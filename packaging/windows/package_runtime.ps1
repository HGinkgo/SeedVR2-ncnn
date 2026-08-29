#!/usr/bin/env pwsh
#
# Stages a Windows x86_64 SeedVR2-ncnn runtime archive (ZIP) bundling the
# LGPL FFmpeg shared libraries produced by build_ffmpeg_lgpl.ps1.
#
# Model weights are never bundled. When -ModelDir is supplied it must be a
# self-contained model package: symbolic links and junctions are rejected.
#
# Usage:
#   package_runtime.ps1 -Binary <path> -FfmpegPrefix <path> -FfmpegSource <path>
#                       -Output <dir> [-ModelDir <path>] [-Msys2Root <path>]

[CmdletBinding()]
param(
    [string]$Binary,
    [string]$FfmpegPrefix,
    [string]$FfmpegSource,
    [string]$Output,
    [string]$ModelDir,
    [string]$Msys2Root
)

$ErrorActionPreference = 'Stop'

$PackageName = 'SeedVR2-ncnn-windows-x86_64'

# 1980-01-01 00:00:00 is the earliest timestamp representable in a ZIP entry and
# is used for every entry so the archive is reproducible.
$ZipEpoch = [DateTimeOffset]::new(1980, 1, 1, 0, 0, 0, [TimeSpan]::Zero)

function Write-Usage {
    @"
Usage: package_runtime.ps1 -Binary <path> -FfmpegPrefix <path> -FfmpegSource <path> -Output <dir> [-ModelDir <path>] [-Msys2Root <path>]

Stages a Windows x86_64 SeedVR2-ncnn runtime archive with a bundled LGPL FFmpeg prefix.
"@ | Write-Host
}

function Fail {
    param([string]$Message)
    Write-Error $Message
    exit 1
}

if ($Args.Count -gt 0) { Write-Usage; Fail "unexpected positional arguments: $($Args -join ' ')" }
foreach ($required in @('Binary', 'FfmpegPrefix', 'FfmpegSource', 'Output')) {
    if (-not (Get-Variable -Name $required -ValueOnly)) { Write-Usage; Fail "-$required is required" }
}

if ($env:OS -ne 'Windows_NT') { Fail "Windows runtime packaging must run on Windows" }
if ([System.Environment]::Is64BitOperatingSystem -eq $false) {
    Fail "Windows runtime packaging requires an x86_64 host"
}

# ---------------------------------------------------------------------------
# Locate the MSYS2 UCRT64 MinGW toolchain (runtime licences live there).
# ---------------------------------------------------------------------------
function Resolve-Msys2Root {
    param([string]$Candidate)

    $roots = @()
    if ($Candidate) { $roots += $Candidate }
    $roots += @(
        'C:\msys64', 'C:\msys2', 'D:\msys64',
        (Join-Path $env:LOCALAPPDATA 'msys2'),
        (Join-Path ${env:ProgramFiles} 'msys64')
    )
    $gccOnPath = Get-Command gcc.exe -ErrorAction SilentlyContinue
    if ($gccOnPath) {
        $binDir = Split-Path -Parent $gccOnPath.Source
        $resolved = (Resolve-Path (Join-Path $binDir '..\..') -ErrorAction SilentlyContinue).Path
        if ($resolved) { $roots += $resolved }
    }

    foreach ($root in $roots) {
        if (-not $root) { continue }
        if (Test-Path (Join-Path $root 'ucrt64\bin\gcc.exe')) { return (Resolve-Path $root).Path }
    }
    return $null
}

$msys2 = Resolve-Msys2Root $Msys2Root
if (-not $msys2) { Fail "MSYS2 UCRT64 MinGW toolchain not found; pass -Msys2Root <path>" }

# ---------------------------------------------------------------------------
# Validate inputs.
# ---------------------------------------------------------------------------
$binaryFull = [System.IO.Path]::GetFullPath($Binary)
$prefixFull = [System.IO.Path]::GetFullPath($FfmpegPrefix)
$sourceFull = [System.IO.Path]::GetFullPath($FfmpegSource)
$outputFull = [System.IO.Path]::GetFullPath($Output)

if (-not (Test-Path $binaryFull -PathType Leaf)) { Fail "release binary is missing: $binaryFull" }
if (-not (Test-Path $prefixFull -PathType Container)) { Fail "FFmpeg prefix does not exist: $prefixFull" }
if (-not (Test-Path $sourceFull -PathType Leaf)) { Fail "FFmpeg source archive does not exist: $sourceFull" }

$prefixBin = Join-Path $prefixFull 'bin'
$prefixLib = Join-Path $prefixFull 'lib'
foreach ($dir in @($prefixBin, $prefixLib)) {
    if (-not (Test-Path $dir -PathType Container)) { Fail "FFmpeg prefix is missing $(Split-Path -Leaf $dir)/: $prefixFull" }
}
foreach ($meta in @('ffmpeg-build-config.txt', 'ffmpeg-license.txt')) {
    if (-not (Test-Path (Join-Path $prefixFull $meta) -PathType Leaf)) {
        Fail "FFmpeg prefix is missing ${meta}: $prefixFull"
    }
}

$configPath = Join-Path $prefixFull 'ffmpeg-build-config.txt'
$expectedSourceSha = $null
foreach ($line in (Get-Content -LiteralPath $configPath)) {
    if ($line -match '^source-sha256=(.+)$') { $expectedSourceSha = $Matches[1].Trim().ToLowerInvariant(); break }
}
if (-not $expectedSourceSha) { Fail "FFmpeg prefix build configuration is missing source-sha256" }

$actualSourceSha = (Get-FileHash -Algorithm SHA256 -LiteralPath $sourceFull).Hash.ToLowerInvariant()
if ($actualSourceSha -ne $expectedSourceSha) {
    Fail "FFmpeg source checksum does not match the build prefix: $sourceFull`n  expected $expectedSourceSha`n  actual   $actualSourceSha"
}

# ---------------------------------------------------------------------------
# PE import helpers. objdump ships with the MinGW toolchain and is the
# MSVC-free equivalent of readelf/dumpbin.
# ---------------------------------------------------------------------------
$objdump = Get-Command objdump.exe -ErrorAction SilentlyContinue
if (-not $objdump) { Fail "objdump is required for runtime dependency validation (MinGW binutils)" }

function Get-PeImports {
    param([string]$File)
    $lines = & objdump.exe -p $File 2>$null
    $names = @()
    foreach ($line in $lines) {
        if ($line -match '^\s*DLL Name:\s*(\S+)\s*$') { $names += $Matches[1] }
    }
    return $names
}

$knownSystemDlls = @(
    'KERNEL32', 'USER32', 'GDI32', 'ADVAPI32', 'SHELL32', 'OLE32', 'OLEAUT32',
    'WS2_32', 'MSVCRT', 'NTDLL', 'CRYPT32', 'SHLWAPI', 'VERSION', 'WINMM',
    'PSAPI', 'RPCRT4', 'SETUPAPI', 'DNSAPI', 'IPHLPAPI', 'MPR', 'NETAPI32',
    'POWRPROF', 'USERENV', 'WTSAPI32', 'BCRYPT', 'NCRYPT', 'SECUR32', 'AUTHZ',
    'CABINET', 'CFGMGR32', 'DEVOBJ', 'WINHTTP', 'WININET', 'WINTRUST',
    'WLDAP32', 'NORMALIZ', 'IMM32', 'COMCTL32', 'COMDLG32', 'DWMAPI', 'DXGI',
    'D3D11', 'D3D12', 'UCRTBASE', 'VCRUNTIME140', 'MSVCP140', 'CONCRT140'
)

function Test-SystemDll {
    param([string]$Name)
    if ($Name -match '^api-ms-win' -or $Name -match '^ext-ms-win') { return $true }
    return ($knownSystemDlls -contains (($Name -replace '\.dll$', '').ToUpperInvariant()))
}

# Graphics drivers must be resolved by the host, never shipped in the package.
$forbiddenDllPatterns = @(
    '^vulkan-1\.dll$', '^vulkan\.dll$', '^nvoglv64\.dll$', '^nvapi64\.dll$',
    '^nvcuda\.dll$', '^amdxc64\.dll$', '^igvk.*\.dll$', '^igd.*\.dll$',
    '^nv-vk.*\.dll$'
)

$binaryImports = Get-PeImports $binaryFull
if (-not ($binaryImports -match '^avformat-.*\.dll$')) {
    Fail "release binary does not import avformat DLL; rebuild with SEEDVR2_ENABLE_FFMPEG=ON"
}

foreach ($library in @('avformat', 'avcodec', 'avutil', 'swscale')) {
    $dll = Get-ChildItem -Path $prefixBin -Filter "$library-*.dll" -File -ErrorAction SilentlyContinue
    if (-not $dll) { Fail "FFmpeg prefix is missing $library-*.dll: $prefixBin" }
}

# ---------------------------------------------------------------------------
# Validate the model directory, when supplied.
# ---------------------------------------------------------------------------
if ($ModelDir) {
    $modelFull = [System.IO.Path]::GetFullPath($ModelDir)
    if (-not (Test-Path $modelFull -PathType Container)) { Fail "model directory does not exist: $modelFull" }

    $reparse = Get-ChildItem -LiteralPath $modelFull -Recurse -Force -ErrorAction SilentlyContinue |
        Where-Object { $_.Attributes -band [System.IO.FileAttributes]::ReparsePoint }
    if ($reparse) {
        Fail "model directory contains symbolic links or junctions; provide a self-contained model package: $modelFull`n  first offender: $($reparse[0].FullName)"
    }

    # Mirror ModelRegistry::resolve(): conditioning/pos_emb.f32 plus at least
    # one complete <height>x<width> variant directory.
    if (-not (Test-Path (Join-Path $modelFull 'conditioning/pos_emb.f32') -PathType Leaf)) {
        Fail "model directory is missing conditioning/pos_emb.f32: $modelFull"
    }

    $requiredStems = @('vae_encode', 'vae_decode', 'dit_input', 'dit_embedding', 'dit_output')
    for ($i = 0; $i -lt 32; $i++) { $requiredStems += ('dit_block_{0:d2}' -f $i) }

    $variants = Get-ChildItem -LiteralPath $modelFull -Directory | Where-Object { $_.Name -match '^\d+x\d+$' }
    $completeVariant = $null
    foreach ($variant in $variants) {
        $complete = $true
        foreach ($stem in $requiredStems) {
            if (-not (Test-Path (Join-Path $variant.FullName "$stem.ncnn.param") -PathType Leaf) -or
                -not (Test-Path (Join-Path $variant.FullName "$stem.ncnn.bin") -PathType Leaf)) {
                $complete = $false
                break
            }
        }
        if ($complete) { $completeVariant = $variant; break }
    }
    if (-not $completeVariant) {
        Fail "model directory has no complete <height>x<width> variant (needs vae_encode, vae_decode, dit_input, dit_embedding, dit_output and dit_block_00..31 as .ncnn.param/.ncnn.bin): $modelFull"
    }
    Write-Host "model variant validated: $($completeVariant.Name)"
}

# ---------------------------------------------------------------------------
# Stage the package tree.
# ---------------------------------------------------------------------------
$stageDir = Join-Path $outputFull $PackageName
$archivePath = Join-Path $outputFull "$PackageName.zip"
if (Test-Path $stageDir) { Fail "release staging directory already exists: $stageDir" }
if (Test-Path $archivePath) { Fail "release archive already exists: $archivePath" }

New-Item -ItemType Directory -Force -Path $outputFull | Out-Null
New-Item -ItemType Directory -Force -Path (Join-Path $stageDir 'bin') | Out-Null
New-Item -ItemType Directory -Force -Path (Join-Path $stageDir 'lib') | Out-Null
New-Item -ItemType Directory -Force -Path (Join-Path $stageDir 'licenses') | Out-Null
New-Item -ItemType Directory -Force -Path (Join-Path $stageDir 'models') | Out-Null

Copy-Item -Force $binaryFull (Join-Path $stageDir 'bin/seedvr2-ncnn.exe')

$launcherTemplate = Join-Path $PSScriptRoot 'seedvr2-ncnn.bat.in'
if (-not (Test-Path $launcherTemplate -PathType Leaf)) {
    Fail "launcher template is missing: $launcherTemplate"
}
# The launcher is written with CRLF endings, as expected by cmd.exe.
$launcherText = (Get-Content -LiteralPath $launcherTemplate -Raw) -replace "`r`n", "`n"
$launcherText = ($launcherText -replace "`n", "`r`n").TrimEnd() + "`r`n"
[System.IO.File]::WriteAllText((Join-Path $stageDir 'seedvr2-ncnn.bat'), $launcherText)

Copy-Item -Force (Join-Path $prefixFull 'ffmpeg-build-config.txt') (Join-Path $stageDir 'licenses/ffmpeg-build-config.txt')
Copy-Item -Force (Join-Path $prefixFull 'ffmpeg-license.txt') (Join-Path $stageDir 'licenses/FFmpeg-LGPL-2.1-or-later.txt')
Copy-Item -Force $sourceFull (Join-Path $stageDir 'licenses/ffmpeg-source.tar.xz')
Copy-Item -Force $PSCommandPath (Join-Path $stageDir 'licenses/SeedVR2-ncnn-package-script.ps1')

# ---------------------------------------------------------------------------
# Copy the runtime DLL closure.
# ---------------------------------------------------------------------------
# FFmpeg's MinGW install layout puts the shared libraries in bin/ and the
# import libraries in lib/, so the DLLs are collected from bin/.
foreach ($dll in (Get-ChildItem -Path $prefixBin -Filter '*.dll' -File)) {
    Copy-Item -Force $dll.FullName (Join-Path $stageDir 'lib')
    Write-Host "  staged ffmpeg dll $($dll.Name)"
}

# Resolve the transitive dependency closure of the payload plus the staged
# DLLs, and copy anything that is not provided by Windows itself.
$searchPaths = New-Object System.Collections.Specialized.StringCollection
$searchPaths.Add($prefixBin) | Out-Null
$searchPaths.Add((Join-Path $msys2 'ucrt64\bin')) | Out-Null

function Resolve-Dll {
    param([string]$Name, [System.Collections.Specialized.StringCollection]$SearchPaths)
    foreach ($dir in $SearchPaths) {
        $candidate = Join-Path $dir $Name
        if (Test-Path $candidate -PathType Leaf) { return (Resolve-Path $candidate).Path }
    }
    $onPath = Get-Command $Name -ErrorAction SilentlyContinue
    if ($onPath) { return $onPath.Source }
    return $null
}

$visited = @{}
$queue = New-Object System.Collections.Generic.Queue[string]
$queue.Enqueue((Join-Path $stageDir 'bin/seedvr2-ncnn.exe'))
$missing = @()

while ($queue.Count -gt 0) {
    $current = $queue.Dequeue()
    $key = $current.ToLowerInvariant()
    if ($visited.ContainsKey($key)) { continue }
    $visited[$key] = $true

    foreach ($dep in (Get-PeImports $current)) {
        if (Test-SystemDll $dep) { continue }

        $forbidden = $false
        foreach ($pattern in $forbiddenDllPatterns) {
            if ($dep -match $pattern) { $forbidden = $true; break }
        }
        if ($forbidden) { Fail "refusing to bundle graphics driver component $dep required by $(Split-Path -Leaf $current); Vulkan and GPU driver libraries must come from the host" }

        $resolved = Resolve-Dll $dep $searchPaths
        if (-not $resolved) { $missing += "$dep (required by $(Split-Path -Leaf $current))"; continue }

        # Anything living under a Windows system directory is owned by the OS.
        $resolvedFull = [System.IO.Path]::GetFullPath($resolved)
        if ($resolvedFull -match '^[A-Za-z]:\\(Windows|WINDOWS)\\') { continue }

        $target = Join-Path $stageDir "lib/$dep"
        if (-not (Test-Path $target)) {
            Copy-Item -Force $resolvedFull $target
            Write-Host "  staged runtime dll $dep"
        }
        $queue.Enqueue($target)
    }
}

if ($missing.Count -gt 0) {
    Fail ("release binary has unresolved DLL dependencies:`n  " + ($missing -join "`n  "))
}

$libDir = Join-Path $stageDir 'lib'
foreach ($library in @('avformat', 'avcodec', 'avutil', 'swscale')) {
    if (-not (Get-ChildItem -Path $libDir -Filter "$library-*.dll" -File -ErrorAction SilentlyContinue)) {
        Fail "package lib/ is missing $library-*.dll"
    }
}

# ---------------------------------------------------------------------------
# Runtime licence texts for the bundled MinGW/GCC runtime DLLs.
# ---------------------------------------------------------------------------
$licenceSource = Join-Path $msys2 'ucrt64/share/licenses'
$licenceMap = @{
    'GCC-runtime-license.txt'         = 'gcc-libs/COPYING.RUNTIME'
    'GPL-3.0.txt'                     = 'gcc-libs/COPYING3'
    'LGPL-2.1.txt'                    = 'gcc-libs/COPYING.LIB'
    'winpthreads-license.txt'         = 'libwinpthread/COPYING'
    'MinGW-w64-runtime-license.txt'   = 'crt/COPYING.MinGW-w64-runtime.txt'
}
foreach ($entry in $licenceMap.GetEnumerator()) {
    $src = Join-Path $licenceSource $entry.Value
    if (-not (Test-Path $src -PathType Leaf)) { Fail "runtime licence text is unavailable: $src" }
    Copy-Item -Force $src (Join-Path $stageDir "licenses/$($entry.Key)")
}

# ---------------------------------------------------------------------------
# Model directory.
# ---------------------------------------------------------------------------
if ($ModelDir) {
    Copy-Item -Recurse -Force (Join-Path $modelFull '.') (Join-Path $stageDir 'models/seedvr2-3b')
}
else {
    $modelReadme = @'
Place a compatible SeedVR2-ncnn model package in this directory as seedvr2-3b/.
Run the launcher with --model-dir models\seedvr2-3b, or provide another model path.

Model weights are distributed separately under their own license and are never
included in this runtime archive. The model directory must be self-contained;
symbolic-link or junction development directories are not valid release model
packages.
'@
    [System.IO.File]::WriteAllText((Join-Path $stageDir 'models/README.md'), ($modelReadme -replace "`r`n", "`n").Replace("`n", "`r`n"))
}

# ---------------------------------------------------------------------------
# SHA-256 manifest.
# ---------------------------------------------------------------------------
$manifestPath = Join-Path $stageDir 'manifest.sha256'
$manifestLines = New-Object System.Collections.Generic.List[string]
foreach ($file in (Get-ChildItem -LiteralPath $stageDir -Recurse -File)) {
    $relative = $file.FullName.Substring($stageDir.Length).TrimStart('\', '/').Replace('\', '/')
    if ($relative -eq 'manifest.sha256') { continue }
    $hash = (Get-FileHash -Algorithm SHA256 -LiteralPath $file.FullName).Hash.ToLowerInvariant()
    $manifestLines.Add("$hash  $relative")
}
$manifestLines.Sort([System.StringComparer]::Ordinal)
[System.IO.File]::WriteAllText($manifestPath, (($manifestLines -join "`r`n") + "`r`n"))

# ---------------------------------------------------------------------------
# Repackage as a reproducible ZIP: stable ordering, fixed timestamps.
# ---------------------------------------------------------------------------
Add-Type -AssemblyName System.IO.Compression
Add-Type -AssemblyName System.IO.Compression.FileSystem

$entries = Get-ChildItem -LiteralPath $stageDir -Recurse -File |
    ForEach-Object {
        [PSCustomObject]@{
            FullName = $_.FullName
            Relative = $_.FullName.Substring($stageDir.Length).TrimStart('\', '/').Replace('\', '/')
        }
    } | Sort-Object -Property Relative -Culture ([System.Globalization.CultureInfo]::InvariantCulture)

$stream = [System.IO.File]::Create($archivePath)
try {
    $zip = New-Object System.IO.Compression.ZipArchive($stream, [System.IO.Compression.ZipArchiveMode]::Create)
    try {
        foreach ($entry in $entries) {
            $relative = "$PackageName/" + $entry.Relative
            $zipEntry = $zip.CreateEntry($relative, [System.IO.Compression.CompressionLevel]::Optimal)
            $zipEntry.LastWriteTime = $ZipEpoch
            $inStream = [System.IO.File]::OpenRead($entry.FullName)
            try {
                $outStream = $zipEntry.Open()
                try { $inStream.CopyTo($outStream) }
                finally { $outStream.Dispose() }
            }
            finally { $inStream.Dispose() }
        }
    }
    finally { $zip.Dispose() }
}
finally { $stream.Dispose() }

Write-Host "package-dir=$stageDir"
Write-Host "package-archive=$archivePath"
