#!/usr/bin/env pwsh
#
# Contract tests for the Windows x86_64 SeedVR2-ncnn runtime package.
#
# This is the Windows counterpart of test_linux_runtime_package.sh. It expects
# the staged package directory and the release ZIP produced by
# packaging/windows/package_runtime.ps1.
#
# Usage:
#   test_windows_runtime_package.ps1 -PackageDir <dir> -Archive <path> [-SkipGpuSmoke]
#   test_windows_runtime_package.ps1 -ReleaseSmoke -Binary <path> -FfmpegPrefix <path>
#                                    -FfmpegSource <path> -Output <dir>

[CmdletBinding()]
param(
    [string]$PackageDir,
    [string]$Archive,
    [string]$Binary,
    [string]$FfmpegPrefix,
    [string]$FfmpegSource,
    [string]$Output,
    [switch]$ReleaseSmoke
)

$ErrorActionPreference = 'Stop'

$PackageName = 'SeedVR2-ncnn-windows-x86_64'

function Fail {
    param([string]$Message)
    Write-Host "FAIL: $Message" -ForegroundColor Red
    exit 1
}

function Pass {
    param([string]$Message)
    Write-Host "  ok  $Message"
}

# ---------------------------------------------------------------------------
# Optionally stage a fresh package from the release build.
# ---------------------------------------------------------------------------
if ($ReleaseSmoke) {
    foreach ($required in @('Binary', 'FfmpegPrefix', 'FfmpegSource', 'Output')) {
        if (-not (Get-Variable -Name $required -ValueOnly)) { Fail "-$required is required with -ReleaseSmoke" }
    }
    $packageScript = Join-Path $PSScriptRoot '..\..\..\packaging\windows\package_runtime.ps1'
    if (-not (Test-Path $packageScript)) { Fail "package script is missing: $packageScript" }

    Write-Host "staging release package"
    & pwsh -NoProfile -ExecutionPolicy Bypass -File $packageScript `
        -Binary $Binary -FfmpegPrefix $FfmpegPrefix -FfmpegSource $FfmpegSource -Output $Output
    if ($LASTEXITCODE -ne 0) { Fail "package_runtime.ps1 exited with $LASTEXITCODE" }

    $PackageDir = Join-Path ([System.IO.Path]::GetFullPath($Output)) $PackageName
    $Archive = Join-Path ([System.IO.Path]::GetFullPath($Output)) "$PackageName.zip"
}

if (-not $PackageDir) { Fail "-PackageDir is required (or use -ReleaseSmoke)" }
if (-not $Archive) { Fail "-Archive is required (or use -ReleaseSmoke)" }

$packageFull = [System.IO.Path]::GetFullPath($PackageDir)
$archiveFull = [System.IO.Path]::GetFullPath($Archive)

Write-Host "package dir : $packageFull"
Write-Host "archive     : $archiveFull"

# ---------------------------------------------------------------------------
# 1. Layout contract.
# ---------------------------------------------------------------------------
if (-not (Test-Path $packageFull -PathType Container)) { Fail "package directory is missing: $packageFull" }
if (-not (Test-Path $archiveFull -PathType Leaf)) { Fail "package archive is missing: $archiveFull" }

foreach ($entry in @('seedvr2-ncnn.bat', 'bin/seedvr2-ncnn.exe', 'models/README.md', 'manifest.sha256')) {
    if (-not (Test-Path (Join-Path $packageFull $entry))) { Fail "package is missing $entry" }
}
foreach ($entry in @('bin', 'lib', 'licenses', 'models')) {
    if (-not (Test-Path (Join-Path $packageFull $entry) -PathType Container)) { Fail "package is missing directory $entry" }
}
Pass "package layout"

# ---------------------------------------------------------------------------
# 2. Launcher runs with a clean PATH and prints usage.
# ---------------------------------------------------------------------------
$launcher = Join-Path $packageFull 'seedvr2-ncnn.bat'
$cleanPath = 'C:\Windows\System32;C:\Windows'

$savedPath = $env:PATH
$env:PATH = $cleanPath
try {
    $helpOutput = & cmd.exe /c "`"$launcher`" --help" 2>&1
    $helpExit = $LASTEXITCODE
}
finally {
    $env:PATH = $savedPath
}
if ($helpExit -ne 0) { Fail "seedvr2-ncnn.bat --help exited with $helpExit (clean PATH)" }
$helpText = $helpOutput -join "`n"
if ($helpText -notmatch 'Usage: seedvr2-ncnn') { Fail "help output does not contain 'Usage: seedvr2-ncnn'" }
Pass "seedvr2-ncnn.bat --help with clean PATH"

# The launcher must not depend on the current working directory: run it from
# an unrelated directory and confirm it still resolves its payload.
$savedPath = $env:PATH
$env:PATH = $cleanPath
try {
    $pushLocation = Get-Location
    Set-Location 'C:\Windows'
    $cwdOutput = & cmd.exe /c "`"$launcher`" --help" 2>&1
    $cwdExit = $LASTEXITCODE
    Set-Location $pushLocation
}
finally {
    $env:PATH = $savedPath
}
if ($cwdExit -ne 0) { Fail "seedvr2-ncnn.bat --help failed when invoked from an unrelated working directory" }
if (($cwdOutput -join "`n") -notmatch 'Usage: seedvr2-ncnn') { Fail "help output from unrelated CWD is missing usage text" }
Pass "launcher is independent of the working directory"

# ---------------------------------------------------------------------------
# 3. No missing DLLs across the whole dependency closure.
# ---------------------------------------------------------------------------
$objdump = Get-Command objdump.exe -ErrorAction SilentlyContinue
if (-not $objdump) { Fail "objdump is required for DLL dependency validation" }

function Get-PeImports {
    param([string]$File)
    $lines = & objdump.exe -p $File 2>$null
    $names = @()
    foreach ($line in $lines) {
        if ($line -match '^\s*DLL Name:\s*(\S+)\s*$') { $names += $Matches[1] }
    }
    return $names
}

$systemDlls = @(
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
    return ($systemDlls -contains (($Name -replace '\.dll$', '').ToUpperInvariant()))
}

$packageLib = Join-Path $packageFull 'lib'
$missing = @()
$visited = @{}

$targets = @((Join-Path $packageFull 'bin/seedvr2-ncnn.exe')) + (Get-ChildItem -Path $packageLib -Filter '*.dll' -File).FullName
$queue = New-Object System.Collections.Generic.Queue[string]
foreach ($t in $targets) { $queue.Enqueue($t) }

while ($queue.Count -gt 0) {
    $current = $queue.Dequeue()
    $key = $current.ToLowerInvariant()
    if ($visited.ContainsKey($key)) { continue }
    $visited[$key] = $true

    foreach ($dep in (Get-PeImports $current)) {
        if (Test-SystemDll $dep) { continue }
        $bundled = Join-Path $packageLib $dep
        if (Test-Path $bundled) { $queue.Enqueue($bundled); continue }
        $missing += "$dep (required by $(Split-Path -Leaf $current))"
    }
}
if ($missing.Count -gt 0) {
    Fail ("missing DLL dependencies:`n  " + ($missing -join "`n  "))
}
Pass "every non-system DLL dependency is bundled ($($visited.Count) PE files scanned)"

# GPU driver components must never be bundled.
foreach ($dll in (Get-ChildItem -Path $packageLib -Filter '*.dll' -File)) {
    if ($dll.Name -match '^(vulkan-1|vulkan|nvoglv64|nvapi64|nvcuda|amdxc64)\.dll$' -or
        $dll.Name -match '^(igvk|igd|nv-vk)') {
        Fail "package bundles a GPU driver component: $($dll.Name)"
    }
}
Pass "no Vulkan or GPU driver DLLs bundled"

# ---------------------------------------------------------------------------
# 4. Manifest integrity.
# ---------------------------------------------------------------------------
$manifestPath = Join-Path $packageFull 'manifest.sha256'
$expectedFiles = @{}
foreach ($line in (Get-Content -LiteralPath $manifestPath)) {
    if ($line -notmatch '^([0-9a-f]{64})\s\s(.+)$') { Fail "manifest line is malformed: $line" }
    $expectedFiles[$Matches[2]] = $Matches[1]
}

$actualFiles = @{}
foreach ($file in (Get-ChildItem -LiteralPath $packageFull -Recurse -File)) {
    $relative = $file.FullName.Substring($packageFull.Length).TrimStart('\', '/').Replace('\', '/')
    if ($relative -eq 'manifest.sha256') { continue }
    $actualFiles[$relative] = (Get-FileHash -Algorithm SHA256 -LiteralPath $file.FullName).Hash.ToLowerInvariant()
}

foreach ($name in $actualFiles.Keys) {
    if (-not $expectedFiles.ContainsKey($name)) { Fail "manifest is missing an entry for $name" }
    if ($expectedFiles[$name] -ne $actualFiles[$name]) { Fail "manifest checksum mismatch for $name" }
}
foreach ($name in $expectedFiles.Keys) {
    if (-not $actualFiles.ContainsKey($name)) { Fail "manifest references a missing file: $name" }
}
Pass "manifest.sha256 verified ($($actualFiles.Count) files)"

# ---------------------------------------------------------------------------
# 5. The archive contents match the staged tree and carry no model weights.
# ---------------------------------------------------------------------------
Add-Type -AssemblyName System.IO.Compression
Add-Type -AssemblyName System.IO.Compression.FileSystem

$zip = [System.IO.Compression.ZipFile]::OpenRead($archiveFull)
try {
    $entries = @()
    foreach ($entry in $zip.Entries) { $entries += $entry.FullName }

    $expectedEntries = @()
    foreach ($name in ($actualFiles.Keys + @('manifest.sha256'))) { $expectedEntries += "$PackageName/$name" }
    $expectedEntries = $expectedEntries | Sort-Object

    $missingEntries = Compare-Object $expectedEntries $entries | Where-Object { $_.SideIndicator -eq '<=' }
    $extraEntries = Compare-Object $expectedEntries $entries | Where-Object { $_.SideIndicator -eq '=>' }
    if ($missingEntries) { Fail "archive is missing entries: $($missingEntries.InputObject -join ', ')" }
    if ($extraEntries) { Fail "archive has unexpected entries: $($extraEntries.InputObject -join ', ')" }

    $sortedCopy = $entries | Sort-Object
    if (($entries -join "`n") -ne ($sortedCopy -join "`n")) { Fail "archive entries are not in stable sorted order" }

    $weightPattern = '\.(ncnn\.bin|ncnn\.param|safetensors|ckpt|pth|gguf|onnx|f32)$'
    foreach ($name in $entries) {
        if ($name -match $weightPattern) { Fail "archive contains model weights: $name" }
        if ($name -match '/models/' -and $name -notmatch '/models/README\.md$') {
            Fail "archive contains non-README model payload: $name"
        }
    }
    Pass "archive matches the staged tree and contains no model weights ($($entries.Count) entries)"

    # Reproducible timestamps.
    foreach ($entry in $zip.Entries) {
        if ($entry.LastWriteTime.Year -ne 1980) { Fail "archive entry has a non-reproducible timestamp: $($entry.FullName) ($($entry.LastWriteTime))" }
    }
    Pass "archive timestamps are normalized"
}
finally { $zip.Dispose() }

# ---------------------------------------------------------------------------
# 6. The models README points at the separately distributed model package.
# ---------------------------------------------------------------------------
$modelReadme = Get-Content -LiteralPath (Join-Path $packageFull 'models/README.md') -Raw
if ($modelReadme -notmatch 'seedvr2-3b') { Fail "models/README.md does not reference seedvr2-3b" }
if ($modelReadme -notmatch 'separately') { Fail "models/README.md does not state that weights ship separately" }
Pass "models/README.md documents the separate model package"

Write-Host "PASS: Windows runtime package contract" -ForegroundColor Green
exit 0
