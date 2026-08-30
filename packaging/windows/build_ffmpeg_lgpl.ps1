#!/usr/bin/env pwsh
#
# Builds a minimal LGPL shared-library FFmpeg prefix for SeedVR2-ncnn on
# Windows x86_64 using the MSYS2 UCRT64 MinGW toolchain.
#
# The resulting prefix is consumed by packaging/windows/package_runtime.ps1.
# Model weights are never downloaded or bundled.
#
# Usage:
#   build_ffmpeg_lgpl.ps1 -Prefix <path> [-SourceTarball <path>] [-Jobs <count>]
#                         [-Msys2Root <path>]

[CmdletBinding()]
param(
    [string]$Prefix,
    [string]$SourceTarball,
    [int]$Jobs = 2,
    [string]$Msys2Root
)

$ErrorActionPreference = 'Stop'

$FfmpegVersion = '8.1.2'
$FfmpegSha256 = '464beb5e7bf0c311e68b45ae2f04e9cc2af88851abb4082231742a74d97b524c'
$FfmpegUrl = "https://ffmpeg.org/releases/ffmpeg-$FfmpegVersion.tar.xz"

function Write-Usage {
    @"
Usage: build_ffmpeg_lgpl.ps1 -Prefix <path> [-SourceTarball <path>] [-Jobs <count>] [-Msys2Root <path>]

Builds FFmpeg $FfmpegVersion as a minimal LGPL shared-library prefix for SeedVR2-ncnn.

  -Prefix          Destination prefix (must not already exist).
  -SourceTarball   Existing ffmpeg-$FfmpegVersion.tar.xz. Downloaded when omitted.
  -Jobs            Parallel make jobs (default 2).
  -Msys2Root       MSYS2 installation root containing ucrt64/ (auto-detected when omitted).
"@ | Write-Host
}

function Fail {
    param([string]$Message)
    Write-Error $Message
    exit 1
}

if ($Args.Count -gt 0) { Write-Usage; Fail "unexpected positional arguments: $($Args -join ' ')" }
if (-not $Prefix) { Write-Usage; Fail "-Prefix is required" }
if ($Jobs -lt 1) { Fail "-Jobs must be a positive integer" }

# This is a Windows-only toolchain script.
if ($env:OS -ne 'Windows_NT') { Fail "Windows FFmpeg build must run on Windows" }
if ([System.Environment]::Is64BitOperatingSystem -eq $false) {
    Fail "Windows FFmpeg build requires an x86_64 host"
}

# ---------------------------------------------------------------------------
# Locate the MSYS2 UCRT64 MinGW toolchain.
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
    # A MinGW gcc already on PATH implies a usable MSYS2 install two levels up.
    $gccOnPath = Get-Command gcc.exe -ErrorAction SilentlyContinue
    if ($gccOnPath) {
        $binDir = Split-Path -Parent $gccOnPath.Source
        $roots += (Resolve-Path (Join-Path $binDir '..\..') -ErrorAction SilentlyContinue).Path
    }

    foreach ($root in $roots) {
        if (-not $root) { continue }
        if (Test-Path (Join-Path $root 'ucrt64\bin\gcc.exe')) {
            return (Resolve-Path $root).Path
        }
    }
    return $null
}

$msys2 = Resolve-Msys2Root $Msys2Root
if (-not $msys2) {
    Fail @"
MSYS2 UCRT64 MinGW toolchain not found.

A Windows FFmpeg source build requires the MSYS2 UCRT64 MinGW environment
(gcc, mingw32-make, nasm). Install MSYS2 from https://www.msys2.org/ and run:

    pacman -S --noconfirm mingw-w64-ucrt-x86_64-gcc mingw-w64-ucrt-x86_64-make
    pacman -S --noconfirm mingw-w64-ucrt-x86_64-nasm

Then re-run with -Msys2Root <path-to-msys2>.
"@
}

$ucrtBin = Join-Path $msys2 'ucrt64\bin'
$bashExe = Join-Path $msys2 'usr\bin\bash.exe'

foreach ($tool in @('gcc.exe', 'g++.exe', 'mingw32-make.exe', 'nasm.exe', 'pkg-config.exe', 'dlltool.exe')) {
    if (-not (Test-Path (Join-Path $ucrtBin $tool))) {
        Fail "required UCRT64 build tool is unavailable: $tool (looked in $ucrtBin)"
    }
}
if (-not (Test-Path $bashExe)) { Fail "MSYS2 bash is unavailable: $bashExe" }

# ---------------------------------------------------------------------------
# Acquire and verify the pinned source tarball.
# ---------------------------------------------------------------------------
$prefixFull = [System.IO.Path]::GetFullPath($Prefix)

if (-not $SourceTarball) {
    $sourceParent = Split-Path -Parent $prefixFull
    $SourceTarball = Join-Path $sourceParent "ffmpeg-$FfmpegVersion.tar.xz"
}
$sourceFull = [System.IO.Path]::GetFullPath($SourceTarball)

New-Item -ItemType Directory -Force -Path (Split-Path -Parent $sourceFull) | Out-Null

if (-not (Test-Path $sourceFull)) {
    Write-Host "Downloading $FfmpegUrl"
    & curl.exe -fL --retry 3 --silent --show-error --output $sourceFull $FfmpegUrl
    if ($LASTEXITCODE -ne 0) { Fail "failed to download FFmpeg source: $FfmpegUrl" }
}
if (-not (Test-Path $sourceFull)) { Fail "FFmpeg source archive does not exist: $sourceFull" }

$actualSha = (Get-FileHash -Algorithm SHA256 -Path $sourceFull).Hash.ToLowerInvariant()
if ($actualSha -ne $FfmpegSha256) {
    Fail "FFmpeg source checksum mismatch: $sourceFull`n  expected $FfmpegSha256`n  actual   $actualSha"
}
Write-Host "ffmpeg source sha256 verified: $actualSha"

if (Test-Path $prefixFull) { Fail "FFmpeg prefix already exists: $prefixFull" }

# ---------------------------------------------------------------------------
# Convert Windows paths to MSYS POSIX paths for the bash build step.
# ---------------------------------------------------------------------------
function ConvertTo-MsysPath {
    param([string]$WindowsPath)
    $p = [System.IO.Path]::GetFullPath($WindowsPath)
    $drive = $p.Substring(0, 1).ToLowerInvariant()
    $rest = $p.Substring(2).Replace('\', '/')
    return "/$drive$rest"
}

$msysPrefix = ConvertTo-MsysPath $prefixFull
$msysSource = ConvertTo-MsysPath $sourceFull

$workDir = Join-Path ([System.IO.Path]::GetTempPath()) ("seedvr2-ffmpeg-build-" + [guid]::NewGuid().ToString('N'))
New-Item -ItemType Directory -Force -Path $workDir | Out-Null
$msysWork = ConvertTo-MsysPath $workDir

# FFmpeg's configure refuses to run under the MSYS environment, so everything
# below runs inside the UCRT64 MINGW environment (MSYSTEM=UCRT64).
$bashScript = @'
set -eu

PREFIX="__PREFIX__"
SRC="__SOURCE__"
WORK="__WORK__"
JOBS=__JOBS__

mkdir -p "$WORK"
tar -xf "$SRC" -C "$WORK"
cd "$WORK/ffmpeg-__VERSION__"

if [ ! -x ./configure ]; then
    echo "FFmpeg source archive has an unexpected layout" >&2
    exit 1
fi

./configure \
    --prefix="$PREFIX" \
    --enable-shared \
    --disable-static \
    --enable-pic \
    --enable-small \
    --disable-programs \
    --disable-doc \
    --disable-debug \
    --disable-avdevice \
    --disable-avfilter \
    --disable-network \
    --disable-autodetect \
    --disable-everything \
    --disable-gpl \
    --disable-nonfree \
    --disable-iconv \
    --disable-swresample \
    --enable-swscale \
    --enable-protocol=file \
    --enable-demuxer=avi,matroska,mov \
    --enable-parser=av1,h264,hevc,mjpeg,mpeg4video,vp9 \
    --enable-decoder=av1,h264,hevc,mjpeg,mpeg4,vp9

mingw32-make -j "$JOBS"
mingw32-make install
'@ -replace '__PREFIX__', $msysPrefix `
    -replace '__SOURCE__', $msysSource `
    -replace '__WORK__', $msysWork `
    -replace '__JOBS__', "$Jobs" `
    -replace '__VERSION__', $FfmpegVersion

# bash rejects CRLF line endings; write the generated script with LF only.
$bashScriptPath = Join-Path $workDir 'build.sh'
[System.IO.File]::WriteAllText($bashScriptPath, ($bashScript -join "`n") + "`n")

Write-Host "Building FFmpeg $FfmpegVersion with MSYS2 UCRT64 MinGW (jobs=$Jobs)"
Write-Host "  msys2  : $msys2"
Write-Host "  prefix : $prefixFull"

# ---------------------------------------------------------------------------
# Run the build.
# ---------------------------------------------------------------------------
$previousMsystem = $env:MSYSTEM
$savedPath = $env:PATH
$env:MSYSTEM = 'UCRT64'
$env:PATH = "$ucrtBin;$env:PATH"
try {
    & $bashExe -lc ("bash '" + (ConvertTo-MsysPath $bashScriptPath) + "'")
    if ($LASTEXITCODE -ne 0) {
        Fail "FFmpeg build failed (exit code $LASTEXITCODE). See the output above for details."
    }
}
finally {
    $env:MSYSTEM = $previousMsystem
    $env:PATH = $savedPath
}

# ---------------------------------------------------------------------------
# Emit build metadata alongside the installed prefix.
# ---------------------------------------------------------------------------
if (-not (Test-Path $prefixFull)) { Fail "FFmpeg build did not create prefix: $prefixFull" }

$configureLine = "./configure --prefix=$prefixFull --enable-shared --disable-static --enable-pic --enable-small --disable-programs --disable-doc --disable-debug --disable-avdevice --disable-avfilter --disable-network --disable-autodetect --disable-everything --disable-gpl --disable-nonfree --disable-iconv --disable-swresample --enable-swscale --enable-protocol=file --enable-demuxer=avi,matroska,mov --enable-parser=av1,h264,hevc,mjpeg,mpeg4video,vp9 --enable-decoder=av1,h264,hevc,mjpeg,mpeg4,vp9"

$configLines = @(
    "ffmpeg-version=$FfmpegVersion",
    "source-url=$FfmpegUrl",
    "source-sha256=$FfmpegSha256",
    "target-os=mingw64",
    "toolchain=msys2-ucrt64-mingw-w64",
    "configure=$configureLine"
)
[System.IO.File]::WriteAllText((Join-Path $prefixFull 'ffmpeg-build-config.txt'), ($configLines -join "`n") + "`n")

$licenseSource = Join-Path $workDir "ffmpeg-$FfmpegVersion\COPYING.LGPLv2.1"
if (-not (Test-Path $licenseSource)) {
    Fail "FFmpeg source archive is missing COPYING.LGPLv2.1"
}
Copy-Item -Force $licenseSource (Join-Path $prefixFull 'ffmpeg-license.txt')

Copy-Item -Force $sourceFull (Join-Path $prefixFull 'ffmpeg-source.tar.xz')
[System.IO.File]::WriteAllText((Join-Path $prefixFull 'source-sha256'), "$FfmpegSha256  ffmpeg-$FfmpegVersion.tar.xz`n")

# ---------------------------------------------------------------------------
# Validate the installed prefix.
# ---------------------------------------------------------------------------
$prefixBin = Join-Path $prefixFull 'bin'
$prefixLib = Join-Path $prefixFull 'lib'
if (-not (Test-Path $prefixBin)) { Fail "FFmpeg prefix is missing bin/: $prefixFull" }
if (-not (Test-Path $prefixLib)) { Fail "FFmpeg prefix is missing lib/: $prefixFull" }

foreach ($library in @('avformat', 'avcodec', 'avutil', 'swscale')) {
    $dll = Get-ChildItem -Path $prefixBin -Filter "$library-*.dll" -File -ErrorAction SilentlyContinue
    if (-not $dll) { Fail "FFmpeg build did not install $library-*.dll in $prefixBin" }
    $importLib = Get-ChildItem -Path $prefixLib -Filter "lib$library.dll.a" -File -ErrorAction SilentlyContinue
    if (-not $importLib) { Fail "FFmpeg build did not install lib$library.dll.a in $prefixLib" }
    Write-Host "  found $($dll[0].Name) + $($importLib[0].Name)"
}

Remove-Item -Recurse -Force $workDir -ErrorAction SilentlyContinue

Write-Host "ffmpeg-prefix=$prefixFull"
Write-Host "ffmpeg-source-sha256=$FfmpegSha256"
