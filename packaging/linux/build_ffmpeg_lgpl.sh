#!/bin/sh

set -eu

ffmpeg_version=8.1.2
ffmpeg_sha256=464beb5e7bf0c311e68b45ae2f04e9cc2af88851abb4082231742a74d97b524c
ffmpeg_url="https://ffmpeg.org/releases/ffmpeg-${ffmpeg_version}.tar.xz"

usage() {
    cat <<EOF
Usage: build_ffmpeg_lgpl.sh --prefix <path> [--source-tarball <path>] [--jobs <count>]

Builds FFmpeg ${ffmpeg_version} as a minimal LGPL shared-library prefix for SeedVR2-ncnn.
EOF
}

die() {
    printf '%s\n' "$*" >&2
    exit 1
}

require_value() {
    if [ "$#" -lt 2 ] || [ -z "$2" ]; then
        usage >&2
        exit 2
    fi
}

prefix=
source_tarball=
jobs=2

while [ "$#" -gt 0 ]; do
    case "$1" in
        --prefix)
            require_value "$1" "${2:-}"
            prefix=$2
            shift 2
            ;;
        --source-tarball)
            require_value "$1" "${2:-}"
            source_tarball=$2
            shift 2
            ;;
        --jobs)
            require_value "$1" "${2:-}"
            jobs=$2
            shift 2
            ;;
        --help|-h)
            usage
            exit 0
            ;;
        *)
            usage >&2
            exit 2
            ;;
    esac
done

[ -n "$prefix" ] || die "--prefix is required"
[ "$(uname -s)" = "Linux" ] || die "FFmpeg runtime build must run on Linux"
case "$(uname -m)" in
    x86_64|amd64) ;;
    *) die "FFmpeg runtime build requires x86_64, got: $(uname -m)" ;;
esac
case "$jobs" in
    ''|*[!0-9]*) die "--jobs must be a positive integer" ;;
esac
[ "$jobs" -gt 0 ] || die "--jobs must be a positive integer"

if [ -n "$source_tarball" ]; then
    [ -f "$source_tarball" ] || die "FFmpeg source archive does not exist: $source_tarball"
else
    source_parent=$(dirname -- "$prefix")
    source_tarball="$source_parent/ffmpeg-${ffmpeg_version}.tar.xz"
    mkdir -p "$source_parent"
    if [ ! -f "$source_tarball" ]; then
        if command -v curl >/dev/null 2>&1; then
            curl -fL --retry 3 --output "$source_tarball" "$ffmpeg_url"
        elif command -v wget >/dev/null 2>&1; then
            wget -O "$source_tarball" "$ffmpeg_url"
        else
            die "curl or wget is required to download FFmpeg source"
        fi
    fi
fi

actual_sha256=$(sha256sum "$source_tarball" | awk '{print $1}')
[ "$actual_sha256" = "$ffmpeg_sha256" ] || die "FFmpeg source checksum mismatch: $source_tarball"
[ ! -e "$prefix" ] || die "FFmpeg prefix already exists: $prefix"

for required_tool in tar make gcc; do
    command -v "$required_tool" >/dev/null 2>&1 || die "required build tool is unavailable: $required_tool"
done

work_dir=$(mktemp -d "${TMPDIR:-/tmp}/seedvr2-ffmpeg-build.XXXXXX")
trap 'rm -rf "$work_dir"' EXIT HUP INT TERM
tar -xf "$source_tarball" -C "$work_dir"
source_dir="$work_dir/ffmpeg-$ffmpeg_version"
[ -x "$source_dir/configure" ] || die "FFmpeg source archive has an unexpected layout"

mkdir -p "$prefix"
(
    cd "$source_dir"
    ./configure \
        --prefix="$prefix" \
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
    make -j "$jobs"
    make install
)

cat >"$prefix/ffmpeg-build-config.txt" <<EOF
ffmpeg-version=$ffmpeg_version
source-url=$ffmpeg_url
source-sha256=$ffmpeg_sha256
configure=./configure --prefix=$prefix --enable-shared --disable-static --enable-pic --enable-small --disable-programs --disable-doc --disable-debug --disable-avdevice --disable-avfilter --disable-network --disable-autodetect --disable-everything --disable-gpl --disable-nonfree --disable-iconv --disable-swresample --enable-swscale --enable-protocol=file --enable-demuxer=avi,matroska,mov --enable-parser=av1,h264,hevc,mjpeg,mpeg4video,vp9 --enable-decoder=av1,h264,hevc,mjpeg,mpeg4,vp9
EOF
cp "$source_tarball" "$prefix/ffmpeg-source.tar.xz"
cp "$source_dir/COPYING.LGPLv2.1" "$prefix/ffmpeg-license.txt"

for library_name in libavformat libavcodec libavutil libswscale; do
    if ! find "$prefix/lib" -maxdepth 1 -type f -name "${library_name}.so.*" -print -quit | grep -q .; then
        die "FFmpeg build did not install ${library_name}.so.*"
    fi
done

printf '%s\n' "ffmpeg-prefix=$prefix"
printf '%s\n' "ffmpeg-source-sha256=$ffmpeg_sha256"
