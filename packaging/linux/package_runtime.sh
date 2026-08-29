#!/bin/sh

set -eu

usage() {
    cat <<'EOF'
Usage: package_runtime.sh --binary <path> --ffmpeg-prefix <path> --ffmpeg-source <path> --output <dir> [--model-dir <path>]

Stages a Linux x86_64 SeedVR2-ncnn runtime archive with a bundled LGPL FFmpeg prefix.
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

binary=
ffmpeg_prefix=
ffmpeg_source=
output_dir=
model_dir=

while [ "$#" -gt 0 ]; do
    case "$1" in
        --binary)
            require_value "$1" "${2:-}"
            binary=$2
            shift 2
            ;;
        --ffmpeg-prefix)
            require_value "$1" "${2:-}"
            ffmpeg_prefix=$2
            shift 2
            ;;
        --ffmpeg-source)
            require_value "$1" "${2:-}"
            ffmpeg_source=$2
            shift 2
            ;;
        --output)
            require_value "$1" "${2:-}"
            output_dir=$2
            shift 2
            ;;
        --model-dir)
            require_value "$1" "${2:-}"
            model_dir=$2
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

[ "$(uname -s)" = "Linux" ] || die "Linux runtime packaging must run on Linux"
case "$(uname -m)" in
    x86_64|amd64) ;;
    *) die "Linux runtime packaging requires x86_64, got: $(uname -m)" ;;
esac

[ -n "$binary" ] || die "--binary is required"
[ -n "$ffmpeg_prefix" ] || die "--ffmpeg-prefix is required"
[ -n "$ffmpeg_source" ] || die "--ffmpeg-source is required"
[ -n "$output_dir" ] || die "--output is required"

[ -d "$ffmpeg_prefix" ] || die "FFmpeg prefix does not exist: $ffmpeg_prefix"
[ -f "$ffmpeg_source" ] || die "FFmpeg source archive does not exist: $ffmpeg_source"
[ -x "$binary" ] || die "release binary is missing or not executable: $binary"

ffmpeg_lib_dir="$ffmpeg_prefix/lib"
[ -d "$ffmpeg_lib_dir" ] || die "FFmpeg prefix is missing lib/: $ffmpeg_prefix"
[ -f "$ffmpeg_prefix/ffmpeg-build-config.txt" ] || die "FFmpeg prefix is missing ffmpeg-build-config.txt"
[ -f "$ffmpeg_prefix/ffmpeg-license.txt" ] || die "FFmpeg prefix is missing ffmpeg-license.txt"

expected_ffmpeg_source_sha256=$(awk -F= '/^source-sha256=/{print $2; exit}' "$ffmpeg_prefix/ffmpeg-build-config.txt")
[ -n "$expected_ffmpeg_source_sha256" ] || die "FFmpeg prefix build configuration is missing source-sha256"
actual_ffmpeg_source_sha256=$(sha256sum "$ffmpeg_source" | awk '{print $1}')
[ "$actual_ffmpeg_source_sha256" = "$expected_ffmpeg_source_sha256" ] || \
    die "FFmpeg source checksum does not match the build prefix: $ffmpeg_source"

require_ffmpeg_library() {
    library_name=$1
    if ! find "$ffmpeg_lib_dir" -maxdepth 1 -type f -name "${library_name}.so.*" -print -quit | grep -q .; then
        die "FFmpeg prefix is missing ${library_name}.so.*: $ffmpeg_lib_dir"
    fi
}

require_ffmpeg_library libavformat
require_ffmpeg_library libavcodec
require_ffmpeg_library libavutil
require_ffmpeg_library libswscale

command -v readelf >/dev/null 2>&1 || die "readelf is required for runtime dependency validation"
if ! readelf -d "$binary" | grep -Fq "Shared library: [libavformat.so"; then
    die "release binary does not link libavformat; rebuild with SEEDVR2_ENABLE_FFMPEG=ON"
fi
if readelf -d "$binary" | grep -Eq '(RPATH|RUNPATH)'; then
    die "release binary has RPATH/RUNPATH; rebuild with -DCMAKE_SKIP_RPATH=ON"
fi

gomp_library=$(ldd "$binary" | awk '/libgomp\.so\.1 =>/ {print $3; exit}')
[ -n "$gomp_library" ] && [ -f "$gomp_library" ] || \
    die "release binary dependency libgomp.so.1 could not be resolved"
gomp_real_library=$(readlink -f "$gomp_library")
[ -f "$gomp_real_library" ] || die "release binary libgomp target is invalid: $gomp_library"
gomp_license=$(find /usr/share/doc -path '/usr/share/doc/gcc-*-base/copyright' -type f -print -quit 2>/dev/null)
[ -n "$gomp_license" ] && [ -r "$gomp_license" ] || \
    die "GCC runtime license text is unavailable under /usr/share/doc/gcc-*-base"
[ -r /usr/share/common-licenses/GPL-3 ] || die "GPL-3 license text is unavailable under /usr/share/common-licenses"

if [ -n "$model_dir" ] && [ ! -d "$model_dir" ]; then
    die "model directory does not exist: $model_dir"
fi

if [ -n "$model_dir" ] && find "$model_dir" -type l -print -quit | grep -q .; then
    die "model directory contains symbolic links; provide a self-contained model package: $model_dir"
fi

package_name=SeedVR2-ncnn-linux-x86_64
stage_dir="$output_dir/$package_name"
archive_path="$output_dir/$package_name.tar.gz"
[ ! -e "$stage_dir" ] || die "release staging directory already exists: $stage_dir"
[ ! -e "$archive_path" ] || die "release archive already exists: $archive_path"

mkdir -p "$output_dir"
mkdir -p "$stage_dir/bin" "$stage_dir/lib" "$stage_dir/licenses" "$stage_dir/models"

cp "$binary" "$stage_dir/bin/seedvr2-ncnn"
cp "$0" "$stage_dir/licenses/SeedVR2-ncnn-package-script.sh"
cp "$ffmpeg_source" "$stage_dir/licenses/ffmpeg-source.tar.xz"
cp "$ffmpeg_prefix/ffmpeg-build-config.txt" "$stage_dir/licenses/ffmpeg-build-config.txt"
cp "$ffmpeg_prefix/ffmpeg-license.txt" "$stage_dir/licenses/FFmpeg-LGPL-2.1-or-later.txt"
cp "$gomp_license" "$stage_dir/licenses/GCC-runtime-license.txt"
cp /usr/share/common-licenses/GPL-3 "$stage_dir/licenses/GPL-3.0.txt"
cp "$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)/seedvr2-ncnn.sh.in" "$stage_dir/seedvr2-ncnn"
chmod 0755 "$stage_dir/seedvr2-ncnn" "$stage_dir/bin/seedvr2-ncnn"

find "$ffmpeg_lib_dir" -maxdepth 1 \( -type f -o -type l \) -name '*.so*' -exec cp -a '{}' "$stage_dir/lib/" \;
cp "$gomp_real_library" "$stage_dir/lib/"
gomp_link_name=$(basename "$gomp_library")
gomp_real_name=$(basename "$gomp_real_library")
if [ "$gomp_link_name" != "$gomp_real_name" ]; then
    ln -s "$gomp_real_name" "$stage_dir/lib/$gomp_link_name"
fi

if [ -n "$model_dir" ]; then
    cp -a "$model_dir" "$stage_dir/models/seedvr2-3b"
else
    cat >"$stage_dir/models/README.md" <<'EOF'
Place a compatible SeedVR2-ncnn model package in this directory as seedvr2-3b/.
Run the launcher with --model-dir models/seedvr2-3b, or provide another model path.
Model weights are distributed separately under their own license. The model
directory must be self-contained; symbolic-link development directories are
not valid release model packages.
EOF
fi

(
    cd "$stage_dir"
    find . \( -type f -o -type l \) ! -name manifest.sha256 -print | LC_ALL=C sort | sed 's#^./##' | xargs sha256sum > manifest.sha256
)

temporary_tar="$output_dir/.${package_name}.tar"
tar --sort=name --mtime='@0' --owner=0 --group=0 --numeric-owner -C "$output_dir" -cf "$temporary_tar" "$package_name"
gzip -n -c "$temporary_tar" >"$archive_path"
rm -f "$temporary_tar"

printf '%s\n' "package-dir=$stage_dir"
printf '%s\n' "package-archive=$archive_path"
