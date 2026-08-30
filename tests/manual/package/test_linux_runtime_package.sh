#!/bin/sh

set -eu

repo_root=$(CDPATH= cd -- "$(dirname -- "$0")/../../.." && pwd)
packager="$repo_root/packaging/linux/package_runtime.sh"
ffmpeg_builder="$repo_root/packaging/linux/build_ffmpeg_lgpl.sh"
temp_dir=$(mktemp -d "${TMPDIR:-/tmp}/seedvr2-package-test.XXXXXX")
trap 'rm -rf "$temp_dir"' EXIT HUP INT TERM

if [ ! -x "$packager" ]; then
    printf '%s\n' "FAIL: package runtime script is missing: $packager" >&2
    exit 1
fi

set +e
"$packager" \
    --binary /bin/true \
    --ffmpeg-prefix "$temp_dir/missing-prefix" \
    --ffmpeg-source "$temp_dir/missing-source.tar.xz" \
    --output "$temp_dir/release" \
    >"$temp_dir/package.log" 2>&1
status=$?
set -e

if [ "$status" -eq 0 ]; then
    printf '%s\n' "FAIL: packager accepted a missing FFmpeg prefix" >&2
    exit 1
fi

if ! grep -Fq "FFmpeg prefix does not exist" "$temp_dir/package.log"; then
    printf '%s\n' "FAIL: missing FFmpeg prefix error was not actionable" >&2
    cat "$temp_dir/package.log" >&2
    exit 1
fi

printf '%s\n' "seedvr2 Linux runtime package contract: ok"

if [ ! -x "$ffmpeg_builder" ]; then
    printf '%s\n' "FAIL: FFmpeg builder script is missing: $ffmpeg_builder" >&2
    exit 1
fi

set +e
"$ffmpeg_builder" \
    --prefix "$temp_dir/ffmpeg-prefix" \
    --source-tarball "$temp_dir/missing-ffmpeg-source.tar.xz" \
    >"$temp_dir/builder.log" 2>&1
status=$?
set -e

if [ "$status" -eq 0 ]; then
    printf '%s\n' "FAIL: FFmpeg builder accepted a missing source tarball" >&2
    exit 1
fi

if ! grep -Fq "FFmpeg source archive does not exist" "$temp_dir/builder.log"; then
    printf '%s\n' "FAIL: missing FFmpeg source error was not actionable" >&2
    cat "$temp_dir/builder.log" >&2
    exit 1
fi

if [ -e "$temp_dir/ffmpeg-prefix" ]; then
    printf '%s\n' "FAIL: FFmpeg builder created a prefix before source validation" >&2
    exit 1
fi

printf '%s\n' "seedvr2 FFmpeg builder contract: ok"

if [ "$#" -eq 0 ]; then
    exit 0
fi

if [ "$#" -ne 1 ]; then
    printf '%s\n' "Usage: test_linux_runtime_package.sh [--release-smoke|--release-video-smoke|--reject-linked-model|--reject-mismatched-source]" >&2
    exit 2
fi

require_release_input() {
    variable_name=$1
    eval "variable_value=\${$variable_name:-}"
    if [ -z "$variable_value" ]; then
        printf '%s\n' "FAIL: --release-smoke requires $variable_name" >&2
        exit 1
    fi
}

require_release_input SEEDVR2_PACKAGE_TEST_BINARY
require_release_input SEEDVR2_PACKAGE_TEST_FFMPEG_PREFIX
require_release_input SEEDVR2_PACKAGE_TEST_FFMPEG_SOURCE

case "$1" in
    --release-smoke)
        release_output="$temp_dir/release-output"
        "$packager" \
            --binary "$SEEDVR2_PACKAGE_TEST_BINARY" \
            --ffmpeg-prefix "$SEEDVR2_PACKAGE_TEST_FFMPEG_PREFIX" \
            --ffmpeg-source "$SEEDVR2_PACKAGE_TEST_FFMPEG_SOURCE" \
            --output "$release_output"

        release_root="$release_output/SeedVR2-ncnn-linux-x86_64"
        for required_path in \
            "$release_root/seedvr2-ncnn" \
            "$release_root/bin/seedvr2-ncnn" \
            "$release_root/lib/libavformat.so" \
            "$release_root/lib/libavcodec.so" \
            "$release_root/lib/libavutil.so" \
            "$release_root/lib/libswscale.so" \
            "$release_root/lib/libgomp.so.1" \
            "$release_root/licenses/FFmpeg-LGPL-2.1-or-later.txt" \
            "$release_root/licenses/GCC-runtime-license.txt" \
            "$release_root/licenses/GPL-3.0.txt" \
            "$release_root/licenses/ffmpeg-source.tar.xz" \
            "$release_root/licenses/ffmpeg-build-config.txt" \
            "$release_root/manifest.sha256" \
            "$release_root/models/README.md" \
            "$release_output/SeedVR2-ncnn-linux-x86_64.tar.gz"; do
            if [ ! -e "$required_path" ]; then
                printf '%s\n' "FAIL: staged package is missing $required_path" >&2
                exit 1
            fi
        done

        if ! tar -tzf "$release_output/SeedVR2-ncnn-linux-x86_64.tar.gz" | grep -Fq 'SeedVR2-ncnn-linux-x86_64/bin/seedvr2-ncnn'; then
            printf '%s\n' "FAIL: archive is missing executable payload" >&2
            exit 1
        fi

        if ! env -i PATH="$PATH" "$release_root/seedvr2-ncnn" --help >"$temp_dir/package-help.log" 2>&1; then
            printf '%s\n' "FAIL: staged launcher could not run --help" >&2
            cat "$temp_dir/package-help.log" >&2
            exit 1
        fi

        if ! grep -Fq 'Usage: seedvr2-ncnn' "$temp_dir/package-help.log"; then
            printf '%s\n' "FAIL: staged launcher did not reach the application" >&2
            cat "$temp_dir/package-help.log" >&2
            exit 1
        fi

        if LD_LIBRARY_PATH="$release_root/lib" ldd "$release_root/bin/seedvr2-ncnn" | grep -Fq 'not found'; then
            printf '%s\n' "FAIL: staged executable has unresolved dynamic libraries" >&2
            exit 1
        fi

        if ! LD_LIBRARY_PATH="$release_root/lib" ldd "$release_root/bin/seedvr2-ncnn" | \
            grep -Fq "libgomp.so.1 => $release_root/lib/libgomp.so.1"; then
            printf '%s\n' "FAIL: staged executable did not resolve libgomp from the package" >&2
            exit 1
        fi

        if readelf --version-info "$release_root/bin/seedvr2-ncnn" | \
            grep -Eq 'GLIBC_2\.(3[6-9]|[4-9][0-9])'; then
            printf '%s\n' "FAIL: staged executable requires glibc newer than the Ubuntu 22.04 release baseline" >&2
            exit 1
        fi

        if ! grep -Fq '  lib/libavformat.so' "$release_root/manifest.sha256"; then
            printf '%s\n' "FAIL: manifest does not cover FFmpeg library links" >&2
            exit 1
        fi

        printf '%s\n' "seedvr2 Linux runtime package release smoke: ok"
        ;;
    --reject-linked-model)
        linked_model_dir="$temp_dir/linked-model"
        model_store="$temp_dir/model-store"
        mkdir -p "$linked_model_dir/128x128" "$model_store"
        : >"$model_store/vae_encode.ncnn.param"
        ln -s ../../model-store/vae_encode.ncnn.param "$linked_model_dir/128x128/vae_encode.ncnn.param"

        set +e
        "$packager" \
            --binary "$SEEDVR2_PACKAGE_TEST_BINARY" \
            --ffmpeg-prefix "$SEEDVR2_PACKAGE_TEST_FFMPEG_PREFIX" \
            --ffmpeg-source "$SEEDVR2_PACKAGE_TEST_FFMPEG_SOURCE" \
            --model-dir "$linked_model_dir" \
            --output "$temp_dir/linked-model-release" \
            >"$temp_dir/linked-model.log" 2>&1
        status=$?
        set -e

        if [ "$status" -eq 0 ]; then
            printf '%s\n' "FAIL: packager accepted a linked model directory" >&2
            exit 1
        fi

        if ! grep -Fq "model directory contains symbolic links" "$temp_dir/linked-model.log"; then
            printf '%s\n' "FAIL: linked model rejection was not actionable" >&2
            cat "$temp_dir/linked-model.log" >&2
            exit 1
        fi

        printf '%s\n' "seedvr2 linked model package contract: ok"
        ;;
    --reject-mismatched-source)
        mismatched_source="$temp_dir/mismatched-source.tar.xz"
        : >"$mismatched_source"

        set +e
        "$packager" \
            --binary "$SEEDVR2_PACKAGE_TEST_BINARY" \
            --ffmpeg-prefix "$SEEDVR2_PACKAGE_TEST_FFMPEG_PREFIX" \
            --ffmpeg-source "$mismatched_source" \
            --output "$temp_dir/mismatched-source-release" \
            >"$temp_dir/mismatched-source.log" 2>&1
        status=$?
        set -e

        if [ "$status" -eq 0 ]; then
            printf '%s\n' "FAIL: packager accepted an FFmpeg source archive that did not build the prefix" >&2
            exit 1
        fi

        if ! grep -Fq "FFmpeg source checksum does not match the build prefix" "$temp_dir/mismatched-source.log"; then
            printf '%s\n' "FAIL: FFmpeg source mismatch rejection was not actionable" >&2
            cat "$temp_dir/mismatched-source.log" >&2
            exit 1
        fi

        printf '%s\n' "seedvr2 FFmpeg source matching contract: ok"
        ;;
    --release-video-smoke)
        require_release_input SEEDVR2_PACKAGE_TEST_MODEL_DIR
        require_release_input SEEDVR2_PACKAGE_TEST_VIDEO_INPUT
        require_release_input SEEDVR2_PACKAGE_TEST_NVIDIA_RUNTIME

        release_output="$temp_dir/release-video-output"
        "$packager" \
            --binary "$SEEDVR2_PACKAGE_TEST_BINARY" \
            --ffmpeg-prefix "$SEEDVR2_PACKAGE_TEST_FFMPEG_PREFIX" \
            --ffmpeg-source "$SEEDVR2_PACKAGE_TEST_FFMPEG_SOURCE" \
            --output "$release_output"

        release_root="$release_output/SeedVR2-ncnn-linux-x86_64"
        output_video="$temp_dir/output.avi"
        nvidia_runtime=$SEEDVR2_PACKAGE_TEST_NVIDIA_RUNTIME
        if ! env -i \
            PATH="$PATH" \
            LD_LIBRARY_PATH="$nvidia_runtime/usr/lib/x86_64-linux-gnu" \
            VK_ICD_FILENAMES="$nvidia_runtime/usr/share/vulkan/icd.d/nvidia_icd.json" \
            __EGL_VENDOR_LIBRARY_FILENAMES="$nvidia_runtime/usr/share/glvnd/egl_vendor.d/10_nvidia.json" \
            "$release_root/seedvr2-ncnn" \
                --model-dir "$SEEDVR2_PACKAGE_TEST_MODEL_DIR" \
                --input "$SEEDVR2_PACKAGE_TEST_VIDEO_INPUT" \
                --output "$output_video" \
                --width 128 \
                --height 128 \
                --gpu-id 0 \
            >"$temp_dir/release-video.log" 2>&1; then
            printf '%s\n' "FAIL: packaged launcher could not process compressed video" >&2
            cat "$temp_dir/release-video.log" >&2
            exit 1
        fi

        if ! grep -Fq 'seedvr2-video-inference: ok' "$temp_dir/release-video.log"; then
            printf '%s\n' "FAIL: packaged video smoke did not reach success marker" >&2
            cat "$temp_dir/release-video.log" >&2
            exit 1
        fi

        frame_stages=$(grep -c '^stage=video-frame index=' "$temp_dir/release-video.log" || true)
        if [ "$frame_stages" -ne 2 ] || [ ! -s "$output_video" ]; then
            printf '%s\n' "FAIL: packaged video smoke did not produce two output frames" >&2
            cat "$temp_dir/release-video.log" >&2
            exit 1
        fi

        printf '%s\n' "seedvr2 Linux runtime package video smoke: ok"
        ;;
    *)
        printf '%s\n' "Usage: test_linux_runtime_package.sh [--release-smoke|--release-video-smoke|--reject-linked-model|--reject-mismatched-source]" >&2
        exit 2
        ;;
esac
