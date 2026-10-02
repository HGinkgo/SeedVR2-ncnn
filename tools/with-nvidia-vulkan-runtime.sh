#!/bin/sh

set -eu

usage() {
    printf '%s\n' \
        "Usage: $0 [command [argument ...]]" \
        "Run a command with the repository's NVIDIA Vulkan runtime selected." \
        "Set SEEDVR2_NVIDIA_VULKAN_ROOT to override the runtime directory."
}

if [ "$#" -eq 0 ]; then
    usage >&2
    exit 2
fi

script_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
repo_root=$(CDPATH= cd -- "$script_dir/.." && pwd)

if [ -n "${SEEDVR2_NVIDIA_VULKAN_ROOT:-}" ]; then
    runtime_root=$SEEDVR2_NVIDIA_VULKAN_ROOT
else
    runtime_root=
    runtime_count=0
    for candidate in "$repo_root"/runtime/nvidia-vulkan-*/usr; do
        [ -d "$candidate" ] || continue
        runtime_root=$candidate
        runtime_count=$((runtime_count + 1))
    done
    if [ "$runtime_count" -ne 1 ]; then
        printf '%s\n' \
            "Unable to identify one repository NVIDIA Vulkan runtime under $repo_root/runtime." \
            "Set SEEDVR2_NVIDIA_VULKAN_ROOT=/path/to/runtime/usr or stage the packaged runtime." >&2
        exit 1
    fi
fi

library_dir=$runtime_root/lib/x86_64-linux-gnu
icd_file=$runtime_root/share/vulkan/icd.d/nvidia_icd.json
egl_file=$runtime_root/share/glvnd/egl_vendor.d/10_nvidia.json

[ -d "$library_dir" ] || {
    printf '%s\n' "NVIDIA Vulkan runtime library directory is missing: $library_dir" >&2
    exit 1
}
[ -f "$icd_file" ] || {
    printf '%s\n' "NVIDIA Vulkan ICD file is missing: $icd_file" >&2
    exit 1
}
[ -f "$egl_file" ] || {
    printf '%s\n' "NVIDIA EGL vendor file is missing: $egl_file" >&2
    exit 1
}

if [ -n "${LD_LIBRARY_PATH:-}" ]; then
    LD_LIBRARY_PATH="$library_dir:$LD_LIBRARY_PATH"
else
    LD_LIBRARY_PATH=$library_dir
fi

export LD_LIBRARY_PATH
export VK_ICD_FILENAMES=$icd_file
export __EGL_VENDOR_LIBRARY_FILENAMES=$egl_file

exec "$@"
