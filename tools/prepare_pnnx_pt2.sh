#!/usr/bin/env bash
# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 SeedVR2-ncnn contributors

set -euo pipefail

repo_root=$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
workspace_root=$(CDPATH= cd -- "$repo_root/.." && pwd)
source_dir="$workspace_root/external/ncnn-pnnx-pt2-independent"
build_dir=""
python_bin="python3"
jobs=4
pinned_commit=c6b351b56fbe32e0381ae00331e3df649b20d7b7
remote_url=https://github.com/Tencent/ncnn.git
patch_file="$repo_root/patches/pnnx-pt2-independent.patch"

usage() {
    cat <<'EOF'
Usage: prepare_pnnx_pt2.sh [--source DIR] [--build DIR] [--python PATH] [--jobs N]

Prepare the pinned official PNNX checkout and restricted independent PT2 frontend without modifying
SeedVR2's third_party/ncnn or installing host-wide dependencies.
Requires git, cmake, ctest, ninja, and Python with PyTorch already available.
EOF
}

while (($#)); do
    case "$1" in
        --source) source_dir=${2:?missing --source value}; shift 2 ;;
        --build) build_dir=${2:?missing --build value}; shift 2 ;;
        --python) python_bin=${2:?missing --python value}; shift 2 ;;
        --jobs) jobs=${2:?missing --jobs value}; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) printf 'Unknown argument: %s\n' "$1" >&2; usage >&2; exit 2 ;;
    esac
done

if ! [[ "$jobs" =~ ^[1-9][0-9]*$ ]]; then
    printf 'ERROR: --jobs must be a positive integer\n' >&2
    exit 2
fi

for command_name in git cmake ctest ninja; do
    if ! command -v "$command_name" >/dev/null 2>&1; then
        printf 'ERROR: required command not found: %s\n' "$command_name" >&2
        exit 1
    fi
done
if ! command -v "$python_bin" >/dev/null 2>&1; then
    printf 'ERROR: Python interpreter not found: %s\n' "$python_bin" >&2
    exit 1
fi
source_dir=$("$python_bin" -c 'import pathlib, sys; print(pathlib.Path(sys.argv[1]).expanduser().resolve())' "$source_dir")
case "$source_dir/" in
    "$repo_root/"*)
        printf '%s\n' 'ERROR: PNNX checkout must be outside the SeedVR2 repository.' >&2
        exit 1
        ;;
esac
torch_dir=$("$python_bin" -c 'import pathlib, torch; print(pathlib.Path(torch.__file__).resolve().parent)')
torch_version=$("$python_bin" -c 'import torch; print(torch.__version__)')
python_version=$("$python_bin" --version 2>&1)

if [[ ! -e "$source_dir" ]]; then
    mkdir -p "$(dirname -- "$source_dir")"
    git clone --no-checkout "$remote_url" "$source_dir"
    git -C "$source_dir" checkout --detach "$pinned_commit"
fi
if ! git -C "$source_dir" rev-parse --show-toplevel >/dev/null 2>&1; then
    printf 'ERROR: external source is not a Git checkout: %s\n' "$source_dir" >&2
    exit 1
fi
actual_commit=$(git -C "$source_dir" rev-parse HEAD)
if [[ "$actual_commit" != "$pinned_commit" ]]; then
    printf 'ERROR: expected PNNX source commit %s, found %s; source was left unchanged\n' \
        "$pinned_commit" "$actual_commit" >&2
    exit 1
fi

if git -C "$source_dir" apply --reverse --check "$patch_file" >/dev/null 2>&1; then
    printf '%s\n' 'Independent PT2 patch is already applied.'
elif git -C "$source_dir" apply --check "$patch_file"; then
    if [[ -n "$(git -C "$source_dir" status --porcelain)" ]]; then
        printf '%s\n' 'ERROR: external PNNX checkout has local changes; refusing to apply the patch.' >&2
        exit 1
    fi
    git -C "$source_dir" apply "$patch_file"
else
    printf '%s\n' 'ERROR: patch does not apply cleanly; external checkout was not changed.' >&2
    exit 1
fi

if [[ -z "$build_dir" ]]; then
    build_dir="$source_dir/build-seedvr2-pt2"
fi

cmake -S "$source_dir/tools/pnnx" -B "$build_dir" -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DPython_EXECUTABLE="$python_bin" \
    -DPython3_EXECUTABLE="$python_bin" \
    -DTorch_INSTALL_DIR="$torch_dir" \
    -DPNNX_TNN2PNNX=OFF \
    -DCMAKE_DISABLE_FIND_PACKAGE_Protobuf=TRUE
cmake --build "$build_dir" --target pnnx --parallel "$jobs"
ctest --test-dir "$build_dir" -R '^test_pt2_independent$' --output-on-failure

printf '\nPNNX PT2 frontend ready\n'
printf '  source commit: %s\n' "$actual_commit"
printf '  Python: %s (%s)\n' "$python_version" "$torch_version"
printf '  executable: %s/src/pnnx\n' "$build_dir"
