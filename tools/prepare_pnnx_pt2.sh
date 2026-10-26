#!/usr/bin/env bash
# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 SeedVR2-ncnn contributors

set -euo pipefail

repo_root=$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
workspace_root=$(CDPATH= cd -- "$repo_root/.." && pwd)
source_dir="$workspace_root/external/ncnn-pnnx-pt2"
build_dir=""
python_bin="python3"
jobs=4

usage() {
    cat <<'EOF'
Usage: prepare_pnnx_pt2.sh [--source DIR] [--build DIR] [--python PATH] [--jobs N]

Prepare the pinned external PNNX checkout and its native ExportedProgram frontend without modifying
SeedVR2's third_party/ncnn or installing host-wide dependencies.
Requires cmake, ctest, ninja, and Python with PyTorch already available.
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

for command_name in cmake ctest ninja; do
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
torch_dir=$("$python_bin" -c 'import pathlib, torch; print(pathlib.Path(torch.__file__).resolve().parent)')
torch_version=$("$python_bin" -c 'import torch; print(torch.__version__)')
python_version=$("$python_bin" --version 2>&1)

if [[ ! -f "$source_dir/tools/pnnx/CMakeLists.txt" ]]; then
    printf 'ERROR: PNNX source tree is missing: %s\n' "$source_dir/tools/pnnx/CMakeLists.txt" >&2
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
cmake --build "$build_dir" --target pnnx_test_exported_program_parameter_roundtrip --parallel "$jobs"
ctest --test-dir "$build_dir" -L '^pt2$' --output-on-failure

printf '\nPNNX PT2 frontend ready\n'
printf '  source tree: %s\n' "$source_dir"
printf '  Python: %s (%s)\n' "$python_version" "$torch_version"
printf '  executable: %s/src/pnnx\n' "$build_dir"
