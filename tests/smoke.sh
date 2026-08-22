#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
    printf 'usage: %s /path/to/seedvr2-ncnn\n' "$0" >&2
    exit 2
fi

binary="$1"

help_output="$($binary --help)"
grep -Fq 'Usage: seedvr2-ncnn' <<<"$help_output"
grep -Fq -- '--version' <<<"$help_output"

version_output="$($binary --version)"
grep -Fq 'SeedVR2-ncnn' <<<"$version_output"
grep -Fq 'ncnn' <<<"$version_output"

if "$binary" --unknown-option >/dev/null 2>&1; then
    printf 'unknown option unexpectedly succeeded\n' >&2
    exit 1
fi
