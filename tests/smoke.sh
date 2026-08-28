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
grep -Fq -- '--model-dir' <<<"$help_output"
grep -Fq -- '--input' <<<"$help_output"
grep -Fq -- '--width' <<<"$help_output"

version_output="$($binary --version)"
grep -Fq 'SeedVR2-ncnn' <<<"$version_output"
grep -Fq 'ncnn' <<<"$version_output"

if "$binary" --unknown-option >/dev/null 2>&1; then
    printf 'unknown option unexpectedly succeeded\n' >&2
    exit 1
fi

smoke_image="$(mktemp --suffix=.ppm)"
trap 'rm -f "$smoke_image"' EXIT
printf 'P6\n2 1\n255\n\377\000\000\000\377\000' >"$smoke_image"
set +e
preflight_output="$($binary --model-dir . --input "$smoke_image" 2>&1)"
preflight_status=$?
set -e
[[ "$preflight_status" -eq 1 ]]
grep -Fq 'input=2x1' <<<"$preflight_output"
grep -Fq 'target=' <<<"$preflight_output"
grep -Fq 'model variant' <<<"$preflight_output"

invalid_video="$(mktemp --suffix=.avi)"
trap 'rm -f "$smoke_image" "$invalid_video"' EXIT
printf 'not an avi' >"$invalid_video"
set +e
video_output="$($binary --model-dir . --input "$invalid_video" --output /tmp/seedvr2-invalid-output.avi 2>&1)"
video_status=$?
set -e
[[ "$video_status" -eq 1 ]]
grep -Fq 'not a RIFF AVI video' <<<"$video_output"
