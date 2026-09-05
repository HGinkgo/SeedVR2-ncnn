# Changelog

## 0.1.1

This release aligns the portable runtime with the current low-resolution
product path:

- Reference-guided low-frequency color reconstruction is now applied to image
  and video output by default while preserving generated high-frequency detail.
- The experimental `--vae-tile-size` path reduces VAE activation residency for
  tiled image and video processing without changing model weights or the
  default full-frame path.
- The CLI adds integer `--scale`, directory image batches, bounded video
  segments, and `--check-model` package validation.
- Model-package validation now verifies every manifest hash, rejects missing or
  extra files and unsafe paths, and requires a package without symbolic links.

The validated target matrix remains `128x128`, `128x256`, and `256x256`.
720p and long-duration video remain outside the supported release boundary.

## 0.1.0

The first low-resolution release line provides:

- FP32 Vulkan image enhancement for the validated `128x128`, `128x256`, and `256x256` targets.
- PNG/JPEG image input and RGB24 AVI video input/output, with optional LGPL FFmpeg input support.
- Dynamic shape planning with a `256x256` area cap, 16-pixel alignment, and bounded CLI errors for larger explicit targets.
- Two-frame internal video batches that reuse the encoder, DiT stack, and decoder loads while preserving staged Vulkan residency.
- Portable Linux and Windows runtime-package builds with dependency and package-contract checks in CI.
- Optional `--profile` timing output for image/video stages and aggregate DiT parameter/bin loading.

Known release limits:

- The model is FP32 and normally needs about 10 GiB or more of Vulkan heap.
- 720p and long-duration video are not part of this release line; 720p currently reaches cumulative VAE encoder Vulkan memory exhaustion on the RTX 3090 baseline.
- Dynamic shapes other than the three validated targets are accepted only within the CLI area/alignment policy and are not individually promised by the release validation matrix.
