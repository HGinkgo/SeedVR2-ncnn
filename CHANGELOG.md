# Changelog

## Unreleased

- Focuses the active product path and CI/package tooling on Linux x86_64 NVIDIA
  Vulkan with BF16 storage and FP32 accumulation where required.
- Replaces five fixed-256 decoder pointwise Conv3D CPU nodes with local Vulkan
  execution and retains OC8 causal Conv3D kernels for two measured nodes.
- Enables ncnn Extractor light mode for fixed-256 DiT tensor lifetime and
  in-place reuse; actual single-frame and 36-frame outputs remain byte-identical
  against the frozen pre-change runtime.
- Adds optional persistent Vulkan pipeline caching for image startup.
- Documents scoped performance results, distinct baselines, and validation
  limits; provides paired 64x64 input and 256x256 output showcase assets.

Older entries below describe the configuration of their historical releases.

## 0.1.2

This release closes the low-resolution Vulkan product path and refreshes the
public showcase:

- Adds opt-in multi-step Euler sampling with `--steps N`; the default one-step
  path remains unchanged and is the release validation path.
- Reuses loaded DiT graphs and Vulkan pipelines within an inference session,
  with optional cold/warm and residency profiling diagnostics.
- Updates the bundled ncnn submodule to the official master revision used by
  the release validation builds.
- Keeps the compact public verification surface and documents the supported
  Linux/Windows x86_64 runtime boundary.
- Replaces the showcase assets with a frame from the official SeedVR2 demo and
  records the source attribution in both READMEs.

The validated target matrix remains `128x128`, `128x256`, and `256x256`.
720p and long-duration video remain outside the supported release boundary.

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
