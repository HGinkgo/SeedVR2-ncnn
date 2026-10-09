# SeedVR2-ncnn

Native C++ / ncnn / Vulkan SeedVR2 image and video enhancement CLI for Linux x86_64.

[中文](README.md) · [Build from source](#build) · [Model](https://modelscope.cn/models/HGinkgo/SeedVR2-ncnn)

## Showcase

An official example frame is downsampled to `64x64` and restored with the fixed `256x256` path:

| Input | Output |
| --- | --- |
| <img src="assets/showcase-image-input-64.png" alt="64x64 input" width="256"> | <img src="assets/showcase-image-output-256.png" alt="256x256 output" width="256"> |

The input comes from the [official SeedVR2 demo](https://huggingface.co/spaces/ByteDance-Seed/SeedVR2-3B).

Video inference result (36 frames, `64x64` input restored to `256x256`, RGB24 AVI):

The comparison GIF shows the `64x64` input on the left and the ncnn Vulkan `256x256` enhanced output on the right:

![Official video input and SeedVR2-ncnn output comparison](assets/showcase-video-comparison-bf16.gif)

<video controls width="256" preload="metadata">
  <source src="assets/showcase-video-output-256.avi" type="video/x-msvideo">
</video>

The complete [input video](assets/showcase-video-input-64.avi) and ncnn Vulkan [output video](assets/showcase-video-output-256.avi) are included. The input is made by bilinearly downsampling the first 36 frames of the official `256x256` example video to `64x64`. This file showcases the ncnn Vulkan inference output and is not an independent video-quality acceptance result.

## Features

- Native C++ inference without Python, PyTorch, or a CUDA runtime.
- Linux NVIDIA Vulkan with BF16 storage and FP32 accumulation where required.
- PNG/JPEG images and RGB24 AVI video.
- Validated fixed targets: `128x128`, `128x256`, and `256x256`.
- Current optimization acceptance targets RTX 3090, one GPU, fixed `256x256` BF16 Vulkan.
- One-step Euler is the default; multi-step sampling and VAE tiling are experimental.
- Output uses reference-guided color reconstruction, retaining generated high-frequency detail while rebuilding low-frequency color from the input.

## Quick Start

Follow [Build](#build) to compile from source, then run the commands below from the repository root.

Download the model:

```bash
modelscope download HGinkgo/SeedVR2-ncnn --local-dir models/seedvr2-3b
```

Image:

```bash
build/seedvr2-ncnn \
  --model-dir models/seedvr2-3b \
  --input input.png \
  --output output.png \
  --width 256 --height 256 \
  --gpu-id 0
```

Video:

```bash
build/seedvr2-ncnn \
  --model-dir models/seedvr2-3b \
  --input input.avi \
  --output output.avi \
  --width 256 --height 256 \
  --frames 36 \
  --gpu-id 0
```

This processes the first 36 input frames and writes an RGB24 AVI. Omit `--frames` to process the whole video. The default build supports AVI; compressed video input requires a build with `-DSEEDVR2_ENABLE_FFMPEG=ON`. The runtime does not require Python, PyTorch, or CUDA. Use `--help` for all options.

Set `SEEDVR2_PIPELINE_CACHE_PATH` to enable persistent Vulkan pipeline caching. A matching cache helps repeated image startup; a missing or incompatible cache is rebuilt:

```bash
mkdir -p "$HOME/.cache/seedvr2-ncnn"
SEEDVR2_PIPELINE_CACHE_PATH="$HOME/.cache/seedvr2-ncnn/pipeline.cache" \
  build/seedvr2-ncnn \
  --model-dir models/seedvr2-3b --input input.png --output output.png --gpu-id 0
```

## Validation

| Target | Image | AVI |
| --- | --- | --- |
| `128x128` | verified | verified, 2 frames |
| `128x256` | verified | - |
| `256x256` | verified | verified, 36 frames |

The fixed `256x256` path retains three optimizations: Vulkan pointwise Conv3D
for five decoder nodes, OC8 output-channel reuse for two causal Conv3D nodes,
and DiT intermediate tensor lifetime and in-place reuse.

The latest runtime comparison measures `343.65 s -> 337.07 s` for one 36-frame
pair, with `3.55%` lower DiT stage latency and byte-identical output. Per-block
peak live temporary memory falls by `84.6%`; whole-video peak GPU memory is
unchanged. See [performance and validation](docs/performance.md) for the separate
frozen baselines. These gains cannot be added or directly compared with older
end-to-end timings.

## Build

Use a C++17 compiler, CMake, Vulkan development files, and the pinned ncnn
submodule:

```bash
git submodule update --init --recursive
cmake -S . -B build \
  -DSEEDVR2_ENABLE_VULKAN=ON \
  -DNCNN_BATCH=ON \
  -DNCNN_BF16=ON \
  -DNCNN_BENCHMARK=OFF \
  -DCMAKE_BUILD_TYPE=Release
cmake --build build --target seedvr2-ncnn --parallel
```

The model directory must be the complete dynamic package, containing
`manifest.sha256` with 75 entries and no symbolic links. Pass that directory
to `--model-dir`.

Inside this source workspace, use the prepared NVIDIA runtime:

```bash
tools/with-nvidia-vulkan-runtime.sh build/seedvr2-ncnn \
  --model-dir models/seedvr2-3b --input input.png --output output.png \
  --width 256 --height 256 --gpu-id 0
```

The helper selects NVIDIA user-space libraries from the ignored `runtime/`
directory. Environments with a configured NVIDIA Vulkan driver can run
`build/seedvr2-ncnn` directly.
Compressed video input additionally requires FFmpeg development files and
`-DSEEDVR2_ENABLE_FFMPEG=ON`. The ModelScope CLI is only a download tool, not an
inference dependency.

## Scope

- Current product line: Linux x86_64 NVIDIA Vulkan.
- Output target is capped at `256x256`; 720p and long video are not validated.
- Model weights and third-party dependencies retain their own licenses.

## License

ncnn uses BSD-3-Clause. SeedVR2 models, weights, and other third-party dependencies retain their respective licenses.
