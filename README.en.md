# SeedVR2-ncnn

Native C++ / ncnn / Vulkan SeedVR2 image and video enhancement CLI for Linux x86_64.

[中文](README.md) · [Runtime](https://github.com/HGinkgo/SeedVR2-ncnn/releases/latest) · [Model](https://modelscope.cn/models/HGinkgo/SeedVR2-ncnn)

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
- One-step Euler is the default; multi-step sampling and VAE tiling are experimental.

## Quick Start

Download the model:

```bash
modelscope download HGinkgo/SeedVR2-ncnn --local-dir models/seedvr2-3b
```

Image:

```bash
./seedvr2-ncnn \
  --model-dir models/seedvr2-3b \
  --input input.png \
  --output output.png \
  --width 256 --height 256 \
  --gpu-id 0
```

Video:

```bash
tools/with-nvidia-vulkan-runtime.sh \
  ./seedvr2-ncnn \
  --model-dir models/seedvr2-3b \
  --input input.avi \
  --output output.avi \
  --width 256 --height 256 \
  --frames 36 \
  --gpu-id 0
```

This processes the first 36 input frames and writes an RGB24 AVI. Omit `--frames` to process the whole video. The default build supports AVI; compressed video input requires a build with `-DSEEDVR2_ENABLE_FFMPEG=ON`. The runtime does not require Python, PyTorch, or CUDA. Use `--help` for all options.

## Validation

Fixed-input regression environment: RTX 3090, driver `580.95.05`, ncnn
`c6b351b56fbe32e0381ae00331e3df649b20d7b7`, BF16 Vulkan.

| Target | Image | AVI |
| --- | --- | --- |
| `128x128` | verified | verified, 2 frames |
| `128x256` | verified | - |
| `256x256` | verified | verified, 36 frames |

The `256x256 / 36-frame` baseline is `660.7 s` video-batch time, `664.8 s`
end-to-end time, and `1515 MiB` peak RSS. This is a performance baseline,
not a cross-engine comparison or a video-quality acceptance result.

## Build

CPU:

```bash
cmake -S . -B build \
  -DSEEDVR2_ENABLE_VULKAN=OFF \
  -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
```

Vulkan:

```bash
cmake -S . -B build-vulkan \
  -DSEEDVR2_ENABLE_VULKAN=ON \
  -DCMAKE_BUILD_TYPE=Release
cmake --build build-vulkan --parallel
```

Compressed video input additionally requires `-DSEEDVR2_ENABLE_FFMPEG=ON`.

## Scope

- Current product line: Linux x86_64 NVIDIA Vulkan.
- Output target is capped at `256x256`; 720p and long video are not validated.
- Model weights and third-party dependencies retain their own licenses.

## License

ncnn uses BSD-3-Clause. SeedVR2 models, weights, and other third-party dependencies retain their respective licenses.
