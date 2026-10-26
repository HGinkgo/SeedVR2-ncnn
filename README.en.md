# SeedVR2-ncnn

Linux x86_64 C++ / ncnn / Vulkan inference for SeedVR2 images and RGB24 AVI video.

## Showcase

Image input/output:

| Input | Output |
| --- | --- |
| <img src="assets/showcase-image-input-64.png" alt="64x64 input" width="256"> | <img src="assets/showcase-image-output-256.png" alt="256x256 output" width="256"> |

Video input/output comparison:

![Video input and output comparison](assets/showcase-video-comparison-bf16.gif)

Full files: [input video](assets/showcase-video-input-64.avi), [output video](assets/showcase-video-output-256.avi).

## Usage

Download the complete ModelScope package:

```bash
modelscope download HGinkgo/SeedVR2-ncnn --local-dir models/seedvr2-3b
```

The default precision is mixed: DiT input, timestep embedding, and the first 9 blocks use FP32; the remaining blocks use BF16. Use `--precision fp32` for a full-FP32 comparison.

```bash
tools/with-nvidia-vulkan-runtime.sh build/seedvr2-ncnn \
  --model-dir models/seedvr2-3b \
  --input input.png --output output.png \
  --width 256 --height 256 --gpu-id 0
```

For video:

```bash
build/seedvr2-ncnn \
  --model-dir models/seedvr2-3b \
  --input input.avi --output output.avi \
  --width 256 --height 256 --gpu-id 0
```

## Build

Requirements: C++17, CMake, Vulkan development files, and the ncnn submodule.

```bash
git submodule update --init --recursive
cmake -S . -B build \
  -DSEEDVR2_ENABLE_VULKAN=ON \
  -DNCNN_BATCH=ON -DNCNN_BF16=ON \
  -DCMAKE_BUILD_TYPE=Release
cmake --build build --target seedvr2-ncnn --parallel
```

Compressed video input additionally requires FFmpeg development files and `-DSEEDVR2_ENABLE_FFMPEG=ON`.

## PT2 / PNNX

PT2 support uses an external ncnn checkout and patch; PNNX source is not vendored in this repository:

```bash
tools/prepare_pnnx_pt2.sh --python /path/to/python --jobs 4
```

The script pins an ncnn commit, applies `patches/pnnx-pt2-independent.patch`, builds PNNX, and runs the basic PT2 tests. See [`patches/README.md`](patches/README.md) for scope and limitations.

## Scope

- Target platform: Linux x86_64 NVIDIA Vulkan.
- Validated output size: up to `256x256`.
- PT2 is an independent conversion validation path; production still uses the TorchScript/ncnn package.

## License

ncnn and model weights retain their own licenses. New project code and patches use BSD-3-Clause.
