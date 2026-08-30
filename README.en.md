# SeedVR2-ncnn

An ncnn/Vulkan implementation of SeedVR2, intended to become a portable image and video super-resolution CLI similar to `zimage-ncnn-vulkan`.

## Build

The base CPU build only requires CMake and a C++17 compiler:

```bash
cmake -S . -B build -DSEEDVR2_ENABLE_VULKAN=OFF -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
```

The Vulkan build requires a local Vulkan SDK, a Vulkan-capable GPU, and a compatible driver:

```bash
cmake -S . -B build-vulkan -DSEEDVR2_ENABLE_VULKAN=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build-vulkan --parallel
```

For compressed video input, also configure `-DSEEDVR2_ENABLE_FFMPEG=ON` and provide FFmpeg development libraries.

## CLI

```bash
./build/seedvr2-ncnn --help
./build/seedvr2-ncnn --version
bash tests/smoke.sh ./build/seedvr2-ncnn
```

The CLI supports PNG/JPEG images and frame-by-frame processing of uncompressed RGB24 AVI. With FFmpeg enabled, common compressed video inputs are also accepted; video output remains RGB24 AVI. Image inference uses the Vulkan build and shape-aware model directories. Use `--memory-budget-mib` to require a minimum Vulkan heap budget before running (default `0`, disabled).

## Hardware

Vulkan inference requires a Vulkan-capable GPU and driver. Model weights are distributed separately; the current FP32 model package needs roughly 10 GiB or more of Vulkan heap, and insufficient memory is reported as an allocation failure. Windows GPU drivers are not bundled.

The Linux x86_64 runtime is built against Ubuntu 22.04 (glibc 2.35); older distributions may require a local build.

## Model

The model package is hosted on [ModelScope](https://modelscope.cn/models/HGinkgo/SeedVR2-ncnn). Download `model-package-128-materialized.tar.zst`, extract it as described in the model repository `README.md`, and pass the extracted directory with `--model-dir`.

## Layout

```text
SeedVR2-ncnn/
├── src/                 # Application source
├── tests/               # CLI smoke test
├── third_party/ncnn/    # ncnn SSH submodule
├── CMakeLists.txt
└── README.en.md         # English documentation
```

## License

ncnn uses the BSD-3-Clause license. SeedVR2 models, weights, and other third-party dependencies are subject to their respective licenses.
