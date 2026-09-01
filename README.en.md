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

The release product targets low-resolution workloads. Automatic mode uniformly resizes each input, aligns the target to 16 pixels, and center-crops it to an area no larger than `256x256`; inputs already below that limit are not upscaled just to fill the area (the minimum legal alignment is still 16 pixels). Explicit `--width/--height` values use the same area limit. The validated release targets are `128x128`, `128x256`, and `256x256`; other dynamic shapes within the limit are not part of the release validation promise.

## Testing

Default CTest contains the fast native unit and backend-contract tests: normally 12 tests for a CPU build and 17 for a Vulkan build.

```bash
cmake -S . -B build -DSEEDVR2_ENABLE_VULKAN=OFF -DSEEDVR2_BUILD_TESTS=ON
cmake --build build --parallel 2
ctest --test-dir build --output-on-failure
```

The model-backed graph-load, golden, and end-to-end tests in `tests/integration/cpp/` are built only when `-DSEEDVR2_BUILD_INTEGRATION_TESTS=ON` is set, and are not part of default CTest. `tests/python/` covers export and model-package tooling; `tests/manual/package/` checks the Linux and Windows runtime-package contracts. See [`tests/README.md`](tests/README.md) for the test-tier map.

## Hardware

Vulkan inference requires a Vulkan-capable GPU and driver. Model weights are distributed separately; the current FP32 model package normally needs roughly 10 GiB or more of Vulkan heap, and insufficient memory is reported as an allocation failure. RTX 3090 is the current low-resolution acceptance baseline; 6 GiB-class GPUs are not an end-to-end support target. Windows GPU drivers are not bundled.

The Linux x86_64 runtime is built against Ubuntu 22.04 (glibc 2.35); older distributions may require a local build.

## Model

The model package is hosted on [ModelScope](https://modelscope.cn/models/HGinkgo/SeedVR2-ncnn). The release uses a flat dynamic package: the extracted directory must contain `manifest.sha256` with 75 records and no symbolic links. Pass that directory with `--model-dir`. Use the model repository `README.md` for the current archive filename; legacy fixed-resolution directories remain a compatibility fallback, not a release validation promise.

## Layout

```text
SeedVR2-ncnn/
├── src/                 # Application source
├── tests/               # Unit, integration, Python, and runtime-contract tests
├── third_party/ncnn/    # ncnn SSH submodule
├── CMakeLists.txt
└── README.en.md         # English documentation
```

## License

ncnn uses the BSD-3-Clause license. SeedVR2 models, weights, and other third-party dependencies are subject to their respective licenses.
