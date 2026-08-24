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

## CLI

```bash
./build/seedvr2-ncnn --help
./build/seedvr2-ncnn --version
bash tests/smoke.sh ./build/seedvr2-ncnn
```

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
