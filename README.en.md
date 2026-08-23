# SeedVR2-ncnn

An ncnn/Vulkan implementation of SeedVR2, intended to become a portable image and video super-resolution CLI similar to `zimage-ncnn-vulkan`.

This revision includes the ncnn submodule and fixed-shape AWA window, QKV/text sequence, and window-attention reference boundaries. The ncnn/Vulkan inference runtime is still under development.

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

## CLI checks

```bash
./build/seedvr2-ncnn --help
./build/seedvr2-ncnn --version
bash tests/smoke.sh ./build/seedvr2-ncnn
```

The current program does not load models or process media.

## Baseline and export

`tools/reference/seedvr2_baseline.py` generates reproducible AWA/VAE tensors. `tools/export_awa_attention.py` creates a fixed-shape TorchScript file for PNNX inspection. Keep generated files under local `.agent/`.

```bash
conda run -n seedvr2-ncnn pytest -q tests/python
conda run -n seedvr2-ncnn python -m tools.reference.seedvr2_baseline awa \
  --size 1,45,80 --windows 4,3,3 --output-dir .agent/golden/awa
conda run -n seedvr2-ncnn python -m tools.export_awa_attention \
  --output-dir .agent/export/awa-attention --shifted
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

This revision contains the project skeleton and export reference modules. ncnn uses the BSD-3-Clause license. Licenses for SeedVR2 models, weights, and future third-party dependencies will be documented before release.
