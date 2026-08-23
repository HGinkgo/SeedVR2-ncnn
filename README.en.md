# SeedVR2-ncnn

An ncnn/Vulkan implementation of SeedVR2, intended to become a portable image and video super-resolution CLI similar to `zimage-ncnn-vulkan`.

This revision is only the minimal buildable skeleton. It includes the ncnn submodule, CLI help/version output, and a CPU build check. SeedVR2 model export, the Adaptive Window Attention custom operator, and actual inference are not included yet.

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

The current program does not load models or process media. Model files, preprocessing, and inference options will be added after the PyTorch baseline and export path are frozen.

## Reference Baseline

`tools/reference/seedvr2_baseline.py` generates reproducible AWA partitions and VAE input/output tensors for export and ncnn operator checks. The VAE baseline requires `SEEDVR2_UPSTREAM_ROOT` and `SEEDVR2_CKPT_DIR`; keep generated files in local `.agent/golden/`.

```bash
conda run -n seedvr2-ncnn pytest -q tests/python
conda run -n seedvr2-ncnn python -m tools.reference.seedvr2_baseline awa \
  --size 1,45,80 --windows 4,3,3 --output-dir .agent/golden/awa
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

This revision contains only the project skeleton. ncnn uses the BSD-3-Clause license. Licenses for SeedVR2 models, weights, and future third-party dependencies will be documented before release.
