# SeedVR2-ncnn

SeedVR2 的 ncnn/Vulkan 实现项目，目标是提供类似 `zimage-ncnn-vulkan` 的便携式图像与视频超分命令行程序。

当前版本是最小可构建骨架：已经接入 ncnn 子模块，提供 CLI 帮助/版本信息和 CPU 构建检查；SeedVR2 模型导出、Adaptive Window Attention 自定义算子和实际推理功能尚未加入。

## 构建

基础 CPU 构建只需要 CMake 和 C++17 编译器：

```bash
cmake -S . -B build -DSEEDVR2_ENABLE_VULKAN=OFF -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
```

启用 Vulkan 构建需要本机安装 Vulkan SDK、支持 Vulkan 的 GPU 和对应驱动：

```bash
cmake -S . -B build-vulkan -DSEEDVR2_ENABLE_VULKAN=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build-vulkan --parallel
```

## CLI 验证

```bash
./build/seedvr2-ncnn --help
./build/seedvr2-ncnn --version
bash tests/smoke.sh ./build/seedvr2-ncnn
```

当前程序不会加载模型或处理媒体文件。模型、预处理和推理参数将在 PyTorch 基线冻结及导出链路完成后加入。

## 参考基线

可用 `tools/reference/seedvr2_baseline.py` 生成可复现的 AWA 分块和 VAE 输入/输出张量，用于后续导出与 ncnn 算子对齐。VAE 基线需要设置 `SEEDVR2_UPSTREAM_ROOT` 和 `SEEDVR2_CKPT_DIR`；生成文件建议保存在本地 `.agent/golden/`。

```bash
conda run -n seedvr2-ncnn pytest -q tests/python
conda run -n seedvr2-ncnn python -m tools.reference.seedvr2_baseline awa \
  --size 1,45,80 --windows 4,3,3 --output-dir .agent/golden/awa
```

## 目录

```text
SeedVR2-ncnn/
├── src/                 # 应用源码
├── tests/               # CLI 烟测
├── third_party/ncnn/    # ncnn SSH 子模块
├── CMakeLists.txt
└── README.en.md         # English documentation
```

## 许可证

本项目当前只包含工程骨架。ncnn 使用其 BSD-3-Clause 许可证；SeedVR2 模型、权重和后续第三方依赖的许可证将在发布前补充。
