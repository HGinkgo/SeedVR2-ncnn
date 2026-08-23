# SeedVR2-ncnn

SeedVR2 的 ncnn/Vulkan 实现项目，目标是提供类似 `zimage-ncnn-vulkan` 的便携式图像与视频超分命令行程序。

当前版本已接入 ncnn 子模块，并冻结了固定形状的 AWA 窗口、QKV/text 序列和窗口 attention 参考边界；ncnn/Vulkan 推理运行时仍在开发中。

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

当前程序不会加载模型或处理媒体文件。

## 基线与导出

`tools/reference/seedvr2_baseline.py` 生成可复现的 AWA/VAE 张量；`tools/export_awa_attention.py` 生成固定形状 TorchScript，供 PNNX 检查。生成文件建议保存在本地 `.agent/`。

```bash
conda run -n seedvr2-ncnn pytest -q tests/python
conda run -n seedvr2-ncnn python -m tools.reference.seedvr2_baseline awa \
  --size 1,45,80 --windows 4,3,3 --output-dir .agent/golden/awa
conda run -n seedvr2-ncnn python -m tools.export_awa_attention \
  --output-dir .agent/export/awa-attention --shifted
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

本项目当前包含工程骨架和导出参考模块。ncnn 使用其 BSD-3-Clause 许可证；SeedVR2 模型、权重和后续第三方依赖的许可证将在发布前补充。
