# SeedVR2-ncnn

SeedVR2 的 ncnn/Vulkan 实现项目，目标是提供类似 `zimage-ncnn-vulkan` 的便携式图像与视频超分命令行程序。

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

## CLI

```bash
./build/seedvr2-ncnn --help
./build/seedvr2-ncnn --version
bash tests/smoke.sh ./build/seedvr2-ncnn
```

当前 CLI 支持 PNG/JPEG 图片，以及未压缩 RGB24 AVI 的逐帧处理。视频输入需要使用 AVI 输出；单图推理使用 Vulkan 构建，模型目录按目标尺寸组织。可用 `--memory-budget-mib` 设置运行前的最小 Vulkan 显存预算（默认 `0`，不预检）。

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

ncnn 使用 BSD-3-Clause 许可证；SeedVR2 模型、权重和其他第三方依赖遵循其各自许可证。
