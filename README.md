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

需要压缩视频输入时，额外配置 `-DSEEDVR2_ENABLE_FFMPEG=ON`，并提供 FFmpeg 开发库。

## CLI

```bash
./build/seedvr2-ncnn --help
./build/seedvr2-ncnn --version
bash tests/smoke.sh ./build/seedvr2-ncnn
```

当前 CLI 支持 PNG/JPEG 图片，以及逐帧处理未压缩 RGB24 AVI。启用 FFmpeg 后可读取常见压缩视频；视频输出仍为 RGB24 AVI。单图推理使用 Vulkan 构建，模型目录按目标尺寸组织。可用 `--memory-budget-mib` 设置运行前的最小 Vulkan 显存预算（默认 `0`，不预检）。

发布版面向低分辨率输入：自动模式会将输入等比缩放、按 16 像素对齐并居中裁剪到不超过 `256x256` 的目标面积；输入本身低于该上限时不会为填满面积而放大（仅保证合法的 16 像素最小对齐）。显式 `--width/--height` 也必须满足同一面积限制。当前正式验证的目标尺寸为 `128x128`、`128x256` 和 `256x256`，其他满足限制的动态尺寸不作为发布验证承诺。

## 硬件要求

Vulkan 运行需要支持 Vulkan 的 GPU 和驱动。模型权重单独分发；当前 FP32 模型包需要约 10 GiB 以上的 Vulkan heap，显存不足时程序会报告分配失败。Windows GPU 驱动不随运行包分发。

Linux x86_64 runtime 以 Ubuntu 22.04（glibc 2.35）为兼容基线构建；较旧发行版可能需要自行构建。

## 模型

模型包托管在 [ModelScope](https://modelscope.cn/models/HGinkgo/SeedVR2-ncnn)。下载 `model-package-128-materialized.tar.zst` 后，按模型仓库中的 `README.md` 解包并通过 `--model-dir` 指定。

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
