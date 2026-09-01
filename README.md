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

## 测试

默认 CTest 只包含快速的原生单元和后端契约测试：CPU 构建通常为 12 项，Vulkan 构建为 17 项。

```bash
cmake -S . -B build -DSEEDVR2_ENABLE_VULKAN=OFF -DSEEDVR2_BUILD_TESTS=ON
cmake --build build --parallel 2
ctest --test-dir build --output-on-failure
```

`tests/integration/cpp/` 中的模型加载、golden 和端到端测试需要本地模型/参考数据，只有显式配置 `-DSEEDVR2_BUILD_INTEGRATION_TESTS=ON` 才会构建，不会进入默认 CTest。`tests/python/` 验证导出和模型包工具；`tests/manual/package/` 验证 Linux/Windows runtime 包契约。各层的用途见 [`tests/README.md`](tests/README.md)。

## 硬件要求

Vulkan 运行需要支持 Vulkan 的 GPU 和驱动。模型权重单独分发；当前 FP32 模型包通常需要约 10 GiB 以上的 Vulkan heap，显存不足时程序会报告分配失败。RTX 3090 是当前低分辨率验收基线；6 GiB 级显卡不属于端到端支持目标。Windows GPU 驱动不随运行包分发。

Linux x86_64 runtime 以 Ubuntu 22.04（glibc 2.35）为兼容基线构建；较旧发行版可能需要自行构建。

## 模型

模型包托管在 [ModelScope](https://modelscope.cn/models/HGinkgo/SeedVR2-ncnn)。发布版使用扁平动态模型包：解包后应包含 `manifest.sha256`（75 条记录），且包内不得有符号链接；通过 `--model-dir` 指定解包目录。模型仓库中的 `README.md` 以当前发布包文件名为准；旧的固定分辨率目录仅作为兼容回退，不是当前发布验证承诺。

## 目录

```text
SeedVR2-ncnn/
├── src/                 # 应用源码
├── tests/               # 单元、集成、Python 和 runtime 契约测试
├── third_party/ncnn/    # ncnn SSH 子模块
├── CMakeLists.txt
└── README.en.md         # English documentation
```

## 许可证

ncnn 使用 BSD-3-Clause 许可证；SeedVR2 模型、权重和其他第三方依赖遵循其各自许可证。
