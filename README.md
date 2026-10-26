# SeedVR2-ncnn

SeedVR2 的 Linux x86_64 C++ / ncnn / Vulkan 推理程序，支持图片和 RGB24 AVI 视频。

## 示例

图片输入/输出：

| 输入 | 输出 |
| --- | --- |
| <img src="assets/showcase-image-input-64.png" alt="64x64 输入" width="256"> | <img src="assets/showcase-image-output-256.png" alt="256x256 输出" width="256"> |

视频输入和输出对比：

![视频输入与输出对比](assets/showcase-video-comparison-bf16.gif)

完整文件：[输入视频](assets/showcase-video-input-64.avi)、[输出视频](assets/showcase-video-output-256.avi)。

## 使用

模型目录使用 ModelScope 发布的完整包：

```bash
modelscope download HGinkgo/SeedVR2-ncnn --local-dir models/seedvr2-3b
```

默认精度为混合模式：DiT 输入、时间嵌入和前 9 个 block 使用 FP32，其余 block 使用 BF16。全 FP32 对照使用 `--precision fp32`。

```bash
tools/with-nvidia-vulkan-runtime.sh build/seedvr2-ncnn \
  --model-dir models/seedvr2-3b \
  --input input.png --output output.png \
  --width 256 --height 256 --gpu-id 0
```

视频输入：

```bash
build/seedvr2-ncnn \
  --model-dir models/seedvr2-3b \
  --input input.avi --output output.avi \
  --width 256 --height 256 --gpu-id 0
```

## 构建

需要 C++17、CMake、Vulkan 开发文件和 ncnn 子模块：

```bash
git submodule update --init --recursive
cmake -S . -B build \
  -DSEEDVR2_ENABLE_VULKAN=ON \
  -DNCNN_BATCH=ON -DNCNN_BF16=ON \
  -DCMAKE_BUILD_TYPE=Release
cmake --build build --target seedvr2-ncnn --parallel
```

压缩视频输入需要额外启用 `-DSEEDVR2_ENABLE_FFMPEG=ON` 并安装 FFmpeg 开发文件。

## PT2 / PNNX

PT2 支持通过外部 ncnn checkout 和 patch 提供，不复制 PNNX 源码到本仓库：

```bash
tools/prepare_pnnx_pt2.sh --python /path/to/python --jobs 4
```

脚本固定 ncnn commit，应用 `patches/pnnx-pt2-independent.patch`，构建 PNNX 并运行基础 PT2 测试。支持范围和限制见 [`patches/README.md`](patches/README.md)。

## 限制

- 当前目标平台为 Linux x86_64 NVIDIA Vulkan。
- 已验证输出上限为 `256x256`。
- PT2 路径为独立转换验证路径，当前生产模型包仍使用 TorchScript/ncnn。

## 许可

ncnn 和模型权重遵循各自许可证。本项目新增代码和 patch 使用 BSD-3-Clause。
