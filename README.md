# SeedVR2-ncnn

面向 Linux x86_64 的 SeedVR2 原生 C++ / ncnn / Vulkan 图像与视频增强命令行工具。

[下载运行包](https://github.com/HGinkgo/SeedVR2-ncnn/releases/latest) | [下载模型](https://modelscope.cn/models/HGinkgo/SeedVR2-ncnn) | [English](README.en.md)

## 效果展示

下列结果使用官方 SeedVR2 示例输入展示；输出图与输入图来自同一帧。当前验证线的最大目标为 `256x256`。

### 官方示例图像增强

左图为 SeedVR2 官方视频示例的一帧降采样到 `64x64` 后的输入，右图为显式 `256x256` 目标的实际输出。展示流程先将同一图像放大到 `256x256` 模型画布，再执行已验证的固定尺寸路径。

| 输入 | 输出 |
| --- | --- |
| <img src="assets/showcase-image-input-64.png" alt="64x64 SeedVR2 官方示例低分辨率输入图" width="256"> | <img src="assets/showcase-image-output-256.png" alt="SeedVR2-ncnn 256x256 官方示例输出图" width="256"> |

### 官方示例视频推理

左侧是官方示例视频输入，右侧是本项目 BF16 Vulkan 路径处理后的输出。仓库同时提供完整的 [输入视频](assets/showcase-video-official-1_1-ncnn-256.avi) 和 [输出视频](assets/showcase-video-output-bf16-256.avi)；下方 GIF 只截取前 12 帧用于 README 预览。

![官方示例视频输入与 SeedVR2-ncnn BF16 输出对比](assets/showcase-video-comparison-bf16.gif)

本次固定运行处理输入视频的前 36 帧，输出为 `256x256` RGB AVI。输入 SHA256 为 `5b16698d7bbafdc00aa4ee87134ea82dcc8976dde59d78ff5db21054c89ae8ac`，输出 SHA256 为 `e940fa0d8dde1edfd7f723fe889db7b8d87e44611d4a75127d1d2f33a5215b55`。

示例素材来自 [SeedVR2 官方示例空间](https://huggingface.co/spaces/ByteDance-Seed/SeedVR2-3B)，对应视频由其公开的 [SeedVR_VideoDemos 数据集](https://huggingface.co/datasets/Iceclear/SeedVR_VideoDemos) 提供。仓库提交其中一帧的裁剪、降采样 `64x64` 输入，以及与该输入配对的 `256x256` 输出；视频素材的版权和使用条件以原作者及数据集说明为准，本项目不主张拥有素材或与版权方存在关联。

## 能做什么

- 本延伸分支使用 BF16 存储和必要的 FP32 累加；`128x128`、`256x256` 图片及 `256x256`/36 帧视频路径已完成固定输入回归。比赛截止时的提交快照保留在 Git 历史中。
- 处理 PNG/JPEG 图片和 RGB24 AVI 视频；视频输出固定为 RGB24 AVI。当前发布边界为 Linux x86_64。
- 根据输入自动规划目标尺寸，或使用 `--width` 和 `--height` 指定目标尺寸。
- 支持单张图片、最多两张图片的一次调用，以及单个视频文件。
- 不提供文生图、文生视频或图生视频生成工作流；输入内容保持为图片或视频增强任务。
- 标准输出会进行参考引导的色彩重建：保留模型生成的高频细节，并从输入重建低频色彩。

## 快速开始

1. 从 [Releases](https://github.com/HGinkgo/SeedVR2-ncnn/releases/latest) 下载并解压对应平台的运行包。
2. 从 [ModelScope](https://modelscope.cn/models/HGinkgo/SeedVR2-ncnn) 下载模型到运行包目录。安装了 ModelScope CLI 时可执行：

```bash
modelscope download HGinkgo/SeedVR2-ncnn --local-dir models/seedvr2-3b
```

3. 在 Linux 运行第一张图片：

```bash
./seedvr2-ncnn \
  --model-dir models/seedvr2-3b \
  --input input.png \
  --output output.png \
  --gpu-id 0
```

将视频处理到指定的低分辨率目标：

```bash
./seedvr2-ncnn \
  --model-dir models/seedvr2-3b \
  --input input.avi \
  --output output.avi \
  --width 128 --height 128 \
  --gpu-id 0
```

使用 `--help` 查看全部参数。模型包独立分发；运行包本身不依赖 Python、PyTorch 或 CUDA。

默认使用经过发布验证的单步采样。`--steps N` 可启用实验性的多步 Euler 采样（`N` 必须为正整数）；它会按步数增加 DiT 计算时间，并改变生成结果：

```bash
./seedvr2-ncnn \
  --model-dir models/seedvr2-3b \
  --input input.png --output output.png \
  --width 256 --height 256 \
  --steps 4 --gpu-id 0
```

批量处理一个目录中的图片时，可以复用同一模型会话：

```bash
./seedvr2-ncnn \
  --model-dir models/seedvr2-3b \
  --input-dir frames \
  --output-dir enhanced \
  --scale 2 \
  --gpu-id 0
```

视频也支持从指定帧开始、限制处理帧数；这适合快速预览较长素材：

```bash
./seedvr2-ncnn \
  --model-dir models/seedvr2-3b \
  --input clip.avi \
  --output preview.avi \
  --width 256 --height 256 \
  --start-frame 12 --frames 48 \
  --gpu-id 0
```

显存紧张时可以启用实验性的 VAE 分块路径。它只改变 VAE 的空间分块与拼接，不改变模型权重或默认路径；分块尺寸必须是 16 的倍数：

```bash
./seedvr2-ncnn \
  --model-dir models/seedvr2-3b \
  --input large.png --output enhanced.png \
  --width 256 --height 256 \
  --vae-tile-size 128 --gpu-id 0
```

发布包可先运行 `--check-model --model-dir models/seedvr2-3b`，在不初始化 Vulkan 的情况下检查动态模型包的 manifest、记录数和符号链接约束。

输出后处理会保留模型生成的高频细节，并用输入参考重建低频色彩。视频模式会将参考帧暂存到进程临时文件中，因此不会把整段视频长期保留在内存中。

## 动态尺寸

发布版的自动模式保持输入宽高比，将目标对齐到 16 像素，并在必要时居中裁剪。目标面积不超过 `256x256`；低于该上限的输入不会仅为填满面积而被放大。显式 `--width` / `--height` 使用相同上限。

| 发布验证目标 | 图片 | AVI |
| --- | --- | --- |
| `128x128` | 已验证 | 已验证（两帧） |
| `128x256` | 已验证 | - |
| `256x256` | 已验证 | 已验证（连续 36 帧） |

其他符合 16 像素对齐和面积上限的动态尺寸可以请求，但不属于当前发布验证承诺。

在不使用 `--vae-tile-size` 的 `256x256` 目标上，运行包会自动将动态 VAE 参数在内存中固定化，并启用 ncnn 的轻量执行模式；模型包文件和其他尺寸的动态路径不变。这是内部的等价执行路径，不需要额外 CLI 选项。

上表的发布验收采用默认单步采样；多步采样保持为实验选项。GPU 端到端验收已在 `256x256`、36 帧视频上通过，四步路径产生预期的不同采样结果，但尚未纳入默认发布质量承诺。

## 输入与输出

| 工作流 | 输入 | 输出 |
| --- | --- | --- |
| 图片 | PNG、JPEG | PNG、JPEG |
| 基础视频 | RGB24 AVI | RGB24 AVI |
| 带 FFmpeg 的运行包 | 常见压缩视频格式 | RGB24 AVI |

图片调用可将 `--input` 和 `--output` 各重复一次，以在一次调用中处理两张图片。视频调用要求一个输入和一个 `.avi` 输出。

## 模型与硬件

模型目录必须是 ModelScope 发布的完整动态包：根目录含有 `manifest.sha256`（75 条记录），且包内不能有符号链接。将该目录传给 `--model-dir`。

当前模型包仍以 FP32 权重格式导入；Vulkan 推理路径使用 BF16 张量存储，并在需要处使用 FP32 累加。固定输入回归基于 RTX 3090、驱动 `580.95.05`、ncnn `c6b351b56fbe32e0381ae00331e3df649b20d7b7` 和模型 manifest SHA256 `a4285a52f34b05408877ffcb97e98a6fccfebea38b636260d1b11fe93cbecee5`。Linux x86_64 运行包以 Ubuntu 22.04（glibc 2.35）为兼容基线构建。

## 当前边界

- 本发布线面向低分辨率目标；720p 与长视频尚未纳入支持或验证范围。
- macOS 不在产品范围内。
- 模型权重和第三方依赖遵循各自许可证。

## 从源码构建

CPU 构建只需要 CMake 和 C++17 编译器：

```bash
cmake -S . -B build -DSEEDVR2_ENABLE_VULKAN=OFF -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
```

Vulkan 构建需要本地 Vulkan SDK、支持 Vulkan 的 GPU 与驱动：

```bash
cmake -S . -B build-vulkan -DSEEDVR2_ENABLE_VULKAN=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build-vulkan --parallel
```

压缩视频输入还需要在配置时加入 `-DSEEDVR2_ENABLE_FFMPEG=ON` 并提供 FFmpeg 开发库。

## 开发与测试

公开验证面保持精简：`SEEDVR2_BUILD_TESTS=ON` 只构建一条快速运行时合同测试，覆盖 CLI、低分辨率规划和 AVI I/O；`tests/smoke.sh` 用于可执行文件的轻量冒烟。模型导出、GPU 数值比对和性能验证保留为本地验收流程，不作为日常仓库测试目标。

```bash
cmake -S . -B build -DSEEDVR2_ENABLE_VULKAN=OFF -DSEEDVR2_BUILD_TESTS=ON
cmake --build build --parallel 2
ctest --test-dir build --output-on-failure
```

## 许可证

ncnn 使用 BSD-3-Clause 许可证。SeedVR2 模型、权重和其他第三方依赖遵循各自许可证。
