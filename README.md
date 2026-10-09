# SeedVR2-ncnn

面向 Linux x86_64 的 SeedVR2 原生 C++ / ncnn / Vulkan 图像与视频增强 CLI。

[源码编译](#build) | [下载模型](https://modelscope.cn/models/HGinkgo/SeedVR2-ncnn) | [English](README.en.md)

## Showcase

官方示例帧经过 `64x64` 降采样后，使用固定 `256x256` 路径恢复：

| 输入 | 输出 |
| --- | --- |
| <img src="assets/showcase-image-input-64.png" alt="64x64 SeedVR2 官方示例低分辨率输入图" width="256"> | <img src="assets/showcase-image-output-256.png" alt="SeedVR2-ncnn 256x256 官方示例输出图" width="256"> |

输入来自 [SeedVR2 官方示例](https://huggingface.co/spaces/ByteDance-Seed/SeedVR2-3B)；输出为本项目保存的配对示例。

视频推理结果（36 帧、`64x64` 输入恢复到 `256x256`、RGB24 AVI）：

下方对比 GIF 左侧为 `64x64` 输入，右侧为 ncnn Vulkan `256x256` 增强输出：

![官方示例视频输入与 SeedVR2-ncnn 输出对比](assets/showcase-video-comparison-bf16.gif)

<video controls width="256" preload="metadata">
  <source src="assets/showcase-video-output-256.avi" type="video/x-msvideo">
</video>

完整[输入视频](assets/showcase-video-input-64.avi)和 ncnn Vulkan [输出视频](assets/showcase-video-output-256.avi)均已提供。输入由官方 `256x256` 示例视频前 36 帧双线性降采样到 `64x64`。该视频用于展示 ncnn Vulkan 推理产物，不作为独立的画质验收结论。

## Features

- 原生 C++ 推理，不依赖 Python、PyTorch 或 CUDA runtime。
- Linux NVIDIA Vulkan，BF16 storage，必要位置使用 FP32 accumulation。
- PNG/JPEG 图片和 RGB24 AVI 视频。
- 固定验证尺寸：`128x128`、`128x256`、`256x256`。
- 当前优化与性能验收固定为 RTX 3090、单 GPU、`256x256` BF16 Vulkan。
- 单步 Euler 为默认路径；多步采样和 VAE tiling 为实验选项。
- 标准输出会进行参考引导的色彩重建：保留模型生成的高频细节，并从输入重建低频色彩。

## Quick Start

先按 [Build](#build) 从源码编译，再在仓库根目录执行以下命令。

下载模型：

```bash
modelscope download HGinkgo/SeedVR2-ncnn --local-dir models/seedvr2-3b
```

图片：

```bash
build/seedvr2-ncnn \
  --model-dir models/seedvr2-3b \
  --input input.png \
  --output output.png \
  --width 256 --height 256 \
  --gpu-id 0
```

视频：

```bash
build/seedvr2-ncnn \
  --model-dir models/seedvr2-3b \
  --input input.avi \
  --output output.avi \
  --width 256 --height 256 \
  --frames 36 \
  --gpu-id 0
```

该命令处理输入视频的前 36 帧，并将结果写入 RGB24 AVI。省略 `--frames` 时处理整个视频；默认构建支持 AVI，压缩视频输入需要使用 `-DSEEDVR2_ENABLE_FFMPEG=ON` 构建。推理不需要 Python、PyTorch 或 CUDA。使用 `--help` 查看完整参数。

重复运行可通过 `SEEDVR2_PIPELINE_CACHE_PATH` 启用持久化 Vulkan pipeline cache。首次运行会生成缓存，后续匹配的 ncnn、GPU 和驱动环境会自动复用；缓存无效时会重新构建：

```bash
mkdir -p "$HOME/.cache/seedvr2-ncnn"
SEEDVR2_PIPELINE_CACHE_PATH="$HOME/.cache/seedvr2-ncnn/pipeline.cache" \
  build/seedvr2-ncnn \
  --model-dir models/seedvr2-3b --input input.png --output output.png --gpu-id 0
```

## Validation

| Target | Image | AVI |
| --- | --- | --- |
| `128x128` | verified | verified, 2 frames |
| `128x256` | verified | - |
| `256x256` | verified | verified, 36 frames |

当前固定 `256x256` 路径保留三项优化：五个 decoder pointwise Conv3D 的 Vulkan 实现、两个 causal Conv3D 的 OC8 输出通道复用，以及 DiT 中间张量生命周期与原地复用。

最近一次 runtime 优化的 36 帧配对验证为 `343.65 s -> 337.07 s`，DiT 阶段耗时降低 `3.55%`，前后输出完全一致。单 block 活跃临时内存降低 `84.6%`；整条视频峰值显存不变。各项优化使用独立冻结基线，详见[性能与验证记录](docs/performance.md)，不能累加收益或直接与历史端到端时间比较。

## Build

模型目录必须是 ModelScope 发布的完整动态包：根目录含有 `manifest.sha256`（75 条记录），且包内不能有符号链接。将该目录传给 `--model-dir`。

构建需要 C++17 编译器、CMake 和 Vulkan 开发文件；使用仓库固定的 ncnn submodule：

```bash
git submodule update --init --recursive
cmake -S . -B build \
  -DSEEDVR2_ENABLE_VULKAN=ON \
  -DNCNN_BATCH=ON \
  -DNCNN_BF16=ON \
  -DNCNN_BENCHMARK=OFF \
  -DCMAKE_BUILD_TYPE=Release
cmake --build build --target seedvr2-ncnn --parallel
```

源码工作区内使用已准备的 NVIDIA runtime 运行：

```bash
tools/with-nvidia-vulkan-runtime.sh build/seedvr2-ncnn \
  --model-dir models/seedvr2-3b --input input.png --output output.png \
  --width 256 --height 256 --gpu-id 0
```

该 helper 使用忽略目录 `runtime/` 中的 NVIDIA 用户态库；已配置 NVIDIA Vulkan 驱动的环境可直接运行 `build/seedvr2-ncnn`。构建压缩视频输入需额外提供 FFmpeg 开发文件并启用 `-DSEEDVR2_ENABLE_FFMPEG=ON`。ModelScope CLI 仅用于下载模型，不是运行时依赖。

## Scope

- 当前产品线只支持 Linux x86_64 NVIDIA Vulkan。
- 输出目标上限为 `256x256`；720p 和长视频未验证。
- 模型权重和第三方依赖遵循各自许可证。

## License

ncnn 使用 BSD-3-Clause。SeedVR2 模型、权重和其他第三方依赖遵循其各自许可证。
