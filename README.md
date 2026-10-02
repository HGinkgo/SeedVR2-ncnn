# SeedVR2-ncnn

面向 Linux x86_64 的 SeedVR2 原生 C++ / ncnn / Vulkan 图像与视频增强 CLI。

[下载运行包](https://github.com/HGinkgo/SeedVR2-ncnn/releases/latest) | [下载模型](https://modelscope.cn/models/HGinkgo/SeedVR2-ncnn) | [English](README.en.md)

## Showcase

官方示例帧经过 `64x64` 降采样后，使用固定 `256x256` 路径恢复：

| 输入 | 输出 |
| --- | --- |
| <img src="assets/showcase-image-input-64.png" alt="64x64 SeedVR2 官方示例低分辨率输入图" width="256"> | <img src="assets/showcase-image-output-256.png" alt="SeedVR2-ncnn 256x256 官方示例输出图" width="256"> |

输入来自 [SeedVR2 官方示例](https://huggingface.co/spaces/ByteDance-Seed/SeedVR2-3B)；输出为本项目保存的配对示例。视频输入和推理输出不作为画质 showcase，避免将尚未完成质量归因的结果误作质量承诺。

## Features

- 原生 C++ 推理，不依赖 Python、PyTorch 或 CUDA runtime。
- Linux NVIDIA Vulkan，BF16 storage，必要位置使用 FP32 accumulation。
- PNG/JPEG 图片和 RGB24 AVI 视频。
- 固定验证尺寸：`128x128`、`128x256`、`256x256`。
- 单步 Euler 为默认路径；多步采样和 VAE tiling 为实验选项。
- 标准输出会进行参考引导的色彩重建：保留模型生成的高频细节，并从输入重建低频色彩。

## Quick Start

下载模型：

```bash
modelscope download HGinkgo/SeedVR2-ncnn --local-dir models/seedvr2-3b
```

图片：

```bash
./seedvr2-ncnn \
  --model-dir models/seedvr2-3b \
  --input input.png \
  --output output.png \
  --gpu-id 0
```

视频：

```bash
./seedvr2-ncnn \
  --model-dir models/seedvr2-3b \
  --input input.avi \
  --output output.avi \
  --width 128 --height 128 \
  --gpu-id 0
```

运行包不需要 Python、PyTorch 或 CUDA。使用 `--help` 查看完整参数。

## Validation

| Target | Image | AVI |
| --- | --- | --- |
| `128x128` | verified | verified, 2 frames |
| `128x256` | verified | - |
| `256x256` | verified | verified, 36 frames |

`256x256 / 36 frames` baseline：video batch `660.7 s`、end-to-end `664.8 s`、peak RSS `1515 MiB`。这是性能基线，不是跨引擎对比，也不代表视频画质验收。

模型 manifest SHA256：`a4285a52f34b05408877ffcb97e98a6fccfebea38b636260d1b11fe93cbecee5`。

## Build

模型目录必须是 ModelScope 发布的完整动态包：根目录含有 `manifest.sha256`（75 条记录），且包内不能有符号链接。将该目录传给 `--model-dir`。

固定输入回归环境：RTX 3090、driver `580.95.05`、ncnn `c6b351b56fbe32e0381ae00331e3df649b20d7b7`、BF16 Vulkan。

## Scope

- 当前产品线只支持 Linux x86_64 NVIDIA Vulkan。
- 输出目标上限为 `256x256`；720p 和长视频未验证。
- 模型权重和第三方依赖遵循各自许可证。

## License

ncnn 使用 BSD-3-Clause。SeedVR2 模型、权重和其他第三方依赖遵循其各自许可证。
