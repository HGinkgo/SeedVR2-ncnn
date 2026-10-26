# SeedVR2-ncnn

面向 Linux x86_64 的 SeedVR2 原生 C++ / ncnn / Vulkan 图像与视频增强 CLI。

[源码编译](#build) | [下载模型](https://modelscope.cn/models/HGinkgo/SeedVR2-ncnn) | [English](README.en.md)

生产默认采用已通过固定 256 六组配对质量验证的混合精度：DiT 输入、timestep embedding 和前 9 个 block 使用 FP32，后续 block 使用 BF16。RTX 3090 峰值显存约 22.9 GiB。`--precision fp32` 可用于全 FP32 对照。ExportedProgram/PT2 → PNNX → ncnn 的固定 256 验证路径仍为 opt-in。

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
- 可选 `--seed 666` 使用可复现的标准正态初始噪声；省略时保留旧 fixture。
- 标准输出会进行参考引导的色彩重建：保留模型生成的高频细节，并从输入重建低频色彩。

固定 `256x256` 混合精度现在为默认配置，正常运行无需设置环境变量：

```bash
tools/with-nvidia-vulkan-runtime.sh build/seedvr2-ncnn \
  --model-dir models/seedvr2-3b --input input.png --output output.png \
  --width 256 --height 256 --gpu-id 0
```

默认让 DIT 输入、时间嵌入和前 9 个 DIT block 使用 FP32，后续 block 使用 BF16。六组固定质量对照均通过；单步最高占用约 22.9 GiB 显存。

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

当前固定 `256x256` 路径保留三项优化：五个 decoder pointwise Conv3D 的 Vulkan 实现、两个 causal Conv3D 的 OC8 输出通道复用，以及 DiT 中间张量生命周期与原地复用。现有 TorchScript VAE encode/decode 边界也已用固定 256x256 golden 在 CPU/Vulkan 完成 smoke；这项验证仍不等于 PT2 VAE 转换或完整 VAE→DiT→VAE 串联。

最近一次 runtime 优化的 36 帧配对验证为 `343.65 s -> 337.07 s`，DiT 阶段耗时降低 `3.55%`，前后输出完全一致。单 block 活跃临时内存降低 `84.6%`；整条视频峰值显存不变。各项优化使用独立冻结基线，不能累加收益或直接与历史端到端时间比较。

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

## PT2 / PNNX 实验路径

2026-10-19 直接导入原始 NaDiT 逐层审计，修正了最终 Ada 的 embedding 切片，
以及 block 31 错误执行文本 Ada/MLP 的问题。
新导出使用逐特征 `[2560,6]` 的首个 triplet，末层保留文本 attention 并跳过
文本 Ada/MLP；导出恢复和打包均拒绝缺少新契约的旧产物。
官方固定 commit 的原始输出 Ada 存在维度不匹配；本项目采用显式兼容契约，
下述历史一致性结果不作为官方采样或画质证据。

仓库内的 [`pnnx/`](pnnx/) 模块直接提供原生 Torch Dynamo ExportedProgram reader，支持
`pnnx model.pt2`，读取 `.pt2` 图、权重和输入输出签名，并继续复用 PNNX 优化和 ncnn
lowering。使用 [`tools/prepare_pnnx_pt2.sh`](tools/prepare_pnnx_pt2.sh) 可构建模块并运行
2 个基础 PT2 测试；更大的算子矩阵和 SeedVR2 全模型验证属于后续集成工作。

已验证静态/有界动态 token 子图、checkpoint MLP/Ada/attention，以及完整固定 block-0、block-1、block-9 的 PT2/TorchScript CPU/Vulkan 对照；固定单帧 PT2 VAE encoder/decoder 也通过 CPU/Vulkan。完整 35 图 stack 的早期结果实际来自 TorchScript，先前的 PT2 来源标记有误；旧 `model-package-pt2-full-stack-256` 只包含 PT2 VAE + TorchScript full-matrix DiT。这些实测结果不能作为 PT2 DiT 全栈证据。

2026-10-19 已用真正 `.pt2` 重新验收固定 256 正向 58-token 和负向 64-token stack：两套各 35 张图均通过 CPU/Vulkan，同输入 PT2/TorchScript 最大输出差为 0；两套串联 Vulkan 的最终最大误差均为 `1.26362e-5`。

真正 PT2 VAE + DiT 候选包已完成 RTX 3090 上的固定 256 图像 CLI：CFG=1/1.5、一步/两步四组均通过，原生产 TorchScript 一步/两步回归也通过。CFG 去噪结束后释放两套 DiT 权重，再加载 VAE 解码，解决了实测显存申请失败。PNG 差异与图级 FP32 验收分开记录，不据此声称画质或性能提升。

BF16 验证发现并修复了 AWA int32 映射被低精度权重上传转换的问题，同时为 RoPE 补齐 dispatch 加上 token 上界。修复后，在相同 PT2 VAE/conditioning 下，TS/PT2 DiT 的 CFG=1 和 1.5 两步完整图像对照，三个张量交接点指纹和最终 PNG SHA256 均一致。PT2 独立进程重复也满足交接点与 PNG 完全一致；正负向栈各 68 阶段的 BF16 重复及两帧逐帧 CFG 视频均通过。视频 CLI 当前逐帧运行 source_t=1 的模型；temporal VAE 和跨帧 attention 尚未实现验证，不能视为时序视频移植完成。

原始纯 BF16 与 Vulkan FP32 对照的六组固定输入验收中有五组未达到质量门槛。现默认采用已通过六组配对门槛的配置：DiT 输入、timestep embedding 和前 9 个 block 使用 FP32，其余使用 BF16；RTX 3090 单步峰值显存约 22.9 GiB。全 FP32 对照仍可用 `--precision fp32` 显式运行。

可选 `--seed` 的标准正态初始噪声已完成五次图像、两次视频 Vulkan 验证：seed=666 的匹配 TS/PT2 和独立 PT2 重复，四个交接指纹与 PNG 完全一致；视频第二帧单独截取后也精确复现。不传 seed 的旧输出保持一致。完整原始预处理和时序模型路径仍未验收；这里的精度结论限于上述固定输入、固定 256 对照。

[`tools/export_dit_stack.py`](tools/export_dit_stack.py) 现提供显式 `--format pt2`，为每张图记录 `.pt2` 哈希和原生转换记录；默认仍为 `torchscript`。CFG runtime 使用独立正向 58-token 和负向 64-token stack，通过 `--cfg-scale` 启用，默认 1；固定 256 的模型包准备器为 [`tools/prepare_pt2_cfg_package.py`](tools/prepare_pt2_cfg_package.py)。当前生产模型包仍是 TorchScript/ncnn，默认推理使用上述混合精度；PT2 端到端替换生产模型包仍未完成。动态/多帧 VAE、文本 encoder 和 CFG rescale 未验收。

[`pnnx/README.md`](pnnx/README.md) 记录 PNNX 模块和基础验证方式。

基础回归可执行 `ctest --test-dir build --output-on-failure`；PNNX `.pt2` 基础转换回归可执行 `tools/prepare_pnnx_pt2.sh --python <含 PyTorch 的 Python>`。

## Scope

- 当前产品线只支持 Linux x86_64 NVIDIA Vulkan。
- 输出目标上限为 `256x256`；720p 和长视频未验证。
- 模型权重和第三方依赖遵循各自许可证。

## License

ncnn 使用 BSD-3-Clause。SeedVR2 模型、权重和其他第三方依赖遵循其各自许可证。
新增 PT2 工具和 patch 使用 [BSD-3-Clause](LICENSES/BSD-3-Clause.txt)。既有项目根 LICENSE 尚未声明；此声明不追溯授权全部原有源码。
