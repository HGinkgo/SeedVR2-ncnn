# SeedVR2-ncnn

面向 Linux x86_64 的 SeedVR2 原生 C++ / ncnn / Vulkan 图像与视频增强 CLI。

[源码编译](#build) | [下载模型](https://modelscope.cn/models/HGinkgo/SeedVR2-ncnn) | [English](README.en.md)

生产默认仍是 TorchScript/ncnn BF16；Vulkan BF16 生产精度验收尚未通过。ExportedProgram/PT2 → 独立 PNNX → ncnn 的固定 256 验证路径已打通，但仍为 opt-in，尚未替换生产包。

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

仓库提供独立编写的受限 ExportedProgram 前端：Python 读取 `.pt2` 图、权重和签名，输出 canonical PNNX IR；独立上游 checkout 中的 C++ reader 复用 PNNX 优化和 ncnn lowering。入口是 [`tools/convert_pt2.py`](tools/convert_pt2.py)，保留 TorchScript baseline，不修改 `third_party/ncnn`，不要求发布 runtime 携带 Python、PyTorch 或 CUDA。

已验证静态/有界动态 token 子图、checkpoint MLP/Ada/attention，以及完整固定 block-0、block-1、block-9 的 PT2/TorchScript CPU/Vulkan 对照；固定单帧 PT2 VAE encoder/decoder 也通过 CPU/Vulkan。完整 35 图 stack 的早期结果实际来自 TorchScript，先前的 PT2 来源标记有误；旧 `model-package-pt2-full-stack-256` 只包含 PT2 VAE + TorchScript full-matrix DiT。这些实测结果不能作为 PT2 DiT 全栈证据。

2026-10-19 已用真正 `.pt2` 重新验收固定 256 正向 58-token 和负向 64-token stack：两套各 35 张图均通过 CPU/Vulkan，同输入 PT2/TorchScript 最大输出差为 0；两套串联 Vulkan 的最终最大误差均为 `1.26362e-5`。

真正 PT2 VAE + DiT 候选包已完成 RTX 3090 上的固定 256 图像 CLI：CFG=1/1.5、一步/两步四组均通过，原生产 TorchScript 一步/两步回归也通过。CFG 去噪结束后释放两套 DiT 权重，再加载 VAE 解码，解决了实测显存申请失败。PNG 差异与图级 FP32 验收分开记录，不据此声称画质或性能提升。

BF16 验证发现并修复了 AWA int32 映射被低精度权重上传转换的问题，同时为 RoPE 补齐 dispatch 加上 token 上界。修复后，在相同 PT2 VAE/conditioning 下，TS/PT2 DiT 的 CFG=1 和 1.5 两步完整图像对照，三个张量交接点指纹和最终 PNG SHA256 均一致。PT2 独立进程重复也满足交接点与 PNG 完全一致；正负向栈各 68 阶段的 BF16 重复及两帧逐帧 CFG 视频均通过。结论限定于记录中的固定输入、模型、binary 和单卡环境；视频尚未验证 temporal VAE 或跨帧 attention。

可选 `--seed` 的标准正态初始噪声已完成五次图像、两次视频 Vulkan 验证：seed=666 的匹配 TS/PT2 和独立 PT2 重复，四个交接指纹与 PNG 完全一致；视频第二帧单独截取后也精确复现。不传 seed 的旧输出保持一致。原始完整预处理、默认 BF16 数值与最终画质仍待验收。

[`tools/export_dit_stack.py`](tools/export_dit_stack.py) 现提供显式 `--format pt2`，为每张图记录 `.pt2` 哈希和独立转换记录；默认仍为 `torchscript`。CFG runtime 使用独立正向 58-token 和负向 64-token stack，通过 `--cfg-scale` 启用，默认 1；固定 256 的模型包准备器为 [`tools/prepare_pt2_cfg_package.py`](tools/prepare_pt2_cfg_package.py)。生产包仍为 TorchScript/BF16；动态/多帧 VAE、文本 encoder、CFG rescale 和生产默认切换未验收。

[`tools/prepare_pnnx_pt2.sh`](tools/prepare_pnnx_pt2.sh) 在仓库外固定官方 ncnn commit 并应用[独立 patch](patches/README.md)。[CPU CI](.github/workflows/pt2.yml) 已配置，尚未远程运行；未声称官方接收。

无需模型或 GPU 的验收工具回归可执行 `python -m unittest tools.test_pt2_runtime_contracts tools.test_dit_output_contract -v`，覆盖输出 Ada 布局、非法签名、过期导出/模型包、交接指纹、seed/帧号、CFG 阶段顺序和 AVI 帧边界。

## Scope

- 当前产品线只支持 Linux x86_64 NVIDIA Vulkan。
- 输出目标上限为 `256x256`；720p 和长视频未验证。
- 模型权重和第三方依赖遵循各自许可证。

## License

ncnn 使用 BSD-3-Clause。SeedVR2 模型、权重和其他第三方依赖遵循其各自许可证。
新增 PT2 工具和 patch 使用 [BSD-3-Clause](LICENSES/BSD-3-Clause.txt)。既有项目根 LICENSE 尚未声明；此声明不追溯授权全部原有源码。
