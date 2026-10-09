# 性能与验证

当前主线是 Linux x86_64、RTX 3090 GPU 0、Vulkan、固定 `256x256`、BF16 存储与必要位置的 FP32 累加。发布的 C++ runtime 不依赖 Python、PyTorch 或 CUDA；导出和参考比较使用独立的 Python 工具。

## 推理结构

```mermaid
flowchart LR
    Input[图片或视频帧] --> Encode[VAE encode]
    Encode --> DiT[32 个 DiT blocks]
    DiT --> Spool[CPU latent spool]
    Spool --> Decode[VAE decode]
    Decode --> Color[参考引导色彩重建]
    Color --> Output[PNG 或 RGB24 AVI]
```

视频按阶段处理，使用 spool 控制资源驻留。当前保留三项生产优化，均通过本地自定义层或上游 ncnn 公共接口实现；`third_party/ncnn` 未修改。

## 三项生产优化

| 方向 | 实现与基线 | 已验证结果 |
| --- | --- | --- |
| Pointwise Conv3D 的 Vulkan 部署 | 固定 decoder 五个 `1x1x1` 节点从 CPU 转到 `SeedVR2PointwiseConv3D`；真实权重、FP32 累加、BF16 输出 | 五层 BF16 输出与 CPU 参考一致；目前保留的证据不足以发布独立的前后加速百分比 |
| Causal Conv3D 输出通道复用 | `conv3d_20`、`conv3d_29`，本地 OC4 Vulkan kernel 对比 OC8，复用输入读取 | 收尾复测为 `1.439x` / `1.463x`，耗时降低 `30.52%` / `31.66%`；BF16 输出一致 |
| DiT 张量生命周期与原地复用 | 固定-256 block 的 Extractor：light mode 关闭对比开启 | 活跃临时内存降低 `84.59%`，分配器预留空间降低 `62.5%`，复制命令减少 `25.62%`；真实视频 DiT 阶段耗时降低 `3.55%` |

Pointwise 替换仅覆盖 `conv3d_11`、`conv3d_19`、`conv3d_23`、`conv3d_28`、`conv3d_32`。OC8 保留 OC4 作为其他形状的有效路径。DiT 使用现有 ncnn 引用计数和 in-place 执行能力，没有新增内存规划器；每个 block 的 video/text 双输出仍在提交前提取。

### OC8 定向复测

2026-10-09，真实 decoder 权重、确定性合成输入，两个 kernel 各 2 次预热和 5 次计时。Release 构建，计时含 Extractor 执行与提交等待，不含模型加载、初始上传和最终下载。

| 节点 | 输入 W×H×D×C | OC4 中位耗时 | OC8 中位耗时 | BF16 输出 |
| --- | --- | --- | --- | --- |
| `conv3d_20` | `128x128x4x512` | `373.693 ms` | `259.639 ms` | 完全一致 |
| `conv3d_29` | `256x256x4x256` | `378.808 ms` | `258.872 ms` | 完全一致 |

加速比使用 `before / after`；耗时降低使用 `1 - after / before`。历史配对 36 帧测试记录端到端耗时降低 `2.09%`、输出一致、RSS 增加约 `2.7 MiB`；它与下面的 runtime 比较使用不同的冻结基线。

### DiT runtime 验证

隔离 block 0 使用官方权重和确定性合成 hidden states：256 个 video tokens、58 个 text tokens、宽度 2560、pack1 BF16。2 次预热与 10 个新 Extractor；计时包含 CPU fallback、同步和诊断开销，排除加载、初始上传和最终下载。两次运行均为 Release、`NCNN_BENCHMARK=ON`。

| 单 block 指标 | Light mode 关闭 | Light mode 开启 |
| --- | --- | --- |
| 中位耗时 | `212.757 ms` | `205.094 ms` |
| 峰值活跃 blob/workspace 请求 | `120.869 MiB` | `18.630 MiB` |
| blob/workspace buffer 预留空间 | `128 MiB` | `48 MiB` |
| 每次执行复制命令 | `12258` | `9118` |
| Dispatch / submit 次数 | `21736 / 7` | `21736 / 7` |

预留空间通过独立的 2 次预热、5 次执行统计唯一 Vulkan buffer 的实际内存需求，排除权重与驱动内存；该测量的时间样本不混入 10 次计时结果。Dispatch 数量与双输出一致性同时验证，未观察到 light mode 引入额外计算。

实际输入验证覆盖完整 32-block 路径，单帧 PNG 和 36 帧 AVI 前后均字节一致。一次顺序 before/after 配对使用同一模型包、输入 AVI、GPU 和参数，持久化 pipeline cache 关闭：

| 36 帧指标 | 冻结的优化前 runtime | 开启 DiT light mode |
| --- | --- | --- |
| 端到端耗时 | `343.65 s` | `337.07 s`；降低 `1.91%` |
| DiT stack call 累计耗时 | `244.3136 s` | `235.6402 s`；降低 `3.55%` |
| 进程峰值 RSS | `1133.36 MiB` | `1187.83 MiB` |
| 设备总显存峰值，NVML 每 100 ms 采样 | `11521.81 MiB` | `11521.81 MiB` |

**84.59% 描述的是单 block 临时张量，不是整模型显存。** 整条视频峰值由 decoder 阶段决定；本次 RSS 增加 `54.46 MiB`。一个配对结果不能建立置信区间，也不能与历史 `658.9 s` 直接计算累计收益。

## 补充参考比较

- 已有自定义 AWA 对比可执行的 PNNX attention 图：`6.215553 -> 1.851003 ms`，`3.36x`。官方 block 0 权重作用于合成 hidden states 后的 QKV，5 次预热、50 个新 Extractor。基线只改写不受支持的 pack/unpack 边界，两层原生 MatMul 走 CPU。双方 BF16 video/text 输出相对 L2 差异约 `0.415% / 0.424%`，并非字节一致；这是已有实现的补充证据。
- 可选持久化 pipeline cache 的历史图片冷启动比较为 `50.26 -> 39.89 s`，降低 `20.6%`；没有证明视频吞吐提升。
- 历史权威参考使用 [SeedVR 官方 PyTorch 实现](https://github.com/ByteDance-Seed/SeedVR)，固定参考版本为 `e4de8c2`。当时 36 帧记录为 PyTorch `617.45 s`、ncnn `658.9 s`；不能据此声称 ncnn 更快。跨框架输出 MAE `9.533195`、RMSE `17.317680`、PSNR `23.362142 dB`，未建立统一的质量验收阈值。内部优化前后字节一致不等于与 PyTorch 数值完全一致。

## 复现与证据边界

保留的本地测量材料位于忽略目录，包含输入与二进制身份、日志、输出、逐次时间样本和小型定向 probe：

| 比较 | 本地入口 |
| --- | --- |
| DiT 生命周期 | `runtime/runtime-memory-probe/README.md`、`result.json` |
| OC4 / OC8 | `runtime/causal-layerbench/closeout_bench.log`、`result.json` |
| AWA 可执行导出图 | `runtime/awa-validation/README.md`、`result.json` |

这些产物与模型权重不提交到 Git。源码构建和实际推理命令见 README；定向 probe 按各自说明使用同一模型包和 `tools/with-nvidia-vulkan-runtime.sh`。测量前先验证完整模型包并冻结比较对象，记录结果时保留各项原有输入与基线，不相加收益，也不把合成单层输入当作真实视频激活。
