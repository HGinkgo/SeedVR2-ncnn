# PNNX PT2 独立 patch

## 默认：独立受限前端

`pnnx-pt2-independent.patch` 基于官方 Tencent/ncnn commit
`c6b351b56fbe32e0381ae00331e3df649b20d7b7`，新增本项目独立编写的
Python ExportedProgram 前端、C++ canonical PNNX IR reader 和专项 CTest。
它不依赖外部 PT2 review 分支，不通过 TorchScript tracing 中转。

支持二维 FP32 Linear（可选 bias）、RMSNorm（affine/no-affine）、SiLU、mul/add，
读取图、权重、输入输出签名和范围。包含 rank-1..5 FP32 lifted 常量/静态激活、
尾维参数广播、受限固定 reshape/slice/select、静态 attention 的 permute/matmul/SDPA、
单帧 VAE 所需 Conv3D/GroupNorm/pad/clamp/slice/exp，以及明确限定为平方的
`pow.Tensor_Scalar` 和单轴 mean/rsqrt。动态 rank-3..5、其他幂指数与未知结构明确失败；
有限动态形状仅支持直接符号 token 轴；原生后端已测范围以支持矩阵为准。
Python 先输出 canonical IR，打过 patch 的 PNNX 读取 IR 并复用现有优化
和 ncnn lowering；当前不是直接执行 `pnnx model.pt2` 的通用 C++ 前端。

在 SeedVR2 仓库外的官方 checkout 应用；不要修改 `third_party/ncnn`：

```bash
git -C ../external/ncnn-pnnx-pt2-independent apply --check \
  "$PWD/patches/pnnx-pt2-independent.patch"
git -C ../external/ncnn-pnnx-pt2-independent apply \
  "$PWD/patches/pnnx-pt2-independent.patch"
```

也可由 `tools/prepare_pnnx_pt2.sh --python /path/to/python --jobs 4`
创建固定 checkout、应用 patch、构建和测试，重复运行会识别已应用 patch。
CTest `test_pt2_independent` 本地 1/1 通过；四项既有 TorchScript CTest
4/4 通过。真实 checkpoint MLP、RMSNorm/residual、物化 Ada、固定 batch=1
Ada embedding 及 MLP Ada embedding 子图的 20 项 Python、70 项 CPU/Vulkan 数值对照通过。固定 256 单帧 PT2 VAE decoder 候选仍是 opt-in，不替换 TorchScript 生产 VAE。

当前独立 diff SHA256 为
`209e99d32e878e8c957331ba2b68f780ad80c97fda8da32433933dd9b4407645`。
已在官方基线的干净临时 Git index 上执行 `git apply --cached --check`。
2026-10-19 另从同一官方基线创建全新隔离 worktree，使用准备脚本应用此 patch、
完成 611 个构建步骤：`test_pt2_independent` 1/1 通过（12.61 秒），上述四项
TorchScript CTest 4/4 通过（7.10 秒）。patch 涉及的所有文件和生成的 PNNX
二进制 SHA256 都与开发 checkout 一致。
输出归一化所需平方分支另有正常转换和非平方指数拒绝回归；早期四个 commit
的 mailbox 不能替代这份更新后的 diff。

patch 为便于上游评审的独立 diff，使用 BSD-3-Clause；不声称官方接收、
完整 PT2 支持或完整 SeedVR2 转换。
