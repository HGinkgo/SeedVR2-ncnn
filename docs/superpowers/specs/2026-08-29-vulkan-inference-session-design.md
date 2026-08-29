# Vulkan Inference Session Design

## Goal

让 AVI 多帧推理在同一个 Vulkan 会话中复用运行时、内存分配器和全部模型 graph，同时保持现有单图片 CLI 行为不变。

## Scope

- 复用一次创建的 Vulkan instance、选定的 `VulkanDevice`、encode/decode allocator。
- 复用两套 VAE graph，以及 DiT 的 `dit_input`、`dit_embedding`、packing、32 个 block 和 `dit_output` graph。
- 每帧只保留输入预处理、上传、latent/patch 中间 tensor、采样和输出后处理的瞬时状态。
- 不改变模型文件格式、分辨率选择规则、AVI 编解码边界或现有 `run_image_inference` 公共接口。
- 单图片路径继续创建一次会话并执行一帧，以保持错误和资源释放语义。

## Architecture

`VulkanInferenceSession` 负责 Vulkan 和 allocator 的 RAII 生命周期，并保存 VAE 与 DiT runner。`DitStackRunner` 保存已加载的 ncnn `Net` 对象；其 `run` 方法接收当前帧的 GPU patch、文本和 timestep，创建 extractor 与瞬时 tensor，但不重复加载 graph。VAE encode/decode 使用会话持有的 allocator 和 Net，帧结束时显式释放所有 GPU 中间结果。

视频 CLI 在打开 AVI 和解析 `ResolutionPlan` 后创建一个会话，循环调用 `run_frame`，最后关闭 writer 并销毁会话。任何初始化错误在首帧前返回；任何帧错误都带有 frame index 和 stage 信息。会话不缓存跨帧 latent 或输出，避免帧间数据污染。

## Error Handling

- Vulkan 初始化、GPU 选择、allocator 获取和 graph 加载失败，沿用现有 diagnostics 格式并标记对应 stage。
- 会话 plan 与后续帧尺寸不匹配时立即失败，不尝试隐式重建 graph。
- 资源释放采用逆序 RAII；失败路径也必须 reclaim allocator、clear Net 和销毁 Vulkan instance。

## Verification

1. 先增加失败测试，锁定两帧调用只发生一次会话初始化和 graph 加载，并验证第二帧仍能产生独立输出。
2. 运行对应 C++ 单元/集成测试，确认失败原因来自缺少复用实现而非测试错误。
3. 实现最小会话和 DiT runner，重复运行测试直至通过。
4. 使用现有 2 帧 16x16 AVI -> 128x128 smoke，验证 stage 顺序、输出帧数、尺寸和 fps；日志中的初始化与 graph-load 计数必须为一次。
5. 保留图片单帧 smoke，确认公共 CLI 路径未回归。长分辨率测试不在本阶段执行。

## Acceptance Criteria

- 2 帧 AVI 在 Vulkan 下成功输出 2 帧，格式仍为 RGB24 AVI，尺寸/fps 不变。
- 同一进程内 `initialize-vulkan`、VAE graph load、DiT graph load 各只出现一次。
- 每帧仍完整经过 `vae-encode -> dit-stack -> v_lerp -> vae-decode`，且 GPU/CPU 资源在会话结束后释放。
- 单图片 CLI、CPU 构建和现有快速 CTest 全部通过。
