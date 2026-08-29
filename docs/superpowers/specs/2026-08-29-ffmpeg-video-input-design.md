# FFmpeg 视频输入适配设计

## 目标

为 `SeedVR2-ncnn` 增加可选的压缩视频输入能力，使常见 MP4/H.264 文件可以进入现有逐帧 Vulkan 推理链路，同时保持无第三方媒体依赖时的 RGB24 AVI 路径。

本阶段不把 H.264/MP4 编码列为首发硬性能力。输出继续使用现有 AVI writer，避免编码器许可、动态库打包和跨平台发行问题阻塞模型产品化。

## 方案

新增 `VideoReader` 门面，隐藏具体 reader 的实现：

- `.avi` 输入优先使用现有 `AviVideoReader`，保证无依赖回退和现有行为不变。
- 原生 AVI 打开失败或其他视频扩展名，在启用 FFmpeg 时使用 `FfmpegVideoReader`。
- FFmpeg reader 通过 libavformat 选择视频流，通过 libavcodec 解码，通过 libswscale 转换为 RGB24；公共接口只暴露 `VideoInfo` 和 `RgbImage`，不向应用层泄露 FFmpeg 类型。
- `main.cpp` 只依赖 `VideoReader`，视频输出仍要求 `.avi`，不支持的输出格式返回明确错误。

FFmpeg 通过 CMake 选项 `SEEDVR2_ENABLE_FFMPEG` 控制，默认关闭。启用时查找 `libavformat`、`libavcodec`、`libavutil` 和 `libswscale` 的开发文件，并动态链接目标系统提供的共享库。未启用或依赖缺失时，构建基础路径仍然可用，压缩输入返回包含构建选项的可操作错误。

## 数据流和生命周期

`VideoReader::open` 完成容器探测、视频流选择、解码器初始化和 RGB 转换上下文创建。`read_next` 逐包送入解码器，处理一个或多个解码帧后返回一帧 `RgbImage`；到达 EOF 时先执行解码器 flush，再报告正常结束。reader 析构释放 FFmpeg 上下文和转换缓冲区。

`VideoInfo` 的宽高来自解码帧/流参数，帧率优先使用平均帧率并保留分子分母；容器无法提供可靠帧数时允许为零，应用仍以实际读到的帧数作为处理结果。

视频解码失败只影响当前命令并返回 `stage=video-decode` 错误；不改变模型推理的 Vulkan/CPU 选择，也不引入静默的模型计算 CPU fallback。

## 构建和发行边界

- 默认构建：不需要 FFmpeg，现有图片和 RGB24 AVI 测试必须保持通过。
- FFmpeg 构建：仅依赖系统动态库；发布包需要随附或明确声明这些库及其许可证。
- 本阶段不启用 GPL 编码器，不引入 `--enable-gpl`、OpenH264、NVENC/NVDEC 或 FFmpeg 源码 vendoring。
- 不处理音频、字幕、多视频流同步、硬件解码、分段/多帧批处理和 MP4 输出。

## 验证

1. 默认 CPU/Vulkan 构建和现有 `test_video_io` 不受影响。
2. 启用 FFmpeg 的构建可以加载一个由系统 `ffmpeg` 生成的两帧小型 MP4/H.264 输入，正确报告宽高和帧率，并逐帧得到 RGB24 数据。
3. 产品 CLI 使用该 MP4 输入完成短时视频 smoke，输出现有 RGB24 AVI，成功标记和帧数/尺寸/fps 检查与 AVI 测试一致。
4. FFmpeg 未启用时，对非 AVI 视频输入给出明确的重新配置提示，而不是误报为损坏 AVI。

## 完成标准

本阶段完成时，FFmpeg 适配可以通过独立 CMake 开关构建；压缩视频输入能够稳定进入现有 Vulkan 视频推理路径；关闭该开关时原有零依赖 AVI 行为和测试完全不变；没有引入 H.264 编码承诺或未验证的媒体依赖。
