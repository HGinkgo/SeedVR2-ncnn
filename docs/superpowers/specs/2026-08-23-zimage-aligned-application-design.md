# SeedVR2-ncnn Application Alignment Design

## Goal

将 `SeedVR2-ncnn` 做成接近 `zimage-ncnn-vulkan` 的可发布应用：一个面向最终用户的 Linux/Windows 命令行程序，运行时只依赖 Vulkan、ncnn、模型文件和发布包内的必要库，不依赖 Python、PyTorch 或 CUDA。

## Reference Constraints

参考项目的工程形态作为首发约束：

- `src/CMakeLists.txt` 是应用构建入口。
- 一个可执行文件负责命令行解析和推理 pipeline 编排。
- 每个模型阶段由独立的 ncnn 封装负责加载 `.param`/`.bin` 和执行。
- 模型目录与可执行文件并列放置，模型路径通过 CLI 指定或使用默认目录。
- 发布重点是 Windows/Linux 便携压缩包，而不是 Python 包或服务端接口。
- CI 首发只覆盖 Linux 和 Windows；macOS 不列入目标平台。

ncnn 子模块继续使用本项目已经验证的 SSH 地址和固定 commit，不照搬参考项目的 HTTPS 子模块地址。

## Application Shape

首发代码按以下边界逐步形成，不提前创建无实现的占位模块：

```text
SeedVR2-ncnn/
├── src/
│   ├── CMakeLists.txt       # canonical application build entry
│   ├── main.cpp             # short CLI and process lifecycle
│   ├── seedvr2_pipeline.*   # input, model stages, output orchestration
│   ├── seedvr2_vae.*        # VAE ncnn model wrapper
│   ├── seedvr2_dit.*        # DiT ncnn model wrapper
│   └── seedvr2_awa.*        # AWA layer contract and implementation
├── src/ncnn/                # SSH ncnn submodule
├── tests/
└── tools/                   # export and reference utilities
```

`main.cpp` 只处理参数、错误码和 pipeline 生命周期；模型加载、预处理、分块策略和输出编码不放入 CLI 文件。Z-Image 的 prompt、ControlNet、LanPaint 等与 SeedVR2 无关的功能不复制。

## Runtime Interface

首个可用推理接口围绕超分任务设计：

- `-i` 输入图片或视频。
- `-o` 输出路径。
- `-m` 模型目录。
- `-g` Vulkan GPU 索引。
- `-s` 输出尺寸或目标分辨率。

只有已经有真实后端实现的选项才进入 CLI；阶段性调试参数保留在测试工具中，不污染最终用户接口。

## Implementation Order

1. 将 CMake canonical entry 和 ncnn 子模块路径对齐到 `src/`，保持干净递归克隆可构建。
2. 完成真实的输入/输出媒体路径和 VAE ncnn wrapper，先让单阶段 pipeline 可运行。
3. 冻结 PyTorch -> PNNX -> ncnn 的 DiT 子图，定义 AWA 自定义层参数和张量布局。
4. 实现 AWA CPU 层并用 golden tensors 对齐，再实现 Vulkan layer 和 shader。
5. 组装完整 SeedVR2 pipeline，加入显存预算、错误信息和 Linux/Windows release packaging。

## Out Of Scope For First Release

- macOS、移动端和服务端 API。
- 7B 模型、量化、多 GPU 和 ControlNet 类扩展。
- 没有后端支持的 CLI 选项或空的扩展目录。

## Completion Standard

首发版本只有在以下条件同时满足时才算完成：

1. Linux 和 Windows 从干净递归克隆可以构建 Release 包。
2. 便携包运行时不需要 Python、PyTorch 或 CUDA。
3. 一个命令可以处理至少一张图片和一个标准视频样例，并支持输入、输出、模型目录、GPU 和尺寸参数。
4. VAE、DiT 和 AWA CPU/Vulkan 路径都通过固定 PyTorch golden 回归，满足预先记录的数值容差。
5. AWA 对整除、非整除、shifted 和 clipped window 输入均正确，Vulkan 验证层无错误。
6. 模型缺失、输入异常、GPU 不可用和显存不足都返回可操作的错误信息。
7. Linux/Windows 发布包和双语 README 可在干净环境完成烟测。
