# SeedVR2-ncnn Roadmap

## 项目定位

`SeedVR2-ncnn` 的目标是构建一个类似 `zimage-ncnn-vulkan` 的独立、便携式 SeedVR2 图像/视频超分命令行应用。底层使用 ncnn 和 Vulkan，运行时不依赖 Python、PyTorch 或 CUDA。

SeedVR2 是单步扩散式视频复原模型。高分辨率输入依赖 Adaptive Window Attention（AWA），因此 AWA 的导出、CPU 参考实现和 Vulkan 实现是本项目的核心技术主线。

## 项目约定

- 项目目录：`SeedVR2-ncnn`
- 命令行程序：`seedvr2-ncnn`
- ncnn 位置：`third_party/ncnn`
- ncnn 依赖方式：SSH Git 子模块，固定到经过验证的 commit
- 主导出链路：PyTorch -> PNNX -> ncnn
- 自定义算子顺序：先 CPU 参考实现，再 Vulkan 实现
- 首个版本不提前加入模型、量化、多 GPU 或移动端的空接口

## 阶段路线

### 0. 基线冻结

固定 SeedVR2 上游代码、权重版本、输入输出预处理和许可证，准备一组小型黄金输入、输出和必要的中间张量。

**完成条件：** PyTorch 参考推理可以稳定复现，后续阶段均使用同一基线进行对齐。

### 1. 应用工程

建立接近 `zimage-ncnn-vulkan` 的应用结构：`src/` 应用层、`third_party/ncnn` 子模块、CMake 构建、中文 `README.md` 和同步英文 `README.en.md`。第一版只包含真实可运行的 CLI、版本信息和构建流程，不加入模型或算子占位目录。

**完成条件：** CPU 配置可以构建并运行 CLI；Vulkan 配置可以单独启用；全新递归克隆可以复现构建。

### 2. 导出路径

建立可重复的 PyTorch -> PNNX -> ncnn 流程，输出模型、转换日志和图结构检查结果。优先将可由现有 ncnn 层表达的子图转换成功，单独记录无法转换的算子。

**完成条件：** 不含自定义 AWA 的模型骨干可以转换、加载，并在固定张量测试上满足数值误差预算。

### 3. AWA 语义验证

从上游实现中确认 AWA 的张量布局、动态窗口规则、padding/mask、相对位置编码和 softmax 精度策略，编写独立 PyTorch 对照测试，覆盖非整除和目标高分辨率尺寸。

**完成条件：** 独立 AWA 参考实现与上游实现一致，并在多组输入形状上通过测试。

### 4. ncnn 自定义层

为 AWA 定义稳定的 PNNX 自定义算子导出规则和明确的参数格式，实现 ncnn CPU 参考层。导出图只使用一种稳定层名，避免模型文件依赖临时实现细节。

**完成条件：** 完整模型可以在 ncnn CPU 后端运行；中间张量和最终输出与 PyTorch 基线在约定容差内对齐。

### 5. Vulkan 加速

实现 AWA Vulkan layer、shader、pipeline 创建、动态 shape 和 ncnn Vulkan packing 支持。先完成正确性验证，再进行算子融合、访存和工作组配置优化。

**完成条件：** Vulkan 输出与 CPU 参考层对齐；目标 GPU 上无验证层错误、无资源泄漏，且端到端性能优于 CPU。

### 6. 推理产品化

加入图片/视频解码与编码、逐帧或分段调度、显存预算、OOM 诊断、tile 策略和 CLI 参数。模型目录旁置，程序保持单一可执行文件和清晰的发布包布局。

**完成条件：** 一个命令可以完成图片和视频超分；模型缺失、GPU 不可用、输入异常和显存不足时都提供可操作的错误信息。

### 7. 发布验证

补充自动构建、回归集、性能基线、第三方许可证清单、双语使用文档和预构建发布包，并在干净环境完成烟测。

## 1.0 收尾标准

只有同时满足以下条件，项目才算完成 `SeedVR2-ncnn 1.0`：

1. 在支持 Vulkan 的 Linux 和 Windows x86_64 主机上，从干净的递归克隆可以构建发布版。
2. 发布包只需要可执行文件、模型文件和必要动态库；运行时不依赖 Python、PyTorch 或 CUDA。
3. `seedvr2-ncnn` 可以处理至少一张图片和一个标准视频样例，输出到指定路径，并支持选择 GPU、模型目录和输出分辨率或倍率。
4. 全模型 ncnn Vulkan 输出通过预定义回归集，与冻结的 PyTorch 基线满足事先写入测试的数值和视觉容差。
5. AWA 对常规整除尺寸、非整除尺寸和目标高分辨率尺寸均验证正确，CPU 与 Vulkan 路径都通过。
6. Vulkan 验证层无错误；重复处理回归视频不崩溃、不出现显存持续增长或输出尺寸错误。
7. 中英文 README 同步，完整说明构建、模型目录、命令示例、硬件要求、已知限制和模型/代码许可证。
8. CI 至少覆盖 Linux CPU 构建、Linux Vulkan 构建和 AWA/端到端回归测试；发布包通过一次干净环境烟测。

## 参考资料

- [SeedVR2 项目主页](https://iceclear.github.io/projects/seedvr2/)
- [SeedVR2 论文](https://arxiv.org/abs/2506.05301)
- [zimage-ncnn-vulkan](https://github.com/nihui/zimage-ncnn-vulkan)
- [Tencent ncnn](https://github.com/Tencent/ncnn)
- [ncnn 自定义层指南](https://github.com/Tencent/ncnn/wiki/how-to-implement-custom-layer-step-by-step)
- [PNNX 文档](https://github.com/Tencent/ncnn/blob/master/tools/pnnx/README.md)
