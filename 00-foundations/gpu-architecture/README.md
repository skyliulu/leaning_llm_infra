# GPU Architecture

这个专题用于整理 GPU 架构基础，当前重点放在 NVIDIA GPU 的硬件层级、CUDA 编程模型、存储与通信路径，以及架构代际演进。

## 当前笔记

- [CUDA 编程模型与 GPU 计算系统综述](./cuda_intro.md)
- [NVIDIA GPU Architecture Evolution](./nvidia_gpu_architecture_evolution.md)

## 阅读重点

这篇笔记适合按三条线阅读：

1. 先读 CUDA 编程模型与 GPU 计算系统综述，理解 Host / Device、kernel、grid / block / thread / warp 以及存储层级。
2. 再理解 GPU 为什么以吞吐量、warp、SM、存储层级为核心设计。
3. 接着看 G80 / Fermi / Kepler / Maxwell / Pascal / Volta / Turing / Ampere / Ada / Hopper / Blackwell / Rubin 各代主要解决了什么瓶颈。
4. 最后把 Tensor Core、RT Core、NVLink、低精度计算、Roofline、occupancy、stall 等概念连接到 LLM 训练和推理系统。

## 资源组织

- `assets/`：当前笔记中使用的架构图和数据流图。
- `examples/`：CUDA 综述配套的最小可运行示例，用于理解 kernel launch、shared memory tiling、stream pipeline 等概念。
- 图片文件名应与 Markdown 中的引用保持一致，避免使用截图工具生成的数字文件名。
