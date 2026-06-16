# GPU Architecture

这个专题用于整理 GPU 架构基础，当前重点放在 NVIDIA GPU 的硬件层级、存储与通信路径、架构代际演进，以及 CUDA 编程模型。

## 当前笔记

- [NVIDIA GPU 架构演进](./nvidia_gpu_architecture_evolution.md)
- [CUDA 编程模型](./cuda_intro.md)

## 阅读重点

这个专题适合按三条线阅读：

1. 先读 NVIDIA GPU 架构演进，理解 GPU 为什么以吞吐量、warp、SM、存储层级和专用加速单元为核心设计。
2. 接着看 G80 / Fermi / Kepler / Maxwell / Pascal / Volta / Turing / Ampere / Ada / Hopper / Blackwell / Rubin 各代主要解决了什么瓶颈。
3. 再读 CUDA 编程模型，理解 Host / Device、kernel、grid / block / thread / warp 如何映射到硬件执行路径。
4. 最后把 Tensor Core、RT Core、NVLink、低精度计算、Roofline、occupancy、stall 等概念连接到 LLM 训练和推理系统。

## 资源组织

- `assets/`：当前笔记中使用的架构图和数据流图。
- 图片文件名应与 Markdown 中的引用保持一致，避免使用截图工具生成的数字文件名。
