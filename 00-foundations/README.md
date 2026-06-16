# 00 共性基础

这个目录整理 LLM Infra 和 RL Infra 共用的基础能力。这里的重点不是把基础知识写成百科，而是把后续理解训练、推理、调度、通信、实验平台时会反复用到的系统概念先打牢。

## 当前专题

- [GPU 架构](./gpu-architecture/README.md)：NVIDIA GPU 硬件层级、存储层级、Tensor Core / RT Core / NVLink、从 G80 到 Rubin / Vera Rubin 的架构演进，以及 CUDA 编程模型。
- [异构分布式计算系统](./heterogeneous-distributed-computing/README.md)：以 Ray 为贯穿案例，梳理 CPU、GPU、对象存储、网络和调度器如何支撑 AI 数据处理、训练、推理和 RL rollout。

## 建议补充方向

### GPU / 加速器

- GPU 架构演进与 CUDA 编程模型
- Kernel、warp、occupancy、访存合并与 shared memory
- Tensor Core、低精度计算与矩阵乘法数据流
- NCCL、NVLink、PCIe 与多 GPU 通信
- Profiling、Roofline、stall analysis 与性能调优

### 计算机系统

- Linux 进程、线程、协程与调度
- 网络通信、RPC、RDMA 与数据传输
- 文件系统、对象存储与数据加载
- 容器、镜像、cgroup、namespace 与运行时

### 异构分布式计算系统

- Ray Core：task、actor、object store、CPU / GPU / custom resource
- CPU-GPU 数据流水：数据读取、预处理、batch inference、训练 step 和 rollout
- KubeRay：Kubernetes 上的 Ray 集群、作业、服务和弹性伸缩
- 资源调度、队列、优先级、配额、backpressure 与故障恢复
- 可观测性、日志、指标、Tracing、object spilling 和 GPU 利用率

### 深度学习系统

- 前向 / 反向传播与自动求导
- 显存管理、activation、optimizer state 与 checkpoint
- 算子执行、图优化、kernel fusion 与编译
- 混合精度训练、通信重叠与吞吐优化

### 工程工具链

- Docker / Kubernetes / Slurm / Ray
- PyTorch distributed、NCCL、CUDA toolkit
- Benchmark、profiling、debugging 与实验复现
- Prometheus / Grafana / OpenTelemetry

## 目录约定

每个基础专题单独建目录，目录内保留自己的 README、正文文章、资源文件和可选示例：

```text
00-foundations/
└── topic-name/
    ├── README.md
    ├── xxx.md                  # 正文文章，文件名使用有语义的 snake_case
    ├── assets/
    └── examples/              # 可选：代码、配置或实验样例
```

`README.md` 只做导览：说明这个专题解决什么问题、推荐先读哪篇、有哪些关键资料。真正的正文文章使用有语义的 snake_case 文件名，例如 `cuda_intro.md`、`nvidia_gpu_architecture_evolution.md`。

如果一个主题开始变大，可以再拆 `concepts/`、`papers/`、`projects/`、`notes/`；在内容还少的时候，先保持扁平，避免过早分层。

## 推荐阅读顺序

1. GPU 架构演进与 CUDA 编程模型
2. PyTorch 训练流程、显存机制与算子执行
3. 多 GPU 通信、NCCL 与分布式训练瓶颈
4. 异构分布式计算、Ray、KubeRay 与 CPU-GPU pipeline
5. Linux、容器、调度与资源管理
6. Profiling、Benchmark 与可观测性
