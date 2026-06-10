# 00 Foundations

这个目录整理 LLM Infra 和 RL Infra 共用的基础能力。这里的重点不是把基础知识写成百科，而是把后续理解训练、推理、调度、通信、实验平台时会反复用到的系统概念先打牢。

## 当前专题

- [GPU Architecture](./gpu-architecture/README.md)：CUDA 编程模型、NVIDIA GPU 硬件层级、存储层级、Tensor Core / RT Core / NVLink，以及从 G80 到 Rubin / Vera Rubin 的架构演进。

## 建议补充方向

### GPU / Accelerator

- GPU 架构与 CUDA 编程模型
- Kernel、warp、occupancy、访存合并与 shared memory
- Tensor Core、低精度计算与矩阵乘法数据流
- NCCL、NVLink、PCIe 与多 GPU 通信
- Profiling、Roofline、stall analysis 与性能调优

### Computer Systems

- Linux 进程、线程、协程与调度
- 网络通信、RPC、RDMA 与数据传输
- 文件系统、对象存储与数据加载
- 容器、镜像、cgroup、namespace 与运行时

### Distributed Systems

- 资源调度、队列、优先级与配额
- 一致性、容错、checkpoint 与恢复
- 分布式存储、消息队列与任务编排
- 可观测性、日志、指标、Tracing 与告警

### Deep Learning Systems

- 前向 / 反向传播与自动求导
- 显存管理、activation、optimizer state 与 checkpoint
- 算子执行、图优化、kernel fusion 与编译
- 混合精度训练、通信重叠与吞吐优化

### Engineering Toolchain

- Docker / Kubernetes / Slurm / Ray
- PyTorch distributed、NCCL、CUDA toolkit
- Benchmark、profiling、debugging 与实验复现
- Prometheus / Grafana / OpenTelemetry

## 目录约定

每个基础专题单独建目录，目录内保留自己的 README、主笔记和资源文件：

```text
00-foundations/
└── topic-name/
    ├── README.md
    ├── main_note.md
    └── assets/
```

如果一个主题开始变大，可以再拆 `concepts/`、`papers/`、`projects/`、`notes/`；在内容还少的时候，先保持扁平，避免过早分层。

## 推荐阅读顺序

1. GPU 架构与 CUDA 基础
2. PyTorch 训练流程、显存机制与算子执行
3. 多 GPU 通信、NCCL 与分布式训练瓶颈
4. Linux、容器、调度与资源管理
5. Profiling、Benchmark 与可观测性
