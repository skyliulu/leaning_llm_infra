# 00 Foundations

这个目录用于整理 **LLM Infra** 和 **RL Infra** 的共性基础能力。

## 建议学习模块

### 1. 计算机系统基础

- Linux 基础
- 进程 / 线程 / 协程
- 网络通信与 RPC
- 文件系统与对象存储
- 容器与虚拟化

### 2. 深度学习系统基础

- 前向 / 反向传播
- 自动求导
- 显存管理
- 算子执行与图优化
- 混合精度训练

### 3. 分布式系统基础

- 调度与资源分配
- 一致性与容错
- 消息队列与任务编排
- 分布式存储
- 可观测性

### 4. GPU / CUDA 基础

- GPU 架构
- CUDA 编程模型
- Kernel 与访存
- 通信库：NCCL
- Profiling 与性能分析

### 5. 工程工具链

- Docker / Kubernetes
- Ray / Slurm
- Prometheus / Grafana
- PyTorch 分布式工具链
- Benchmark / Profiling 工具

## 建议笔记模板

每个主题可按以下模板沉淀：

- 背景与问题定义
- 关键概念
- 核心组件 / 数据流 / 控制流
- 常见瓶颈
- 代表项目 / 论文 / 博客
- 自己的理解与疑问

## 推荐优先级

1. Linux + Docker + GPU 基础
2. PyTorch 训练流程与显存机制
3. 分布式训练通信基础
4. K8s / Ray / 调度系统
5. Profiling / Benchmark / 可观测性
