# leaning_llm_infra

一个围绕 **LLM Infra** 和 **RL Infra** 的个人学习资料收集库，主要用于沉淀：

- 学习路线
- 主题拆分
- 阅读笔记
- 系统设计理解
- 论文 / 项目 / 代码索引

## 目录结构

```text
.
├── 00-foundations/
│   └── README.md
├── 01-llm-infra/
│   └── README.md
└── 02-rl-infra/
    └── README.md
```

## 子领域划分

### 00-foundations

偏基础与共性能力，适合作为 LLM Infra / RL Infra 的前置知识。

- 计算机系统基础：OS、网络、存储、并发、容器化
- 深度学习系统基础：训练流程、显存、算子、通信
- 分布式系统基础：一致性、调度、容错、资源管理
- GPU / CUDA 基础：Kernel、带宽、吞吐、访存
- 工程工具链：Linux、Docker、K8s、Ray、监控与 profiling

### 01-llm-infra

聚焦大模型训练、推理、服务化与生产系统。

- 数据处理与 Tokenization
- 预训练 / SFT 训练流水线
- 并行训练：DP / TP / PP / ZeRO / FSDP
- 推理加速与 Serving：KV Cache、PagedAttention、Batching、量化
- RAG / Agent Runtime / Memory / Tool Use
- 评测、可观测性与成本优化
- 安全、权限、配额与多租户

### 02-rl-infra

聚焦强化学习训练系统、环境系统和大规模实验平台。

- 环境与仿真平台
- 采样、回放与数据管线
- 单机 / 分布式训练架构
- Offline RL / Batch RL 基础设施
- 多智能体 / 自博弈 / 大规模实验编排
- RLHF / RLAIF / Post-training 与 LLM 对接
- 评测、奖励建模、安全与实验管理

## 使用建议

建议在每个主题目录下按下面方式继续补充个人笔记：

```text
主题目录/
├── README.md                # 主题导览
├── concepts/               # 核心概念
├── papers/                 # 论文阅读
├── projects/               # 开源项目拆解
└── notes/                  # 自己的总结与实践记录
```

每篇笔记可以优先回答 4 个问题：

1. 这个模块解决什么问题？
2. 核心设计权衡是什么？
3. 代表性系统 / 论文 / 项目有哪些？
4. 我后续值得继续深挖的问题是什么？

## 当前阅读入口

- [00-foundations](./00-foundations/README.md)
- [01-llm-infra](./01-llm-infra/README.md)
- [02-rl-infra](./02-rl-infra/README.md)