# Learning LLM Infra

这个仓库用于沉淀我对 **LLM Infra**、**RL Infra** 以及相关基础系统的学习笔记。它不是资料堆放区，而是一个逐步长出来的学习地图：每个主题尽量回答“它解决什么问题、核心机制是什么、系统瓶颈在哪里、和其他模块如何连接”。

## 当前入口

```text
.
├── 00-foundations/          # 共性基础：GPU、系统、分布式、工程工具链
│   ├── README.md
│   └── gpu-architecture/
│       ├── README.md
│       ├── nvidia_gpu_architecture_evolution.md
│       └── assets/
├── 01-llm-infra/            # LLM 训练、推理、服务化与平台工程
│   └── README.md
└── 02-rl-infra/             # RL 训练系统、环境、采样与实验平台
    └── README.md
```

## 内容边界

### 00-foundations

放 LLM Infra / RL Infra 都会依赖的底层知识。这里的笔记不追求覆盖所有基础课程，而是优先服务后续工程理解。

- GPU / CUDA / 加速器架构
- Linux、网络、存储、并发与容器
- 深度学习训练基础、显存、算子与通信
- 分布式系统、调度、容错与资源管理
- Profiling、Benchmark、可观测性与工程工具链

### 01-llm-infra

放大模型系统链路相关内容，重点是训练、推理、服务化和生产平台。

- 数据处理、Tokenization、数据配比与版本管理
- Pretraining / SFT / Post-training 训练流水线
- DP / TP / PP / ZeRO / FSDP 等并行训练机制
- KV Cache、PagedAttention、Batching、量化与推理服务
- RAG、Agent Runtime、Memory、Tool Use
- 评测、可观测性、成本、安全、多租户与发布治理

### 02-rl-infra

放强化学习系统相关内容，重点是环境、采样、训练、评测和大规模实验。

- 环境抽象、仿真平台与任务封装
- Rollout、Replay Buffer、采样与数据管线
- Actor / Learner、参数同步与分布式训练
- Offline RL、Batch RL 与数据集基础设施
- 多智能体、自博弈、实验编排与结果管理
- RLHF / RLAIF / Post-training 与 LLM Infra 的交叉部分

## 目录组织约定

后续新增主题时，优先按“领域 / 主题 / 资料”三级组织，而不是按资料类型直接堆在根目录。

```text
领域目录/
└── topic-name/
    ├── README.md            # 主题导览、阅读顺序、关键问题
    ├── xxx.md               # 主笔记或专题文章
    └── assets/              # 当前主题笔记使用的图片、图表、附件
```

命名规则：

- 目录名使用英文 kebab-case，例如 `gpu-architecture`、`llm-serving`。
- Markdown 文件名使用英文 snake_case，例如 `nvidia_gpu_architecture_evolution.md`。
- 图片放在同主题目录的 `assets/` 下，文件名与文档引用保持语义一致。
- 根目录只保留全局入口，不直接放具体专题笔记。

## 笔记写法

每篇笔记尽量围绕这几个问题展开：

1. 这个模块解决什么系统问题？
2. 它的核心数据流 / 控制流是什么？
3. 关键设计权衡和瓶颈在哪里？
4. 代表性系统、论文、项目或官方资料有哪些？
5. 它和 LLM Infra / RL Infra 的哪一层连接？

## 阅读入口

- [00-foundations](./00-foundations/README.md)
- [01-llm-infra](./01-llm-infra/README.md)
- [02-rl-infra](./02-rl-infra/README.md)
