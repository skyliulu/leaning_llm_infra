# Learning LLM Infra

这个仓库用于沉淀我对 **LLM Infra**、**RL Infra** 以及相关底层系统的学习笔记。它不是资料堆放区，也不是单篇文章集合，而是一张逐步长出来的学习地图：每个主题尽量回答“它解决什么问题、核心机制是什么、系统瓶颈在哪里、和其他模块如何连接”。

## 项目主线

### 00 共性基础

这一部分放 LLM Infra / RL Infra 都会反复依赖的底层能力。重点不是写成通用百科，而是服务后续工程理解：为什么训练会被显存、通信、调度、IO 或可观测性卡住，以及这些问题在真实系统里怎样出现。

当前重点：

- [GPU 架构](./00-foundations/gpu-architecture/README.md)：CUDA 编程模型、GPU 硬件层级、存储层级、Tensor Core / RT Core / NVLink，以及 NVIDIA GPU 架构演进。

后续会继续围绕计算机系统、分布式系统、深度学习系统和工程工具链补充。

### 01 大模型基础设施

这一部分关注大模型从数据到训练、推理、服务化、评测和平台治理的完整链路。共性系统知识不在这里重复展开，而是重点解释这些基础能力如何组合成 LLM 系统。

计划覆盖：

- 数据基础设施：语料清洗、去重、配比、版本、数据加载与污染检测。
- 训练系统：pretraining、SFT、post-training、checkpoint、并行策略、显存优化与训练吞吐。
- 推理与服务化：KV Cache、continuous batching、PagedAttention、量化、多副本调度、延迟与成本。
- RAG 与 Agent Runtime：检索、上下文组装、tool calling、session memory、权限、状态与失败恢复。
- 评测与平台治理：离线/在线评测、可观测性、成本、权限、多租户、安全和发布治理。

### 02 强化学习基础设施

这一部分关注强化学习算法要稳定、大规模、可复现地运行时，需要哪些系统组件。它既包括传统 RL 的环境、采样、训练和实验平台，也包括 LLM 后训练中的 RLHF / RLAIF / preference optimization。

当前重点：

- [强化学习基础理论](./02-rl-infra/rl-theory-foundations/rl_theory_foundations.md)：从 MDP、Bellman 方程、动态规划、Monte Carlo、TD、函数近似、policy gradient、actor-critic，到 PPO、DPO、GRPO 和长程 Agent RL 的基础理论主线。

后续会继续围绕环境系统、rollout 与数据管线、训练系统、实验平台、Offline RL 数据集、安全评测和后训练系统补充。

## 内容组织原则

新增主题优先按“领域 / 专题 / 文章与资源”组织：

```text
领域目录/
└── topic-name/
    ├── README.md            # 专题导览、阅读顺序、关键问题
    ├── xxx.md               # 正文文章，文件名使用有语义的 snake_case
    ├── assets/              # 图片、图表、附件
    └── examples/            # 可选：代码、配置或实验样例
```

约定：

- 目录名使用英文 `kebab-case`，例如 `gpu-architecture`、`inference-serving`。
- Markdown 正文文件使用英文 `snake_case`，例如 `cuda_intro.md`、`rl_theory_foundations.md`。
- `README.md` 负责导览，不承载长篇正文。
- 图片、SVG、示例代码放在同一专题目录下，保持源码仓库和博客站点使用同一套相对路径。
- 如果某个主题积累到多篇内容，再拆成 `concepts/`、`papers/`、`projects/`、`notes/`；内容少时先保持扁平。

## 写作关注点

每篇笔记尽量围绕这些问题展开：

1. 这个模块解决什么系统问题？
2. 它处在 LLM Infra / RL Infra 的哪一层？
3. 输入、输出、状态和核心数据流是什么？
4. 关键设计权衡、瓶颈和失败模式在哪里？
5. 代表性系统、论文、项目或官方资料有哪些？

理论文章还需要额外注意：先建立直观例子，再进入形式化定义；公式推导要完整但不能淹没主线；图解应服务理解，而不是装饰。

## 项目级工作流

仓库中维护了两个项目级 skill，用于把可复用的写作和制图经验固化下来：

- `write-technical-blog`（路径：`.agents/skills/write-technical-blog/SKILL.md`）：用于资料分析、术语铺垫、案例设计、公式推导、章节衔接、引用和发布前检查。
- `design-technical-diagrams`（路径：`.agents/skills/design-technical-diagrams/SKILL.md`）：用于设计论文风格技术图、选择 SVG/PNG/绘图库/imagegen，以及完成结构检查和渲染验收。

根目录的 `AGENTS.md` 记录了给后续 Agent 使用的项目级协作规范。
