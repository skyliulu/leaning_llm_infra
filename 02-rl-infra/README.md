# 02 RL Infra

这个目录整理强化学习基础设施相关笔记，重点关注环境、采样、训练、评测和大规模实验平台。

这里不把重点放在单个 RL 算法推导上，而是关注算法要稳定、大规模、可复现地运行时，需要哪些系统组件。和 LLM 相关的 RLHF / RLAIF / post-training 内容也可以放在这里，但需要说明它和 [01-llm-infra](../01-llm-infra/README.md) 的连接点。

## 建议专题

### Environment Systems

- Gym / Env 抽象、任务封装与环境版本管理
- 模拟器、真实系统、在线环境与离线数据环境
- 并行环境执行、状态同步、reset 与 episode 管理
- 状态、动作、奖励、终止条件和安全约束定义

### Rollout and Data Pipeline

- Rollout worker、actor pool 与采样调度
- Experience collection、trajectory storage 与 replay buffer
- On-policy 数据新鲜度、off-policy 数据复用与优先级采样
- 数据持久化、回放、过滤、重加权与样本质量分析

### Training Systems

- Actor / Learner 架构与参数同步
- On-policy、off-policy、offline RL 的系统差异
- 单机、多机、异步、同步训练流程
- Checkpoint、恢复、随机性控制与复现实验
- 吞吐、样本效率、稳定性和资源利用率之间的权衡

### Experiment Platform

- 实验编排、配置管理和超参数搜索
- 多任务、多智能体、自博弈和联赛系统
- 指标记录、结果对比、回归测试与实验归档
- 失败任务恢复、资源调度、队列与优先级管理

### Offline RL and Dataset Infra

- 静态数据集管理、质量评估和覆盖度分析
- OPE（Off-policy Evaluation）与评测可信度
- 数据切分、版本、污染、偏差和复现
- Batch RL 数据管线与训练任务解耦

### RLHF and Post-training

- Preference data pipeline、reward model 训练与数据版本
- PPO、DPO、GRPO 等训练流程的系统边界
- Online feedback loop、采样服务、模型服务和训练任务联动
- Reward hacking、评测、安全约束与人工审核闭环

### Safety and Evaluation

- 泛化评测、稳定性评测和压力测试
- Reward hacking、specification gaming 与异常行为分析
- 安全约束、干预机制和训练过程可观测性
- 不同 seed、环境版本和策略版本之间的可比性

## 推荐新增目录

后续可以按专题逐步新增目录：

```text
02-rl-infra/
├── environment-systems/
├── rollout-data-pipeline/
├── training-systems/
├── experiment-platform/
├── offline-rl-dataset-infra/
├── rlhf-post-training/
└── safety-evaluation/
```

每个专题目录内先保持简单：

```text
topic-name/
├── README.md
├── main_note.md
└── assets/
```

等某个专题变大后，再拆 `concepts/`、`papers/`、`projects/`、`notes/`。

## 推荐阅读顺序

1. 先看 `environment-systems`，明确 RL 系统里的任务边界和数据来源。
2. 再看 `rollout-data-pipeline`，理解样本如何产生、存储、复用和调度。
3. 接着看 `training-systems`，把 actor、learner、参数同步和 checkpoint 串起来。
4. 然后看 `experiment-platform`，理解为什么 RL 实验管理比普通监督训练更麻烦。
5. 最后看 `rlhf-post-training` 和 `safety-evaluation`，连接到 LLM 后训练与安全评测。

## 笔记关注点

每篇 RL Infra 笔记尽量回答：

1. 这个模块服务的是环境、采样、训练、评测还是实验管理？
2. 数据是在线产生、离线读取，还是两者混合？
3. 系统瓶颈在样本生成、训练吞吐、同步通信、评测成本还是实验稳定性？
4. 算法假设对基础设施提出了什么要求？
5. 哪些问题最容易破坏可复现性和结果可信度？
