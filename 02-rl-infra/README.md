# 02 RL Infra

这个目录用于整理强化学习相关基础设施，重点关注“**环境 - 采样 - 训练 - 评测 - 实验平台**”。

## 子领域拆分

### 1. 环境与仿真平台

- Gym / Env 抽象
- 模拟器与任务封装
- 并行环境执行
- 状态 / 动作 / 奖励定义
- 环境版本管理

### 2. 采样与数据管线

- Rollout worker
- Experience collection
- Replay buffer
- 优先级采样
- 数据持久化与回放

### 3. 训练系统

- On-policy / Off-policy 流程
- Learner / Actor 架构
- 参数同步
- 分布式训练
- Checkpoint 与恢复

### 4. Offline RL / Batch RL

- 静态数据集管理
- Dataset quality
- OPE（Off-policy Evaluation）
- 数据过滤与重加权
- 训练可复现性

### 5. 大规模实验平台

- 实验编排
- 超参数搜索
- 多任务 / 多智能体训练
- 自博弈系统
- 结果对比与回归

### 6. RLHF / Post-training

- Preference data pipeline
- Reward model training
- PPO / DPO / GRPO 等训练流程
- Online feedback loop
- 与 LLM Serving / Data Infra 的联动

### 7. 评测与安全

- Reward hacking 分析
- 泛化评测
- 稳定性评测
- 安全约束
- 训练过程可观测性

## 建议记录方式

可以从下面几个角度持续补充：

1. 算法假设依赖什么样的基础设施？
2. 训练吞吐和样本效率如何平衡？
3. 环境、采样器、训练器之间如何解耦？
4. 哪些模块最容易成为实验不稳定来源？

## 可优先跟进的代表方向

- Gymnasium / EnvPool / Isaac Gym
- Ray RLlib / CleanRL / Acme
- Replay buffer 与大规模 rollout 系统
- Offline RL 数据集基础设施
- RLHF / Post-training 与 LLM Infra 的交叉地带
