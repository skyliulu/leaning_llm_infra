# 01 LLM Infra

这个目录用于整理大模型相关基础设施，重点关注“**训练 - 推理 - 服务 - 评测**”全链路。

## 子领域拆分

### 1. 数据处理与 Tokenization

- 语料采集与清洗
- 去重与质量过滤
- Tokenizer 设计
- 数据配比与 curriculum
- 数据版本管理

### 2. 训练流水线

- Pretraining
- SFT
- Checkpoint 管理
- 容错恢复
- 实验配置管理

### 3. 并行与分布式训练

- Data Parallel
- Tensor Parallel
- Pipeline Parallel
- ZeRO / FSDP
- 通信优化与带宽瓶颈

### 4. 推理与 Serving

- Continuous batching
- KV Cache
- PagedAttention
- 量化推理
- 多模型部署与弹性伸缩

### 5. RAG / Agent Runtime

- 检索系统
- 向量数据库
- Prompt 编排
- Tool calling
- Memory / Session 管理

### 6. 评测与可观测性

- 离线评测
- 在线评测
- 延迟 / 吞吐 / 成本分析
- 日志、指标、Tracing
- 回归测试

### 7. 平台治理

- 权限与配额
- 多租户
- 成本控制
- 安全与审计
- 发布与灰度

## 建议记录方式

可以围绕以下问题记录每个主题：

1. 典型系统瓶颈在哪里？
2. 核心优化手段是什么？
3. 训练与推理的工程差异是什么？
4. 开源项目各自解决了哪一层问题？

## 可优先跟进的代表方向

- Megatron / DeepSpeed / FSDP
- vLLM / TensorRT-LLM / SGLang
- RAG pipeline 与向量检索系统
- LLM Serving 的可观测性与成本优化
- RLHF 与后训练系统如何接入 LLM 平台
