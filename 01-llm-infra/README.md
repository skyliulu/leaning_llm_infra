# 01 LLM Infra

这个目录整理大模型基础设施相关笔记，重点关注从数据到训练、推理、服务化、评测和平台治理的完整链路。

这里不重复记录 GPU、Linux、分布式系统等通用基础；这些内容优先放在 [00-foundations](../00-foundations/README.md)。本目录更关注这些基础能力如何组合成 LLM 系统。

## 建议专题

### Data Infrastructure

- 语料采集、清洗、去重与质量过滤
- Tokenization、数据配比、curriculum 与数据混合
- 数据版本、数据血缘、数据审计与污染检测
- 大规模数据加载、缓存、shuffle 与训练吞吐

### Training Systems

- Pretraining、SFT、continued pretraining 与 post-training 流水线
- Checkpoint、容错恢复、实验配置与训练状态管理
- Data Parallel、Tensor Parallel、Pipeline Parallel、Sequence Parallel
- ZeRO、FSDP、offload、activation checkpointing 与显存优化
- 通信重叠、straggler、训练吞吐与集群利用率

### Inference and Serving

- Prefill / decode 分离与 KV Cache 管理
- Continuous batching、PagedAttention、speculative decoding
- 量化、并行推理、模型切分与多副本调度
- 多模型服务、弹性伸缩、限流、降级与灰度发布
- 延迟、吞吐、显存占用和成本之间的权衡

### RAG and Agent Runtime

- 检索、rerank、query rewrite 与上下文组装
- 向量数据库、索引构建、召回质量与延迟优化
- Tool calling、session memory、workflow orchestration
- Agent runtime 的权限、状态、可观测性与失败恢复

### Evaluation and Observability

- 离线评测、在线评测、A/B、回归测试
- 延迟、吞吐、错误率、成本、GPU 利用率和队列指标
- 日志、metrics、tracing、prompt / response 审计
- 数据漂移、模型退化、安全问题与发布回滚

### Platform Governance

- 权限、配额、多租户与资源隔离
- 成本核算、预算控制、容量规划
- 模型注册、版本管理、发布治理与审计
- 安全、隐私、合规和企业集成

## 推荐新增目录

后续可以按专题逐步新增目录：

```text
01-llm-infra/
├── data-infra/
├── training-systems/
├── distributed-training/
├── inference-serving/
├── rag-agent-runtime/
├── evaluation-observability/
└── platform-governance/
```

每个专题目录内先保持简单：

```text
topic-name/
├── README.md
├── main_note.md
└── assets/
```

等某个专题积累到多篇笔记后，再拆成 `concepts/`、`papers/`、`projects/`、`notes/`。

## 推荐阅读顺序

1. 先从 `training-systems` 理解一次 LLM 训练流水线如何跑起来。
2. 再看 `distributed-training`，把并行策略、显存优化和通信瓶颈串起来。
3. 接着看 `inference-serving`，理解训练系统和在线服务系统的差异。
4. 然后补 `evaluation-observability`，把系统是否稳定、有效、可回归说清楚。
5. 最后看 `rag-agent-runtime` 和 `platform-governance`，理解应用层和生产平台如何接入。

## 笔记关注点

每篇 LLM Infra 笔记尽量回答：

1. 这个模块处在 LLM 系统链路的哪一层？
2. 输入、输出、状态和关键数据流是什么？
3. 主要瓶颈是算力、显存、通信、IO、调度还是质量评估？
4. 常见工程方案分别牺牲了什么、换来了什么？
5. 代表性系统、论文、项目或官方文档有哪些？
