# 生产推理服务栈与可观测性

版本日期：2026-06-16

---

## 摘要

推理引擎能在单机上高效生成 token，并不等于生产推理服务已经完成。生产服务还需要 API 网关、鉴权、限流、多模型路由、灰度发布、弹性伸缩、观测、成本归因、故障恢复和安全治理。vLLM、SGLang、TensorRT-LLM、TGI 解决的是执行层和部分服务层问题；Ray Serve、KServe、Kubernetes、Triton、NVIDIA Dynamo 等更偏平台和编排层。

本文从一个企业内部多模型推理平台出发，梳理生产推理服务栈的分层、主流系统定位、关键指标和排障路径。重点是帮助读者区分“推理 runtime”、“模型服务框架”和“生产平台”。

**阅读定位**：这篇文章扩展 [大模型推理服务化系统综述](./llm_inference_serving_overview.md) 中生产服务章节，适合在理解 KV Cache、调度和优化手段后阅读。

---

## 目录

- 第 1 章 生产服务栈的分层
- 第 2 章 主流推理 runtime 与平台组件
- 第 3 章 发布、路由与多租户治理
- 第 4 章 可观测性：从 API 到 GPU 和 KV Cache
- 第 5 章 Benchmark 与容量规划
- 第 6 章 工程检查清单与参考资料

---

## 第 1 章 生产服务栈的分层

**本章主线**：先把生产服务拆层，避免把 vLLM、KServe、Ray Serve、Triton 等系统混在一个层面比较。

### 1.1 一个企业推理平台的最小形态

假设企业内部要提供多个模型：通用聊天、代码生成、embedding、rerank、reward model 和私有微调模型。平台需要：

- 对外暴露 OpenAI-compatible API；
- 支持 streaming；
- 按部门、应用和模型做鉴权与配额；
- 多副本路由和灰度发布；
- 监控 TTFT、TPOT、错误率、GPU 利用率和成本；
- 支持模型版本回滚；
- 支持批量评测和在线服务共存。

这已经远超一个推理引擎能单独解决的范围。

### 1.2 分层模型

| 层次 | 职责 | 典型组件 |
|---|---|---|
| API / Gateway | 认证、限流、协议、streaming、审计 | Nginx、Envoy、API Gateway、LiteLLM |
| Routing / Policy | 多模型路由、灰度、配额、优先级 | 自研 router、Ray Serve、KServe InferenceGraph |
| Serving Runtime | batch、KV Cache、sampling、GPU 执行 | vLLM、SGLang、TensorRT-LLM、TGI |
| Cluster Platform | Pod、GPU 节点、弹性、服务发现 | Kubernetes、KubeRay、KServe、Triton |
| Observability | metrics、logs、traces、profiles | Prometheus、Grafana、OpenTelemetry、Nsight |
| Governance | 成本、安全、合规、模型版本 | Model registry、IAM、审计系统 |

不同团队可以从不同层切入。研究团队可能先用 vLLM server；平台团队需要把 runtime 放进 Kubernetes 和观测体系；企业服务团队还要处理权限、成本和审计。

**本章小结**：生产服务栈不是单层系统。Runtime 负责高效生成 token，平台负责把它变成可靠、可治理、可扩展的服务。

## 第 2 章 主流推理 runtime 与平台组件

**本章主线**：对比主流系统定位，避免把不同层次的工具直接排名。

### 2.1 Runtime 层

| 系统 | 定位 | 代表能力 | 注意点 |
|---|---|---|---|
| vLLM | 通用高吞吐 LLM serving runtime | PagedAttention、continuous batching、OpenAI-compatible server、prefix caching、speculative decoding、metrics | 生态活跃，适合作为默认候选 |
| SGLang | 结构化生成和高吞吐 runtime | RadixAttention、structured outputs、tool / reasoning parser、PD disaggregation、observability | 适合复杂生成程序和结构化输出 |
| TensorRT-LLM | NVIDIA GPU 优化 runtime | TensorRT engine、in-flight batching、paged KV cache、量化、多 GPU / 多节点 | 更偏 NVIDIA 生产性能，需要考虑 engine 构建 |
| TGI | Hugging Face text generation server | streaming、continuous batching、tensor parallelism、Prometheus metrics、PagedAttention | 官方文档已标注 maintenance mode，新项目需谨慎 |
| llama.cpp / MLX / local engines | 本地和边缘推理 | CPU / consumer GPU、本地部署 | 更适合本地和轻量场景 |

### 2.2 平台和编排层

| 系统 | 定位 | 和 runtime 的关系 |
|---|---|---|
| Kubernetes | 容器和集群调度基础 | 管 Pod、节点、GPU device plugin、服务发现 |
| KServe | Kubernetes 上的模型服务框架 | 管 InferenceService、autoscaling、模型协议 |
| Ray Serve | Python-native 服务编排 | 管副本、路由、组合 DAG，可承载 vLLM 等 runtime |
| NVIDIA Triton | 通用 inference server | 支持多 backend，可和 TensorRT / TensorRT-LLM 生态结合 |
| NVIDIA Dynamo / llm-d 等 | LLM serving 平台化组件 | 更偏多节点、KV routing、disaggregation、生产 stack |

选型时应问：你需要的是 runtime，还是服务编排，还是完整平台？把 runtime 当平台会缺少治理；把平台当 runtime 又可能忽略 GPU 执行效率。

**本章小结**：vLLM / SGLang / TensorRT-LLM 是执行层候选；KServe / Ray Serve / Kubernetes / Triton 是平台层候选。生产系统通常是组合，而不是单选。

## 第 3 章 发布、路由与多租户治理

**本章主线**：生产推理服务必须处理模型版本、流量路由和租户隔离。

### 3.1 模型发布不是替换文件

模型发布涉及：

1. 下载权重和 tokenizer。
2. 构建或加载 engine。
3. Warmup，避免首批请求冷启动。
4. 灰度导流。
5. 监控质量、延迟和错误率。
6. 异常时回滚。

如果新版本 tokenizer、chat template、LoRA adapter 或系统提示词变化，prefix cache 和 KV Cache 也需要正确失效。否则可能出现缓存污染或错误复用。

### 3.2 路由策略

常见路由策略包括：

- 按模型名路由。
- 按租户、项目或 API key 路由。
- 按上下文长度路由到不同副本。
- 按 SLO 路由到低延迟或低成本池。
- 按灰度比例路由到新版本。
- 按 GPU 池和 KV Cache locality 路由。

对于长上下文或 PD disaggregation，路由层还要考虑 cache locality。把请求随意打散到任意副本，可能破坏 prefix cache 命中率。

### 3.3 多租户治理

多租户平台需要：

- 配额：requests/min、tokens/min、concurrency、max context。
- 优先级：生产流量、实验流量、批处理流量区分。
- 隔离：不同租户是否共享模型副本和 KV Cache。
- 成本归因：按 token、GPU time、模型和租户计费。
- 安全审计：prompt / response、工具调用、敏感信息。

这些能力不一定由推理 runtime 提供，但 runtime 必须暴露足够的指标和控制点。

**本章小结**：生产推理平台的难点不只是“跑得快”，还包括“谁能用、用多少、用哪个版本、出问题怎么回滚、成本算给谁”。

## 第 4 章 可观测性：从 API 到 GPU 和 KV Cache

**本章主线**：推理服务排障必须把 API、队列、调度、GPU、KV Cache 和平台指标串起来。

### 4.1 指标分层

| 层次 | 核心指标 |
|---|---|
| API | QPS、错误率、限流、客户端取消、stream 中断 |
| Queue | waiting time、queue length、admission reject、priority |
| Scheduler | running requests、batch size、prefill/decode mix、chunk size |
| GPU | prefill latency、decode latency、tokens/s、GPU utilization、HBM bandwidth |
| KV Cache | used blocks、free blocks、hit rate、eviction、offload bytes、transfer latency |
| Platform | replica health、pod restart、cold start、model loading、node pressure |
| Cost | cost/token、GPU hours、tenant attribution、idle capacity |

### 4.2 Trace 比单点指标更重要

一次慢请求需要能串起：

1. 到达 API gateway 的时间。
2. 排队时间。
3. 被哪个副本接收。
4. prefill 开始和结束。
5. 每个 decode step 或聚合 decode latency。
6. KV Cache 分配、命中、offload 或 transfer。
7. streaming response 的完成或中断。

没有 trace，TTFT 高只能猜；有 trace，才能知道是排队、prefill、tokenizer、cache transfer 还是网络导致。

### 4.3 常见告警

推理服务建议至少设置：

- TTFT p99 超过 SLO；
- TPOT p99 超过 SLO；
- queue length 持续增长；
- KV Cache 使用率超过阈值；
- prefix cache hit rate 异常下降；
- GPU utilization 低但 queue 高；
- replica restart 或 cold start 增多；
- error rate 或 client cancel rate 上升；
- cost/token 异常上升。

**本章小结**：可观测性不是最后加的仪表盘，而是推理平台能否稳定运营的基础。没有分层指标和 trace，优化只能靠猜。

## 第 5 章 Benchmark 与容量规划

**本章主线**：容量规划必须以 workload 分布和 SLO 为输入，而不是只看单机 tokens/s。

### 5.1 Benchmark 输入

一个有用的 benchmark 应说明：

- 输入长度分布；
- 输出长度分布；
- 到达过程：固定并发、Poisson arrival、trace replay；
- 生成参数：temperature、top-p、beam search、structured output；
- 模型版本和 tokenizer；
- GPU 型号、显存、网络；
- runtime 版本、并行策略、量化方式；
- SLO：TTFT、TPOT、E2E latency。

### 5.2 容量规划方法

容量规划可以按以下步骤做：

1. 用真实或近似流量 trace 估计输入/输出长度分布。
2. 在目标 runtime 上测单副本性能。
3. 记录满足 SLO 的 goodput，而不是最大 throughput。
4. 加入租户优先级和长上下文流量。
5. 评估扩容后的冷启动、权重加载和 warmup。
6. 计算 cost/token 和峰值冗余。

### 5.3 不同流量池分开规划

短聊天、长 RAG、批处理、Agent rollout 不一定应该共享同一个副本池。分池可以降低干扰：

- 短请求池：低 TTFT / TPOT，限制 max context。
- 长上下文池：更大 KV Cache 预算，可能支持 offload。
- 批处理池：更追求吞吐和成本。
- 实验池：低优先级，可抢占或降级。

**本章小结**：容量规划不是根据平均 QPS 乘一个系数，而是根据长度分布、SLO、KV Cache、冷启动和租户策略决定副本池。

## 第 6 章 工程检查清单与参考资料

### 6.1 生产检查清单

1. API 层是否支持鉴权、限流、streaming 和审计？
2. Runtime 是否暴露 TTFT、TPOT、batch、KV Cache 指标？
3. 是否支持灰度、回滚和模型 warmup？
4. 是否能按租户做配额和成本归因？
5. 是否能按请求类型分池？
6. 是否有 trace 串起 API、queue、prefill、decode 和 KV Cache？
7. 是否有冷启动和扩容压测？
8. 是否有长上下文和缓存失效策略？

## 参考资料

- vLLM Team, [vLLM Documentation](https://docs.vllm.ai/en/latest/)
- vLLM Team, [Production Metrics](https://docs.vllm.ai/en/latest/)
- SGLang Team, [SGLang Documentation](https://docs.sglang.io/)
- Hugging Face, [Text Generation Inference Documentation](https://huggingface.co/docs/text-generation-inference/)
- NVIDIA, [TensorRT-LLM Documentation](https://docs.nvidia.com/tensorrt-llm/)
- Ray Team, [Ray Serve Documentation](https://docs.ray.io/en/latest/serve/index.html)
- KServe, [KServe Documentation](https://kserve.github.io/website/)
- NVIDIA, [Triton Inference Server Documentation](https://docs.nvidia.com/deeplearning/triton-inference-server/)
- OpenTelemetry, [OpenTelemetry Documentation](https://opentelemetry.io/docs/)

**全文小结**：生产推理服务是 runtime、路由、平台、观测和治理的组合。真正可用的平台不仅要生成 token 快，还要可发布、可观测、可隔离、可回滚、可核算成本。
