# 推理优化方法索引：量化、Speculative Decoding 与并行推理

版本日期：2026-06-16

---

## 说明

这篇文章原本把量化、speculative decoding 和并行推理放在一起讲。随着专题继续扩展，这三类技术已经不适合放在同一篇长文中：它们都属于推理优化，但优化对象、数学结构和系统代价完全不同。

因此，本文保留为索引页，帮助读者判断应该进入哪一篇扩展文章。

## 如何选择阅读路径

| 你关心的问题 | 应阅读 | 核心对象 |
|---|---|---|
| 模型放不进 GPU，长上下文并发低，想降低显存和带宽 | [量化与低精度推理](./quantization_and_low_precision_inference.md) | 权重、activation、KV Cache、低精度 kernel |
| 输出很长，decode step 太多，想减少目标模型 forward 次数 | [Speculative Decoding 与解码加速](./speculative_decoding_and_decode_acceleration.md) | draft model、target verification、接受率、KV 状态 |
| 单副本需要多 GPU，MoE expert 跨卡，通信影响延迟 | [多 GPU 并行推理与 MoE Serving](./parallel_inference_and_moe_serving.md) | tensor / pipeline / context / expert / replica parallel |

## 三者之间的关系

量化、speculative decoding 和并行推理可以组合，但它们不解决同一个瓶颈：

- 量化降低每步计算涉及的数据表示成本，主要影响显存、带宽和 kernel。
- Speculative decoding 降低 target model 的顺序 decode 步数，主要影响长输出 TPOT。
- 并行推理改变模型和请求在多 GPU 上的放置，主要影响容量、吞吐和通信。

如果一个 70B 模型服务同时面对短聊天、长 RAG 和代码生成，真实系统可能会同时使用三类手段：用量化降低权重和 KV Cache 成本，用 speculative decoding 加速长输出，用 tensor / expert / replica parallel 承载单副本容量和在线流量。但工程上应该分别验证每类手段的收益，而不是把它们当成一个“优化套餐”。

## 推荐顺序

1. 先读 [KV Cache、PagedAttention 与 Prefix Caching](./kv_cache_paged_attention_and_prefix_caching.md)，理解推理服务的核心状态。
2. 再读 [Prefill / Decode 调度与分离部署](./prefill_decode_scheduling_and_disaggregation.md)，理解在线请求如何被调度。
3. 根据瓶颈进入量化、解码加速或并行推理专题。
4. 最后读 [生产推理服务栈与可观测性](./production_serving_stack_and_observability.md)，把 runtime 优化放回生产平台。

## 参考资料

- vLLM Team, [vLLM Documentation](https://docs.vllm.ai/en/latest/)
- SGLang Team, [SGLang Documentation](https://docs.sglang.io/)
- NVIDIA, [TensorRT-LLM Documentation](https://docs.nvidia.com/tensorrt-llm/)
