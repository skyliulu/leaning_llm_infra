# 推理与服务化

这个专题整理大模型从“能在单卡上生成文本”到“能稳定承载线上请求”的系统链路。它不重复 CUDA、GPU 架构和通用分布式调度的基础知识，而是关注这些能力如何组合成 LLM inference serving：请求如何排队，prefill 和 decode 如何占用 GPU，KV Cache 如何管理，多副本如何调度，以及线上系统如何在延迟、吞吐、显存和成本之间取舍。

## 当前笔记

- [大模型推理服务化系统综述](./llm_inference_serving_overview.md)
- [KV Cache、PagedAttention 与 Prefix Caching](./kv_cache_paged_attention_and_prefix_caching.md)
- [Prefill / Decode 调度与分离部署](./prefill_decode_scheduling_and_disaggregation.md)
- [量化与低精度推理](./quantization_and_low_precision_inference.md)
- [Speculative Decoding 与解码加速](./speculative_decoding_and_decode_acceleration.md)
- [多 GPU 并行推理与 MoE Serving](./parallel_inference_and_moe_serving.md)
- [推理优化方法索引：量化、Speculative Decoding 与并行推理](./quantization_speculative_decoding_and_parallelism.md)
- [生产推理服务栈与可观测性](./production_serving_stack_and_observability.md)

## 阅读重点

建议按四条线阅读：

1. 先理解一次请求的生命周期：API、tokenization、prefill、decode、sampling、streaming response。
2. 再理解核心状态：KV Cache、batch、request queue、model replica 和 GPU memory。
3. 接着看优化手段：continuous batching、PagedAttention、prefix caching、量化、speculative decoding 和多 GPU 并行推理。
4. 最后进入生产视角：多模型服务、弹性伸缩、限流、灰度发布、观测指标和成本核算。

## 后续可扩展方向

- 将 `KV Cache` 文章继续扩展为长上下文推理和跨节点 cache transfer 专题。
- 将 `Prefill / Decode` 文章继续扩展为 vLLM、Sarathi-Serve、DistServe、SGLang PD disaggregation 的系统对比。
- 将 `量化与低精度推理` 继续扩展为 FP8 / INT4 / KV quantization 的质量评估和 kernel 细节。
- 将 `Speculative Decoding 与解码加速` 继续扩展为 Medusa、EAGLE、n-gram speculation 和 runtime 调度对比。
- 将 `多 GPU 并行推理与 MoE Serving` 继续扩展为 DeepSeek 类 MoE、MLA / hybrid attention 和多节点拓扑案例。
- 将 `生产服务栈` 文章继续扩展为可运行的 Kubernetes / Ray Serve / KServe 部署样例和观测仪表盘。

## 资源组织

- `assets/`：请求生命周期、调度、缓存和部署结构相关图。
- 正文引用优先使用论文、官方文档和公开项目文档，不引用本地路径。
