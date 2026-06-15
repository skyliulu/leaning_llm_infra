# 推理与服务化

这个专题整理大模型从“能在单卡上生成文本”到“能稳定承载线上请求”的系统链路。它不重复 CUDA、GPU 架构和通用分布式调度的基础知识，而是关注这些能力如何组合成 LLM inference serving：请求如何排队，prefill 和 decode 如何占用 GPU，KV Cache 如何管理，多副本如何调度，以及线上系统如何在延迟、吞吐、显存和成本之间取舍。

## 当前笔记

- [大模型推理服务化系统综述](./llm_inference_serving_overview.md)

## 阅读重点

建议按四条线阅读：

1. 先理解一次请求的生命周期：API、tokenization、prefill、decode、sampling、streaming response。
2. 再理解核心状态：KV Cache、batch、request queue、model replica 和 GPU memory。
3. 接着看优化手段：continuous batching、PagedAttention、prefix caching、speculative decoding、quantization 和并行推理。
4. 最后进入生产视角：多模型服务、弹性伸缩、限流、灰度发布、观测指标和成本核算。

## 后续可扩展文章

- `kv_cache_paged_attention_and_prefix_caching.md`：围绕 KV Cache 公式、PagedAttention、prefix caching、KV offload 和 cache transfer 展开。
- `prefill_decode_scheduling_and_disaggregation.md`：解释 continuous batching、chunked prefill、prefill / decode disaggregation 和 SLO-aware scheduling。
- `quantization_speculative_decoding_and_parallelism.md`：对比量化、speculative decoding、并行推理和 kernel / runtime 优化的收益与代价。
- `production_serving_stack_and_observability.md`：梳理 vLLM、SGLang、TensorRT-LLM、TGI、Kubernetes / KServe / Ray Serve / Triton 等生产栈和观测体系。

## 资源组织

- `assets/`：请求生命周期、调度、缓存和部署结构相关图。
- 正文引用优先使用论文、官方文档和公开项目文档，不引用本地路径。
