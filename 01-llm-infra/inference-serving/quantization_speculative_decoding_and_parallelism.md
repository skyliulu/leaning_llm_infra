# 量化、Speculative Decoding 与并行推理

版本日期：2026-06-16

---

## 摘要

大模型推理优化经常被混成一组名词：量化、speculative decoding、tensor parallel、FlashAttention、CUDA Graph、MoE parallel。它们确实都可能让服务更快或更省，但优化对象完全不同。量化主要降低权重、activation 或 KV Cache 的存储和带宽成本；speculative decoding 试图减少大模型 decode step 数量；并行推理解决单卡放不下或吞吐不足；kernel / runtime 优化则减少底层执行和调度开销。

本文按瓶颈分类梳理这些方法。目标不是给出固定选型结论，而是帮助读者判断：当前服务慢，是显存、带宽、decode 顺序性、通信、kernel、CPU 控制面，还是调度问题。

**阅读定位**：这篇文章扩展 [大模型推理服务化系统综述](./llm_inference_serving_overview.md) 中的优化分类章节。建议先读 [KV Cache、PagedAttention 与 Prefix Caching](./kv_cache_paged_attention_and_prefix_caching.md)，再读本文。

---

## 目录

- 第 1 章 按瓶颈分类推理优化
- 第 2 章 量化：显存和带宽优化
- 第 3 章 Speculative Decoding：减少目标模型步数
- 第 4 章 并行推理：让模型跨 GPU 运行
- 第 5 章 Kernel、Runtime 与 CPU 控制面优化
- 第 6 章 工程选型清单与参考资料

---

## 第 1 章 按瓶颈分类推理优化

**本章主线**：先建立分类框架，避免把所有优化手段都归为“更快”。

### 1.1 一个慢请求可能有多种原因

同样是用户觉得慢，底层原因可能完全不同：

| 症状 | 可能瓶颈 | 优先考虑 |
|---|---|---|
| 模型放不进单卡 | 权重显存 | weight quantization、tensor parallel |
| 长上下文并发低 | KV Cache 显存 | KV quantization、PagedAttention、offload |
| 首 token 慢 | prefill、排队、tokenization | chunked prefill、PD 分离、CPU 优化 |
| 流式 token 慢 | decode step、显存带宽 | speculative decoding、batching、KV 优化 |
| 多卡后不快 | 通信 | parallelism 拓扑、batch、placement |
| GPU 利用率低 | batch、CPU 调度、kernel launch | continuous batching、CUDA Graph、profiling |

优化前先定位瓶颈。否则可能出现典型误用：明明是 queueing 导致 TTFT 高，却去做 weight quantization；明明是 KV Cache 爆显存，却只调 tensor parallel。

### 1.2 贯穿例子：70B 模型服务

假设要部署一个 70B decoder-only 模型。服务同时面对短聊天、长 RAG 和代码生成。单张 GPU 放不下权重，多卡后通信明显；长 RAG 请求吃掉大量 KV Cache；代码生成输出很长，decode 阶段时间占比高。

这个服务可能需要组合：

- 权重量化或 tensor parallel 解决模型放置；
- PagedAttention 和 KV quantization 控制长上下文显存；
- continuous batching 和 chunked prefill 控制 TTFT / TPOT；
- speculative decoding 加速长输出；
- CUDA Graph 和 fused kernels 降低小 batch 开销。

**本章小结**：推理优化不是菜单式叠加，而是瓶颈驱动的组合设计。

## 第 2 章 量化：显存和带宽优化

**本章主线**：量化不是单一技术。要区分 weight、activation 和 KV Cache 量化，并分别评估质量、kernel 和硬件支持。

### 2.1 Weight quantization

Weight quantization 把模型权重从 FP16 / BF16 压到 INT8、INT4、FP8 或其他格式。收益主要有两个：

- 降低权重显存，让更大模型放入有限 GPU。
- 降低读取权重的带宽压力，提高部分场景吞吐。

常见路线包括 GPTQ、AWQ、bitsandbytes、FP8 和 TensorRT-LLM / vLLM 支持的多种量化后端。不同方法对校准数据、硬件支持和模型质量的要求不同。

### 2.2 Activation 和 KV Cache quantization

Activation quantization 关注推理过程中的中间激活，KV Cache quantization 关注缓存的 key / value。对长上下文 serving，KV quantization 特别重要，因为 KV Cache 会随 token 数和并发数增长。

KV quantization 的风险是 attention 质量。Key / value 的误差会影响后续所有 decode step，尤其在长上下文检索、代码生成和推理任务中，需要用 workload 级指标验证，而不是只看 perplexity 或短 benchmark。

### 2.3 量化的系统代价

量化可能引入：

- 额外 dequantize / quantize kernel。
- 不同 shape 下 kernel 性能差异。
- 模型质量回退。
- 和 tensor parallel、MoE、LoRA、speculative decoding 的兼容问题。
- 观测和回滚复杂度。

因此，生产中常见做法是分层评估：先验证质量，再验证单机吞吐和延迟，最后在真实请求分布下验证 SLO 和成本。

**本章小结**：量化优化的是存储和带宽，不是自动优化所有延迟。权重量化解决模型放置和带宽，KV 量化解决长上下文显存，但都必须验证质量和 kernel 支持。

## 第 3 章 Speculative Decoding：减少目标模型步数

**本章主线**：Speculative decoding 针对 decode 顺序性。它试图让小模型先猜多个 token，再让大模型验证。

### 3.1 标准 decode 的顺序瓶颈

自回归 decode 每一步只能生成一个 token。即使每步计算很快，长输出仍然要执行很多轮目标模型 forward。对大模型来说，每步还要读取大量权重和 KV Cache。

Speculative decoding 的直觉是：如果一个便宜 draft model 能猜出多个可能 token，target model 可以一次验证多个 token。被接受的 token 越多，目标模型执行步数越少。

### 3.2 基本流程

一个典型 speculative decoding 流程：

1. Draft model 根据当前上下文生成 `k` 个候选 token。
2. Target model 对这些候选 token 做一次或少数几次验证。
3. 按接受规则保留一段候选 token。
4. 如果某个 token 被拒绝，回退到 target model 分布采样。
5. 将接受的 token 追加到上下文，继续下一轮。

Leviathan 等人的工作强调在保持目标模型输出分布的前提下加速 decode；后续系统又发展出 n-gram speculation、Medusa、EAGLE、MLP draft、multi-token prediction 等变体。

### 3.3 什么时候有效

Speculative decoding 的收益取决于：

- Draft model 是否显著便宜。
- Draft token 接受率是否高。
- 输出是否足够长。
- Draft 与 target 的并行或调度开销是否低。
- 额外模型是否占用不可接受的显存。

如果接受率低，系统会频繁回退；如果输出很短，draft setup 成本可能不值；如果 GPU 已被显存卡住，额外 draft model 可能反而降低并发。

### 3.4 和量化、batching 的关系

Speculative decoding 和量化可以组合，但优化对象不同：

- 量化降低每步成本。
- Speculative decoding 降低目标模型步数。
- Continuous batching 提高在线 batch 利用率。
- KV Cache 管理决定这些请求能否同时驻留。

生产系统需要一起看。如果 speculative decoding 增加了 KV Cache、draft model 显存或调度复杂度，就要用 SLO 内 goodput 验证收益。

**本章小结**：Speculative decoding 针对自回归 decode 的顺序瓶颈。它不是“无损必快”的按钮，收益由 draft 成本、接受率、输出长度和系统调度共同决定。

## 第 4 章 并行推理：让模型跨 GPU 运行

**本章主线**：并行推理解决单卡容量和吞吐问题，但通信和调度代价会直接影响在线延迟。

### 4.1 Tensor Parallel

Tensor parallel 把单层矩阵计算切到多张 GPU。它适合单层权重太大或希望提高大 batch 吞吐的场景。代价是每层需要通信，例如 all-reduce 或 all-gather。在线小 batch 下，通信占比可能很高。

### 4.2 Pipeline Parallel

Pipeline parallel 把模型不同层放在不同 GPU 上。它能降低单卡权重占用，但会引入流水气泡。Prefill 和 decode 的计算时间差异会让气泡更明显，因此 Sarathi-Serve 等工作也关注 pipeline parallel 下 prefill/decode 不均衡。

### 4.3 Context Parallel 和长上下文

Context parallel 沿序列维度切分长上下文 attention。它适合极长上下文，但会引入 attention 相关通信和 KV Cache 分布问题。对长 RAG、代码库问答和长文档处理，这类方法可能比单纯 tensor parallel 更相关。

### 4.4 Expert Parallel

MoE 模型中，每个 token 只路由到部分 expert。Expert parallel 把不同 expert 放到不同 GPU 上。它的主要问题是 token routing、负载均衡和 all-to-all 通信。线上请求分布变化可能造成某些 expert 热点。

### 4.5 Replica Parallel

Replica parallel 是最简单的服务扩展方式：多份模型副本分摊请求。它适合中小模型和在线服务，但会重复占用权重显存。对于超大模型，单副本本身就可能需要 tensor parallel 或 pipeline parallel。

**本章小结**：并行推理不是训练并行的直接复制。在线推理要把通信、KV Cache、streaming、batch 动态变化和 SLO 一起考虑。

## 第 5 章 Kernel、Runtime 与 CPU 控制面优化

**本章主线**：当瓶颈不在算法层，而在执行层和控制面时，需要看 kernel、CUDA Graph、编译和 CPU 调度。

### 5.1 Attention 和 fused kernels

FlashAttention、FlashInfer、fused MoE kernel、fused norm / activation / quantization kernel 等优化，目标是减少 HBM 访问、中间 tensor 和 kernel 数量。它们通常和模型结构、dtype、GPU 架构强相关。

### 5.2 CUDA Graph

CUDA Graph 可以降低重复 kernel launch 的 CPU 开销，适合 shape 稳定的执行路径。但在线 serving 的 batch 和 sequence length 经常变化，系统需要在动态性和 graph capture 收益之间权衡。

### 5.3 编译与模型加载

torch.compile、TensorRT engine 构建、fast safetensors loading、weight streaming 等优化，分别影响运行时效率和冷启动速度。对弹性伸缩和灰度发布来说，模型加载时间和 warmup 时间也是生产指标。

### 5.4 CPU 控制面

Tokenizer、HTTP server、JSON 序列化、streaming、队列锁、Python multiprocessing 都可能成为瓶颈。GPU 利用率低时，不要只看 GPU kernel，也要看 CPU timeline。

**本章小结**：执行层优化通常很有效，但依赖硬件、shape 和 runtime。它们应该在 profiling 后使用，而不是替代系统层调度和缓存管理。

## 第 6 章 工程选型清单与参考资料

### 6.1 选型清单

1. 模型是否能单卡放下？不能，先看量化和并行。
2. 长上下文是否导致 KV Cache 爆显存？是，先看 PagedAttention、KV quantization、offload。
3. 输出很长且 TPOT 高？看 speculative decoding 和 decode batching。
4. 多卡后收益低？看通信拓扑、batch、parallelism 策略。
5. GPU 利用率低？看 CPU tokenization、调度、CUDA timeline。
6. 扩容慢？看权重加载、engine 构建、warmup。

## 参考资料

- vLLM Team, [Quantization](https://docs.vllm.ai/en/latest/features/quantization/)
- vLLM Team, [Speculative Decoding](https://docs.vllm.ai/en/latest/features/spec_decode/)
- Yaniv Leviathan et al., [Fast Inference from Transformers via Speculative Decoding](https://arxiv.org/abs/2211.17192)
- Charlie Chen et al., [Accelerating Large Language Model Decoding with Speculative Sampling](https://arxiv.org/abs/2302.01318)
- Tianle Cai et al., [Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads](https://arxiv.org/abs/2401.10774)
- NVIDIA, [TensorRT-LLM Documentation](https://docs.nvidia.com/tensorrt-llm/)
- SGLang Team, [Speculative Decoding and Quantization](https://docs.sglang.io/)
- vLLM Team, [Parallelism and Scaling](https://docs.vllm.ai/en/latest/)

**全文小结**：量化、speculative decoding、并行和 kernel 优化分别作用在不同瓶颈上。真正的推理优化要从 profiling 和 SLO 出发，把这些手段组合成稳定的服务策略。
