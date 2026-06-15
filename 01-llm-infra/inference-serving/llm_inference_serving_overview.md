# 大模型推理服务化系统综述

版本日期：2026-06-15

---

## 摘要

大模型推理服务化解决的不是“模型能不能生成下一个 token”，而是“在多用户、多请求、多副本、多 GPU 的环境中，如何稳定、低延迟、低成本地持续生成 token”。训练系统追求长时间高吞吐，推理系统则同时面对用户可感知延迟、请求到达随机性、上下文长度差异、KV Cache 显存压力、调度公平性和线上故障恢复。

本文用一个在线聊天服务作为贯穿例子：用户请求进入 API gateway 后先被 tokenizer 转成 token，随后进入调度器；模型先做 prefill 处理完整 prompt，再进入逐 token 的 decode 循环；生成结果以 streaming response 返回。围绕这条链路，本文解释 LLM 推理服务的核心对象、性能瓶颈和常见优化手段，并把 vLLM、Orca、TensorRT-LLM、SGLang、Hugging Face TGI 等系统放到同一张系统地图中理解。

**文档定位**：本文是推理与服务化专题的入口文章。它不替代某个推理引擎的 API 手册，而是建立“请求生命周期 → GPU 执行路径 → KV Cache 状态 → 调度策略 → 生产指标”的系统框架。

**前置知识**：读者最好已经了解 Transformer decoder 的自回归生成、GPU 显存与 kernel 执行的基本概念。CUDA 和 GPU 细节可先阅读 [CUDA 编程模型与 GPU 计算系统综述](../../00-foundations/gpu-architecture/cuda_intro.md)。

---

## 目录

- 第 1 章 推理服务到底在调度什么
- 第 2 章 Prefill、Decode 与 KV Cache
- 第 3 章 Continuous Batching 与 PagedAttention
- 第 4 章 并行、量化与 speculative decoding
- 第 5 章 从单模型引擎到生产服务
- 第 6 章 学习路线与参考资料

---

## 第 1 章 推理服务到底在调度什么

**本章主线**：先把推理服务从“调用模型”改写成“管理一批随时间变化的请求”。然后定义 request、batch、replica、queue 和 streaming response，最后用一张图说明一次请求跨越了哪些系统边界。

### 1.1 在线推理的输入不是一个静态 batch

离线推理通常可以提前准备一个固定 batch：输入长度相近，输出长度可控，系统只需尽量把 GPU 填满。在线 LLM 服务不同，请求按随机时间到达，prompt 长度不同，输出长度也要到生成结束后才知道。一个用户可能只问一句短问题，另一个用户可能带着长上下文生成代码，还有用户可能中途断开连接。

因此，推理服务调度的对象至少包括五类状态：

| 对象 | 含义 | 为什么重要 |
|---|---|---|
| Request | 一个用户请求及其 prompt、生成参数和连接状态 | 决定输入长度、输出预算、优先级和取消逻辑 |
| Queue | 等待进入模型执行的请求集合 | 决定排队延迟、限流和公平性 |
| Batch | 当前一起执行的请求片段 | 决定 GPU 利用率、padding 开销和尾延迟 |
| Replica | 一份可执行模型副本，通常绑定一组 GPU | 决定容量、隔离和故障影响范围 |
| KV Cache | 每个请求已生成上下文的 key / value 状态 | 决定显存占用和 decode 吞吐 |

这也是为什么 LLM serving 不能只看模型结构。它更像一个状态密集型的在线系统：每个请求都在不断增长，每个 decode step 都会改写 KV Cache，每个新请求都可能加入当前执行队列。

### 1.2 贯穿例子：多用户聊天服务

考虑一个面向研发团队的内部聊天服务。它使用一个 decoder-only LLM，提供 OpenAI-compatible API，要求首 token 延迟不要太高，回答过程要流式返回，并且要在夜间批量任务和白天在线请求之间共享 GPU 集群。

一次请求会经历以下步骤：

1. API gateway 接收请求，完成认证、限流和参数校验。
2. Tokenizer 把 prompt 转成 token ids，并估算上下文长度。
3. Scheduler 决定请求进入哪个模型副本、什么时候和哪些请求一起执行。
4. Prefill 阶段一次性处理 prompt，为每一层 attention 写入 KV Cache。
5. Decode 阶段逐 token 生成，每一步读取过去的 KV Cache 并追加新 token 的 KV。
6. Streaming response 把 token 增量返回给用户，同时观测系统记录 TTFT、TPOT、吞吐、错误率和 GPU 使用率。

下图把这条路径拆成请求平面、调度平面、GPU 执行平面和观测平面。

![LLM 推理请求生命周期](assets/llm_inference_request_lifecycle.svg)

<small>图 1-1：LLM 推理请求生命周期。在线推理的核心不是一次静态 forward，而是让新请求、运行中请求、KV Cache 和 GPU batch 在持续变化的状态中保持可控。</small>

**本章小结**：推理服务的基本单位不是模型 forward，而是随时间变化的 request state。理解后续优化前，必须先把请求、队列、batch、replica 和 KV Cache 看成同一个系统中的状态对象。

## 第 2 章 Prefill、Decode 与 KV Cache

**本章主线**：先区分 prefill 和 decode 的资源特征，再解释 KV Cache 为什么成为长上下文和高并发推理的核心状态，最后说明显存管理为什么会直接决定吞吐和成本。

### 2.1 Prefill 偏计算，Decode 偏显存带宽和调度

Transformer 自回归生成可以分成两个阶段。**Prefill** 处理完整 prompt，计算所有输入 token 的 hidden states，并为每一层 attention 生成 key / value。这个阶段的矩阵乘法较大，通常更容易把 GPU 算力打满。**Decode** 每次只生成一个或少量新 token，它需要读取已有上下文的 KV Cache，计算新 token 的 attention 和 MLP，再采样得到下一个 token。

这两个阶段的瓶颈不同：

| 阶段 | 输入形态 | 主要状态 | 典型瓶颈 |
|---|---|---|---|
| Prefill | 长 prompt，一次处理多个 token | prompt tokens、临时 activation、初始 KV Cache | GEMM 计算、长上下文 attention、短请求混入导致排队 |
| Decode | 每个请求每步新增 token | 已有 KV Cache、采样状态、输出流 | 显存带宽、batch 动态变化、kernel launch / 调度开销 |

如果系统把 prefill 和 decode 混在同一个简单 FIFO batch 中，长 prompt 可能阻塞正在 decode 的短请求，导致用户看到的 token 间隔变大；反过来，如果系统只追求 decode 的低延迟，GPU 又可能在小 batch 下空转。

### 2.2 KV Cache 是推理系统的核心状态

KV Cache 保存每层 attention 中历史 token 的 key 和 value。没有 KV Cache，生成第 $t$ 个 token 时就要重新计算前 $1 \ldots t-1$ 个 token 的 key / value，成本会随输出长度快速膨胀。使用 KV Cache 后，decode step 只需要计算新 token 的 key / value，再读取历史缓存参与 attention。

但 KV Cache 也带来三个系统问题：

- **容量问题**：上下文越长、batch 越大、层数越多，KV Cache 占用越大。
- **碎片问题**：请求长度不同、完成时间不同，缓存空间会出现不规则空洞。
- **迁移问题**：当请求跨副本、跨 GPU 或发生恢复时，KV Cache 很难像普通参数一样随意搬动。

因此，高性能推理引擎往往把 KV Cache 当成一等公民。vLLM 的 PagedAttention 把 KV Cache 管理借鉴为分页式块管理，核心目标就是减少显存碎片并支持更灵活的请求调度。TensorRT-LLM、SGLang 和其他推理栈也都围绕 KV Cache、prefix reuse、offload 或缓存复用提供优化。

**本章小结**：Prefill 和 decode 的资源画像不同，KV Cache 则把两者连接成一个持续增长的状态系统。后续的 batching、并行和生产调度，本质上都要先回答“如何让 KV Cache 占用可控且不浪费 GPU”。

## 第 3 章 Continuous Batching 与 PagedAttention

**本章主线**：先说明静态 batching 为什么不适合在线生成，再解释 continuous batching 如何把新请求插入正在运行的 decode 循环，最后把 PagedAttention 放到显存管理而不是单个 attention 算子的角度理解。

### 3.1 静态 batch 会浪费在线请求的时间

传统 batch 推理通常等待一批请求凑齐后一起执行，然后等这批请求全部完成。LLM 生成长度差异很大，如果系统必须等待最长的请求结束，短请求完成后占用的 batch 槽位会被浪费；如果系统频繁开新 batch，又会牺牲 GPU 利用率。

Continuous batching 的直觉是：decode 是一个循环，每一步之后都会有请求完成，也可能有新请求到达。调度器不必等整批请求结束，而是在每个或若干个 decode step 边界更新 batch 成员，让新请求尽快进入执行，同时释放已完成请求的缓存和槽位。

Orca 把这种思想系统化：它不是把一个请求当成不可切分的任务，而是把生成过程拆成 iteration 级别的调度问题。后续 vLLM、TGI、SGLang 等推理系统都围绕类似的在线 batch 管理能力演进。

### 3.2 PagedAttention 解决的是 KV Cache 管理问题

PagedAttention 容易被误解成“一个更快的 attention kernel”。它真正重要的系统含义是：把每个请求的 KV Cache 切成固定大小的 block，并用逻辑块到物理块的映射管理缓存。这样一来，请求上下文不需要占用一整段连续显存，多个请求也可以更灵活地共享、释放和复用缓存块。

这个设计带来的收益包括：

- 减少不同请求长度造成的显存碎片。
- 更容易支持 continuous batching 下的动态请求加入和退出。
- 支持 prefix caching 等复用策略，让相同前缀的请求共享部分缓存。
- 让系统用更接近操作系统内存管理的方式理解 KV Cache。

对工程读者来说，PagedAttention 的价值不只在论文性能数字，而在它提供了一种思维方式：推理服务的瓶颈不是单个算子孤立变慢，而是请求状态、缓存分配和调度策略相互放大。

**本章小结**：Continuous batching 解决“请求何时进入 GPU”的问题，PagedAttention 解决“请求状态如何占用显存”的问题。二者组合后，LLM serving 才能在高并发下同时提高吞吐和降低排队等待。

## 第 4 章 并行、量化与 speculative decoding

**本章主线**：先把优化手段按资源维度分类，再说明每种手段牺牲什么、换来什么，避免把所有推理优化都混成“更快”。

### 4.1 并行推理解决单副本装不下或跑不快

当模型参数、KV Cache 或请求吞吐超过单 GPU 能力时，推理系统需要并行策略：

| 策略 | 解决的问题 | 代价 |
|---|---|---|
| Tensor Parallel | 单层权重和计算切到多 GPU | 每层通信增加，batch 小时通信占比高 |
| Pipeline Parallel | 不同层放到不同 GPU | 流水气泡和跨阶段调度复杂 |
| Expert Parallel | MoE 模型中按专家分布计算 | 路由、负载均衡和 all-to-all 通信 |
| Replica Parallel | 多份完整模型副本分摊请求 | 参数显存重复，占用更多 GPU |

训练系统中的并行策略不能直接照搬到在线推理。推理更关心单请求尾延迟、batch 动态变化、KV Cache 分布和流式输出，因此并行策略需要和服务调度一起设计。

### 4.2 量化和 speculative decoding 不是同一种优化

**量化**把权重、activation 或 KV Cache 使用更低精度表示，目标是减少显存占用、提高带宽效率或使用更快 kernel。它的风险是数值误差、模型质量下降和硬件 / kernel 支持差异。

**Speculative decoding** 使用一个较小或更便宜的 draft model 先提出多个候选 token，再由目标模型验证。它的目标是减少目标模型 decode step 数量。它的风险是接受率不稳定、系统复杂度增加，以及 draft model 本身也要占用计算资源。

二者可以组合，但优化对象不同：量化主要改变每一步的计算和内存成本；speculative decoding 改变需要执行多少步目标模型。

**本章小结**：推理优化要按瓶颈分类。模型装不下时看并行和量化；显存碎片严重时看 KV Cache 管理；decode 步数成为瓶颈时看 speculative decoding；线上请求波动大时优先看调度和限流。

## 第 5 章 从单模型引擎到生产服务

**本章主线**：先区分推理引擎和生产服务，再整理线上系统必须观测的指标，最后给出一个选型框架。

### 5.1 推理引擎只是服务栈的一层

vLLM、TensorRT-LLM、SGLang 和 TGI 主要解决模型执行与请求调度问题，但生产服务还需要更多层：

- API 层：认证、限流、配额、OpenAI-compatible schema、streaming 协议。
- 路由层：多模型路由、多副本负载均衡、灰度发布和回滚。
- 执行层：推理引擎、GPU kernel、KV Cache、batching 和并行策略。
- 平台层：Kubernetes / Ray Serve / KServe、GPU 资源池、弹性伸缩和故障恢复。
- 观测层：latency、throughput、error、GPU utilization、queue length、cache usage 和 cost。

如果只评估单机 benchmark，很容易忽略线上系统中真正痛的地方：请求分布变化、长上下文突发、某个租户打满 KV Cache、某个副本尾延迟抬高、回滚后缓存失效、GPU 利用率高但用户仍然等待。

### 5.2 指标要按用户体验和系统资源同时设计

推理服务常见指标可以分成两组：

| 指标 | 关注点 | 典型解释 |
|---|---|---|
| TTFT | 首 token 延迟 | 用户多久看到模型开始回答，受排队和 prefill 影响大 |
| TPOT | 每 token 时间 | 流式输出是否平滑，受 decode 调度和 batch 影响大 |
| E2E latency | 完整响应延迟 | 受输出长度和系统负载影响大 |
| Throughput | tokens/s 或 requests/s | GPU 使用效率和系统容量 |
| KV Cache usage | 显存状态 | 长上下文、并发和碎片化压力 |
| Queue length | 排队状态 | 是否需要限流、扩容或分级服务 |
| Cost per token | 成本 | 选型、并行、量化和容量规划的共同结果 |

选型时可以用一个简单判断：如果团队正在做模型实验，先用易集成的 serving 框架；如果追求通用高吞吐，重点评估 vLLM / SGLang；如果部署深度绑定 NVIDIA GPU、需要 engine 化和极致 kernel 优化，评估 TensorRT-LLM；如果已有 Hugging Face 生态集成，TGI 仍可作为参考，但需要注意其维护状态和后续迁移路径。

**本章小结**：生产推理服务不是单个模型引擎。它需要 API、路由、调度、GPU 执行、观测和治理共同工作。最终目标不是某个 benchmark 最快，而是在目标成本下稳定满足用户体验指标。

## 第 6 章 学习路线与参考资料

**本章主线**：把前面的系统地图转成后续学习顺序，并列出可继续深入的代表性资料。

### 6.1 推荐学习顺序

1. 先读 vLLM / PagedAttention，理解 KV Cache 管理为什么是系统问题。
2. 再读 Orca，理解 iteration-level scheduling 和 continuous batching。
3. 接着看 TensorRT-LLM，理解高性能 GPU runtime、kernel、quantization 和 engine 化部署。
4. 然后看 SGLang，理解结构化生成、RadixAttention、prefix / cache reuse 和服务运行时如何结合。
5. 最后结合 Ray Serve、Kubernetes、KServe 或自研平台，看多模型、多副本、多租户、灰度和观测如何落地。

## 参考资料

- vLLM Team, [vLLM Documentation](https://docs.vllm.ai/en/latest/)
- Woosuk Kwon et al., [Efficient Memory Management for Large Language Model Serving with PagedAttention](https://arxiv.org/abs/2309.06180)
- Gyeong-In Yu et al., [Orca: A Distributed Serving System for Transformer-Based Generative Models](https://www.usenix.org/conference/osdi22/presentation/yu)
- NVIDIA, [TensorRT-LLM Documentation](https://docs.nvidia.com/tensorrt-llm/)
- SGLang Team, [SGLang Documentation](https://docs.sglang.ai/)
- Lianmin Zheng et al., [SGLang: Efficient Execution of Structured Language Model Programs](https://arxiv.org/abs/2312.07104)
- Hugging Face, [Text Generation Inference Documentation](https://huggingface.co/docs/text-generation-inference/)

**全文小结**：推理服务化的学习主线应从请求状态和 KV Cache 出发，而不是从某个工具的命令行参数出发。工具会迭代，但 prefill / decode、batching、cache、并行、观测和成本之间的关系会长期存在。
