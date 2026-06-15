# 异构分布式计算系统

这个专题整理 AI Infra 中更贴近训练、推理、数据处理和 RL rollout 的分布式计算基础。这里的重点不是从 Paxos / Raft 开始写传统分布式系统教材，而是先回答一个更工程化的问题：当一个 workload 同时依赖 CPU、GPU、内存、对象存储、网络和调度器时，系统如何表达任务、状态、资源约束和失败恢复。

Ray 是本专题的主要贯穿案例。它的 task / actor / object store / resource model 很适合连接 CPU 数据处理、GPU 训练与推理、Ray Serve 在线服务、Ray Train 分布式训练、Ray Data 数据流水线、KubeRay 集群部署，以及后续 RL / RLHF rollout workload。

## 当前笔记

- [面向 AI Infra 的异构分布式计算系统综述](./heterogeneous_distributed_computing_overview.md)

## 阅读重点

建议按五条线阅读：

1. 先理解异构资源：CPU、GPU、内存、对象存储、网络和调度器分别承担什么职责。
2. 再理解 Ray Core：task、actor、object reference、placement group、CPU/GPU/custom resource。
3. 接着看 CPU-GPU pipeline：数据读取、预处理、batch inference、训练 step 和 rollout 如何串成流水。
4. 然后进入生产部署：KubeRay、autoscaling、故障恢复、资源隔离和观测。
5. 最后把它连接到 LLM Infra / RL Infra：推理服务、分布式训练、数据处理、环境模拟和后训练系统。

## 后续可扩展文章

- `ray_core_task_actor_and_resource_model.md`：系统解释 Ray 的任务、Actor、对象引用和资源调度模型。
- `cpu_gpu_pipeline_and_ray_data.md`：围绕 Ray Data、batch inference 和训练数据流水展开。
- `ray_train_serve_and_rl_workloads.md`：比较 Ray Train、Ray Serve、RLlib / verl 一类 workload 的系统形态。
- `kuberay_scheduling_autoscaling_and_observability.md`：整理 KubeRay 在 Kubernetes 上的部署、弹性和观测问题。

## 资源组织

- `assets/`：异构计算流水、Ray 运行时抽象、KubeRay 部署结构等图。
- 正文引用优先使用 Ray 官方文档、论文、公开项目文档和云原生官方资料。
