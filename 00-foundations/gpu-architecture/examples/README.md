# CUDA 示例代码

这个目录放 `cuda_intro.md` 配套的最小示例，目标不是替代完整 CUDA 教程，而是把文档里的执行模型、访存模式和异步流水落到可读代码上。

## 示例列表

| 文件 | 说明 | 关注点 |
|---|---|---|
| `vector_add.cu` | 最小向量加法 | Host / Device 内存、kernel launch、grid-stride loop、错误检查 |
| `tiled_matmul.cu` | shared memory tiled GEMM | block tile、shared memory、边界处理、CPU 校验 |
| `stream_pipeline.cu` | 双 stream 分块流水 | pinned memory、async copy、stream、event 计时 |

## 编译方式

如果本机安装了 CUDA Toolkit，可以在本目录运行：

```bash
nvcc -O2 -std=c++17 vector_add.cu -o vector_add
nvcc -O2 -std=c++17 tiled_matmul.cu -o tiled_matmul
nvcc -O2 -std=c++17 stream_pipeline.cu -o stream_pipeline
```

运行：

```bash
./vector_add
./tiled_matmul
./stream_pipeline
```

## Profiling 建议

```bash
nsys profile -t cuda,nvtx -o vector_add_report ./vector_add
ncu --set full ./tiled_matmul
```

如果环境没有 NVIDIA GPU 或 CUDA Toolkit，可以先阅读源码中的注释；这些示例尽量保持单文件、少依赖，方便之后复制到 GPU 节点上实验。
