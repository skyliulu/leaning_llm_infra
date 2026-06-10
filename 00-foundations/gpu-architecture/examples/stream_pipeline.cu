#include <cuda_runtime.h>

#include <cmath>
#include <iostream>
#include <vector>

#define CUDA_CHECK(call)                                                        \
    do {                                                                       \
        cudaError_t err__ = (call);                                             \
        if (err__ != cudaSuccess) {                                             \
            std::cerr << "CUDA error: " << cudaGetErrorString(err__)           \
                      << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
            std::exit(EXIT_FAILURE);                                            \
        }                                                                      \
    } while (0)

__global__ void saxpy_kernel(const float* x, float* y, float alpha, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = tid; i < n; i += stride) {
        y[i] = alpha * x[i] + y[i];
    }
}

int main() {
    constexpr int n = 1 << 22;
    constexpr int num_streams = 2;
    constexpr int chunk = n / num_streams;
    constexpr int threads = 256;
    constexpr int blocks = 256;
    constexpr float alpha = 2.5f;
    const size_t bytes = n * sizeof(float);

    float *h_x = nullptr, *h_y = nullptr;
    CUDA_CHECK(cudaMallocHost(&h_x, bytes));  // pinned memory enables true async H2D/D2H copies.
    CUDA_CHECK(cudaMallocHost(&h_y, bytes));

    for (int i = 0; i < n; ++i) {
        h_x[i] = static_cast<float>(i % 100) / 100.0f;
        h_y[i] = 1.0f;
    }

    float *d_x = nullptr, *d_y = nullptr;
    CUDA_CHECK(cudaMalloc(&d_x, bytes));
    CUDA_CHECK(cudaMalloc(&d_y, bytes));

    cudaStream_t streams[num_streams];
    for (int i = 0; i < num_streams; ++i) {
        CUDA_CHECK(cudaStreamCreate(&streams[i]));
    }

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));

    for (int s = 0; s < num_streams; ++s) {
        const int offset = s * chunk;
        const int this_chunk = (s == num_streams - 1) ? (n - offset) : chunk;
        const size_t chunk_bytes = this_chunk * sizeof(float);

        CUDA_CHECK(cudaMemcpyAsync(d_x + offset, h_x + offset, chunk_bytes,
                                   cudaMemcpyHostToDevice, streams[s]));
        CUDA_CHECK(cudaMemcpyAsync(d_y + offset, h_y + offset, chunk_bytes,
                                   cudaMemcpyHostToDevice, streams[s]));
        saxpy_kernel<<<blocks, threads, 0, streams[s]>>>(d_x + offset, d_y + offset, alpha, this_chunk);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaMemcpyAsync(h_y + offset, d_y + offset, chunk_bytes,
                                   cudaMemcpyDeviceToHost, streams[s]));
    }

    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float elapsed_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));

    for (int i = 0; i < n; i += 4099) {
        const float expected = alpha * h_x[i] + 1.0f;
        if (std::fabs(h_y[i] - expected) > 1e-5f) {
            std::cerr << "Mismatch at " << i << ": got " << h_y[i]
                      << ", expected " << expected << std::endl;
            return EXIT_FAILURE;
        }
    }

    for (int i = 0; i < num_streams; ++i) CUDA_CHECK(cudaStreamDestroy(streams[i]));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_x));
    CUDA_CHECK(cudaFree(d_y));
    CUDA_CHECK(cudaFreeHost(h_x));
    CUDA_CHECK(cudaFreeHost(h_y));

    std::cout << "stream_pipeline passed, elapsed_ms=" << elapsed_ms << std::endl;
    return EXIT_SUCCESS;
}
