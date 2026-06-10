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

__global__ void vector_add_kernel(const float* a, const float* b, float* c, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    // grid-stride loop lets the same kernel handle arbitrary input size.
    for (int i = tid; i < n; i += stride) {
        c[i] = a[i] + b[i];
    }
}

int main() {
    constexpr int n = 1 << 20;
    constexpr int threads_per_block = 256;
    constexpr int blocks = 256;
    const size_t bytes = n * sizeof(float);

    std::vector<float> h_a(n), h_b(n), h_c(n);
    for (int i = 0; i < n; ++i) {
        h_a[i] = static_cast<float>(i) * 0.5f;
        h_b[i] = static_cast<float>(i) * 2.0f;
    }

    float *d_a = nullptr, *d_b = nullptr, *d_c = nullptr;
    CUDA_CHECK(cudaMalloc(&d_a, bytes));
    CUDA_CHECK(cudaMalloc(&d_b, bytes));
    CUDA_CHECK(cudaMalloc(&d_c, bytes));

    CUDA_CHECK(cudaMemcpy(d_a, h_a.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b.data(), bytes, cudaMemcpyHostToDevice));

    vector_add_kernel<<<blocks, threads_per_block>>>(d_a, d_b, d_c, n);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(h_c.data(), d_c, bytes, cudaMemcpyDeviceToHost));

    for (int i = 0; i < n; ++i) {
        const float expected = h_a[i] + h_b[i];
        if (std::fabs(h_c[i] - expected) > 1e-5f) {
            std::cerr << "Mismatch at " << i << ": got " << h_c[i]
                      << ", expected " << expected << std::endl;
            return EXIT_FAILURE;
        }
    }

    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_c));

    std::cout << "vector_add passed, n=" << n << std::endl;
    return EXIT_SUCCESS;
}
