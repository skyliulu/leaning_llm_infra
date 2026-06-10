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

constexpr int TILE = 16;

// C[M, N] = A[M, K] * B[K, N]
__global__ void tiled_matmul_kernel(const float* A, const float* B, float* C,
                                    int M, int N, int K) {
    __shared__ float tile_a[TILE][TILE];
    __shared__ float tile_b[TILE][TILE];

    const int row = blockIdx.y * TILE + threadIdx.y;
    const int col = blockIdx.x * TILE + threadIdx.x;

    float acc = 0.0f;

    for (int t = 0; t < (K + TILE - 1) / TILE; ++t) {
        const int a_col = t * TILE + threadIdx.x;
        const int b_row = t * TILE + threadIdx.y;

        tile_a[threadIdx.y][threadIdx.x] = (row < M && a_col < K) ? A[row * K + a_col] : 0.0f;
        tile_b[threadIdx.y][threadIdx.x] = (b_row < K && col < N) ? B[b_row * N + col] : 0.0f;

        __syncthreads();

        #pragma unroll
        for (int i = 0; i < TILE; ++i) {
            acc += tile_a[threadIdx.y][i] * tile_b[i][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = acc;
    }
}

float cpu_ref(const std::vector<float>& A, const std::vector<float>& B,
              int M, int N, int K, int row, int col) {
    float acc = 0.0f;
    for (int i = 0; i < K; ++i) {
        acc += A[row * K + i] * B[i * N + col];
    }
    return acc;
}

int main() {
    constexpr int M = 127;
    constexpr int N = 131;
    constexpr int K = 129;

    std::vector<float> h_A(M * K), h_B(K * N), h_C(M * N);
    for (int i = 0; i < M * K; ++i) h_A[i] = static_cast<float>((i % 17) - 8) / 17.0f;
    for (int i = 0; i < K * N; ++i) h_B[i] = static_cast<float>((i % 13) - 6) / 13.0f;

    float *d_A = nullptr, *d_B = nullptr, *d_C = nullptr;
    CUDA_CHECK(cudaMalloc(&d_A, h_A.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_B, h_B.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_C, h_C.size() * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), h_A.size() * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B.data(), h_B.size() * sizeof(float), cudaMemcpyHostToDevice));

    dim3 block(TILE, TILE);
    dim3 grid((N + TILE - 1) / TILE, (M + TILE - 1) / TILE);
    tiled_matmul_kernel<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(h_C.data(), d_C, h_C.size() * sizeof(float), cudaMemcpyDeviceToHost));

    for (int row = 0; row < M; row += 17) {
        for (int col = 0; col < N; col += 19) {
            const float expected = cpu_ref(h_A, h_B, M, N, K, row, col);
            if (std::fabs(h_C[row * N + col] - expected) > 1e-3f) {
                std::cerr << "Mismatch at (" << row << ", " << col << "): got "
                          << h_C[row * N + col] << ", expected " << expected << std::endl;
                return EXIT_FAILURE;
            }
        }
    }

    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));

    std::cout << "tiled_matmul passed: " << M << "x" << K << " times " << K << "x" << N << std::endl;
    return EXIT_SUCCESS;
}
