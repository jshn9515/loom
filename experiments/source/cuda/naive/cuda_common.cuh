#pragma once

#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>

constexpr int kBlockSize = 256;

inline int ceil_div(std::int64_t n, int d) { return static_cast<int>((n + d - 1) / d); }

inline int choose_block_count(std::int64_t n) { return ceil_div(n, kBlockSize); }

inline int check_cuda(cudaError_t error) { return error == cudaSuccess ? 0 : static_cast<int>(error); }

__global__ inline void reduce_sum_partials_kernel(const float* input, float* partials, std::int64_t n) {
  __shared__ float shared[kBlockSize];
  std::int64_t idx = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  std::int64_t stride = static_cast<std::int64_t>(gridDim.x) * blockDim.x;
  float accum = 0.0f;
  for (std::int64_t i = idx; i < n; i += stride) {
    accum += input[i];
  }
  shared[threadIdx.x] = accum;
  __syncthreads();
  for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
    if (threadIdx.x < offset) {
      shared[threadIdx.x] += shared[threadIdx.x + offset];
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    partials[blockIdx.x] = shared[0];
  }
}

inline int launch_reduce_sum_from_partials(float* partials, std::int64_t n, float* out) {
  if (n <= 0) {
    if (partials != nullptr) {
      cudaFree(partials);
    }
    return check_cuda(cudaMemset(out, 0, sizeof(float)));
  }

  std::int64_t current_n = n;
  cudaError_t status = cudaSuccess;
  while (current_n > 1) {
    int current_blocks = choose_block_count(current_n);
    float* next_partials = nullptr;
    status = cudaMalloc(&next_partials, sizeof(float) * current_blocks);
    if (status != cudaSuccess) {
      cudaFree(partials);
      return static_cast<int>(status);
    }
    reduce_sum_partials_kernel<<<current_blocks, kBlockSize>>>(partials, next_partials, current_n);
    status = cudaGetLastError();
    cudaFree(partials);
    if (status != cudaSuccess) {
      cudaFree(next_partials);
      return static_cast<int>(status);
    }
    partials = next_partials;
    current_n = current_blocks;
  }

  status = cudaMemcpy(out, partials, sizeof(float), cudaMemcpyDeviceToDevice);
  cudaFree(partials);
  return static_cast<int>(status == cudaSuccess ? cudaGetLastError() : status);
}
