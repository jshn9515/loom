#pragma once

#include <cuda_runtime.h>

#include <cstdint>

constexpr int kBlockSize = 256;
constexpr int kMaxBlocks = 4096;
constexpr int kWarpSize = 32;

inline int ceil_div(std::int64_t n, int d) { return static_cast<int>((n + d - 1) / d); }

inline int choose_block_count(std::int64_t n) {
  int blocks = ceil_div(n, kBlockSize);
  return blocks < kMaxBlocks ? blocks : kMaxBlocks;
}

inline int check_cuda(cudaError_t error) { return error == cudaSuccess ? 0 : static_cast<int>(error); }

inline std::size_t workspace_bytes(std::int64_t n) {
  if (n <= 0) {
    return 0;
  }
  return static_cast<std::size_t>(choose_block_count(n)) * sizeof(float) * 2;
}

template <typename T>
inline T* workspace_ptr(void* workspace, std::size_t bytes) {
  return reinterpret_cast<T*>(static_cast<unsigned char*>(workspace) + bytes);
}

__device__ inline float warp_reduce_sum(float value) {
  for (int offset = kWarpSize / 2; offset > 0; offset >>= 1) {
    value += __shfl_down_sync(0xffffffff, value, offset);
  }
  return value;
}

template <int BLOCK_SIZE>
__device__ inline float block_reduce_sum(float value) {
  __shared__ float shared[BLOCK_SIZE / kWarpSize];
  int lane = threadIdx.x & (kWarpSize - 1);
  int warp_id = threadIdx.x / kWarpSize;
  value = warp_reduce_sum(value);
  if (lane == 0) {
    shared[warp_id] = value;
  }
  __syncthreads();
  value = (threadIdx.x < BLOCK_SIZE / kWarpSize) ? shared[lane] : 0.0f;
  if (warp_id == 0) {
    value = warp_reduce_sum(value);
  }
  return value;
}

template <int BLOCK_SIZE>
__global__ void reduce_sum_kernel(const float* __restrict__ input, float* __restrict__ partials,
                                  std::int64_t n) {
  float accum = 0.0f;
  std::int64_t idx = static_cast<std::int64_t>(blockIdx.x) * BLOCK_SIZE + threadIdx.x;
  std::int64_t stride = static_cast<std::int64_t>(gridDim.x) * BLOCK_SIZE;
  for (std::int64_t i = idx; i < n; i += stride) {
    accum += input[i];
  }
  accum = block_reduce_sum<BLOCK_SIZE>(accum);
  if (threadIdx.x == 0) {
    partials[blockIdx.x] = accum;
  }
}

template <int BLOCK_SIZE>
__global__ void reduce_sum_final_kernel(const float* __restrict__ input, float* __restrict__ out,
                                        std::int64_t n) {
  float accum = 0.0f;
  for (std::int64_t i = threadIdx.x; i < n; i += BLOCK_SIZE) {
    accum += input[i];
  }
  accum = block_reduce_sum<BLOCK_SIZE>(accum);
  if (threadIdx.x == 0) {
    out[0] = accum;
  }
}

inline int launch_reduce_sum_from_partials(float* current, std::int64_t current_n, float* out,
                                           void* workspace, std::size_t workspace_size) {
  if (current_n <= 0) {
    return check_cuda(cudaMemset(out, 0, sizeof(float)));
  }
  if (current_n <= kBlockSize) {
    reduce_sum_final_kernel<kBlockSize><<<1, kBlockSize>>>(current, out, current_n);
    cudaError_t status = cudaGetLastError();
    return status == cudaSuccess ? 0 : static_cast<int>(status);
  }
  std::size_t current_bytes = static_cast<std::size_t>(current_n) * sizeof(float);
  std::size_t next_bytes = static_cast<std::size_t>(choose_block_count(current_n)) * sizeof(float);
  if (current_bytes + next_bytes > workspace_size) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  float* buffer_a = workspace_ptr<float>(workspace, 0);
  float* buffer_b = workspace_ptr<float>(workspace, current_bytes);
  float* next = buffer_b;
  while (current_n > 1) {
    int blocks = choose_block_count(current_n);
    reduce_sum_kernel<kBlockSize><<<blocks, kBlockSize>>>(current, next, current_n);
    cudaError_t status = cudaGetLastError();
    if (status != cudaSuccess) {
      return static_cast<int>(status);
    }
    current = next;
    next = (next == buffer_a) ? buffer_b : buffer_a;
    current_n = blocks;
  }
  cudaError_t status = cudaMemcpy(out, current, sizeof(float), cudaMemcpyDeviceToDevice);
  return status == cudaSuccess ? 0 : static_cast<int>(status);
}
