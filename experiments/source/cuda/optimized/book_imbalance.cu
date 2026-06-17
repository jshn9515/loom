#include "cuda_common.cuh"

__global__ void book_imbalance_kernel(const float* __restrict__ bid, const float* __restrict__ ask,
                                      float epsilon, float* __restrict__ out, std::int64_t n) {
  std::int64_t idx = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  std::int64_t stride = static_cast<std::int64_t>(gridDim.x) * blockDim.x;
  for (; idx < n; idx += stride) {
    out[idx] = (bid[idx] - ask[idx]) / (bid[idx] + ask[idx] + epsilon);
  }
}

extern "C" int book_imbalance_run(float epsilon, const float* bid, const float* ask, float* out,
                                  std::int64_t n) {
  int blocks = choose_block_count(n);
  book_imbalance_kernel<<<blocks, kBlockSize>>>(bid, ask, epsilon, out, n);
  return static_cast<int>(cudaGetLastError());
}
