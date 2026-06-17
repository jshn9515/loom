#include "cuda_common.cuh"

__global__ void scaled_book_signal_kernel(const float* __restrict__ bid, const float* __restrict__ ask, float scale,
                                          float epsilon, float* __restrict__ out, std::int64_t n) {
  std::int64_t idx = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  std::int64_t stride = static_cast<std::int64_t>(gridDim.x) * blockDim.x;
  for (; idx < n; idx += stride) {
    out[idx] = scale * ((bid[idx] - ask[idx]) / (bid[idx] + ask[idx] + epsilon));
  }
}

extern "C" int scaled_book_signal_run(float scale, float epsilon, const float* bid, const float* ask, float* out,
                                      std::int64_t n) {
  int blocks = choose_block_count(n);
  scaled_book_signal_kernel<<<blocks, kBlockSize>>>(bid, ask, scale, epsilon, out, n);
  return static_cast<int>(cudaGetLastError());
}
