#include "cuda_common.cuh"

__global__ void mixed_book_signal_kernel(const float* bid, const float* ask, float scale, float epsilon,
                                         float threshold, float* out, std::int64_t n) {
  std::int64_t idx = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  std::int64_t stride = static_cast<std::int64_t>(gridDim.x) * blockDim.x;
  for (std::int64_t i = idx; i < n; i += stride) {
    float depth = bid[i] + ask[i];
    float value = scale * ((bid[i] - ask[i]) / (depth + epsilon));
    out[i] = depth > threshold ? value : 0.0f;
  }
}

extern "C" int mixed_book_signal_run(float scale, float epsilon, float threshold, const float* bid,
                                     const float* ask, float* out, std::int64_t n) {
  if (n <= 0) {
    return 0;
  }
  int blocks = choose_block_count(n);
  mixed_book_signal_kernel<<<blocks, kBlockSize>>>(bid, ask, scale, epsilon, threshold, out, n);
  return check_cuda(cudaGetLastError());
}
