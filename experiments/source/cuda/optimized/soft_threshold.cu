#include "cuda_common.cuh"

__global__ void soft_threshold_kernel(const float* __restrict__ x, float threshold, float* __restrict__ out,
                                      std::int64_t n) {
  std::int64_t idx = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  std::int64_t stride = static_cast<std::int64_t>(gridDim.x) * blockDim.x;
  for (; idx < n; idx += stride) {
    float value = x[idx];
    out[idx] = value > threshold ? value - threshold : (value < -threshold ? value + threshold : 0.0f);
  }
}

extern "C" int soft_threshold_run(float threshold, const float* x, float* out, std::int64_t n) {
  int blocks = choose_block_count(n);
  soft_threshold_kernel<<<blocks, kBlockSize>>>(x, threshold, out, n);
  return static_cast<int>(cudaGetLastError());
}
