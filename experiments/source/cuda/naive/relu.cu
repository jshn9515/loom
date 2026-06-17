#include "cuda_common.cuh"

extern "C" __global__ void relu_kernel(const float* x, float* out, std::int64_t n) {
  std::int64_t idx = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  std::int64_t stride = static_cast<std::int64_t>(gridDim.x) * blockDim.x;
  for (; idx < n; idx += stride) {
    float value = x[idx];
    out[idx] = value > 0.0f ? value : 0.0f;
  }
}

extern "C" int relu_run(const float* x, float* out, std::int64_t n) {
  int blocks = choose_block_count(n);
  relu_kernel<<<blocks, kBlockSize>>>(x, out, n);
  return static_cast<int>(cudaGetLastError());
}
