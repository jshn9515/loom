#include "cuda_common.cuh"

__global__ void affine_clamp_kernel(const float* __restrict__ x, float scale, float bias, float lo, float hi,
                                    float* __restrict__ out, std::int64_t n) {
  std::int64_t idx = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  std::int64_t stride = static_cast<std::int64_t>(gridDim.x) * blockDim.x;
  for (; idx < n; idx += stride) {
    float value = scale * x[idx] + bias;
    if (value < lo) value = lo;
    if (value > hi) value = hi;
    out[idx] = value;
  }
}

extern "C" int affine_clamp_run(float scale, float bias, float lo, float hi, const float* x, float* out,
                                std::int64_t n) {
  int blocks = choose_block_count(n);
  affine_clamp_kernel<<<blocks, kBlockSize>>>(x, scale, bias, lo, hi, out, n);
  return static_cast<int>(cudaGetLastError());
}
