#include "cuda_common.cuh"

extern "C" __global__ void saxpy_kernel(float a, const float* x, const float* y, float* out,
                                         std::int64_t n) {
  std::int64_t idx = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  std::int64_t stride = static_cast<std::int64_t>(gridDim.x) * blockDim.x;
  for (; idx < n; idx += stride) {
    out[idx] = a * x[idx] + y[idx];
  }
}

extern "C" int saxpy_run(float a, const float* x, const float* y, float* out, std::int64_t n) {
  int blocks = choose_block_count(n);
  saxpy_kernel<<<blocks, kBlockSize>>>(a, x, y, out, n);
  return static_cast<int>(cudaGetLastError());
}
