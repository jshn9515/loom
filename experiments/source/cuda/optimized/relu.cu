#include "cuda_common.cuh"

__global__ void relu_vector_kernel(const float* __restrict__ x, float* __restrict__ out,
                                   std::int64_t n) {
  std::int64_t vector_idx = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  std::int64_t vector_stride = static_cast<std::int64_t>(gridDim.x) * blockDim.x;
  std::int64_t vector_count = (n + 3) / 4;
  for (; vector_idx < vector_count; vector_idx += vector_stride) {
    std::int64_t base = vector_idx * 4;
    if (base + 3 < n) {
      float4 xv = reinterpret_cast<const float4*>(x)[vector_idx];
      float4 result = {
          xv.x > 0.0f ? xv.x : 0.0f,
          xv.y > 0.0f ? xv.y : 0.0f,
          xv.z > 0.0f ? xv.z : 0.0f,
          xv.w > 0.0f ? xv.w : 0.0f,
      };
      reinterpret_cast<float4*>(out)[vector_idx] = result;
    } else {
      for (int lane = 0; lane < 4; ++lane) {
        std::int64_t idx = base + lane;
        if (idx < n) {
          float value = x[idx];
          out[idx] = value > 0.0f ? value : 0.0f;
        }
      }
    }
  }
}

extern "C" int relu_run(const float* x, float* out, std::int64_t n) {
  int blocks = choose_block_count(ceil_div(n, 4));
  relu_vector_kernel<<<blocks, kBlockSize>>>(x, out, n);
  return static_cast<int>(cudaGetLastError());
}
