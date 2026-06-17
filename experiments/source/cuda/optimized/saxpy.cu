#include "cuda_common.cuh"

__global__ void saxpy_vector_kernel(float a, const float* __restrict__ x, const float* __restrict__ y,
                                    float* __restrict__ out, std::int64_t n) {
  std::int64_t vector_idx = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  std::int64_t vector_stride = static_cast<std::int64_t>(gridDim.x) * blockDim.x;
  std::int64_t vector_count = (n + 3) / 4;
  for (; vector_idx < vector_count; vector_idx += vector_stride) {
    std::int64_t base = vector_idx * 4;
    if (base + 3 < n) {
      float4 xv = reinterpret_cast<const float4*>(x)[vector_idx];
      float4 yv = reinterpret_cast<const float4*>(y)[vector_idx];
      float4 result = {a * xv.x + yv.x, a * xv.y + yv.y, a * xv.z + yv.z, a * xv.w + yv.w};
      reinterpret_cast<float4*>(out)[vector_idx] = result;
    } else {
      for (int lane = 0; lane < 4; ++lane) {
        std::int64_t idx = base + lane;
        if (idx < n) {
          out[idx] = a * x[idx] + y[idx];
        }
      }
    }
  }
}

extern "C" int saxpy_run(float a, const float* x, const float* y, float* out, std::int64_t n) {
  int blocks = choose_block_count(ceil_div(n, 4));
  saxpy_vector_kernel<<<blocks, kBlockSize>>>(a, x, y, out, n);
  return static_cast<int>(cudaGetLastError());
}
