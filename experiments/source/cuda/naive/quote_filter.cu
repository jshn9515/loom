#include "cuda_common.cuh"

__global__ void quote_filter_kernel(const float* bid, const float* ask, float threshold, float* out,
                                    std::int64_t n) {
  std::int64_t idx = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  std::int64_t stride = static_cast<std::int64_t>(gridDim.x) * blockDim.x;
  for (; idx < n; idx += stride) {
    float spread = ask[idx] - bid[idx];
    out[idx] = spread > threshold ? spread : 0.0f;
  }
}

extern "C" int quote_filter_run(float threshold, const float* bid, const float* ask, float* out, std::int64_t n) {
  int blocks = choose_block_count(n);
  quote_filter_kernel<<<blocks, kBlockSize>>>(bid, ask, threshold, out, n);
  return static_cast<int>(cudaGetLastError());
}
