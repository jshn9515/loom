#include "cuda_common.cuh"

__global__ void affine_score_reduce_partial_kernel(const float* x, float scale, float bias, float threshold,
                                                   float* partials, std::int64_t n) {
  __shared__ float shared[kBlockSize];
  std::int64_t idx = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  std::int64_t stride = static_cast<std::int64_t>(gridDim.x) * blockDim.x;
  float accum = 0.0f;
  for (std::int64_t i = idx; i < n; i += stride) {
    float value = scale * x[i] + bias;
    accum += value > threshold ? value : 0.0f;
  }
  shared[threadIdx.x] = accum;
  __syncthreads();
  for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
    if (threadIdx.x < offset) {
      shared[threadIdx.x] += shared[threadIdx.x + offset];
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    partials[blockIdx.x] = shared[0];
  }
}

extern "C" int affine_score_reduce_run(float scale, float bias, float threshold, const float* x, float* out,
                                       std::int64_t n) {
  if (n <= 0) {
    return check_cuda(cudaMemset(out, 0, sizeof(float)));
  }
  int blocks = choose_block_count(n);
  float* partials = nullptr;
  cudaError_t status = cudaMalloc(&partials, sizeof(float) * blocks);
  if (status != cudaSuccess) {
    return static_cast<int>(status);
  }
  affine_score_reduce_partial_kernel<<<blocks, kBlockSize>>>(x, scale, bias, threshold, partials, n);
  status = cudaGetLastError();
  if (status != cudaSuccess) {
    cudaFree(partials);
    return static_cast<int>(status);
  }
  return launch_reduce_sum_from_partials(partials, blocks, out);
}
