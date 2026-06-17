#include "cuda_common.cuh"

template <int BLOCK_SIZE>
__global__ void affine_score_reduce_partial_kernel(const float* __restrict__ x, float scale, float bias,
                                                   float threshold, float* __restrict__ partials,
                                                   std::int64_t n) {
  float accum = 0.0f;
  std::int64_t idx = static_cast<std::int64_t>(blockIdx.x) * BLOCK_SIZE + threadIdx.x;
  std::int64_t stride = static_cast<std::int64_t>(gridDim.x) * BLOCK_SIZE;
  for (std::int64_t i = idx; i < n; i += stride) {
    float value = scale * x[i] + bias;
    accum += value > threshold ? value : 0.0f;
  }
  accum = block_reduce_sum<BLOCK_SIZE>(accum);
  if (threadIdx.x == 0) {
    partials[blockIdx.x] = accum;
  }
}

extern "C" std::size_t affine_score_reduce_workspace_size(std::int64_t n) { return workspace_bytes(n); }

extern "C" int affine_score_reduce_run(float scale, float bias, float threshold, const float* x, float* out,
                                       std::int64_t n, void* workspace, std::size_t workspace_size) {
  if (n <= 0) {
    return check_cuda(cudaMemset(out, 0, sizeof(float)));
  }
  if (workspace_bytes(n) > workspace_size || workspace == nullptr) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  float* partials = workspace_ptr<float>(workspace, 0);
  int blocks = choose_block_count(n);
  affine_score_reduce_partial_kernel<kBlockSize><<<blocks, kBlockSize>>>(x, scale, bias, threshold, partials, n);
  cudaError_t status = cudaGetLastError();
  if (status != cudaSuccess) {
    return static_cast<int>(status);
  }
  return launch_reduce_sum_from_partials(partials, blocks, out, workspace, workspace_size);
}
