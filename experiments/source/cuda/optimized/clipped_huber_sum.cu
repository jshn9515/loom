#include "cuda_common.cuh"

template <int BLOCK_SIZE>
__global__ void clipped_huber_sum_partial_kernel(const float* __restrict__ x, const float* __restrict__ y,
                                                 float delta, float cap, float* __restrict__ partials,
                                                 std::int64_t n) {
  float accum = 0.0f;
  std::int64_t idx = static_cast<std::int64_t>(blockIdx.x) * BLOCK_SIZE + threadIdx.x;
  std::int64_t stride = static_cast<std::int64_t>(gridDim.x) * BLOCK_SIZE;
  for (std::int64_t i = idx; i < n; i += stride) {
    float d = x[i] - y[i];
    float abs_d = fabsf(d);
    float quadratic = 0.5f * d * d;
    float linear = delta * (abs_d - 0.5f * delta);
    float value = abs_d > delta ? linear : quadratic;
    accum += value > cap ? cap : value;
  }
  accum = block_reduce_sum<BLOCK_SIZE>(accum);
  if (threadIdx.x == 0) {
    partials[blockIdx.x] = accum;
  }
}

extern "C" std::size_t clipped_huber_sum_workspace_size(std::int64_t n) { return workspace_bytes(n); }

extern "C" int clipped_huber_sum_run(float delta, float cap, const float* x, const float* y, float* out,
                                     std::int64_t n, void* workspace, std::size_t workspace_size) {
  if (n <= 0) {
    return check_cuda(cudaMemset(out, 0, sizeof(float)));
  }
  if (workspace_bytes(n) > workspace_size || workspace == nullptr) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  float* partials = workspace_ptr<float>(workspace, 0);
  int blocks = choose_block_count(n);
  clipped_huber_sum_partial_kernel<kBlockSize><<<blocks, kBlockSize>>>(x, y, delta, cap, partials, n);
  cudaError_t status = cudaGetLastError();
  if (status != cudaSuccess) {
    return static_cast<int>(status);
  }
  return launch_reduce_sum_from_partials(partials, blocks, out, workspace, workspace_size);
}
