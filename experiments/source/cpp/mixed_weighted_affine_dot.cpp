#include <loom/loom.hpp>

LOOM_ENTRY float mixed_weighted_affine_dot(float weight, float scale, float bias, loom::Tensor1 x, loom::Tensor1 y) {
  auto contribution = [=](float xi, float yi) -> float {
    auto transformed = (scale * xi) + bias;
    return weight * transformed * yi;
  };
  return loom::Tensor1::reduce_sum(loom::Tensor1::map2(contribution, x, y));
}
