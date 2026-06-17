#include <loom/loom.hpp>

LOOM_ENTRY float piecewise_weighted_dot_reordered(float weight_pos, float weight_neg, loom::Tensor1 x, loom::Tensor1 y) {
  auto contribution = [=](float xi, float yi) -> float {
    auto prod = yi * xi;
    auto weight = 0.0f < xi ? weight_pos : weight_neg;
    return prod * weight;
  };
  return loom::Tensor1::reduce_sum(loom::Tensor1::map2(contribution, x, y));
}
