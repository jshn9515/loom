#include <loom/loom.hpp>

LOOM_ENTRY float piecewise_weighted_dot(float weight_pos, float weight_neg, loom::Tensor1 x, loom::Tensor1 y) {
  auto contribution = [=](float xi, float yi) -> float {
    auto prod = xi * yi;
    return xi > 0.0f ? weight_pos * prod : weight_neg * prod;
  };
  return loom::Tensor1::reduce_sum(loom::Tensor1::map2(contribution, x, y));
}
