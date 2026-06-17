#include <loom/loom.hpp>

LOOM_ENTRY float weighted_dot_pipeline(float weight, loom::Tensor1 x, loom::Tensor1 y) {
  auto weighted_term = [=](float xi, float yi) -> float {
    auto prod = xi * yi;
    return weight * prod;
  };
  return loom::Tensor1::reduce_sum(loom::Tensor1::map2(weighted_term, x, y));
}
