#include <loom/loom.hpp>

LOOM_ENTRY float weighted_dot(float weight, loom::Tensor1 x, loom::Tensor1 y) {
  auto contribution = [=](float xi, float yi) -> float { return weight * xi * yi; };
  return loom::Tensor1::reduce_sum(loom::Tensor1::map2(contribution, x, y));
}
