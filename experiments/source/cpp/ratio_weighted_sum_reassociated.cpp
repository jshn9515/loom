#include <loom/loom.hpp>

LOOM_ENTRY float ratio_weighted_sum_reassociated(float scale, float epsilon, loom::Tensor1 x, loom::Tensor1 y) {
  auto contribution = [=](float xi, float yi) -> float {
    auto scaled = xi * scale;
    return scaled / (epsilon + yi);
  };
  return loom::Tensor1::reduce_sum(loom::Tensor1::map2(contribution, x, y));
}
