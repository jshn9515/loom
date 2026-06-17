#include <loom/loom.hpp>

LOOM_ENTRY float mse_sum(loom::Tensor1 x, loom::Tensor1 y) {
  auto contribution = [](float xi, float yi) -> float {
    auto d = xi - yi;
    return d * d;
  };
  return loom::Tensor1::reduce_sum(loom::Tensor1::map2(contribution, x, y));
}
