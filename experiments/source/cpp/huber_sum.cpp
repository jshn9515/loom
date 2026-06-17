#include <loom/loom.hpp>

LOOM_ENTRY float huber_sum(float delta, loom::Tensor1 x, loom::Tensor1 y) {
  auto contribution = [=](float xi, float yi) -> float {
    auto d = xi - yi;
    return d > delta ? delta * (d - (0.5f * delta))
         : d < (-delta) ? delta * ((-d) - (0.5f * delta))
         : 0.5f * d * d;
  };
  return loom::Tensor1::reduce_sum(loom::Tensor1::map2(contribution, x, y));
}
