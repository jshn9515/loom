#include <loom/loom.hpp>

LOOM_ENTRY float clipped_huber_sum(float delta, float cap, loom::Tensor1 x, loom::Tensor1 y) {
  auto contribution = [=](float xi, float yi) -> float {
    auto d = xi - yi;
    auto abs_d = d >= 0.0f ? d : -d;
    auto quadratic = 0.5f * d * d;
    auto linear = delta * (abs_d - (0.5f * delta));
    auto value = abs_d > delta ? linear : quadratic;
    return value > cap ? cap : value;
  };
  return loom::Tensor1::reduce_sum(loom::Tensor1::map2(contribution, x, y));
}
