#include <loom/loom.hpp>

LOOM_ENTRY float clipped_huber_sum_reordered(float delta, float cap, loom::Tensor1 x, loom::Tensor1 y) {
  auto contribution = [=](float xi, float yi) -> float {
    auto d = xi - yi;
    auto abs_d = d < 0.0f ? 0.0f - d : d;
    auto half_delta = 0.5f * delta;
    auto quadratic = (d * d) * 0.5f;
    auto linear = delta * (abs_d - half_delta);
    auto value = delta < abs_d ? linear : quadratic;
    return cap < value ? cap : value;
  };
  return loom::Tensor1::reduce_sum(loom::Tensor1::map2(contribution, x, y));
}
