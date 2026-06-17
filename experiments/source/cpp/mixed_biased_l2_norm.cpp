#include <loom/loom.hpp>

LOOM_ENTRY float mixed_biased_l2_norm(float scale, float bias, loom::Tensor1 x) {
  auto square_shifted = [=](float xi) -> float {
    auto value = (scale * xi) + bias;
    return value * value;
  };
  return loom::Tensor1::reduce_sum(loom::Tensor1::map(square_shifted, x));
}
