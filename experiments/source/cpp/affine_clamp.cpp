#include <loom/loom.hpp>

LOOM_ENTRY loom::Tensor1 affine_clamp(float scale, float bias, float lo, float hi, loom::Tensor1 x) {
  auto clamp = [=](float xi) -> float {
    auto value = (scale * xi) + bias;
    return value < lo ? lo : value > hi ? hi : value;
  };
  return loom::Tensor1::map(clamp, x);
}
