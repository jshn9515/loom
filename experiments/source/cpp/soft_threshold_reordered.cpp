#include <loom/loom.hpp>

LOOM_ENTRY loom::Tensor1 soft_threshold_reordered(float threshold, loom::Tensor1 x) {
  auto shrink = [=](float xi) -> float {
    return xi < (0.0f - threshold) ? xi + threshold : threshold < xi ? xi - threshold : 0.0f;
  };
  return loom::Tensor1::map(shrink, x);
}
