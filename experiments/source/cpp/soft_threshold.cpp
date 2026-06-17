#include <loom/loom.hpp>

LOOM_ENTRY loom::Tensor1 soft_threshold(float threshold, loom::Tensor1 x) {
  auto shrink = [=](float xi) -> float {
    return xi > threshold ? xi - threshold : xi < (-threshold) ? xi + threshold : 0.0f;
  };
  return loom::Tensor1::map(shrink, x);
}
