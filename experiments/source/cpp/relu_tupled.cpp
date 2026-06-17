#include <loom/loom.hpp>

LOOM_ENTRY loom::Tensor1 relu_tupled(loom::Tensor1 x) {
  auto thresholds = loom::tuple(0.0f, 1.0f);
  auto [zero, _one] = thresholds;
  auto clip = [zero](float xi) -> float { return xi > zero ? xi : zero; };
  return loom::Tensor1::map(clip, x);
}
