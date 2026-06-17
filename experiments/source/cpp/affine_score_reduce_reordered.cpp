#include <loom/loom.hpp>

LOOM_ENTRY float affine_score_reduce_reordered(float scale, float bias, float threshold, loom::Tensor1 x) {
  auto score = [=](float xi) -> float {
    auto shifted = bias + (xi * scale);
    return threshold < shifted ? shifted : 0.0f;
  };
  return loom::Tensor1::reduce_sum(loom::Tensor1::map(score, x));
}
