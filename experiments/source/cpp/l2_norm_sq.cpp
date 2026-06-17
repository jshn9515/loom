#include <loom/loom.hpp>

LOOM_ENTRY float l2_norm_sq(loom::Tensor1 x) {
  auto square = [](float xi) -> float { return xi * xi; };
  auto squares = loom::Tensor1::map(square, x);
  return loom::Tensor1::reduce_sum(squares);
}
