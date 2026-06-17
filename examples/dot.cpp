#include <loom/loom.hpp>

LOOM_ENTRY float dot(loom::Tensor1 x, loom::Tensor1 y) {
  auto multiply = [](float xi, float yi) -> float { return xi * yi; };
  auto products = loom::Tensor1::map2(multiply, x, y);
  return loom::Tensor1::reduce_sum(products);
}
