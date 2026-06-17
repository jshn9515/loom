#include <loom/loom.hpp>

LOOM_ENTRY loom::Tensor1 saxpy(float a, loom::Tensor1 x, loom::Tensor1 y) {
  auto blend = [a](float xi, float yi) -> float { return (a * xi) + yi; };
  return loom::Tensor1::map2(blend, x, y);
}

LOOM_ENTRY loom::Tensor1 relu(loom::Tensor1 x) {
  auto activate = [](float xi) -> float { return xi > 0.0f ? xi : 0.0f; };
  return loom::Tensor1::map(activate, x);
}

LOOM_ENTRY float dot(loom::Tensor1 x, loom::Tensor1 y) {
  auto multiply = [](float xi, float yi) -> float { return xi * yi; };
  auto products = loom::Tensor1::map2(multiply, x, y);
  return loom::Tensor1::reduce_sum(products);
}
