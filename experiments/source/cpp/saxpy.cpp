#include <loom/loom.hpp>

LOOM_ENTRY loom::Tensor1 saxpy(float a, loom::Tensor1 x, loom::Tensor1 y) {
  auto blend = [a](float xi, float yi) -> float { return (a * xi) + yi; };
  return loom::Tensor1::map2(blend, x, y);
}
