#include <loom/loom.hpp>

LOOM_ENTRY loom::Tensor1 saxpy_curried(float a, loom::Tensor1 x, loom::Tensor1 y) {
  auto blend = [](float a, float xi, float yi) -> float { return (a * xi) + yi; };
  return loom::Tensor1::map2(loom::partial(blend, a), x, y);
}
