#include <loom/loom.hpp>

LOOM_ENTRY loom::Tensor1 relu_guard_reordered(loom::Tensor1 x) {
  auto activate = [](float xi) -> float { return 0.0f < xi ? xi : 0.0f; };
  return loom::Tensor1::map(activate, x);
}
