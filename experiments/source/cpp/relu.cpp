#include <loom/loom.hpp>

LOOM_ENTRY loom::Tensor1 relu(loom::Tensor1 x) {
  auto activate = [](float xi) -> float { return xi > 0.0f ? xi : 0.0f; };
  return loom::Tensor1::map(activate, x);
}
