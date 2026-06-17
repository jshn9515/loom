#include <loom/loom.hpp>

LOOM_ENTRY loom::Tensor1 bad(loom::Tensor1 x) {
  auto window = [](float xi) -> float {
    auto value = xi;
    for (int i = 0; i < 1; ++i) {
      value = value + xi;
    }
    return value;
  };
  return loom::Tensor1::map(window, x);
}
