#include <loom/loom.hpp>

LOOM_ENTRY loom::Tensor1 bad(loom::Tensor1 x) {
  auto captures_tensor = [x](float xi) -> float {
    return xi + loom::Tensor1::reduce_sum(x);
  };
  return loom::Tensor1::map(captures_tensor, x);
}
