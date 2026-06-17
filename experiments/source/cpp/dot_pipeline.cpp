#include <loom/loom.hpp>

LOOM_ENTRY float dot_pipeline(loom::Tensor1 x, loom::Tensor1 y) {
  auto mul = [](float xi, float yi) -> float { return xi * yi; };
  auto reduce = [](loom::Tensor1 t) -> float { return loom::Tensor1::reduce_sum(t); };
  return reduce(loom::Tensor1::map2(mul, x, y));
}
