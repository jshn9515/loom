#include <loom/loom.hpp>

LOOM_ENTRY float inventory_penalty_sum(float target, loom::Tensor1 pos) {
  auto penalty = [=](float pi) -> float {
    auto d = pi - target;
    return d * d;
  };
  return loom::Tensor1::reduce_sum(loom::Tensor1::map(penalty, pos));
}
