#include <loom/loom.hpp>

float mystery(float value);

LOOM_ENTRY float bad(loom::Tensor1 x) {
  auto value = loom::Tensor1::reduce_sum(x);
  return mystery(value);
}
