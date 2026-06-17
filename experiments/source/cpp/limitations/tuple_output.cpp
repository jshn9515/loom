#include <loom/loom.hpp>

LOOM_ENTRY auto bad(loom::Tensor1 x) {
  auto sum = loom::Tensor1::reduce_sum(x);
  return loom::tuple(sum, sum);
}
