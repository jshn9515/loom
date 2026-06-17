#include <loom/loom.hpp>

LOOM_ENTRY loom::Tensor1 bad(loom::Tensor1 x) {
  for (int i = 0; i < 1; ++i) {
  }
  return x;
}
