#include <loom/loom.hpp>

LOOM_ENTRY loom::Tensor1 book_imbalance(float epsilon, loom::Tensor1 bid, loom::Tensor1 ask) {
  auto imbalance = [=](float bi, float ai) -> float {
    return (bi - ai) / (bi + ai + epsilon);
  };
  return loom::Tensor1::map2(imbalance, bid, ask);
}
