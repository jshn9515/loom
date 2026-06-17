#include <loom/loom.hpp>

LOOM_ENTRY loom::Tensor1 quote_filter(float threshold, loom::Tensor1 bid, loom::Tensor1 ask) {
  auto filter_quote = [=](float bi, float ai) -> float {
    auto spread = ai - bi;
    return spread > threshold ? spread : 0.0f;
  };
  return loom::Tensor1::map2(filter_quote, bid, ask);
}
