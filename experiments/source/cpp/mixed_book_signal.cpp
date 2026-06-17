#include <loom/loom.hpp>

LOOM_ENTRY loom::Tensor1 mixed_book_signal(float scale, float epsilon, float threshold, loom::Tensor1 bid, loom::Tensor1 ask) {
  auto signal = [=](float bid_i, float ask_i) -> float {
    auto depth = bid_i + ask_i;
    return depth > threshold ? scale * ((bid_i - ask_i) / (depth + epsilon)) : 0.0f;
  };
  return loom::Tensor1::map2(signal, bid, ask);
}
