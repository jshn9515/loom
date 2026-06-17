#include <loom/loom.hpp>

LOOM_ENTRY loom::Tensor1 scaled_book_signal(float scale, float epsilon, loom::Tensor1 bid, loom::Tensor1 ask) {
  auto signal = [=](float bid_i, float ask_i) -> float {
    return scale * ((bid_i - ask_i) / (bid_i + ask_i + epsilon));
  };
  return loom::Tensor1::map2(signal, bid, ask);
}
