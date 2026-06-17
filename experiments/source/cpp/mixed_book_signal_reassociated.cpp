#include <loom/loom.hpp>

LOOM_ENTRY loom::Tensor1 mixed_book_signal_reassociated(float scale, float epsilon, float threshold, loom::Tensor1 bid, loom::Tensor1 ask) {
  auto signal = [=](float bid_i, float ask_i) -> float {
    auto depth = ask_i + bid_i;
    auto spread = bid_i - ask_i;
    auto normalized = spread / (epsilon + depth);
    return threshold < depth ? normalized * scale : 0.0f;
  };
  return loom::Tensor1::map2(signal, bid, ask);
}
