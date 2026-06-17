#include <loom/loom.hpp>

LOOM_ENTRY float signal_clip_reduce(float scale, float epsilon, float clip, loom::Tensor1 bid, loom::Tensor1 ask) {
  auto contribution = [=](float bid_i, float ask_i) -> float {
    auto base = scale * ((bid_i - ask_i) / (bid_i + ask_i + epsilon));
    return base > clip ? clip : base < -clip ? -clip : base;
  };
  return loom::Tensor1::reduce_sum(loom::Tensor1::map2(contribution, bid, ask));
}
