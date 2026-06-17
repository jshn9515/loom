#include <loom/loom.hpp>

LOOM_ENTRY float signal_clip_reduce_reassociated(float scale, float epsilon, float clip, loom::Tensor1 bid, loom::Tensor1 ask) {
  auto contribution = [=](float bid_i, float ask_i) -> float {
    auto depth = ask_i + bid_i;
    auto raw = ((bid_i - ask_i) / (depth + epsilon)) * scale;
    auto neg_clip = 0.0f - clip;
    return raw < neg_clip ? neg_clip : clip < raw ? clip : raw;
  };
  return loom::Tensor1::reduce_sum(loom::Tensor1::map2(contribution, bid, ask));
}
