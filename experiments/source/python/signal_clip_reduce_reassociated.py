import loom


@loom.entry
def signal_clip_reduce_reassociated(
    scale: float, epsilon: float, clip: float, bid: loom.Tensor1, ask: loom.Tensor1
) -> float:
    def contribution(bid_i: float, ask_i: float) -> float:
        depth = ask_i + bid_i
        raw = ((bid_i - ask_i) / (depth + epsilon)) * scale
        neg_clip = 0.0 - clip
        return neg_clip if raw < neg_clip else clip if clip < raw else raw

    return loom.Tensor1.reduce_sum(loom.Tensor1.map2(contribution, bid, ask))
