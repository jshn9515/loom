import loom


@loom.entry
def signal_clip_reduce(scale: float, epsilon: float, clip: float, bid: loom.Tensor1, ask: loom.Tensor1) -> float:
    def contribution(bid_i: float, ask_i: float) -> float:
        base = scale * ((bid_i - ask_i) / (bid_i + ask_i + epsilon))
        return clip if base > clip else -clip if base < -clip else base

    return loom.Tensor1.reduce_sum(loom.Tensor1.map2(contribution, bid, ask))
