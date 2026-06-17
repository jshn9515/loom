import loom


@loom.entry
def mixed_book_signal(scale: float, epsilon: float, threshold: float, bid: loom.Tensor1, ask: loom.Tensor1) -> loom.Tensor1:
    def signal(bid_i: float, ask_i: float) -> float:
        depth = bid_i + ask_i
        return scale * ((bid_i - ask_i) / (depth + epsilon)) if depth > threshold else 0.0

    return loom.Tensor1.map2(signal, bid, ask)
