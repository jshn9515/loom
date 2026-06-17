import loom


@loom.entry
def mixed_book_signal_reassociated(
    scale: float, epsilon: float, threshold: float, bid: loom.Tensor1, ask: loom.Tensor1
) -> loom.Tensor1:
    def signal(bid_i: float, ask_i: float) -> float:
        depth = ask_i + bid_i
        spread = bid_i - ask_i
        normalized = spread / (epsilon + depth)
        return normalized * scale if threshold < depth else 0.0

    return loom.Tensor1.map2(signal, bid, ask)
