import loom


@loom.entry
def scaled_book_signal(scale: float, epsilon: float, bid: loom.Tensor1, ask: loom.Tensor1) -> loom.Tensor1:
    return loom.Tensor1.map2(
        lambda bid_i, ask_i: scale * ((bid_i - ask_i) / (bid_i + ask_i + epsilon)),
        bid,
        ask,
    )
