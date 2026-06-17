import loom


@loom.entry
def book_imbalance(epsilon: float, bid: loom.Tensor1, ask: loom.Tensor1) -> loom.Tensor1:
    return loom.Tensor1.map2(lambda bi, ai: (bi - ai) / (bi + ai + epsilon), bid, ask)
