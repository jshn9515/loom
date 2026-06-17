import loom


@loom.entry
def quote_filter(threshold: float, bid: loom.Tensor1, ask: loom.Tensor1) -> loom.Tensor1:
    def filter_quote(bi: float, ai: float) -> float:
        spread = ai - bi
        return spread if spread > threshold else 0.0

    return loom.Tensor1.map2(filter_quote, bid, ask)
