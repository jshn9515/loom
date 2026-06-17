import loom


@loom.entry
def soft_threshold(threshold: float, x: loom.Tensor1) -> loom.Tensor1:
    def shrink(xi: float) -> float:
        return xi - threshold if xi > threshold else xi + threshold if xi < (-threshold) else 0.0

    return loom.Tensor1.map(shrink, x)
