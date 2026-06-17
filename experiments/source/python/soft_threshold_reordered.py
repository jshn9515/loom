import loom


@loom.entry
def soft_threshold_reordered(threshold: float, x: loom.Tensor1) -> loom.Tensor1:
    def shrink(xi: float) -> float:
        return xi + threshold if xi < (0.0 - threshold) else xi - threshold if threshold < xi else 0.0

    return loom.Tensor1.map(shrink, x)
