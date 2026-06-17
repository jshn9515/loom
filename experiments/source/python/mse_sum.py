import loom


@loom.entry
def mse_sum(x: loom.Tensor1, y: loom.Tensor1) -> float:
    def contribution(xi: float, yi: float) -> float:
        d = xi - yi
        return d * d

    return loom.Tensor1.reduce_sum(loom.Tensor1.map2(contribution, x, y))
