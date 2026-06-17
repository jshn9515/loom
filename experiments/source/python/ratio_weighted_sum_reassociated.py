import loom


@loom.entry
def ratio_weighted_sum_reassociated(scale: float, epsilon: float, x: loom.Tensor1, y: loom.Tensor1) -> float:
    def contribution(xi: float, yi: float) -> float:
        scaled = xi * scale
        return scaled / (epsilon + yi)

    return loom.Tensor1.reduce_sum(loom.Tensor1.map2(contribution, x, y))
