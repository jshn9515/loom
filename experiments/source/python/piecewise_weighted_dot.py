import loom


@loom.entry
def piecewise_weighted_dot(weight_pos: float, weight_neg: float, x: loom.Tensor1, y: loom.Tensor1) -> float:
    def contribution(xi: float, yi: float) -> float:
        prod = xi * yi
        return weight_pos * prod if xi > 0.0 else weight_neg * prod

    return loom.Tensor1.reduce_sum(loom.Tensor1.map2(contribution, x, y))
