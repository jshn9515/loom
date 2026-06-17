import loom


@loom.entry
def weighted_dot_pipeline(weight: float, x: loom.Tensor1, y: loom.Tensor1) -> float:
    def weighted_term(xi: float, yi: float) -> float:
        prod = xi * yi
        return weight * prod

    return loom.Tensor1.reduce_sum(loom.Tensor1.map2(weighted_term, x, y))
