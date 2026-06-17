import loom


@loom.entry
def piecewise_weighted_dot_reordered(weight_pos: float, weight_neg: float, x: loom.Tensor1, y: loom.Tensor1) -> float:
    def contribution(xi: float, yi: float) -> float:
        prod = yi * xi
        weight = weight_pos if 0.0 < xi else weight_neg
        return prod * weight

    return loom.Tensor1.reduce_sum(loom.Tensor1.map2(contribution, x, y))
