import loom


@loom.entry
def ratio_weighted_sum(scale: float, epsilon: float, x: loom.Tensor1, y: loom.Tensor1) -> float:
    return loom.Tensor1.reduce_sum(loom.Tensor1.map2(lambda xi, yi: (scale * xi) / (yi + epsilon), x, y))
