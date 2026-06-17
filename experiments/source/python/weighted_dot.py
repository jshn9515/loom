import loom


@loom.entry
def weighted_dot(weight: float, x: loom.Tensor1, y: loom.Tensor1) -> float:
    return loom.Tensor1.reduce_sum(loom.Tensor1.map2(lambda xi, yi: weight * xi * yi, x, y))
