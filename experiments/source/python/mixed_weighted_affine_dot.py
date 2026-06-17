import loom


@loom.entry
def mixed_weighted_affine_dot(weight: float, scale: float, bias: float, x: loom.Tensor1, y: loom.Tensor1) -> float:
    def contribution(xi: float, yi: float) -> float:
        transformed = (scale * xi) + bias
        return weight * transformed * yi

    return loom.Tensor1.reduce_sum(loom.Tensor1.map2(contribution, x, y))
