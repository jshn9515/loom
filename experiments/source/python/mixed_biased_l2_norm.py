import loom


@loom.entry
def mixed_biased_l2_norm(scale: float, bias: float, x: loom.Tensor1) -> float:
    def square_shifted(xi: float) -> float:
        value = (scale * xi) + bias
        return value * value

    return loom.Tensor1.reduce_sum(loom.Tensor1.map(square_shifted, x))
