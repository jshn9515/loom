import loom


@loom.entry
def l2_norm_sq(x: loom.Tensor1) -> float:
    squares = loom.Tensor1.map(lambda xi: xi * xi, x)
    return loom.Tensor1.reduce_sum(squares)
