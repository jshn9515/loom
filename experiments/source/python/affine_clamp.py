import loom


@loom.entry
def affine_clamp(scale: float, bias: float, lo: float, hi: float, x: loom.Tensor1) -> loom.Tensor1:
    def clamp(xi: float) -> float:
        value = (scale * xi) + bias
        return lo if value < lo else hi if value > hi else value

    return loom.Tensor1.map(clamp, x)
