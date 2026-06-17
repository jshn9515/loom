import loom


@loom.entry
def affine_score_reduce(scale: float, bias: float, threshold: float, x: loom.Tensor1) -> float:
    def score(xi: float) -> float:
        value = (scale * xi) + bias
        return value if value > threshold else 0.0

    return loom.Tensor1.reduce_sum(loom.Tensor1.map(score, x))
