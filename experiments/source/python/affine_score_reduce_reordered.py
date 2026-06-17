import loom


@loom.entry
def affine_score_reduce_reordered(scale: float, bias: float, threshold: float, x: loom.Tensor1) -> float:
    def score(xi: float) -> float:
        shifted = bias + (xi * scale)
        return shifted if threshold < shifted else 0.0

    return loom.Tensor1.reduce_sum(loom.Tensor1.map(score, x))
