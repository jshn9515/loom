import loom


@loom.entry
def relu_tupled(x: loom.Tensor1) -> loom.Tensor1:
    thresholds = (0.0, 1.0)
    zero, _one = thresholds

    def clip(xi: float) -> float:
        return xi if xi > zero else zero

    return loom.Tensor1.map(clip, x)
