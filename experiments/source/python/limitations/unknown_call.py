import loom


def bad_scalar(x: float) -> float:
    return x + 1.0


@loom.entry
def bad(x: loom.Tensor1) -> loom.Tensor1:
    return loom.Tensor1.map(lambda xi: bad_scalar(xi), x)
