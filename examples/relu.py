import loom


@loom.entry
def relu(x: loom.Tensor1) -> loom.Tensor1:
    return loom.Tensor1.map(lambda xi: xi if xi > 0.0 else 0.0, x)
