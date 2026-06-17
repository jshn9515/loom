import loom


@loom.entry
def bad(x: loom.Tensor1) -> loom.Tensor1:
    return loom.Tensor1.map(lambda xi: xi + loom.Tensor1.reduce_sum(x), x)
