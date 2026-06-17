import loom


@loom.entry
def bad(x: loom.Tensor1) -> tuple[loom.Tensor1, loom.Tensor1]:
    return (x, x)
