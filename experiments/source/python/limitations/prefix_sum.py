import loom


@loom.entry
def bad(x: loom.Tensor1) -> loom.Tensor1:
    def prefix(value: float) -> float:
        return prefix(value)

    return loom.Tensor1.map(prefix, x)
