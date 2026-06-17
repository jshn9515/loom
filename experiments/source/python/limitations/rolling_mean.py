import loom


@loom.entry
def bad(x: loom.Tensor1) -> loom.Tensor1:
    def rolling(value: float) -> float:
        return rolling(value)

    return loom.Tensor1.map(rolling, x)
