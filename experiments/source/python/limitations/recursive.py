import loom


@loom.entry
def bad(x: loom.Tensor1) -> loom.Tensor1:
    def loop(y: loom.Tensor1) -> loom.Tensor1:
        return loop(y)

    return loop(x)
