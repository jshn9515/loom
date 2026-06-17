import loom


@loom.entry
def relu_guard_reordered(x: loom.Tensor1) -> loom.Tensor1:
    def activate(xi: float) -> float:
        return xi if 0.0 < xi else 0.0

    return loom.Tensor1.map(activate, x)
