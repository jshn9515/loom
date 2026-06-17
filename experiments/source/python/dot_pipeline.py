import loom


@loom.entry
def dot_pipeline(x: loom.Tensor1, y: loom.Tensor1) -> float:
    def mul(xi: float, yi: float) -> float:
        return xi * yi

    def reduce(t: loom.Tensor1) -> float:
        return loom.Tensor1.reduce_sum(t)

    return reduce(loom.Tensor1.map2(mul, x, y))
