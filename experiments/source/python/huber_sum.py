import loom


@loom.entry
def huber_sum(delta: float, x: loom.Tensor1, y: loom.Tensor1) -> float:
    def contribution(xi: float, yi: float) -> float:
        d = xi - yi
        return (
            delta * (d - (0.5 * delta))
            if d > delta
            else delta * ((-d) - (0.5 * delta))
            if d < (-delta)
            else 0.5 * d * d
        )

    return loom.Tensor1.reduce_sum(loom.Tensor1.map2(contribution, x, y))
