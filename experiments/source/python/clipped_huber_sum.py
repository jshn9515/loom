import loom


@loom.entry
def clipped_huber_sum(delta: float, cap: float, x: loom.Tensor1, y: loom.Tensor1) -> float:
    def contribution(xi: float, yi: float) -> float:
        d = xi - yi
        abs_d = d if d >= 0.0 else -d
        quadratic = 0.5 * d * d
        linear = delta * (abs_d - (0.5 * delta))
        value = linear if abs_d > delta else quadratic
        return cap if value > cap else value

    return loom.Tensor1.reduce_sum(loom.Tensor1.map2(contribution, x, y))
