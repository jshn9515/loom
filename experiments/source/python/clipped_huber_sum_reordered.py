import loom


@loom.entry
def clipped_huber_sum_reordered(delta: float, cap: float, x: loom.Tensor1, y: loom.Tensor1) -> float:
    def contribution(xi: float, yi: float) -> float:
        d = xi - yi
        abs_d = 0.0 - d if d < 0.0 else d
        half_delta = 0.5 * delta
        quadratic = (d * d) * 0.5
        linear = delta * (abs_d - half_delta)
        value = linear if delta < abs_d else quadratic
        return cap if cap < value else value

    return loom.Tensor1.reduce_sum(loom.Tensor1.map2(contribution, x, y))
