import loom


@loom.entry
def inventory_penalty_sum(target: float, pos: loom.Tensor1) -> float:
    def penalty(pi: float) -> float:
        d = pi - target
        return d * d

    return loom.Tensor1.reduce_sum(loom.Tensor1.map(penalty, pos))
