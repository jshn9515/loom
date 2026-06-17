import loom


@loom.entry
def saxpy_curried(a: float, x: loom.Tensor1, y: loom.Tensor1) -> loom.Tensor1:
    def blend(a: float, xi: float, yi: float) -> float:
        return a * xi + yi

    return loom.Tensor1.map2(blend(a), x, y)
