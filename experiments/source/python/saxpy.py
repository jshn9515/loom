import loom


@loom.entry
def saxpy(a: float, x: loom.Tensor1, y: loom.Tensor1) -> loom.Tensor1:
    return loom.Tensor1.map2(lambda xi, yi: a * xi + yi, x, y)
