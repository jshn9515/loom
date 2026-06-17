import loom


@loom.entry
def saxpy(a: float, x: loom.Tensor1, y: loom.Tensor1) -> loom.Tensor1:
    return loom.Tensor1.map2(lambda xi, yi: a * xi + yi, x, y)


@loom.entry
def relu(x: loom.Tensor1) -> loom.Tensor1:
    return loom.Tensor1.map(lambda xi: xi if xi > 0.0 else 0.0, x)


@loom.entry
def dot(x: loom.Tensor1, y: loom.Tensor1) -> float:
    products = loom.Tensor1.map2(lambda xi, yi: xi * yi, x, y)
    return loom.Tensor1.reduce_sum(products)
