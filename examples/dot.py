import loom


@loom.entry
def dot(x: loom.Tensor1, y: loom.Tensor1) -> float:
    products = loom.Tensor1.map2(lambda xi, yi: xi * yi, x, y)
    return loom.Tensor1.reduce_sum(products)
