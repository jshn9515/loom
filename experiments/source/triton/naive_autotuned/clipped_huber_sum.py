import torch
import triton
import triton.language as tl

from common import ELEMENTWISE_CONFIGS, check_tensor, collect_autotune_state, reduce_sum, size_bucket


@triton.autotune(configs=ELEMENTWISE_CONFIGS, key=["size_bucket"])
@triton.jit
def clipped_huber_sum_kernel(out_ptr, n, size_bucket, x_ptr, y_ptr, delta, cap, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    y = tl.load(y_ptr + offs, mask=mask, other=0.0)
    d = x - y
    abs_d = tl.abs(d)
    quadratic = 0.5 * d * d
    linear = delta * (abs_d - (0.5 * delta))
    value = tl.where(abs_d > delta, linear, quadratic)
    tl.store(out_ptr + offs, tl.where(value > cap, cap, value), mask=mask)


def run(delta: float, cap: float, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    check_tensor("x", x)
    check_tensor("y", y)
    if y.numel() != x.numel():
        raise ValueError("all tensor inputs must have the same shape")
    temp = torch.empty_like(x)
    grid = lambda meta: (triton.cdiv(x.numel(), meta["BLOCK_SIZE"]),)
    clipped_huber_sum_kernel[grid](temp, x.numel(), size_bucket(x.numel()), x, y, float(delta), float(cap))
    return reduce_sum(temp)


def autotune_state() -> dict[str, object]:
    return collect_autotune_state({"clipped_huber_sum_kernel": clipped_huber_sum_kernel})
