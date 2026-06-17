import torch
import triton
import triton.language as tl

from common import BLOCK_SIZE, NUM_WARPS, check_tensor, reduce_sum


@triton.jit
def huber_kernel(out_ptr, n, x_ptr, y_ptr, delta, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    y = tl.load(y_ptr + offs, mask=mask, other=0.0)
    d = x - y
    abs_d = tl.where(d > 0.0, d, -d)
    quadratic = 0.5 * d * d
    linear = delta * (abs_d - (0.5 * delta))
    tl.store(out_ptr + offs, tl.where(abs_d > delta, linear, quadratic), mask=mask)


def run(delta: float, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    check_tensor("x", x)
    check_tensor("y", y)
    if y.numel() != x.numel():
        raise ValueError("all tensor inputs must have the same shape")
    temp = torch.empty_like(x)
    grid = (triton.cdiv(x.numel(), BLOCK_SIZE),)
    huber_kernel[grid](temp, x.numel(), x, y, float(delta), BLOCK_SIZE=BLOCK_SIZE, num_warps=NUM_WARPS)
    return reduce_sum(temp)
