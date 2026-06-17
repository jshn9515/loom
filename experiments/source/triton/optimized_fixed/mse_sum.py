import torch
import triton
import triton.language as tl

from common import BLOCK_SIZE, NUM_WARPS, REDUCE_BLOCK_SIZE, check_tensor, reduce_sum


@triton.jit
def mse_sum_kernel(partial_ptr, n, x_ptr, y_ptr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    y = tl.load(y_ptr + offs, mask=mask, other=0.0)
    d = x - y
    partial = tl.sum(d * d, axis=0)
    tl.store(partial_ptr + pid, partial)


def run(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    check_tensor("x", x)
    check_tensor("y", y)
    if y.numel() != x.numel():
        raise ValueError("all tensor inputs must have the same shape")
    grid = (triton.cdiv(x.numel(), BLOCK_SIZE),)
    partial = torch.empty((triton.cdiv(x.numel(), REDUCE_BLOCK_SIZE),), device=x.device, dtype=torch.float32)
    mse_sum_kernel[grid](partial, x.numel(), x, y, BLOCK_SIZE=BLOCK_SIZE, num_warps=NUM_WARPS)
    return reduce_sum(partial)
