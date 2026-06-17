import torch
import triton
import triton.language as tl

from common import BLOCK_SIZE, NUM_WARPS, check_tensor, reduce_sum


@triton.jit
def mixed_weighted_affine_dot_kernel(out_ptr, n, x_ptr, y_ptr, weight, scale, bias, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    y = tl.load(y_ptr + offs, mask=mask, other=0.0)
    transformed = scale * x + bias
    tl.store(out_ptr + offs, weight * transformed * y, mask=mask)


def run(weight: float, scale: float, bias: float, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    check_tensor("x", x)
    check_tensor("y", y)
    if y.numel() != x.numel():
        raise ValueError("all tensor inputs must have the same shape")
    temp = torch.empty_like(x)
    grid = (triton.cdiv(x.numel(), BLOCK_SIZE),)
    mixed_weighted_affine_dot_kernel[grid](
        temp, x.numel(), x, y, float(weight), float(scale), float(bias), BLOCK_SIZE=BLOCK_SIZE, num_warps=NUM_WARPS
    )
    return reduce_sum(temp)
