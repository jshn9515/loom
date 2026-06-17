import torch
import triton
import triton.language as tl

from common import BLOCK_SIZE, NUM_WARPS, check_tensor, reduce_sum


@triton.jit
def affine_score_reduce_kernel(out_ptr, n, x_ptr, scale, bias, threshold, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    value = scale * x + bias
    tl.store(out_ptr + offs, tl.where(value > threshold, value, 0.0), mask=mask)


def run(scale: float, bias: float, threshold: float, x: torch.Tensor) -> torch.Tensor:
    check_tensor("x", x)
    temp = torch.empty_like(x)
    grid = (triton.cdiv(x.numel(), BLOCK_SIZE),)
    affine_score_reduce_kernel[grid](
        temp,
        x.numel(),
        x,
        float(scale),
        float(bias),
        float(threshold),
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=NUM_WARPS,
    )
    return reduce_sum(temp)
