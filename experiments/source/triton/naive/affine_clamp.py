import torch
import triton
import triton.language as tl

from common import BLOCK_SIZE, NUM_WARPS, check_tensor


@triton.jit
def affine_clamp_kernel(out_ptr, n, x_ptr, scale, bias, lo, hi, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    value = scale * x + bias
    value = tl.where(value < lo, lo, value)
    value = tl.where(value > hi, hi, value)
    tl.store(out_ptr + offs, value, mask=mask)


def run(scale: float, bias: float, lo: float, hi: float, x: torch.Tensor) -> torch.Tensor:
    check_tensor("x", x)
    out = torch.empty_like(x)
    grid = (triton.cdiv(x.numel(), BLOCK_SIZE),)
    affine_clamp_kernel[grid](
        out, x.numel(), x, float(scale), float(bias), float(lo), float(hi), BLOCK_SIZE=BLOCK_SIZE, num_warps=NUM_WARPS
    )
    return out
