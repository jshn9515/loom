import torch
import triton
import triton.language as tl

from common import BLOCK_SIZE, NUM_WARPS, check_tensor


@triton.jit
def soft_threshold_kernel(out_ptr, n, x_ptr, threshold, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    out = tl.where(x > threshold, x - threshold, tl.where(x < -threshold, x + threshold, 0.0))
    tl.store(out_ptr + offs, out, mask=mask)


def run(threshold: float, x: torch.Tensor) -> torch.Tensor:
    check_tensor("x", x)
    out = torch.empty_like(x)
    grid = (triton.cdiv(x.numel(), BLOCK_SIZE),)
    soft_threshold_kernel[grid](out, x.numel(), x, float(threshold), BLOCK_SIZE=BLOCK_SIZE, num_warps=NUM_WARPS)
    return out
