import torch
import triton
import triton.language as tl

from common import BLOCK_SIZE, NUM_WARPS, check_tensor


@triton.jit
def relu_kernel(out_ptr, n, x_ptr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    tl.store(out_ptr + offs, tl.where(x > 0.0, x, 0.0), mask=mask)


def run(x: torch.Tensor) -> torch.Tensor:
    check_tensor("x", x)
    out = torch.empty_like(x)
    grid = (triton.cdiv(x.numel(), BLOCK_SIZE),)
    relu_kernel[grid](out, x.numel(), x, BLOCK_SIZE=BLOCK_SIZE, num_warps=NUM_WARPS)
    return out
