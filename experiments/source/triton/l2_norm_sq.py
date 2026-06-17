import torch
import triton
import triton.language as tl

from common import BLOCK_SIZE, NUM_WARPS, check_tensor, reduce_sum


@triton.jit
def square_kernel(out_ptr, n, x_ptr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    value = x * x
    tl.store(out_ptr + offs, value, mask=mask)


def run(x: torch.Tensor) -> torch.Tensor:
    check_tensor("x", x)
    temp = torch.empty_like(x)
    grid = (triton.cdiv(x.numel(), BLOCK_SIZE),)
    square_kernel[grid](temp, x.numel(), x, BLOCK_SIZE=BLOCK_SIZE, num_warps=NUM_WARPS)
    return reduce_sum(temp)

