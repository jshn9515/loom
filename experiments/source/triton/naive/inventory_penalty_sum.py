import torch
import triton
import triton.language as tl

from common import BLOCK_SIZE, NUM_WARPS, check_tensor, reduce_sum


@triton.jit
def inventory_penalty_kernel(out_ptr, n, pos_ptr, target, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n
    pos = tl.load(pos_ptr + offs, mask=mask, other=0.0)
    d = pos - target
    tl.store(out_ptr + offs, d * d, mask=mask)


def run(target: float, pos: torch.Tensor) -> torch.Tensor:
    check_tensor("pos", pos)
    temp = torch.empty_like(pos)
    grid = (triton.cdiv(pos.numel(), BLOCK_SIZE),)
    inventory_penalty_kernel[grid](temp, pos.numel(), pos, float(target), BLOCK_SIZE=BLOCK_SIZE, num_warps=NUM_WARPS)
    return reduce_sum(temp)
