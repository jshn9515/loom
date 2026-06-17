import torch
import triton
import triton.language as tl

from common import BLOCK_SIZE, NUM_WARPS, REDUCE_BLOCK_SIZE, check_tensor, reduce_sum


@triton.jit
def inventory_penalty_kernel(partial_ptr, n, pos_ptr, target, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n
    pos = tl.load(pos_ptr + offs, mask=mask, other=0.0)
    d = pos - target
    partial = tl.sum(d * d, axis=0)
    tl.store(partial_ptr + pid, partial)


def run(target: float, pos: torch.Tensor) -> torch.Tensor:
    check_tensor("pos", pos)
    grid = (triton.cdiv(pos.numel(), BLOCK_SIZE),)
    partial = torch.empty((triton.cdiv(pos.numel(), REDUCE_BLOCK_SIZE),), device=pos.device, dtype=torch.float32)
    inventory_penalty_kernel[grid](partial, pos.numel(), pos, float(target), BLOCK_SIZE=BLOCK_SIZE, num_warps=NUM_WARPS)
    return reduce_sum(partial)
