import torch
import triton
import triton.language as tl

from common import BLOCK_SIZE, NUM_WARPS, REDUCE_BLOCK_SIZE, check_tensor, reduce_sum


@triton.jit
def signal_clip_reduce_kernel(partial_ptr, n, bid_ptr, ask_ptr, scale, epsilon, clip, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n
    bid = tl.load(bid_ptr + offs, mask=mask, other=0.0)
    ask = tl.load(ask_ptr + offs, mask=mask, other=0.0)
    value = scale * ((bid - ask) / (bid + ask + epsilon))
    clipped = tl.where(value > clip, clip, tl.where(value < -clip, -clip, value))
    tl.store(partial_ptr + pid, tl.sum(clipped, axis=0))


def run(scale: float, epsilon: float, clip: float, bid: torch.Tensor, ask: torch.Tensor) -> torch.Tensor:
    check_tensor("bid", bid)
    check_tensor("ask", ask)
    if ask.numel() != bid.numel():
        raise ValueError("all tensor inputs must have the same shape")
    grid = (triton.cdiv(bid.numel(), BLOCK_SIZE),)
    partial = torch.empty((triton.cdiv(bid.numel(), REDUCE_BLOCK_SIZE),), device=bid.device, dtype=torch.float32)
    signal_clip_reduce_kernel[grid](
        partial,
        bid.numel(),
        bid,
        ask,
        float(scale),
        float(epsilon),
        float(clip),
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=NUM_WARPS,
    )
    return reduce_sum(partial)
