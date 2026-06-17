import torch
import triton
import triton.language as tl

from common import BLOCK_SIZE, NUM_WARPS, check_tensor


@triton.jit
def book_imbalance_kernel(out_ptr, n, bid_ptr, ask_ptr, epsilon, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n
    bid = tl.load(bid_ptr + offs, mask=mask, other=0.0)
    ask = tl.load(ask_ptr + offs, mask=mask, other=0.0)
    tl.store(out_ptr + offs, (bid - ask) / (bid + ask + epsilon), mask=mask)


def run(epsilon: float, bid: torch.Tensor, ask: torch.Tensor) -> torch.Tensor:
    check_tensor("bid", bid)
    check_tensor("ask", ask)
    if ask.numel() != bid.numel():
        raise ValueError("all tensor inputs must have the same shape")
    out = torch.empty_like(bid)
    grid = (triton.cdiv(bid.numel(), BLOCK_SIZE),)
    book_imbalance_kernel[grid](out, bid.numel(), bid, ask, float(epsilon), BLOCK_SIZE=BLOCK_SIZE, num_warps=NUM_WARPS)
    return out
