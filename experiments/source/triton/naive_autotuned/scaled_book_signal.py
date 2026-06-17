import torch
import triton
import triton.language as tl

from common import ELEMENTWISE_CONFIGS, check_tensor, collect_autotune_state, size_bucket


@triton.autotune(configs=ELEMENTWISE_CONFIGS, key=["size_bucket"])
@triton.jit
def scaled_book_signal_kernel(out_ptr, n, size_bucket, bid_ptr, ask_ptr, scale, epsilon, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n
    bid = tl.load(bid_ptr + offs, mask=mask, other=0.0)
    ask = tl.load(ask_ptr + offs, mask=mask, other=0.0)
    tl.store(out_ptr + offs, scale * ((bid - ask) / (bid + ask + epsilon)), mask=mask)


def run(scale: float, epsilon: float, bid: torch.Tensor, ask: torch.Tensor) -> torch.Tensor:
    check_tensor("bid", bid)
    check_tensor("ask", ask)
    if ask.numel() != bid.numel():
        raise ValueError("all tensor inputs must have the same shape")
    out = torch.empty_like(bid)
    grid = lambda meta: (triton.cdiv(bid.numel(), meta["BLOCK_SIZE"]),)
    scaled_book_signal_kernel[grid](out, bid.numel(), size_bucket(bid.numel()), bid, ask, float(scale), float(epsilon))
    return out


def autotune_state() -> dict[str, object]:
    return collect_autotune_state({"scaled_book_signal_kernel": scaled_book_signal_kernel})
