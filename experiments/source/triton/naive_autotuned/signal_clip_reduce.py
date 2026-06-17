import torch
import triton
import triton.language as tl

from common import ELEMENTWISE_CONFIGS, check_tensor, collect_autotune_state, reduce_sum, size_bucket


@triton.autotune(configs=ELEMENTWISE_CONFIGS, key=["size_bucket"])
@triton.jit
def signal_clip_reduce_kernel(out_ptr, n, size_bucket, bid_ptr, ask_ptr, scale, epsilon, clip, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n
    bid = tl.load(bid_ptr + offs, mask=mask, other=0.0)
    ask = tl.load(ask_ptr + offs, mask=mask, other=0.0)
    value = scale * ((bid - ask) / (bid + ask + epsilon))
    clipped = tl.where(value > clip, clip, tl.where(value < -clip, -clip, value))
    tl.store(out_ptr + offs, clipped, mask=mask)


def run(scale: float, epsilon: float, clip: float, bid: torch.Tensor, ask: torch.Tensor) -> torch.Tensor:
    check_tensor("bid", bid)
    check_tensor("ask", ask)
    if ask.numel() != bid.numel():
        raise ValueError("all tensor inputs must have the same shape")
    temp = torch.empty_like(bid)
    grid = lambda meta: (triton.cdiv(bid.numel(), meta["BLOCK_SIZE"]),)
    signal_clip_reduce_kernel[grid](
        temp,
        bid.numel(),
        size_bucket(bid.numel()),
        bid,
        ask,
        float(scale),
        float(epsilon),
        float(clip),
    )
    return reduce_sum(temp)


def autotune_state() -> dict[str, object]:
    return collect_autotune_state({"signal_clip_reduce_kernel": signal_clip_reduce_kernel})
