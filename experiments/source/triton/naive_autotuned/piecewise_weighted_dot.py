import torch
import triton
import triton.language as tl

from common import ELEMENTWISE_CONFIGS, check_tensor, collect_autotune_state, reduce_sum, size_bucket


@triton.autotune(configs=ELEMENTWISE_CONFIGS, key=["size_bucket"])
@triton.jit
def piecewise_weighted_dot_kernel(out_ptr, n, size_bucket, x_ptr, y_ptr, weight_pos, weight_neg, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    y = tl.load(y_ptr + offs, mask=mask, other=0.0)
    prod = x * y
    weight = tl.where(x > 0.0, weight_pos, weight_neg)
    tl.store(out_ptr + offs, weight * prod, mask=mask)


def run(weight_pos: float, weight_neg: float, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    check_tensor("x", x)
    check_tensor("y", y)
    if y.numel() != x.numel():
        raise ValueError("all tensor inputs must have the same shape")
    temp = torch.empty_like(x)
    grid = lambda meta: (triton.cdiv(x.numel(), meta["BLOCK_SIZE"]),)
    piecewise_weighted_dot_kernel[grid](
        temp,
        x.numel(),
        size_bucket(x.numel()),
        x,
        y,
        float(weight_pos),
        float(weight_neg),
    )
    return reduce_sum(temp)


def autotune_state() -> dict[str, object]:
    return collect_autotune_state({"piecewise_weighted_dot_kernel": piecewise_weighted_dot_kernel})
