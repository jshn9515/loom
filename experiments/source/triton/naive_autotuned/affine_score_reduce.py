import torch
import triton
import triton.language as tl

from common import ELEMENTWISE_CONFIGS, check_tensor, collect_autotune_state, reduce_sum, size_bucket


@triton.autotune(configs=ELEMENTWISE_CONFIGS, key=["size_bucket"])
@triton.jit
def affine_score_reduce_kernel(out_ptr, n, size_bucket, x_ptr, scale, bias, threshold, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    value = scale * x + bias
    tl.store(out_ptr + offs, tl.where(value > threshold, value, 0.0), mask=mask)


def run(scale: float, bias: float, threshold: float, x: torch.Tensor) -> torch.Tensor:
    check_tensor("x", x)
    temp = torch.empty_like(x)
    grid = lambda meta: (triton.cdiv(x.numel(), meta["BLOCK_SIZE"]),)
    affine_score_reduce_kernel[grid](
        temp,
        x.numel(),
        size_bucket(x.numel()),
        x,
        float(scale),
        float(bias),
        float(threshold),
    )
    return reduce_sum(temp)


def autotune_state() -> dict[str, object]:
    return collect_autotune_state({"affine_score_reduce_kernel": affine_score_reduce_kernel})
