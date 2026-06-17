import torch
import triton
import triton.language as tl

from common import ELEMENTWISE_CONFIGS, check_tensor, collect_autotune_state, size_bucket


@triton.autotune(configs=ELEMENTWISE_CONFIGS, key=["size_bucket"])
@triton.jit
def affine_clamp_kernel(out_ptr, n, size_bucket, x_ptr, scale, bias, lo, hi, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    value = scale * x + bias
    value = tl.where(value < lo, lo, value)
    value = tl.where(value > hi, hi, value)
    tl.store(out_ptr + offs, value, mask=mask)


def run(scale: float, bias: float, lo: float, hi: float, x: torch.Tensor) -> torch.Tensor:
    check_tensor("x", x)
    out = torch.empty_like(x)
    grid = lambda meta: (triton.cdiv(x.numel(), meta["BLOCK_SIZE"]),)
    affine_clamp_kernel[grid](out, x.numel(), size_bucket(x.numel()), x, float(scale), float(bias), float(lo), float(hi))
    return out


def autotune_state() -> dict[str, object]:
    return collect_autotune_state({"affine_clamp_kernel": affine_clamp_kernel})
