import torch
import triton
import triton.language as tl

from common import ELEMENTWISE_CONFIGS, check_tensor, collect_autotune_state, size_bucket


@triton.autotune(configs=ELEMENTWISE_CONFIGS, key=["size_bucket"])
@triton.jit
def soft_threshold_kernel(out_ptr, n, size_bucket, x_ptr, threshold, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    out = tl.where(x > threshold, x - threshold, tl.where(x < -threshold, x + threshold, 0.0))
    tl.store(out_ptr + offs, out, mask=mask)


def run(threshold: float, x: torch.Tensor) -> torch.Tensor:
    check_tensor("x", x)
    out = torch.empty_like(x)
    grid = lambda meta: (triton.cdiv(x.numel(), meta["BLOCK_SIZE"]),)
    soft_threshold_kernel[grid](out, x.numel(), size_bucket(x.numel()), x, float(threshold))
    return out


def autotune_state() -> dict[str, object]:
    return collect_autotune_state({"soft_threshold_kernel": soft_threshold_kernel})
