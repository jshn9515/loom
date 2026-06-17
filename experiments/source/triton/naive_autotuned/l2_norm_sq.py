import torch
import triton
import triton.language as tl

from common import REDUCE_CONFIGS, check_tensor, collect_autotune_state, reduce_sum, reduce_sum_kernel, size_bucket


@triton.autotune(configs=REDUCE_CONFIGS, key=["size_bucket"])
@triton.jit
def l2_norm_sq_kernel(out_ptr, n, size_bucket, x_ptr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    tl.store(out_ptr + offs, x * x, mask=mask)


def run(x: torch.Tensor) -> torch.Tensor:
    check_tensor("x", x)
    temp = torch.empty_like(x)
    grid = lambda meta: (triton.cdiv(x.numel(), meta["BLOCK_SIZE"]),)
    l2_norm_sq_kernel[grid](temp, x.numel(), size_bucket(x.numel()), x)
    return reduce_sum(temp)


def autotune_state() -> dict[str, object]:
    return collect_autotune_state({"l2_norm_sq_kernel": l2_norm_sq_kernel, "reduce_sum_kernel": reduce_sum_kernel})
