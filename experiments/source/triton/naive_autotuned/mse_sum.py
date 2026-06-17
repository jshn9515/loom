import torch
import triton
import triton.language as tl

from common import REDUCE_CONFIGS, check_tensor, collect_autotune_state, reduce_sum, reduce_sum_kernel, size_bucket


@triton.autotune(configs=REDUCE_CONFIGS, key=["size_bucket"])
@triton.jit
def mse_sum_kernel(out_ptr, n, size_bucket, x_ptr, y_ptr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    y = tl.load(y_ptr + offs, mask=mask, other=0.0)
    d = x - y
    tl.store(out_ptr + offs, d * d, mask=mask)


def run(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    check_tensor("x", x)
    check_tensor("y", y)
    if y.numel() != x.numel():
        raise ValueError("all tensor inputs must have the same shape")
    temp = torch.empty_like(x)
    grid = lambda meta: (triton.cdiv(x.numel(), meta["BLOCK_SIZE"]),)
    mse_sum_kernel[grid](temp, x.numel(), size_bucket(x.numel()), x, y)
    return reduce_sum(temp)


def autotune_state() -> dict[str, object]:
    return collect_autotune_state({"mse_sum_kernel": mse_sum_kernel, "reduce_sum_kernel": reduce_sum_kernel})
