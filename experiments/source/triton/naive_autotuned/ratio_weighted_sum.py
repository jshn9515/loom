import torch
import triton
import triton.language as tl

from common import ELEMENTWISE_CONFIGS, check_tensor, collect_autotune_state, reduce_sum, size_bucket


@triton.autotune(configs=ELEMENTWISE_CONFIGS, key=["size_bucket"])
@triton.jit
def ratio_weighted_sum_kernel(out_ptr, n, size_bucket, x_ptr, y_ptr, scale, epsilon, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    y = tl.load(y_ptr + offs, mask=mask, other=0.0)
    tl.store(out_ptr + offs, (scale * x) / (y + epsilon), mask=mask)


def run(scale: float, epsilon: float, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    check_tensor("x", x)
    check_tensor("y", y)
    if y.numel() != x.numel():
        raise ValueError("all tensor inputs must have the same shape")
    temp = torch.empty_like(x)
    grid = lambda meta: (triton.cdiv(x.numel(), meta["BLOCK_SIZE"]),)
    ratio_weighted_sum_kernel[grid](temp, x.numel(), size_bucket(x.numel()), x, y, float(scale), float(epsilon))
    return reduce_sum(temp)


def autotune_state() -> dict[str, object]:
    return collect_autotune_state({"ratio_weighted_sum_kernel": ratio_weighted_sum_kernel})
