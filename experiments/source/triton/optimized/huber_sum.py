import torch
import triton
import triton.language as tl

from common import REDUCE_CONFIGS, REDUCE_MIN_BLOCK_SIZE, check_tensor, collect_autotune_state, reduce_sum, reduce_sum_kernel, selected_block_size, size_bucket


@triton.autotune(configs=REDUCE_CONFIGS, key=["size_bucket"])
@triton.jit
def huber_kernel(partial_ptr, n, size_bucket, x_ptr, y_ptr, delta, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    y = tl.load(y_ptr + offs, mask=mask, other=0.0)
    d = x - y
    abs_d = tl.where(d > 0.0, d, -d)
    quadratic = 0.5 * d * d
    linear = delta * (abs_d - (0.5 * delta))
    partial = tl.sum(tl.where(abs_d > delta, linear, quadratic), axis=0)
    tl.store(partial_ptr + pid, partial)


def run(delta: float, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    check_tensor("x", x)
    check_tensor("y", y)
    if y.numel() != x.numel():
        raise ValueError("all tensor inputs must have the same shape")
    grid = lambda meta: (triton.cdiv(x.numel(), meta["BLOCK_SIZE"]),)
    partial = torch.zeros((triton.cdiv(x.numel(), REDUCE_MIN_BLOCK_SIZE),), device=x.device, dtype=torch.float32)
    huber_kernel[grid](partial, x.numel(), size_bucket(x.numel()), x, y, float(delta))
    active = triton.cdiv(x.numel(), selected_block_size(huber_kernel, REDUCE_MIN_BLOCK_SIZE))
    return reduce_sum(partial[:active])


def autotune_state() -> dict[str, object]:
    return collect_autotune_state({"huber_kernel": huber_kernel, "reduce_sum_kernel": reduce_sum_kernel})
