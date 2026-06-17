import torch
import triton
import triton.language as tl

from common import REDUCE_CONFIGS, REDUCE_MIN_BLOCK_SIZE, check_tensor, collect_autotune_state, reduce_sum, reduce_sum_kernel, selected_block_size, size_bucket


@triton.autotune(configs=REDUCE_CONFIGS, key=["size_bucket"])
@triton.jit
def dot_kernel(partial_ptr, n, size_bucket, x_ptr, y_ptr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    y = tl.load(y_ptr + offs, mask=mask, other=0.0)
    partial = tl.sum(x * y, axis=0)
    tl.store(partial_ptr + pid, partial)


def run(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    check_tensor("x", x)
    check_tensor("y", y)
    if y.numel() != x.numel():
        raise ValueError("all tensor inputs must have the same shape")
    grid = lambda meta: (triton.cdiv(x.numel(), meta["BLOCK_SIZE"]),)
    partial = torch.zeros((triton.cdiv(x.numel(), REDUCE_MIN_BLOCK_SIZE),), device=x.device, dtype=torch.float32)
    dot_kernel[grid](partial, x.numel(), size_bucket(x.numel()), x, y)
    active = triton.cdiv(x.numel(), selected_block_size(dot_kernel, REDUCE_MIN_BLOCK_SIZE))
    return reduce_sum(partial[:active])


def autotune_state() -> dict[str, object]:
    return collect_autotune_state({"dot_kernel": dot_kernel, "reduce_sum_kernel": reduce_sum_kernel})
