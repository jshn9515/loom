import torch
import triton
import triton.language as tl

from common import REDUCE_CONFIGS, REDUCE_MIN_BLOCK_SIZE, check_tensor, collect_autotune_state, reduce_sum, reduce_sum_kernel, selected_block_size, size_bucket


@triton.autotune(configs=REDUCE_CONFIGS, key=["size_bucket"])
@triton.jit
def l2_norm_sq_kernel(partial_ptr, n, size_bucket, x_ptr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    partial = tl.sum(x * x, axis=0)
    tl.store(partial_ptr + pid, partial)


def run(x: torch.Tensor) -> torch.Tensor:
    check_tensor("x", x)
    grid = lambda meta: (triton.cdiv(x.numel(), meta["BLOCK_SIZE"]),)
    partial = torch.zeros((triton.cdiv(x.numel(), REDUCE_MIN_BLOCK_SIZE),), device=x.device, dtype=torch.float32)
    l2_norm_sq_kernel[grid](partial, x.numel(), size_bucket(x.numel()), x)
    active = triton.cdiv(x.numel(), selected_block_size(l2_norm_sq_kernel, REDUCE_MIN_BLOCK_SIZE))
    return reduce_sum(partial[:active])


def autotune_state() -> dict[str, object]:
    return collect_autotune_state({"l2_norm_sq_kernel": l2_norm_sq_kernel, "reduce_sum_kernel": reduce_sum_kernel})
