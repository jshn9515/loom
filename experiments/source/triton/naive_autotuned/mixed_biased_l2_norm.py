import torch
import triton
import triton.language as tl

from common import REDUCE_CONFIGS, REDUCE_MIN_BLOCK_SIZE, check_tensor, collect_autotune_state, reduce_sum, selected_block_size, size_bucket


@triton.autotune(configs=REDUCE_CONFIGS, key=["size_bucket"])
@triton.jit
def mixed_biased_l2_norm_kernel(partial_ptr, n, size_bucket, x_ptr, scale, bias, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    value = scale * x + bias
    tl.store(partial_ptr + pid, tl.sum(value * value, axis=0))


def run(scale: float, bias: float, x: torch.Tensor) -> torch.Tensor:
    check_tensor("x", x)
    grid = lambda meta: (triton.cdiv(x.numel(), meta["BLOCK_SIZE"]),)
    partial = torch.zeros((triton.cdiv(x.numel(), REDUCE_MIN_BLOCK_SIZE),), device=x.device, dtype=torch.float32)
    mixed_biased_l2_norm_kernel[grid](partial, x.numel(), size_bucket(x.numel()), x, float(scale), float(bias))
    active = triton.cdiv(x.numel(), selected_block_size(mixed_biased_l2_norm_kernel, REDUCE_MIN_BLOCK_SIZE))
    return reduce_sum(partial[:active])


def autotune_state() -> dict[str, object]:
    return collect_autotune_state({"mixed_biased_l2_norm_kernel": mixed_biased_l2_norm_kernel})
