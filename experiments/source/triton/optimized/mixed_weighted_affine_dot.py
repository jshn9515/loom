import torch
import triton
import triton.language as tl

from common import REDUCE_CONFIGS, REDUCE_MIN_BLOCK_SIZE, check_tensor, collect_autotune_state, reduce_sum, selected_block_size, size_bucket


@triton.autotune(configs=REDUCE_CONFIGS, key=["size_bucket"])
@triton.jit
def mixed_weighted_affine_dot_kernel(partial_ptr, n, size_bucket, x_ptr, y_ptr, weight, scale, bias, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    y = tl.load(y_ptr + offs, mask=mask, other=0.0)
    transformed = scale * x + bias
    tl.store(partial_ptr + pid, tl.sum(weight * transformed * y, axis=0))


def run(weight: float, scale: float, bias: float, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    check_tensor("x", x)
    check_tensor("y", y)
    if y.numel() != x.numel():
        raise ValueError("all tensor inputs must have the same shape")
    grid = lambda meta: (triton.cdiv(x.numel(), meta["BLOCK_SIZE"]),)
    partial = torch.zeros((triton.cdiv(x.numel(), REDUCE_MIN_BLOCK_SIZE),), device=x.device, dtype=torch.float32)
    mixed_weighted_affine_dot_kernel[grid](partial, x.numel(), size_bucket(x.numel()), x, y, float(weight), float(scale), float(bias))
    active = triton.cdiv(x.numel(), selected_block_size(mixed_weighted_affine_dot_kernel, REDUCE_MIN_BLOCK_SIZE))
    return reduce_sum(partial[:active])


def autotune_state() -> dict[str, object]:
    return collect_autotune_state({"mixed_weighted_affine_dot_kernel": mixed_weighted_affine_dot_kernel})
