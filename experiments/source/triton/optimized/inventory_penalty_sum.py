import torch
import triton
import triton.language as tl

from common import REDUCE_CONFIGS, REDUCE_MIN_BLOCK_SIZE, check_tensor, collect_autotune_state, reduce_sum, reduce_sum_kernel, selected_block_size, size_bucket


@triton.autotune(configs=REDUCE_CONFIGS, key=["size_bucket"])
@triton.jit
def inventory_penalty_kernel(partial_ptr, n, size_bucket, pos_ptr, target, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n
    pos = tl.load(pos_ptr + offs, mask=mask, other=0.0)
    d = pos - target
    partial = tl.sum(d * d, axis=0)
    tl.store(partial_ptr + pid, partial)


def run(target: float, pos: torch.Tensor) -> torch.Tensor:
    check_tensor("pos", pos)
    grid = lambda meta: (triton.cdiv(pos.numel(), meta["BLOCK_SIZE"]),)
    partial = torch.zeros((triton.cdiv(pos.numel(), REDUCE_MIN_BLOCK_SIZE),), device=pos.device, dtype=torch.float32)
    inventory_penalty_kernel[grid](partial, pos.numel(), size_bucket(pos.numel()), pos, float(target))
    active = triton.cdiv(pos.numel(), selected_block_size(inventory_penalty_kernel, REDUCE_MIN_BLOCK_SIZE))
    return reduce_sum(partial[:active])


def autotune_state() -> dict[str, object]:
    return collect_autotune_state({"inventory_penalty_kernel": inventory_penalty_kernel, "reduce_sum_kernel": reduce_sum_kernel})
