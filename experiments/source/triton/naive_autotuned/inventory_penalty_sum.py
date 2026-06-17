import torch
import triton
import triton.language as tl

from common import REDUCE_CONFIGS, check_tensor, collect_autotune_state, reduce_sum, reduce_sum_kernel, size_bucket


@triton.autotune(configs=REDUCE_CONFIGS, key=["size_bucket"])
@triton.jit
def inventory_penalty_kernel(out_ptr, n, size_bucket, pos_ptr, target, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n
    pos = tl.load(pos_ptr + offs, mask=mask, other=0.0)
    d = pos - target
    tl.store(out_ptr + offs, d * d, mask=mask)


def run(target: float, pos: torch.Tensor) -> torch.Tensor:
    check_tensor("pos", pos)
    temp = torch.empty_like(pos)
    grid = lambda meta: (triton.cdiv(pos.numel(), meta["BLOCK_SIZE"]),)
    inventory_penalty_kernel[grid](temp, pos.numel(), size_bucket(pos.numel()), pos, float(target))
    return reduce_sum(temp)


def autotune_state() -> dict[str, object]:
    return collect_autotune_state({"inventory_penalty_kernel": inventory_penalty_kernel, "reduce_sum_kernel": reduce_sum_kernel})
