import torch
import triton
import triton.language as tl

from common import ELEMENTWISE_CONFIGS, check_tensor, collect_autotune_state, size_bucket


@triton.autotune(configs=ELEMENTWISE_CONFIGS, key=["size_bucket"])
@triton.jit
def relu_kernel(out_ptr, n, size_bucket, x_ptr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    tl.store(out_ptr + offs, tl.where(x > 0.0, x, 0.0), mask=mask)


def run(x: torch.Tensor) -> torch.Tensor:
    check_tensor("x", x)
    out = torch.empty_like(x)
    grid = lambda meta: (triton.cdiv(x.numel(), meta["BLOCK_SIZE"]),)
    relu_kernel[grid](out, x.numel(), size_bucket(x.numel()), x)
    return out


def autotune_state() -> dict[str, object]:
    return collect_autotune_state({"relu_kernel": relu_kernel})
