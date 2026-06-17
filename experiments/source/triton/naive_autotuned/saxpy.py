import torch
import triton
import triton.language as tl

from common import ELEMENTWISE_CONFIGS, check_tensor, collect_autotune_state, size_bucket


@triton.autotune(configs=ELEMENTWISE_CONFIGS, key=["size_bucket"])
@triton.jit
def saxpy_kernel(out_ptr, n, size_bucket, x_ptr, y_ptr, a, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    y = tl.load(y_ptr + offs, mask=mask, other=0.0)
    tl.store(out_ptr + offs, a * x + y, mask=mask)


def run(a: float, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    check_tensor("x", x)
    check_tensor("y", y)
    if y.numel() != x.numel():
        raise ValueError("all tensor inputs must have the same shape")
    out = torch.empty_like(x)
    grid = lambda meta: (triton.cdiv(x.numel(), meta["BLOCK_SIZE"]),)
    saxpy_kernel[grid](out, x.numel(), size_bucket(x.numel()), x, y, float(a))
    return out


def autotune_state() -> dict[str, object]:
    return collect_autotune_state({"saxpy_kernel": saxpy_kernel})
