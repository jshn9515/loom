import torch
import triton
import triton.language as tl


BLOCK_SIZE = 1024
NUM_WARPS = 8
REDUCE_BLOCK_SIZE = 1024


def check_tensor(name: str, value: torch.Tensor) -> None:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if not value.is_cuda:
        raise ValueError(f"{name} must be a CUDA tensor")
    if value.dtype != torch.float32:
        raise ValueError(f"{name} must have dtype torch.float32")
    if value.ndim != 1:
        raise ValueError(f"{name} must be rank-1")
    if not value.is_contiguous():
        raise ValueError(f"{name} must be contiguous")


@triton.jit
def reduce_sum_kernel(input_ptr, partial_ptr, n, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n
    values = tl.load(input_ptr + offs, mask=mask, other=0.0)
    partial = tl.sum(values, axis=0)
    tl.store(partial_ptr + pid, partial)


@triton.jit
def reduce_sum_final_kernel(input_ptr, out_ptr, n, BLOCK_SIZE: tl.constexpr):
    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < n
    values = tl.load(input_ptr + offs, mask=mask, other=0.0)
    total = tl.sum(values, axis=0)
    tl.store(out_ptr, total)


def reduce_sum(value: torch.Tensor) -> torch.Tensor:
    current = value
    while current.numel() > 1:
        if current.numel() <= REDUCE_BLOCK_SIZE:
            out = torch.empty((1,), device=current.device, dtype=torch.float32)
            reduce_sum_final_kernel[(1,)](
                current,
                out,
                current.numel(),
                BLOCK_SIZE=REDUCE_BLOCK_SIZE,
                num_warps=NUM_WARPS,
            )
            return out
        grid = (triton.cdiv(current.numel(), REDUCE_BLOCK_SIZE),)
        partial = torch.empty((grid[0],), device=current.device, dtype=torch.float32)
        reduce_sum_kernel[grid](
            current,
            partial,
            current.numel(),
            BLOCK_SIZE=REDUCE_BLOCK_SIZE,
            num_warps=NUM_WARPS,
        )
        current = partial
    return current
