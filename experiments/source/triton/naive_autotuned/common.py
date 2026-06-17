import torch
import triton
import triton.language as tl


ELEMENTWISE_CONFIGS = [
    triton.Config({"BLOCK_SIZE": 256}, num_warps=4, num_stages=2),
    triton.Config({"BLOCK_SIZE": 512}, num_warps=4, num_stages=3),
    triton.Config({"BLOCK_SIZE": 1024}, num_warps=8, num_stages=4),
]
REDUCE_CONFIGS = [
    triton.Config({"BLOCK_SIZE": 256}, num_warps=4, num_stages=2),
    triton.Config({"BLOCK_SIZE": 512}, num_warps=4, num_stages=3),
    triton.Config({"BLOCK_SIZE": 1024}, num_warps=8, num_stages=4),
]
REDUCE_BLOCK_SIZE = 1024
REDUCE_MIN_BLOCK_SIZE = 256
BUCKET_UPPER_BOUNDS = [262144, 4194304]


def size_bucket(n: int) -> int:
    for index, bound in enumerate(BUCKET_UPPER_BOUNDS):
        if n <= bound:
            return index
    return len(BUCKET_UPPER_BOUNDS)


def _config_to_dict(config):
    if config is None:
        return None
    return {
        "meta": dict(getattr(config, "kwargs", {})),
        "num_warps": getattr(config, "num_warps", None),
        "num_stages": getattr(config, "num_stages", None),
    }


def collect_autotune_state(kernels: dict[str, object]) -> dict[str, object]:
    rendered = {}
    for name, kernel in kernels.items():
        cache = {}
        for key, config in getattr(kernel, "cache", {}).items():
            cache[str(key)] = _config_to_dict(config)
        rendered[name] = {
            "best_config": _config_to_dict(getattr(kernel, "best_config", None)),
            "cache": cache,
        }
    return {
        "enabled": True,
        "bucket_upper_bounds": BUCKET_UPPER_BOUNDS,
        "kernels": rendered,
    }


def selected_block_size(kernel, default: int) -> int:
    config = getattr(kernel, "best_config", None)
    if config is None:
        return default
    kwargs = getattr(config, "kwargs", {})
    return int(kwargs.get("BLOCK_SIZE", default))


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


@triton.autotune(configs=REDUCE_CONFIGS, key=["size_bucket"])
@triton.jit
def reduce_sum_kernel(input_ptr, partial_ptr, n, size_bucket, BLOCK_SIZE: tl.constexpr):
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
            reduce_sum_final_kernel[(1,)](current, out, current.numel(), BLOCK_SIZE=REDUCE_BLOCK_SIZE, num_warps=8)
            return out
        grid = lambda meta: (triton.cdiv(current.numel(), meta["BLOCK_SIZE"]),)
        partial = torch.zeros(
            (triton.cdiv(current.numel(), REDUCE_MIN_BLOCK_SIZE),),
            device=current.device,
            dtype=torch.float32,
        )
        reduce_sum_kernel[grid](current, partial, current.numel(), size_bucket(current.numel()))
        active = triton.cdiv(current.numel(), selected_block_size(reduce_sum_kernel, REDUCE_MIN_BLOCK_SIZE))
        current = partial[:active]
    return current
