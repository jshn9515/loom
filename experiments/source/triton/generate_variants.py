from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parent

NAIVE_COMMON = """import torch
import triton
import triton.language as tl


BLOCK_SIZE = 256
NUM_WARPS = 4


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


def reduce_sum(value: torch.Tensor) -> torch.Tensor:
    current = value
    while current.numel() > 1:
        grid = (triton.cdiv(current.numel(), BLOCK_SIZE),)
        partial = torch.empty((grid[0],), device=current.device, dtype=torch.float32)
        reduce_sum_kernel[grid](current, partial, current.numel(), BLOCK_SIZE=BLOCK_SIZE, num_warps=NUM_WARPS)
        current = partial
    return current
"""

AUTOTUNED_COMMON = """import torch
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
"""

OPTIMIZED_FIXED_COMMON = """import torch
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
"""


ELEMENTWISE_SPECS = {
    "saxpy": {
        "kernel_name": "saxpy_kernel",
        "signature": "a: float, x: torch.Tensor, y: torch.Tensor",
        "checks": ['check_tensor("x", x)', 'check_tensor("y", y)'],
        "shape_guard": "if y.numel() != x.numel():\n        raise ValueError(\"all tensor inputs must have the same shape\")",
        "out_tensor": "x",
        "numel_tensor": "x",
        "kernel_params": "x_ptr, y_ptr, a",
        "body": "\n".join(
            [
                "x = tl.load(x_ptr + offs, mask=mask, other=0.0)",
                "    y = tl.load(y_ptr + offs, mask=mask, other=0.0)",
                "    tl.store(out_ptr + offs, a * x + y, mask=mask)",
            ]
        ),
        "launch_args": "x, y, float(a)",
    },
    "relu": {
        "kernel_name": "relu_kernel",
        "signature": "x: torch.Tensor",
        "checks": ['check_tensor("x", x)'],
        "shape_guard": "",
        "out_tensor": "x",
        "numel_tensor": "x",
        "kernel_params": "x_ptr",
        "body": "\n".join(
            [
                "x = tl.load(x_ptr + offs, mask=mask, other=0.0)",
                "    tl.store(out_ptr + offs, tl.where(x > 0.0, x, 0.0), mask=mask)",
            ]
        ),
        "launch_args": "x",
    },
    "soft_threshold": {
        "kernel_name": "soft_threshold_kernel",
        "signature": "threshold: float, x: torch.Tensor",
        "checks": ['check_tensor("x", x)'],
        "shape_guard": "",
        "out_tensor": "x",
        "numel_tensor": "x",
        "kernel_params": "x_ptr, threshold",
        "body": "\n".join(
            [
                "x = tl.load(x_ptr + offs, mask=mask, other=0.0)",
                "    out = tl.where(x > threshold, x - threshold, tl.where(x < -threshold, x + threshold, 0.0))",
                "    tl.store(out_ptr + offs, out, mask=mask)",
            ]
        ),
        "launch_args": "x, float(threshold)",
    },
    "book_imbalance": {
        "kernel_name": "book_imbalance_kernel",
        "signature": "epsilon: float, bid: torch.Tensor, ask: torch.Tensor",
        "checks": ['check_tensor("bid", bid)', 'check_tensor("ask", ask)'],
        "shape_guard": "if ask.numel() != bid.numel():\n        raise ValueError(\"all tensor inputs must have the same shape\")",
        "out_tensor": "bid",
        "numel_tensor": "bid",
        "kernel_params": "bid_ptr, ask_ptr, epsilon",
        "body": "\n".join(
            [
                "bid = tl.load(bid_ptr + offs, mask=mask, other=0.0)",
                "    ask = tl.load(ask_ptr + offs, mask=mask, other=0.0)",
                "    tl.store(out_ptr + offs, (bid - ask) / (bid + ask + epsilon), mask=mask)",
            ]
        ),
        "launch_args": "bid, ask, float(epsilon)",
    },
    "quote_filter": {
        "kernel_name": "quote_filter_kernel",
        "signature": "threshold: float, bid: torch.Tensor, ask: torch.Tensor",
        "checks": ['check_tensor("bid", bid)', 'check_tensor("ask", ask)'],
        "shape_guard": "if ask.numel() != bid.numel():\n        raise ValueError(\"all tensor inputs must have the same shape\")",
        "out_tensor": "bid",
        "numel_tensor": "bid",
        "kernel_params": "bid_ptr, ask_ptr, threshold",
        "body": "\n".join(
            [
                "bid = tl.load(bid_ptr + offs, mask=mask, other=0.0)",
                "    ask = tl.load(ask_ptr + offs, mask=mask, other=0.0)",
                "    spread = ask - bid",
                "    tl.store(out_ptr + offs, tl.where(spread > threshold, spread, 0.0), mask=mask)",
            ]
        ),
        "launch_args": "bid, ask, float(threshold)",
    },
}

REDUCTION_SPECS = {
    "l2_norm_sq": {
        "kernel_name": "l2_norm_sq_kernel",
        "signature": "x: torch.Tensor",
        "checks": ['check_tensor("x", x)'],
        "shape_guard": "",
        "temp_tensor": "x",
        "numel_tensor": "x",
        "naive_kernel_params": "x_ptr",
        "naive_body": "\n".join(
            [
                "x = tl.load(x_ptr + offs, mask=mask, other=0.0)",
                "    tl.store(out_ptr + offs, x * x, mask=mask)",
            ]
        ),
        "naive_launch_args": "x",
        "optimized_kernel_params": "x_ptr",
        "optimized_body": "\n".join(
            [
                "x = tl.load(x_ptr + offs, mask=mask, other=0.0)",
                "    partial = tl.sum(x * x, axis=0)",
                "    tl.store(partial_ptr + pid, partial)",
            ]
        ),
        "optimized_launch_args": "x",
    },
    "dot": {
        "kernel_name": "dot_kernel",
        "signature": "x: torch.Tensor, y: torch.Tensor",
        "checks": ['check_tensor("x", x)', 'check_tensor("y", y)'],
        "shape_guard": "if y.numel() != x.numel():\n        raise ValueError(\"all tensor inputs must have the same shape\")",
        "temp_tensor": "x",
        "numel_tensor": "x",
        "naive_kernel_params": "x_ptr, y_ptr",
        "naive_body": "\n".join(
            [
                "x = tl.load(x_ptr + offs, mask=mask, other=0.0)",
                "    y = tl.load(y_ptr + offs, mask=mask, other=0.0)",
                "    tl.store(out_ptr + offs, x * y, mask=mask)",
            ]
        ),
        "naive_launch_args": "x, y",
        "optimized_kernel_params": "x_ptr, y_ptr",
        "optimized_body": "\n".join(
            [
                "x = tl.load(x_ptr + offs, mask=mask, other=0.0)",
                "    y = tl.load(y_ptr + offs, mask=mask, other=0.0)",
                "    partial = tl.sum(x * y, axis=0)",
                "    tl.store(partial_ptr + pid, partial)",
            ]
        ),
        "optimized_launch_args": "x, y",
    },
    "mse_sum": {
        "kernel_name": "mse_sum_kernel",
        "signature": "x: torch.Tensor, y: torch.Tensor",
        "checks": ['check_tensor("x", x)', 'check_tensor("y", y)'],
        "shape_guard": "if y.numel() != x.numel():\n        raise ValueError(\"all tensor inputs must have the same shape\")",
        "temp_tensor": "x",
        "numel_tensor": "x",
        "naive_kernel_params": "x_ptr, y_ptr",
        "naive_body": "\n".join(
            [
                "x = tl.load(x_ptr + offs, mask=mask, other=0.0)",
                "    y = tl.load(y_ptr + offs, mask=mask, other=0.0)",
                "    d = x - y",
                "    tl.store(out_ptr + offs, d * d, mask=mask)",
            ]
        ),
        "naive_launch_args": "x, y",
        "optimized_kernel_params": "x_ptr, y_ptr",
        "optimized_body": "\n".join(
            [
                "x = tl.load(x_ptr + offs, mask=mask, other=0.0)",
                "    y = tl.load(y_ptr + offs, mask=mask, other=0.0)",
                "    d = x - y",
                "    partial = tl.sum(d * d, axis=0)",
                "    tl.store(partial_ptr + pid, partial)",
            ]
        ),
        "optimized_launch_args": "x, y",
    },
    "huber_sum": {
        "kernel_name": "huber_kernel",
        "signature": "delta: float, x: torch.Tensor, y: torch.Tensor",
        "checks": ['check_tensor("x", x)', 'check_tensor("y", y)'],
        "shape_guard": "if y.numel() != x.numel():\n        raise ValueError(\"all tensor inputs must have the same shape\")",
        "temp_tensor": "x",
        "numel_tensor": "x",
        "naive_kernel_params": "x_ptr, y_ptr, delta",
        "naive_body": "\n".join(
            [
                "x = tl.load(x_ptr + offs, mask=mask, other=0.0)",
                "    y = tl.load(y_ptr + offs, mask=mask, other=0.0)",
                "    d = x - y",
                "    abs_d = tl.where(d > 0.0, d, -d)",
                "    quadratic = 0.5 * d * d",
                "    linear = delta * (abs_d - (0.5 * delta))",
                "    tl.store(out_ptr + offs, tl.where(abs_d > delta, linear, quadratic), mask=mask)",
            ]
        ),
        "naive_launch_args": "x, y, float(delta)",
        "optimized_kernel_params": "x_ptr, y_ptr, delta",
        "optimized_body": "\n".join(
            [
                "x = tl.load(x_ptr + offs, mask=mask, other=0.0)",
                "    y = tl.load(y_ptr + offs, mask=mask, other=0.0)",
                "    d = x - y",
                "    abs_d = tl.where(d > 0.0, d, -d)",
                "    quadratic = 0.5 * d * d",
                "    linear = delta * (abs_d - (0.5 * delta))",
                "    partial = tl.sum(tl.where(abs_d > delta, linear, quadratic), axis=0)",
                "    tl.store(partial_ptr + pid, partial)",
            ]
        ),
        "optimized_launch_args": "x, y, float(delta)",
    },
    "inventory_penalty_sum": {
        "kernel_name": "inventory_penalty_kernel",
        "signature": "target: float, pos: torch.Tensor",
        "checks": ['check_tensor("pos", pos)'],
        "shape_guard": "",
        "temp_tensor": "pos",
        "numel_tensor": "pos",
        "naive_kernel_params": "pos_ptr, target",
        "naive_body": "\n".join(
            [
                "pos = tl.load(pos_ptr + offs, mask=mask, other=0.0)",
                "    d = pos - target",
                "    tl.store(out_ptr + offs, d * d, mask=mask)",
            ]
        ),
        "naive_launch_args": "pos, float(target)",
        "optimized_kernel_params": "pos_ptr, target",
        "optimized_body": "\n".join(
            [
                "pos = tl.load(pos_ptr + offs, mask=mask, other=0.0)",
                "    d = pos - target",
                "    partial = tl.sum(d * d, axis=0)",
                "    tl.store(partial_ptr + pid, partial)",
            ]
        ),
        "optimized_launch_args": "pos, float(target)",
    },
}


def render_elementwise_fixed(spec: dict[str, str]) -> str:
    checks = "\n    ".join(spec["checks"])
    shape_guard = f"\n    {spec['shape_guard']}" if spec["shape_guard"] else ""
    return f"""import torch
import triton
import triton.language as tl

from common import BLOCK_SIZE, NUM_WARPS, check_tensor


@triton.jit
def {spec['kernel_name']}(out_ptr, n, {spec['kernel_params']}, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n
    {spec['body']}


def run({spec['signature']}) -> torch.Tensor:
    {checks}{shape_guard}
    out = torch.empty_like({spec['out_tensor']})
    grid = (triton.cdiv({spec['numel_tensor']}.numel(), BLOCK_SIZE),)
    {spec['kernel_name']}[grid](out, {spec['numel_tensor']}.numel(), {spec['launch_args']}, BLOCK_SIZE=BLOCK_SIZE, num_warps=NUM_WARPS)
    return out
"""


def render_elementwise_autotuned(spec: dict[str, str]) -> str:
    checks = "\n    ".join(spec["checks"])
    shape_guard = f"\n    {spec['shape_guard']}" if spec["shape_guard"] else ""
    return f"""import torch
import triton
import triton.language as tl

from common import ELEMENTWISE_CONFIGS, check_tensor, collect_autotune_state, size_bucket


@triton.autotune(configs=ELEMENTWISE_CONFIGS, key=["size_bucket"])
@triton.jit
def {spec['kernel_name']}(out_ptr, n, size_bucket, {spec['kernel_params']}, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n
    {spec['body']}


def run({spec['signature']}) -> torch.Tensor:
    {checks}{shape_guard}
    out = torch.empty_like({spec['out_tensor']})
    grid = lambda meta: (triton.cdiv({spec['numel_tensor']}.numel(), meta["BLOCK_SIZE"]),)
    {spec['kernel_name']}[grid](out, {spec['numel_tensor']}.numel(), size_bucket({spec['numel_tensor']}.numel()), {spec['launch_args']})
    return out


def autotune_state() -> dict[str, object]:
    return collect_autotune_state({{"{spec['kernel_name']}": {spec['kernel_name']}}})
"""


def render_reduction_naive_autotuned(spec: dict[str, str]) -> str:
    checks = "\n    ".join(spec["checks"])
    shape_guard = f"\n    {spec['shape_guard']}" if spec["shape_guard"] else ""
    return f"""import torch
import triton
import triton.language as tl

from common import REDUCE_CONFIGS, check_tensor, collect_autotune_state, reduce_sum, reduce_sum_kernel, size_bucket


@triton.autotune(configs=REDUCE_CONFIGS, key=["size_bucket"])
@triton.jit
def {spec['kernel_name']}(out_ptr, n, size_bucket, {spec['naive_kernel_params']}, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n
    {spec['naive_body']}


def run({spec['signature']}) -> torch.Tensor:
    {checks}{shape_guard}
    temp = torch.empty_like({spec['temp_tensor']})
    grid = lambda meta: (triton.cdiv({spec['numel_tensor']}.numel(), meta["BLOCK_SIZE"]),)
    {spec['kernel_name']}[grid](temp, {spec['numel_tensor']}.numel(), size_bucket({spec['numel_tensor']}.numel()), {spec['naive_launch_args']})
    return reduce_sum(temp)


def autotune_state() -> dict[str, object]:
    return collect_autotune_state({{"{spec['kernel_name']}": {spec['kernel_name']}, "reduce_sum_kernel": reduce_sum_kernel}})
"""


def render_reduction_naive_fixed(spec: dict[str, str]) -> str:
    checks = "\n    ".join(spec["checks"])
    shape_guard = f"\n    {spec['shape_guard']}" if spec["shape_guard"] else ""
    return f"""import torch
import triton
import triton.language as tl

from common import BLOCK_SIZE, NUM_WARPS, check_tensor, reduce_sum


@triton.jit
def {spec['kernel_name']}(out_ptr, n, {spec['naive_kernel_params']}, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n
    {spec['naive_body']}


def run({spec['signature']}) -> torch.Tensor:
    {checks}{shape_guard}
    temp = torch.empty_like({spec['temp_tensor']})
    grid = (triton.cdiv({spec['numel_tensor']}.numel(), BLOCK_SIZE),)
    {spec['kernel_name']}[grid](temp, {spec['numel_tensor']}.numel(), {spec['naive_launch_args']}, BLOCK_SIZE=BLOCK_SIZE, num_warps=NUM_WARPS)
    return reduce_sum(temp)
"""


def render_reduction_optimized_fixed(spec: dict[str, str]) -> str:
    checks = "\n    ".join(spec["checks"])
    shape_guard = f"\n    {spec['shape_guard']}" if spec["shape_guard"] else ""
    return f"""import torch
import triton
import triton.language as tl

from common import BLOCK_SIZE, NUM_WARPS, REDUCE_BLOCK_SIZE, check_tensor, reduce_sum


@triton.jit
def {spec['kernel_name']}(partial_ptr, n, {spec['optimized_kernel_params']}, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n
    {spec['optimized_body']}


def run({spec['signature']}) -> torch.Tensor:
    {checks}{shape_guard}
    grid = (triton.cdiv({spec['numel_tensor']}.numel(), BLOCK_SIZE),)
    partial = torch.empty((triton.cdiv({spec['numel_tensor']}.numel(), REDUCE_BLOCK_SIZE),), device={spec['numel_tensor']}.device, dtype=torch.float32)
    {spec['kernel_name']}[grid](partial, {spec['numel_tensor']}.numel(), {spec['optimized_launch_args']}, BLOCK_SIZE=BLOCK_SIZE, num_warps=NUM_WARPS)
    return reduce_sum(partial)
"""


def render_reduction_optimized_autotuned(spec: dict[str, str]) -> str:
    checks = "\n    ".join(spec["checks"])
    shape_guard = f"\n    {spec['shape_guard']}" if spec["shape_guard"] else ""
    return f"""import torch
import triton
import triton.language as tl

from common import REDUCE_CONFIGS, REDUCE_MIN_BLOCK_SIZE, check_tensor, collect_autotune_state, reduce_sum, reduce_sum_kernel, selected_block_size, size_bucket


@triton.autotune(configs=REDUCE_CONFIGS, key=["size_bucket"])
@triton.jit
def {spec['kernel_name']}(partial_ptr, n, size_bucket, {spec['optimized_kernel_params']}, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n
    {spec['optimized_body']}


def run({spec['signature']}) -> torch.Tensor:
    {checks}{shape_guard}
    grid = lambda meta: (triton.cdiv({spec['numel_tensor']}.numel(), meta["BLOCK_SIZE"]),)
    partial = torch.zeros((triton.cdiv({spec['numel_tensor']}.numel(), REDUCE_MIN_BLOCK_SIZE),), device={spec['numel_tensor']}.device, dtype=torch.float32)
    {spec['kernel_name']}[grid](partial, {spec['numel_tensor']}.numel(), size_bucket({spec['numel_tensor']}.numel()), {spec['optimized_launch_args']})
    active = triton.cdiv({spec['numel_tensor']}.numel(), selected_block_size({spec['kernel_name']}, REDUCE_MIN_BLOCK_SIZE))
    return reduce_sum(partial[:active])


def autotune_state() -> dict[str, object]:
    return collect_autotune_state({{"{spec['kernel_name']}": {spec['kernel_name']}, "reduce_sum_kernel": reduce_sum_kernel}})
"""


def write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def main() -> None:
    write(ROOT / "naive" / "common.py", NAIVE_COMMON)
    write(ROOT / "naive_autotuned" / "common.py", AUTOTUNED_COMMON)
    write(ROOT / "optimized_fixed" / "common.py", OPTIMIZED_FIXED_COMMON)
    write(ROOT / "optimized" / "common.py", AUTOTUNED_COMMON)

    for name, spec in ELEMENTWISE_SPECS.items():
        write(ROOT / "naive" / f"{name}.py", render_elementwise_fixed(spec))
        write(ROOT / "naive_autotuned" / f"{name}.py", render_elementwise_autotuned(spec))
        write(ROOT / "optimized_fixed" / f"{name}.py", render_elementwise_fixed(spec))
        write(ROOT / "optimized" / f"{name}.py", render_elementwise_autotuned(spec))

    for name, spec in REDUCTION_SPECS.items():
        write(ROOT / "naive" / f"{name}.py", render_reduction_naive_fixed(spec))
        write(ROOT / "naive_autotuned" / f"{name}.py", render_reduction_naive_autotuned(spec))
        write(ROOT / "optimized_fixed" / f"{name}.py", render_reduction_optimized_fixed(spec))
        write(ROOT / "optimized" / f"{name}.py", render_reduction_optimized_autotuned(spec))


if __name__ == "__main__":
    main()
