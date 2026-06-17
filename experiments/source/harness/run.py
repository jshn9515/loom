from __future__ import annotations

import argparse
import csv
import ctypes
import hashlib
import importlib.util
import json
import math
import os
import pathlib
import random
import shutil
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass
from typing import Any, Callable

os.environ.setdefault("MPLCONFIGDIR", "/tmp/loom-matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import torch


ROOT = pathlib.Path(__file__).resolve().parents[3]
EXPERIMENTS_ROOT = ROOT / "experiments"
SOURCE_ROOT = EXPERIMENTS_ROOT / "source"
WORKLOAD_CONFIG_DIR = SOURCE_ROOT / "data" / "workloads"
PROFILE_CONFIG_DIR = SOURCE_ROOT / "data" / "profiles"
TUNING_DATA_PATH = SOURCE_ROOT / "data" / "tuning" / "settings.json"
EVAL_DATA_PATH = SOURCE_ROOT / "data" / "evaluation" / "settings.json"
RESULTS_ROOT = EXPERIMENTS_ROOT / "results" / "local"
WORK_ROOT = EXPERIMENTS_ROOT / "_work"
LOOM_BIN = ROOT / "_build" / "default" / "src" / "loom_cli" / "main.exe"
LOOM_BUILD_TARGET = "src/loom_cli/main.exe"
HELD_OUT_GENERALIZATION_KERNELS = [
    "affine_score_reduce_reordered",
    "clipped_huber_sum_reordered",
    "mixed_book_signal_reassociated",
    "piecewise_weighted_dot_reordered",
    "ratio_weighted_sum_reassociated",
    "relu_guard_reordered",
    "signal_clip_reduce_reassociated",
    "soft_threshold_reordered",
]
HELD_OUT_CUDA_COMPARISON_IMPLEMENTATIONS = [
    "loom_cuda_fixed",
    "loom_full_fixed",
    "triton_optimized_fixed",
    "cuda_naive",
    "cuda_optimized",
]
CURRENT_NON_HELD_OUT_KERNELS = [
    "affine_clamp",
    "affine_score_reduce",
    "book_imbalance",
    "clipped_huber_sum",
    "dot",
    "dot_pipeline",
    "huber_sum",
    "inventory_penalty_sum",
    "l2_norm_sq",
    "mixed_biased_l2_norm",
    "mixed_book_signal",
    "mixed_weighted_affine_dot",
    "mse_sum",
    "piecewise_weighted_dot",
    "quote_filter",
    "ratio_weighted_sum",
    "relu",
    "relu_tupled",
    "saxpy",
    "saxpy_curried",
    "scaled_book_signal",
    "signal_clip_reduce",
    "soft_threshold",
    "weighted_dot",
    "weighted_dot_pipeline",
]
CURRENT_CUDA_FOCUSED_KERNELS = [
    "affine_clamp",
    "scaled_book_signal",
    "relu_tupled",
    "relu",
    "saxpy",
    "saxpy_curried",
    "soft_threshold",
    "book_imbalance",
    "mixed_book_signal",
    "quote_filter",
    "signal_clip_reduce",
    "clipped_huber_sum",
]
CURRENT_CUDA_PASS_KERNELS = list(CURRENT_NON_HELD_OUT_KERNELS)
CURRENT_CUDA_MILESTONE_SIZES = [131072, 2097152, 8388608]
CURRENT_CUDA_COMPARISON_IMPLEMENTATIONS = [
    "loom_cuda_fixed",
    "loom_full_fixed",
    "triton_naive_fixed",
    "triton_optimized_fixed",
    "cuda_naive",
    "cuda_optimized",
]
CURRENT_SECURED_GAP_THRESHOLD = 1.02
CURRENT_TRITON_FOCUSED_KERNELS = [
    "weighted_dot",
    "weighted_dot_pipeline",
    "mixed_weighted_affine_dot",
    "dot",
    "dot_pipeline",
    "mse_sum",
    "inventory_penalty_sum",
    "l2_norm_sq",
    "saxpy",
    "saxpy_curried",
    "relu",
    "relu_tupled",
    "affine_clamp",
    "scaled_book_signal",
    "quote_filter",
    "book_imbalance",
]
CURRENT_TRITON_PASS_KERNELS = list(CURRENT_NON_HELD_OUT_KERNELS)
CURRENT_TRITON_COMPARISON_IMPLEMENTATIONS = [
    "loom_full_fixed",
    "loom_triton_previous_fixed",
    "triton_naive_fixed",
    "triton_optimized_fixed",
]
CURRENT_TRITON_FIXED_EXTERNAL_IMPLEMENTATIONS = [
    "triton_naive_fixed",
    "triton_optimized_fixed",
]
CURRENT_TRITON_SECURED_GAP_THRESHOLD = 1.05

EXTERNAL_BASELINE_SPECS = [
    {
        "key": "triton_naive_fixed",
        "label": "Triton Naive Fixed",
        "color": "#ffdd99",
        "kind": "triton",
        "variant": "naive",
        "autotuned": False,
    },
    {
        "key": "triton_naive_autotuned",
        "label": "Triton Naive Autotuned",
        "color": "#ffbb78",
        "kind": "triton",
        "variant": "naive",
        "autotuned": True,
    },
    {
        "key": "triton_optimized_fixed",
        "label": "Triton Optimized Fixed",
        "color": "#ff9966",
        "kind": "triton",
        "variant": "optimized",
        "autotuned": False,
    },
    {
        "key": "triton_optimized_autotuned",
        "label": "Triton Optimized Autotuned",
        "color": "#ff7f0e",
        "kind": "triton",
        "variant": "optimized",
        "autotuned": True,
    },
    {
        "key": "cuda_naive",
        "label": "CUDA Naive",
        "color": "#98df8a",
        "kind": "cuda",
        "variant": "naive",
        "autotuned": False,
    },
    {
        "key": "cuda_optimized",
        "label": "CUDA Optimized",
        "color": "#2ca02c",
        "kind": "cuda",
        "variant": "optimized",
        "autotuned": False,
    },
]
REFERENCE_IMPLEMENTATION = "loom_none_fixed"
REFERENCE_KERNELS = {
    "saxpy",
    "relu",
    "l2_norm_sq",
    "dot",
    "weighted_dot",
    "ratio_weighted_sum",
    "piecewise_weighted_dot",
    "mse_sum",
    "huber_sum",
    "soft_threshold",
    "clipped_huber_sum",
    "affine_clamp",
    "affine_score_reduce",
    "book_imbalance",
    "scaled_book_signal",
    "signal_clip_reduce",
    "quote_filter",
    "inventory_penalty_sum",
    "mixed_biased_l2_norm",
    "mixed_book_signal",
    "mixed_weighted_affine_dot",
}
TRITON_VARIANT_DIRS = ("naive", "naive_autotuned", "optimized_fixed", "optimized")
CUDA_VARIANT_DIRS = ("naive", "optimized")

TUNING_MEASUREMENT_FIELDS = [
    "kernel",
    "implementation",
    "implementation_label",
    "implementation_kind",
    "autotuned",
    "optimization_flags",
    "size",
    "bucket",
    "dataset_id",
    "dataset_index",
    "dataset_seed",
    "dataset_seed_offset",
    "seconds",
    "application_domain",
    "workload_class",
    "application",
]

COMPILE_MEASUREMENT_FIELDS = [
    "kernel",
    "implementation",
    "implementation_label",
    "implementation_kind",
    "autotuned",
    "optimization_flags",
    "run_index",
    "seconds",
    "application_domain",
    "workload_class",
    "application",
]

RUNTIME_MEASUREMENT_FIELDS = [
    "kernel",
    "implementation",
    "implementation_label",
    "implementation_kind",
    "autotuned",
    "optimization_flags",
    "size",
    "dataset_id",
    "dataset_index",
    "dataset_seed",
    "dataset_seed_offset",
    "run_index",
    "seconds",
    "application_domain",
    "workload_class",
    "application",
]

VERIFICATION_MEASUREMENT_FIELDS = [
    "kernel",
    "implementation",
    "implementation_label",
    "implementation_kind",
    "autotuned",
    "optimization_flags",
    "phase",
    "size",
    "dataset_id",
    "dataset_index",
    "dataset_seed",
    "dataset_seed_offset",
    "status",
    "max_abs_diff",
    "max_rel_diff",
    "expected_checksum",
    "actual_checksum",
    "verification_mode",
    "failure_message",
    "application_domain",
    "workload_class",
    "application",
]

SUMMARY_FIELDS = [
    "kernel",
    "implementation",
    "implementation_label",
    "implementation_kind",
    "autotuned",
    "optimization_flags",
    "size",
    "median_ms",
    "q1_ms",
    "q3_ms",
    "speedup_vs_loom_none_fixed",
    "capability_expectation_met",
    "application_domain",
    "workload_class",
    "application",
]

FRONTEND_MEASUREMENT_FIELDS = [
    "kernel",
    "frontend",
    "phase",
    "run_index",
    "seconds",
    "status",
    "application_domain",
    "workload_class",
    "application",
]

FRONTEND_SUMMARY_FIELDS = [
    "kernel",
    "frontend",
    "phase",
    "median_ms",
    "runs",
    "application_domain",
    "workload_class",
    "application",
]

FRONTEND_PARITY_FIELDS = [
    "kernel",
    "frontend",
    "tensor_ir_match",
    "kernel_plan_match",
    "status",
]

FRONTEND_RUNTIME_MEASUREMENT_FIELDS = [
    "kernel",
    "frontend",
    "loom_profile",
    "backend",
    "size",
    "dataset_id",
    "dataset_index",
    "dataset_seed",
    "dataset_seed_offset",
    "run_index",
    "seconds",
    "application_domain",
    "workload_class",
    "application",
]

FRONTEND_RUNTIME_VERIFICATION_FIELDS = [
    "kernel",
    "frontend",
    "loom_profile",
    "backend",
    "size",
    "dataset_id",
    "dataset_index",
    "dataset_seed",
    "dataset_seed_offset",
    "status",
    "max_abs_diff",
    "max_rel_diff",
    "expected_checksum",
    "actual_checksum",
    "failure_message",
    "application_domain",
    "workload_class",
    "application",
]

FRONTEND_RUNTIME_SUMMARY_FIELDS = [
    "kernel",
    "frontend",
    "loom_profile",
    "backend",
    "size",
    "median_ms",
    "q1_ms",
    "q3_ms",
    "ratio_vs_ocaml",
    "application_domain",
    "workload_class",
    "application",
]

GAP_FIELDS = [
    "kernel",
    "size",
    "loom_implementation",
    "loom_label",
    "loom_median_ms",
    "best_external_implementation",
    "best_external_label",
    "best_external_median_ms",
    "gap_ratio",
    "application_domain",
    "workload_class",
    "application",
]

GAP_BY_CLASS_FIELDS = [
    "workload_class",
    "application_domain",
    "median_gap_to_best_fixed_triton",
    "worst_gap_to_best_fixed_triton",
    "loom_wins_vs_best_fixed_triton",
    "case_count",
]

COMPLETED_UNIT_FIELDS = [
    "stage",
    "kernel",
    "implementation",
    "size",
    "dataset_id",
    "dataset_seed",
    "dataset_seed_offset",
    "status",
    "detail",
]


@dataclass
class RunOptions:
    kernels: list[str]
    warmup: int
    runtime_repetitions: int
    compile_repetitions: int
    tuning_sizes: list[int]
    evaluation_sizes: list[int]
    tuning_seed_offsets: list[int]
    evaluation_seed_offsets: list[int]
    results_dir: pathlib.Path
    work_dir: pathlib.Path
    run_tuning: bool
    run_compile: bool
    run_verification: bool
    run_runtime: bool
    mode: str
    candidate_groups: list[str]
    baseline_groups: list[str]
    implementation_filter: list[str]
    benchmark_cases: dict[tuple[str, str], set[int] | None]


@dataclass
class RawResultWriters:
    tuning: AppendCsvWriter
    compile: AppendCsvWriter
    runtime: AppendCsvWriter
    verification: AppendCsvWriter
    completed_units: AppendCsvWriter
    dataset_manifest_jsonl: AppendJsonlWriter
    verification_failures_json: pathlib.Path

    def flush(self) -> None:
        self.tuning.flush()
        self.compile.flush()
        self.runtime.flush()
        self.verification.flush()
        self.completed_units.flush()
        self.dataset_manifest_jsonl.flush()


def compile_results_dir(results_dir: pathlib.Path) -> pathlib.Path:
    return results_dir / "compile"


def runtime_results_dir(results_dir: pathlib.Path) -> pathlib.Path:
    return results_dir / "runtime"


def frontend_results_dir(results_dir: pathlib.Path) -> pathlib.Path:
    return results_dir / "frontend"


def shared_results_dir(results_dir: pathlib.Path) -> pathlib.Path:
    return results_dir / "shared"


def reset_path(path: pathlib.Path) -> None:
    if path.is_dir():
        shutil.rmtree(path)
    elif path.exists():
        path.unlink()


def prepare_results_tree(results_dir: pathlib.Path) -> None:
    for path in (
        compile_results_dir(results_dir) / "raw",
        compile_results_dir(results_dir) / "plots",
        runtime_results_dir(results_dir) / "raw",
        runtime_results_dir(results_dir) / "plots",
        runtime_results_dir(results_dir) / "summaries",
    ):
        reset_path(path)
        ensure_dir(path)
    shared_raw = shared_results_dir(results_dir) / "raw"
    ensure_dir(shared_raw)
    for name in (
        "dataset_manifest.json",
        "dataset_manifest.jsonl",
        "audit_report.json",
        "generalizability_audit.json",
        "environment.json",
        "environment.txt",
        "completed_units.csv",
        "run_state.json",
        "capability_checks.csv",
    ):
        reset_path(shared_raw / name)
    reset_path(runtime_results_dir(results_dir) / "report.md")


def build_raw_result_writers(results_dir: pathlib.Path) -> RawResultWriters:
    return RawResultWriters(
        tuning=AppendCsvWriter(
            runtime_results_dir(results_dir) / "raw" / "tuning_measurements.csv",
            TUNING_MEASUREMENT_FIELDS,
        ),
        compile=AppendCsvWriter(
            compile_results_dir(results_dir) / "raw" / "compile_measurements.csv",
            COMPILE_MEASUREMENT_FIELDS,
        ),
        runtime=AppendCsvWriter(
            runtime_results_dir(results_dir) / "raw" / "runtime_measurements.csv",
            RUNTIME_MEASUREMENT_FIELDS,
        ),
        verification=AppendCsvWriter(
            runtime_results_dir(results_dir) / "raw" / "verification_measurements.csv",
            VERIFICATION_MEASUREMENT_FIELDS,
        ),
        completed_units=AppendCsvWriter(
            shared_results_dir(results_dir) / "raw" / "completed_units.csv",
            COMPLETED_UNIT_FIELDS,
        ),
        dataset_manifest_jsonl=AppendJsonlWriter(shared_results_dir(results_dir) / "raw" / "dataset_manifest.jsonl"),
        verification_failures_json=runtime_results_dir(results_dir) / "raw" / "verification_failures.json",
    )


def load_json(path: pathlib.Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def discover_workload_configs() -> list[dict[str, Any]]:
    configs = []
    for path in sorted(WORKLOAD_CONFIG_DIR.glob("*.json")):
        config = load_json(path)
        config["_config_path"] = str(path.relative_to(ROOT))
        configs.append(config)
    return configs


def discover_loom_profiles() -> list[dict[str, Any]]:
    profiles: list[dict[str, Any]] = []
    for path in sorted(PROFILE_CONFIG_DIR.glob("*.json")):
        payload = load_json(path)
        items = payload.get("profiles", [payload])
        for profile in items:
            record = dict(profile)
            record["_config_path"] = str(path.relative_to(ROOT))
            record["backend"] = str(record.get("backend", "triton"))
            record["kind"] = f"loom-{record['backend']}"
            record["autotuned"] = bool(record.get("autotuned", False))
            record["optimizations"] = list(record.get("optimizations", []))
            record["comparison_modes"] = list(record.get("comparison_modes", []))
            profiles.append(record)
    profiles.sort(key=lambda item: (int(item.get("sort_order", 0)), item["key"]))
    return profiles


def implementation_label_map(specs: list[dict[str, Any]]) -> dict[str, str]:
    return {spec["key"]: spec["label"] for spec in specs}


def implementation_color_map(specs: list[dict[str, Any]]) -> dict[str, str]:
    return {spec["key"]: spec["color"] for spec in specs}


LOOM_PROFILE_SPECS = discover_loom_profiles()
LOOM_PROFILE_MAP = {spec["key"]: spec for spec in LOOM_PROFILE_SPECS}
EXTERNAL_IMPLEMENTATION_MAP = {spec["key"]: spec for spec in EXTERNAL_BASELINE_SPECS}
ALL_IMPLEMENTATION_SPECS = LOOM_PROFILE_SPECS + EXTERNAL_BASELINE_SPECS
ALL_IMPLEMENTATION_MAP = {spec["key"]: spec for spec in ALL_IMPLEMENTATION_SPECS}
ALL_IMPLEMENTATION_LABELS = implementation_label_map(ALL_IMPLEMENTATION_SPECS)
ALL_IMPLEMENTATION_COLORS = implementation_color_map(ALL_IMPLEMENTATION_SPECS)
LOOM_INTERNAL_ORDER = tuple(
    spec["key"] for spec in LOOM_PROFILE_SPECS if "internal" in spec["comparison_modes"]
)
LOOM_SEARCH_ORDER = tuple(
    spec["key"] for spec in LOOM_PROFILE_SPECS if "search" in spec["comparison_modes"]
)
LOOM_VS_OTHERS_ORDER = tuple(
    [spec["key"] for spec in LOOM_PROFILE_SPECS if "external" in spec["comparison_modes"]]
    + [spec["key"] for spec in EXTERNAL_BASELINE_SPECS]
)
TUNED_IMPLEMENTATIONS = tuple(
    [spec["key"] for spec in LOOM_PROFILE_SPECS if spec["autotuned"]]
    + [spec["key"] for spec in EXTERNAL_BASELINE_SPECS if spec["kind"] == "triton" and spec["autotuned"]]
)


def load_config() -> dict[str, Any]:
    workloads = discover_workload_configs()
    tuning = load_json(TUNING_DATA_PATH)
    evaluation = load_json(EVAL_DATA_PATH)
    benchmark_workloads = sorted((item for item in workloads if item["kind"] == "benchmark"), key=lambda item: item["name"])
    capability_cases = sorted((item for item in workloads if item.get("capability_expected") is not None), key=lambda item: item["name"])
    return {
        "supported_kernels": benchmark_workloads,
        "capability_cases": capability_cases,
        "seed": int(tuning["seed"]),
        "compile_probe_size": int(tuning["compile_probe_size"]),
        "tuning_sizes": list(tuning["tuning_sizes"]),
        "tuning_seed_offsets": [int(value) for value in tuning.get("tuning_seed_offsets", [0])],
        "autotune_config": str(ROOT / tuning["autotune_config"]),
        "optimizer_config": str(ROOT / tuning["optimizer_config"]),
        "warmup_repetitions": int(evaluation["warmup_repetitions"]),
        "runtime_repetitions": int(evaluation["runtime_repetitions"]),
        "compile_repetitions": int(evaluation["compile_repetitions"]),
        "evaluation_sizes": list(evaluation["evaluation_sizes"]),
        "evaluation_seed_offsets": [int(value) for value in evaluation.get("evaluation_seed_offsets", [0])],
        "loom_profiles": LOOM_PROFILE_SPECS,
        "external_implementations": [spec["key"] for spec in EXTERNAL_BASELINE_SPECS],
        "loom_internal_order": list(LOOM_INTERNAL_ORDER),
        "loom_search_order": list(LOOM_SEARCH_ORDER),
        "loom_vs_others_order": list(LOOM_VS_OTHERS_ORDER),
    }


def filter_config_to_implementations(config: dict[str, Any], allowed_keys: set[str]) -> dict[str, Any]:
    filtered = {
        **config,
        "loom_profiles": [profile for profile in config["loom_profiles"] if profile["key"] in allowed_keys],
        "external_implementations": [
            key for key in config["external_implementations"] if key in allowed_keys
        ],
        "loom_internal_order": [key for key in config["loom_internal_order"] if key in allowed_keys],
        "loom_search_order": [key for key in config.get("loom_search_order", []) if key in allowed_keys],
        "loom_vs_others_order": [key for key in config["loom_vs_others_order"] if key in allowed_keys],
    }
    return filtered


def require_known_implementations(keys: list[str]) -> None:
    unknown = [key for key in keys if key not in ALL_IMPLEMENTATION_MAP]
    if unknown:
        raise SystemExit(f"unknown implementation/group key(s): {', '.join(sorted(unknown))}")


def assert_performance_sources_are_ocaml(config: dict[str, Any]) -> None:
    non_ocaml = [
        f"{kernel['name']}:{kernel.get('input_kind')}:{kernel.get('source_path')}"
        for kernel in config["supported_kernels"]
        if kernel.get("input_kind") != "ocaml"
        or "/ocaml/" not in f"/{kernel.get('source_path', '')}"
    ]
    if non_ocaml:
        raise SystemExit(
            "performance benchmarks with external groups must use OCaml sources only: "
            + ", ".join(non_ocaml)
        )


def parse_benchmark_cases(
    raw_cases: list[str] | None,
    supported_kernel_names: set[str],
) -> dict[tuple[str, str], set[int] | None]:
    cases: dict[tuple[str, str], set[int] | None] = {}
    for raw_case in raw_cases or []:
        parts = raw_case.split(":")
        if len(parts) not in {2, 3} or not all(parts):
            raise SystemExit(
                "--benchmark-case expects IMPLEMENTATION:KERNEL or IMPLEMENTATION:KERNEL:SIZE"
            )
        implementation, kernel = parts[0], parts[1]
        require_known_implementations([implementation])
        if kernel not in supported_kernel_names:
            raise SystemExit(f"unknown benchmark kernel in --benchmark-case: {kernel}")
        key = (implementation, kernel)
        if len(parts) == 2:
            cases[key] = None
            continue
        try:
            size = int(parts[2])
        except ValueError as exc:
            raise SystemExit(f"invalid benchmark-case size: {parts[2]}") from exc
        if size <= 0:
            raise SystemExit(f"invalid benchmark-case size: {size}")
        if cases.get(key) is None and key in cases:
            continue
        cases.setdefault(key, set())
        assert cases[key] is not None
        cases[key].add(size)
    return cases


def benchmark_cases_to_json(
    cases: dict[tuple[str, str], set[int] | None]
) -> list[dict[str, Any]]:
    rows = []
    for (implementation, kernel), sizes in sorted(cases.items()):
        rows.append(
            {
                "implementation": implementation,
                "kernel": kernel,
                "sizes": [] if sizes is None else sorted(sizes),
            }
        )
    return rows


def selected_for_kernel(options: RunOptions, implementation: str, kernel: str) -> bool:
    if options.implementation_filter and implementation not in options.implementation_filter:
        return False
    if options.benchmark_cases:
        return (implementation, kernel) in options.benchmark_cases
    return True


def selected_sizes_for_kernel(
    options: RunOptions,
    implementation: str,
    kernel: str,
) -> list[int]:
    if not selected_for_kernel(options, implementation, kernel):
        return []
    if not options.benchmark_cases:
        return list(options.evaluation_sizes)
    sizes = options.benchmark_cases[(implementation, kernel)]
    if sizes is None:
        return list(options.evaluation_sizes)
    return [size for size in options.evaluation_sizes if size in sizes]


def current_git_revision() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        return "unknown"
    return result.stdout.strip() or "unknown"


def kernel_tuning_sizes(kernel: dict[str, Any], default_sizes: list[int]) -> list[int]:
    return [int(size) for size in kernel.get("tuning_sizes", default_sizes)]


def kernel_compile_probe_size(kernel: dict[str, Any], default_size: int) -> int:
    return int(kernel.get("compile_probe_size", default_size))


def triton_variant_dir(implementation_key: str) -> str:
    if implementation_key == "triton_naive_fixed":
        return "naive"
    if implementation_key == "triton_naive_autotuned":
        return "naive_autotuned"
    if implementation_key == "triton_optimized_fixed":
        return "optimized_fixed"
    if implementation_key == "triton_optimized_autotuned":
        return "optimized"
    raise ValueError(f"unknown Triton implementation {implementation_key}")


def ensure_cuda_environment() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for experiments, but torch.cuda.is_available() is false")
    if shutil.which("nvcc") is None:
        raise SystemExit("CUDA experiments require nvcc on PATH")


def ensure_loom_built() -> None:
    subprocess.run(["dune", "build", LOOM_BUILD_TARGET], cwd=ROOT, check=True)


def ensure_dir(path: pathlib.Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_csv(path: pathlib.Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


class AppendCsvWriter:
    def __init__(self, path: pathlib.Path, fieldnames: list[str]) -> None:
        self.path = path
        self.fieldnames = fieldnames
        ensure_dir(path.parent)
        self._header_written = path.exists() and path.stat().st_size > 0
        self._buffer: list[dict[str, Any]] = []

    def append_row(self, row: dict[str, Any]) -> None:
        self.append_rows([row])

    def append_rows(self, rows: list[dict[str, Any]]) -> None:
        if not rows:
            return
        self._buffer.extend(rows)

    def flush(self) -> None:
        if not self._buffer:
            return
        with open(self.path, "a", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=self.fieldnames, lineterminator="\n")
            if not self._header_written:
                writer.writeheader()
                self._header_written = True
            writer.writerows(self._buffer)
            handle.flush()
            os.fsync(handle.fileno())
        self._buffer.clear()


class AppendJsonlWriter:
    def __init__(self, path: pathlib.Path) -> None:
        self.path = path
        ensure_dir(path.parent)
        self._buffer: list[dict[str, Any]] = []

    def append(self, payload: dict[str, Any]) -> None:
        self._buffer.append(payload)

    def flush(self) -> None:
        if not self._buffer:
            return
        with open(self.path, "a", encoding="utf-8") as handle:
            for payload in self._buffer:
                handle.write(json.dumps(payload, sort_keys=True))
                handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        self._buffer.clear()


def write_json(path: pathlib.Path, payload: dict[str, Any]) -> None:
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def write_text(path: pathlib.Path, text: str) -> None:
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(text)


def read_csv(path: pathlib.Path) -> list[dict[str, Any]]:
    with open(path, "r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def read_csv_if_exists(path: pathlib.Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return read_csv(path)


def read_jsonl(path: pathlib.Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def module_from_path(path: pathlib.Path, module_name: str):
    sys.path.insert(0, str(path.parent))
    prior_common = sys.modules.pop("common", None)
    try:
        spec = importlib.util.spec_from_file_location(module_name, path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"unable to import module from {path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.pop(0)
        sys.modules.pop("common", None)
        sys.modules.pop(module_name, None)
        if prior_common is not None:
            sys.modules["common"] = prior_common


def add_implementation_legend(ax: plt.Axes, implementation_order: tuple[str, ...]) -> None:
    handles = [
        plt.Rectangle((0, 0), 1, 1, facecolor=ALL_IMPLEMENTATION_COLORS[key], alpha=0.75)
        for key in implementation_order
    ]
    ax.legend(handles, [ALL_IMPLEMENTATION_LABELS[key] for key in implementation_order], loc="best")


def kernel_config_map(kernels: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {kernel["name"]: kernel for kernel in kernels}


def plot_scale_mode(kernel_configs: dict[str, dict[str, Any]], kernel: str) -> str:
    config = kernel_configs.get(kernel, {})
    return str(config.get("plot_scale_mode", "linear_only"))


def grouped_positions(group_count: int, implementations: tuple[str, ...]) -> tuple[list[float], list[float]]:
    stride = len(implementations) + 1.0
    positions: list[float] = []
    centers: list[float] = []
    for group_index in range(group_count):
        base = group_index * stride
        impl_positions = [base + offset + 1.0 for offset in range(len(implementations))]
        positions.extend(impl_positions)
        centers.append(sum(impl_positions) / len(impl_positions))
    return positions, centers


def make_tensor(seed: int, size: int) -> torch.Tensor:
    generator = torch.Generator(device="cuda")
    generator.manual_seed(seed)
    return torch.randn(size, device="cuda", dtype=torch.float32, generator=generator)


def dataset_seed(base_seed: int, seed_offset: int) -> int:
    return int(base_seed) + int(seed_offset)


def make_kernel_inputs(
    kernel: dict[str, Any], size: int, base_seed: int, seed_offset: int = 0
) -> dict[str, Any]:
    inputs: dict[str, Any] = {}
    seed = dataset_seed(base_seed, seed_offset)
    for offset, name in enumerate(kernel["tensor_inputs"]):
        inputs[name] = make_tensor(seed + (offset * 104729), size)
    for name, value in kernel["scalar_args"].items():
        inputs[name] = float(value)
    return inputs


def tensor_checksum(value: torch.Tensor) -> str:
    digest = hashlib.sha256()
    digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()[:16]


def scalar_arg_checksum(kernel: dict[str, Any]) -> str:
    rendered = json.dumps(kernel.get("scalar_args", {}), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(rendered.encode("utf-8")).hexdigest()[:16]


def dataset_id(kernel: str, phase: str, size: int, seed_offset: int) -> str:
    return f"{phase}:{kernel}:n{size}:seed{seed_offset}"


def dataset_manifest_entry(
    kernel: dict[str, Any],
    phase: str,
    size: int,
    base_seed: int,
    seed_offset: int,
    inputs: dict[str, Any],
) -> dict[str, Any]:
    seed = dataset_seed(base_seed, seed_offset)
    tensor_checksums = {
        name: tensor_checksum(value)
        for name, value in inputs.items()
        if isinstance(value, torch.Tensor)
    }
    tensor_shapes = {
        name: list(value.shape)
        for name, value in inputs.items()
        if isinstance(value, torch.Tensor)
    }
    return {
        "dataset_id": dataset_id(kernel["name"], phase, size, seed_offset),
        "phase": phase,
        "kernel": kernel["name"],
        "logical_kernel": logical_kernel_name(kernel),
        "size": int(size),
        "seed": seed,
        "seed_offset": int(seed_offset),
        "tensor_checksums": tensor_checksums,
        "tensor_shapes": tensor_shapes,
        "scalar_args": dict(kernel.get("scalar_args", {})),
        "scalar_args_checksum": scalar_arg_checksum(kernel),
    }


def verification_mode(kernel: dict[str, Any]) -> str:
    return str(kernel.get("verification_mode", "exact_reference"))


def verification_tolerances(kernel: dict[str, Any]) -> tuple[float, float]:
    return float(kernel.get("verification_rtol", 1e-4)), float(kernel.get("verification_atol", 1e-5))


def logical_kernel_name(kernel: dict[str, Any]) -> str:
    return str(kernel.get("baseline_name", kernel["name"]))


def kernel_bucket(size: int, bucket_upper_bounds: list[int]) -> int:
    for index, bound in enumerate(bucket_upper_bounds):
        if size <= bound:
            return index
    return len(bucket_upper_bounds)


def reference_output(kernel_name: str, inputs: dict[str, Any]) -> torch.Tensor:
    if kernel_name == "saxpy":
        return inputs["a"] * inputs["x"] + inputs["y"]
    if kernel_name == "relu":
        return torch.relu(inputs["x"])
    if kernel_name == "l2_norm_sq":
        return torch.sum(inputs["x"] * inputs["x"]).reshape(1)
    if kernel_name == "dot":
        return torch.sum(inputs["x"] * inputs["y"]).reshape(1)
    if kernel_name == "weighted_dot":
        return torch.sum(inputs["weight"] * inputs["x"] * inputs["y"]).reshape(1)
    if kernel_name == "mixed_weighted_affine_dot":
        value = inputs["scale"] * inputs["x"] + inputs["bias"]
        return torch.sum(inputs["weight"] * value * inputs["y"]).reshape(1)
    if kernel_name == "mixed_biased_l2_norm":
        value = inputs["scale"] * inputs["x"] + inputs["bias"]
        return torch.sum(value * value).reshape(1)
    if kernel_name == "ratio_weighted_sum":
        return torch.sum((inputs["scale"] * inputs["x"]) / (inputs["y"] + inputs["epsilon"])).reshape(1)
    if kernel_name == "piecewise_weighted_dot":
        weight = torch.where(inputs["x"] > 0.0, torch.full_like(inputs["x"], inputs["weight_pos"]), torch.full_like(inputs["x"], inputs["weight_neg"]))
        return torch.sum(weight * inputs["x"] * inputs["y"]).reshape(1)
    if kernel_name == "mse_sum":
        d = inputs["x"] - inputs["y"]
        return torch.sum(d * d).reshape(1)
    if kernel_name == "huber_sum":
        d = inputs["x"] - inputs["y"]
        abs_d = torch.abs(d)
        delta = inputs["delta"]
        quadratic = 0.5 * d * d
        linear = delta * (abs_d - (0.5 * delta))
        return torch.sum(torch.where(abs_d > delta, linear, quadratic)).reshape(1)
    if kernel_name == "soft_threshold":
        threshold = inputs["threshold"]
        x = inputs["x"]
        return torch.where(x > threshold, x - threshold, torch.where(x < -threshold, x + threshold, torch.zeros_like(x)))
    if kernel_name == "clipped_huber_sum":
        d = inputs["x"] - inputs["y"]
        abs_d = torch.abs(d)
        delta = inputs["delta"]
        cap = inputs["cap"]
        quadratic = 0.5 * d * d
        linear = delta * (abs_d - (0.5 * delta))
        value = torch.where(abs_d > delta, linear, quadratic)
        return torch.minimum(value, torch.full_like(value, cap)).sum().reshape(1)
    if kernel_name == "affine_clamp":
        value = inputs["scale"] * inputs["x"] + inputs["bias"]
        return torch.clamp(value, min=inputs["lo"], max=inputs["hi"])
    if kernel_name == "affine_score_reduce":
        value = inputs["scale"] * inputs["x"] + inputs["bias"]
        return torch.where(value > inputs["threshold"], value, torch.zeros_like(value)).sum().reshape(1)
    if kernel_name == "book_imbalance":
        return (inputs["bid"] - inputs["ask"]) / (inputs["bid"] + inputs["ask"] + inputs["epsilon"])
    if kernel_name == "scaled_book_signal":
        return inputs["scale"] * ((inputs["bid"] - inputs["ask"]) / (inputs["bid"] + inputs["ask"] + inputs["epsilon"]))
    if kernel_name == "mixed_book_signal":
        depth = inputs["bid"] + inputs["ask"]
        value = inputs["scale"] * ((inputs["bid"] - inputs["ask"]) / (depth + inputs["epsilon"]))
        return torch.where(depth > inputs["threshold"], value, torch.zeros_like(value))
    if kernel_name == "signal_clip_reduce":
        value = inputs["scale"] * ((inputs["bid"] - inputs["ask"]) / (inputs["bid"] + inputs["ask"] + inputs["epsilon"]))
        return torch.clamp(value, min=-inputs["clip"], max=inputs["clip"]).sum().reshape(1)
    if kernel_name == "quote_filter":
        spread = inputs["ask"] - inputs["bid"]
        return torch.where(spread > inputs["threshold"], spread, torch.zeros_like(spread))
    if kernel_name == "inventory_penalty_sum":
        d = inputs["pos"] - inputs["target"]
        return torch.sum(d * d).reshape(1)
    raise ValueError(f"unknown kernel {kernel_name}")


def compare_outputs(actual: torch.Tensor, expected: torch.Tensor) -> tuple[float, float]:
    abs_diff = torch.max(torch.abs(actual - expected)).item()
    denom = torch.maximum(torch.abs(expected), torch.full_like(expected, 1e-8))
    rel_diff = torch.max(torch.abs(actual - expected) / denom).item()
    return float(abs_diff), float(rel_diff)


def assert_close(actual: torch.Tensor, expected: torch.Tensor, kernel: dict[str, Any]) -> tuple[float, float]:
    rtol, atol = verification_tolerances(kernel)
    torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol)
    return compare_outputs(actual, expected)


_CUDA_STABILITY_TENSORS: tuple[torch.Tensor, torch.Tensor] | None = None
_CUDA_BENCHMARK_PREWARMED = False


def cuda_stability_warmup(iterations: int = 4) -> None:
    if not torch.cuda.is_available():
        return
    global _CUDA_STABILITY_TENSORS
    if _CUDA_STABILITY_TENSORS is None:
        lhs = torch.full((4 * 1024 * 1024,), 0.25, device="cuda", dtype=torch.float32)
        rhs = torch.full((4 * 1024 * 1024,), 0.5, device="cuda", dtype=torch.float32)
        _CUDA_STABILITY_TENSORS = (lhs, rhs)
    lhs, rhs = _CUDA_STABILITY_TENSORS
    for _ in range(iterations):
        torch.add(lhs, rhs, out=lhs)
        torch.mul(lhs, 0.5, out=lhs)
    torch.cuda.synchronize()


def cuda_benchmark_prewarm(seconds: float = 10.0) -> None:
    if not torch.cuda.is_available():
        return
    global _CUDA_BENCHMARK_PREWARMED
    if _CUDA_BENCHMARK_PREWARMED:
        return
    _CUDA_BENCHMARK_PREWARMED = True
    deadline = time.perf_counter() + seconds
    while time.perf_counter() < deadline:
        cuda_stability_warmup()


def cuda_timing_block_prewarm(seconds: float = 0.25) -> None:
    if not torch.cuda.is_available():
        return
    deadline = time.perf_counter() + seconds
    while time.perf_counter() < deadline:
        cuda_stability_warmup()


def timed_runs(fn: Callable[[], torch.Tensor], warmup: int, repetitions: int) -> list[float]:
    cuda_timing_block_prewarm()
    for _ in range(warmup):
        _ = fn()
        torch.cuda.synchronize()
    measurements: list[float] = []
    for _ in range(repetitions):
        torch.cuda.synchronize()
        start = time.perf_counter()
        _ = fn()
        torch.cuda.synchronize()
        measurements.append(time.perf_counter() - start)
    return measurements


def bucket_upper_bounds_for_autotune(module) -> list[int]:
    metadata = getattr(module, "_LOOM_AUTOTUNE_METADATA", None)
    if not isinstance(metadata, dict):
        return []
    bounds = metadata.get("bucket_upper_bounds", [])
    return [int(value) for value in bounds]


def loom_compile(
    source: pathlib.Path,
    entry: str,
    input_kind: str,
    out_dir: pathlib.Path,
    backend: str,
    optimizations: list[str],
    optimizer_config: str | None = None,
    autotune_config: str | None = None,
    cuda_platform: str | None = None,
) -> None:
    ensure_dir(out_dir)
    command = [
        str(LOOM_BIN),
        "compile",
        str(source),
        "--input-kind",
        input_kind,
        "--entry",
        entry,
        "--target",
        backend,
        "--out",
        str(out_dir),
        "--emit",
        "all",
    ]
    if optimizer_config is not None:
        command.extend(["--opt-config", optimizer_config])
    if backend == "cuda" and cuda_platform is not None:
        command.extend(["--cuda-platform", cuda_platform])
    for optimization in optimizations:
        command.extend(["--enable-opt", optimization])
    if backend == "triton" and autotune_config is not None:
        command.extend(["--autotune-config", autotune_config])
    result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip())


def measure_loom_compile(
    kernel: dict[str, Any],
    profile: dict[str, Any],
    repetitions: int,
    work_dir: pathlib.Path,
    autotune_config: str,
    optimizer_config: str,
    probe_size: int,
    seed: int,
) -> list[float]:
    source = EXPERIMENTS_ROOT / kernel["source_path"]
    rows = []
    config = autotune_config if profile["autotuned"] else None
    max_attempts = 3
    for index in range(repetitions):
        out_dir = work_dir / "compile" / profile["key"] / kernel["name"] / f"run_{index}"
        cache_dir = work_dir / "compile" / f"{profile['key']}_cache" / kernel["name"] / f"run_{index}"
        payload_path = work_dir / "compile" / f"{profile['key']}_jobs" / kernel["name"] / f"run_{index}.json"
        if out_dir.exists():
            shutil.rmtree(out_dir)
        if cache_dir.exists():
            shutil.rmtree(cache_dir)
        ensure_dir(payload_path.parent)
        ensure_dir(cache_dir)
        write_json(
            payload_path,
            {
                "source": str(source),
                "entry": kernel["entry"],
                "input_kind": kernel["input_kind"],
                "out_dir": str(out_dir),
                "backend": profile.get("backend", "triton"),
                "optimizations": list(profile["optimizations"]),
                "cuda_platform": profile.get("cuda_platform"),
                "optimizer_config": optimizer_config,
                "autotune_config": config,
                "probe_size": int(probe_size),
                "seed": int(seed + index),
                "kernel_name": kernel["name"],
                "module_name": f"{profile['key']}_compile_{kernel['name']}_{index}",
            },
        )
        env = os.environ.copy()
        env["TRITON_CACHE_DIR"] = str(cache_dir)
        result = None
        for attempt in range(max_attempts):
            result = subprocess.run(
                [
                    sys.executable,
                    str(pathlib.Path(__file__).resolve()),
                    "--internal-loom-compile-json",
                    str(payload_path),
                ],
                cwd=ROOT,
                env=env,
                capture_output=True,
                text=True,
            )
            if result.returncode == 0:
                break
            stderr = result.stderr.strip()
            transient_cuda_loss = (
                profile.get("backend", "triton") == "triton"
                and (
                    "no CUDA-capable device is detected" in stderr
                    or "cudaErrorNoDevice" in stderr
                )
            )
            if not transient_cuda_loss or attempt == max_attempts - 1:
                raise RuntimeError(stderr)
            time.sleep(2.0 * (attempt + 1))
        assert result is not None
        rows.append(float(result.stdout.strip()))
    return rows


FRONTEND_COMPARISON_FRONTENDS = ("ocaml", "python", "cpp")
FRONTEND_COMPARISON_BASELINE = "ocaml"


def frontend_source_path(kernel: dict[str, Any], frontend: str) -> str:
    source = pathlib.Path(str(kernel["source_path"]))
    if frontend == "ocaml":
        return str(source)
    parts = list(source.parts)
    try:
        index = parts.index("ocaml")
    except ValueError as exc:
        raise RuntimeError(f"{kernel['name']}: cannot derive {frontend} source from {source}") from exc
    if frontend == "python":
        parts[index] = "python"
        return str(pathlib.Path(*parts).with_suffix(".py"))
    if frontend == "cpp":
        parts[index] = "cpp"
        return str(pathlib.Path(*parts).with_suffix(".cpp"))
    raise ValueError(f"unknown frontend {frontend}")


def frontend_source(kernel: dict[str, Any], frontend: str) -> pathlib.Path:
    return EXPERIMENTS_ROOT / frontend_source_path(kernel, frontend)


def frontend_ocaml_source(kernel: dict[str, Any]) -> pathlib.Path:
    return EXPERIMENTS_ROOT / kernel["source_path"]


def frontend_kernel_variant(kernel: dict[str, Any], frontend: str) -> dict[str, Any]:
    return {
        **kernel,
        "input_kind": frontend,
        "source_path": frontend_source_path(kernel, frontend),
    }


def run_frontend_command(source: pathlib.Path, entry: str, frontend: str) -> None:
    result = subprocess.run(
        [
            str(LOOM_BIN),
            "front-ir",
            str(source),
            "--input-kind",
            frontend,
            "--entry",
            entry,
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip())


def run_frontend_compile(
    source: pathlib.Path,
    entry: str,
    frontend: str,
    out_dir: pathlib.Path,
) -> None:
    if out_dir.exists():
        shutil.rmtree(out_dir)
    result = subprocess.run(
        [
            str(LOOM_BIN),
            "compile",
            str(source),
            "--input-kind",
            frontend,
            "--entry",
            entry,
            "--target",
            "triton",
            "--out",
            str(out_dir),
            "--emit",
            "all",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip())


def plot_frontend_comparison(summary_rows: list[dict[str, Any]], output_dir: pathlib.Path) -> None:
    ensure_dir(output_dir)
    kernels = sorted({str(row["kernel"]) for row in summary_rows})
    frontends = FRONTEND_COMPARISON_FRONTENDS
    phases = ("front-ir", "full-compile")
    labels = {"front-ir": "FrontIR lowering", "full-compile": "Full compile"}
    fig, axes = plt.subplots(1, 2, figsize=(18, max(10, len(kernels) * 0.34)), sharey=True)
    if hasattr(axes, "ravel"):
        axes_list = list(axes.ravel())
    elif isinstance(axes, (list, tuple)):
        axes_list = list(axes)
    else:
        axes = [axes]
        axes_list = list(axes)
    y_base = list(range(len(kernels)))
    bar_height = 0.24
    colors = {"ocaml": "#4c78a8", "python": "#f58518", "cpp": "#54a24b"}
    by_key = {
        (str(row["kernel"]), str(row["frontend"]), str(row["phase"])): float(row["median_ms"])
        for row in summary_rows
    }
    for ax, phase in zip(axes_list, phases, strict=True):
        for index, frontend in enumerate(frontends):
            center = (len(frontends) - 1) / 2.0
            offsets = [y + ((index - center) * bar_height) for y in y_base]
            values = [by_key.get((kernel, frontend, phase), 0.0) for kernel in kernels]
            ax.barh(offsets, values, height=bar_height, label=frontend.upper(), color=colors[frontend])
        ax.set_title(labels[phase])
        ax.set_xlabel("median milliseconds")
        ax.grid(axis="x", alpha=0.25)
        ax.set_yticks(y_base, kernels)
        ax.invert_yaxis()
    axes_list[0].set_ylabel("benchmark")
    axes_list[-1].legend(loc="lower right")
    fig.suptitle("Loom frontend comparison: OCaml vs Python vs C++")
    fig.tight_layout()
    fig.savefig(output_dir / "frontend_comparison.png", dpi=180)
    plt.close(fig)


def percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return values[0]
    position = (len(values) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return values[int(position)]
    weight = position - lower
    return values[lower] * (1.0 - weight) + values[upper] * weight


def summarize_frontend_runtime(runtime_rows: list[dict[str, Any]], kernels: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str, int], list[float]] = {}
    metadata = {kernel["name"]: kernel for kernel in kernels}
    for row in runtime_rows:
        grouped.setdefault(
            (
                str(row["kernel"]),
                str(row["frontend"]),
                str(row["loom_profile"]),
                int(row["size"]),
            ),
            [],
        ).append(float(row["seconds"]))
    medians = {key: statistics.median(values) * 1000.0 for key, values in grouped.items()}
    summary_rows: list[dict[str, Any]] = []
    for (kernel, frontend, profile, size), values in sorted(grouped.items()):
        sorted_values = sorted(values)
        median_ms = statistics.median(sorted_values) * 1000.0
        q1_ms = percentile(sorted_values, 0.25) * 1000.0
        q3_ms = percentile(sorted_values, 0.75) * 1000.0
        baseline_key = (kernel, FRONTEND_COMPARISON_BASELINE, profile, size)
        baseline_ms = medians.get(baseline_key)
        ratio = "" if baseline_ms is None or baseline_ms == 0.0 else median_ms / baseline_ms
        kernel_meta = metadata[kernel]
        summary_rows.append(
            {
                "kernel": kernel,
                "frontend": frontend,
                "loom_profile": profile,
                "backend": str(LOOM_PROFILE_MAP[profile].get("backend", "triton")),
                "size": size,
                "median_ms": median_ms,
                "q1_ms": q1_ms,
                "q3_ms": q3_ms,
                "ratio_vs_ocaml": ratio,
                "application_domain": kernel_meta.get("application_domain", ""),
                "workload_class": kernel_meta.get("workload_class", ""),
                "application": kernel_meta.get("application", ""),
            }
        )
    return summary_rows


def plot_frontend_runtime_summary(summary_rows: list[dict[str, Any]], output_dir: pathlib.Path) -> None:
    ensure_dir(output_dir)
    ratio_rows = [
        row
        for row in summary_rows
        if row["frontend"] != FRONTEND_COMPARISON_BASELINE and row["ratio_vs_ocaml"] != ""
    ]
    if not ratio_rows:
        return
    profiles = sorted({str(row["loom_profile"]) for row in ratio_rows})
    sizes = sorted({int(row["size"]) for row in ratio_rows})
    frontends = sorted({str(row["frontend"]) for row in ratio_rows})
    fig, axes = plt.subplots(
        len(profiles) * len(frontends),
        len(sizes),
        figsize=(max(12, len(sizes) * 5.0), max(6, len(profiles) * len(frontends) * 5.0)),
        squeeze=False,
        sharey=True,
    )
    for profile_index, profile in enumerate(profiles):
        for frontend_index, frontend in enumerate(frontends):
            row_index = profile_index * len(frontends) + frontend_index
            for col_index, size in enumerate(sizes):
                ax = axes[row_index][col_index]
                rows = [
                    row
                    for row in ratio_rows
                    if row["loom_profile"] == profile
                    and row["frontend"] == frontend
                    and int(row["size"]) == size
                ]
                rows.sort(key=lambda row: float(row["ratio_vs_ocaml"]))
                kernels = [str(row["kernel"]) for row in rows]
                values = [float(row["ratio_vs_ocaml"]) for row in rows]
                colors = [
                    "#2ca02c" if value <= 1.02 else "#ff7f0e" if value <= 1.10 else "#d62728"
                    for value in values
                ]
                ax.barh(range(len(rows)), values, color=colors)
                ax.axvline(1.0, color="black", linewidth=1.0, linestyle="--")
                ax.axvline(1.02, color="#777777", linewidth=0.8, linestyle=":")
                ax.set_title(f"{frontend.upper()} / OCaml, {profile} @ {size}")
                ax.set_xlabel(f"{frontend.upper()} / OCaml median runtime")
                ax.set_yticks(range(len(rows)), kernels)
                ax.invert_yaxis()
                ax.grid(axis="x", alpha=0.25)
    fig.suptitle("Loom cross-frontend runtime parity: Python and C++ vs OCaml")
    fig.tight_layout()
    fig.savefig(output_dir / "frontend_runtime_ratio.png", dpi=180)
    plt.close(fig)


def run_frontend_runtime_comparison(
    kernels: list[dict[str, Any]],
    args: argparse.Namespace,
    raw_dir: pathlib.Path,
    summary_dir: pathlib.Path,
    plot_dir: pathlib.Path,
) -> None:
    ensure_cuda_environment()
    cuda_benchmark_prewarm()
    profiles = ["loom_full_fixed", "loom_cuda_fixed"]
    sizes = args.size if args.size else list(load_config()["evaluation_sizes"])
    seed_offsets = list(load_config()["evaluation_seed_offsets"])
    warmup = args.warmup if args.warmup is not None else 5
    repetitions = args.runtime_repetitions if args.runtime_repetitions is not None else int(load_config()["runtime_repetitions"])
    work_dir = args.work_dir / "frontend_runtime"
    reset_path(work_dir)
    ensure_dir(work_dir)
    runtime_rows: list[dict[str, Any]] = []
    verification_rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    config = load_config()
    for kernel in kernels:
        for profile_key in profiles:
            base_profile = LOOM_PROFILE_MAP[profile_key]
            runners: dict[str, Callable[..., torch.Tensor]] = {}
            for frontend in FRONTEND_COMPARISON_FRONTENDS:
                profile = dict(base_profile)
                profile["key"] = f"frontend_{frontend}_{profile_key}"
                runners[frontend] = load_loom_runtime(
                    frontend_kernel_variant(kernel, frontend),
                    profile,
                    work_dir,
                    config["autotune_config"],
                    config["optimizer_config"],
                )
            for size in sizes:
                for dataset_index, seed_offset in enumerate(seed_offsets):
                    inputs = make_kernel_inputs(kernel, size, config["seed"] + size, seed_offset)
                    expected = reference_output(logical_kernel_name(kernel), inputs)
                    dataset = dataset_id(kernel["name"], "frontend", size, seed_offset)
                    for frontend in FRONTEND_COMPARISON_FRONTENDS:
                        runner = runners[frontend]
                        actual: torch.Tensor | None = None
                        try:
                            actual = runner(**inputs)
                            torch.cuda.synchronize()
                            max_abs_diff, max_rel_diff = assert_close(actual, expected, kernel)
                            expected_checksum = tensor_checksum(expected)
                            actual_checksum = tensor_checksum(actual)
                            status = "pass"
                            failure_message = ""
                        except Exception as exc:
                            torch.cuda.synchronize()
                            status = "fail"
                            failure_message = str(exc)
                            expected_checksum = tensor_checksum(expected)
                            if actual is not None:
                                max_abs_diff, max_rel_diff = compare_outputs(actual, expected)
                                actual_checksum = tensor_checksum(actual)
                            else:
                                max_abs_diff, max_rel_diff = math.inf, math.inf
                                actual_checksum = ""
                            failures.append(
                                {
                                    "kernel": kernel["name"],
                                    "frontend": frontend,
                                    "loom_profile": profile_key,
                                    "size": size,
                                    "message": failure_message,
                                }
                            )
                        verification_rows.append(
                            {
                                "kernel": kernel["name"],
                                "frontend": frontend,
                                "loom_profile": profile_key,
                                "backend": str(base_profile.get("backend", "triton")),
                                "size": size,
                                "dataset_id": dataset,
                                "dataset_index": dataset_index,
                                "dataset_seed": dataset_seed(config["seed"] + size, seed_offset),
                                "dataset_seed_offset": seed_offset,
                                "status": status,
                                "max_abs_diff": max_abs_diff,
                                "max_rel_diff": max_rel_diff,
                                "expected_checksum": expected_checksum,
                                "actual_checksum": actual_checksum,
                                "failure_message": failure_message,
                                "application_domain": kernel.get("application_domain", ""),
                                "workload_class": kernel.get("workload_class", ""),
                                "application": kernel.get("application", ""),
                            }
                        )
                        if status != "pass":
                            continue
                        measurements = timed_runs(
                            lambda runner=runner, inputs=inputs: runner(**inputs),
                            warmup,
                            repetitions,
                        )
                        for run_index, seconds in enumerate(measurements):
                            runtime_rows.append(
                                {
                                    "kernel": kernel["name"],
                                    "frontend": frontend,
                                    "loom_profile": profile_key,
                                    "backend": str(base_profile.get("backend", "triton")),
                                    "size": size,
                                    "dataset_id": dataset,
                                    "dataset_index": dataset_index,
                                    "dataset_seed": dataset_seed(config["seed"] + size, seed_offset),
                                    "dataset_seed_offset": seed_offset,
                                    "run_index": run_index,
                                    "seconds": seconds,
                                    "application_domain": kernel.get("application_domain", ""),
                                    "workload_class": kernel.get("workload_class", ""),
                                    "application": kernel.get("application", ""),
                                }
                            )
    write_csv(raw_dir / "frontend_runtime_measurements.csv", FRONTEND_RUNTIME_MEASUREMENT_FIELDS, runtime_rows)
    write_csv(raw_dir / "frontend_runtime_verification.csv", FRONTEND_RUNTIME_VERIFICATION_FIELDS, verification_rows)
    write_json(raw_dir / "frontend_runtime_failures.json", {"failures": failures})
    if failures:
        raise SystemExit("frontend runtime verification failed; see frontend/raw/frontend_runtime_failures.json")
    runtime_summary_rows = summarize_frontend_runtime(runtime_rows, kernels)
    write_csv(summary_dir / "frontend_runtime_summary.csv", FRONTEND_RUNTIME_SUMMARY_FIELDS, runtime_summary_rows)
    plot_frontend_runtime_summary(runtime_summary_rows, plot_dir)


def run_frontend_comparison(config: dict[str, Any], args: argparse.Namespace) -> None:
    ensure_loom_built()
    selected = set(args.kernel or [])
    kernels = [
        kernel
        for kernel in config["supported_kernels"]
        if not selected or kernel["name"] in selected
    ]
    raw_dir = frontend_results_dir(args.results_dir) / "raw"
    summary_dir = frontend_results_dir(args.results_dir) / "summaries"
    plot_dir = frontend_results_dir(args.results_dir) / "plots"
    reset_path(raw_dir)
    reset_path(summary_dir)
    reset_path(plot_dir)
    ensure_dir(raw_dir)
    ensure_dir(summary_dir)
    ensure_dir(plot_dir)
    work_dir = args.work_dir / "frontend_comparison"
    reset_path(work_dir)
    ensure_dir(work_dir)
    repetitions = args.compile_repetitions if args.compile_repetitions is not None else 3
    measurement_rows: list[dict[str, Any]] = []
    parity_rows: list[dict[str, Any]] = []
    compiled_dirs: dict[tuple[str, str], pathlib.Path] = {}
    for kernel in kernels:
        sources = {frontend: frontend_source(kernel, frontend) for frontend in FRONTEND_COMPARISON_FRONTENDS}
        for frontend, source in sources.items():
            if not source.exists():
                raise RuntimeError(f"{kernel['name']}: missing {frontend} source {source.relative_to(ROOT)}")
            for phase in ("front-ir", "full-compile"):
                for run_index in range(repetitions):
                    out_dir = work_dir / frontend / kernel["name"] / f"run_{run_index}"
                    start = time.perf_counter()
                    if phase == "front-ir":
                        run_frontend_command(source, kernel["entry"], frontend)
                    else:
                        run_frontend_compile(source, kernel["entry"], frontend, out_dir)
                        if run_index == 0:
                            compiled_dirs[(frontend, kernel["name"])] = out_dir
                    seconds = time.perf_counter() - start
                    measurement_rows.append(
                        {
                            "kernel": kernel["name"],
                            "frontend": frontend,
                            "phase": phase,
                            "run_index": run_index,
                            "seconds": seconds,
                            "status": "ok",
                            "application_domain": kernel.get("application_domain", ""),
                            "workload_class": kernel.get("workload_class", ""),
                            "application": kernel.get("application", ""),
                        }
                    )
        baseline_dir = compiled_dirs.get((FRONTEND_COMPARISON_BASELINE, kernel["name"]))
        for frontend in FRONTEND_COMPARISON_FRONTENDS:
            if frontend == FRONTEND_COMPARISON_BASELINE:
                continue
            frontend_dir = compiled_dirs.get((frontend, kernel["name"]))
            tensor_match = False
            plan_match = False
            if baseline_dir is not None and frontend_dir is not None:
                tensor_match = load_json(baseline_dir / "tensor_ir.json") == load_json(frontend_dir / "tensor_ir.json")
                plan_match = load_json(baseline_dir / "kernel_plan.json") == load_json(frontend_dir / "kernel_plan.json")
            parity_rows.append(
                {
                    "kernel": kernel["name"],
                    "frontend": frontend,
                    "tensor_ir_match": "yes" if tensor_match else "no",
                    "kernel_plan_match": "yes" if plan_match else "no",
                    "status": "ok" if tensor_match and plan_match else "mismatch",
                }
            )
    write_csv(raw_dir / "frontend_measurements.csv", FRONTEND_MEASUREMENT_FIELDS, measurement_rows)
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for row in measurement_rows:
        grouped.setdefault((row["kernel"], row["frontend"], row["phase"]), []).append(row)
    summary_rows: list[dict[str, Any]] = []
    for (kernel_name, frontend, phase), rows in sorted(grouped.items()):
        values = [float(row["seconds"]) for row in rows]
        kernel = next(item for item in kernels if item["name"] == kernel_name)
        summary_rows.append(
            {
                "kernel": kernel_name,
                "frontend": frontend,
                "phase": phase,
                "median_ms": statistics.median(values) * 1000.0,
                "runs": len(values),
                "application_domain": kernel.get("application_domain", ""),
                "workload_class": kernel.get("workload_class", ""),
                "application": kernel.get("application", ""),
            }
        )
    write_csv(summary_dir / "frontend_summary.csv", FRONTEND_SUMMARY_FIELDS, summary_rows)
    write_csv(summary_dir / "frontend_parity.csv", FRONTEND_PARITY_FIELDS, parity_rows)
    plot_frontend_comparison(summary_rows, plot_dir)
    if args.frontend_runtime:
        run_frontend_runtime_comparison(kernels, args, raw_dir, summary_dir, plot_dir)


class PythonRuntime:
    def __init__(
        self,
        implementation_key: str,
        module,
        runner: Callable[..., torch.Tensor],
        cache_dir: pathlib.Path | None = None,
    ):
        self.implementation_key = implementation_key
        self.module = module
        self.runner = runner
        self.cache_dir = cache_dir

    def __call__(self, **inputs: Any) -> torch.Tensor:
        if self.cache_dir is not None:
            os.environ["TRITON_CACHE_DIR"] = str(self.cache_dir)
        return self.runner(**inputs)

    def autotune_state(self) -> dict[str, Any]:
        state_fn = getattr(self.module, "__loom_autotune_state__", None)
        if callable(state_fn):
            return state_fn()
        state_fn = getattr(self.module, "autotune_state", None)
        if callable(state_fn):
            return state_fn()
        return {}


class LoomCudaRuntime:
    def __init__(
        self,
        implementation_key: str,
        kernel: dict[str, Any],
        manifest_path: pathlib.Path,
    ):
        self.implementation_key = implementation_key
        self.kernel = kernel
        self.manifest_path = manifest_path
        manifest = load_json(manifest_path)
        entries = manifest.get("entries", [])
        if len(entries) != 1:
            raise RuntimeError(f"expected exactly one generated CUDA entry in {manifest_path}")
        self.entry = entries[0]
        artifact_path = pathlib.Path(str(manifest["artifact_path"]))
        if not artifact_path.is_absolute() and not artifact_path.exists():
            artifact_path = manifest_path.parent / artifact_path
        self.lib = ctypes.CDLL(str(artifact_path))
        self.export = getattr(self.lib, str(self.entry["symbol_name"]))
        self.no_workspace_export = None
        no_workspace_symbol = self.entry.get("no_workspace_symbol")
        if no_workspace_symbol is not None:
            self.no_workspace_export = getattr(self.lib, str(no_workspace_symbol))
        self.workspace_size_fn = getattr(self.lib, str(self.entry["workspace_symbol"]))
        self.workspace_size_fn.argtypes = [ctypes.c_longlong]
        self.workspace_size_fn.restype = ctypes.c_size_t
        self.workspace_cache: dict[int, torch.Tensor] = {}
        self.workspace_arg_cache: dict[int, tuple[ctypes.c_void_p | None, ctypes.c_size_t]] = {}
        self._configure_export()

    def _configure_export(self) -> None:
        argtypes: list[Any] = []
        for param in self.entry.get("params", []):
            kind = str(param["kind"])
            if kind == "scalar-f32":
                argtypes.append(ctypes.c_float)
            elif kind == "tensor1-f32":
                argtypes.append(ctypes.c_void_p)
            else:
                raise ValueError(f"unsupported generated Loom CUDA param kind {kind}")
        argtypes.extend(
            [ctypes.c_longlong, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t]
        )
        self.export.argtypes = argtypes
        self.export.restype = ctypes.c_int
        if self.no_workspace_export is not None:
            fast_argtypes = list(argtypes[:-2])
            self.no_workspace_export.argtypes = fast_argtypes
            self.no_workspace_export.restype = ctypes.c_int

    def _workspace_for_size(self, size: int) -> tuple[ctypes.c_void_p | None, ctypes.c_size_t]:
        cached = self.workspace_arg_cache.get(size)
        if cached is not None:
            return cached
        workspace_size = int(self.workspace_size_fn(size))
        if workspace_size <= 0:
            result = (None, ctypes.c_size_t(0))
            self.workspace_arg_cache[size] = result
            return result
        workspace = self.workspace_cache.get(size)
        if workspace is None or workspace.numel() < workspace_size:
            workspace = torch.empty(workspace_size, device="cuda", dtype=torch.uint8)
            self.workspace_cache[size] = workspace
        result = (ctypes.c_void_p(workspace.data_ptr()), ctypes.c_size_t(workspace_size))
        self.workspace_arg_cache[size] = result
        return result

    def __call__(self, **inputs: Any) -> torch.Tensor:
        size = next(value.numel() for value in inputs.values() if isinstance(value, torch.Tensor))
        if self.entry["result_kind"] == "tensor":
            output_tensor_name = next(name for name in self.kernel["tensor_inputs"] if name in inputs)
            out = torch.empty_like(inputs[output_tensor_name])
        else:
            out = torch.empty((1,), device="cuda", dtype=torch.float32)
        args: list[Any] = []
        for param in self.entry.get("params", []):
            name = str(param["name"])
            kind = str(param["kind"])
            if kind == "scalar-f32":
                args.append(ctypes.c_float(float(inputs[name])))
            elif kind == "tensor1-f32":
                args.append(ctypes.c_void_p(inputs[name].data_ptr()))
            else:  # pragma: no cover - defensive
                raise ValueError(f"unsupported generated Loom CUDA param kind {kind}")
        if self.no_workspace_export is not None:
            args.extend(
                [
                    ctypes.c_longlong(size),
                    ctypes.c_void_p(out.data_ptr()),
                ]
            )
            error = self.no_workspace_export(*args)
            if error != 0:
                raise RuntimeError(f"generated Loom CUDA backend returned error code {error}")
            return out
        workspace_ptr, workspace_size = self._workspace_for_size(size)
        args.extend(
            [
                ctypes.c_longlong(size),
                ctypes.c_void_p(out.data_ptr()),
                workspace_ptr,
                workspace_size,
            ]
        )
        error = self.export(*args)
        if error != 0:
            raise RuntimeError(f"generated Loom CUDA backend returned error code {error}")
        return out

    def autotune_state(self) -> dict[str, Any]:
        return {}


def load_loom_runtime(
    kernel: dict[str, Any],
    profile: dict[str, Any],
    work_dir: pathlib.Path,
    autotune_config: str,
    optimizer_config: str,
) -> Callable[..., torch.Tensor]:
    implementation_key = profile["key"]
    out_dir = work_dir / "runtime" / implementation_key / kernel["name"]
    if out_dir.exists():
        shutil.rmtree(out_dir)
    cache_dir = work_dir / "runtime_cache" / implementation_key / kernel["name"]
    if cache_dir.exists():
        shutil.rmtree(cache_dir)
    ensure_dir(cache_dir)
    config = autotune_config if profile["autotuned"] else None
    loom_compile(
        EXPERIMENTS_ROOT / kernel["source_path"],
        kernel["entry"],
        kernel["input_kind"],
        out_dir,
        str(profile.get("backend", "triton")),
        list(profile["optimizations"]),
        optimizer_config=optimizer_config,
        autotune_config=config,
        cuda_platform=profile.get("cuda_platform"),
    )
    if str(profile.get("backend", "triton")) == "cuda":
        return LoomCudaRuntime(implementation_key, kernel, out_dir / "manifest.json")
    module = module_from_path(out_dir / f"{kernel['name']}_triton.py", f"{implementation_key}_{kernel['name']}")
    return PythonRuntime(implementation_key, module, getattr(module, kernel["entry"]), cache_dir=cache_dir)


def triton_module_path(kernel_name: str, variant: str) -> pathlib.Path:
    return SOURCE_ROOT / "triton" / variant / f"{kernel_name}.py"


def internal_triton_compile(module_path: pathlib.Path, kernel_name: str, probe_size: int, seed: int) -> float:
    kernel = next(item for item in load_config()["supported_kernels"] if item["name"] == kernel_name)
    inputs = make_kernel_inputs(kernel, probe_size, seed)
    start = time.perf_counter()
    module = module_from_path(module_path, f"triton_compile_{kernel_name}_{module_path.parent.name}")
    torch.cuda.synchronize()
    _ = module.run(**inputs)
    torch.cuda.synchronize()
    return time.perf_counter() - start


def internal_loom_compile_from_json(payload_path: pathlib.Path) -> float:
    payload = load_json(payload_path)
    source = pathlib.Path(str(payload["source"]))
    out_dir = pathlib.Path(str(payload["out_dir"]))
    kernel_name = str(payload["kernel_name"])
    entry = str(payload["entry"])
    backend = str(payload.get("backend", "triton"))
    kernel = next(item for item in load_config()["supported_kernels"] if item["name"] == kernel_name)
    start = time.perf_counter()
    loom_compile(
        source,
        entry,
        str(payload["input_kind"]),
        out_dir,
        backend,
        [str(item) for item in payload.get("optimizations", [])],
        optimizer_config=payload.get("optimizer_config"),
        autotune_config=payload.get("autotune_config"),
        cuda_platform=payload.get("cuda_platform"),
    )
    if backend == "cuda":
        return time.perf_counter() - start
    inputs = make_kernel_inputs(
        kernel,
        int(payload["probe_size"]),
        int(payload["seed"]),
    )
    module = module_from_path(out_dir / f"{kernel_name}_triton.py", str(payload["module_name"]))
    torch.cuda.synchronize()
    _ = getattr(module, entry)(**inputs)
    torch.cuda.synchronize()
    return time.perf_counter() - start


def measure_triton_compile(
    kernel: dict[str, Any],
    implementation_key: str,
    repetitions: int,
    work_dir: pathlib.Path,
    probe_size: int,
    seed: int,
) -> list[float]:
    variant_dir = triton_variant_dir(implementation_key)
    module_path = triton_module_path(logical_kernel_name(kernel), variant_dir)
    measurements: list[float] = []
    for index in range(repetitions):
        cache_dir = work_dir / "compile" / f"{implementation_key}_cache" / kernel["name"] / f"run_{index}"
        ensure_dir(cache_dir)
        env = os.environ.copy()
        env["TRITON_CACHE_DIR"] = str(cache_dir)
        result = subprocess.run(
            [
                sys.executable,
                str(pathlib.Path(__file__).resolve()),
                "--internal-triton-compile",
                str(module_path),
                kernel["name"],
                str(probe_size),
                str(seed + index),
            ],
            cwd=ROOT,
            env=env,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(result.stderr.strip())
        measurements.append(float(result.stdout.strip()))
    return measurements


def load_triton_runtime(kernel: dict[str, Any], implementation_key: str, work_dir: pathlib.Path) -> PythonRuntime:
    variant_dir = triton_variant_dir(implementation_key)
    cache_dir = work_dir / "runtime_cache" / implementation_key / kernel["name"]
    if cache_dir.exists():
        shutil.rmtree(cache_dir)
    ensure_dir(cache_dir)
    module = module_from_path(
        triton_module_path(logical_kernel_name(kernel), variant_dir),
        f"{implementation_key}_{kernel['name']}",
    )
    return PythonRuntime(
        implementation_key,
        module,
        module.run,
        cache_dir=cache_dir,
    )


def cuda_source_path(kernel_name: str, variant: str) -> pathlib.Path:
    return SOURCE_ROOT / "cuda" / variant / f"{kernel_name}.cu"


def cuda_export_name(kernel_name: str) -> str:
    return f"{kernel_name}_run"


def cuda_workspace_symbol(kernel_name: str) -> str:
    return f"{kernel_name}_workspace_size"


def compile_cuda_shared(source: pathlib.Path, output: pathlib.Path) -> None:
    ensure_dir(output.parent)
    major, minor = torch.cuda.get_device_capability(0)
    arch = f"sm_{major}{minor}"
    result = subprocess.run(
        [
            "nvcc",
            "-O3",
            "-std=c++17",
            "-arch",
            arch,
            "--shared",
            "-Xcompiler",
            "-fPIC",
            "-I",
            str(source.parent),
            "-o",
            str(output),
            str(source),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip())


def measure_cuda_compile(
    kernel: dict[str, Any],
    variant: str,
    repetitions: int,
    work_dir: pathlib.Path,
    probe_size: int,
    seed: int,
) -> list[float]:
    source = cuda_source_path(logical_kernel_name(kernel), variant)
    measurements: list[float] = []
    for index in range(repetitions):
        output = work_dir / "compile" / f"cuda_{variant}" / kernel["name"] / f"run_{index}.so"
        if output.exists():
            output.unlink()
        inputs = make_kernel_inputs(kernel, probe_size, seed + index)
        start = time.perf_counter()
        compile_cuda_shared(source, output)
        runtime = CudaRuntime(f"cuda_{variant}", kernel, output)
        torch.cuda.synchronize()
        _ = runtime(**inputs)
        torch.cuda.synchronize()
        measurements.append(time.perf_counter() - start)
    return measurements


class CudaRuntime:
    def __init__(self, implementation_key: str, kernel: dict[str, Any], library_path: pathlib.Path):
        self.implementation_key = implementation_key
        self.kernel = kernel
        self.logical_name = logical_kernel_name(kernel)
        self.lib = ctypes.CDLL(str(library_path))
        self.export = getattr(self.lib, cuda_export_name(self.logical_name))
        self.workspace_size_fn = None
        self.workspace_cache: dict[int, torch.Tensor] = {}
        self.workspace_arg_cache: dict[int, tuple[ctypes.c_void_p | None, ctypes.c_size_t]] = {}
        self.variant = ALL_IMPLEMENTATION_MAP[implementation_key]["variant"]
        self._configure_export()

    def _configure_export(self) -> None:
        kernel_name = self.logical_name
        optimized = self.variant == "optimized"
        if kernel_name == "saxpy":
            argtypes = [
                ctypes.c_float,
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_longlong,
            ]
        elif kernel_name == "relu":
            argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_longlong]
        elif kernel_name in {"soft_threshold", "inventory_penalty_sum"}:
            if kernel_name == "soft_threshold":
                argtypes = [ctypes.c_float, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_longlong]
            else:
                if optimized:
                    self.workspace_size_fn = getattr(self.lib, cuda_workspace_symbol(kernel_name))
                    self.workspace_size_fn.argtypes = [ctypes.c_longlong]
                    self.workspace_size_fn.restype = ctypes.c_size_t
                    argtypes = [
                        ctypes.c_float,
                        ctypes.c_void_p,
                        ctypes.c_void_p,
                        ctypes.c_longlong,
                        ctypes.c_void_p,
                        ctypes.c_size_t,
                    ]
                else:
                    argtypes = [ctypes.c_float, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_longlong]
        elif kernel_name in {
            "l2_norm_sq",
            "dot",
            "mse_sum",
            "huber_sum",
            "weighted_dot",
            "mixed_biased_l2_norm",
            "mixed_weighted_affine_dot",
            "clipped_huber_sum",
            "ratio_weighted_sum",
            "piecewise_weighted_dot",
            "affine_score_reduce",
            "signal_clip_reduce",
        }:
            if optimized:
                self.workspace_size_fn = getattr(self.lib, cuda_workspace_symbol(kernel_name))
                self.workspace_size_fn.argtypes = [ctypes.c_longlong]
                self.workspace_size_fn.restype = ctypes.c_size_t
            if kernel_name == "l2_norm_sq":
                argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_longlong]
                if optimized:
                    argtypes = argtypes + [ctypes.c_void_p, ctypes.c_size_t]
            elif kernel_name == "dot" or kernel_name == "mse_sum":
                argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_longlong]
                if optimized:
                    argtypes = argtypes + [ctypes.c_void_p, ctypes.c_size_t]
            elif kernel_name == "weighted_dot":
                argtypes = [ctypes.c_float, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_longlong]
                if optimized:
                    argtypes = argtypes + [ctypes.c_void_p, ctypes.c_size_t]
            elif kernel_name == "mixed_weighted_affine_dot":
                argtypes = [
                    ctypes.c_float,
                    ctypes.c_float,
                    ctypes.c_float,
                    ctypes.c_void_p,
                    ctypes.c_void_p,
                    ctypes.c_void_p,
                    ctypes.c_longlong,
                ]
                if optimized:
                    argtypes = argtypes + [ctypes.c_void_p, ctypes.c_size_t]
            elif kernel_name == "mixed_biased_l2_norm":
                argtypes = [
                    ctypes.c_float,
                    ctypes.c_float,
                    ctypes.c_void_p,
                    ctypes.c_void_p,
                    ctypes.c_longlong,
                ]
                if optimized:
                    argtypes = argtypes + [ctypes.c_void_p, ctypes.c_size_t]
            elif kernel_name == "ratio_weighted_sum":
                argtypes = [ctypes.c_float, ctypes.c_float, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_longlong]
                if optimized:
                    argtypes = argtypes + [ctypes.c_void_p, ctypes.c_size_t]
            elif kernel_name == "piecewise_weighted_dot":
                argtypes = [
                    ctypes.c_float,
                    ctypes.c_float,
                    ctypes.c_void_p,
                    ctypes.c_void_p,
                    ctypes.c_void_p,
                    ctypes.c_longlong,
                ]
                if optimized:
                    argtypes = argtypes + [ctypes.c_void_p, ctypes.c_size_t]
            elif kernel_name == "affine_score_reduce":
                argtypes = [
                    ctypes.c_float,
                    ctypes.c_float,
                    ctypes.c_float,
                    ctypes.c_void_p,
                    ctypes.c_void_p,
                    ctypes.c_longlong,
                ]
                if optimized:
                    argtypes = argtypes + [ctypes.c_void_p, ctypes.c_size_t]
            elif kernel_name == "signal_clip_reduce":
                argtypes = [
                    ctypes.c_float,
                    ctypes.c_float,
                    ctypes.c_float,
                    ctypes.c_void_p,
                    ctypes.c_void_p,
                    ctypes.c_void_p,
                    ctypes.c_longlong,
                ]
                if optimized:
                    argtypes = argtypes + [ctypes.c_void_p, ctypes.c_size_t]
            else:
                if kernel_name == "huber_sum":
                    argtypes = [ctypes.c_float, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_longlong]
                else:
                    argtypes = [ctypes.c_float, ctypes.c_float, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_longlong]
                if optimized:
                    argtypes = argtypes + [ctypes.c_void_p, ctypes.c_size_t]
        elif kernel_name in {"book_imbalance", "quote_filter", "scaled_book_signal", "mixed_book_signal"}:
            if kernel_name == "scaled_book_signal":
                argtypes = [
                    ctypes.c_float,
                    ctypes.c_float,
                    ctypes.c_void_p,
                    ctypes.c_void_p,
                    ctypes.c_void_p,
                    ctypes.c_longlong,
                ]
            elif kernel_name == "mixed_book_signal":
                argtypes = [
                    ctypes.c_float,
                    ctypes.c_float,
                    ctypes.c_float,
                    ctypes.c_void_p,
                    ctypes.c_void_p,
                    ctypes.c_void_p,
                    ctypes.c_longlong,
                ]
            else:
                argtypes = [
                    ctypes.c_float,
                    ctypes.c_void_p,
                    ctypes.c_void_p,
                    ctypes.c_void_p,
                    ctypes.c_longlong,
                ]
        elif kernel_name == "affine_clamp":
            argtypes = [
                ctypes.c_float,
                ctypes.c_float,
                ctypes.c_float,
                ctypes.c_float,
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_longlong,
            ]
        else:
            raise ValueError(f"unknown kernel {kernel_name}")
        self.export.argtypes = argtypes
        self.export.restype = ctypes.c_int

    def _workspace_for_size(self, size: int) -> tuple[ctypes.c_void_p | None, ctypes.c_size_t]:
        if self.workspace_size_fn is None:
            return None, ctypes.c_size_t(0)
        cached = self.workspace_arg_cache.get(size)
        if cached is not None:
            return cached
        workspace_size = int(self.workspace_size_fn(size))
        if workspace_size <= 0:
            result = (None, ctypes.c_size_t(0))
            self.workspace_arg_cache[size] = result
            return result
        workspace = self.workspace_cache.get(size)
        if workspace is None or workspace.numel() < workspace_size:
            workspace = torch.empty(workspace_size, device="cuda", dtype=torch.uint8)
            self.workspace_cache[size] = workspace
        result = (ctypes.c_void_p(workspace.data_ptr()), ctypes.c_size_t(workspace_size))
        self.workspace_arg_cache[size] = result
        return result

    def __call__(self, **inputs: Any) -> torch.Tensor:
        size = next(value.numel() for value in inputs.values() if isinstance(value, torch.Tensor))
        if self.kernel["result_kind"] == "tensor":
            output_tensor_name = next(name for name in self.kernel["tensor_inputs"] if name in inputs)
            out = torch.empty_like(inputs[output_tensor_name])
        else:
            out = torch.empty((1,), device="cuda", dtype=torch.float32)
        workspace_ptr, workspace_size = self._workspace_for_size(size)
        name = self.logical_name
        optimized = self.variant == "optimized"
        if name == "saxpy":
            error = self.export(
                ctypes.c_float(float(inputs["a"])),
                ctypes.c_void_p(inputs["x"].data_ptr()),
                ctypes.c_void_p(inputs["y"].data_ptr()),
                ctypes.c_void_p(out.data_ptr()),
                ctypes.c_longlong(size),
            )
        elif name == "relu":
            error = self.export(
                ctypes.c_void_p(inputs["x"].data_ptr()),
                ctypes.c_void_p(out.data_ptr()),
                ctypes.c_longlong(size),
            )
        elif name == "soft_threshold":
            error = self.export(
                ctypes.c_float(float(inputs["threshold"])),
                ctypes.c_void_p(inputs["x"].data_ptr()),
                ctypes.c_void_p(out.data_ptr()),
                ctypes.c_longlong(size),
            )
        elif name == "affine_clamp":
            error = self.export(
                ctypes.c_float(float(inputs["scale"])),
                ctypes.c_float(float(inputs["bias"])),
                ctypes.c_float(float(inputs["lo"])),
                ctypes.c_float(float(inputs["hi"])),
                ctypes.c_void_p(inputs["x"].data_ptr()),
                ctypes.c_void_p(out.data_ptr()),
                ctypes.c_longlong(size),
            )
        elif name == "book_imbalance":
            error = self.export(
                ctypes.c_float(float(inputs["epsilon"])),
                ctypes.c_void_p(inputs["bid"].data_ptr()),
                ctypes.c_void_p(inputs["ask"].data_ptr()),
                ctypes.c_void_p(out.data_ptr()),
                ctypes.c_longlong(size),
            )
        elif name == "scaled_book_signal":
            error = self.export(
                ctypes.c_float(float(inputs["scale"])),
                ctypes.c_float(float(inputs["epsilon"])),
                ctypes.c_void_p(inputs["bid"].data_ptr()),
                ctypes.c_void_p(inputs["ask"].data_ptr()),
                ctypes.c_void_p(out.data_ptr()),
                ctypes.c_longlong(size),
            )
        elif name == "mixed_book_signal":
            error = self.export(
                ctypes.c_float(float(inputs["scale"])),
                ctypes.c_float(float(inputs["epsilon"])),
                ctypes.c_float(float(inputs["threshold"])),
                ctypes.c_void_p(inputs["bid"].data_ptr()),
                ctypes.c_void_p(inputs["ask"].data_ptr()),
                ctypes.c_void_p(out.data_ptr()),
                ctypes.c_longlong(size),
            )
        elif name == "quote_filter":
            error = self.export(
                ctypes.c_float(float(inputs["threshold"])),
                ctypes.c_void_p(inputs["bid"].data_ptr()),
                ctypes.c_void_p(inputs["ask"].data_ptr()),
                ctypes.c_void_p(out.data_ptr()),
                ctypes.c_longlong(size),
            )
        elif name == "inventory_penalty_sum":
            if optimized:
                error = self.export(
                    ctypes.c_float(float(inputs["target"])),
                    ctypes.c_void_p(inputs["pos"].data_ptr()),
                    ctypes.c_void_p(out.data_ptr()),
                    ctypes.c_longlong(size),
                    workspace_ptr,
                    workspace_size,
                )
            else:
                error = self.export(
                    ctypes.c_float(float(inputs["target"])),
                    ctypes.c_void_p(inputs["pos"].data_ptr()),
                    ctypes.c_void_p(out.data_ptr()),
                    ctypes.c_longlong(size),
                )
        elif name == "l2_norm_sq":
            if optimized:
                error = self.export(
                    ctypes.c_void_p(inputs["x"].data_ptr()),
                    ctypes.c_void_p(out.data_ptr()),
                    ctypes.c_longlong(size),
                    workspace_ptr,
                    workspace_size,
                )
            else:
                error = self.export(
                    ctypes.c_void_p(inputs["x"].data_ptr()),
                    ctypes.c_void_p(out.data_ptr()),
                    ctypes.c_longlong(size),
                )
        elif name in {"dot", "mse_sum"}:
            if optimized:
                error = self.export(
                    ctypes.c_void_p(inputs["x"].data_ptr()),
                    ctypes.c_void_p(inputs["y"].data_ptr()),
                    ctypes.c_void_p(out.data_ptr()),
                    ctypes.c_longlong(size),
                    workspace_ptr,
                    workspace_size,
                )
            else:
                error = self.export(
                    ctypes.c_void_p(inputs["x"].data_ptr()),
                    ctypes.c_void_p(inputs["y"].data_ptr()),
                    ctypes.c_void_p(out.data_ptr()),
                    ctypes.c_longlong(size),
                )
        elif name == "weighted_dot":
            if optimized:
                error = self.export(
                    ctypes.c_float(float(inputs["weight"])),
                    ctypes.c_void_p(inputs["x"].data_ptr()),
                    ctypes.c_void_p(inputs["y"].data_ptr()),
                    ctypes.c_void_p(out.data_ptr()),
                    ctypes.c_longlong(size),
                    workspace_ptr,
                    workspace_size,
                )
            else:
                error = self.export(
                    ctypes.c_float(float(inputs["weight"])),
                    ctypes.c_void_p(inputs["x"].data_ptr()),
                    ctypes.c_void_p(inputs["y"].data_ptr()),
                    ctypes.c_void_p(out.data_ptr()),
                    ctypes.c_longlong(size),
                )
        elif name == "mixed_weighted_affine_dot":
            if optimized:
                error = self.export(
                    ctypes.c_float(float(inputs["weight"])),
                    ctypes.c_float(float(inputs["scale"])),
                    ctypes.c_float(float(inputs["bias"])),
                    ctypes.c_void_p(inputs["x"].data_ptr()),
                    ctypes.c_void_p(inputs["y"].data_ptr()),
                    ctypes.c_void_p(out.data_ptr()),
                    ctypes.c_longlong(size),
                    workspace_ptr,
                    workspace_size,
                )
            else:
                error = self.export(
                    ctypes.c_float(float(inputs["weight"])),
                    ctypes.c_float(float(inputs["scale"])),
                    ctypes.c_float(float(inputs["bias"])),
                    ctypes.c_void_p(inputs["x"].data_ptr()),
                    ctypes.c_void_p(inputs["y"].data_ptr()),
                    ctypes.c_void_p(out.data_ptr()),
                    ctypes.c_longlong(size),
                )
        elif name == "mixed_biased_l2_norm":
            if optimized:
                error = self.export(
                    ctypes.c_float(float(inputs["scale"])),
                    ctypes.c_float(float(inputs["bias"])),
                    ctypes.c_void_p(inputs["x"].data_ptr()),
                    ctypes.c_void_p(out.data_ptr()),
                    ctypes.c_longlong(size),
                    workspace_ptr,
                    workspace_size,
                )
            else:
                error = self.export(
                    ctypes.c_float(float(inputs["scale"])),
                    ctypes.c_float(float(inputs["bias"])),
                    ctypes.c_void_p(inputs["x"].data_ptr()),
                    ctypes.c_void_p(out.data_ptr()),
                    ctypes.c_longlong(size),
                )
        elif name == "ratio_weighted_sum":
            if optimized:
                error = self.export(
                    ctypes.c_float(float(inputs["scale"])),
                    ctypes.c_float(float(inputs["epsilon"])),
                    ctypes.c_void_p(inputs["x"].data_ptr()),
                    ctypes.c_void_p(inputs["y"].data_ptr()),
                    ctypes.c_void_p(out.data_ptr()),
                    ctypes.c_longlong(size),
                    workspace_ptr,
                    workspace_size,
                )
            else:
                error = self.export(
                    ctypes.c_float(float(inputs["scale"])),
                    ctypes.c_float(float(inputs["epsilon"])),
                    ctypes.c_void_p(inputs["x"].data_ptr()),
                    ctypes.c_void_p(inputs["y"].data_ptr()),
                    ctypes.c_void_p(out.data_ptr()),
                    ctypes.c_longlong(size),
                )
        elif name == "piecewise_weighted_dot":
            if optimized:
                error = self.export(
                    ctypes.c_float(float(inputs["weight_pos"])),
                    ctypes.c_float(float(inputs["weight_neg"])),
                    ctypes.c_void_p(inputs["x"].data_ptr()),
                    ctypes.c_void_p(inputs["y"].data_ptr()),
                    ctypes.c_void_p(out.data_ptr()),
                    ctypes.c_longlong(size),
                    workspace_ptr,
                    workspace_size,
                )
            else:
                error = self.export(
                    ctypes.c_float(float(inputs["weight_pos"])),
                    ctypes.c_float(float(inputs["weight_neg"])),
                    ctypes.c_void_p(inputs["x"].data_ptr()),
                    ctypes.c_void_p(inputs["y"].data_ptr()),
                    ctypes.c_void_p(out.data_ptr()),
                    ctypes.c_longlong(size),
                )
        elif name == "affine_score_reduce":
            if optimized:
                error = self.export(
                    ctypes.c_float(float(inputs["scale"])),
                    ctypes.c_float(float(inputs["bias"])),
                    ctypes.c_float(float(inputs["threshold"])),
                    ctypes.c_void_p(inputs["x"].data_ptr()),
                    ctypes.c_void_p(out.data_ptr()),
                    ctypes.c_longlong(size),
                    workspace_ptr,
                    workspace_size,
                )
            else:
                error = self.export(
                    ctypes.c_float(float(inputs["scale"])),
                    ctypes.c_float(float(inputs["bias"])),
                    ctypes.c_float(float(inputs["threshold"])),
                    ctypes.c_void_p(inputs["x"].data_ptr()),
                    ctypes.c_void_p(out.data_ptr()),
                    ctypes.c_longlong(size),
                )
        elif name == "signal_clip_reduce":
            if optimized:
                error = self.export(
                    ctypes.c_float(float(inputs["scale"])),
                    ctypes.c_float(float(inputs["epsilon"])),
                    ctypes.c_float(float(inputs["clip"])),
                    ctypes.c_void_p(inputs["bid"].data_ptr()),
                    ctypes.c_void_p(inputs["ask"].data_ptr()),
                    ctypes.c_void_p(out.data_ptr()),
                    ctypes.c_longlong(size),
                    workspace_ptr,
                    workspace_size,
                )
            else:
                error = self.export(
                    ctypes.c_float(float(inputs["scale"])),
                    ctypes.c_float(float(inputs["epsilon"])),
                    ctypes.c_float(float(inputs["clip"])),
                    ctypes.c_void_p(inputs["bid"].data_ptr()),
                    ctypes.c_void_p(inputs["ask"].data_ptr()),
                    ctypes.c_void_p(out.data_ptr()),
                    ctypes.c_longlong(size),
                )
        elif name == "huber_sum":
            if optimized:
                error = self.export(
                    ctypes.c_float(float(inputs["delta"])),
                    ctypes.c_void_p(inputs["x"].data_ptr()),
                    ctypes.c_void_p(inputs["y"].data_ptr()),
                    ctypes.c_void_p(out.data_ptr()),
                    ctypes.c_longlong(size),
                    workspace_ptr,
                    workspace_size,
                )
            else:
                error = self.export(
                    ctypes.c_float(float(inputs["delta"])),
                    ctypes.c_void_p(inputs["x"].data_ptr()),
                    ctypes.c_void_p(inputs["y"].data_ptr()),
                    ctypes.c_void_p(out.data_ptr()),
                    ctypes.c_longlong(size),
                )
        elif name == "clipped_huber_sum":
            if optimized:
                error = self.export(
                    ctypes.c_float(float(inputs["delta"])),
                    ctypes.c_float(float(inputs["cap"])),
                    ctypes.c_void_p(inputs["x"].data_ptr()),
                    ctypes.c_void_p(inputs["y"].data_ptr()),
                    ctypes.c_void_p(out.data_ptr()),
                    ctypes.c_longlong(size),
                    workspace_ptr,
                    workspace_size,
                )
            else:
                error = self.export(
                    ctypes.c_float(float(inputs["delta"])),
                    ctypes.c_float(float(inputs["cap"])),
                    ctypes.c_void_p(inputs["x"].data_ptr()),
                    ctypes.c_void_p(inputs["y"].data_ptr()),
                    ctypes.c_void_p(out.data_ptr()),
                    ctypes.c_longlong(size),
                )
        else:
            raise ValueError(f"unknown kernel {name}")
        if error != 0:
            raise RuntimeError(f"CUDA baseline returned error code {error}")
        return out

    def autotune_state(self) -> dict[str, Any]:
        return {}


def load_cuda_runtime(kernel: dict[str, Any], implementation_key: str, work_dir: pathlib.Path) -> CudaRuntime:
    variant = ALL_IMPLEMENTATION_MAP[implementation_key]["variant"]
    assert isinstance(variant, str)
    output = work_dir / "runtime" / implementation_key / kernel["name"] / f"{kernel['name']}.so"
    if output.exists():
        output.unlink()
    compile_cuda_shared(cuda_source_path(logical_kernel_name(kernel), variant), output)
    return CudaRuntime(implementation_key, kernel, output)


def command_output(command: list[str]) -> tuple[int, str]:
    result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
    return result.returncode, result.stdout.strip() if result.stdout else result.stderr.strip()


def collect_environment_info() -> dict[str, Any]:
    devices = []
    for index in range(torch.cuda.device_count()):
        properties = torch.cuda.get_device_properties(index)
        devices.append(
            {
                "index": index,
                "name": properties.name,
                "total_memory_bytes": properties.total_memory,
                "compute_capability": f"{properties.major}.{properties.minor}",
                "multi_processor_count": properties.multi_processor_count,
            }
        )
    query_fields = "name,driver_version,compute_cap,pci.bus_id,memory.total"
    query_rc, query_text = command_output(
        ["nvidia-smi", f"--query-gpu={query_fields}", "--format=csv,noheader"]
    )
    full_rc, full_text = command_output(["nvidia-smi"])
    info = {
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "python_version": sys.version,
        "cuda_device_count": torch.cuda.device_count(),
        "devices": devices,
        "nvidia_smi_query_returncode": query_rc,
        "nvidia_smi_query": query_text.splitlines() if query_text else [],
        "nvidia_smi_returncode": full_rc,
        "nvidia_smi_text": full_text,
    }
    try:
        import triton

        info["triton_version"] = getattr(triton, "__version__", "unknown")
    except Exception as exc:  # pragma: no cover - defensive
        info["triton_version"] = f"unavailable: {exc}"
    return info


def dump_environment_info(results_dir: pathlib.Path) -> dict[str, Any]:
    info = collect_environment_info()
    write_json(results_dir / "raw" / "environment.json", info)
    write_text(results_dir / "raw" / "environment.txt", info.get("nvidia_smi_text", "") + "\n")
    return info


def implementation_execution_order(
    implementation_keys: list[str], kernel: str, phase: str, size: int, seed: int
) -> list[str]:
    stable_seed = int(
        hashlib.sha256(f"{kernel}:{phase}:{size}:{seed}".encode("utf-8")).hexdigest()[:16],
        16,
    )
    order = list(implementation_keys)
    random.Random(stable_seed).shuffle(order)
    return order


def audit_workloads(config: dict[str, Any], results_dir: pathlib.Path) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []
    catalog = load_json(SOURCE_ROOT / "data" / "common" / "workloads.json")
    supported_catalog = {item["name"]: item for item in catalog.get("supported_kernels", [])}
    capability_catalog = {item["name"]: item for item in catalog.get("capability_cases", [])}
    supported_names = {kernel["name"] for kernel in config["supported_kernels"]}
    capability_names = {case["name"] for case in config["capability_cases"]}

    if supported_names != set(supported_catalog):
        missing = sorted(supported_names - set(supported_catalog))
        extra = sorted(set(supported_catalog) - supported_names)
        if missing:
            errors.append(f"common catalog missing supported kernels: {', '.join(missing)}")
        if extra:
            warnings.append(f"common catalog has extra supported kernels: {', '.join(extra)}")
    if capability_names != set(capability_catalog):
        missing = sorted(capability_names - set(capability_catalog))
        extra = sorted(set(capability_catalog) - capability_names)
        if missing:
            errors.append(f"common catalog missing capability cases: {', '.join(missing)}")
        if extra:
            warnings.append(f"common catalog has extra capability cases: {', '.join(extra)}")

    for kernel in config["supported_kernels"]:
        if logical_kernel_name(kernel) not in REFERENCE_KERNELS:
            errors.append(f"{kernel['name']}: missing PyTorch reference implementation")
        source_path = EXPERIMENTS_ROOT / kernel["source_path"]
        if not source_path.exists():
            errors.append(f"{kernel['name']}: missing source path {source_path.relative_to(ROOT)}")
        catalog_entry = supported_catalog.get(kernel["name"])
        if catalog_entry is None:
            continue
        for field in (
            "baseline_name",
            "entry",
            "tensor_inputs",
            "result_kind",
            "application_domain",
            "workload_class",
        ):
            if catalog_entry.get(field) != kernel.get(field):
                errors.append(f"{kernel['name']}: drift in field '{field}' between workload config and common catalog")
        logical_name = logical_kernel_name(kernel)
        for variant in TRITON_VARIANT_DIRS:
            path = SOURCE_ROOT / "triton" / variant / f"{logical_name}.py"
            if not path.exists():
                errors.append(f"{kernel['name']}: missing Triton baseline {path.relative_to(ROOT)}")
        for variant in CUDA_VARIANT_DIRS:
            path = SOURCE_ROOT / "cuda" / variant / f"{logical_name}.cu"
            if not path.exists():
                errors.append(f"{kernel['name']}: missing CUDA baseline {path.relative_to(ROOT)}")

    for case in config["capability_cases"]:
        catalog_entry = capability_catalog.get(case["name"])
        if catalog_entry is None:
            continue
        for field in ("entry", "description"):
            if catalog_entry.get(field) != case.get(field):
                warnings.append(f"{case['name']}: capability drift in field '{field}'")

    report = {
        "status": "passed" if not errors else "failed",
        "supported_kernel_count": len(config["supported_kernels"]),
        "capability_case_count": len(config["capability_cases"]),
        "errors": errors,
        "warnings": warnings,
    }
    write_json(results_dir / "raw" / "audit_report.json", report)
    if errors:
        raise SystemExit("experiment audit failed; see shared/raw/audit_report.json")
    return report


def audit_cuda_backend_generalizability(results_dir: pathlib.Path) -> dict[str, Any]:
    backend_paths = [
        ROOT / "src" / "loom_backend_cuda" / "cuda_plan.ml",
        ROOT / "src" / "loom_backend_cuda" / "cuda_backend.ml",
    ]
    entry_name_lines: list[dict[str, Any]] = []
    benchmark_name_lines: list[dict[str, Any]] = []
    baseline_names = set(CURRENT_NON_HELD_OUT_KERNELS) | {
        "ratio_weighted_sum",
        "piecewise_weighted_dot",
        "affine_score_reduce",
    }
    for path in backend_paths:
        if not path.exists():
            continue
        for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            stripped = line.strip()
            if "entry_name" in stripped and ("List.mem" in stripped or "String.equal" in stripped):
                entry_name_lines.append(
                    {
                        "path": str(path.relative_to(ROOT)),
                        "line": lineno,
                        "code": stripped,
                    }
                )
            if any(f'"{name}"' in stripped for name in baseline_names):
                benchmark_name_lines.append(
                    {
                        "path": str(path.relative_to(ROOT)),
                        "line": lineno,
                        "code": stripped,
                    }
                )
    report = {
        "status": "warning" if entry_name_lines or benchmark_name_lines else "passed",
        "held_out_generalization_kernels": list(HELD_OUT_GENERALIZATION_KERNELS),
        "entry_name_policy_sites": entry_name_lines,
        "benchmark_name_literal_sites": benchmark_name_lines,
        "diagnosis": (
            "CUDA backend still contains name-sensitive policy sites; held-out workloads "
            "exercise equivalent expression shapes to catch brittle optimization dispatch."
            if entry_name_lines or benchmark_name_lines
            else "No obvious name-sensitive CUDA backend policy sites were found."
        ),
    }
    write_json(results_dir / "raw" / "generalizability_audit.json", report)
    return report


def row_with_kernel_metadata(kernel: dict[str, Any], row: dict[str, Any]) -> dict[str, Any]:
    return {
        **row,
        "application_domain": kernel["application_domain"],
        "workload_class": kernel["workload_class"],
        "application": kernel["application"],
    }


def row_with_implementation_metadata(row: dict[str, Any], implementation_key: str) -> dict[str, Any]:
    spec = ALL_IMPLEMENTATION_MAP[implementation_key]
    return {
        **row,
        "implementation_label": spec["label"],
        "implementation_kind": spec["kind"],
        "autotuned": "yes" if spec.get("autotuned", False) else "no",
        "optimization_flags": ",".join(spec.get("optimizations", [])),
    }


def record_dataset_manifest(
    manifest: dict[str, dict[str, Any]],
    kernel: dict[str, Any],
    phase: str,
    size: int,
    base_seed: int,
    seed_offset: int,
    inputs: dict[str, Any],
) -> None:
    entry = dataset_manifest_entry(kernel, phase, size, base_seed, seed_offset, inputs)
    manifest[entry["dataset_id"]] = entry


def record_dataset_manifest_streaming(
    seen_dataset_ids: set[str],
    writers: RawResultWriters,
    kernel: dict[str, Any],
    phase: str,
    size: int,
    base_seed: int,
    seed_offset: int,
    inputs: dict[str, Any],
) -> None:
    entry = dataset_manifest_entry(kernel, phase, size, base_seed, seed_offset, inputs)
    if entry["dataset_id"] in seen_dataset_ids:
        return
    seen_dataset_ids.add(entry["dataset_id"])
    writers.dataset_manifest_jsonl.append(entry)


def append_completed_unit(
    writers: RawResultWriters,
    stage: str,
    kernel: str,
    implementation: str,
    size: int | None,
    dataset_id_value: str,
    dataset_seed_value: int | str,
    dataset_seed_offset: int | str,
    status: str,
    detail: str = "",
) -> None:
    writers.completed_units.append_row(
        {
            "stage": stage,
            "kernel": kernel,
            "implementation": implementation,
            "size": "" if size is None else size,
            "dataset_id": dataset_id_value,
            "dataset_seed": dataset_seed_value,
            "dataset_seed_offset": dataset_seed_offset,
            "status": status,
            "detail": detail,
        }
    )
    writers.flush()


def finalize_dataset_manifest(results_dir: pathlib.Path) -> list[dict[str, Any]]:
    manifest_path = shared_results_dir(results_dir) / "raw" / "dataset_manifest.jsonl"
    datasets = read_jsonl(manifest_path)
    deduped: dict[str, dict[str, Any]] = {}
    for item in datasets:
        deduped[item["dataset_id"]] = item
    rows = sorted(deduped.values(), key=lambda item: item["dataset_id"])
    write_json(shared_results_dir(results_dir) / "raw" / "dataset_manifest.json", {"datasets": rows})
    return rows


def run_capability_checks(
    config: dict[str, Any],
    results_dir: pathlib.Path,
    work_dir: pathlib.Path,
    plot_results_dir: pathlib.Path | None = None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    out_base = work_dir / "capabilities"
    for case in config["capability_cases"]:
        out_dir = out_base / case["name"]
        if out_dir.exists():
            shutil.rmtree(out_dir)
        source = EXPERIMENTS_ROOT / case["source_path"]
        result = subprocess.run(
            [
                str(LOOM_BIN),
                "compile",
                str(source),
                "--input-kind",
                case.get("input_kind", "ocaml"),
                "--entry",
                case["entry"],
                "--target",
                "triton",
                "--out",
                str(out_dir),
                "--emit",
                "all",
            ],
            cwd=ROOT,
            capture_output=True,
            text=True,
        )
        observed = "supported" if result.returncode == 0 else "unsupported"
        diagnostic = ""
        if result.stderr:
            diagnostic = result.stderr.strip().splitlines()[0]
        expected = case["capability_expected"]
        rows.append(
            {
                "case": case["name"],
                "description": case["description"],
                "expected": expected,
                "observed": observed,
                "matched_expectation": "yes" if observed == expected else "no",
                "diagnostic": diagnostic,
            }
        )

    write_csv(
        results_dir / "raw" / "capability_checks.csv",
        ["case", "description", "expected", "observed", "matched_expectation", "diagnostic"],
        rows,
    )

    plot_capability_summary(rows, plot_results_dir or results_dir)
    return rows


def plot_capability_summary(rows: list[dict[str, Any]], results_dir: pathlib.Path) -> None:
    labels = [row["case"] for row in rows]
    values = [1 if row["observed"] == "supported" else 0 for row in rows]
    colors = ["#2ca02c" if row["matched_expectation"] == "yes" else "#d62728" for row in rows]
    fig, ax = plt.subplots(figsize=(16, 5))
    ax.bar(labels, values, color=colors)
    ax.set_ylim(-0.1, 1.1)
    ax.set_yticks([0, 1], ["unsupported", "supported"])
    ax.set_title("Loom Capability and Limitation Checks (higher is better)")
    plt.setp(ax.get_xticklabels(), rotation=25, ha="right")
    fig.tight_layout()
    ensure_dir(results_dir / "plots")
    fig.savefig(results_dir / "plots" / "capability_summary.png", dpi=160)
    plt.close(fig)


def loom_profiles_by_mode(config: dict[str, Any], mode: str) -> list[dict[str, Any]]:
    return [profile for profile in config["loom_profiles"] if mode in profile["comparison_modes"]]


def autotuned_loom_profiles(config: dict[str, Any]) -> list[dict[str, Any]]:
    return [profile for profile in config["loom_profiles"] if profile["autotuned"]]


def summarize_runtime(runtime_rows: list[dict[str, Any]], capability_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    capability_map = {row["case"]: row["matched_expectation"] for row in capability_rows}
    grouped: dict[tuple[str, str, int], list[float]] = {}
    metadata: dict[str, dict[str, Any]] = {}
    for row in runtime_rows:
        grouped.setdefault((row["kernel"], row["implementation"], int(row["size"])), []).append(float(row["seconds"]))
        metadata[row["kernel"]] = {
            "application_domain": row["application_domain"],
            "workload_class": row["workload_class"],
            "application": row["application"],
        }

    reference_medians: dict[tuple[str, int], float] = {}
    for (kernel, implementation, size), values in grouped.items():
        if implementation == REFERENCE_IMPLEMENTATION:
            reference_medians[(kernel, size)] = statistics.median(values)

    summary_rows: list[dict[str, Any]] = []
    for (kernel, implementation, size), values in sorted(grouped.items()):
        median = statistics.median(values)
        if len(values) >= 2:
            q1, _, q3 = statistics.quantiles(values, n=4, method="inclusive")
        else:
            q1 = median
            q3 = median
        reference_median = reference_medians.get((kernel, size), median)
        summary_rows.append(
            {
                "kernel": kernel,
                "implementation": implementation,
                "implementation_label": ALL_IMPLEMENTATION_LABELS[implementation],
                "implementation_kind": ALL_IMPLEMENTATION_MAP[implementation]["kind"],
                "autotuned": "yes" if ALL_IMPLEMENTATION_MAP[implementation].get("autotuned", False) else "no",
                "optimization_flags": ",".join(ALL_IMPLEMENTATION_MAP[implementation].get("optimizations", [])),
                "size": size,
                "median_ms": median * 1000.0,
                "q1_ms": q1 * 1000.0,
                "q3_ms": q3 * 1000.0,
                "speedup_vs_loom_none_fixed": reference_median / median,
                "capability_expectation_met": capability_map.get(kernel, "n/a"),
                **metadata[kernel],
            }
        )
    return summary_rows


def filter_summary_rows(summary_rows: list[dict[str, Any]], implementation_order: tuple[str, ...]) -> list[dict[str, Any]]:
    allowed = set(implementation_order)
    return [row for row in summary_rows if row["implementation"] in allowed]


def summarize_gap_vs_best_external(
    summary_rows: list[dict[str, Any]],
    loom_key: str,
    external_order: tuple[str, ...],
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for row in summary_rows:
        grouped.setdefault((row["kernel"], int(row["size"])), []).append(row)
    rows: list[dict[str, Any]] = []
    external_set = set(external_order) - {
        "loom_none_fixed",
        "loom_none_autotuned",
        "loom_full_fixed",
        "loom_full_autotuned",
    }
    for (kernel, size), items in sorted(grouped.items()):
        loom_row = next((row for row in items if row["implementation"] == loom_key), None)
        external_rows = [row for row in items if row["implementation"] in external_set]
        if loom_row is None or not external_rows:
            continue
        best_external = min(external_rows, key=lambda row: float(row["median_ms"]))
        loom_median = float(loom_row["median_ms"])
        best_external_median = float(best_external["median_ms"])
        rows.append(
            {
                "kernel": kernel,
                "size": size,
                "loom_implementation": loom_row["implementation"],
                "loom_label": loom_row["implementation_label"],
                "loom_median_ms": loom_median,
                "best_external_implementation": best_external["implementation"],
                "best_external_label": best_external["implementation_label"],
                "best_external_median_ms": best_external_median,
                "gap_ratio": loom_median / best_external_median if best_external_median > 0.0 else math.inf,
                "application_domain": loom_row["application_domain"],
                "workload_class": loom_row["workload_class"],
                "application": loom_row["application"],
            }
        )
    return rows


def summarize_gap_by_class(gap_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in gap_rows:
        grouped.setdefault(str(row["workload_class"]), []).append(row)
    rows: list[dict[str, Any]] = []
    for workload_class, items in sorted(grouped.items()):
        gap_values = [float(item["gap_ratio"]) for item in items]
        wins = sum(1 for item in items if float(item["gap_ratio"]) < 1.0)
        rows.append(
            {
                "workload_class": workload_class,
                "application_domain": items[0]["application_domain"],
                "median_gap_to_best_fixed_triton": statistics.median(gap_values),
                "worst_gap_to_best_fixed_triton": max(gap_values),
                "loom_wins_vs_best_fixed_triton": wins,
                "case_count": len(items),
            }
        )
    return rows


def summarize_top_losses(
    gap_rows: list[dict[str, Any]], limit: int = 20
) -> list[dict[str, Any]]:
    return sorted(
        gap_rows,
        key=lambda row: float(row["gap_ratio"]),
        reverse=True,
    )[:limit]


OPTIMIZATION_PROGRESS_FIELDS = [
    "kernel",
    "size",
    "candidate",
    "candidate_label",
    "baseline",
    "baseline_label",
    "candidate_median_ms",
    "baseline_median_ms",
    "speedup_vs_baseline",
    "application_domain",
    "workload_class",
    "application",
]


def summarize_optimization_progress(
    summary_rows: list[dict[str, Any]],
    candidate_groups: list[str],
    baseline_groups: list[str],
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int], dict[str, dict[str, Any]]] = {}
    for row in summary_rows:
        grouped.setdefault((str(row["kernel"]), int(row["size"])), {})[str(row["implementation"])] = row
    rows: list[dict[str, Any]] = []
    for (kernel, size), impl_rows in sorted(grouped.items()):
        for candidate in candidate_groups:
            candidate_row = impl_rows.get(candidate)
            if candidate_row is None:
                continue
            for baseline in baseline_groups:
                baseline_row = impl_rows.get(baseline)
                if baseline_row is None:
                    continue
                candidate_median = float(candidate_row["median_ms"])
                baseline_median = float(baseline_row["median_ms"])
                rows.append(
                    {
                        "kernel": kernel,
                        "size": size,
                        "candidate": candidate,
                        "candidate_label": candidate_row["implementation_label"],
                        "baseline": baseline,
                        "baseline_label": baseline_row["implementation_label"],
                        "candidate_median_ms": candidate_median,
                        "baseline_median_ms": baseline_median,
                        "speedup_vs_baseline": baseline_median / candidate_median if candidate_median > 0.0 else math.inf,
                        "application_domain": candidate_row["application_domain"],
                        "workload_class": candidate_row["workload_class"],
                        "application": candidate_row["application"],
                    }
                )
    return rows


def style_boxplot(ax: plt.Axes, data: list[list[float]], positions: list[float], colors: list[str]) -> None:
    box = ax.boxplot(data, positions=positions, widths=0.7, patch_artist=True)
    for patch, color in zip(box["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)


def first_external_index(implementation_order: tuple[str, ...]) -> int | None:
    for index, implementation in enumerate(implementation_order):
        if not str(ALL_IMPLEMENTATION_MAP[implementation]["kind"]).startswith("loom-"):
            return index
    return None


def add_loom_external_separators(
    ax: plt.Axes,
    positions: list[float],
    implementation_order: tuple[str, ...],
    group_count: int,
) -> None:
    boundary_index = first_external_index(implementation_order)
    if boundary_index is None or boundary_index == 0:
        return
    group_width = len(implementation_order)
    for group_index in range(group_count):
        offset = group_index * group_width
        left = positions[offset + boundary_index - 1]
        right = positions[offset + boundary_index]
        ax.axvline(
            (left + right) / 2.0,
            color="#4a5568",
            linestyle=(0, (4, 4)),
            linewidth=1.0,
            alpha=0.55,
            zorder=0,
        )


def plot_compile_boxplots(
    compile_rows: list[dict[str, Any]],
    results_dir: pathlib.Path,
    kernels: list[str],
    kernel_configs: dict[str, dict[str, Any]],
    implementation_order: tuple[str, ...],
    output_subdir: str,
) -> None:
    if not implementation_order:
        return
    plot_dir = results_dir / "plots" / output_subdir
    ensure_dir(plot_dir)
    for kernel in kernels:
        workload_class = str(kernel_configs.get(kernel, {}).get("workload_class", "unknown-workload"))
        data = []
        colors = []
        for implementation in implementation_order:
            values = [
                float(row["seconds"]) * 1000.0
                for row in compile_rows
                if row["kernel"] == kernel and row["implementation"] == implementation
            ]
            data.append(values)
            colors.append(ALL_IMPLEMENTATION_COLORS[implementation])
        positions, _ = grouped_positions(1, implementation_order)
        scale_mode = plot_scale_mode(kernel_configs, kernel)
        use_dual_scale = scale_mode == "linear_and_log" and all(
            values and all(value > 0.0 for value in values) for values in data
        )
        figure_width = max(14, len(implementation_order) * 1.6)
        if use_dual_scale:
            fig, axes = plt.subplots(2, 1, figsize=(figure_width, 9), sharex=True)
            for ax, scale_label, use_log in (
                (axes[0], "linear scale", False),
                (axes[1], "log scale", True),
            ):
                style_boxplot(ax, data, positions, colors)
                add_loom_external_separators(ax, positions, implementation_order, 1)
                ax.set_xticks(positions, [ALL_IMPLEMENTATION_LABELS[key] for key in implementation_order])
                plt.setp(ax.get_xticklabels(), rotation=18, ha="right")
                ax.set_ylabel("Time to first result (ms)")
                ax.set_title(
                    f"{workload_class}: {kernel} time-to-first-result distribution (lower is better, {scale_label})"
                )
                if use_log:
                    ax.set_yscale("log")
                add_implementation_legend(ax, implementation_order)
        else:
            fig, ax = plt.subplots(figsize=(figure_width, 5))
            style_boxplot(ax, data, positions, colors)
            add_loom_external_separators(ax, positions, implementation_order, 1)
            ax.set_xticks(positions, [ALL_IMPLEMENTATION_LABELS[key] for key in implementation_order])
            plt.setp(ax.get_xticklabels(), rotation=18, ha="right")
            ax.set_ylabel("Time to first result (ms)")
            ax.set_title(
                f"{workload_class}: {kernel} time-to-first-result distribution (lower is better, linear scale)"
            )
            add_implementation_legend(ax, implementation_order)
        fig.tight_layout()
        fig.savefig(plot_dir / f"{kernel}_compile_boxplot.png", dpi=160)
        plt.close(fig)


def plot_runtime_boxplots(
    runtime_rows: list[dict[str, Any]],
    results_dir: pathlib.Path,
    kernels: list[str],
    sizes: list[int],
    kernel_configs: dict[str, dict[str, Any]],
    implementation_order: tuple[str, ...],
    output_subdir: str,
) -> None:
    if not implementation_order:
        return
    plot_dir = results_dir / "plots" / output_subdir
    ensure_dir(plot_dir)
    for kernel in kernels:
        workload_class = str(kernel_configs.get(kernel, {}).get("workload_class", "unknown-workload"))
        data = []
        colors = []
        size_labels = []
        for size in sizes:
            for implementation in implementation_order:
                values = [
                    float(row["seconds"]) * 1000.0
                    for row in runtime_rows
                    if row["kernel"] == kernel
                    and row["implementation"] == implementation
                    and int(row["size"]) == size
                ]
                data.append(values)
                colors.append(ALL_IMPLEMENTATION_COLORS[implementation])
            size_labels.append(f"2^{int(math.log2(size))}")
        positions, centers = grouped_positions(len(sizes), implementation_order)
        scale_mode = plot_scale_mode(kernel_configs, kernel)
        use_dual_scale = scale_mode == "linear_and_log" and all(
            values and all(value > 0.0 for value in values) for values in data
        )
        figure_width = max(18, len(implementation_order) * 2.0)
        if use_dual_scale:
            fig, axes = plt.subplots(2, 1, figsize=(figure_width, 11), sharex=True)
            plotting_axes = (
                (axes[0], "linear scale", False),
                (axes[1], "log scale", True),
            )
        else:
            fig, axis = plt.subplots(figsize=(figure_width, 6))
            plotting_axes = ((axis, "linear scale", False),)
        for ax, scale_label, use_log in plotting_axes:
            style_boxplot(ax, data, positions, colors)
            add_loom_external_separators(ax, positions, implementation_order, len(sizes))
            ax.set_xticks(centers, size_labels)
            ax.set_xlabel("Held-out evaluation input size")
            ax.set_ylabel("Runtime latency (ms)")
            ax.set_title(
                f"{workload_class}: {kernel} runtime distribution (lower is better, {scale_label})"
            )
            if use_log:
                ax.set_yscale("log")
            add_implementation_legend(ax, implementation_order)
        fig.tight_layout()
        fig.savefig(plot_dir / f"{kernel}_runtime_boxplot.png", dpi=160)
        plt.close(fig)


def positive_geomean(values: list[float]) -> float:
    positive = [value for value in values if value > 0.0 and math.isfinite(value)]
    if not positive:
        return math.nan
    return math.exp(sum(math.log(value) for value in positive) / len(positive))


def plot_suite_summary(
    summary_rows: list[dict[str, Any]],
    results_dir: pathlib.Path,
    kernels: list[str],
    sizes: list[int],
    output_name: str = "suite_summary_cuda_gap.png",
    title: str = "Suite summary: best Loom Triton and CUDA runtime gaps vs best external Triton/CUDA",
) -> None:
    if not summary_rows:
        return
    external_kinds = {"triton", "cuda"}
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = {}
    metadata: dict[str, dict[str, str]] = {}
    for row in summary_rows:
        kernel = str(row["kernel"])
        grouped.setdefault((kernel, int(row["size"])), []).append(row)
        metadata[kernel] = {
            "workload_class": str(row.get("workload_class", "unknown-workload")),
            "application_domain": str(row.get("application_domain", "unknown-domain")),
        }

    rows: list[dict[str, Any]] = []
    for kernel in kernels:
        triton_ratios: list[float] = []
        cuda_ratios: list[float] = []
        best_labels: list[str] = []
        for size in sizes:
            items = grouped.get((kernel, int(size)), [])
            loom_triton_rows = [
                row for row in items if row.get("implementation_kind") == "loom-triton"
            ]
            loom_cuda_rows = [
                row for row in items if row.get("implementation_kind") == "loom-cuda"
            ]
            external_rows = [
                row for row in items if row.get("implementation_kind") in external_kinds
            ]
            if not external_rows:
                triton_ratios.append(math.nan)
                cuda_ratios.append(math.nan)
                best_labels.append("n/a")
                continue
            best_external = min(external_rows, key=lambda row: float(row["median_ms"]))
            best_external_median = float(best_external["median_ms"])
            if not loom_triton_rows or best_external_median <= 0.0:
                triton_ratios.append(math.nan)
            else:
                best_loom_triton = min(
                    loom_triton_rows, key=lambda row: float(row["median_ms"])
                )
                triton_ratios.append(float(best_loom_triton["median_ms"]) / best_external_median)
            if not loom_cuda_rows or best_external_median <= 0.0:
                cuda_ratios.append(math.nan)
            else:
                best_loom_cuda = min(loom_cuda_rows, key=lambda row: float(row["median_ms"]))
                cuda_ratios.append(float(best_loom_cuda["median_ms"]) / best_external_median)
            best_labels.append(str(best_external["implementation_label"]))
        if any(math.isfinite(value) for value in triton_ratios + cuda_ratios):
            rows.append(
                {
                    "kernel": kernel,
                    "workload_class": metadata.get(kernel, {}).get(
                        "workload_class", "unknown-workload"
                    ),
                    "triton_ratios": triton_ratios,
                    "cuda_ratios": cuda_ratios,
                    "best_labels": best_labels,
                    "triton_geomean": positive_geomean(triton_ratios),
                    "cuda_geomean": positive_geomean(cuda_ratios),
                    "sort_geomean": positive_geomean(triton_ratios + cuda_ratios),
                    "worst": max(
                        [
                            value
                            for value in triton_ratios + cuda_ratios
                            if math.isfinite(value)
                        ],
                        default=math.nan,
                    ),
                }
            )

    if not rows:
        return
    rows.sort(key=lambda row: (str(row["workload_class"]), -float(row["sort_geomean"]), str(row["kernel"])))
    triton_size_geomeans = [
        positive_geomean(
            [
                float(row["triton_ratios"][size_index])
                for row in rows
                if math.isfinite(float(row["triton_ratios"][size_index]))
            ]
        )
        for size_index in range(len(sizes))
    ]
    cuda_size_geomeans = [
        positive_geomean(
            [
                float(row["cuda_ratios"][size_index])
                for row in rows
                if math.isfinite(float(row["cuda_ratios"][size_index]))
            ]
        )
        for size_index in range(len(sizes))
    ]
    triton_suite_geomean = positive_geomean(
        [
            float(value)
            for row in rows
            for value in row["triton_ratios"]
            if math.isfinite(float(value))
        ]
    )
    cuda_suite_geomean = positive_geomean(
        [
            float(value)
            for row in rows
            for value in row["cuda_ratios"]
            if math.isfinite(float(value))
        ]
    )

    column_groups = [f"2^{int(math.log2(size))}" for size in sizes] + ["kernel gmean"]
    matrix: list[list[float]] = []
    for row in rows:
        row_values: list[float] = []
        triton_values = list(row["triton_ratios"]) + [float(row["triton_geomean"])]
        cuda_values = list(row["cuda_ratios"]) + [float(row["cuda_geomean"])]
        for triton_value, cuda_value in zip(triton_values, cuda_values):
            row_values.extend([float(triton_value), float(cuda_value)])
        matrix.append(row_values)
    bottom_values: list[float] = []
    for triton_value, cuda_value in zip(
        triton_size_geomeans + [triton_suite_geomean],
        cuda_size_geomeans + [cuda_suite_geomean],
    ):
        bottom_values.extend([float(triton_value), float(cuda_value)])
    matrix.append(bottom_values)

    fig_height = max(10.0, 0.42 * (len(rows) + 1) + 3.4)
    fig, ax = plt.subplots(figsize=(17.8, fig_height))
    norm = TwoSlopeNorm(vmin=0.5, vcenter=1.0, vmax=1.5)
    image = ax.imshow(matrix, cmap="RdYlGn_r", norm=norm, aspect="auto")

    for y_index, row_values in enumerate(matrix):
        for x_index, value in enumerate(row_values):
            label = "n/a" if not math.isfinite(float(value)) else f"{float(value):.2f}x"
            ax.text(
                x_index,
                y_index,
                label,
                ha="center",
                va="center",
                fontsize=7,
                fontweight="bold" if y_index == len(matrix) - 1 else "normal",
            )

    subcolumn_labels: list[str] = []
    group_centers: list[float] = []
    for group_index, _group_label in enumerate(column_groups):
        subcolumn_labels.extend(["T", "C"])
        group_centers.append(group_index * 2 + 0.5)
    y_labels = [f"{row['kernel']} ({row['workload_class']})" for row in rows] + [
        "size gmean"
    ]
    ax.set_xticks(range(len(subcolumn_labels)), subcolumn_labels)
    ax.set_xticks(group_centers, column_groups, minor=True)
    ax.set_yticks(range(len(y_labels)), y_labels)
    ax.tick_params(axis="x", which="major", rotation=0, pad=2, labelsize=8)
    ax.tick_params(axis="x", which="minor", pad=18, length=0, labelsize=10)
    ax.set_xlabel("Held-out evaluation input size")

    values = [
        float(value)
        for row in rows
        for value in row["triton_ratios"] + row["cuda_ratios"]
        if math.isfinite(float(value))
    ]
    wins = sum(1 for value in values if value < 1.0)
    parity = sum(1 for value in values if 1.0 <= value <= 1.05)
    lag = sum(1 for value in values if value > 1.05)
    subtitle = (
        f"suite gmean: Loom Triton {triton_suite_geomean:.2f}x, Loom CUDA {cuda_suite_geomean:.2f}x | wins {wins} | parity within 5% {parity} | lag >5% {lag}"
    )
    fig.suptitle(title, y=0.985, fontsize=14)
    fig.text(
        0.37,
        0.952,
        "Green means generated Loom is faster; red means slower. Each size has paired T/C subcolumns.",
        ha="left",
        va="center",
        fontsize=10,
        color="#2d3748",
    )
    fig.text(0.37, 0.928, subtitle, ha="left", va="center", fontsize=10, color="#2d3748")
    fig.text(
        0.37,
        0.020,
        "T = fastest generated Loom Triton profile; C = fastest generated Loom CUDA profile. Denominator is the fastest external Triton/CUDA baseline for the same kernel and size.",
        ha="left",
        va="center",
        fontsize=8,
        color="#4a5568",
    )

    last_class: str | None = None
    for row_index, row in enumerate(rows):
        workload_class = str(row["workload_class"])
        if last_class is not None and workload_class != last_class:
            ax.axhline(row_index - 0.5, color="#718096", linewidth=0.8, alpha=0.55)
        last_class = workload_class
    ax.axhline(len(rows) - 0.5, color="#2d3748", linewidth=1.2, alpha=0.75)
    for group_index in range(1, len(column_groups)):
        ax.axvline(group_index * 2 - 0.5, color="#a0aec0", linewidth=0.65, alpha=0.75)
    ax.axvline(len(sizes) * 2 - 0.5, color="#2d3748", linewidth=1.2, alpha=0.75)
    ax.set_xlim(-0.5, len(subcolumn_labels) - 0.5)
    ax.set_ylim(len(matrix) - 0.5, -0.5)

    colorbar = fig.colorbar(image, ax=ax, fraction=0.025, pad=0.02)
    colorbar.set_label("Loom backend median / best external median")
    fig.subplots_adjust(left=0.34, right=0.88, top=0.90, bottom=0.08)
    plot_dir = results_dir / "plots"
    ensure_dir(plot_dir)
    fig.savefig(plot_dir / output_name, dpi=180)
    plt.close(fig)


def filter_generalization_rows(summary_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    allowed = set(HELD_OUT_GENERALIZATION_KERNELS)
    return [row for row in summary_rows if str(row["kernel"]) in allowed]


def write_generalization_summaries(
    summary_rows: list[dict[str, Any]], results_dir: pathlib.Path
) -> None:
    generalization_rows = filter_generalization_rows(summary_rows)
    if not generalization_rows:
        return
    summary_dir = runtime_results_dir(results_dir) / "summaries"
    write_csv(
        summary_dir / "generalization_summary.csv",
        SUMMARY_FIELDS,
        generalization_rows,
    )
    gap_rows = summarize_gap_vs_best_external(
        generalization_rows,
        "loom_cuda_fixed",
        tuple(HELD_OUT_CUDA_COMPARISON_IMPLEMENTATIONS),
    )
    write_csv(
        summary_dir / "generalization_cuda_vs_best_external_gap.csv",
        GAP_FIELDS,
        gap_rows,
    )
    write_csv(
        summary_dir / "generalization_cuda_top_losses.csv",
        GAP_FIELDS,
        summarize_top_losses(gap_rows),
    )


def filter_current_non_held_out_rows(summary_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    allowed = set(CURRENT_NON_HELD_OUT_KERNELS)
    return [row for row in summary_rows if str(row["kernel"]) in allowed]


def write_current_cuda_summaries(
    summary_rows: list[dict[str, Any]], results_dir: pathlib.Path
) -> None:
    current_rows = filter_current_non_held_out_rows(summary_rows)
    if not current_rows:
        return
    summary_dir = runtime_results_dir(results_dir) / "summaries"
    write_csv(
        summary_dir / "current_non_held_out_summary.csv",
        SUMMARY_FIELDS,
        current_rows,
    )
    gap_rows = summarize_gap_vs_best_external(
        current_rows,
        "loom_cuda_fixed",
        tuple(CURRENT_CUDA_COMPARISON_IMPLEMENTATIONS),
    )
    write_csv(
        summary_dir / "current_cuda_vs_best_external_gap.csv",
        GAP_FIELDS,
        gap_rows,
    )
    write_csv(
        summary_dir / "current_cuda_top_losses.csv",
        GAP_FIELDS,
        summarize_top_losses(gap_rows),
    )
    secured_rows = [
        row
        for row in sorted(gap_rows, key=lambda item: (str(item["kernel"]), int(item["size"])))
        if float(row["gap_ratio"]) <= CURRENT_SECURED_GAP_THRESHOLD
    ]
    write_csv(
        summary_dir / "current_cuda_secured_win_guard.csv",
        GAP_FIELDS,
        secured_rows,
    )


def write_current_triton_summaries(
    summary_rows: list[dict[str, Any]], results_dir: pathlib.Path
) -> None:
    current_rows = filter_current_non_held_out_rows(summary_rows)
    if not current_rows:
        return
    summary_dir = runtime_results_dir(results_dir) / "summaries"
    gap_rows = summarize_gap_vs_best_external(
        current_rows,
        "loom_full_fixed",
        tuple(CURRENT_TRITON_FIXED_EXTERNAL_IMPLEMENTATIONS),
    )
    if not gap_rows:
        return
    write_csv(
        summary_dir / "current_triton_vs_best_fixed_triton_gap.csv",
        GAP_FIELDS,
        gap_rows,
    )
    write_csv(
        summary_dir / "current_triton_top_losses.csv",
        GAP_FIELDS,
        summarize_top_losses(gap_rows),
    )
    secured_rows = [
        row
        for row in sorted(gap_rows, key=lambda item: (str(item["kernel"]), int(item["size"])))
        if float(row["gap_ratio"]) <= CURRENT_TRITON_SECURED_GAP_THRESHOLD
    ]
    write_csv(
        summary_dir / "current_triton_secured_win_guard.csv",
        GAP_FIELDS,
        secured_rows,
    )


def regenerate_plots(
    config: dict[str, Any],
    results_dir: pathlib.Path,
    kernels: list[str],
    evaluation_sizes: list[int],
) -> None:
    kernel_configs = kernel_config_map(config["supported_kernels"])
    capability_path = shared_results_dir(results_dir) / "raw" / "capability_checks.csv"
    compile_root = compile_results_dir(results_dir)
    runtime_root = runtime_results_dir(results_dir)
    compile_path = compile_root / "raw" / "compile_measurements.csv"
    runtime_path = runtime_root / "raw" / "runtime_measurements.csv"
    if capability_path.exists():
        plot_capability_summary(read_csv(capability_path), runtime_root)
    if compile_path.exists():
        compile_rows = read_csv(compile_path)
        plot_compile_boxplots(
            compile_rows,
            compile_root,
            kernels,
            kernel_configs,
            tuple(config["loom_internal_order"]),
            "loom_internal",
        )
        plot_compile_boxplots(
            compile_rows,
            compile_root,
            kernels,
            kernel_configs,
            tuple(config["loom_vs_others_order"]),
            "loom_vs_others",
        )
    if runtime_path.exists():
        runtime_rows = read_csv(runtime_path)
        capability_rows = read_csv(capability_path) if capability_path.exists() else []
        summary_rows = summarize_runtime(runtime_rows, capability_rows)
        plot_suite_summary(summary_rows, runtime_root, kernels, evaluation_sizes)
        generalization_kernels = [
            kernel for kernel in kernels if kernel in set(HELD_OUT_GENERALIZATION_KERNELS)
        ]
        if generalization_kernels:
            plot_suite_summary(
                filter_generalization_rows(summary_rows),
                runtime_root,
                generalization_kernels,
                evaluation_sizes,
                output_name="suite_summary_generalization_gap.png",
                title="Generalization summary: held-out Loom backend runtime gaps vs best external Triton/CUDA",
            )
        plot_runtime_boxplots(
            runtime_rows,
            runtime_root,
            kernels,
            evaluation_sizes,
            kernel_configs,
            tuple(config["loom_internal_order"]),
            "loom_internal",
        )
        plot_runtime_boxplots(
            runtime_rows,
            runtime_root,
            kernels,
            evaluation_sizes,
            kernel_configs,
            tuple(config["loom_vs_others_order"]),
            "loom_vs_others",
        )


def finalize_results(
    config: dict[str, Any],
    results_dir: pathlib.Path,
    kernels: list[str],
    evaluation_sizes: list[int],
) -> None:
    run_state_path = shared_results_dir(results_dir) / "raw" / "run_state.json"
    run_state = load_json(run_state_path) if run_state_path.exists() else {}
    capability_rows = read_csv_if_exists(shared_results_dir(results_dir) / "raw" / "capability_checks.csv")
    runtime_rows = read_csv_if_exists(runtime_results_dir(results_dir) / "raw" / "runtime_measurements.csv")
    verification_rows = read_csv_if_exists(
        runtime_results_dir(results_dir) / "raw" / "verification_measurements.csv"
    )
    compile_rows = read_csv_if_exists(compile_results_dir(results_dir) / "raw" / "compile_measurements.csv")
    assert_complete_raw_results(run_state, compile_rows, verification_rows, runtime_rows)
    tuning_rows = read_csv_if_exists(runtime_results_dir(results_dir) / "raw" / "tuning_measurements.csv")
    summary_rows = summarize_runtime(runtime_rows, capability_rows) if runtime_rows else []
    if summary_rows:
        write_csv(runtime_results_dir(results_dir) / "summaries" / "summary.csv", SUMMARY_FIELDS, summary_rows)
        write_csv(
            runtime_results_dir(results_dir) / "summaries" / "loom_internal_summary.csv",
            SUMMARY_FIELDS,
            filter_summary_rows(summary_rows, tuple(config["loom_internal_order"])),
        )
        write_csv(
            runtime_results_dir(results_dir) / "summaries" / "loom_vs_others_summary.csv",
            SUMMARY_FIELDS,
            filter_summary_rows(summary_rows, tuple(config["loom_vs_others_order"])),
        )
        write_csv(
            runtime_results_dir(results_dir) / "summaries" / "loom_search_summary.csv",
            SUMMARY_FIELDS,
            filter_summary_rows(summary_rows, tuple(config.get("loom_search_order", []))),
        )
        gap_rows = summarize_gap_vs_best_external(
            summary_rows,
            "loom_full_fixed",
            tuple(config["loom_vs_others_order"]),
        )
        write_csv(runtime_results_dir(results_dir) / "summaries" / "loom_vs_best_external_gap.csv", GAP_FIELDS, gap_rows)
        write_csv(
            runtime_results_dir(results_dir) / "summaries" / "full_suite_gap_by_class.csv",
            GAP_BY_CLASS_FIELDS,
            summarize_gap_by_class(gap_rows),
        )
        write_csv(
            runtime_results_dir(results_dir) / "summaries" / "full_suite_top_losses.csv",
            GAP_FIELDS,
            summarize_top_losses(gap_rows),
        )
        if run_state.get("mode") == "optimization-pass":
            write_csv(
                runtime_results_dir(results_dir) / "summaries" / "optimization_progress.csv",
                OPTIMIZATION_PROGRESS_FIELDS,
                summarize_optimization_progress(
                    summary_rows,
                    [str(item) for item in run_state.get("candidate_groups", [])],
                    [str(item) for item in run_state.get("baseline_groups", [])],
                ),
            )
        write_generalization_summaries(summary_rows, results_dir)
        write_current_cuda_summaries(summary_rows, results_dir)
        write_current_triton_summaries(summary_rows, results_dir)
    dataset_manifest_rows = finalize_dataset_manifest(results_dir)
    regenerate_plots(config, results_dir, kernels, evaluation_sizes)
    verification_failures_path = runtime_results_dir(results_dir) / "raw" / "verification_failures.json"
    verification_failures = (
        load_json(verification_failures_path)
        if verification_failures_path.exists()
        else {"failures": []}
    )
    if verification_failures.get("failures"):
        raise SystemExit("verification failed; see runtime/raw/verification_failures.json")
    audit_report = load_json(shared_results_dir(results_dir) / "raw" / "audit_report.json")
    environment = load_json(shared_results_dir(results_dir) / "raw" / "environment.json")
    tuning_decisions_path = runtime_results_dir(results_dir) / "raw" / "tuning_decisions.json"
    tuning_decisions = load_json(tuning_decisions_path) if tuning_decisions_path.exists() else {}
    generate_report(
        {
            **config,
            "run_mode": run_state.get("mode", "unknown"),
            "warmup_repetitions": run_state.get("warmup", config.get("warmup_repetitions")),
            "runtime_repetitions": run_state.get("runtime_repetitions", config.get("runtime_repetitions")),
            "compile_repetitions": run_state.get("compile_repetitions", config.get("compile_repetitions")),
            "evaluation_sizes": evaluation_sizes,
            "tuning_sizes": run_state.get("tuning_sizes", config.get("tuning_sizes", [])),
            "tuning_seed_offsets": run_state.get("tuning_seed_offsets", config.get("tuning_seed_offsets", [])),
            "evaluation_seed_offsets": run_state.get("evaluation_seed_offsets", config.get("evaluation_seed_offsets", [])),
            "candidate_groups": [str(item) for item in run_state.get("candidate_groups", [])],
            "baseline_groups": [str(item) for item in run_state.get("baseline_groups", [])],
        },
        environment,
        audit_report,
        verification_rows,
        dataset_manifest_rows,
        tuning_rows,
        tuning_decisions,
        summary_rows,
        runtime_results_dir(results_dir) / "report.md",
    )


def tune_python_runtime(
    kernel: dict[str, Any],
    implementation_key: str,
    runner: PythonRuntime,
    tuning_sizes: list[int],
    seed: int,
    seed_offsets: list[int],
    autotune_bounds: list[int],
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    dataset_manifest: dict[str, dict[str, Any]] = {}
    for size in tuning_sizes:
        for dataset_index, seed_offset in enumerate(seed_offsets):
            inputs = make_kernel_inputs(kernel, size, seed + size, seed_offset)
            record_dataset_manifest(dataset_manifest, kernel, "tuning", size, seed + size, seed_offset, inputs)
            expected = reference_output(logical_kernel_name(kernel), inputs)
            torch.cuda.synchronize()
            start = time.perf_counter()
            actual = runner(**inputs)
            torch.cuda.synchronize()
            seconds = time.perf_counter() - start
            assert_close(actual, expected, kernel)
            rows.append(
                row_with_kernel_metadata(
                    kernel,
                    {
                        "kernel": kernel["name"],
                        "implementation": implementation_key,
                        "size": size,
                        "bucket": kernel_bucket(size, autotune_bounds),
                        "dataset_id": dataset_id(kernel["name"], "tuning", size, seed_offset),
                        "dataset_index": dataset_index,
                        "dataset_seed": dataset_seed(seed + size, seed_offset),
                        "dataset_seed_offset": seed_offset,
                        "seconds": seconds,
                    },
                )
            )
    return rows, runner.autotune_state(), dataset_manifest


def run_tuning(
    config: dict[str, Any],
    options: RunOptions,
    writers: RawResultWriters,
    seen_dataset_ids: set[str],
    tuned_runners: dict[str, dict[str, PythonRuntime]] | None = None,
) -> tuple[dict[str, Any], dict[str, dict[str, PythonRuntime]]]:
    tuning_decisions: dict[str, Any] = {}
    tuned_runners = tuned_runners or {}
    tuned_loom_profiles = autotuned_loom_profiles(config)
    for kernel in config["supported_kernels"]:
        if kernel["name"] not in options.kernels:
            continue
        tuned_runners.setdefault(kernel["name"], {})
        sizes = kernel_tuning_sizes(kernel, options.tuning_sizes)
        for profile in tuned_loom_profiles:
            implementation_key = profile["key"]
            if not selected_for_kernel(options, implementation_key, kernel["name"]):
                continue
            runner = load_loom_runtime(
                kernel,
                profile,
                options.work_dir,
                config["autotune_config"],
                config["optimizer_config"],
            )
            autotune_bounds = bucket_upper_bounds_for_autotune(runner.module)
            rows, state, _ = tune_python_runtime(
                kernel,
                implementation_key,
                runner,
                sizes,
                config["seed"],
                options.tuning_seed_offsets,
                autotune_bounds,
            )
            for row in rows:
                inputs = make_kernel_inputs(
                    kernel,
                    int(row["size"]),
                    config["seed"] + int(row["size"]),
                    int(row["dataset_seed_offset"]),
                )
                record_dataset_manifest_streaming(
                    seen_dataset_ids,
                    writers,
                    kernel,
                    "tuning",
                    int(row["size"]),
                    config["seed"] + int(row["size"]),
                    int(row["dataset_seed_offset"]),
                    inputs,
                )
                output_row = row_with_implementation_metadata(row, implementation_key)
                writers.tuning.append_row(output_row)
                append_completed_unit(
                    writers,
                    "tuning",
                    kernel["name"],
                    implementation_key,
                    int(row["size"]),
                    str(row["dataset_id"]),
                    int(row["dataset_seed"]),
                    int(row["dataset_seed_offset"]),
                    "completed",
                )
            tuned_runners[kernel["name"]][implementation_key] = runner
            tuning_decisions.setdefault(kernel["name"], {})[implementation_key] = state
        for implementation_key in config["external_implementations"]:
            if implementation_key not in TUNED_IMPLEMENTATIONS:
                continue
            if implementation_key in LOOM_PROFILE_MAP:
                continue
            if not selected_for_kernel(options, implementation_key, kernel["name"]):
                continue
            if ALL_IMPLEMENTATION_MAP[implementation_key]["kind"] == "triton":
                runner = load_triton_runtime(kernel, implementation_key, options.work_dir)
                autotune_bounds = list(getattr(runner.module, "BUCKET_UPPER_BOUNDS", []))
            else:
                continue
            rows, state, _ = tune_python_runtime(
                kernel,
                implementation_key,
                runner,
                sizes,
                config["seed"],
                options.tuning_seed_offsets,
                autotune_bounds,
            )
            for row in rows:
                inputs = make_kernel_inputs(
                    kernel,
                    int(row["size"]),
                    config["seed"] + int(row["size"]),
                    int(row["dataset_seed_offset"]),
                )
                record_dataset_manifest_streaming(
                    seen_dataset_ids,
                    writers,
                    kernel,
                    "tuning",
                    int(row["size"]),
                    config["seed"] + int(row["size"]),
                    int(row["dataset_seed_offset"]),
                    inputs,
                )
                output_row = row_with_implementation_metadata(row, implementation_key)
                writers.tuning.append_row(output_row)
                append_completed_unit(
                    writers,
                    "tuning",
                    kernel["name"],
                    implementation_key,
                    int(row["size"]),
                    str(row["dataset_id"]),
                    int(row["dataset_seed"]),
                    int(row["dataset_seed_offset"]),
                    "completed",
                )
            tuned_runners[kernel["name"]][implementation_key] = runner
            tuning_decisions.setdefault(kernel["name"], {})[implementation_key] = state
    return tuning_decisions, tuned_runners


def run_performance(
    config: dict[str, Any],
    options: RunOptions,
    tuned_runners: dict[str, dict[str, PythonRuntime]],
    writers: RawResultWriters,
    seen_dataset_ids: set[str],
) -> list[dict[str, Any]]:
    verification_failures: list[dict[str, Any]] = []
    loom_profiles = config["loom_profiles"]
    if options.run_runtime:
        cuda_benchmark_prewarm()
    for kernel in config["supported_kernels"]:
        if kernel["name"] not in options.kernels:
            continue

        if options.run_compile:
            compile_measurements: dict[str, list[float]] = {}
            for implementation_key in config["external_implementations"]:
                if not selected_for_kernel(options, implementation_key, kernel["name"]):
                    continue
                spec = ALL_IMPLEMENTATION_MAP[implementation_key]
                if spec["kind"] == "triton":
                    compile_measurements[implementation_key] = measure_triton_compile(
                        kernel,
                        implementation_key,
                        options.compile_repetitions,
                        options.work_dir,
                        kernel_compile_probe_size(kernel, config["compile_probe_size"]),
                        config["seed"],
                    )
                elif spec["kind"] == "cuda":
                    compile_measurements[implementation_key] = measure_cuda_compile(
                        kernel,
                        spec["variant"],
                        options.compile_repetitions,
                        options.work_dir,
                        kernel_compile_probe_size(kernel, config["compile_probe_size"]),
                        config["seed"],
                    )
            for profile in loom_profiles:
                if not selected_for_kernel(options, profile["key"], kernel["name"]):
                    continue
                compile_measurements[profile["key"]] = measure_loom_compile(
                    kernel,
                    profile,
                    options.compile_repetitions,
                    options.work_dir,
                    config["autotune_config"],
                    config["optimizer_config"],
                    kernel_compile_probe_size(kernel, config["compile_probe_size"]),
                    config["seed"],
                )
            for implementation, measurements in compile_measurements.items():
                compile_rows = [
                    row_with_implementation_metadata(
                        row_with_kernel_metadata(
                            kernel,
                            {
                                "kernel": kernel["name"],
                                "implementation": implementation,
                                "run_index": index,
                                "seconds": seconds,
                            },
                        ),
                        implementation,
                    )
                    for index, seconds in enumerate(measurements)
                ]
                writers.compile.append_rows(compile_rows)
                append_completed_unit(
                    writers,
                    "compile",
                    kernel["name"],
                    implementation,
                    None,
                    "",
                    config["seed"],
                    "",
                    "completed",
                    detail=f"runs={len(measurements)}",
                )

        runners: dict[str, Callable[..., torch.Tensor]] = {}
        for implementation_key in config["external_implementations"]:
            if not selected_for_kernel(options, implementation_key, kernel["name"]):
                continue
            spec = ALL_IMPLEMENTATION_MAP[implementation_key]
            if spec["kind"] == "triton" and spec["autotuned"]:
                runners[implementation_key] = tuned_runners[kernel["name"]][implementation_key]
            elif spec["kind"] == "triton":
                runners[implementation_key] = load_triton_runtime(kernel, implementation_key, options.work_dir)
            elif spec["kind"] == "cuda":
                runners[implementation_key] = load_cuda_runtime(kernel, implementation_key, options.work_dir)
        for profile in loom_profiles:
            if not selected_for_kernel(options, profile["key"], kernel["name"]):
                continue
            if profile["autotuned"]:
                runners[profile["key"]] = tuned_runners[kernel["name"]][profile["key"]]
            else:
                runners[profile["key"]] = load_loom_runtime(
                    kernel,
                    profile,
                    options.work_dir,
                    config["autotune_config"],
                    config["optimizer_config"],
                )

        for size in options.evaluation_sizes:
            size_implementations = [
                implementation
                for implementation in runners
                if size in selected_sizes_for_kernel(options, implementation, kernel["name"])
            ]
            if not size_implementations:
                continue
            for dataset_index, seed_offset in enumerate(options.evaluation_seed_offsets):
                inputs = make_kernel_inputs(kernel, size, config["seed"] + size, seed_offset)
                phase = "evaluation"
                record_dataset_manifest_streaming(
                    seen_dataset_ids,
                    writers,
                    kernel,
                    phase,
                    size,
                    config["seed"] + size,
                    seed_offset,
                    inputs,
                )
                expected = reference_output(logical_kernel_name(kernel), inputs)
                ordered_implementations = implementation_execution_order(
                    size_implementations, kernel["name"], phase, size, dataset_seed(config["seed"] + size, seed_offset)
                )
                for implementation in ordered_implementations:
                    runner = runners[implementation]
                    if options.run_verification and verification_mode(kernel) == "exact_reference":
                        actual: torch.Tensor | None = None
                        try:
                            actual = runner(**inputs)
                            torch.cuda.synchronize()
                            max_abs_diff, max_rel_diff = assert_close(actual, expected, kernel)
                            actual_checksum = tensor_checksum(actual)
                            expected_checksum = tensor_checksum(expected)
                            status = "pass"
                            failure_message = ""
                        except Exception as exc:
                            torch.cuda.synchronize()
                            if actual is not None:
                                max_abs_diff, max_rel_diff = compare_outputs(actual, expected)
                                actual_checksum = tensor_checksum(actual)
                            else:
                                max_abs_diff, max_rel_diff = math.inf, math.inf
                                actual_checksum = ""
                            expected_checksum = tensor_checksum(expected)
                            status = "fail"
                            failure_message = str(exc)
                            verification_failures.append(
                                {
                                    "kernel": kernel["name"],
                                    "implementation": implementation,
                                    "size": size,
                                    "dataset_id": dataset_id(kernel["name"], phase, size, seed_offset),
                                    "dataset_seed": dataset_seed(config["seed"] + size, seed_offset),
                                    "message": failure_message,
                                    "max_abs_diff": max_abs_diff,
                                    "max_rel_diff": max_rel_diff,
                                }
                            )
                    elif options.run_verification:
                        status = "skipped"
                        failure_message = ""
                        max_abs_diff = 0.0
                        max_rel_diff = 0.0
                        expected_checksum = ""
                        actual_checksum = ""
                    else:
                        status = "completed"
                        failure_message = ""
                    if options.run_verification:
                        writers.verification.append_row(
                            row_with_implementation_metadata(
                                row_with_kernel_metadata(
                                    kernel,
                                    {
                                        "kernel": kernel["name"],
                                        "implementation": implementation,
                                        "phase": phase,
                                        "size": size,
                                        "dataset_id": dataset_id(kernel["name"], phase, size, seed_offset),
                                        "dataset_index": dataset_index,
                                        "dataset_seed": dataset_seed(config["seed"] + size, seed_offset),
                                        "dataset_seed_offset": seed_offset,
                                        "status": status,
                                        "max_abs_diff": max_abs_diff,
                                        "max_rel_diff": max_rel_diff,
                                        "expected_checksum": expected_checksum,
                                        "actual_checksum": actual_checksum,
                                        "verification_mode": verification_mode(kernel),
                                        "failure_message": failure_message,
                                    },
                                ),
                                implementation,
                            )
                        )
                        append_completed_unit(
                            writers,
                            "verification",
                            kernel["name"],
                            implementation,
                            size,
                            dataset_id(kernel["name"], phase, size, seed_offset),
                            dataset_seed(config["seed"] + size, seed_offset),
                            seed_offset,
                            status,
                            detail=failure_message,
                        )
                    if options.run_verification and status == "fail":
                        continue
                    if not options.run_runtime:
                        continue
                    measurements = timed_runs(
                        lambda runner=runner, inputs=inputs: runner(**inputs),
                        options.warmup,
                        options.runtime_repetitions,
                    )
                    runtime_output_rows: list[dict[str, Any]] = []
                    for index, seconds in enumerate(measurements):
                        runtime_output_rows.append(
                            row_with_implementation_metadata(
                                row_with_kernel_metadata(
                                    kernel,
                                    {
                                        "kernel": kernel["name"],
                                        "implementation": implementation,
                                        "size": size,
                                        "dataset_id": dataset_id(kernel["name"], phase, size, seed_offset),
                                        "dataset_index": dataset_index,
                                        "dataset_seed": dataset_seed(config["seed"] + size, seed_offset),
                                        "dataset_seed_offset": seed_offset,
                                        "run_index": index,
                                        "seconds": seconds,
                                    },
                                ),
                                implementation,
                            )
                        )
                    writers.runtime.append_rows(runtime_output_rows)
                    append_completed_unit(
                        writers,
                        "runtime",
                        kernel["name"],
                        implementation,
                        size,
                        dataset_id(kernel["name"], phase, size, seed_offset),
                        dataset_seed(config["seed"] + size, seed_offset),
                        seed_offset,
                        "completed",
                        detail=f"runs={len(measurements)}",
                    )

    write_json(writers.verification_failures_json, {"failures": verification_failures})
    return verification_failures


def generate_report(
    config: dict[str, Any],
    environment: dict[str, Any],
    audit_report: dict[str, Any],
    verification_rows: list[dict[str, Any]],
    dataset_manifest: list[dict[str, Any]],
    tuning_rows: list[dict[str, Any]],
    tuning_decisions: dict[str, Any],
    summary_rows: list[dict[str, Any]],
    output_path: pathlib.Path,
) -> None:
    grouped_summary: dict[str, list[dict[str, Any]]] = {}
    for row in summary_rows:
        grouped_summary.setdefault(row["kernel"], []).append(row)
    profile_lines = []
    for profile in config["loom_profiles"]:
        flags = ", ".join(profile["optimizations"]) if profile["optimizations"] else "none"
        modes = ", ".join(profile["comparison_modes"])
        autotuned = "yes" if profile["autotuned"] else "no"
        cuda_platform = profile.get("cuda_platform")
        platform_note = f", cuda_platform `{cuda_platform}`" if cuda_platform else ""
        profile_lines.append(
            f"- `{profile['key']}`: label `{profile['label']}`, autotuned `{autotuned}`, modes `{modes}`{platform_note}, opts `{flags}`"
        )
    platform_specific_profiles = [
        profile
        for profile in config["loom_profiles"]
        if str(profile.get("backend", "triton")) == "cuda"
        and profile.get("cuda_platform") not in (None, "generic")
    ]
    lines = [
        "# Loom Experiment Report",
        "",
        "## Environment",
        "",
        f"- torch: `{environment.get('torch_version', 'unknown')}`",
        f"- torch CUDA: `{environment.get('torch_cuda_version', 'unknown')}`",
        f"- triton: `{environment.get('triton_version', 'unknown')}`",
    ]
    lines.extend(["", "## Audit", ""])
    lines.append(f"- status: `{audit_report.get('status', 'unknown')}`")
    lines.append(f"- errors: `{len(audit_report.get('errors', []))}`")
    lines.append(f"- warnings: `{len(audit_report.get('warnings', []))}`")
    lines.extend(["", "## Run Mode", ""])
    lines.append(f"- mode: `{config.get('run_mode', 'unknown')}`")
    candidate_groups = [str(item) for item in config.get("candidate_groups", [])]
    baseline_groups = [str(item) for item in config.get("baseline_groups", [])]
    if candidate_groups:
        lines.append(f"- candidate groups: {', '.join(f'`{item}`' for item in candidate_groups)}")
    if baseline_groups:
        lines.append(f"- baseline groups: {', '.join(f'`{item}`' for item in baseline_groups)}")
    lines.extend(["", "## Dataset Protocol", ""])
    lines.append(f"- warmup repetitions: `{config.get('warmup_repetitions', 'unknown')}`")
    lines.append(f"- runtime repetitions: `{config.get('runtime_repetitions', 'unknown')}`")
    lines.append(f"- compile repetitions: `{config.get('compile_repetitions', 'unknown')}`")
    lines.append(
        f"- tuning seed offsets: {', '.join(str(value) for value in config.get('tuning_seed_offsets', [])) or 'none'}"
    )
    lines.append(
        f"- evaluation seed offsets: {', '.join(str(value) for value in config.get('evaluation_seed_offsets', [])) or 'none'}"
    )
    lines.append(f"- dataset manifest entries: `{len(dataset_manifest)}`")
    lines.extend(["", "## Verification", ""])
    verification_pass = sum(1 for row in verification_rows if row["status"] == "pass")
    verification_fail = sum(1 for row in verification_rows if row["status"] == "fail")
    verification_skip = sum(1 for row in verification_rows if row["status"] == "skipped")
    lines.append(f"- pass rows: `{verification_pass}`")
    lines.append(f"- fail rows: `{verification_fail}`")
    lines.append(f"- skipped rows: `{verification_skip}`")
    for device in environment.get("devices", []):
        lines.append(
            f"- GPU {device['index']}: `{device['name']}` cc `{device['compute_capability']}` memory `{device['total_memory_bytes']}` bytes"
        )
    lines.extend(["", "## Loom Profiles", ""])
    lines.extend(profile_lines)
    if platform_specific_profiles:
        lines.extend(["", "## Footnotes", ""])
        lines.append(
            "- Some public Loom CUDA results use platform-specific CUDA backend "
            "selection via `--cuda-platform current`; those results are tuned for "
            "the benchmark host and should not be read as fully portable generic "
            "CUDA codegen numbers."
        )
    lines.extend(["", "## Tuning", ""])
    if tuning_rows:
        lines.append(
            f"- tuned implementations: {', '.join(ALL_IMPLEMENTATION_LABELS[key] for key in TUNED_IMPLEMENTATIONS)}"
        )
        lines.append(
            f"- tuning sizes: {', '.join(f'2^{int(math.log2(size))}' for size in config['tuning_sizes'])}"
        )
    else:
        lines.append("- tuning was skipped")
    lines.extend(["", "## Workloads", ""])
    search_impls = set(config.get("loom_search_order", []))
    for kernel in config["supported_kernels"]:
        rows = grouped_summary.get(kernel["name"], [])
        if not rows:
            continue
        lines.extend(
            [
                f"### {kernel['name']}",
                "",
                f"- description: {kernel['description']}",
                f"- application: {kernel['application']}",
                f"- application_domain: `{kernel['application_domain']}`",
                f"- workload_class: `{kernel['workload_class']}`",
            ]
        )
        for size in config["evaluation_sizes"]:
            size_rows = [row for row in rows if int(row["size"]) == size]
            if not size_rows:
                continue
            internal_rows = [row for row in size_rows if row["implementation"] in config["loom_internal_order"]]
            external_rows = [row for row in size_rows if row["implementation"] in config["loom_vs_others_order"]]
            size_rows.sort(key=lambda row: float(row["median_ms"]))
            winner = size_rows[0]
            lines.append(
                f"- held-out 2^{int(math.log2(size))}: best overall `{winner['implementation_label']}` at `{float(winner['median_ms']):.4f} ms`"
            )
            if internal_rows:
                internal_rows.sort(key=lambda row: float(row["median_ms"]))
                best_internal = internal_rows[0]
                lines.append(
                    f"- held-out 2^{int(math.log2(size))}: best Loom profile `{best_internal['implementation_label']}` at `{float(best_internal['median_ms']):.4f} ms`"
                )
            search_rows = [row for row in size_rows if row["implementation"] in search_impls]
            if search_rows:
                search_rows.sort(key=lambda row: float(row["median_ms"]))
                best_search = search_rows[0]
                lines.append(
                    f"- held-out 2^{int(math.log2(size))}: best Loom search profile `{best_search['implementation_label']}` at `{float(best_search['median_ms']):.4f} ms`"
                )
            if external_rows:
                external_rows.sort(key=lambda row: float(row["median_ms"]))
                best_external = external_rows[0]
                lines.append(
                    f"- held-out 2^{int(math.log2(size))}: best Loom-vs-others comparison winner `{best_external['implementation_label']}` at `{float(best_external['median_ms']):.4f} ms`"
                )
        if kernel["name"] in tuning_decisions:
            lines.append(f"- tuning_state_keys: {', '.join(sorted(tuning_decisions[kernel['name']].keys()))}")
        lines.append("")
    write_text(output_path, "\n".join(lines))


def assert_complete_raw_results(
    run_state: dict[str, Any],
    compile_rows: list[dict[str, Any]],
    verification_rows: list[dict[str, Any]],
    runtime_rows: list[dict[str, Any]],
) -> None:
    expected_kernels = [str(item) for item in run_state.get("kernels", [])]
    expected_sizes = [int(item) for item in run_state.get("evaluation_sizes", [])]
    if not expected_kernels:
        return
    run_cases = run_state.get("benchmark_cases", [])
    implementations = [str(item) for item in run_state.get("implementation_filter", [])]
    if run_cases:
        expected_compile = {
            (str(item["kernel"]), str(item["implementation"]))
            for item in run_cases
        }
        expected_runtime = set()
        for item in run_cases:
            sizes = [int(size) for size in item.get("sizes", [])] or expected_sizes
            for size in sizes:
                expected_runtime.add((str(item["kernel"]), str(item["implementation"]), size))
        compile_pairs = {(row["kernel"], row["implementation"]) for row in compile_rows}
        verification_pairs = {
            (row["kernel"], row["implementation"], int(row["size"])) for row in verification_rows
        }
        runtime_pairs = {
            (row["kernel"], row["implementation"], int(row["size"])) for row in runtime_rows
        }
        missing_compile = sorted(
            f"{kernel}:{implementation}"
            for kernel, implementation in expected_compile
            if (kernel, implementation) not in compile_pairs
        )
        missing_verification = sorted(
            f"{kernel}:{implementation}@{size}"
            for kernel, implementation, size in expected_runtime
            if (kernel, implementation, size) not in verification_pairs
        )
        missing_runtime = sorted(
            f"{kernel}:{implementation}@{size}"
            for kernel, implementation, size in expected_runtime
            if (kernel, implementation, size) not in runtime_pairs
        )
    elif implementations:
        expected_compile = {
            (kernel, implementation)
            for kernel in expected_kernels
            for implementation in implementations
        }
        expected_runtime = {
            (kernel, implementation, size)
            for kernel in expected_kernels
            for implementation in implementations
            for size in expected_sizes
        }
        compile_pairs = {(row["kernel"], row["implementation"]) for row in compile_rows}
        verification_pairs = {
            (row["kernel"], row["implementation"], int(row["size"])) for row in verification_rows
        }
        runtime_pairs = {
            (row["kernel"], row["implementation"], int(row["size"])) for row in runtime_rows
        }
        missing_compile = sorted(
            f"{kernel}:{implementation}"
            for kernel, implementation in expected_compile
            if (kernel, implementation) not in compile_pairs
        )
        missing_verification = sorted(
            f"{kernel}:{implementation}@{size}"
            for kernel, implementation, size in expected_runtime
            if (kernel, implementation, size) not in verification_pairs
        )
        missing_runtime = sorted(
            f"{kernel}:{implementation}@{size}"
            for kernel, implementation, size in expected_runtime
            if (kernel, implementation, size) not in runtime_pairs
        )
    else:
        compile_kernels = {row["kernel"] for row in compile_rows}
        verification_pairs = {(row["kernel"], int(row["size"])) for row in verification_rows}
        runtime_pairs = {(row["kernel"], int(row["size"])) for row in runtime_rows}
        missing_compile = sorted(kernel for kernel in expected_kernels if kernel not in compile_kernels)
        missing_verification = sorted(
            f"{kernel}@{size}"
            for kernel in expected_kernels
            for size in expected_sizes
            if (kernel, size) not in verification_pairs
        )
        missing_runtime = sorted(
            f"{kernel}@{size}"
            for kernel in expected_kernels
            for size in expected_sizes
            if (kernel, size) not in runtime_pairs
        )
    if missing_compile or missing_verification or missing_runtime:
        details: list[str] = []
        if missing_compile:
            details.append("compile missing: " + ", ".join(missing_compile))
        if missing_verification:
            details.append("verification missing: " + ", ".join(missing_verification))
        if missing_runtime:
            details.append("runtime missing: " + ", ".join(missing_runtime))
        raise SystemExit("incomplete raw results; " + "; ".join(details))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--preset",
        choices=[
            "current-cuda-focused",
            "current-cuda-pass",
            "current-cuda-milestone",
            "current-triton-focused",
            "current-triton-pass",
            "current-triton-milestone",
        ],
        help="apply a named current benchmark preset",
    )
    parser.add_argument("--all", action="store_true", help="run capability, audit, tuning, verification, evaluation, plotting, and report generation")
    parser.add_argument(
        "--optimization-pass",
        action="store_true",
        help="run only candidate groups plus accepted baseline groups for optimization-pass comparison",
    )
    parser.add_argument("--candidate-group", action="append", help="implementation/group key to treat as the modified candidate")
    parser.add_argument("--baseline-group", action="append", help="implementation/group key to treat as the accepted former-self baseline")
    parser.add_argument("--implementation", action="append", help="run only selected implementation/group key(s)")
    parser.add_argument(
        "--benchmark-case",
        action="append",
        help="run only IMPLEMENTATION:KERNEL or IMPLEMENTATION:KERNEL:SIZE combinations",
    )
    parser.add_argument("--kernel", action="append", help="run only selected kernel(s)")
    parser.add_argument("--warmup", type=int)
    parser.add_argument("--runtime-repetitions", type=int)
    parser.add_argument("--compile-repetitions", type=int)
    parser.add_argument("--size", action="append", type=int, help="override held-out evaluation size(s)")
    parser.add_argument("--tuning-size", action="append", type=int, help="override tuning size(s)")
    parser.add_argument("--results-dir", type=pathlib.Path, default=RESULTS_ROOT)
    parser.add_argument("--work-dir", type=pathlib.Path, default=WORK_ROOT)
    parser.add_argument("--list-kernels", action="store_true")
    parser.add_argument(
        "--frontend-comparison",
        action="store_true",
        help="benchmark OCaml, Python, and C++ frontend lowering and compile parity separately from backend runs",
    )
    parser.add_argument(
        "--frontend-runtime",
        action="store_true",
        help="extend --frontend-comparison with isolated OCaml/Python/C++ Loom runtime benchmarks",
    )
    parser.add_argument(
        "--benchmark-set",
        choices=["all", "tuning", "held-out"],
        default="all",
        help="select the full benchmark set, current non-held-out tuning set, or held-out generalization set",
    )
    parser.add_argument("--tune-only", action="store_true")
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument(
        "--smoke-only",
        action="store_true",
        help="run compile and verification only, skipping runtime timing and excluding autotuned groups",
    )
    parser.add_argument("--plots-only", action="store_true", help="regenerate plots from existing CSV outputs")
    parser.add_argument("--finalize-only", action="store_true", help="rebuild summaries, plots, manifest, and report from existing raw outputs")
    parser.add_argument("--fixed-only", action="store_true", help="run only fixed-path Loom and external implementations")
    parser.add_argument(
        "--public-only",
        action="store_true",
        help="run only public comparison groups, excluding internal-only profiles",
    )
    parser.add_argument("--internal-triton-compile", nargs=4, metavar=("MODULE", "KERNEL", "SIZE", "SEED"))
    parser.add_argument("--internal-loom-compile-json", type=pathlib.Path)
    return parser.parse_args()


def apply_named_preset(args: argparse.Namespace) -> None:
    if args.preset is None:
        return
    if not any(
        [
            args.all,
            args.optimization_pass,
            args.smoke_only,
            args.tune_only,
            args.eval_only,
            args.audit_only,
            args.plots_only,
            args.finalize_only,
            args.fixed_only,
        ]
    ):
        args.fixed_only = True
    triton_preset = args.preset.startswith("current-triton-")
    if not args.size:
        args.size = list(CURRENT_CUDA_MILESTONE_SIZES)
    if not args.implementation:
        args.implementation = (
            list(CURRENT_TRITON_COMPARISON_IMPLEMENTATIONS)
            if triton_preset
            else list(CURRENT_CUDA_COMPARISON_IMPLEMENTATIONS)
        )
    if args.compile_repetitions is None:
        args.compile_repetitions = 1

    if args.preset in {"current-cuda-focused", "current-triton-focused"}:
        if not args.kernel:
            args.kernel = (
                list(CURRENT_TRITON_FOCUSED_KERNELS)
                if triton_preset
                else list(CURRENT_CUDA_FOCUSED_KERNELS)
            )
        if args.runtime_repetitions is None:
            args.runtime_repetitions = 8
        if args.warmup is None:
            args.warmup = 3
    elif args.preset in {"current-cuda-pass", "current-triton-pass"}:
        if not args.kernel:
            args.kernel = (
                list(CURRENT_TRITON_PASS_KERNELS)
                if triton_preset
                else list(CURRENT_CUDA_PASS_KERNELS)
            )
        if args.runtime_repetitions is None:
            args.runtime_repetitions = 8
        if args.warmup is None:
            args.warmup = 3
    elif args.preset in {"current-cuda-milestone", "current-triton-milestone"}:
        if not args.kernel:
            args.kernel = (
                list(CURRENT_TRITON_PASS_KERNELS)
                if triton_preset
                else list(CURRENT_CUDA_PASS_KERNELS)
            )
        if args.runtime_repetitions is None:
            args.runtime_repetitions = 20
        if args.warmup is None:
            args.warmup = 5


def smoke_gate_args(args: argparse.Namespace) -> list[str]:
    smoke_results_dir = args.work_dir / "smoke_gate_results"
    smoke_work_dir = args.work_dir / "smoke_gate_work"
    command = [
        sys.executable,
        str(pathlib.Path(__file__).resolve()),
        "--smoke-only",
        "--results-dir",
        str(smoke_results_dir),
        "--work-dir",
        str(smoke_work_dir),
    ]
    for kernel in args.kernel or []:
        command.extend(["--kernel", kernel])
    for size in args.size or []:
        command.extend(["--size", str(size)])
    for tuning_size in args.tuning_size or []:
        command.extend(["--tuning-size", str(tuning_size)])
    for implementation in args.implementation or []:
        command.extend(["--implementation", implementation])
    for benchmark_case in args.benchmark_case or []:
        command.extend(["--benchmark-case", benchmark_case])
    if args.benchmark_set != "all":
        command.extend(["--benchmark-set", args.benchmark_set])
    if args.public_only:
        command.append("--public-only")
    if args.fixed_only:
        command.append("--fixed-only")
    if args.warmup is not None:
        command.extend(["--warmup", str(args.warmup)])
    return command


def main() -> int:
    args = parse_args()
    apply_named_preset(args)
    if args.internal_triton_compile:
        module, kernel_name, size, seed = args.internal_triton_compile
        seconds = internal_triton_compile(pathlib.Path(module), kernel_name, int(size), int(seed))
        print(seconds)
        return 0
    if args.internal_loom_compile_json:
        seconds = internal_loom_compile_from_json(args.internal_loom_compile_json)
        print(seconds)
        return 0
    if args.frontend_runtime and not args.frontend_comparison:
        raise SystemExit("--frontend-runtime requires --frontend-comparison")
    if args.frontend_comparison:
        config = load_config()
        supported_kernel_names = [kernel["name"] for kernel in config["supported_kernels"]]
        if args.benchmark_set == "tuning" and not args.kernel:
            args.kernel = list(CURRENT_NON_HELD_OUT_KERNELS)
        elif args.benchmark_set == "held-out" and not args.kernel:
            args.kernel = list(HELD_OUT_GENERALIZATION_KERNELS)
        if args.kernel:
            unknown = sorted(set(args.kernel) - set(supported_kernel_names))
            if unknown:
                raise SystemExit("unknown kernel(s): " + ", ".join(unknown))
        run_frontend_comparison(config, args)
        return 0

    if args.optimization_pass and not (args.candidate_group and args.baseline_group):
        raise SystemExit("--optimization-pass requires at least one --candidate-group and one --baseline-group")
    if args.all and args.optimization_pass:
        raise SystemExit("--all and --optimization-pass are mutually exclusive")
    if args.smoke_only and args.optimization_pass:
        raise SystemExit("--smoke-only and --optimization-pass are mutually exclusive")
    if args.all and not args.smoke_only:
        subprocess.run(smoke_gate_args(args), cwd=ROOT, check=True)

    config = load_config()
    assert_performance_sources_are_ocaml(config)
    if args.public_only:
        public_loom_keys = {
            profile["key"]
            for profile in config["loom_profiles"]
            if "external" in profile["comparison_modes"]
        }
        config = filter_config_to_implementations(
            config,
            public_loom_keys | set(config["external_implementations"]),
        )
    fixed_path_mode = args.fixed_only or args.smoke_only or args.optimization_pass
    if fixed_path_mode:
        config["loom_profiles"] = [profile for profile in config["loom_profiles"] if not profile["autotuned"]]
        config["external_implementations"] = [
            key for key in config["external_implementations"] if not ALL_IMPLEMENTATION_MAP[key].get("autotuned", False)
        ]
        allowed = {profile["key"] for profile in config["loom_profiles"]} | set(config["external_implementations"])
        config["loom_internal_order"] = [key for key in config["loom_internal_order"] if key in allowed]
        config["loom_search_order"] = [key for key in config["loom_search_order"] if key in allowed]
        config["loom_vs_others_order"] = [key for key in config["loom_vs_others_order"] if key in allowed]
    candidate_groups = list(args.candidate_group or [])
    baseline_groups = list(args.baseline_group or [])
    if args.optimization_pass:
        require_known_implementations(candidate_groups + baseline_groups)
        autotuned_selected = [
            key for key in candidate_groups + baseline_groups if ALL_IMPLEMENTATION_MAP[key].get("autotuned", False)
        ]
        if autotuned_selected:
            raise SystemExit(
                "optimization-pass only supports fixed-path groups; remove autotuned selections: "
                + ", ".join(sorted(autotuned_selected))
            )
        config = filter_config_to_implementations(config, set(candidate_groups) | set(baseline_groups))
    run_state_path = shared_results_dir(args.results_dir) / "raw" / "run_state.json"
    if (args.plots_only or args.finalize_only) and run_state_path.exists():
        persisted_run_state = load_json(run_state_path)
        persisted_mode = str(persisted_run_state.get("mode", "full"))
        if persisted_mode in {"fixed-only", "smoke-only", "optimization-pass"}:
            config["loom_profiles"] = [profile for profile in config["loom_profiles"] if not profile["autotuned"]]
            config["external_implementations"] = [
                key
                for key in config["external_implementations"]
                if not ALL_IMPLEMENTATION_MAP[key].get("autotuned", False)
            ]
            allowed = {profile["key"] for profile in config["loom_profiles"]} | set(config["external_implementations"])
            config["loom_internal_order"] = [key for key in config["loom_internal_order"] if key in allowed]
            config["loom_search_order"] = [key for key in config["loom_search_order"] if key in allowed]
            config["loom_vs_others_order"] = [key for key in config["loom_vs_others_order"] if key in allowed]
        if persisted_mode == "optimization-pass":
            persisted_allowed = {
                str(item) for item in persisted_run_state.get("candidate_groups", [])
            } | {str(item) for item in persisted_run_state.get("baseline_groups", [])}
            config = filter_config_to_implementations(config, persisted_allowed)
        persisted_implementations = {
            str(item) for item in persisted_run_state.get("implementation_filter", [])
        }
        if persisted_implementations:
            config = filter_config_to_implementations(config, persisted_implementations)

    supported_kernel_names = [kernel["name"] for kernel in config["supported_kernels"]]
    if args.benchmark_set == "tuning":
        allowed_benchmark_set = set(CURRENT_NON_HELD_OUT_KERNELS)
    elif args.benchmark_set == "held-out":
        allowed_benchmark_set = set(HELD_OUT_GENERALIZATION_KERNELS)
    else:
        allowed_benchmark_set = set(supported_kernel_names)
    if args.kernel:
        outside = sorted(set(args.kernel) - allowed_benchmark_set)
        if outside and args.benchmark_set != "all":
            raise SystemExit(
                f"--benchmark-set {args.benchmark_set} excludes kernel(s): "
                + ", ".join(outside)
            )
    elif args.benchmark_set != "all":
        args.kernel = [name for name in supported_kernel_names if name in allowed_benchmark_set]
    benchmark_cases = parse_benchmark_cases(args.benchmark_case, set(supported_kernel_names))
    implementation_filter = list(dict.fromkeys(args.implementation or []))
    if implementation_filter:
        require_known_implementations(implementation_filter)
    if benchmark_cases:
        case_implementations = {implementation for implementation, _ in benchmark_cases}
        if implementation_filter:
            missing_cases = sorted(case_implementations - set(implementation_filter))
            if missing_cases:
                raise SystemExit(
                    "--benchmark-case implementation(s) not present in --implementation: "
                    + ", ".join(missing_cases)
                )
        else:
            implementation_filter = sorted(case_implementations)
        case_kernels = {kernel for _, kernel in benchmark_cases}
        if args.kernel:
            missing_kernels = sorted(case_kernels - set(args.kernel))
            if missing_kernels:
                raise SystemExit(
                    "--benchmark-case kernel(s) not present in --kernel: "
                    + ", ".join(missing_kernels)
                )
        else:
            args.kernel = sorted(case_kernels)
        case_sizes = sorted(
            {
                size
                for sizes in benchmark_cases.values()
                if sizes is not None
                for size in sizes
            }
        )
        if case_sizes:
            args.size = sorted(set(args.size or []) | set(case_sizes))
    if implementation_filter:
        config = filter_config_to_implementations(config, set(implementation_filter))
    if args.list_kernels:
        for name in supported_kernel_names:
            print(name)
        return 0

    kernels = args.kernel if args.kernel else supported_kernel_names
    if args.plots_only:
        regenerate_plots(
            config,
            args.results_dir,
            kernels,
            args.size if args.size else list(config["evaluation_sizes"]),
        )
        return 0
    if args.finalize_only:
        finalize_results(
            config,
            args.results_dir,
            kernels,
            args.size if args.size else list(config["evaluation_sizes"]),
        )
        return 0

    run_tuning_flag = args.all or args.tune_only or args.smoke_only or args.optimization_pass or not args.eval_only
    run_compile_flag = args.all or args.eval_only or args.smoke_only or args.optimization_pass or not args.tune_only
    run_verification_flag = run_compile_flag
    run_runtime_flag = False if args.smoke_only else (args.all or args.eval_only or args.optimization_pass or not args.tune_only)
    default_runtime_repetitions = 5 if args.optimization_pass else int(config["runtime_repetitions"])
    default_compile_repetitions = 1 if (args.optimization_pass or args.smoke_only) else int(config["compile_repetitions"])
    run_mode = (
        "optimization-pass"
        if args.optimization_pass
        else "smoke-only"
        if args.smoke_only
        else "fixed-only"
        if fixed_path_mode
        else "public"
        if args.public_only
        else "full"
    )
    options = RunOptions(
        kernels=kernels,
        warmup=args.warmup if args.warmup is not None else int(config["warmup_repetitions"]),
        runtime_repetitions=args.runtime_repetitions
        if args.runtime_repetitions is not None
        else default_runtime_repetitions,
        compile_repetitions=args.compile_repetitions
        if args.compile_repetitions is not None
        else default_compile_repetitions,
        tuning_sizes=args.tuning_size if args.tuning_size else list(config["tuning_sizes"]),
        evaluation_sizes=args.size if args.size else list(config["evaluation_sizes"]),
        tuning_seed_offsets=list(config["tuning_seed_offsets"]),
        evaluation_seed_offsets=list(config["evaluation_seed_offsets"]),
        results_dir=args.results_dir,
        work_dir=args.work_dir,
        run_tuning=run_tuning_flag,
        run_compile=run_compile_flag,
        run_verification=run_verification_flag,
        run_runtime=run_runtime_flag,
        mode=run_mode,
        candidate_groups=candidate_groups,
        baseline_groups=baseline_groups,
        implementation_filter=implementation_filter,
        benchmark_cases=benchmark_cases,
    )

    prepare_results_tree(options.results_dir)
    ensure_dir(options.work_dir)
    writers = build_raw_result_writers(options.results_dir)
    seen_dataset_ids: set[str] = set()
    write_json(
        shared_results_dir(options.results_dir) / "raw" / "run_state.json",
        {
            "mode": options.mode,
            "fixed_only": fixed_path_mode,
            "kernels": kernels,
            "warmup": options.warmup,
            "runtime_repetitions": options.runtime_repetitions,
            "compile_repetitions": options.compile_repetitions,
            "tuning_sizes": options.tuning_sizes,
            "evaluation_sizes": options.evaluation_sizes,
            "tuning_seed_offsets": options.tuning_seed_offsets,
            "evaluation_seed_offsets": options.evaluation_seed_offsets,
            "candidate_groups": options.candidate_groups,
            "baseline_groups": options.baseline_groups,
            "implementation_filter": options.implementation_filter,
            "benchmark_cases": benchmark_cases_to_json(options.benchmark_cases),
            "benchmark_set": args.benchmark_set,
            "comparison_reference_revision": current_git_revision(),
        },
    )

    audit_report = audit_workloads(config, shared_results_dir(options.results_dir))
    audit_cuda_backend_generalizability(shared_results_dir(options.results_dir))
    if args.audit_only:
        return 0
    ensure_cuda_environment()
    ensure_loom_built()
    environment = dump_environment_info(shared_results_dir(options.results_dir))
    capability_rows = run_capability_checks(
        config,
        shared_results_dir(options.results_dir),
        options.work_dir,
        plot_results_dir=runtime_results_dir(options.results_dir),
    )

    tuning_decisions: dict[str, Any] = {}
    tuned_runners: dict[str, dict[str, PythonRuntime]] = {}
    if options.run_tuning:
        tuning_decisions, tuned_runners = run_tuning(config, options, writers, seen_dataset_ids)
    else:
        _, tuned_runners = run_tuning(
            config,
            RunOptions(
                **{
                    **options.__dict__,
                    "runtime_repetitions": 0,
                    "compile_repetitions": 0,
                    "run_tuning": True,
                    "run_compile": False,
                    "run_verification": False,
                    "run_runtime": False,
                    "implementation_filter": options.implementation_filter,
                    "benchmark_cases": options.benchmark_cases,
                }
            ),
            writers,
            seen_dataset_ids,
        )
    write_json(runtime_results_dir(options.results_dir) / "raw" / "tuning_decisions.json", tuning_decisions)

    verification_failures: list[dict[str, Any]] = []
    if options.run_compile or options.run_verification or options.run_runtime:
        verification_failures = run_performance(config, options, tuned_runners, writers, seen_dataset_ids)
        if verification_failures:
            raise SystemExit("verification failed; see runtime/raw/verification_failures.json")

    tuning_rows = read_csv_if_exists(runtime_results_dir(options.results_dir) / "raw" / "tuning_measurements.csv")
    compile_rows = read_csv_if_exists(compile_results_dir(options.results_dir) / "raw" / "compile_measurements.csv")
    runtime_rows = read_csv_if_exists(runtime_results_dir(options.results_dir) / "raw" / "runtime_measurements.csv")
    verification_rows = read_csv_if_exists(
        runtime_results_dir(options.results_dir) / "raw" / "verification_measurements.csv"
    )
    summary_rows: list[dict[str, Any]] = []
    dataset_manifest_rows = finalize_dataset_manifest(options.results_dir)
    if options.run_runtime:
        summary_rows = summarize_runtime(runtime_rows, capability_rows)
        write_csv(
            runtime_results_dir(options.results_dir) / "summaries" / "summary.csv",
            SUMMARY_FIELDS,
            summary_rows,
        )
        write_csv(
            runtime_results_dir(options.results_dir) / "summaries" / "loom_internal_summary.csv",
            SUMMARY_FIELDS,
            filter_summary_rows(summary_rows, tuple(config["loom_internal_order"])),
        )
        write_csv(
            runtime_results_dir(options.results_dir) / "summaries" / "loom_vs_others_summary.csv",
            SUMMARY_FIELDS,
            filter_summary_rows(summary_rows, tuple(config["loom_vs_others_order"])),
        )
        write_csv(
            runtime_results_dir(options.results_dir) / "summaries" / "loom_search_summary.csv",
            SUMMARY_FIELDS,
            filter_summary_rows(summary_rows, tuple(config.get("loom_search_order", []))),
        )
        gap_rows = summarize_gap_vs_best_external(
            summary_rows,
            "loom_full_fixed",
            tuple(config["loom_vs_others_order"]),
        )
        write_csv(
            runtime_results_dir(options.results_dir) / "summaries" / "loom_vs_best_external_gap.csv",
            GAP_FIELDS,
            gap_rows,
        )
        write_csv(
            runtime_results_dir(options.results_dir) / "summaries" / "full_suite_gap_by_class.csv",
            GAP_BY_CLASS_FIELDS,
            summarize_gap_by_class(gap_rows),
        )
        write_csv(
            runtime_results_dir(options.results_dir) / "summaries" / "full_suite_top_losses.csv",
            GAP_FIELDS,
            summarize_top_losses(gap_rows),
        )
        write_generalization_summaries(summary_rows, options.results_dir)
        write_current_cuda_summaries(summary_rows, options.results_dir)
        write_current_triton_summaries(summary_rows, options.results_dir)
        if options.mode == "optimization-pass":
            write_csv(
                runtime_results_dir(options.results_dir) / "summaries" / "optimization_progress.csv",
                OPTIMIZATION_PROGRESS_FIELDS,
                summarize_optimization_progress(summary_rows, options.candidate_groups, options.baseline_groups),
            )
        kernel_configs = kernel_config_map(config["supported_kernels"])
        plot_compile_boxplots(
            compile_rows,
            compile_results_dir(options.results_dir),
            kernels,
            kernel_configs,
            tuple(config["loom_internal_order"]),
            "loom_internal",
        )
        plot_compile_boxplots(
            compile_rows,
            compile_results_dir(options.results_dir),
            kernels,
            kernel_configs,
            tuple(config["loom_vs_others_order"]),
            "loom_vs_others",
        )
        plot_runtime_boxplots(
            runtime_rows,
            runtime_results_dir(options.results_dir),
            kernels,
            options.evaluation_sizes,
            kernel_configs,
            tuple(config["loom_internal_order"]),
            "loom_internal",
        )
        plot_runtime_boxplots(
            runtime_rows,
            runtime_results_dir(options.results_dir),
            kernels,
            options.evaluation_sizes,
            kernel_configs,
            tuple(config["loom_vs_others_order"]),
            "loom_vs_others",
        )
        plot_suite_summary(
            summary_rows,
            runtime_results_dir(options.results_dir),
            kernels,
            options.evaluation_sizes,
        )
        generalization_kernels = [
            kernel for kernel in kernels if kernel in set(HELD_OUT_GENERALIZATION_KERNELS)
        ]
        if generalization_kernels:
            plot_suite_summary(
                filter_generalization_rows(summary_rows),
                runtime_results_dir(options.results_dir),
                generalization_kernels,
                options.evaluation_sizes,
                output_name="suite_summary_generalization_gap.png",
                title="Generalization summary: held-out Loom backend runtime gaps vs best external Triton/CUDA",
            )
    generate_report(
        {
            **config,
            "run_mode": options.mode,
            "warmup_repetitions": options.warmup,
            "runtime_repetitions": options.runtime_repetitions,
            "compile_repetitions": options.compile_repetitions,
            "evaluation_sizes": options.evaluation_sizes,
            "tuning_sizes": options.tuning_sizes,
            "tuning_seed_offsets": options.tuning_seed_offsets,
            "evaluation_seed_offsets": options.evaluation_seed_offsets,
            "candidate_groups": options.candidate_groups,
            "baseline_groups": options.baseline_groups,
            "implementation_filter": options.implementation_filter,
            "benchmark_cases": benchmark_cases_to_json(options.benchmark_cases),
        },
        environment,
        audit_report,
        verification_rows,
        dataset_manifest_rows,
        tuning_rows,
        tuning_decisions,
        summary_rows,
        runtime_results_dir(options.results_dir) / "report.md",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
