import pathlib
import shutil
import subprocess
import tempfile
import unittest
import csv
import json

try:
    import torch
except Exception:
    torch = None


ROOT = pathlib.Path(__file__).resolve().parents[2]
BIN = ROOT / "_build" / "default" / "src" / "loom_cli" / "main.exe"


def run_cmd(*args):
    return subprocess.run(args, cwd=ROOT, capture_output=True, text=True)


class LoomExperimentsTests(unittest.TestCase):
    def test_list_kernels(self):
        result = run_cmd("uv", "run", "python", "experiments/source/harness/run.py", "--list-kernels")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("saxpy", result.stdout)
        self.assertIn("dot", result.stdout)
        self.assertIn("saxpy_curried", result.stdout)
        self.assertIn("book_imbalance", result.stdout)
        self.assertIn("ratio_weighted_sum", result.stdout)

    def test_performance_workloads_use_ocaml_sources(self):
        workload_dir = ROOT / "experiments" / "source" / "data" / "workloads"
        for path in sorted(workload_dir.glob("*.json")):
            payload = json.loads(path.read_text())
            if payload.get("kind") != "benchmark":
                continue
            self.assertEqual(payload.get("input_kind"), "ocaml", payload["name"])
            self.assertIn("source/ocaml/", payload.get("source_path", ""), payload["name"])

    def test_python_frontend_supported_workloads_compile(self):
        tmp = pathlib.Path(tempfile.mkdtemp(prefix="loom_python_frontend_parity_"))
        try:
            workload_dir = ROOT / "experiments" / "source" / "data" / "workloads"
            workloads = []
            for path in sorted(workload_dir.glob("*.json")):
                payload = json.loads(path.read_text())
                if payload.get("kind") == "benchmark":
                    workloads.append(payload)
            self.assertTrue(workloads)
            for workload in workloads:
                source = pathlib.Path(str(workload["source_path"]))
                parts = list(source.parts)
                parts[parts.index("ocaml")] = "python"
                python_source = (ROOT / "experiments" / pathlib.Path(*parts)).with_suffix(".py")
                out_dir = tmp / workload["name"]
                result = run_cmd(
                    str(BIN),
                    "compile",
                    str(python_source),
                    "--input-kind",
                    "python",
                    "--entry",
                    workload["entry"],
                    "--target",
                    "triton",
                    "--out",
                    str(out_dir),
                    "--emit",
                    "all",
                )
                self.assertEqual(result.returncode, 0, f"{workload['name']}: {result.stderr}")
                self.assertTrue((out_dir / "tensor_ir.json").exists(), workload["name"])
                self.assertTrue((out_dir / "kernel_plan.json").exists(), workload["name"])
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_cpp_frontend_supported_workloads_compile(self):
        tmp = pathlib.Path(tempfile.mkdtemp(prefix="loom_cpp_frontend_parity_"))
        try:
            workload_dir = ROOT / "experiments" / "source" / "data" / "workloads"
            workloads = []
            for path in sorted(workload_dir.glob("*.json")):
                payload = json.loads(path.read_text())
                if payload.get("kind") == "benchmark":
                    workloads.append(payload)
            self.assertTrue(workloads)
            for workload in workloads:
                source = pathlib.Path(str(workload["source_path"]))
                parts = list(source.parts)
                parts[parts.index("ocaml")] = "cpp"
                cpp_source = (ROOT / "experiments" / pathlib.Path(*parts)).with_suffix(".cpp")
                out_dir = tmp / workload["name"]
                result = run_cmd(
                    str(BIN),
                    "compile",
                    str(cpp_source),
                    "--input-kind",
                    "cpp",
                    "--entry",
                    workload["entry"],
                    "--target",
                    "triton",
                    "--out",
                    str(out_dir),
                    "--emit",
                    "all",
                )
                self.assertEqual(result.returncode, 0, f"{workload['name']}: {result.stderr}")
                self.assertTrue((out_dir / "tensor_ir.json").exists(), workload["name"])
                self.assertTrue((out_dir / "kernel_plan.json").exists(), workload["name"])
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_python_frontend_limitations_are_rejected(self):
        tmp = pathlib.Path(tempfile.mkdtemp(prefix="loom_python_frontend_negative_"))
        try:
            cases = [
                "prefix_sum",
                "recursive",
                "rolling_mean",
                "tensor_capture",
                "tuple_output",
                "unknown_call",
            ]
            for name in cases:
                result = run_cmd(
                    str(BIN),
                    "compile",
                    str(ROOT / "experiments" / "source" / "python" / "limitations" / f"{name}.py"),
                    "--input-kind",
                    "python",
                    "--entry",
                    "bad",
                    "--target",
                    "triton",
                    "--out",
                    str(tmp / name),
                    "--emit",
                    "all",
                )
                self.assertNotEqual(result.returncode, 0, name)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_cpp_frontend_limitations_are_rejected(self):
        tmp = pathlib.Path(tempfile.mkdtemp(prefix="loom_cpp_frontend_negative_"))
        try:
            cases = [
                "prefix_sum",
                "recursive",
                "rolling_mean",
                "tensor_capture",
                "tuple_output",
                "unknown_call",
            ]
            for name in cases:
                result = run_cmd(
                    str(BIN),
                    "compile",
                    str(ROOT / "experiments" / "source" / "cpp" / "limitations" / f"{name}.cpp"),
                    "--input-kind",
                    "cpp",
                    "--entry",
                    "bad",
                    "--target",
                    "triton",
                    "--out",
                    str(tmp / name),
                    "--emit",
                    "all",
                )
                self.assertNotEqual(result.returncode, 0, name)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_python_frontend_comparison_mode(self):
        tmp = pathlib.Path(tempfile.mkdtemp(prefix="loom_frontend_comparison_"))
        try:
            results_dir = tmp / "results"
            work_dir = tmp / "work"
            result = run_cmd(
                "uv",
                "run",
                "python",
                "experiments/source/harness/run.py",
                "--frontend-comparison",
                "--kernel",
                "saxpy",
                "--compile-repetitions",
                "1",
                "--results-dir",
                str(results_dir),
                "--work-dir",
                str(work_dir),
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertTrue((results_dir / "frontend" / "raw" / "frontend_measurements.csv").exists())
            self.assertTrue((results_dir / "frontend" / "summaries" / "frontend_summary.csv").exists())
            parity = (results_dir / "frontend" / "summaries" / "frontend_parity.csv").read_text()
            self.assertIn("saxpy", parity)
            self.assertIn("python", parity)
            self.assertIn("cpp", parity)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    @unittest.skipIf(torch is None or not torch.cuda.is_available(), "CUDA runtime unavailable")
    def test_python_frontend_runtime_comparison_mode(self):
        tmp = pathlib.Path(tempfile.mkdtemp(prefix="loom_frontend_runtime_comparison_"))
        try:
            results_dir = tmp / "results"
            work_dir = tmp / "work"
            result = run_cmd(
                "uv",
                "run",
                "python",
                "experiments/source/harness/run.py",
                "--frontend-comparison",
                "--frontend-runtime",
                "--kernel",
                "saxpy",
                "--size",
                "4096",
                "--compile-repetitions",
                "1",
                "--runtime-repetitions",
                "1",
                "--warmup",
                "1",
                "--results-dir",
                str(results_dir),
                "--work-dir",
                str(work_dir),
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertTrue((results_dir / "frontend" / "raw" / "frontend_runtime_measurements.csv").exists())
            self.assertTrue((results_dir / "frontend" / "raw" / "frontend_runtime_verification.csv").exists())
            self.assertTrue((results_dir / "frontend" / "summaries" / "frontend_runtime_summary.csv").exists())
            self.assertTrue((results_dir / "frontend" / "plots" / "frontend_runtime_ratio.png").exists())
            verification = (results_dir / "frontend" / "raw" / "frontend_runtime_verification.csv").read_text()
            self.assertIn("python", verification)
            self.assertIn("cpp", verification)
            self.assertIn("pass", verification)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    @unittest.skipIf(torch is None or not torch.cuda.is_available(), "CUDA runtime unavailable")
    def test_experiment_smoke(self):
        tmp = pathlib.Path(tempfile.mkdtemp(prefix="loom_experiments_"))
        try:
            results_dir = tmp / "results"
            work_dir = tmp / "work"
            result = run_cmd(
                "uv",
                "run",
                "python",
                "experiments/source/harness/run.py",
                "--kernel",
                "saxpy",
                "--warmup",
                "1",
                "--runtime-repetitions",
                "2",
                "--compile-repetitions",
                "1",
                "--size",
                "4096",
                "--tuning-size",
                "2048",
                "--results-dir",
                str(results_dir),
                "--work-dir",
                str(work_dir),
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertTrue((results_dir / "compile" / "raw" / "compile_measurements.csv").exists())
            self.assertTrue((results_dir / "runtime" / "raw" / "runtime_measurements.csv").exists())
            self.assertTrue((results_dir / "runtime" / "raw" / "tuning_measurements.csv").exists())
            self.assertTrue((results_dir / "runtime" / "raw" / "tuning_decisions.json").exists())
            self.assertTrue((results_dir / "runtime" / "raw" / "verification_measurements.csv").exists())
            self.assertTrue((results_dir / "runtime" / "raw" / "verification_failures.json").exists())
            self.assertTrue((results_dir / "shared" / "raw" / "capability_checks.csv").exists())
            self.assertTrue((results_dir / "shared" / "raw" / "audit_report.json").exists())
            self.assertTrue((results_dir / "shared" / "raw" / "dataset_manifest.json").exists())
            self.assertTrue((results_dir / "shared" / "raw" / "environment.json").exists())
            self.assertTrue((results_dir / "shared" / "raw" / "environment.txt").exists())
            self.assertTrue((results_dir / "runtime" / "plots" / "loom_internal" / "saxpy_runtime_boxplot.png").exists())
            self.assertTrue((results_dir / "runtime" / "plots" / "loom_vs_others" / "saxpy_runtime_boxplot.png").exists())
            self.assertTrue((results_dir / "runtime" / "plots" / "capability_summary.png").exists())
            self.assertTrue((results_dir / "runtime" / "summaries" / "summary.csv").exists())
            self.assertTrue((results_dir / "runtime" / "summaries" / "loom_internal_summary.csv").exists())
            self.assertTrue((results_dir / "runtime" / "summaries" / "loom_vs_others_summary.csv").exists())
            self.assertTrue((results_dir / "runtime" / "summaries" / "loom_search_summary.csv").exists())
            self.assertTrue((results_dir / "runtime" / "summaries" / "loom_vs_best_external_gap.csv").exists())
            self.assertTrue((results_dir / "runtime" / "summaries" / "full_suite_gap_by_class.csv").exists())
            self.assertTrue((results_dir / "runtime" / "summaries" / "full_suite_top_losses.csv").exists())
            self.assertTrue((results_dir / "runtime" / "report.md").exists())
            with open(results_dir / "compile" / "raw" / "compile_measurements.csv", newline="", encoding="utf-8") as handle:
                compile_rows = list(csv.DictReader(handle))
            with open(results_dir / "runtime" / "raw" / "runtime_measurements.csv", newline="", encoding="utf-8") as handle:
                runtime_rows = list(csv.DictReader(handle))
            compile_impls = {row["implementation"] for row in compile_rows if row["kernel"] == "saxpy"}
            runtime_impls = {row["implementation"] for row in runtime_rows if row["kernel"] == "saxpy"}
            external_expected = {
                "loom_none_fixed",
                "loom_none_autotuned",
                "loom_full_fixed",
                "loom_full_autotuned",
                "loom_cuda_none_fixed",
                "loom_cuda_fixed",
                "triton_naive_fixed",
                "triton_naive_autotuned",
                "triton_optimized_fixed",
                "triton_optimized_autotuned",
                "cuda_naive",
                "cuda_optimized",
            }
            self.assertTrue(external_expected.issubset(compile_impls))
            self.assertTrue(external_expected.issubset(runtime_impls))
            self.assertIn("loom_reduction_core_fixed", compile_impls)
            self.assertIn("loom_pointwise_guarded_fixed", runtime_impls)
            self.assertIn("torch_version", (results_dir / "shared" / "raw" / "environment.json").read_text())
            self.assertIn("\"status\": \"passed\"", (results_dir / "shared" / "raw" / "audit_report.json").read_text())
            self.assertIn("application_domain", (results_dir / "runtime" / "summaries" / "summary.csv").read_text())
            self.assertIn("optimization_flags", (results_dir / "runtime" / "summaries" / "summary.csv").read_text())
            self.assertIn("dataset_id", (results_dir / "runtime" / "raw" / "verification_measurements.csv").read_text())
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    @unittest.skipIf(torch is None or not torch.cuda.is_available(), "CUDA runtime unavailable")
    def test_l2_norm_sq_tune_only_autotuned_path(self):
        tmp = pathlib.Path(tempfile.mkdtemp(prefix="loom_experiments_tune_only_"))
        try:
            results_dir = tmp / "results"
            work_dir = tmp / "work"
            result = run_cmd(
                "uv",
                "run",
                "python",
                "experiments/source/harness/run.py",
                "--tune-only",
                "--kernel",
                "l2_norm_sq",
                "--tuning-size",
                "65536",
                "--results-dir",
                str(results_dir),
                "--work-dir",
                str(work_dir),
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertTrue((results_dir / "runtime" / "raw" / "tuning_measurements.csv").exists())
            tuning_csv = (results_dir / "runtime" / "raw" / "tuning_measurements.csv").read_text()
            self.assertIn("loom_full_autotuned", tuning_csv)
            self.assertIn("triton_optimized_autotuned", tuning_csv)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    @unittest.skipIf(torch is None or not torch.cuda.is_available(), "CUDA runtime unavailable")
    def test_smoke_only_skips_runtime_and_autotuned_groups(self):
        tmp = pathlib.Path(tempfile.mkdtemp(prefix="loom_experiments_smoke_only_"))
        try:
            results_dir = tmp / "results"
            work_dir = tmp / "work"
            result = run_cmd(
                "uv",
                "run",
                "python",
                "experiments/source/harness/run.py",
                "--smoke-only",
                "--kernel",
                "saxpy",
                "--size",
                "4096",
                "--tuning-size",
                "2048",
                "--results-dir",
                str(results_dir),
                "--work-dir",
                str(work_dir),
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            with open(
                results_dir / "runtime" / "raw" / "verification_measurements.csv",
                newline="",
                encoding="utf-8",
            ) as handle:
                verification_rows = list(csv.DictReader(handle))
            if (results_dir / "runtime" / "raw" / "runtime_measurements.csv").exists():
                with open(
                    results_dir / "runtime" / "raw" / "runtime_measurements.csv",
                    newline="",
                    encoding="utf-8",
                ) as handle:
                    runtime_rows = list(csv.DictReader(handle))
            else:
                runtime_rows = []
            self.assertTrue(verification_rows)
            self.assertFalse(runtime_rows)
            self.assertTrue(all(row["autotuned"] == "no" for row in verification_rows))
            run_state = (results_dir / "shared" / "raw" / "run_state.json").read_text()
            self.assertIn('"mode": "smoke-only"', run_state)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    @unittest.skipIf(torch is None or not torch.cuda.is_available(), "CUDA runtime unavailable")
    def test_optimization_pass_tracks_only_candidate_vs_baseline(self):
        tmp = pathlib.Path(tempfile.mkdtemp(prefix="loom_experiments_opt_pass_"))
        try:
            results_dir = tmp / "results"
            work_dir = tmp / "work"
            result = run_cmd(
                "uv",
                "run",
                "python",
                "experiments/source/harness/run.py",
                "--optimization-pass",
                "--candidate-group",
                "loom_full_fixed",
                "--baseline-group",
                "loom_none_fixed",
                "--kernel",
                "saxpy",
                "--size",
                "4096",
                "--tuning-size",
                "2048",
                "--runtime-repetitions",
                "2",
                "--compile-repetitions",
                "1",
                "--results-dir",
                str(results_dir),
                "--work-dir",
                str(work_dir),
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            with open(
                results_dir / "runtime" / "raw" / "runtime_measurements.csv",
                newline="",
                encoding="utf-8",
            ) as handle:
                runtime_rows = list(csv.DictReader(handle))
            implementations = {row["implementation"] for row in runtime_rows}
            self.assertEqual(implementations, {"loom_full_fixed", "loom_none_fixed"})
            progress_csv = (results_dir / "runtime" / "summaries" / "optimization_progress.csv").read_text()
            self.assertIn("speedup_vs_baseline", progress_csv)
            self.assertIn("loom_full_fixed", progress_csv)
            run_state = (results_dir / "shared" / "raw" / "run_state.json").read_text()
            self.assertIn('"mode": "optimization-pass"', run_state)
            self.assertIn('"candidate_groups"', run_state)
            self.assertIn('"baseline_groups"', run_state)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    @unittest.skipIf(torch is None or not torch.cuda.is_available(), "CUDA runtime unavailable")
    def test_current_cuda_pass_preset_applies_fast_loop_defaults(self):
        tmp = pathlib.Path(tempfile.mkdtemp(prefix="loom_experiments_current_cuda_"))
        try:
            results_dir = tmp / "results"
            work_dir = tmp / "work"
            result = run_cmd(
                "uv",
                "run",
                "python",
                "experiments/source/harness/run.py",
                "--preset",
                "current-cuda-pass",
                "--implementation",
                "loom_cuda_fixed",
                "--kernel",
                "saxpy",
                "--size",
                "4096",
                "--tuning-size",
                "2048",
                "--results-dir",
                str(results_dir),
                "--work-dir",
                str(work_dir),
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            run_state = json.loads((results_dir / "shared" / "raw" / "run_state.json").read_text())
            self.assertEqual(run_state["mode"], "fixed-only")
            self.assertEqual(run_state["runtime_repetitions"], 8)
            self.assertEqual(run_state["compile_repetitions"], 1)
            self.assertEqual(run_state["warmup"], 3)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    @unittest.skipIf(torch is None or not torch.cuda.is_available(), "CUDA runtime unavailable")
    def test_current_cuda_focused_preset_defaults_to_current_groups(self):
        tmp = pathlib.Path(tempfile.mkdtemp(prefix="loom_experiments_current_cuda_"))
        try:
            results_dir = tmp / "results"
            work_dir = tmp / "work"
            result = run_cmd(
                "uv",
                "run",
                "python",
                "experiments/source/harness/run.py",
                "--preset",
                "current-cuda-focused",
                "--smoke-only",
                "--kernel",
                "saxpy",
                "--size",
                "4096",
                "--tuning-size",
                "2048",
                "--results-dir",
                str(results_dir),
                "--work-dir",
                str(work_dir),
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            run_state = json.loads((results_dir / "shared" / "raw" / "run_state.json").read_text())
            self.assertEqual(run_state["implementation_filter"], [
                "loom_cuda_fixed",
                "loom_full_fixed",
                "triton_naive_fixed",
                "triton_optimized_fixed",
                "cuda_naive",
                "cuda_optimized",
            ])
            self.assertEqual(run_state["runtime_repetitions"], 8)
            self.assertEqual(run_state["compile_repetitions"], 1)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    @unittest.skipIf(torch is None or not torch.cuda.is_available(), "CUDA runtime unavailable")
    def test_current_triton_focused_preset_defaults_to_current_groups(self):
        tmp = pathlib.Path(tempfile.mkdtemp(prefix="loom_experiments_current_triton_"))
        try:
            results_dir = tmp / "results"
            work_dir = tmp / "work"
            result = run_cmd(
                "uv",
                "run",
                "python",
                "experiments/source/harness/run.py",
                "--preset",
                "current-triton-focused",
                "--smoke-only",
                "--kernel",
                "dot",
                "--size",
                "4096",
                "--tuning-size",
                "2048",
                "--results-dir",
                str(results_dir),
                "--work-dir",
                str(work_dir),
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            run_state = json.loads((results_dir / "shared" / "raw" / "run_state.json").read_text())
            self.assertEqual(run_state["implementation_filter"], [
                "loom_full_fixed",
                "loom_triton_previous_fixed",
                "triton_naive_fixed",
                "triton_optimized_fixed",
            ])
            self.assertEqual(run_state["runtime_repetitions"], 8)
            self.assertEqual(run_state["compile_repetitions"], 1)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
