import ctypes
import importlib.util
import json
import pathlib
import shutil
import subprocess
import sys
import tempfile
import unittest

import numpy as np

try:
    import torch
except Exception:
    torch = None


ROOT = pathlib.Path(__file__).resolve().parents[2]
BIN = ROOT / "_build" / "default" / "src" / "loom_cli" / "main.exe"


def run_cmd(*args):
    return subprocess.run(args, cwd=ROOT, capture_output=True, text=True)


class LoomCliTests(unittest.TestCase):
    def test_golden_outputs(self):
        out_dir = pathlib.Path(tempfile.mkdtemp(prefix="loom_test_golden_"))
        try:
            saxpy = run_cmd(
                str(BIN),
                "compile",
                "examples/saxpy.ml",
                "--entry",
                "saxpy",
                "--target",
                "triton",
                "--out",
                str(out_dir / "saxpy"),
                "--emit",
                "all",
            )
            self.assertEqual(saxpy.returncode, 0, saxpy.stderr)
            l2 = run_cmd(
                str(BIN),
                "compile",
                "examples/l2_norm_sq.ml",
                "--entry",
                "l2_norm_sq",
                "--target",
                "triton",
                "--out",
                str(out_dir / "l2_norm_sq"),
                "--emit",
                "all",
            )
            self.assertEqual(l2.returncode, 0, l2.stderr)
            self.assertEqual(
                (out_dir / "saxpy" / "loom_lambda.json").read_text().strip(),
                (ROOT / "test" / "golden" / "saxpy_loom_lambda.json").read_text().strip(),
            )
            self.assertEqual(
                json.loads((out_dir / "l2_norm_sq" / "tensor_ir.json").read_text()),
                json.loads((ROOT / "test" / "golden" / "l2_norm_sq_tensor_ir.json").read_text()),
            )
        finally:
            shutil.rmtree(out_dir, ignore_errors=True)

    def test_list_entries(self):
        result = run_cmd(str(BIN), "list-entries", "examples/saxpy.ml")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("saxpy", result.stdout)

    def test_python_list_entries(self):
        result = run_cmd(str(BIN), "list-entries", "examples/saxpy.py", "--input-kind", "python")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("saxpy", result.stdout)
        self.assertIn("a:scalar-f32", result.stdout)
        self.assertIn("x:tensor1-f32", result.stdout)

    def test_python_front_ir_command(self):
        result = run_cmd(
            str(BIN),
            "front-ir",
            "examples/saxpy.py",
            "--input-kind",
            "python",
            "--entry",
            "saxpy",
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        payload = json.loads(result.stdout)
        self.assertEqual(payload["entry"], "saxpy")
        self.assertEqual(payload["params"][0]["type"], "float")
        self.assertEqual(payload["params"][1]["type"], "tensor1<f32>")

    def test_cpp_list_entries(self):
        result = run_cmd(str(BIN), "list-entries", "examples/saxpy.cpp", "--input-kind", "cpp")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("saxpy", result.stdout)
        self.assertIn("a:scalar-f32", result.stdout)
        self.assertIn("x:tensor1-f32", result.stdout)

    def test_cpp_front_ir_command(self):
        result = run_cmd(
            str(BIN),
            "front-ir",
            "examples/saxpy.cpp",
            "--input-kind",
            "cpp",
            "--entry",
            "saxpy",
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        payload = json.loads(result.stdout)
        self.assertEqual(payload["entry"], "saxpy")
        self.assertEqual(payload["params"][0]["type"], "float")
        self.assertEqual(payload["params"][1]["type"], "tensor1<f32>")

    def test_list_opts(self):
        result = run_cmd(str(BIN), "list-opts")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("Optimization: [Scalar Constant Folding]", result.stdout)
        self.assertIn("--enable-opt elementwise-fusion", result.stdout)
        self.assertIn("Optimization: [Reduce Map Fusion]", result.stdout)
        self.assertIn("--enable-opt reduction-tree-plan", result.stdout)
        self.assertIn("Optimization: [Materialization Choice]", result.stdout)
        self.assertIn("--enable-opt reduction-stage-sizing", result.stdout)

    def test_compile_emits_artifacts(self):
        out_dir = pathlib.Path(tempfile.mkdtemp(prefix="loom_test_"))
        try:
            result = run_cmd(
                str(BIN),
                "compile",
                "examples/saxpy.ml",
                "--entry",
                "saxpy",
                "--target",
                "triton",
                "--out",
                str(out_dir),
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            expected = {
                "front_ir.json",
                "lambda.sexp",
                "loom_lambda.json",
                "tensor_ir.json",
                "kernel_plan.json",
                "triton_plan.json",
                "backend_analysis.json",
                "pipeline.json",
                "manifest.json",
                "report.md",
                "saxpy_triton.py",
            }
            self.assertTrue(expected.issubset({p.name for p in out_dir.iterdir()}))
            manifest = json.loads((out_dir / "manifest.json").read_text())
            self.assertEqual(manifest["target_backend"], "triton")
            pipeline = json.loads((out_dir / "pipeline.json").read_text())
            self.assertIn("kernel_plan", pipeline)
            self.assertIn("triton_plan", pipeline)
        finally:
            shutil.rmtree(out_dir, ignore_errors=True)

    def test_python_compile_emits_artifacts(self):
        out_dir = pathlib.Path(tempfile.mkdtemp(prefix="loom_test_python_"))
        try:
            result = run_cmd(
                str(BIN),
                "compile",
                "examples/saxpy.py",
                "--input-kind",
                "python",
                "--entry",
                "saxpy",
                "--target",
                "triton",
                "--out",
                str(out_dir),
                "--emit",
                "all",
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertTrue((out_dir / "front_ir.json").exists())
            self.assertTrue((out_dir / "tensor_ir.json").exists())
            self.assertTrue((out_dir / "kernel_plan.json").exists())
            self.assertTrue((out_dir / "saxpy_triton.py").exists())
            self.assertIn("unavailable for this frontend", (out_dir / "lambda.sexp").read_text())
        finally:
            shutil.rmtree(out_dir, ignore_errors=True)

    def test_cpp_compile_emits_artifacts(self):
        out_dir = pathlib.Path(tempfile.mkdtemp(prefix="loom_test_cpp_"))
        try:
            result = run_cmd(
                str(BIN),
                "compile",
                "examples/saxpy.cpp",
                "--input-kind",
                "cpp",
                "--entry",
                "saxpy",
                "--target",
                "triton",
                "--out",
                str(out_dir),
                "--emit",
                "all",
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertTrue((out_dir / "front_ir.json").exists())
            self.assertTrue((out_dir / "tensor_ir.json").exists())
            self.assertTrue((out_dir / "kernel_plan.json").exists())
            self.assertTrue((out_dir / "saxpy_triton.py").exists())
            self.assertIn("unavailable for this frontend", (out_dir / "lambda.sexp").read_text())
        finally:
            shutil.rmtree(out_dir, ignore_errors=True)

    def test_compile_accepts_backend_alias(self):
        out_dir = pathlib.Path(tempfile.mkdtemp(prefix="loom_test_backend_alias_"))
        try:
            result = run_cmd(
                str(BIN),
                "compile",
                "examples/saxpy.ml",
                "--entry",
                "saxpy",
                "--backend",
                "triton",
                "--out",
                str(out_dir),
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertTrue((out_dir / "saxpy_triton.py").exists())
            self.assertTrue((out_dir / "manifest.json").exists())
        finally:
            shutil.rmtree(out_dir, ignore_errors=True)

    def test_cuda_specific_optimization_rejected_for_triton(self):
        out_dir = pathlib.Path(tempfile.mkdtemp(prefix="loom_test_cuda_opt_"))
        try:
            result = run_cmd(
                str(BIN),
                "compile",
                "examples/saxpy.ml",
                "--entry",
                "saxpy",
                "--target",
                "triton",
                "--out",
                str(out_dir),
                "--emit",
                "all",
                "--enable-opt",
                "cuda-reduction-norm-plan",
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("CUDA-specific optimization flags", result.stderr)
        finally:
            shutil.rmtree(out_dir, ignore_errors=True)

    def test_target_specific_emit_rejected_for_wrong_backend(self):
        out_dir = pathlib.Path(tempfile.mkdtemp(prefix="loom_test_bad_emit_"))
        try:
            result = run_cmd(
                str(BIN),
                "compile",
                "examples/saxpy.ml",
                "--entry",
                "saxpy",
                "--target",
                "triton",
                "--out",
                str(out_dir / "triton"),
                "--emit",
                "cuda",
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("--emit cuda cannot be used with --target triton", result.stderr)

            result = run_cmd(
                str(BIN),
                "compile",
                "examples/saxpy.ml",
                "--entry",
                "saxpy",
                "--target",
                "cuda",
                "--out",
                str(out_dir / "cuda"),
                "--emit",
                "triton-plan",
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn(
                "--emit triton-plan cannot be used with --target cuda",
                result.stderr,
            )
        finally:
            shutil.rmtree(out_dir, ignore_errors=True)

    def test_cuda_target_rejects_triton_autotune_without_nvcc(self):
        out_dir = pathlib.Path(tempfile.mkdtemp(prefix="loom_test_cuda_autotune_"))
        try:
            result = run_cmd(
                str(BIN),
                "compile",
                "examples/saxpy.ml",
                "--entry",
                "saxpy",
                "--target",
                "cuda",
                "--out",
                str(out_dir),
                "--autotune",
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("does not support Triton autotuning flags", result.stderr)
        finally:
            shutil.rmtree(out_dir, ignore_errors=True)

    @unittest.skipIf(
        torch is None or not torch.cuda.is_available() or shutil.which("nvcc") is None,
        "CUDA compile target unavailable",
    )
    def test_compile_cuda_emits_artifacts_and_runtime(self):
        out_dir = pathlib.Path(tempfile.mkdtemp(prefix="loom_test_cuda_"))
        try:
            result = run_cmd(
                str(BIN),
                "compile",
                "examples/saxpy.ml",
                "--entry",
                "saxpy",
                "--target",
                "cuda",
                "--out",
                str(out_dir),
                "--emit",
                "all",
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            expected = {
                "front_ir.json",
                "lambda.sexp",
                "loom_lambda.json",
                "tensor_ir.json",
                "kernel_plan.json",
                "cuda_plan.json",
                "backend_analysis.json",
                "pipeline.json",
                "manifest.json",
                "report.md",
                "saxpy_cuda.cu",
                "saxpy_cuda.h",
                "libsaxpy.so",
            }
            self.assertTrue(expected.issubset({p.name for p in out_dir.iterdir()}))

            manifest = json.loads((out_dir / "manifest.json").read_text())
            self.assertEqual(manifest["target_backend"], "cuda")
            self.assertEqual(manifest["mode"], "compile")
            self.assertEqual(len(manifest["entries"]), 1)
            pipeline = json.loads((out_dir / "pipeline.json").read_text())
            self.assertIn("kernel_plan", pipeline)
            self.assertIn("cuda_plan", pipeline)
            entry = manifest["entries"][0]
            self.assertEqual(entry["entry_name"], "saxpy")
            self.assertEqual(entry["result_kind"], "tensor")

            lib = ctypes.CDLL(str(out_dir / "libsaxpy.so"))
            workspace_fn = getattr(lib, entry["workspace_symbol"])
            workspace_fn.argtypes = [ctypes.c_longlong]
            workspace_fn.restype = ctypes.c_size_t
            self.assertEqual(workspace_fn(4096), 0)

            export = getattr(lib, entry["symbol_name"])
            export.argtypes = [
                ctypes.c_float,
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_longlong,
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_size_t,
            ]
            export.restype = ctypes.c_int

            n = 4096
            a = 2.0
            x = torch.randn(n, device="cuda", dtype=torch.float32)
            y = torch.randn(n, device="cuda", dtype=torch.float32)
            out = torch.empty_like(x)
            status = export(
                ctypes.c_float(a),
                ctypes.c_void_p(x.data_ptr()),
                ctypes.c_void_p(y.data_ptr()),
                ctypes.c_longlong(n),
                ctypes.c_void_p(out.data_ptr()),
                None,
                ctypes.c_size_t(0),
            )
            self.assertEqual(status, 0)
            torch.testing.assert_close(out, a * x + y, rtol=1e-4, atol=1e-5)
        finally:
            shutil.rmtree(out_dir, ignore_errors=True)

    def test_negative_unknown_call(self):
        out_dir = pathlib.Path(tempfile.mkdtemp(prefix="loom_test_neg_"))
        try:
            result = run_cmd(
                str(BIN),
                "compile",
                "test/negative/unknown_call.ml",
                "--entry",
                "bad",
                "--target",
                "triton",
                "--out",
                str(out_dir),
                "--emit",
                "all",
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("unknown function call inside scalar lambda", result.stderr)
        finally:
            shutil.rmtree(out_dir, ignore_errors=True)

    def test_extended_ocaml_features_compile(self):
        out_dir = pathlib.Path(tempfile.mkdtemp(prefix="loom_test_extended_"))
        try:
            cases = [
                ("examples/saxpy_curried.ml", "saxpy_curried"),
                ("examples/relu_tupled.ml", "relu_tupled"),
                ("examples/dot_pipeline.ml", "dot_pipeline"),
            ]
            for source, entry in cases:
                result = run_cmd(
                    str(BIN),
                    "compile",
                    source,
                    "--entry",
                    entry,
                    "--target",
                    "triton",
                    "--out",
                    str(out_dir / entry),
                    "--emit",
                    "all",
                )
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertTrue((out_dir / entry / "front_ir.json").exists())
                self.assertTrue((out_dir / entry / "pipeline.json").exists())
        finally:
            shutil.rmtree(out_dir, ignore_errors=True)

    def test_compile_from_front_ir(self):
        out_dir = pathlib.Path(tempfile.mkdtemp(prefix="loom_test_front_ir_"))
        try:
            ocaml_out = out_dir / "ocaml"
            result = run_cmd(
                str(BIN),
                "compile",
                "examples/saxpy.ml",
                "--entry",
                "saxpy",
                "--target",
                "triton",
                "--out",
                str(ocaml_out),
                "--emit",
                "all",
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            front_ir = ocaml_out / "front_ir.json"
            front_out = out_dir / "front_ir"
            result = run_cmd(
                str(BIN),
                "compile",
                str(front_ir),
                "--input-kind",
                "front-ir",
                "--entry",
                "saxpy",
                "--target",
                "triton",
                "--out",
                str(front_out),
                "--emit",
                "all",
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(
                (ocaml_out / "tensor_ir.json").read_text().strip(),
                (front_out / "tensor_ir.json").read_text().strip(),
            )
            self.assertEqual(
                (ocaml_out / "kernel_plan.json").read_text().strip(),
                (front_out / "kernel_plan.json").read_text().strip(),
            )
            self.assertEqual(
                (ocaml_out / "triton_plan.json").read_text().strip(),
                (front_out / "triton_plan.json").read_text().strip(),
            )
        finally:
            shutil.rmtree(out_dir, ignore_errors=True)

    def test_compile_with_optimization_flags(self):
        out_dir = pathlib.Path(tempfile.mkdtemp(prefix="loom_test_opts_"))
        try:
            result = run_cmd(
                str(BIN),
                "compile",
                "examples/saxpy.ml",
                "--entry",
                "saxpy",
                "--target",
                "triton",
                "--out",
                str(out_dir),
                "--emit",
                "all",
                "--enable-opt",
                "scalar-const-fold",
                "--enable-opt",
                "normalized-dce",
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            manifest = json.loads((out_dir / "manifest.json").read_text())
            self.assertEqual(
                manifest["optimizations"]["enabled"],
                ["scalar-const-fold", "normalized-dce"],
            )
            report_text = (out_dir / "report.md").read_text()
            self.assertIn("enabled optimizations: scalar-const-fold, normalized-dce", report_text)
        finally:
            shutil.rmtree(out_dir, ignore_errors=True)

    def test_compile_with_autotune(self):
        out_dir = pathlib.Path(tempfile.mkdtemp(prefix="loom_test_autotune_"))
        try:
            result = run_cmd(
                str(BIN),
                "compile",
                "examples/saxpy.ml",
                "--entry",
                "saxpy",
                "--target",
                "triton",
                "--out",
                str(out_dir),
                "--emit",
                "all",
                "--autotune",
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            module_text = (out_dir / "saxpy_triton.py").read_text()
            self.assertIn("@triton.autotune", module_text)
            manifest_text = (out_dir / "manifest.json").read_text()
            self.assertIn("\"autotune\"", manifest_text)
        finally:
            shutil.rmtree(out_dir, ignore_errors=True)

    @unittest.skipIf(torch is None or not torch.cuda.is_available(), "CUDA runtime unavailable")
    def test_runtime_saxpy(self):
        out_dir = pathlib.Path(tempfile.mkdtemp(prefix="loom_test_runtime_"))
        try:
            result = run_cmd(
                str(BIN),
                "compile",
                "examples/saxpy.ml",
                "--entry",
                "saxpy",
                "--target",
                "triton",
                "--out",
                str(out_dir),
                "--emit",
                "all",
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            module_path = out_dir / "saxpy_triton.py"
            spec = importlib.util.spec_from_file_location("saxpy_triton", module_path)
            module = importlib.util.module_from_spec(spec)
            assert spec.loader is not None
            spec.loader.exec_module(module)
            n = 4096
            a = 2.5
            x = torch.randn(n, device="cuda", dtype=torch.float32)
            y = torch.randn(n, device="cuda", dtype=torch.float32)
            actual = module.saxpy(a, x, y)
            expected = a * x + y
            torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-5)
        finally:
            shutil.rmtree(out_dir, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
