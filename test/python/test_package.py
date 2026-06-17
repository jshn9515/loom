import ctypes
import json
import pathlib
import shutil
import subprocess
import tempfile
import unittest

try:
    import torch
except Exception:
    torch = None


ROOT = pathlib.Path(__file__).resolve().parents[2]
BIN = ROOT / "_build" / "default" / "src" / "loom_cli" / "main.exe"
FIXTURES = ROOT / "test" / "fixtures"


def run_cmd(*args):
    return subprocess.run(args, cwd=ROOT, capture_output=True, text=True)


def device_ptr(tensor):
    return ctypes.c_void_p(tensor.data_ptr())


class LoomPackageTests(unittest.TestCase):
    @unittest.skipIf(torch is None or not torch.cuda.is_available(), "CUDA runtime unavailable")
    def test_package_shared_library_runtime(self):
        out_dir = pathlib.Path(tempfile.mkdtemp(prefix="loom_test_package_shared_"))
        try:
            result = run_cmd(
                str(BIN),
                "package",
                "--project",
                str(FIXTURES / "package_project"),
                "--out",
                str(out_dir),
            )
            self.assertEqual(result.returncode, 0, result.stderr)

            artifact = out_dir / "libpackage_project.so"
            header = out_dir / "include" / "loom" / "package_project.h"
            source = out_dir / "src-gen" / "package_project.cu"
            manifest_path = out_dir / "manifest.json"
            report_path = out_dir / "report.md"
            self.assertTrue(artifact.exists())
            self.assertTrue(header.exists())
            self.assertTrue(source.exists())
            self.assertTrue(manifest_path.exists())
            self.assertTrue(report_path.exists())
            self.assertTrue((out_dir / "entries" / "loom_kernels_saxpy" / "tensor_ir.json").exists())
            self.assertTrue((out_dir / "entries" / "loom_kernels_dot" / "kernel_plan.json").exists())
            self.assertTrue((out_dir / "entries" / "loom_kernels_dot" / "cuda_plan.json").exists())
            self.assertTrue((out_dir / "entries" / "loom_kernels_dot" / "backend_analysis.json").exists())

            manifest = json.loads(manifest_path.read_text())
            self.assertEqual(manifest["target_backend"], "cuda")
            self.assertEqual(manifest["artifact_kind"], "shared")
            self.assertEqual(
                sorted(entry["entry_name"] for entry in manifest["entries"]),
                ["dot", "relu", "saxpy"],
            )

            lib = ctypes.CDLL(str(artifact))

            saxpy_ws = lib.loom_kernels_saxpy_workspace_size
            saxpy_ws.argtypes = [ctypes.c_longlong]
            saxpy_ws.restype = ctypes.c_size_t
            self.assertEqual(saxpy_ws(4096), 0)

            saxpy = lib.loom_kernels_saxpy
            saxpy.argtypes = [
                ctypes.c_float,
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_longlong,
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_size_t,
            ]
            saxpy.restype = ctypes.c_int

            n = 4096
            a = 2.5
            x = torch.randn(n, device="cuda", dtype=torch.float32)
            y = torch.randn(n, device="cuda", dtype=torch.float32)
            out = torch.empty_like(x)
            status = saxpy(
                ctypes.c_float(a),
                device_ptr(x),
                device_ptr(y),
                ctypes.c_longlong(n),
                device_ptr(out),
                None,
                ctypes.c_size_t(0),
            )
            self.assertEqual(status, 0)
            torch.testing.assert_close(out, a * x + y, rtol=1e-4, atol=1e-5)

            dot_ws = lib.loom_kernels_dot_workspace_size
            dot_ws.argtypes = [ctypes.c_longlong]
            dot_ws.restype = ctypes.c_size_t
            workspace_size = dot_ws(n)
            self.assertGreater(workspace_size, 0)

            dot = lib.loom_kernels_dot
            dot.argtypes = [
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_longlong,
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_size_t,
            ]
            dot.restype = ctypes.c_int

            scalar_out = torch.empty(1, device="cuda", dtype=torch.float32)
            workspace = torch.empty(workspace_size, device="cuda", dtype=torch.uint8)
            status = dot(
                device_ptr(x),
                device_ptr(y),
                ctypes.c_longlong(n),
                device_ptr(scalar_out),
                device_ptr(workspace),
                ctypes.c_size_t(workspace_size),
            )
            self.assertEqual(status, 0)
            expected = torch.sum(x * y)
            torch.testing.assert_close(scalar_out[0], expected, rtol=1e-4, atol=1e-5)
        finally:
            shutil.rmtree(out_dir, ignore_errors=True)

    def test_package_static_library_with_filters(self):
        out_dir = pathlib.Path(tempfile.mkdtemp(prefix="loom_test_package_static_"))
        try:
            result = run_cmd(
                str(BIN),
                "package",
                "--project",
                str(FIXTURES / "package_project"),
                "--out",
                str(out_dir),
                "--kind",
                "static",
                "--module",
                "kernels",
                "--entry",
                "relu",
                "--enable-opt",
                "elementwise-plan-specialize",
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            artifact = out_dir / "libpackage_project.a"
            manifest = json.loads((out_dir / "manifest.json").read_text())
            self.assertTrue(artifact.exists())
            self.assertEqual(manifest["target_backend"], "cuda")
            self.assertEqual(manifest["artifact_kind"], "static")
            self.assertEqual(manifest["optimizations"]["enabled"], ["elementwise-plan-specialize"])
            self.assertEqual(len(manifest["entries"]), 1)
            self.assertEqual(manifest["entries"][0]["entry_name"], "relu")
            self.assertEqual(manifest["entries"][0]["symbol_name"], "loom_kernels_relu")
        finally:
            shutil.rmtree(out_dir, ignore_errors=True)

    def test_python_package_static_library_with_filters(self):
        out_dir = pathlib.Path(tempfile.mkdtemp(prefix="loom_test_python_package_static_"))
        try:
            result = run_cmd(
                str(BIN),
                "package",
                "--project",
                str(FIXTURES / "python_package_project"),
                "--input-kind",
                "python",
                "--out",
                str(out_dir),
                "--kind",
                "static",
                "--module",
                "kernels",
                "--entry",
                "relu",
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            artifact = out_dir / "libpython_package_project.a"
            manifest = json.loads((out_dir / "manifest.json").read_text())
            self.assertTrue(artifact.exists())
            self.assertEqual(manifest["target_backend"], "cuda")
            self.assertEqual(manifest["artifact_kind"], "static")
            self.assertEqual(len(manifest["entries"]), 1)
            self.assertEqual(manifest["entries"][0]["entry_name"], "relu")
            self.assertEqual(manifest["entries"][0]["symbol_name"], "loom_kernels_relu")
            self.assertTrue((out_dir / "entries" / "loom_kernels_relu" / "tensor_ir.json").exists())
        finally:
            shutil.rmtree(out_dir, ignore_errors=True)

    def test_cpp_package_static_library_with_filters(self):
        out_dir = pathlib.Path(tempfile.mkdtemp(prefix="loom_test_cpp_package_static_"))
        try:
            result = run_cmd(
                str(BIN),
                "package",
                "--project",
                str(FIXTURES / "cpp_package_project"),
                "--input-kind",
                "cpp",
                "--out",
                str(out_dir),
                "--kind",
                "static",
                "--module",
                "kernels",
                "--entry",
                "relu",
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            artifact = out_dir / "libcpp_package_project.a"
            manifest = json.loads((out_dir / "manifest.json").read_text())
            self.assertTrue(artifact.exists())
            self.assertEqual(manifest["target_backend"], "cuda")
            self.assertEqual(manifest["artifact_kind"], "static")
            self.assertEqual(len(manifest["entries"]), 1)
            self.assertEqual(manifest["entries"][0]["entry_name"], "relu")
            self.assertEqual(manifest["entries"][0]["symbol_name"], "loom_kernels_relu")
            self.assertTrue((out_dir / "entries" / "loom_kernels_relu" / "tensor_ir.json").exists())
        finally:
            shutil.rmtree(out_dir, ignore_errors=True)

    def test_package_auto_discovers_cpp_project(self):
        out_dir = pathlib.Path(tempfile.mkdtemp(prefix="loom_test_cpp_package_auto_"))
        try:
            result = run_cmd(
                str(BIN),
                "package",
                "--project",
                str(FIXTURES / "cpp_package_project"),
                "--input-kind",
                "auto",
                "--out",
                str(out_dir),
                "--kind",
                "static",
                "--module",
                "kernels",
                "--entry",
                "relu",
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            manifest = json.loads((out_dir / "manifest.json").read_text())
            self.assertEqual(len(manifest["entries"]), 1)
            self.assertEqual(manifest["entries"][0]["entry_name"], "relu")
        finally:
            shutil.rmtree(out_dir, ignore_errors=True)

    def test_package_rejects_empty_project(self):
        out_dir = pathlib.Path(tempfile.mkdtemp(prefix="loom_test_package_empty_"))
        try:
            result = run_cmd(
                str(BIN),
                "package",
                "--project",
                str(FIXTURES / "empty_project"),
                "--out",
                str(out_dir),
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("[@loom.entry]", result.stderr)
        finally:
            shutil.rmtree(out_dir, ignore_errors=True)

    def test_package_rejects_unsupported_entry(self):
        out_dir = pathlib.Path(tempfile.mkdtemp(prefix="loom_test_package_unsupported_"))
        try:
            result = run_cmd(
                str(BIN),
                "package",
                "--project",
                str(FIXTURES / "unsupported_project"),
                "--out",
                str(out_dir),
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("unknown function call inside scalar lambda", result.stderr)
        finally:
            shutil.rmtree(out_dir, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
