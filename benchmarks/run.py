import pathlib
import shutil
import subprocess
import tempfile
import time


ROOT = pathlib.Path(__file__).resolve().parents[1]
BIN = ROOT / "_build" / "default" / "src" / "loom_cli" / "main.exe"
EXAMPLES = ["saxpy", "relu", "l2_norm_sq", "dot"]


def compile_example(name: str, out_dir: pathlib.Path) -> None:
    source = ROOT / "examples" / f"{name}.ml"
    result = subprocess.run(
        [
            str(BIN),
            "compile",
            str(source),
            "--entry",
            name,
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
        raise SystemExit(result.stderr)


def main() -> None:
    work = pathlib.Path(tempfile.mkdtemp(prefix="loom_bench_"))
    try:
        rows = []
        for name in EXAMPLES:
            out_dir = work / name
            t0 = time.perf_counter()
            compile_example(name, out_dir)
            dt = time.perf_counter() - t0
            rows.append((name, dt, out_dir / f"{name}_triton.py"))

        print("| entry | compile_seconds | module |")
        print("| --- | ---: | --- |")
        for name, seconds, module_path in rows:
            print(f"| {name} | {seconds:.4f} | {module_path} |")
        print()
        print("Runtime benchmarking is intentionally skipped when no explicit CUDA harness is configured.")
    finally:
        shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    main()

