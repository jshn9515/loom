OCAML_BIN := _build/default/src/loom_cli/main.exe
KERNEL ?= saxpy
ARGS ?=

.PHONY: ocaml-build python-env test bench experiments experiments-kernel bench-image bench-container dgx-sync-bench clean

ocaml-build:
	dune build $(OCAML_BIN)

python-env:
	uv sync

test: ocaml-build
	uv run python -m unittest discover -s test/python -p 'test_*.py'

bench: ocaml-build
	uv run python benchmarks/run.py

experiments: ocaml-build
	uv run python experiments/source/harness/run.py --all

experiments-kernel: ocaml-build
	uv run python experiments/source/harness/run.py --kernel $(KERNEL)

bench-image:
	scripts/bench/build-image.sh

bench-container:
	scripts/bench/run-container.sh -- $(ARGS)

dgx-sync-bench:
	scripts/bench/tea-dev-benchmark.sh

clean:
	dune clean
	rm -rf build
	rm -rf .venv
	rm -rf experiments/_work
