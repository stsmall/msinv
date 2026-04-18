.PHONY: venv dev test figures clean

RUST_TARGET := $(shell rustc -vV | grep ^host | cut -d' ' -f2)

venv: .venv/.ok

.venv/.ok:
	uv venv .venv --python 3.12 --allow-existing
	uv pip install maturin pip numpy tskit 'msprime>=1.2' pytest pytest-timeout matplotlib
	@touch $@

dev: venv
	.venv/bin/maturin develop --release --target $(RUST_TARGET)

test: dev
	.venv/bin/python -m pytest tests/hull/ --ignore=tests/hull/test_stdpopsim_validation.py

figures: dev
	.venv/bin/python examples/make_figures.py

clean:
	rm -rf .venv/ build/ dist/ *.egg-info/
	find . -name "__pycache__" -type d -exec rm -rf {} +
	find . -name "*.pyc" -delete
	find . -name "*.so" -not -path "./rust/*" -delete
