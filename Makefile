.PHONY: test test-hull test-smc bakeoff build clean install dev-install

test:
	pytest tests/

test-hull:
	pytest tests/hull/

test-smc:
	pytest tests/test_*.py

bakeoff:
	python examples/bakeoff.py

build:
	python -m build

install:
	pip install .

dev-install:
	pip install -e ".[test,plots]"

clean:
	rm -rf build/ dist/ *.egg-info/ msinv/__pycache__ tests/__pycache__
	find . -name "__pycache__" -type d -exec rm -rf {} +
	find . -name "*.pyc" -delete
