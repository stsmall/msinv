.PHONY: test figures build clean install dev-install

test:
	pytest tests/hull/

figures:
	python examples/make_figures.py

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
