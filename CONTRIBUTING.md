# Contributing to msinv

Contributions welcome. Please:

1. **Open an issue first** for any non-trivial change, so we can discuss the design.
2. **Fork and submit a PR** with:
   - A clear description of the change
   - New or updated tests if adding functionality
   - Documentation updates if changing the API

## Development setup

```bash
git clone https://github.com/YOUR_USERNAME/msinv.git
cd msinv
pip install -e ".[test]"
make test  # should pass 46/46
```

## Code style

- Follow PEP 8 for Python
- Use numpy-style docstrings
- Keep functions focused; prefer composition over inheritance
- Numpy arrays for numerical work, native Python for tree structures

## Adding a new feature

1. Write a test first in `tests/`
2. Implement in `msinv/simulator.py`
3. Update docstring and `docs/api.md`
4. Add an example if applicable in `examples/`
5. Run `make test` to verify 46/46 still pass

## Reporting bugs

Include:
- Python version
- msinv version
- Minimal reproducer
- Expected vs actual behavior

## Priorities for contributions

- More application examples (Drosophila inversions, plant inversions)
- Full C extension (current one has demography bugs)
- ABC framework wrapper for inference
- Integration with stdpopsim
- Documentation improvements

## License

By contributing, you agree that your contributions will be licensed under the MIT license (see [LICENSE](LICENSE)).
