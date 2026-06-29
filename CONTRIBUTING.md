# Contributing to pychartai

Thank you for considering a contribution! This document explains how to get the project running locally, the conventions we follow, and how to submit a pull request.

---

## Table of Contents

1. [Development setup](#development-setup)
2. [Running the test suite](#running-the-test-suite)
3. [Code style](#code-style)
4. [Submitting changes](#submitting-changes)
5. [Reporting bugs](#reporting-bugs)
6. [Feature requests](#feature-requests)

---

## Development setup

```bash
# 1. Fork and clone
git clone https://github.com/cemakpolat/pychartai.git
cd pychartai

# 2. Create a virtual environment
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 3. Install in editable mode with dev extras
pip install -e ".[dev,api,viz-plotly]"

# 4. Copy the env template and fill in any keys you need
cp .env.example .env
```

For local LLM testing (no API key needed):

```bash
# Install Ollama — https://ollama.com
ollama pull llama3.2
ollama serve
```

---

## Running the test suite

```bash
# All tests (no LLM required)
pytest

# With coverage report
pytest --cov=src --cov-report=term-missing

# Docker sandbox tests (requires Docker Desktop running)
pytest tests/test_docker_sandbox.py -v
```

### Cloud connector integration tests

These run the S3/Azure/GCS connectors against local emulators (MinIO, Azurite,
fake-gcs-server). They are skipped unless explicitly enabled:

```bash
# Start the emulator stack
docker compose -f docker-compose.cloud-test.yml up -d

# Run the integration tests
PYCHARTAI_CLOUD_IT=1 pytest tests/integration/test_cloud_integration.py -v

# Tear down
docker compose -f docker-compose.cloud-test.yml down -v
```

CI runs these automatically via `.github/workflows/cloud-integration.yml`.

All 313+ tests must pass before a PR can be merged.  
No test should require a live LLM call — use mocks/stubs.

---

## Code style

- **Formatting**: [Black](https://black.readthedocs.io/) — `black src/ tests/`
- **Import ordering**: [isort](https://pycli.readthedocs.io/isort/) — `isort src/ tests/`
- **Type hints**: all public functions and class methods must have complete type annotations
- **Docstrings**: NumPy style for all public symbols
- **No bare `except:`** — always catch a specific exception type
- **No `print()` in library code** — use `logging.getLogger('pychartai')`

---

## Submitting changes

1. **Create a branch** from `main`:
   ```bash
   git checkout -b feat/my-feature
   ```
2. **Write tests** covering the new behaviour.  PRs without tests will not be merged.
3. **Update `CHANGELOG.md`** — add an entry under `[Unreleased]`.
4. **Open a pull request** against `main` with a clear description of _what_ and _why_.

PRs that change the public API require a corresponding docs update in `docs/`.

---

## Reporting bugs

Open a [GitHub Issue](https://github.com/cemakpolat/pychartai/issues) and include:

- pychartai version (`python -c "import pychartai; print(pychartai.__version__)"`)
- Python version and OS
- Minimal reproducible example
- Full traceback

---

## Feature requests

Open a GitHub Issue with the label **`enhancement`**.  
Please describe the use-case, not just the proposed solution.

---

## Security vulnerabilities

See [SECURITY.md](SECURITY.md) for responsible disclosure instructions.  
**Do not open a public GitHub Issue for security vulnerabilities.**
