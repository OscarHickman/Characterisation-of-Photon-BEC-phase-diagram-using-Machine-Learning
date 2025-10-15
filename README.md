# Characterisation-of-Photon-BEC-phase-diagram-using-Machine-Learning

Repository containing analysis and machine learning experiments from a Bachelor's project.

Contents
- `src/photon_bec/` — package modules
- `requirements.txt` — Python dependencies
- `pyproject.toml` — formatting and project metadata


Quickstart

1. Create a virtual environment and activate it (recommended):

```bash
python -m venv .venv
source .venv/bin/activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. (Optional) Format code with Black and isort (if you have pre-commit installed):

```bash
pre-commit run --all-files
```

Usage
-----

The package exposes `photon_bec.binary` and `photon_bec.length`.

Run the examples from the repository root:

```bash
# Add src to PYTHONPATH and run the example modules
python -c "import sys; sys.path.insert(0, 'src'); import examples.run_binary_example"
python -c "import sys; sys.path.insert(0, 'src'); import examples.run_length_example"

# or using -m (requires package discovery via src in sys.path)
python -m examples.run_binary_example
python -m examples.run_length_example
```

To build/train models you will need TensorFlow and SciPy for interpolation.
