venv=.venv
PYTHON=${venv}/bin/python
PIP=${venv}/bin/pip
ACTIVATE=. ${venv}/bin/activate

.PHONY: all setup install precommit examples test clean

all: setup

setup: ${venv}/bin/activate
	${PIP} install --upgrade pip
	${PIP} install -r requirements.txt
	# ensure pre-commit hooks are available
	${PIP} install pre-commit ruff || true
	${PYTHON} -m pre_commit install || true

${venv}/bin/activate:
	python -m venv ${venv}
	@echo "created venv at ${venv}"

precommit:
	${PYTHON} -m pre_commit run --all-files

examples:
	${PYTHON} examples/run_binary_example.py
	${PYTHON} examples/run_length_example.py
	${PYTHON} examples/train_binary_with_tf.py

test:
	# run tests inside venv with src on PYTHONPATH
	PYTHONPATH=src ${PYTHON} -m pytest -q

clean:
	rm -rf ${venv}