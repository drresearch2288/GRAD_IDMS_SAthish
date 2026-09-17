.PHONY: setup test lint train preprocess

PYTHON ?= .venv/bin/python
PIP ?= .venv/bin/pip

setup:
	python3.12 -m venv .venv
	$(PIP) install -U pip
	$(PIP) install -r requirements.txt

test:
	$(PYTHON) -m pytest tests

lint:
	$(PYTHON) -m ruff check src scripts tests

train:
	$(PYTHON) scripts/train.py --tiny

preprocess:
	$(PYTHON) scripts/preprocess.py
