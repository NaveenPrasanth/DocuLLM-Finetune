.PHONY: install install-dev lint test data train evaluate pipeline clean

install:
	pip install -r requirements.txt
	pip install -e .

install-dev:
	pip install -r requirements-dev.txt
	pip install -e .

lint:
	ruff check src/ tests/ scripts/
	ruff format --check src/ tests/ scripts/

format:
	ruff check --fix src/ tests/ scripts/
	ruff format src/ tests/ scripts/

typecheck:
	mypy src/

test:
	pytest tests/ -v --tb=short

test-cov:
	pytest tests/ -v --cov=src --cov-report=term-missing

data:
	python scripts/download_data.py
	python scripts/prepare_dataset.py

synthetic:
	python scripts/generate_synthetic.py

prompts:
	python scripts/run_prompt_experiments.py

train:
	python scripts/train.py

evaluate:
	python scripts/evaluate.py

pipeline:
	python scripts/run_pipeline.py

clean:
	rm -rf __pycache__ .pytest_cache .mypy_cache .ruff_cache
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
