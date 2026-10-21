.PHONY: install run test

install:
	pip install -e ".[dev]"

# Download the UCI extract (checked against its SHA-256), train, evaluate: results/metrics.json
# and results/RESULTS.md. About 3 minutes on one core.
run:
	readmit run

test:
	ruff check src tests
	pytest -q
