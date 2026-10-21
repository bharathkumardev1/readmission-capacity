.PHONY: install run export loadtest lock image test

install:
	pip install -e ".[dev]"

# Download the UCI extract (checked against its SHA-256), train, evaluate: results/metrics.json
# and results/RESULTS.md. About 3 minutes on one core.
run:
	readmit run

# Trees, calibration, capacity curves and the call list for the site's demo.
export:
	readmit export --out build/web

# Service on one CPU, load generator on the other: results/load.json.
loadtest:
	bash scripts/loadtest.sh

lock:
	uv pip compile pyproject.toml --extra service -c constraints.txt -o requirements.lock \
	  --python-version 3.12 --python-platform x86_64-manylinux_2_28 --generate-hashes \
	  --custom-compile-command "make lock"

image:
	docker build -t readmit .

# Lint and pytest, then the browser scorer against a fresh export (needs `make run` first).
test:
	ruff check src tests scripts
	pytest -q
	readmit export --out web/test/data
	cd web && npm test
