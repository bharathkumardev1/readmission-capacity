#!/usr/bin/env bash
# Load test on a 2-CPU machine: the service pinned to CPU 0, the load generator to CPU 1, so
# the two don't compete. Needs `make run` first for the model in artifacts/.
set -euo pipefail

export READMIT_MODEL_DIR=${READMIT_MODEL_DIR:-artifacts} OMP_NUM_THREADS=1

taskset -c 0 uvicorn readmit.service:app --host 127.0.0.1 --port 8000 --workers 1 --log-level warning &
server=$!
trap 'kill $server' EXIT
until curl -sf http://127.0.0.1:8000/healthz >/dev/null; do sleep 1; done

taskset -c 1 readmit loadtest --batches 1,100 --levels 1,2,4,8 --seconds 30 \
  --server-note "1 uvicorn worker pinned to CPU 0, load generator on CPU 1"
