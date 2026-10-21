"""Closed-loop load test against a running scoring service.

Each of C concurrent clients posts a batch of test encounters, waits for the answer and
posts the next, for a fixed time. Per level it reports requests and encounters per second,
latency percentiles of successful requests, the error rate and the server's resident memory
from its own ``/metrics``.

Cost per million encounters prices one vCPU at half the on-demand rate of an AWS c7i.large
(2 vCPUs, $0.08925 an hour in us-east-1 on 2026-10-21, from instances.vantage.sh and
calculator.holori.com). It assumes the throughput measured here carries over to that vCPU,
which I haven't measured.
"""

from __future__ import annotations

import asyncio
import json
import time

import numpy as np

USD_PER_VCPU_HOUR = 0.08925 / 2


async def _client(client, url: str, batches: list[list[dict]], start: int, stride: int, deadline: float,
                  out: list) -> None:
    i = start
    while time.perf_counter() < deadline:
        batch = batches[i % len(batches)]
        i += stride
        t0 = time.perf_counter()
        try:
            status = (await client.post(url, json={"encounters": batch})).status_code
        except Exception:
            status = -1
        out.append((time.perf_counter() - t0, status, len(batch)))


async def _memory_mb(client, base: str) -> float | None:
    try:
        text = (await client.get(f"{base}/metrics")).text
    except Exception:
        return None
    for line in text.splitlines():
        if line.startswith("process_resident_memory_bytes"):
            return round(float(line.split()[-1]) / 2**20, 1)
    return None


async def run_level(base: str, batches: list[list[dict]], concurrency: int, seconds: float) -> dict:
    import httpx

    url = f"{base}/score"
    limits = httpx.Limits(max_connections=concurrency, max_keepalive_connections=concurrency)
    async with httpx.AsyncClient(timeout=30.0, limits=limits) as client:
        for b in batches[:5]:
            await client.post(url, json={"encounters": b})
        out: list[tuple[float, int, int]] = []
        t0 = time.perf_counter()
        await asyncio.gather(*(_client(client, url, batches, c, concurrency, t0 + seconds, out)
                               for c in range(concurrency)))
        wall = time.perf_counter() - t0
        mem = await _memory_mb(client, base)
    ok = [(lat, n) for lat, st, n in out if st == 200]
    lat = np.array([x for x, _ in ok]) * 1000
    eps = sum(n for _, n in ok) / wall
    errors = sum(1 for _, st, _ in out if st != 200)
    return {
        "concurrency": concurrency, "batch": len(batches[0]), "seconds": round(wall, 2),
        "requests": len(out), "errors": errors, "error_rate": round(errors / max(len(out), 1), 4),
        "requests_per_s": round(len(ok) / wall, 1), "encounters_per_s": round(eps, 1),
        "p50_ms": round(float(np.percentile(lat, 50)), 2) if len(lat) else None,
        "p95_ms": round(float(np.percentile(lat, 95)), 2) if len(lat) else None,
        "p99_ms": round(float(np.percentile(lat, 99)), 2) if len(lat) else None,
        "server_rss_mb": mem,
        "usd_per_million_encounters": round(USD_PER_VCPU_HOUR / (eps * 3600) * 1e6, 4) if eps else None,
    }


def run(base: str, rows: list[dict], batch: int, levels=(1, 2, 4, 8), seconds: float = 30.0) -> list[dict]:
    batches = [rows[i : i + batch] for i in range(0, len(rows) - batch + 1, batch)]
    out = []
    for c in levels:
        row = asyncio.run(run_level(base, batches, c, seconds))
        print(json.dumps(row), flush=True)
        out.append(row)
    return out
