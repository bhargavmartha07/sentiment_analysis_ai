"""Latency and throughput benchmark for the sentiment service.

    docker compose exec api python scripts/benchmark.py

Reports p50/p90/p99 latency and throughput for the synchronous endpoint, the
batched endpoint and the full asynchronous round trip (submit -> poll -> result).
Results depend heavily on the host; run it on the target hardware before quoting
numbers.
"""

from __future__ import annotations

import argparse
import os
import statistics
import time
import uuid
from concurrent.futures import ThreadPoolExecutor

import requests

POSITIVE = "I absolutely love this product. Fantastic quality and it arrived a day early."
NEGATIVE = "This was a complete waste of money. Terrible quality and support was useless."


def percentile(values: list[float], fraction: float) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    index = min(int(round(fraction * (len(ordered) - 1))), len(ordered) - 1)
    return ordered[index]


def summarise(name: str, latencies_ms: list[float], errors: int = 0) -> None:
    if not latencies_ms:
        print(f"{name:<28} no successful samples")
        return
    mean = statistics.fmean(latencies_ms)
    print(
        f"{name:<28} n={len(latencies_ms):<5} "
        f"p50={percentile(latencies_ms, 0.50):7.2f} ms  "
        f"p90={percentile(latencies_ms, 0.90):7.2f} ms  "
        f"p99={percentile(latencies_ms, 0.99):7.2f} ms  "
        f"mean={mean:7.2f} ms  "
        f"max={max(latencies_ms):7.2f} ms"
        + (f"  errors={errors}" if errors else "")
    )


def timed_request(session: requests.Session, url: str, payload: dict) -> tuple[float, int]:
    started = time.perf_counter()
    response = session.post(url, json=payload, timeout=60)
    return (time.perf_counter() - started) * 1000, response.status_code


def bench_sync(base_url: str, requests_count: int, concurrency: int) -> None:
    session = requests.Session()
    url = f"{base_url}/api/sentiment/sync"
    payloads = [{"text": POSITIVE if index % 2 == 0 else NEGATIVE} for index in range(requests_count)]

    def run(payload: dict) -> tuple[float, int]:
        return timed_request(session, url, payload)

    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        results = list(pool.map(run, payloads))

    latencies = [value for value, status in results if status == 200]
    summarise(f"POST /sync (c={concurrency})", latencies, errors=len(results) - len(latencies))


def bench_batch(base_url: str, batch_size: int, batches: int) -> None:
    session = requests.Session()
    url = f"{base_url}/api/sentiment/batch"
    texts = [POSITIVE if index % 2 == 0 else NEGATIVE for index in range(batch_size)]

    started = time.perf_counter()
    statuses = [
        session.post(url, json={"texts": texts}, timeout=120).status_code for _ in range(batches)
    ]
    elapsed = time.perf_counter() - started
    processed = batch_size * batches
    print(
        f"{'POST /batch (size=' + str(batch_size) + ')':<28} "
        f"{processed / elapsed:8.1f} texts/s  (statuses ok: {statuses.count(200)}/{batches})"
    )


def bench_async(base_url: str, job_count: int, concurrency: int) -> None:
    session = requests.Session()

    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        submissions = list(
            pool.map(
                lambda _: session.post(f"{base_url}/api/sentiment/async", json={"text": POSITIVE}, timeout=60).json(),
                range(job_count),
            )
        )
    job_ids = [item["job_id"] for item in submissions if "job_id" in item]

    def await_result(job_id: str) -> tuple[float, bool]:
        began = time.perf_counter()
        while time.perf_counter() - began < 120:
            response = session.get(f"{base_url}/api/sentiment/results/{job_id}", timeout=30)
            if response.status_code == 200:
                return (time.perf_counter() - began) * 1000, True
            time.sleep(0.2)
        return (time.perf_counter() - began) * 1000, False

    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        outcomes = list(pool.map(await_result, job_ids))

    total_elapsed = time.perf_counter() - started
    completed = [value for value, ok in outcomes if ok]
    summarise("async submit -> result", completed, errors=len(outcomes) - len(completed))
    print(
        f"{'async end-to-end throughput':<28} {len(completed) / total_elapsed:8.1f} jobs/s "
        f"over {len(completed)} jobs"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default=os.getenv("BASE_URL", "http://localhost:8000"))
    parser.add_argument("--requests", type=int, default=200)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--batches", type=int, default=10)
    parser.add_argument("--async-jobs", type=int, default=25)
    parser.add_argument("--skip-async", action="store_true")
    args = parser.parse_args()

    base_url = args.base_url.rstrip("/")
    print(f"Benchmarking {base_url}")
    print("=" * 100)
    bench_sync(base_url, args.requests, 1)
    bench_sync(base_url, args.requests, args.concurrency)
    bench_batch(base_url, args.batch_size, args.batches)
    if not args.skip_async:
        bench_async(base_url, args.async_jobs, args.concurrency)
    print("=" * 100)
    print(f"run id {uuid.uuid4().hex[:8]} - numbers are indicative, not a published benchmark")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
