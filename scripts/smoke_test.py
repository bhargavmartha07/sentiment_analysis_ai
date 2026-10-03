"""End-to-end smoke test against a running stack.

    docker compose exec api python scripts/smoke_test.py

Exercises the synchronous path, the asynchronous queue path, result retrieval
and error handling, printing a readable report. Exits non-zero on the first
failure, which makes it usable as a post-deploy gate.
"""

from __future__ import annotations

import json
import os
import sys
import time
import uuid

import requests

BASE_URL = os.getenv("BASE_URL", "http://localhost:8000").rstrip("/")
TIMEOUT = float(os.getenv("SMOKE_TIMEOUT", "30"))
POLL_INTERVAL = 0.5

POSITIVE_TEXT = "I love this product! It arrived early and works perfectly."
NEGATIVE_TEXT = "This movie was terrible. A complete waste of time and money."

_failures: list[str] = []


def check(label: str, condition: bool, detail: str = "") -> None:
    marker = "\033[32mPASS\033[0m" if condition else "\033[31mFAIL\033[0m"
    print(f"[{marker}] {label}{f' - {detail}' if detail else ''}")
    if not condition:
        _failures.append(label)


def poll_for_result(session: requests.Session, job_id: str, timeout: float = 45.0) -> dict | None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        response = session.get(f"{BASE_URL}/api/sentiment/results/{job_id}", timeout=TIMEOUT)
        if response.status_code == 200:
            return response.json()
        time.sleep(POLL_INTERVAL)
    return None


def main() -> int:
    session = requests.Session()
    print(f"Smoke testing {BASE_URL}\n" + "-" * 62)

    health = session.get(f"{BASE_URL}/health", timeout=TIMEOUT)
    check("GET /health returns 200", health.status_code == 200, f"status={health.status_code}")
    if health.ok:
        print("        " + json.dumps(health.json()))

    ready = session.get(f"{BASE_URL}/health/ready", timeout=TIMEOUT)
    check("GET /health/ready returns 200", ready.status_code == 200, f"status={ready.status_code}")
    if ready.ok:
        for name, status in (ready.json().get("dependencies") or {}).items():
            print(f"        dependency {name}: {status['status']}")

    started = time.perf_counter()
    response = session.post(f"{BASE_URL}/api/sentiment/sync", json={"text": POSITIVE_TEXT}, timeout=TIMEOUT)
    elapsed_ms = (time.perf_counter() - started) * 1000
    check("POST /sync returns 200", response.status_code == 200, f"{elapsed_ms:.1f} ms")
    if response.ok:
        body = response.json()
        print("        " + json.dumps(body))
        check("  positive text labelled positive", body["sentiment"] == "positive", body["sentiment"])
        check("  score within [0, 1]", 0.0 <= body["score"] <= 1.0, str(body["score"]))

    response = session.post(f"{BASE_URL}/api/sentiment/sync", json={"text": NEGATIVE_TEXT}, timeout=TIMEOUT)
    if response.ok:
        body = response.json()
        print("        " + json.dumps(body))
        check("  negative text labelled negative", body["sentiment"] == "negative", body["sentiment"])

    response = session.post(f"{BASE_URL}/api/sentiment/sync", json={"text": "   "}, timeout=TIMEOUT)
    check("POST /sync with blank text returns 400", response.status_code == 400, f"status={response.status_code}")
    if response.content:
        print("        " + json.dumps(response.json()))

    response = session.post(f"{BASE_URL}/api/sentiment/async", json={"text": POSITIVE_TEXT}, timeout=TIMEOUT)
    check("POST /async returns 202", response.status_code == 202, f"status={response.status_code}")
    job_id = response.json().get("job_id") if response.ok else None
    if job_id:
        print("        " + json.dumps(response.json()))
        result = poll_for_result(session, job_id)
        check("GET /results/{job_id} eventually returns 200", result is not None, f"job_id={job_id}")
        if result:
            print("        " + json.dumps(result))
            check("  stored text matches the input", result["text"] == POSITIVE_TEXT)
            check("  timestamp is UTC", str(result["timestamp"]).endswith("Z") or "+00:00" in str(result["timestamp"]))

    response = session.get(f"{BASE_URL}/api/sentiment/results/{uuid.uuid4()}", timeout=TIMEOUT)
    check("GET /results with unknown id returns 404", response.status_code == 404, f"status={response.status_code}")
    if response.content:
        print("        " + json.dumps(response.json()))

    response = session.get(f"{BASE_URL}/openapi.json", timeout=TIMEOUT)
    check("GET /openapi.json returns a schema", response.ok and bool(response.json().get("paths")))

    print("-" * 62)
    if _failures:
        print(f"\033[31m{len(_failures)} check(s) failed:\033[0m " + ", ".join(_failures))
        return 1
    print("\033[32mAll smoke checks passed.\033[0m")
    return 0


if __name__ == "__main__":
    sys.exit(main())
