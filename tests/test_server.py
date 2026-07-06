"""
tests/test_server.py — Integration tests for the EdgeAccord FastAPI server.

Runs without a live Redis/Celery broker by using the in-process thread pool.
Uses FastAPI's built-in TestClient (synchronous) + httpx for async WebSocket.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient


# ── import application ────────────────────────────────────────────────────────
@pytest.fixture(scope="module")
def client():
    from host.server import app
    with TestClient(app) as c:
        yield c


# ── /health ───────────────────────────────────────────────────────────────────

def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


# ── POST /api/v1/build ────────────────────────────────────────────────────────

def test_post_build_returns_job_id(client):
    r = client.post("/api/v1/build", json={})
    assert r.status_code == 202
    data = r.json()
    assert "job_id" in data
    assert data["status"] == "queued"


def test_post_build_with_config_override(client):
    r = client.post("/api/v1/build", json={"config_override": {"dry_run": True}})
    assert r.status_code == 202
    assert "job_id" in r.json()


# ── POST /api/v1/benchmark ────────────────────────────────────────────────────

def test_post_benchmark_returns_job_id(client):
    r = client.post("/api/v1/benchmark", json={})
    assert r.status_code == 202
    data = r.json()
    assert "job_id" in data
    assert data["status"] == "queued"


# ── GET /api/v1/jobs/{job_id} ─────────────────────────────────────────────────

def test_get_job_status_known(client):
    # Dispatch a build job then poll it
    r = client.post("/api/v1/build", json={})
    job_id = r.json()["job_id"]

    # Poll until terminal or timeout (10 s is plenty for a quickly-failing pio call)
    deadline = time.time() + 10
    status = None
    while time.time() < deadline:
        poll = client.get(f"/api/v1/jobs/{job_id}")
        assert poll.status_code == 200
        status = poll.json().get("status")
        if status in ("success", "failed", "error"):
            break
        time.sleep(0.5)

    # We don't assert success/fail because pio may not be available in all CI envs;
    # we just assert the job lifecycle works end-to-end.
    assert status in ("running", "success", "failed", "error")


def test_get_job_status_unknown(client):
    r = client.get("/api/v1/jobs/nonexistent-job-id")
    assert r.status_code == 404


# ── GET /api/v1/jobs ──────────────────────────────────────────────────────────

def test_list_jobs(client):
    client.post("/api/v1/build", json={})
    r = client.get("/api/v1/jobs")
    assert r.status_code == 200
    assert isinstance(r.json(), dict)


# ── GET /api/v1/telemetry ─────────────────────────────────────────────────────

def test_get_telemetry_present(client, tmp_path, monkeypatch):
    """Telemetry returns 200 when benchmark_results.json exists."""
    import host.server as server_module

    fake_results = {
        "timestamp": "2026-08-18T00:00:00",
        "latency_ms": {"avg": 0.38, "min": 0.03, "max": 8.62, "samples": 30},
        "static_memory_profile": {
            "tensor_arena_bytes": 81920,
            "bss_data_bytes": 27188,
            "total_static_ram_bytes": 109108,
            "flash_used_bytes": 540317,
        },
        "accuracy_evaluation": {"total_samples": 30, "correct_predictions": 5, "accuracy": 0.1667},
    }
    fake_path = tmp_path / "benchmark_results.json"
    fake_path.write_text(json.dumps(fake_results), encoding="utf-8")
    monkeypatch.setattr(server_module, "BENCHMARK_RESULTS_PATH", fake_path)

    r = client.get("/api/v1/telemetry")
    assert r.status_code == 200
    body = r.json()
    assert "latency_ms" in body
    assert body["latency_ms"]["avg"] == pytest.approx(0.38)


def test_get_telemetry_missing(client, tmp_path, monkeypatch):
    """Telemetry returns 404 when benchmark_results.json is absent."""
    import host.server as server_module
    monkeypatch.setattr(server_module, "BENCHMARK_RESULTS_PATH", tmp_path / "nonexistent.json")
    r = client.get("/api/v1/telemetry")
    assert r.status_code == 404


# ── GET /api/v1/logs ──────────────────────────────────────────────────────────

def test_get_recent_logs(client, tmp_path, monkeypatch):
    import host.server as server_module
    fake_log = tmp_path / "accord_log.jsonl"
    fake_log.write_text(
        json.dumps({"phase": "test", "status": "ok"}) + "\n", encoding="utf-8"
    )
    monkeypatch.setattr(server_module, "LOG_PATH", fake_log)
    r = client.get("/api/v1/logs?n=5")
    assert r.status_code == 200
    assert r.json()["count"] >= 1


# ── WebSocket /ws/logs/{job_id} ───────────────────────────────────────────────

def test_websocket_unknown_job(client):
    with client.websocket_connect("/ws/logs/nonexistent") as ws:
        msg = ws.receive_json()
        assert "error" in msg
