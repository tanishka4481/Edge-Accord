"""
host/server.py — FastAPI REST + WebSocket server for EdgeAccord.

Run with:
    uvicorn host.server:app --reload --port 8000
"""
from __future__ import annotations

import asyncio
import json
import uuid
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from .tasks import (
    dispatch_benchmark_sim,
    dispatch_compile_firmware,
    get_job,
    list_jobs,
)

ROOT_DIR = Path(__file__).resolve().parents[1]
BENCHMARK_RESULTS_PATH = ROOT_DIR / "benchmark_results.json"
LOG_PATH = ROOT_DIR / "accord_log.jsonl"

app = FastAPI(
    title="EdgeAccord API",
    description="REST + WebSocket API for the EdgeAccord embedded-ML negotiation pipeline.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── request / response schemas ────────────────────────────────────────────────

class BuildRequest(BaseModel):
    config_override: Optional[Dict[str, Any]] = None


class BenchmarkRequest(BaseModel):
    model_path: Optional[str] = None


class JobResponse(BaseModel):
    job_id: str
    status: str
    message: str


# ── endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health")
async def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.post("/api/v1/build", response_model=JobResponse, status_code=202)
async def post_build(req: BuildRequest = BuildRequest()) -> JobResponse:
    """Dispatch a background firmware compilation job via `pio run`."""
    job_id = dispatch_compile_firmware(req.config_override)
    return JobResponse(
        job_id=job_id,
        status="queued",
        message="Firmware compilation job dispatched.",
    )


@app.post("/api/v1/benchmark", response_model=JobResponse, status_code=202)
async def post_benchmark(req: BenchmarkRequest = BenchmarkRequest()) -> JobResponse:
    """Dispatch a background TFLite benchmark simulation job."""
    job_id = dispatch_benchmark_sim(req.model_path)
    return JobResponse(
        job_id=job_id,
        status="queued",
        message="Benchmark simulation job dispatched.",
    )


@app.get("/api/v1/jobs/{job_id}")
async def get_job_status(job_id: str) -> Dict[str, Any]:
    """Poll the status and results of a dispatched job."""
    job = get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found.")
    # Omit heavy log blobs from status polls
    safe = {k: v for k, v in job.items() if k != "logs"}
    return safe


@app.get("/api/v1/jobs")
async def list_all_jobs() -> Dict[str, Any]:
    """List all tracked jobs (status only, no log blobs)."""
    return {jid: {k: v for k, v in data.items() if k != "logs"} for jid, data in list_jobs().items()}


@app.get("/api/v1/telemetry")
async def get_telemetry() -> Dict[str, Any]:
    """Return latest benchmark metrics from benchmark_results.json."""
    if not BENCHMARK_RESULTS_PATH.exists():
        raise HTTPException(status_code=404, detail="benchmark_results.json not found. Run a benchmark first.")
    data = json.loads(BENCHMARK_RESULTS_PATH.read_text(encoding="utf-8"))
    return data


@app.get("/api/v1/logs")
async def get_recent_log_entries(n: int = 10) -> Dict[str, Any]:
    """Return the last N entries from accord_log.jsonl."""
    if not LOG_PATH.exists():
        return {"entries": []}
    lines = LOG_PATH.read_text(encoding="utf-8").strip().splitlines()
    entries = []
    for line in lines[-n:]:
        try:
            entries.append(json.loads(line))
        except Exception:
            pass
    return {"count": len(entries), "entries": entries}


# ── WebSocket log streaming ───────────────────────────────────────────────────

@app.websocket("/ws/logs/{job_id}")
async def ws_logs(websocket: WebSocket, job_id: str) -> None:
    """Stream stdout/stderr logs for a job in real-time over WebSocket."""
    await websocket.accept()
    poll_interval = 0.5
    max_wait = 300  # 5 minutes
    elapsed = 0.0
    last_len = 0

    try:
        while elapsed < max_wait:
            job = get_job(job_id)
            if job is None:
                await websocket.send_json({"error": f"Job '{job_id}' not found."})
                break

            logs: str = job.get("logs", "")
            if len(logs) > last_len:
                new_chunk = logs[last_len:]
                await websocket.send_text(new_chunk)
                last_len = len(logs)

            status = job.get("status", "")
            if status in ("success", "failed", "error"):
                await websocket.send_json({"status": status, "finished": True})
                break

            await asyncio.sleep(poll_interval)
            elapsed += poll_interval

        if elapsed >= max_wait:
            await websocket.send_json({"error": "Timed out waiting for job completion."})
    except WebSocketDisconnect:
        pass
    finally:
        await websocket.close()
