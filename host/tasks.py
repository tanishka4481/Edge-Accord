"""
host/tasks.py — Async Celery worker tasks for EdgeAccord build & benchmark pipeline.
Executes host/negotiator.py directly for constraint solving and build retries.
Supports a Redis broker (preferred) with automatic fallback to an in-process
thread-pool executor so the server runs without Redis for local development.
"""
from __future__ import annotations

import datetime
import json
import subprocess
import sys
import threading
import uuid
from pathlib import Path
from typing import Any, Dict, Optional

# ── paths ────────────────────────────────────────────────────────────────────
ROOT_DIR = Path(__file__).resolve().parents[1]
FIRMWARE_DIR = ROOT_DIR / "firmware"
LOG_PATH = ROOT_DIR / "accord_log.jsonl"
BENCHMARK_RESULTS_PATH = ROOT_DIR / "benchmark_results.json"
BUILD_FIRMWARE_BIN = FIRMWARE_DIR / ".pio" / "build" / "esp32dev" / "firmware.bin"

# ── in-process task store (used when Celery/Redis unavailable) ───────────────
_JOB_STORE: Dict[str, Dict[str, Any]] = {}
_JOB_LOCK = threading.Lock()

def _set_job(job_id: str, data: Dict[str, Any]) -> None:
    with _JOB_LOCK:
        _JOB_STORE[job_id] = data

def get_job(job_id: str) -> Optional[Dict[str, Any]]:
    with _JOB_LOCK:
        return _JOB_STORE.get(job_id)

def list_jobs() -> Dict[str, Dict[str, Any]]:
    with _JOB_LOCK:
        return dict(_JOB_STORE)

# ── Celery setup (optional) ──────────────────────────────────────────────────
try:
    from celery import Celery  # type: ignore
    _celery_app = Celery(
        "edgeaccord",
        broker="redis://localhost:6379/0",
        backend="redis://localhost:6379/0",
    )
    _celery_app.conf.update(
        task_serializer="json",
        result_serializer="json",
        accept_content=["json"],
        task_track_started=True,
    )
    CELERY_AVAILABLE = True
except ImportError:
    CELERY_AVAILABLE = False
    _celery_app = None

# ── helpers ──────────────────────────────────────────────────────────────────

def _parse_pio_memory(output: str) -> Dict[str, int]:
    """Extract RAM/Flash bytes from build/negotiator output."""
    import re
    ram = re.search(r"RAM:\s*\[[^\]]+\]\s*[\d.]+%\s*\(used\s*([\d,]+)\s*bytes", output)
    flash = re.search(r"Flash:\s*\[[^\]]+\]\s*[\d.]+%\s*\(used\s*([\d,]+)\s*bytes", output)
    return {
        "ram_used_bytes": int(ram.group(1).replace(",", "")) if ram else 0,
        "flash_used_bytes": int(flash.group(1).replace(",", "")) if flash else 0,
    }

def _append_log(record: Dict[str, Any]) -> None:
    with LOG_PATH.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record) + "\n")

# ── core task implementations ─────────────────────────────────────────────────

def execute_negotiator_build(job_id: str, config_override: Dict[str, Any]) -> Dict[str, Any]:
    """
    Executes host/negotiator.py directly for constraint solving and build iterations.
    """
    _set_job(job_id, {"status": "running", "started_at": datetime.datetime.now().isoformat(), "logs": ""})
    
    phase = config_override.get("phase", "shape_search")
    max_rounds = config_override.get("max_rounds", 5)
    use_stub_agent = config_override.get("use_stub_agent", True)
    
    cmd = [
        sys.executable,
        "-m", "host.negotiator",
        "--phase", str(phase),
        "--max-rounds", str(max_rounds),
    ]
    if use_stub_agent:
        cmd.append("--use-stub-agent")
    if config_override.get("toy_data", False):
        cmd.append("--toy-data")
    if "epochs" in config_override:
        cmd.extend(["--epochs", str(config_override["epochs"])])
        
    try:
        process = subprocess.Popen(
            cmd,
            cwd=ROOT_DIR,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        
        logs_collected = []
        while True:
            line = process.stdout.readline()
            if not line and process.poll() is not None:
                break
            if line:
                logs_collected.append(line)
                _set_job(job_id, {
                    "status": "running",
                    "started_at": datetime.datetime.now().isoformat(),
                    "logs": "".join(logs_collected),
                })
                
        rc = process.poll()
        combined_logs = "".join(logs_collected)
        memory = _parse_pio_memory(combined_logs)
        success = (rc == 0)
        firmware_bin_exists = BUILD_FIRMWARE_BIN.exists()
        
        record = {
            "job_id": job_id,
            "phase": "negotiator_build_task",
            "timestamp": datetime.datetime.now().isoformat(),
            "success": success,
            "return_code": rc,
            **memory,
            "firmware_bin_exists": firmware_bin_exists,
            "config_override": config_override,
        }
        _append_log(record)
        
        job_result = {
            "status": "success" if success else "failed",
            "finished_at": datetime.datetime.now().isoformat(),
            "logs": combined_logs,
            **record,
        }
        _set_job(job_id, job_result)
        return job_result
        
    except Exception as exc:
        err_res = {"status": "error", "error": str(exc), "finished_at": datetime.datetime.now().isoformat()}
        _set_job(job_id, err_res)
        return err_res


def execute_benchmark_sim(job_id: str, model_path: str) -> Dict[str, Any]:
    """Runs the local TFLite benchmark and stores structured results."""
    _set_job(job_id, {"status": "running", "started_at": datetime.datetime.now().isoformat(), "logs": ""})
    cmd = [sys.executable, str(ROOT_DIR / "host" / "benchmark.py")]
    try:
        result = subprocess.run(
            cmd, cwd=ROOT_DIR, capture_output=True, text=True, check=False, timeout=600
        )
        success = result.returncode == 0
        metrics: Dict[str, Any] = {}
        if BENCHMARK_RESULTS_PATH.exists():
            metrics = json.loads(BENCHMARK_RESULTS_PATH.read_text(encoding="utf-8"))
        record = {
            "job_id": job_id,
            "phase": "benchmark_sim_task",
            "timestamp": datetime.datetime.now().isoformat(),
            "success": success,
            "model_path": model_path,
            **metrics,
        }
        _append_log(record)
        job_result = {
            "status": "success" if success else "failed",
            "finished_at": datetime.datetime.now().isoformat(),
            "logs": result.stdout + result.stderr,
            **record,
        }
        _set_job(job_id, job_result)
        return job_result
    except Exception as exc:
        err_res = {"status": "error", "error": str(exc), "finished_at": datetime.datetime.now().isoformat()}
        _set_job(job_id, err_res)
        return err_res


# ── Celery Task Registrations ─────────────────────────────────────────────────
if CELERY_AVAILABLE and _celery_app:
    @_celery_app.task(name="host.tasks.compile_firmware_task")
    def compile_firmware_task(config_override: Dict[str, Any] | None = None) -> Dict[str, Any]:
        job_id = str(uuid.uuid4())
        return execute_negotiator_build(job_id, config_override or {})

    @_celery_app.task(name="host.tasks.benchmark_sim_task")
    def benchmark_sim_task(model_path: str | None = None) -> Dict[str, Any]:
        job_id = str(uuid.uuid4())
        model_path = model_path or str(ROOT_DIR / "build" / "final" / "model.tflite")
        return execute_benchmark_sim(job_id, model_path)


# ── public dispatch functions ─────────────────────────────────────────────────

def dispatch_compile_firmware(config_override: Dict[str, Any] | None = None) -> str:
    """
    Dispatch build job executing host/negotiator.py; returns job_id.
    """
    job_id = str(uuid.uuid4())
    config_override = config_override or {}
    t = threading.Thread(target=execute_negotiator_build, args=(job_id, config_override), daemon=True)
    t.start()
    return job_id


def dispatch_benchmark_sim(model_path: str | None = None) -> str:
    """Dispatch benchmark simulation job; returns job_id."""
    job_id = str(uuid.uuid4())
    model_path = model_path or str(ROOT_DIR / "build" / "final" / "model.tflite")
    t = threading.Thread(target=execute_benchmark_sim, args=(job_id, model_path), daemon=True)
    t.start()
    return job_id
