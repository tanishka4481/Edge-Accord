"""
host/telemetry.py — Prometheus-compatible metrics exporter for EdgeAccord.

Exports:
  edge_accord_flash_bytes_used
  edge_accord_ram_bytes_used
  edge_accord_inference_latency_ms (alias to avg latency)
  edge_accord_inference_latency_ms_avg
  edge_accord_inference_latency_ms_min
  edge_accord_inference_latency_ms_max
  edge_accord_accuracy_ratio

Usage:
    python -m host.telemetry               # print current metrics as text
    python -m host.telemetry --json        # dump as JSON
    python -m host.telemetry --serve 9090  # start HTTP /metrics endpoint on port 9090
"""
from __future__ import annotations

import argparse
import json
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any, Dict

ROOT_DIR = Path(__file__).resolve().parents[1]
BENCHMARK_RESULTS_PATH = ROOT_DIR / "benchmark_results.json"
LOG_PATH = ROOT_DIR / "accord_log.jsonl"


# ── metric collection ─────────────────────────────────────────────────────────

def collect_metrics() -> Dict[str, Any]:
    """Gather latest metrics from benchmark_results.json and accord_log.jsonl."""
    metrics: Dict[str, Any] = {
        "edge_accord_flash_bytes_used": 0,
        "edge_accord_ram_bytes_used": 0,
        "edge_accord_inference_latency_ms": 0.0,
        "edge_accord_inference_latency_ms_avg": 0.0,
        "edge_accord_inference_latency_ms_min": 0.0,
        "edge_accord_inference_latency_ms_max": 0.0,
        "edge_accord_accuracy_ratio": 0.0,
        "edge_accord_latency_samples": 0,
    }

    # Pull from benchmark_results.json (authoritative for latency + accuracy)
    if BENCHMARK_RESULTS_PATH.exists():
        try:
            data = json.loads(BENCHMARK_RESULTS_PATH.read_text(encoding="utf-8"))
            lat = data.get("latency_ms", {})
            mem = data.get("static_memory_profile", {})
            acc = data.get("accuracy_evaluation", {})
            avg_lat = float(lat.get("avg", 0.0))
            metrics["edge_accord_inference_latency_ms"] = avg_lat
            metrics["edge_accord_inference_latency_ms_avg"] = avg_lat
            metrics["edge_accord_inference_latency_ms_min"] = float(lat.get("min", 0.0))
            metrics["edge_accord_inference_latency_ms_max"] = float(lat.get("max", 0.0))
            metrics["edge_accord_latency_samples"] = int(lat.get("samples", 0))
            metrics["edge_accord_ram_bytes_used"] = int(mem.get("total_static_ram_bytes", 0))
            metrics["edge_accord_flash_bytes_used"] = int(mem.get("flash_used_bytes", 0))
            metrics["edge_accord_accuracy_ratio"] = float(acc.get("accuracy", 0.0))
        except Exception:
            pass

    # Fall back / supplement from accord_log.jsonl if benchmark_results has 0 bytes for RAM/Flash
    if (metrics["edge_accord_flash_bytes_used"] == 0 or metrics["edge_accord_ram_bytes_used"] == 0) and LOG_PATH.exists():
        try:
            for line in reversed(LOG_PATH.read_text(encoding="utf-8").strip().splitlines()):
                if not line.strip():
                    continue
                entry = json.loads(line)
                if "flash_used" in entry and entry["flash_used"] is not None:
                    metrics["edge_accord_flash_bytes_used"] = int(entry["flash_used"])
                    metrics["edge_accord_ram_bytes_used"] = int(entry.get("ram_used") or 0)
                    break
                if "flash_used_bytes" in entry and entry["flash_used_bytes"]:
                    metrics["edge_accord_flash_bytes_used"] = int(entry["flash_used_bytes"])
                    metrics["edge_accord_ram_bytes_used"] = int(entry.get("ram_used_bytes") or 0)
                    break
                if "static_memory_profile" in entry:
                    mem = entry["static_memory_profile"]
                    if mem.get("flash_used_bytes"):
                        metrics["edge_accord_flash_bytes_used"] = int(mem["flash_used_bytes"])
                        metrics["edge_accord_ram_bytes_used"] = int(mem.get("total_static_ram_bytes", 0))
                        break
                if "constraint" in entry and entry["constraint"]:
                    c = entry["constraint"]
                    if c.get("flash_used_bytes"):
                        metrics["edge_accord_flash_bytes_used"] = int(c["flash_used_bytes"])
                        metrics["edge_accord_ram_bytes_used"] = int(c.get("ram_used_bytes", 0))
                        break
        except Exception:
            pass

    return metrics


# ── Prometheus text format ────────────────────────────────────────────────────

def to_prometheus_text(metrics: Dict[str, Any]) -> str:
    lines = [
        "# HELP edge_accord_flash_bytes_used Firmware Flash usage in bytes",
        "# TYPE edge_accord_flash_bytes_used gauge",
        f"edge_accord_flash_bytes_used {metrics['edge_accord_flash_bytes_used']}",
        "# HELP edge_accord_ram_bytes_used Static RAM usage in bytes",
        "# TYPE edge_accord_ram_bytes_used gauge",
        f"edge_accord_ram_bytes_used {metrics['edge_accord_ram_bytes_used']}",
        "# HELP edge_accord_inference_latency_ms TFLite Invoke() latency in ms",
        "# TYPE edge_accord_inference_latency_ms gauge",
        f"edge_accord_inference_latency_ms {metrics['edge_accord_inference_latency_ms']}",
        "# HELP edge_accord_inference_latency_ms_avg Average TFLite Invoke() latency in ms",
        "# TYPE edge_accord_inference_latency_ms_avg gauge",
        f"edge_accord_inference_latency_ms_avg {metrics['edge_accord_inference_latency_ms_avg']}",
        "# HELP edge_accord_inference_latency_ms_min Min TFLite Invoke() latency in ms",
        "# TYPE edge_accord_inference_latency_ms_min gauge",
        f"edge_accord_inference_latency_ms_min {metrics['edge_accord_inference_latency_ms_min']}",
        "# HELP edge_accord_inference_latency_ms_max Max TFLite Invoke() latency in ms",
        "# TYPE edge_accord_inference_latency_ms_max gauge",
        f"edge_accord_inference_latency_ms_max {metrics['edge_accord_inference_latency_ms_max']}",
        "# HELP edge_accord_accuracy_ratio Inference accuracy ratio (0.0–1.0)",
        "# TYPE edge_accord_accuracy_ratio gauge",
        f"edge_accord_accuracy_ratio {metrics['edge_accord_accuracy_ratio']}",
    ]
    return "\n".join(lines) + "\n"


# ── HTTP server ───────────────────────────────────────────────────────────────

class MetricsHandler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):  # suppress default access logs
        pass

    def do_GET(self):
        if self.path == "/metrics":
            body = to_prometheus_text(collect_metrics()).encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/plain; version=0.0.4; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif self.path in ("/health", "/"):
            body = b'{"status":"ok"}'
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(body)
        else:
            self.send_response(404)
            self.end_headers()


def serve(port: int = 9090) -> None:
    print(f"[telemetry] Prometheus /metrics server listening on http://0.0.0.0:{port}/metrics")
    HTTPServer(("0.0.0.0", port), MetricsHandler).serve_forever()


# ── CLI entrypoint ────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="EdgeAccord telemetry exporter")
    parser.add_argument("--json", action="store_true", help="Output metrics as JSON")
    parser.add_argument("--serve", type=int, metavar="PORT", default=0, help="Start HTTP /metrics server on PORT")
    args = parser.parse_args()

    if args.serve:
        serve(args.serve)
    else:
        metrics = collect_metrics()
        if args.json:
            print(json.dumps(metrics, indent=2))
        else:
            print(to_prometheus_text(metrics))


if __name__ == "__main__":
    main()
