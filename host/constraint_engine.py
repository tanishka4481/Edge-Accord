from __future__ import annotations

import json
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .types import ConstraintResult

RAM_PATTERN = re.compile(
    r"RAM:\s*\[[^\]]+\]\s*(?P<percent>[\d.]+)%\s*\(used\s*(?P<used>[\d,]+)\s*bytes\s*from\s*(?P<total>[\d,]+)\s*bytes\)",
    re.IGNORECASE,
)
FLASH_PATTERN = re.compile(
    r"Flash:\s*\[[^\]]+\]\s*(?P<percent>[\d.]+)%\s*\(used\s*(?P<used>[\d,]+)\s*bytes\s*from\s*(?P<total>[\d,]+)\s*bytes\)",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class HardwareBudget:
    usable_ram_bytes: int
    flash_partition_bytes: int
    tensor_arena_bytes: int = 0

    @classmethod
    def from_json(cls, path: Path) -> "HardwareBudget":
        payload = json.loads(path.read_text(encoding="utf-8"))
        return cls(
            usable_ram_bytes=int(payload["usable_ram_bytes"]),
            flash_partition_bytes=int(payload["flash_partition_bytes"]),
            tensor_arena_bytes=int(payload.get("tensor_arena_bytes", 0)),
        )


def _clean_int(value: str) -> int:
    return int(value.replace(",", ""))


def parse_platformio_output(output: str) -> dict[str, int]:
    ram_match = RAM_PATTERN.search(output)
    flash_match = FLASH_PATTERN.search(output)
    if not ram_match or not flash_match:
        raise ValueError("Could not parse PlatformIO memory summary from build output")

    return {
        "ram_used_bytes": _clean_int(ram_match.group("used")),
        "ram_total_bytes": _clean_int(ram_match.group("total")),
        "flash_used_bytes": _clean_int(flash_match.group("used")),
        "flash_total_bytes": _clean_int(flash_match.group("total")),
    }


class ConstraintEngine:
    def __init__(self, firmware_dir: Path | str, budget: HardwareBudget | None = None, pio_executable: str = "pio"):
        self.firmware_dir = Path(firmware_dir)
        self.budget = budget
        self.pio_executable = pio_executable

    def stage_model_header(self, header_path: Path) -> Path:
        target = self.firmware_dir / "src" / "model_data.h"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(header_path.read_text(encoding="utf-8"), encoding="utf-8")
        return target

    def run(self, header_path: Path | None = None) -> ConstraintResult:
        if header_path is not None:
            self.stage_model_header(header_path)

        if shutil.which(self.pio_executable) is None:
            raise FileNotFoundError(
                f"{self.pio_executable} not found. Install PlatformIO CLI or set pio_executable to its path."
            )

        command = [self.pio_executable, "run"]
        completed = subprocess.run(
            command,
            cwd=self.firmware_dir,
            capture_output=True,
            text=True,
            check=False,
        )

        combined_output = (completed.stdout or "") + "\n" + (completed.stderr or "")
        summary = parse_platformio_output(combined_output)
        violations: list[str] = []
        budget = self.budget
        if budget is not None:
            if summary["ram_used_bytes"] > budget.usable_ram_bytes:
                overflow = summary["ram_used_bytes"] - budget.usable_ram_bytes
                violations.append(f"RAM exceeded by {overflow} bytes")
            if summary["flash_used_bytes"] > budget.flash_partition_bytes:
                overflow = summary["flash_used_bytes"] - budget.flash_partition_bytes
                violations.append(f"Flash exceeded by {overflow} bytes")

        return ConstraintResult(
            passed=not violations and completed.returncode == 0,
            ram_used_bytes=summary["ram_used_bytes"],
            ram_total_bytes=summary["ram_total_bytes"],
            flash_used_bytes=summary["flash_used_bytes"],
            flash_total_bytes=summary["flash_total_bytes"],
            violations=violations,
            command=" ".join(command),
            return_code=completed.returncode,
            stdout=completed.stdout,
            stderr=completed.stderr,
        )
