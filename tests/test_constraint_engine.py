from pathlib import Path

import pytest

from host.constraint_engine import HardwareBudget, parse_platformio_output


FIXTURE = Path(__file__).parent / "fixtures" / "pio_memory_sample.txt"


def test_parse_platformio_output_extracts_ram_and_flash():
    summary = parse_platformio_output(FIXTURE.read_text(encoding="utf-8"))

    assert summary["ram_used_bytes"] == 92928
    assert summary["ram_total_bytes"] == 327680
    assert summary["flash_used_bytes"] == 838656
    assert summary["flash_total_bytes"] == 1310720


def test_hardware_budget_reads_json(tmp_path: Path):
    spec_path = tmp_path / "hardware_specs.json"
    spec_path.write_text(
        "{\n"
        '  "usable_ram_bytes": 12345,\n'
        '  "flash_partition_bytes": 54321\n'
        "}\n",
        encoding="utf-8",
    )

    budget = HardwareBudget.from_json(spec_path)
    assert budget.usable_ram_bytes == 12345
    assert budget.flash_partition_bytes == 54321
