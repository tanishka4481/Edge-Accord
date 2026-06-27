from dataclasses import replace
from pathlib import Path

import pytest

from host.negotiator import Negotiator
from host.types import CompilationArtifact, ConstraintResult, ModelConfig, Proposal


class FakeConstraintEngine:
    def __init__(self):
        self.round_index = 0
        self.staged_paths: list[Path] = []

    def stage_model_header(self, header_path: Path) -> Path:
        self.staged_paths.append(header_path)
        return header_path

    def run(self):
        self.round_index += 1
        if self.round_index == 1:
            return ConstraintResult(
                passed=False,
                ram_used_bytes=340000,
                ram_total_bytes=327680,
                flash_used_bytes=900000,
                flash_total_bytes=1310720,
                violations=["RAM exceeded by 32320 bytes"],
                command="pio run",
                return_code=1,
                stdout="",
                stderr="",
            )
        return ConstraintResult(
            passed=True,
            ram_used_bytes=120000,
            ram_total_bytes=327680,
            flash_used_bytes=700000,
            flash_total_bytes=1310720,
            violations=[],
            command="pio run",
            return_code=0,
            stdout="",
            stderr="",
        )


class FixedAgent:
    def __init__(self, proposals):
        self.proposals = list(proposals)
        self.calls = 0

    def next_proposal(self, context):
        proposal = self.proposals[min(self.calls, len(self.proposals) - 1)]
        self.calls += 1
        return proposal


def fake_compile_proposal(proposal, base_config=None, output_root=None, source_mode="random_init", seed=None):
    config = base_config or ModelConfig()
    if proposal.technique == "prune_filters":
        config = replace(config, conv_filters=(8, 12, 16), dense_units=16)
    header_path = Path(output_root) / "model_data.h"
    header_path.parent.mkdir(parents=True, exist_ok=True)
    header_path.write_text("#pragma once\n", encoding="utf-8")
    return CompilationArtifact(
        config=config,
        tflite_path=Path(output_root) / "model.tflite",
        header_path=header_path,
        byte_size=128,
        parameter_count=1024,
        source_mode=source_mode,
    )


def test_shape_search_converges_with_rejected_rounds(monkeypatch, tmp_path: Path):
    from host import negotiator as negotiator_module

    proposals = [
        Proposal(
            technique="prune_filters",
            target="whole_model",
            params={"prune_ratio": 0.5},
            rationale="Reduce conv widths.",
        )
    ]
    monkeypatch.setattr(negotiator_module, "compile_proposal", fake_compile_proposal)

    negotiator = Negotiator(log_path=tmp_path / "accord_log.jsonl", agent=FixedAgent(proposals))
    negotiator.constraint_engine = FakeConstraintEngine()

    record = negotiator.run_shape_search(max_rounds=3)
    assert record.accepted is True
    assert record.constraint is not None
    assert record.constraint.passed is True
    assert len(negotiator.records) == 2
    assert (tmp_path / "accord_log.jsonl").exists()


@pytest.mark.integration
def test_real_platformio_path_is_optional_but_supported():
    pytest.importorskip("tensorflow")
    import shutil

    if shutil.which("pio") is None:
        pytest.skip("PlatformIO CLI is not installed in this environment")
