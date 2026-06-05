from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Literal

Technique = Literal["quantize_int8", "prune_filters", "reduce_input_features"]
Target = Literal["whole_model", "layer_name"]
Phase = Literal["shape_search", "train"]

DEFAULT_LABELS = ("yes", "no", "stop", "go", "unknown", "silence")


@dataclass(frozen=True)
class ModelConfig:
    time_steps: int = 40
    mfcc_bins: int = 10
    classes: tuple[str, ...] = DEFAULT_LABELS
    conv_filters: tuple[int, int, int] = (16, 24, 32)
    dense_units: int = 32
    dropout_rate: float = 0.1
    learning_rate: float = 0.001
    seed: int = 7

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["classes"] = list(self.classes)
        payload["conv_filters"] = list(self.conv_filters)
        return payload


@dataclass(frozen=True)
class Proposal:
    technique: Technique
    target: Target
    params: dict[str, Any] = field(default_factory=dict)
    rationale: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "technique": self.technique,
            "target": self.target,
            "params": self.params,
            "rationale": self.rationale,
        }


@dataclass(frozen=True)
class CompilationArtifact:
    config: ModelConfig
    tflite_path: Path
    header_path: Path
    byte_size: int
    parameter_count: int
    source_mode: Literal["random_init", "trained"]

    def to_dict(self) -> dict[str, Any]:
        return {
            "config": self.config.to_dict(),
            "tflite_path": str(self.tflite_path),
            "header_path": str(self.header_path),
            "byte_size": self.byte_size,
            "parameter_count": self.parameter_count,
            "source_mode": self.source_mode,
        }


@dataclass(frozen=True)
class ConstraintResult:
    passed: bool
    ram_used_bytes: int | None
    ram_total_bytes: int | None
    flash_used_bytes: int | None
    flash_total_bytes: int | None
    violations: list[str]
    command: str
    return_code: int
    stdout: str
    stderr: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "ram_used_bytes": self.ram_used_bytes,
            "ram_total_bytes": self.ram_total_bytes,
            "flash_used_bytes": self.flash_used_bytes,
            "flash_total_bytes": self.flash_total_bytes,
            "violations": self.violations,
            "command": self.command,
            "return_code": self.return_code,
            "stdout": self.stdout,
            "stderr": self.stderr,
        }


@dataclass(frozen=True)
class NegotiationRecord:
    round_index: int
    phase: Phase
    proposal: Proposal
    artifact: CompilationArtifact
    constraint: ConstraintResult | None = None
    accepted: bool = False
    accuracy: float | None = None
    metrics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "round_index": self.round_index,
            "phase": self.phase,
            "proposal": self.proposal.to_dict(),
            "artifact": self.artifact.to_dict(),
            "constraint": None if self.constraint is None else self.constraint.to_dict(),
            "accepted": self.accepted,
            "accuracy": self.accuracy,
            "metrics": self.metrics,
        }
