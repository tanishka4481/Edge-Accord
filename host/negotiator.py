from __future__ import annotations

import argparse
import json
import os
from dataclasses import replace
from pathlib import Path
from typing import Any

from .compiler import build_base_config, compile_proposal, summarise_artifact
from .constraint_engine import ConstraintEngine, HardwareBudget
from .quant_agent import AgentContext, BaseQuantAgent, GeminiQuantAgent, StubQuantAgent, normalize_proposal
from .types import NegotiationRecord, Phase, Proposal
from .train_model import train_model

ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_LOG_PATH = ROOT_DIR / "accord_log.jsonl"
DEFAULT_FIRMWARE_DIR = ROOT_DIR / "firmware"
DEFAULT_HARDWARE_SPEC = ROOT_DIR / "host" / "hardware_specs.json"
DEFAULT_BUILD_DIR = ROOT_DIR / "build"


class Negotiator:
    def __init__(
        self,
        firmware_dir: Path | str = DEFAULT_FIRMWARE_DIR,
        hardware_spec_path: Path | str = DEFAULT_HARDWARE_SPEC,
        log_path: Path | str = DEFAULT_LOG_PATH,
        pio_executable: str = "pio",
        agent: BaseQuantAgent | None = None,
    ):
        self.firmware_dir = Path(firmware_dir)
        self.hardware_spec_path = Path(hardware_spec_path)
        self.log_path = Path(log_path)
        self.agent = agent
        self.constraint_engine = ConstraintEngine(
            self.firmware_dir,
            budget=HardwareBudget.from_json(self.hardware_spec_path),
            pio_executable=pio_executable,
        )
        self.current_config = build_base_config()
        self.records: list[NegotiationRecord] = []

    def _append_log(self, record: NegotiationRecord) -> None:
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with self.log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record.to_dict(), indent=None) + "\n")

    def _run_round(self, round_index: int, phase: Phase, proposal: Proposal) -> NegotiationRecord:
        artifact = compile_proposal(
            proposal,
            base_config=self.current_config,
            output_root=DEFAULT_BUILD_DIR / f"round_{round_index:02d}",
            source_mode="random_init" if phase == "shape_search" else "trained",
            seed=self.current_config.seed + round_index,
        )
        self.constraint_engine.stage_model_header(artifact.header_path)
        result = None
        if phase == "shape_search":
            result = self.constraint_engine.run()
        accepted = result is not None and result.passed
        record = NegotiationRecord(
            round_index=round_index,
            phase=phase,
            proposal=proposal,
            artifact=artifact,
            constraint=result,
            accepted=accepted,
            metrics=summarise_artifact(artifact),
        )
        self.records.append(record)
        self._append_log(record)
        self.current_config = artifact.config
        return record

    def run_shape_search(self, max_rounds: int = 5) -> NegotiationRecord:
        if self.agent is None:
            self.agent = GeminiQuantAgent()

        current_proposal = Proposal(
            technique="quantize_int8",
            target="whole_model",
            params={},
            rationale="Start with a full int8 baseline before pruning or feature reduction.",
        )
        last_record = None
        for round_index in range(max_rounds):
            if round_index > 0:
                context = AgentContext(
                    config=self.current_config,
                    violations=[] if last_record is None or last_record.constraint is None else last_record.constraint.violations,
                    round_index=round_index,
                    phase="shape_search",
                    last_proposal=None if last_record is None else last_record.proposal,
                )
                current_proposal = self.agent.next_proposal(context)
                if isinstance(current_proposal, dict):
                    current_proposal = normalize_proposal(current_proposal)

            record = self._run_round(round_index, "shape_search", current_proposal)
            last_record = record
            if record.accepted:
                return record
        raise RuntimeError("Shape search did not converge within the configured round budget")

    def run_training_phase(self, data_dir: Path | None = None, epochs: int = 10, toy_data: bool = True) -> dict[str, Any]:
        metrics = train_model(data_dir=data_dir, epochs=epochs, use_toy_data=toy_data)
        proposal = Proposal(
            technique="quantize_int8",
            target="whole_model",
            params={},
            rationale="Re-verify the trained winning shape against the hardware budget.",
        )
        artifact = compile_proposal(
            proposal,
            base_config=self.current_config,
            output_root=DEFAULT_BUILD_DIR / "final",
            source_mode="trained",
            seed=self.current_config.seed,
        )
        self.constraint_engine.stage_model_header(artifact.header_path)
        result = self.constraint_engine.run()
        record = NegotiationRecord(
            round_index=len(self.records),
            phase="train",
            proposal=proposal,
            artifact=artifact,
            constraint=result,
            accepted=result.passed,
            accuracy=float(metrics.get("val_accuracy", metrics.get("accuracy", 0.0))),
            metrics={**metrics, **summarise_artifact(artifact)},
        )
        self.records.append(record)
        self._append_log(record)
        return {"training": metrics, "constraint": result.to_dict(), "artifact": artifact.to_dict()}


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the EdgeAccord negotiation loop.")
    parser.add_argument("--phase", choices=["shape_search", "train"], default="shape_search")
    parser.add_argument("--max-rounds", type=int, default=5)
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--toy-data", action="store_true")
    parser.add_argument("--pio", type=str, default=os.getenv("PIO_EXE", "pio"))
    parser.add_argument("--use-stub-agent", action="store_true")
    return parser


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()
    agent = StubQuantAgent([]) if args.use_stub_agent else None
    negotiator = Negotiator(pio_executable=args.pio, agent=agent)
    if args.phase == "shape_search":
        negotiator.run_shape_search(max_rounds=args.max_rounds)
    else:
        negotiator.run_training_phase(data_dir=args.data_dir, epochs=args.epochs, toy_data=args.toy_data)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
