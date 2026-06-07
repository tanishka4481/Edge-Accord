from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Any, Sequence

from .types import ModelConfig, Proposal

ALLOWED_TECHNIQUES = {"quantize_int8", "prune_filters", "reduce_input_features"}
ALLOWED_TARGETS = {"whole_model", "layer_name"}


class AgentError(RuntimeError):
    pass


@dataclass
class AgentContext:
    config: ModelConfig
    violations: list[str]
    round_index: int
    phase: str = "shape_search"
    last_proposal: Proposal | None = None

    def to_prompt_dict(self) -> dict[str, Any]:
        return {
            "round_index": self.round_index,
            "phase": self.phase,
            "current_config": self.config.to_dict(),
            "violations": self.violations,
            "last_proposal": None if self.last_proposal is None else self.last_proposal.to_dict(),
        }


class BaseQuantAgent:
    def next_proposal(self, context: AgentContext) -> Proposal:
        raise NotImplementedError


class StubQuantAgent(BaseQuantAgent):
    def __init__(self, proposals: Sequence[Proposal]):
        self._proposals = list(proposals)
        self._index = 0

    def next_proposal(self, context: AgentContext) -> Proposal:
        if not self._proposals:
            return Proposal(
                technique="quantize_int8",
                target="whole_model",
                params={},
                rationale="Default to int8 quantization when no proposals are configured.",
            )
        proposal = self._proposals[min(self._index, len(self._proposals) - 1)]
        self._index += 1
        return proposal


class GeminiQuantAgent(BaseQuantAgent):
    def __init__(self, api_key: str | None = None, model_name: str = "gemini-1.5-flash"):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise AgentError("GEMINI_API_KEY is not set")
        self.model_name = model_name

    def _load_client(self):
        try:
            import google.generativeai as genai  # type: ignore
        except ImportError as exc:  # pragma: no cover
            raise AgentError("google-generativeai is required for GeminiQuantAgent") from exc
        genai.configure(api_key=self.api_key)
        return genai

    def _prompt(self, context: AgentContext) -> str:
        return (
            "You are EdgeAccord's compression planner. Return only valid JSON matching this schema: "
            '{"technique":"quantize_int8|prune_filters|reduce_input_features","target":"whole_model|layer_name",'
            '"params":{},"rationale":"short string"}.\n\n'
            f"Context: {json.dumps(context.to_prompt_dict(), indent=2)}\n\n"
            "Rules: do not propose int4, do not propose hardware design, do not use LangChain/CrewAI, "
            "prefer the smallest change that could fix the reported violation, and explain the rationale briefly."
        )

    def _extract_json(self, text: str) -> dict[str, Any]:
        text = text.strip()
        fenced = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, re.DOTALL)
        if fenced:
            text = fenced.group(1)
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1:
            raise AgentError("Model response did not contain JSON")
        return json.loads(text[start : end + 1])

    def next_proposal(self, context: AgentContext) -> Proposal:
        genai = self._load_client()
        model = genai.GenerativeModel(self.model_name)
        prompt = self._prompt(context)

        last_error = None
        for _ in range(3):
            response = model.generate_content(prompt)
            try:
                payload = self._extract_json(response.text or "")
                proposal = Proposal(
                    technique=str(payload["technique"]),
                    target=str(payload["target"]),
                    params=dict(payload.get("params", {})),
                    rationale=str(payload.get("rationale", "")),
                )
                if proposal.technique not in ALLOWED_TECHNIQUES:
                    raise AgentError(f"Unsupported technique: {proposal.technique}")
                if proposal.target not in ALLOWED_TARGETS:
                    raise AgentError(f"Unsupported target: {proposal.target}")
                return proposal
            except Exception as exc:  # pragma: no cover - retry path depends on remote model output
                last_error = exc
                prompt = prompt + f"\n\nThe previous response was invalid because: {exc}. Return only JSON."

        raise AgentError("Failed to obtain a valid proposal from Gemini") from last_error


def normalize_proposal(payload: dict[str, Any]) -> Proposal:
    technique = str(payload.get("technique", "")).strip()
    target = str(payload.get("target", "")).strip()
    params = dict(payload.get("params", {}))
    rationale = str(payload.get("rationale", "")).strip()
    if technique not in ALLOWED_TECHNIQUES:
        raise AgentError(f"Unsupported technique: {technique}")
    if target not in ALLOWED_TARGETS:
        raise AgentError(f"Unsupported target: {target}")
    return Proposal(technique=technique, target=target, params=params, rationale=rationale)
