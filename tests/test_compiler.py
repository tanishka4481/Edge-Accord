from pathlib import Path

import pytest

from host.compiler import apply_proposal_to_config, emit_model_header, make_model, convert_to_tflite_bytes
from host.types import ModelConfig, Proposal


def test_emit_model_header_writes_expected_symbols(tmp_path: Path):
    header_path = tmp_path / "model_data.h"
    emit_model_header(b"\x01\x02\x03\x04", header_path)

    content = header_path.read_text(encoding="utf-8")
    assert "g_model[]" in content
    assert "g_model_len" in content
    assert "0x01" in content


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def test_pruning_reduces_tflite_size_when_tensorflow_is_available():
    tf = pytest.importorskip("tensorflow")

    base_config = ModelConfig(time_steps=20, mfcc_bins=8, conv_filters=(8, 12, 16), dense_units=16)
    base_model = make_model(base_config, seed=1)
    base_tflite = convert_to_tflite_bytes(base_model, base_config)

    proposal = Proposal(
        technique="prune_filters",
        target="whole_model",
        params={"prune_ratio": 0.5},
        rationale="Shrink the network.",
    )
    pruned_config = apply_proposal_to_config(base_config, proposal)
    pruned_model = make_model(pruned_config, seed=1)
    pruned_tflite = convert_to_tflite_bytes(pruned_model, pruned_config)

    assert pruned_config.conv_filters[0] < base_config.conv_filters[0]
    assert len(pruned_tflite) < len(base_tflite)
