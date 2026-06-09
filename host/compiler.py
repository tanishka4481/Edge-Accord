from __future__ import annotations

import json
import math
import os
from dataclasses import replace
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .types import CompilationArtifact, ModelConfig, Proposal

DEFAULT_OUTPUT_ROOT = Path("build")
MODEL_HEADER_NAME = "model_data.h"
MODEL_TFLITE_NAME = "model.tflite"


class TensorFlowUnavailableError(RuntimeError):
    pass


def _load_tensorflow():
    try:
        import tensorflow as tf  # type: ignore
    except ImportError as exc:  # pragma: no cover - exercised in environments without TF
        raise TensorFlowUnavailableError(
            "TensorFlow is required for model compilation and training. Install requirements.txt first."
        ) from exc
    return tf


def _normalize_to_multiple(value: int, multiple: int = 4, minimum: int = 4) -> int:
    value = max(minimum, int(value))
    if value % multiple == 0:
        return value
    return max(minimum, multiple * math.ceil(value / multiple))


def build_base_config() -> ModelConfig:
    return ModelConfig()


def apply_proposal_to_config(config: ModelConfig, proposal: Proposal) -> ModelConfig:
    params = dict(proposal.params)
    if proposal.technique == "quantize_int8":
        return config

    if proposal.technique == "prune_filters":
        prune_ratio = float(params.get("prune_ratio", 0.25))
        prune_ratio = min(max(prune_ratio, 0.0), 0.95)
        scale = 1.0 - prune_ratio
        target_layer = params.get("layer_name")
        filters = list(config.conv_filters)
        if proposal.target == "whole_model" or not target_layer:
            filters = [_normalize_to_multiple(max(4, round(current * scale))) for current in filters]
        else:
            index_map = {"conv1": 0, "conv2": 1, "conv3": 2}
            if target_layer in index_map:
                index = index_map[target_layer]
                filters[index] = _normalize_to_multiple(max(4, round(filters[index] * scale)))
        dense_units = _normalize_to_multiple(max(8, round(config.dense_units * scale)))
        return replace(config, conv_filters=tuple(filters), dense_units=dense_units)

    if proposal.technique == "reduce_input_features":
        scale = float(params.get("scale", params.get("prune_ratio", 0.25)))
        scale = min(max(scale, 0.05), 0.95)
        time_steps = int(params.get("time_steps", round(config.time_steps * (1.0 - scale))))
        mfcc_bins = int(params.get("mfcc_bins", round(config.mfcc_bins * (1.0 - scale))))
        return replace(
            config,
            time_steps=max(8, time_steps),
            mfcc_bins=max(4, mfcc_bins),
        )

    return config


def make_model(config: ModelConfig, seed: int | None = None):
    tf = _load_tensorflow()
    if seed is None:
        seed = config.seed
    tf.keras.utils.set_random_seed(seed)

    inputs = tf.keras.Input(shape=(config.time_steps, config.mfcc_bins, 1), name="spectrogram")
    x = tf.keras.layers.Conv2D(config.conv_filters[0], (3, 3), padding="same", activation="relu", name="conv1")(inputs)
    x = tf.keras.layers.BatchNormalization(name="bn1")(x)
    x = tf.keras.layers.SeparableConv2D(config.conv_filters[1], (3, 3), padding="same", activation="relu", name="sepconv2")(x)
    x = tf.keras.layers.MaxPooling2D((2, 2), name="pool1")(x)
    x = tf.keras.layers.SeparableConv2D(config.conv_filters[2], (3, 3), padding="same", activation="relu", name="sepconv3")(x)
    x = tf.keras.layers.GlobalAveragePooling2D(name="gap")(x)
    x = tf.keras.layers.Dense(config.dense_units, activation="relu", name="dense1")(x)
    outputs = tf.keras.layers.Dense(len(config.classes), activation="softmax", name="class_logits")(x)
    return tf.keras.Model(inputs=inputs, outputs=outputs, name="edgeaccord_kws")


def _representative_dataset(config: ModelConfig, num_samples: int = 32) -> Iterable[list[np.ndarray]]:
    shape = (1, config.time_steps, config.mfcc_bins, 1)
    for index in range(num_samples):
        sample = np.zeros(shape, dtype=np.float32)
        sample.fill((index % 7) / 6.0 - 0.5)
        yield [sample]


def convert_to_tflite_bytes(model, config: ModelConfig, representative_data: Iterable[list[np.ndarray]] | None = None) -> bytes:
    tf = _load_tensorflow()
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    if representative_data is None:
        converter.representative_dataset = lambda: _representative_dataset(config)
    else:
        converter.representative_dataset = lambda: representative_data
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    return converter.convert()


def emit_model_header(tflite_bytes: bytes, header_path: Path) -> Path:
    header_path.parent.mkdir(parents=True, exist_ok=True)
    byte_lines: list[str] = []
    for offset in range(0, len(tflite_bytes), 12):
        chunk = tflite_bytes[offset : offset + 12]
        byte_lines.append("    " + ", ".join(f"0x{byte:02x}" for byte in chunk) + ",")

    header_content = [
        "#pragma once",
        "",
        "#include <cstddef>",
        "#include <cstdint>",
        "",
        "alignas(16) const unsigned char g_model[] = {",
        *byte_lines,
        "};",
        "",
        "const unsigned int g_model_len = sizeof(g_model);",
        "",
    ]
    header_path.write_text("\n".join(header_content), encoding="utf-8")
    return header_path


def compile_proposal(
    proposal: Proposal,
    base_config: ModelConfig | None = None,
    output_root: Path | str = DEFAULT_OUTPUT_ROOT,
    source_mode: str = "random_init",
    seed: int | None = None,
) -> CompilationArtifact:
    config = base_config or build_base_config()
    config = apply_proposal_to_config(config, proposal)

    model = make_model(config, seed=seed)
    tflite_bytes = convert_to_tflite_bytes(model, config)

    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    tflite_path = output_root / MODEL_TFLITE_NAME
    tflite_path.write_bytes(tflite_bytes)

    header_path = output_root / MODEL_HEADER_NAME
    emit_model_header(tflite_bytes, header_path)

    parameter_count = int(model.count_params())
    return CompilationArtifact(
        config=config,
        tflite_path=tflite_path,
        header_path=header_path,
        byte_size=len(tflite_bytes),
        parameter_count=parameter_count,
        source_mode=source_mode,  # type: ignore[arg-type]
    )


def summarise_artifact(artifact: CompilationArtifact) -> dict[str, Any]:
    return {
        "byte_size": artifact.byte_size,
        "parameter_count": artifact.parameter_count,
        "source_mode": artifact.source_mode,
        "config": artifact.config.to_dict(),
    }
