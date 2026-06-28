"""
host/train.py — Accuracy-focused retraining script for EdgeAccord DS-CNN KWS model.
Loads real speech commands from data/, applies audio augmentations, computes MFCC features,
trains DS-CNN network for 25 epochs, quantizes to int8 TFLite, and emits model_data.h.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

# Ensure host directory in path
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from host.compiler import (
    build_base_config,
    convert_to_tflite_bytes,
    emit_model_header,
    make_model,
)
from host.train_model import waveform_to_mfcc
from host.types import ModelConfig


def load_raw_audio_dataset(data_dir: Path, target_classes: List[str]) -> Tuple[List[np.ndarray], List[int]]:
    """
    Scans data/ and loads all .wav files as 16kHz float32 arrays normalized to [-1.0, 1.0].
    Pads or truncates to exactly 16000 samples (1.0 second).
    """
    import tensorflow as tf

    samples: List[np.ndarray] = []
    labels: List[int] = []

    for class_idx, class_name in enumerate(target_classes):
        class_folder = data_dir / class_name
        if not class_folder.exists():
            continue

        wav_paths = list(class_folder.glob("*.wav"))
        print(f"Loading {len(wav_paths):3d} audio samples for class '{class_name}' (label {class_idx})")

        for wav_path in wav_paths:
            try:
                audio_binary = tf.io.read_file(str(wav_path))
                waveform, _ = tf.audio.decode_wav(audio_binary, desired_channels=1)
                waveform = tf.squeeze(waveform, axis=-1)  # (samples,)
                
                # Ensure 16000 length
                length = tf.shape(waveform)[0]
                if length < 16000:
                    padding = tf.zeros([16000 - length], dtype=tf.float32)
                    waveform = tf.concat([waveform, padding], axis=0)
                else:
                    waveform = waveform[:16000]

                samples.append(waveform.numpy())
                labels.append(class_idx)
            except Exception as e:
                print(f"  [!] Skipping corrupted {wav_path.name}: {e}")

    return samples, labels


def augment_waveform(waveform: np.ndarray) -> np.ndarray:
    """
    Applies audio data augmentation:
      - Random time shift (+/- 1600 samples = +/- 100ms)
      - Random volume scaling (0.7 to 1.3)
      - Additive Gaussian noise (SNR control)
    """
    # 1. Random time shift
    shift = random.randint(-1600, 1600)
    if shift > 0:
        waveform = np.pad(waveform, (shift, 0), mode='constant')[:16000]
    elif shift < 0:
        waveform = np.pad(waveform, (0, -shift), mode='constant')[-shift:]
        if len(waveform) < 16000:
            waveform = np.pad(waveform, (0, 16000 - len(waveform)), mode='constant')

    # 2. Random volume scaling
    gain = random.uniform(0.75, 1.25)
    waveform = waveform * gain

    # 3. Additive subtle background noise
    noise_level = random.uniform(0.001, 0.008)
    noise = np.random.normal(0, noise_level, waveform.shape)
    waveform = waveform + noise

    return np.clip(waveform, -1.0, 1.0).astype(np.float32)


def generate_augmented_dataset(
    raw_samples: List[np.ndarray],
    raw_labels: List[int],
    config: ModelConfig,
    augment_factor: int = 8,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Expands dataset with augmentations and transforms all waveforms into MFCC spectrograms.
    """
    import tensorflow as tf

    all_waveforms = []
    all_labels = []

    for wave, label in zip(raw_samples, raw_labels):
        # Keep original
        all_waveforms.append(wave)
        all_labels.append(label)

        # Generate augmented variations
        for _ in range(augment_factor):
            aug_wave = augment_waveform(wave)
            all_waveforms.append(aug_wave)
            all_labels.append(label)

    print(f"\nTotal training/validation dataset size: {len(all_waveforms)} samples")

    # Batch process waveforms to MFCC spectrograms
    waveforms_tensor = tf.constant(np.array(all_waveforms, dtype=np.float32))
    waveforms_expanded = tf.expand_dims(waveforms_tensor, axis=-1)  # (N, 16000, 1)

    mfccs = waveform_to_mfcc(waveforms_expanded, config)  # (N, time_steps, mfcc_bins, 1)
    one_hot_labels = tf.keras.utils.to_categorical(all_labels, num_classes=len(config.classes))

    return mfccs.numpy(), one_hot_labels


def train_and_export(
    data_dir: Path = Path("data"),
    epochs: int = 35,
    learning_rate: float = 0.002,
    batch_size: int = 32,
    output_dir: Path = Path("build/final"),
) -> Dict[str, float]:
    """
    Retrains the DS-CNN keyword spotting model on real audio files,
    quantizes to int8 TFLite, saves build/final/model.tflite, and emits firmware/src/model_data.h.
    """
    import tensorflow as tf

    config = build_base_config()
    print("=== Starting EdgeAccord KWS Retraining Loop ===")
    print(f"Target classes: {config.classes}")
    print(f"Spectrogram shape: ({config.time_steps}, {config.mfcc_bins}, 1)")

    # 1. Load real audio samples
    raw_samples, raw_labels = load_raw_audio_dataset(data_dir, list(config.classes))
    if not raw_samples:
        raise ValueError(f"No audio files found in {data_dir}!")

    # 2. Augment and compute MFCC features
    X, Y = generate_augmented_dataset(raw_samples, raw_labels, config, augment_factor=12)

    # Shuffle dataset
    indices = np.arange(len(X))
    np.random.seed(config.seed)
    np.random.shuffle(indices)
    X = X[indices]
    Y = Y[indices]

    # Split 85% train / 15% val
    val_split = int(len(X) * 0.85)
    X_train, X_val = X[:val_split], X[val_split:]
    Y_train, Y_val = Y[:val_split], Y[val_split:]

    # 3. Create DS-CNN Model Architecture
    model = make_model(config, seed=config.seed)
    
    lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=learning_rate,
        decay_steps=epochs * (len(X_train) // batch_size),
        alpha=0.1
    )
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    print(f"\n--- Training DS-CNN for {epochs} Epochs ---")
    history = model.fit(
        X_train,
        Y_train,
        validation_data=(X_val, Y_val),
        epochs=epochs,
        batch_size=batch_size,
        verbose=1,
    )

    final_train_acc = float(history.history["accuracy"][-1])
    final_val_acc = float(history.history["val_accuracy"][-1])
    print(f"\nFinal Training Accuracy:   {final_train_acc * 100:.2f}%")
    print(f"Final Validation Accuracy: {final_val_acc * 100:.2f}%")

    # 4. Quantize to full int8 using real representative MFCC data
    print("\n--- Quantizing Model to int8 TFLite ---")
    def representative_dataset():
        for i in range(min(100, len(X_train))):
            yield [X_train[i:i+1].astype(np.float32)]

    tflite_bytes = convert_to_tflite_bytes(model, config, representative_data=representative_dataset())
    print(f"Quantized TFLite Model Size: {len(tflite_bytes)} bytes ({len(tflite_bytes) / 1024:.2f} KB)")

    # 5. Export build artifacts
    output_dir.mkdir(parents=True, exist_ok=True)
    tflite_path = output_dir / "model.tflite"
    tflite_path.write_bytes(tflite_bytes)
    print(f"[+] Saved: {tflite_path}")

    # Emit model_data.h to build/final and firmware/src/
    header_path_build = output_dir / "model_data.h"
    emit_model_header(tflite_bytes, header_path_build)

    header_path_firmware = Path("firmware/src/model_data.h")
    emit_model_header(tflite_bytes, header_path_firmware)
    print(f"[+] Emitted C header: {header_path_firmware}")

    return {
        "train_accuracy": final_train_acc,
        "val_accuracy": final_val_acc,
        "model_size_bytes": len(tflite_bytes),
    }


def main():
    parser = argparse.ArgumentParser(description="Retrain EdgeAccord KWS model on real audio samples.")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--lr", type=float, default=0.001)
    args = parser.parse_args()

    train_and_export(
        data_dir=args.data_dir,
        epochs=args.epochs,
        learning_rate=args.lr,
    )


if __name__ == "__main__":
    main()
