from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import numpy as np

from .compiler import build_base_config, make_model
from .types import ModelConfig


class TensorFlowUnavailableError(RuntimeError):
    pass


def _load_tensorflow():
    try:
        import tensorflow as tf  # type: ignore
    except ImportError as exc:  # pragma: no cover
        raise TensorFlowUnavailableError(
            "TensorFlow is required for training. Install requirements.txt first."
        ) from exc
    return tf

def make_toy_dataset(config: ModelConfig, samples_per_class: int = 64):
    tf = _load_tensorflow()
    rng = np.random.default_rng(config.seed)
    total_samples = samples_per_class * len(config.classes)
    x = rng.normal(size=(total_samples, config.time_steps, config.mfcc_bins, 1)).astype(np.float32)
    labels = np.repeat(np.arange(len(config.classes), dtype=np.int32), samples_per_class)
    y = tf.keras.utils.to_categorical(labels, num_classes=len(config.classes))
    dataset = tf.data.Dataset.from_tensor_slices((x, y))
    return dataset.shuffle(total_samples, seed=config.seed).batch(32).prefetch(tf.data.AUTOTUNE)


# def load_dataset_from_directory(data_dir: Path, config: ModelConfig, validation_split: float = 0.2):
#     tf = _load_tensorflow()
#     class_names = list(config.classes)
#     if not data_dir.exists():
#         return None, None

#     audio_ds = tf.keras.utils.audio_dataset_from_directory(
#         data_dir,
#         batch_size=32,
#         output_sequence_length=16000,
#         validation_split=validation_split,
#         subset="both",
#         seed=config.seed,
#         class_names=class_names,
#     )
#     train_ds, val_ds = audio_ds

#     def transform(waveform, label):
#         mfcc = waveform_to_mfcc(waveform, config)
#         label = tf.one_hot(tf.cast(label, tf.int32), depth=len(config.classes))
#         return mfcc, label

#     train_ds = train_ds.map(transform, num_parallel_calls=tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE)
#     val_ds = val_ds.map(transform, num_parallel_calls=tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE)
#     return train_ds, val_ds

def waveform_to_mfcc(waveform, config: ModelConfig):
    tf = _load_tensorflow()
    waveform = tf.cast(waveform, tf.float32)

    # 1. Squeeze the channel dimension: (batch, 16000, 1) -> (batch, 16000)
    waveform = tf.squeeze(waveform, axis=-1)

    # 2. STFT over time axis -> (batch, time_frames, fft_bins)
    stft = tf.signal.stft(waveform, frame_length=640, frame_step=320, fft_length=1024)
    spectrogram = tf.abs(stft)

    # 3. Project Frequency Bins to Mel Scale -> (batch, time_frames, mel_bins)
    num_spectrogram_bins = stft.shape[-1]
    mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(
        num_mel_bins=config.mfcc_bins,
        num_spectrogram_bins=num_spectrogram_bins,
        sample_rate=16000,
        lower_edge_hertz=80.0,
        upper_edge_hertz=7600.0,
    )
    mel_spectrogram = tf.matmul(spectrogram, mel_weight_matrix)
    log_mel = tf.math.log(mel_spectrogram + 1e-6)

    # 4. Extract MFCC -> (batch, time_frames, mfcc_bins)
    mfcc = tf.signal.mfccs_from_log_mel_spectrograms(log_mel)[..., : config.mfcc_bins]

    # 5. Add channel dim for Conv2D -> (batch, time_frames, mfcc_bins, 1)
    mfcc = tf.expand_dims(mfcc, axis=-1)

    # 6. Resize to (time_steps, mfcc_bins) -> (batch, config.time_steps, config.mfcc_bins, 1)
    resized_mfcc = tf.image.resize(mfcc, [config.time_steps, config.mfcc_bins])

    return tf.cast(resized_mfcc, tf.float32)


def load_dataset_from_directory(data_dir: Path, config: ModelConfig, validation_split: float = 0.2):
    tf = _load_tensorflow()
    class_names = list(config.classes)
    if not data_dir.exists():
        return None, None

    audio_ds = tf.keras.utils.audio_dataset_from_directory(
        data_dir,
        batch_size=32,
        output_sequence_length=16000,
        validation_split=validation_split,
        subset="both",
        seed=config.seed,
        class_names=class_names,
    )
    train_ds, val_ds = audio_ds

    def transform(waveform, label):
        mfcc = waveform_to_mfcc(waveform, config)
        label = tf.one_hot(tf.cast(label, tf.int32), depth=len(config.classes))
        return mfcc, label

    train_ds = train_ds.map(transform, num_parallel_calls=tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE)
    val_ds = val_ds.map(transform, num_parallel_calls=tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE)
    return train_ds, val_ds

def train_model(
    data_dir: Path | None = None,
    output_dir: Path | str = Path("build") / "trained",
    epochs: int = 10,
    use_toy_data: bool = False,
) -> dict[str, float | str]:
    tf = _load_tensorflow()
    config = build_base_config()
    model = make_model(config, seed=config.seed)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=config.learning_rate),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    train_ds = val_ds = None
    if not use_toy_data and data_dir is not None:
        train_ds, val_ds = load_dataset_from_directory(Path(data_dir), config)

    if train_ds is None or val_ds is None:
        dataset = make_toy_dataset(config)
        train_count = int(dataset.cardinality().numpy())
        val_count = max(1, train_count // 5)
        val_ds = dataset.take(val_count)
        train_ds = dataset.skip(val_count)

    history = model.fit(train_ds, validation_data=val_ds, epochs=epochs, verbose=2)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "trained_model.keras"
    model.save(model_path)

    final_metrics = {
        "loss": float(history.history["loss"][-1]),
        "accuracy": float(history.history.get("accuracy", [0.0])[-1]),
        "val_loss": float(history.history.get("val_loss", [0.0])[-1]),
        "val_accuracy": float(history.history.get("val_accuracy", [0.0])[-1]),
        "model_path": str(model_path),
    }
    (output_dir / "training_report.json").write_text(json.dumps(final_metrics, indent=2), encoding="utf-8")
    return final_metrics


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train the EdgeAccord keyword spotting model.")
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("build") / "trained")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--toy-data", action="store_true")
    return parser


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()
    train_model(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        epochs=args.epochs,
        use_toy_data=args.toy_data,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
