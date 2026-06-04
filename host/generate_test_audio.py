import struct
from pathlib import Path

def main():
    root_dir = Path(__file__).resolve().parents[1]
    stop_dir = root_dir / "data" / "stop"
    output_path = root_dir / "firmware" / "src" / "test_audio.h"

    # Find the first WAV file in data/stop/
    wav_files = list(stop_dir.glob("*.wav"))
    if not wav_files:
        raise FileNotFoundError(f"No WAV files found in {stop_dir}")

    selected_wav = wav_files[0]
    print(f"Reading sample WAV file: {selected_wav}")

    with open(selected_wav, "rb") as f:
        wav_data = f.read()

    # Verify WAV header basics
    if len(wav_data) < 44 or wav_data[:4] != b"RIFF" or wav_data[8:12] != b"WAVE":
        raise ValueError(f"File {selected_wav} is not a valid RIFF/WAVE file")

    # The PCM data starts at byte 44
    pcm_bytes = wav_data[44:]
    # Each sample is 16-bit signed int (2 bytes)
    num_samples = len(pcm_bytes) // 2

    # Unpack PCM bytes into signed 16-bit integers
    samples = struct.unpack(f"<{num_samples}h", pcm_bytes[:num_samples * 2])

    print(f"Extracted {len(samples)} samples from WAV file.")

    # Write C header
    header_lines = [
        "#pragma once",
        "",
        "#include <cstddef>",
        "#include <cstdint>",
        "",
        "// Auto-generated test audio from speech command dataset",
        f"// Source file: {selected_wav.name}",
        f"const size_t G_TEST_AUDIO_LEN = {len(samples)};",
        "const int16_t G_TEST_AUDIO_DATA[] = {",
    ]

    # Format the samples nicely (12 per line)
    for i in range(0, len(samples), 12):
        chunk = samples[i : i + 12]
        header_lines.append("    " + ", ".join(map(str, chunk)) + ",")

    header_lines.append("};")
    header_lines.append("")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(header_lines), encoding="utf-8")
    print(f"Successfully generated {output_path}")

if __name__ == "__main__":
    main()
