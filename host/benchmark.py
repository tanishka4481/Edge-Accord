import argparse
import datetime
import json
import time
import sys
from pathlib import Path
import numpy as np

# Add root folder to sys.path to enable imports
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from host.compiler import build_base_config
from host.train_model import waveform_to_mfcc

def get_latest_memory_usage(log_path: Path):
    ram_used = 0
    flash_used = 0
    if log_path.exists():
        with open(log_path, "r", encoding="utf-8") as f:
            for line in reversed(f.readlines()):
                if not line.strip():
                    continue
                try:
                    data = json.loads(line)
                    # Support flash_device log format
                    if "ram_used_bytes" in data and "flash_used_bytes" in data:
                        ram_used = data["ram_used_bytes"]
                        flash_used = data["flash_used_bytes"]
                        break
                    # Support negotiator log format
                    elif "constraint" in data and data["constraint"] is not None:
                        c = data["constraint"]
                        ram_used = c.get("ram_used_bytes", 0)
                        flash_used = c.get("flash_used_bytes", 0)
                        if ram_used and flash_used:
                            break
                except Exception:
                    pass
    return ram_used, flash_used

def load_and_preprocess_wav(wav_path: Path, config, tf):
    # Load WAV file
    file_contents = tf.io.read_file(str(wav_path))
    audio, sample_rate = tf.audio.decode_wav(file_contents, desired_channels=1)
    
    # Force length to 16000
    if len(audio) < 16000:
        padding = 16000 - len(audio)
        audio = tf.pad(audio, [[0, padding], [0, 0]])
    else:
        audio = audio[:16000]
        
    waveform = tf.expand_dims(audio, axis=0) # shape (1, 16000, 1)
    mfcc = waveform_to_mfcc(waveform, config) # shape (1, time_steps, mfcc_bins, 1)
    return mfcc

def run_local_benchmark(data_dir: Path, tflite_path: Path, log_path: Path, results_path: Path):
    print("=== Starting Local TFLite Benchmark ===")
    print(f"Loading TFLite model from: {tflite_path}")
    
    try:
        import tensorflow as tf
    except ImportError:
        print("Error: TensorFlow is required for running the local benchmark.")
        sys.exit(1)
        
    config = build_base_config()
    
    # Find all WAV files in data/
    wav_files = list(data_dir.glob("**/*.wav"))
    if not wav_files:
        print(f"Error: No WAV files found in data directory: {data_dir}")
        sys.exit(1)
    
    print(f"Found {len(wav_files)} WAV files to evaluate.")
    
    # Initialize TFLite interpreter
    interpreter = tf.lite.Interpreter(model_path=str(tflite_path))
    interpreter.allocate_tensors()
    
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]
    
    is_input_quantized = (input_details['dtype'] == np.int8)
    if is_input_quantized:
        scale, zero_point = input_details['quantization']
        print(f"Model expects quantized int8 input (scale: {scale}, zero_point: {zero_point})")
    else:
        print("Model expects float32 input")
        
    is_output_quantized = (output_details['dtype'] == np.int8)
    if is_output_quantized:
        o_scale, o_zero_point = output_details['quantization']
        print(f"Model produces quantized int8 output (scale: {o_scale}, zero_point: {o_zero_point})")
    else:
        print("Model produces float32 output")
        
    latencies = []
    correct_count = 0
    total_count = 0
    predictions_per_class = {}
    
    # Initialize predictions dictionary for classes
    for c_name in config.classes:
        predictions_per_class[c_name] = {"total": 0, "correct": 0}
        
    for wav_path in wav_files:
        target_label = wav_path.parent.name
        if target_label not in config.classes:
            # Skip files outside yes/no/stop/go/unknown/silence
            continue
            
        # Load and preprocess
        mfcc = load_and_preprocess_wav(wav_path, config, tf)
        
        # Quantize if necessary
        if is_input_quantized:
            input_data = (mfcc.numpy() / scale) + zero_point
            input_data = np.round(input_data).clip(-128, 127).astype(np.int8)
        else:
            input_data = mfcc.numpy().astype(np.float32)
            
        interpreter.set_tensor(input_details['index'], input_data)
        
        # Invoke TFLite model & measure latency
        t0 = time.perf_counter()
        interpreter.invoke()
        t1 = time.perf_counter()
        
        latency_ms = (t1 - t0) * 1000.0
        latencies.append(latency_ms)
        
        # Process output
        output_data = interpreter.get_tensor(output_details['index'])
        if is_output_quantized:
            probs = (output_data.astype(np.float32) - o_zero_point) * o_scale
        else:
            probs = output_data.astype(np.float32)
            
        pred_idx = np.argmax(probs[0])
        pred_label = config.classes[pred_idx]
        
        is_correct = (pred_label == target_label)
        if is_correct:
            correct_count += 1
        total_count += 1
        
        predictions_per_class[target_label]["total"] += 1
        if is_correct:
            predictions_per_class[target_label]["correct"] += 1
            
        print(f"Sample: {wav_path.name} | Target: {target_label} | Pred: {pred_label} | Latency: {latency_ms:.2f} ms | Correct: {is_correct}")

    # Compute statistics
    avg_latency = sum(latencies) / len(latencies) if latencies else 0.0
    min_latency = min(latencies) if latencies else 0.0
    max_latency = max(latencies) if latencies else 0.0
    accuracy = correct_count / total_count if total_count > 0 else 0.0
    
    # Read static memory usage from accord_log.jsonl
    ram_used, flash_used = get_latest_memory_usage(log_path)
    
    tensor_arena_bytes = 80 * 1024 # 81920 bytes
    bss_data_bytes = max(0, ram_used - tensor_arena_bytes) if ram_used else 0
    
    results = {
        "timestamp": datetime.datetime.now().isoformat(),
        "latency_ms": {
            "avg": round(avg_latency, 2),
            "min": round(min_latency, 2),
            "max": round(max_latency, 2),
            "samples": len(latencies)
        },
        "static_memory_profile": {
            "tensor_arena_bytes": tensor_arena_bytes,
            "bss_data_bytes": bss_data_bytes,
            "total_static_ram_bytes": ram_used,
            "flash_used_bytes": flash_used
        },
        "accuracy_evaluation": {
            "total_samples": total_count,
            "correct_predictions": correct_count,
            "accuracy": round(accuracy, 4),
            "predictions_per_class": predictions_per_class
        }
    }
    
    # Save to benchmark_results.json
    results_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nStructured results exported to: {results_path}")
    print("=" * 40)
    print("BENCHMARK SUMMARY")
    print("=" * 40)
    print(f"Avg Latency:  {avg_latency:.2f} ms")
    print(f"Accuracy:     {accuracy * 100:.2f}% ({correct_count}/{total_count})")
    print(f"Static RAM:   {ram_used} bytes (Arena: {tensor_arena_bytes} B, BSS/Data: {bss_data_bytes} B)")
    print(f"Flash Size:   {flash_used} bytes")
    print("=" * 40)

def main():
    parser = argparse.ArgumentParser(description="Inference latency and accuracy benchmark utility.")
    parser.add_argument("--port", type=str, default=None, help="Serial port of ESP32 (if running on-device)")
    parser.add_argument("--baud", type=int, default=115200, help="Baud rate (default: 115200)")
    parser.add_argument("--count", type=int, default=10, help="Number of samples to collect (for serial)")
    args = parser.parse_args()

    data_dir = ROOT_DIR / "data"
    log_path = ROOT_DIR / "accord_log.jsonl"
    results_path = ROOT_DIR / "benchmark_results.json"
    
    # Determine model path
    tflite_path = ROOT_DIR / "build" / "final" / "model.tflite"
    if not tflite_path.exists():
        candidates = list(ROOT_DIR.glob("build/**/*.tflite"))
        if candidates:
            tflite_path = candidates[0]
        else:
            print("Error: Could not find any .tflite model under build/ directory.")
            sys.exit(1)

    # If serial port is specified, run on-device serial monitor benchmark
    if args.port:
        try:
            import serial  # type: ignore
        except ImportError:
            print("Error: 'pyserial' package is not installed. Please install it using: pip install pyserial")
            sys.exit(1)

        print(f"Connecting to ESP32 on port {args.port} at {args.baud} baud...")
        try:
            ser = serial.Serial(args.port, args.baud, timeout=2.0)
        except Exception as e:
            print(f"Error opening serial port: {e}")
            sys.exit(1)

        time.sleep(1.0)
        ser.reset_input_buffer()
        print(f"Collecting {args.count} inference latency samples...")

        latencies = []
        import re
        latency_pattern = re.compile(r"Inference completed in (\d+)\s*ms", re.IGNORECASE)

        try:
            while len(latencies) < args.count:
                line = ser.readline().decode('utf-8', errors='ignore').strip()
                if line:
                    print(f"[Serial]: {line}")
                    match = latency_pattern.search(line)
                    if match:
                        latency = int(match.group(1))
                        latencies.append(latency)
                        print(f"--> Sample {len(latencies)}/{args.count}: {latency} ms")
        except KeyboardInterrupt:
            print("\nBenchmark interrupted by user.")
        finally:
            ser.close()

        if latencies:
            avg_lat = sum(latencies) / len(latencies)
            min_lat = min(latencies)
            max_lat = max(latencies)
            print("\n" + "=" * 40)
            print("ON-DEVICE BENCHMARK RESULTS")
            print("=" * 40)
            print(f"Samples Collected: {len(latencies)}")
            print(f"Average Latency:   {avg_lat:.2f} ms")
            print("=" * 40)
        else:
            print("\nNo latency samples collected.")
    else:
        # Run local TF Lite simulation benchmark
        run_local_benchmark(data_dir, tflite_path, log_path, results_path)

if __name__ == "__main__":
    main()
