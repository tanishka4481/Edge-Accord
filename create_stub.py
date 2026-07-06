import os
import struct
import wave
import random

# Target labels for the keyword spotting task
LABELS = ["yes", "no", "stop", "go", "unknown", "silence"]
DATA_DIR = "C:\\Users\\HP\\Documents\\edge-accord\\data"

def create_synthetic_wav(filepath, duration_sec=1.0, sample_rate=16000):
    """Generates a simple 1-second 16kHz PCM WAV file containing noise/tones."""
    num_samples = int(duration_sec * sample_rate)
    
    with wave.open(filepath, 'wb') as wav_file:
        wav_file.setnchannels(1)      # Mono
        wav_file.setsampwidth(2)     # 16-bit PCM
        wav_file.setframerate(sample_rate)
        
        # Write random 16-bit audio samples (-32768 to 32767)
        frames = bytearray()
        for _ in range(num_samples):
            sample = random.randint(-2000, 2000)
            frames.extend(struct.pack('<h', sample))
            
        wav_file.writeframes(frames)

def generate_stub_dataset(samples_per_label=5):
    """Creates directory structure and populates with stub wav files."""
    for label in LABELS:
        label_dir = os.path.join(DATA_DIR, label)
        os.makedirs(label_dir, exist_ok=True)
        
        for i in range(samples_per_label):
            wav_path = os.path.join(label_dir, f"stub_{i}.wav")
            if not os.path.exists(wav_path):
                create_synthetic_wav(wav_path)
                
    print(f"✅ Stub dataset generated in: {DATA_DIR}")

if __name__ == "__main__":
    generate_stub_dataset()