import os
import urllib.request
import tarfile
from pathlib import Path

DATA_DIR = Path(r"C:\Users\HP\Documents\edge-accord\data")
DATASET_URL = "http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz"
TAR_PATH = DATA_DIR / "speech_commands.tar.gz"

DATA_DIR.mkdir(parents=True, exist_ok=True)

target_categories = ["yes", "no", "stop", "go", "silence", "unknown"]
counts = {cat: 0 for cat in target_categories}

print("1. Downloading Google Speech Commands subset archive...")
headers = {'User-Agent': 'Mozilla/5.0'}
req = urllib.request.Request(DATASET_URL, headers=headers)

with urllib.request.urlopen(req) as response, open(TAR_PATH, 'wb') as out_file:
    chunk_size = 1024 * 1024
    while True:
        chunk = response.read(chunk_size)
        if not chunk:
            break
        out_file.write(chunk)

print("2. Extracting target audio files into subdirectories...")
with tarfile.open(TAR_PATH, "r:gz") as tar:
    for member in tar.getmembers():
        if not member.isfile() or not member.name.endswith(".wav"):
            continue
        
        # Determine category from path
        parts = Path(member.name).parts
        category = parts[0] if len(parts) > 1 else "unknown"
        
        if category in target_categories and counts[category] < 5:
            target_folder = DATA_DIR / category
            target_folder.mkdir(parents=True, exist_ok=True)
            
            # Extract single file directly into target folder
            file_obj = tar.extractfile(member)
            if file_obj:
                out_path = target_folder / Path(member.name).name
                with open(out_path, "wb") as f:
                    f.write(file_obj.read())
                counts[category] += 1
                print(f"   [+] Saved {category}/{out_path.name}")

print("3. Cleaning up archive...")
if TAR_PATH.exists():
    TAR_PATH.unlink()

print("\n--- Summary of Extracted Audio Files ---")
for cat, count in counts.items():
    print(f"  {cat:<10}: {count} files")