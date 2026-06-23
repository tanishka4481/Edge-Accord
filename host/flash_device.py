import argparse
import datetime
import json
import subprocess
import sys
from pathlib import Path

# Add root folder to sys.path to enable imports
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from host.constraint_engine import parse_platformio_output

RAM_LIMIT_BYTES = 320 * 1024         # 327680 bytes (320KB)
FLASH_LIMIT_BYTES = 1363148         # 1.3MB (1.3 * 1024 * 1024 = 1363148 bytes)

def run_simulation(firmware_dir: Path, log_path: Path):
    print("=== Starting Simulated Flashing (Phase 4 Build Verification) ===")
    
    # Run pio run
    print("Running PlatformIO build: pio run")
    build_cmd = ["pio", "run"]
    completed = subprocess.run(
        build_cmd,
        cwd=firmware_dir,
        capture_output=True,
        text=True,
        check=False
    )
    
    combined_output = (completed.stdout or "") + "\n" + (completed.stderr or "")
    print(combined_output)
    
    if completed.returncode != 0:
        print(f"Error: Compilation failed with return code {completed.returncode}!")
        sys.exit(completed.returncode)
        
    # Parse RAM and Flash usage
    try:
        summary = parse_platformio_output(combined_output)
        ram_used = summary["ram_used_bytes"]
        flash_used = summary["flash_used_bytes"]
    except Exception as e:
        print(f"Error: Failed to parse PlatformIO output: {e}")
        sys.exit(1)
        
    print(f"Parsed RAM used:   {ram_used} bytes")
    print(f"Parsed Flash used: {flash_used} bytes")
    
    # Check firmware.bin existence
    firmware_bin = firmware_dir / ".pio" / "build" / "esp32dev" / "firmware.bin"
    bin_exists = firmware_bin.exists()
    print(f"Checking if firmware.bin exists at '{firmware_bin}': {bin_exists}")
    assert bin_exists, f"Error: firmware.bin was not generated at {firmware_bin}!"
    
    # Assert constraints
    print(f"Asserting RAM footprint < 320KB (Limit: {RAM_LIMIT_BYTES} B): ", end="")
    if ram_used < RAM_LIMIT_BYTES:
        print("PASS")
    else:
        print("FAIL")
        raise AssertionError(f"RAM usage ({ram_used} B) exceeds the 320KB limit ({RAM_LIMIT_BYTES} B)!")
        
    print(f"Asserting Flash footprint < 1.3MB (Limit: {FLASH_LIMIT_BYTES} B): ", end="")
    if flash_used < FLASH_LIMIT_BYTES:
        print("PASS")
    else:
        print("FAIL")
        raise AssertionError(f"Flash usage ({flash_used} B) exceeds the 1.3MB limit ({FLASH_LIMIT_BYTES} B)!")
        
    # Create build record
    record = {
        "timestamp": datetime.datetime.now().isoformat(),
        "phase": "phase_4_simulated_flash",
        "success": True,
        "ram_used_bytes": ram_used,
        "flash_used_bytes": flash_used,
        "firmware_bin_exists": bin_exists,
        "message": "Simulated flashing successful. RAM and Flash within limits."
    }
    
    # Log record to accord_log.jsonl
    print(f"Appending build status record to log file: {log_path}")
    with log_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")
        
    print("=== Simulated Flashing Success ===")

def main():
    parser = argparse.ArgumentParser(description="Flash ESP32 device or perform simulated compilation/verification.")
    parser.add_argument("--simulation", action="store_true", help="Run simulated flashing & compile verification only")
    args = parser.parse_args()
    
    firmware_dir = ROOT_DIR / "firmware"
    log_path = ROOT_DIR / "accord_log.jsonl"
    
    if args.simulation:
        run_simulation(firmware_dir, log_path)
    else:
        print("=== Step 1: Uploading firmware to ESP32 device ===")
        upload_cmd = ["pio", "run", "-t", "upload"]
        
        res = subprocess.run(upload_cmd, cwd=firmware_dir)
        if res.returncode != 0:
            print("Error: Firmware upload failed!")
            sys.exit(res.returncode)
            
        print("\n=== Step 2: Launching PlatformIO serial monitor ===")
        monitor_cmd = ["pio", "device", "monitor"]
        try:
            subprocess.run(monitor_cmd, cwd=firmware_dir)
        except KeyboardInterrupt:
            print("\nSerial monitor stopped.")

if __name__ == "__main__":
    main()
