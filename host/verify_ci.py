import os
import subprocess
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]

def run_command(command, cwd, env=None):
    print(f"\nRunning command: {' '.join(command)}")
    print(f"Directory: {cwd}")
    
    process = subprocess.Popen(
        command,
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=env
    )
    
    while True:
        output = process.stdout.readline()
        if output == '' and process.poll() is not None:
            break
        if output:
            print(output.strip())
            
    rc = process.poll()
    if rc != 0:
        print(f"Command failed with exit code: {rc}")
        sys.exit(rc)
    else:
        print("Command succeeded.")

def is_docker_running():
    try:
        res = subprocess.run(
            ["docker", "info"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False
        )
        return res.returncode == 0
    except Exception:
        return False

def main():
    print("=== Start Local CI/CD Containerized Build Verification ===")
    
    docker_ok = is_docker_running()
    
    if docker_ok:
        print("Docker daemon detected! Running containerized verification.")
        
        # 1. Build the Docker image
        print("\n--- Step 1: Building Docker Image (edge-accord-ci) ---")
        build_cmd = ["docker", "build", "-t", "edge-accord-ci", "."]
        run_command(build_cmd, ROOT_DIR)
        
        # 2. Compile firmware inside the Docker container
        print("\n--- Step 2: Running PlatformIO compilation inside Docker ---")
        abs_root = str(ROOT_DIR.resolve())
        pio_cmd = [
            "docker", "run", "--rm",
            "-v", f"{abs_root}:/workspace",
            "-w", "/workspace",
            "edge-accord-ci",
            "pio", "run"
        ]
        run_command(pio_cmd, ROOT_DIR)
        
        # Check if firmware.bin exists after compilation
        firmware_bin = ROOT_DIR / "firmware" / ".pio" / "build" / "esp32dev" / "firmware.bin"
        print(f"\nVerifying firmware.bin generation at '{firmware_bin}'...")
        if firmware_bin.exists():
            print("Success: firmware.bin was generated correctly inside the container!")
        else:
            print("Error: firmware.bin not found!")
            sys.exit(1)
            
        # 3. Optional: Run simulation if token is available
        wokwi_token = os.getenv("WOKWI_CLI_TOKEN")
        if wokwi_token:
            print("\n--- Step 3: Running Wokwi CLI Simulation inside Docker ---")
            wokwi_cmd = [
                "docker", "run", "--rm",
                "-v", f"{abs_root}:/workspace",
                "-w", "/workspace",
                "-e", f"WOKWI_CLI_TOKEN={wokwi_token}",
                "edge-accord-ci",
                "wokwi-cli", "firmware", "--timeout", "10000", "--expect-text", "Best class:"
            ]
            run_command(wokwi_cmd, ROOT_DIR)
        else:
            print("\n--- Step 3: Skipping Wokwi CLI Simulation (WOKWI_CLI_TOKEN not set) ---")

        # 4. Run pytest inside the Docker container
        print("\n--- Step 4: Running Integration Tests inside Docker ---")
        pytest_cmd = [
            "docker", "run", "--rm",
            "-v", f"{abs_root}:/workspace",
            "-w", "/workspace",
            "edge-accord-ci",
            "pytest", "tests/"
        ]
        run_command(pytest_cmd, ROOT_DIR)
        
    else:
        print("WARNING: Docker daemon is not running or not installed.")
        print("Falling back to local host-side verification of PlatformIO and Pytest...")
        
        # 1. Run local PlatformIO build
        print("\n--- Step 1: Running PlatformIO compilation locally ---")
        pio_cmd = ["pio", "run"]
        run_command(pio_cmd, ROOT_DIR / "firmware")
        
        # Check if firmware.bin exists
        firmware_bin = ROOT_DIR / "firmware" / ".pio" / "build" / "esp32dev" / "firmware.bin"
        print(f"\nVerifying firmware.bin generation at '{firmware_bin}'...")
        if firmware_bin.exists():
            print("Success: firmware.bin exists!")
        else:
            print("Error: firmware.bin not found!")
            sys.exit(1)
            
        # 2. Run local tests
        print("\n--- Step 2: Running Pytest locally ---")
        pytest_cmd = ["pytest"]
        run_command(pytest_cmd, ROOT_DIR)
        
    print("\n=== Local CI/CD Build Verification SUCCESS! ===")

if __name__ == "__main__":
    main()
