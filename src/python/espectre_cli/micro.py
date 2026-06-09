"""Micro-ESPectre device workflow commands."""

from __future__ import annotations

import hashlib
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import List, Tuple

from .common import (
    FIRMWARE_CACHE_DIR,
    FIRMWARE_HASHES,
    FIRMWARE_NAME_PREFIX,
    FIRMWARE_RELEASE_URL,
    MICRO_CHIP_CHOICES,
    Fore,
    PYTHON_SRC_DIR,
    Style,
    detect_chip_type,
    get_serial_port,
    prompt_chip_type,
)


def _calculate_sha256(filepath: Path) -> str:
    sha256 = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


def _require_mpremote() -> None:
    try:
        subprocess.run(["mpremote", "--version"], capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print(f"{Fore.RED}❌ mpremote not found. Install it with:{Style.RESET_ALL}")
        print("   pip install mpremote")
        raise SystemExit(1)


def _reset_device(port: str) -> None:
    time.sleep(0.5)
    try:
        subprocess.run(
            ["mpremote", "connect", port, "exec", "import machine; machine.reset()"],
            timeout=5,
            capture_output=True,
        )
        print(f"{Fore.GREEN}ESP32 reset completed{Style.RESET_ALL}")
    except Exception:
        print(f"{Fore.GREEN}ESP32 reset completed{Style.RESET_ALL}")


def download_firmware(chip: str, firmware_dir: Path = FIRMWARE_CACHE_DIR) -> Path:
    """Download firmware from GitHub releases if not already cached or verified."""
    chip_suffix_map = {
        "esp32": "",
        "c3": "C3",
        "s3": "S3",
        "c5": "C5",
        "c6": "C6",
    }
    chip_suffix = chip_suffix_map.get(chip, chip.upper())
    firmware_name = f"{FIRMWARE_NAME_PREFIX}{chip_suffix}.bin" if chip_suffix else "ESP32_CSI.bin"
    firmware_path = firmware_dir / firmware_name
    expected_hash = FIRMWARE_HASHES.get(firmware_name)

    if firmware_path.exists():
        if expected_hash:
            current_hash = _calculate_sha256(firmware_path)
            if current_hash == expected_hash:
                print(f"{Fore.GREEN}✅ Using cached firmware: {firmware_name} (hash verified){Style.RESET_ALL}")
                return firmware_path
            print(f"{Fore.YELLOW}⚠️  Cached firmware hash mismatch, re-downloading...{Style.RESET_ALL}")
            print(f"   Expected: {expected_hash[:16]}...")
            print(f"   Found:    {current_hash[:16]}...")
            firmware_path.unlink()
        else:
            print(f"{Fore.GREEN}✅ Using cached firmware: {firmware_name}{Style.RESET_ALL}")
            return firmware_path

    firmware_dir.mkdir(parents=True, exist_ok=True)
    url = f"{FIRMWARE_RELEASE_URL}/{firmware_name}"
    print(f"{Fore.YELLOW}📥 Downloading firmware from GitHub...{Style.RESET_ALL}")
    print(f"{Fore.CYAN}   URL: {url}{Style.RESET_ALL}")

    try:
        with urllib.request.urlopen(url, timeout=60) as response:
            total_size = int(response.headers.get("content-length", 0))
            downloaded = 0
            chunk_size = 8192
            with open(firmware_path, "wb") as f:
                while True:
                    chunk = response.read(chunk_size)
                    if not chunk:
                        break
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_size:
                        progress = (downloaded * 100) // total_size
                        print(
                            f"\r{Fore.YELLOW}   Progress: {progress}% ({downloaded // 1024} KB){Style.RESET_ALL}",
                            end="",
                            flush=True,
                        )
            print()

        if expected_hash:
            downloaded_hash = _calculate_sha256(firmware_path)
            if downloaded_hash != expected_hash:
                print(f"{Fore.RED}❌ Downloaded firmware hash mismatch!{Style.RESET_ALL}")
                print(f"   Expected: {expected_hash[:16]}...")
                print(f"   Got:      {downloaded_hash[:16]}...")
                firmware_path.unlink()
                raise SystemExit(1)
            print(f"{Fore.GREEN}✅ Firmware downloaded and verified: {firmware_name}{Style.RESET_ALL}")
        else:
            print(f"{Fore.GREEN}✅ Firmware downloaded: {firmware_name}{Style.RESET_ALL}")
        return firmware_path
    except urllib.error.URLError as e:
        print(f"{Fore.RED}❌ Failed to download firmware: {e}{Style.RESET_ALL}")
        print(f"{Fore.YELLOW}   Check your internet connection or download manually from:{Style.RESET_ALL}")
        print(f"{Fore.CYAN}   https://github.com/francescopace/micropython-esp32-csi/releases{Style.RESET_ALL}")
        raise SystemExit(1)


def flash_firmware(args) -> None:
    """Flash MicroPython firmware to ESP32 using esptool."""
    try:
        import esptool
    except ImportError:
        print(f"{Fore.RED}❌ esptool not found. Install it with:{Style.RESET_ALL}")
        print("   pip install esptool")
        raise SystemExit(1)

    port = get_serial_port(args.port)
    chip = args.chip
    if not chip:
        chip = detect_chip_type(port)
        if not chip:
            print(f"\n{Fore.YELLOW}💡 Tip: If the chip is not responding, try:{Style.RESET_ALL}")
            print("   1. Hold the BOOT button on your ESP32")
            print("   2. Press and release the RESET button (while holding BOOT)")
            print("   3. Release the BOOT button")
            print("   4. Try flashing again")
            print()
            chip = prompt_chip_type()
            if not chip:
                raise SystemExit(1)

    if args.firmware:
        firmware_path = Path(args.firmware)
        if not firmware_path.exists():
            print(f"{Fore.RED}❌ Firmware not found: {firmware_path}{Style.RESET_ALL}")
            raise SystemExit(1)
    else:
        firmware_path = download_firmware(chip, FIRMWARE_CACHE_DIR)

    print(f"{Fore.MAGENTA}╔═══════════════════════════════════════════════════════════╗{Style.RESET_ALL}")
    print(f"{Fore.MAGENTA}║          μESPectre - Flashing MicroPython Firmware        ║{Style.RESET_ALL}")
    print(f"{Fore.MAGENTA}╚═══════════════════════════════════════════════════════════╝{Style.RESET_ALL}")
    print()
    print(f"{Fore.CYAN}Chip:     {chip.upper()}{Style.RESET_ALL}")
    print(f"{Fore.CYAN}Port:     {port}{Style.RESET_ALL}")
    print(f"{Fore.CYAN}Firmware: {firmware_path.name}{Style.RESET_ALL}")
    print()

    chip_name_map = {
        "esp32": "esp32",
        "c3": "esp32c3",
        "s3": "esp32s3",
        "c5": "esp32c5",
        "c6": "esp32c6",
    }
    chip_name = chip_name_map.get(chip, "esp32")
    base_args = ["--chip", chip_name, "--port", port, "--baud", "460800"]

    try:
        if args.erase:
            print(f"{Fore.YELLOW}1️⃣  Erasing flash...{Style.RESET_ALL}")
            esptool.main(base_args + ["erase-flash"])
            print(f"{Fore.GREEN}✅ Flash erased{Style.RESET_ALL}\n")
            print(f"{Fore.YELLOW}⏳ Waiting for chip to stabilize...{Style.RESET_ALL}")
            time.sleep(2)

        print(f"{Fore.YELLOW}2️⃣  Flashing firmware...{Style.RESET_ALL}")
        flash_offset_map = {
            "esp32": "0x1000",
            "c3": "0x0",
            "s3": "0x0",
            "c5": "0x2000",
            "c6": "0x0",
        }
        flash_offset = flash_offset_map.get(chip, "0x0")

        max_retries = 3
        for attempt in range(max_retries):
            try:
                if attempt > 0:
                    print(f"{Fore.YELLOW}🔄 Retry attempt {attempt + 1}/{max_retries}...{Style.RESET_ALL}")
                    time.sleep(2)
                esptool.main(
                    base_args
                    + [
                        "--before",
                        "default-reset",
                        "--after",
                        "hard-reset",
                        "write-flash",
                        "--flash-mode",
                        "dio",
                        "--flash-freq",
                        "40m",
                        "--flash-size",
                        "detect",
                        flash_offset,
                        str(firmware_path),
                    ]
                )
                print()
                print(f"{Fore.GREEN}✅ Firmware flashed successfully!{Style.RESET_ALL}")
                print()
                print(f"{Fore.CYAN}Next steps:{Style.RESET_ALL}")
                print("  1. cp src/python/config_local.py.example src/python/config_local.py")
                print("  2. Edit src/python/config_local.py with your credentials")
                print(f"  3. {Fore.GREEN}./espectre micro deploy{Style.RESET_ALL}")
                print(f"  4. {Fore.GREEN}./espectre micro run{Style.RESET_ALL}")
                print()
                return
            except Exception as e:
                if attempt < max_retries - 1:
                    print(f"{Fore.YELLOW}⚠️  Attempt {attempt + 1} failed: {e}{Style.RESET_ALL}")
                    continue
                raise
    except Exception as e:
        print(f"\n{Fore.RED}❌ Error flashing firmware: {e}{Style.RESET_ALL}")
        print(f"\n{Fore.YELLOW}Troubleshooting tips:{Style.RESET_ALL}")
        print("  1. Try holding the BOOT button while connecting")
        print("  2. Use a different USB cable (data cable, not charge-only)")
        print("  3. Try a different USB port (preferably USB 2.0)")
        print("  4. Ensure no other programs are using the serial port")
        print(f"  5. Try with --erase flag: {Fore.GREEN}./espectre micro flash --erase{Style.RESET_ALL}")
        print()
        raise SystemExit(1)


def deploy_code(args) -> None:
    """Deploy Python code to a MicroPython device using mpremote."""
    _require_mpremote()
    port = get_serial_port(args.port)

    config_local_path = PYTHON_SRC_DIR / "config_local.py"
    if not config_local_path.exists():
        print(f"{Fore.RED}❌ src/python/config_local.py not found!{Style.RESET_ALL}")
        print(f"\n{Fore.YELLOW}Create it from the template:{Style.RESET_ALL}")
        print("  cp src/python/config_local.py.example src/python/config_local.py")
        print("  # Then edit src/python/config_local.py with your credentials")
        print()
        raise SystemExit(1)

    print(f"{Fore.MAGENTA}╔═══════════════════════════════════════════════════════════╗{Style.RESET_ALL}")
    print(f"{Fore.MAGENTA}║            μESPectre - Deploying Code to Device           ║{Style.RESET_ALL}")
    print(f"{Fore.MAGENTA}╚═══════════════════════════════════════════════════════════╝{Style.RESET_ALL}")
    print()
    print(f"{Fore.CYAN}Port: {port}{Style.RESET_ALL}")
    print()

    try:
        health = subprocess.run(
            ["mpremote", "connect", port, "exec", 'print("MP_OK")'],
            capture_output=True,
            text=True,
        )
        if health.returncode != 0 or "MP_OK" not in (health.stdout or ""):
            print(f"{Fore.RED}❌ Device is not running a valid MicroPython firmware{Style.RESET_ALL}")
            print(f"{Fore.YELLOW}   Serial output suggests boot failure (e.g. invalid header).{Style.RESET_ALL}")
            print(f"\n{Fore.CYAN}Recommended fix:{Style.RESET_ALL}")
            print(f"  {Fore.GREEN}./espectre micro flash --erase --chip c5{Style.RESET_ALL}")
            print(f"  {Fore.GREEN}./espectre micro deploy{Style.RESET_ALL}")
            print()
            raise SystemExit(1)

        print(f"{Fore.YELLOW}📁 Creating directories...{Style.RESET_ALL}")
        subprocess.run(["mpremote", "connect", port, "mkdir", ":src"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
        subprocess.run(["mpremote", "connect", port, "mkdir", ":src/mqtt"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)

        print(f"{Fore.YELLOW}📤 Uploading files...{Style.RESET_ALL}")
        files_to_upload: List[Tuple[str, str]] = [
            (str(PYTHON_SRC_DIR / "__init__.py"), ":src/"),
            (str(PYTHON_SRC_DIR / "config.py"), ":src/"),
            (str(PYTHON_SRC_DIR / "config_local.py"), ":src/"),
            (str(PYTHON_SRC_DIR / "utils.py"), ":src/"),
            (str(PYTHON_SRC_DIR / "threshold.py"), ":src/"),
            (str(PYTHON_SRC_DIR / "filters.py"), ":src/"),
            (str(PYTHON_SRC_DIR / "features.py"), ":src/"),
            (str(PYTHON_SRC_DIR / "segmentation.py"), ":src/"),
            (str(PYTHON_SRC_DIR / "detector_interface.py"), ":src/"),
            (str(PYTHON_SRC_DIR / "runtime_policy.py"), ":src/"),
            (str(PYTHON_SRC_DIR / "mvs_detector.py"), ":src/"),
            (str(PYTHON_SRC_DIR / "ml_detector.py"), ":src/"),
            (str(PYTHON_SRC_DIR / "ml_weights.py"), ":src/"),
            (str(PYTHON_SRC_DIR / "traffic_generator.py"), ":src/"),
            (str(PYTHON_SRC_DIR / "main.py"), ":src/"),
            (str(PYTHON_SRC_DIR / "csi_streamer.py"), ":src/"),
            (str(PYTHON_SRC_DIR / "mqtt" / "__init__.py"), ":src/mqtt/"),
            (str(PYTHON_SRC_DIR / "mqtt" / "handler.py"), ":src/mqtt/"),
            (str(PYTHON_SRC_DIR / "mqtt" / "commands.py"), ":src/mqtt/"),
        ]
        for src, dst in files_to_upload:
            if not Path(src).exists():
                print(f"{Fore.RED}  ❌ File not found: {src}{Style.RESET_ALL}")
                continue
            print(f"  {src} → {dst}")
            subprocess.run(["mpremote", "connect", port, "cp", src, dst], check=True, capture_output=True)

        print()
        print(f"{Fore.GREEN}✅ Deployment complete!{Style.RESET_ALL}")
        print()
        print(f"{Fore.CYAN}To run the application:{Style.RESET_ALL}")
        print("  ./espectre micro run")
        print()
    except subprocess.CalledProcessError as e:
        print(f"\n{Fore.RED}❌ Error during deployment: {e}{Style.RESET_ALL}")
        raise SystemExit(1)
    except Exception as e:
        print(f"\n{Fore.RED}❌ Unexpected error: {e}{Style.RESET_ALL}")
        raise SystemExit(1)


def stream_csi(args) -> None:
    """Stream CSI data via UDP for real-time visualization."""
    _require_mpremote()
    port = get_serial_port(args.port)
    dest_ip = args.ip
    if not dest_ip:
        print(f"{Fore.RED}❌ Destination IP address required{Style.RESET_ALL}")
        print(f"\n{Fore.YELLOW}Usage: ./espectre micro stream --ip <PC_IP_ADDRESS>{Style.RESET_ALL}")
        print(f"\n{Fore.CYAN}Example: ./espectre micro stream --ip 192.168.1.100{Style.RESET_ALL}")
        raise SystemExit(1)

    duration = args.duration if args.duration else 0
    print(f"{Fore.MAGENTA}╔═══════════════════════════════════════════════════════════╗{Style.RESET_ALL}")
    print(f"{Fore.MAGENTA}║           μESPectre - CSI UDP Streaming                   ║{Style.RESET_ALL}")
    print(f"{Fore.MAGENTA}╚═══════════════════════════════════════════════════════════╝{Style.RESET_ALL}")
    print()
    print(f"{Fore.CYAN}Destination: {dest_ip}:5001{Style.RESET_ALL}")
    print(f"{Fore.CYAN}Duration:    {'infinite' if duration == 0 else f'{duration}s'}{Style.RESET_ALL}")
    print()
    print(f"{Fore.YELLOW}On another terminal (PC), run:{Style.RESET_ALL}")
    print(f"  ./espectre micro collect --label <name> --duration <sec>  {Fore.CYAN}# Collect labeled data{Style.RESET_ALL}")
    print(f"  ./espectre micro detect --log-turbulence               {Fore.CYAN}# Debug live motion inference{Style.RESET_ALL}")
    print()

    process = None
    try:
        exec_cmd = f"from src.csi_streamer import stream_csi; stream_csi('{dest_ip}', duration_sec={duration})"
        process = subprocess.Popen(["mpremote", "connect", port, "exec", exec_cmd])
        process.wait()
    except subprocess.CalledProcessError as e:
        print(f"\n{Fore.RED}❌ Error: {e}{Style.RESET_ALL}")
        raise SystemExit(1)
    except KeyboardInterrupt:
        print(f"\n{Fore.YELLOW}Streaming stopped - cleaning up ESP32...{Style.RESET_ALL}")
        if process:
            process.terminate()
            try:
                process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                process.kill()
        _reset_device(port)
    except Exception as e:
        print(f"\n{Fore.RED}❌ Unexpected error: {e}{Style.RESET_ALL}")
        raise SystemExit(1)


def run_application(args) -> None:
    """Run the MicroPython application on ESP32."""
    _require_mpremote()
    port = get_serial_port(args.port)

    print(f"{Fore.MAGENTA}╔═══════════════════════════════════════════════════════════╗{Style.RESET_ALL}")
    print(f"{Fore.MAGENTA}║           μESPectre - Running Application                 ║{Style.RESET_ALL}")
    print(f"{Fore.MAGENTA}╚═══════════════════════════════════════════════════════════╝{Style.RESET_ALL}")
    print()
    print(f"{Fore.YELLOW}🚀 Starting application...{Style.RESET_ALL}")
    print()

    process = None
    try:
        process = subprocess.Popen(["mpremote", "connect", port, "run", "src/main.py"])
        process.wait()
    except subprocess.CalledProcessError as e:
        print(f"\n{Fore.RED}❌ Error: {e}{Style.RESET_ALL}")
        raise SystemExit(1)
    except KeyboardInterrupt:
        print(f"\n{Fore.YELLOW}Application stopped - cleaning up ESP32...{Style.RESET_ALL}")
        if process:
            process.terminate()
            try:
                process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                process.kill()
        _reset_device(port)
    except Exception as e:
        print(f"\n{Fore.RED}❌ Unexpected error: {e}{Style.RESET_ALL}")
        raise SystemExit(1)


def verify_installation(args) -> None:
    """Verify MicroPython firmware and deployed code."""
    port = get_serial_port(args.port)
    print(f"{Fore.MAGENTA}╔═══════════════════════════════════════════════════════════╗{Style.RESET_ALL}")
    print(f"{Fore.MAGENTA}║             μESPectre - Verifying Installation            ║{Style.RESET_ALL}")
    print(f"{Fore.MAGENTA}╚═══════════════════════════════════════════════════════════╝{Style.RESET_ALL}")
    print()

    all_ok = True

    print(f"{Fore.YELLOW}🔍 Checking CSI firmware support...{Style.RESET_ALL}")
    try:
        result = subprocess.run(
            [
                "mpremote",
                "connect",
                port,
                "exec",
                "import network; wlan = network.WLAN(network.STA_IF); "
                "csi_methods = [m for m in dir(wlan) if m.startswith('csi_')]; "
                "print(','.join(csi_methods) if csi_methods else 'NONE')",
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        csi_methods = result.stdout.strip()
        if csi_methods and csi_methods != "NONE":
            print(f"{Fore.GREEN}✅ CSI methods available: {csi_methods}{Style.RESET_ALL}")
        else:
            print(f"{Fore.RED}❌ CSI methods not found in firmware{Style.RESET_ALL}")
            print(f"{Fore.YELLOW}   Hint: Flash the CSI-enabled firmware:{Style.RESET_ALL}")
            print("   ./espectre micro flash --erase")
            all_ok = False
    except subprocess.CalledProcessError as e:
        print(f"{Fore.RED}❌ Failed to check CSI support: {e.stderr.strip()}{Style.RESET_ALL}")
        all_ok = False
    print()

    print(f"{Fore.YELLOW}🔍 Checking MicroPython version...{Style.RESET_ALL}")
    try:
        result = subprocess.run(
            ["mpremote", "connect", port, "exec", "import sys; print(sys.implementation.version)"],
            capture_output=True,
            text=True,
            check=True,
        )
        print(f"{Fore.GREEN}✅ MicroPython version: {result.stdout.strip()}{Style.RESET_ALL}")
    except subprocess.CalledProcessError:
        print(f"{Fore.RED}❌ Failed to get MicroPython version{Style.RESET_ALL}")
        all_ok = False
    print()

    print(f"{Fore.YELLOW}🔍 Checking deployed files...{Style.RESET_ALL}")
    try:
        result = subprocess.run(
            ["mpremote", "connect", port, "exec", 'import os; print(os.listdir("/src"))'],
            capture_output=True,
            text=True,
            check=True,
        )
        print(f"{Fore.GREEN}✅ Source files found: {result.stdout.strip()}{Style.RESET_ALL}")
    except subprocess.CalledProcessError:
        print(f"{Fore.RED}❌ Source files not found{Style.RESET_ALL}")
        print(f"{Fore.YELLOW}   Hint: Deploy the code first:{Style.RESET_ALL}")
        print("   ./espectre micro deploy")
        all_ok = False
    print()

    print(f"{Fore.YELLOW}🔍 Checking configuration...{Style.RESET_ALL}")
    try:
        result = subprocess.run(
            ["mpremote", "connect", port, "exec", 'import os; print("config_local.py" in os.listdir("/src"))'],
            capture_output=True,
            text=True,
            check=True,
        )
        if "True" in result.stdout:
            print(f"{Fore.GREEN}✅ config_local.py found{Style.RESET_ALL}")
        else:
            print(f"{Fore.YELLOW}⚠️  config_local.py not found (will use defaults from config.py){Style.RESET_ALL}")
    except subprocess.CalledProcessError:
        print(f"{Fore.YELLOW}⚠️  Could not check config_local.py{Style.RESET_ALL}")
    print()

    if all_ok:
        print(f"{Fore.GREEN}╔═══════════════════════════════════════════════════════════╗{Style.RESET_ALL}")
        print(f"{Fore.GREEN}║    μESPectre - ✅ Installation Verified Successfully!     ║{Style.RESET_ALL}")
        print(f"{Fore.GREEN}╚═══════════════════════════════════════════════════════════╝{Style.RESET_ALL}")
        print()
        print(f"{Fore.CYAN}You can now run the application:{Style.RESET_ALL}")
        print("  ./espectre micro run")
        print()
        return

    print(f"{Fore.RED}╔═══════════════════════════════════════════════════════════╗{Style.RESET_ALL}")
    print(f"{Fore.RED}║    μESPectre - ❌ Some checks failed                       ║{Style.RESET_ALL}")
    print(f"{Fore.RED}╚═══════════════════════════════════════════════════════════╝{Style.RESET_ALL}")
    print()
    print(f"{Fore.YELLOW}Please fix the issues above and try again.{Style.RESET_ALL}")
    print()
    raise SystemExit(1)
