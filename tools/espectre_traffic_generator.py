#!/usr/bin/env python3
"""
ESPectre Traffic Generator

Generates UDP traffic to trigger CSI extraction on ESPectre devices.
Works on all platforms: Linux, macOS, Windows, Home Assistant.

Usage:
  python3 espectre_traffic_generator.py start         # Start in background
  python3 espectre_traffic_generator.py stop          # Stop running instance
  python3 espectre_traffic_generator.py status        # Check if running
  python3 espectre_traffic_generator.py run           # Run in foreground (Ctrl+C to stop)

Configuration:
  Edit TARGETS, PORT, RATE below.

Home Assistant integration:
  See SETUP.md for command_line switch configuration.

Author: Francesco Pace <francesco.pace@gmail.com>
Thanks to: https://github.com/phoenixtechnam

License: GPLv3
"""
import socket
import time
import signal
import sys
import os
import subprocess

# ============= CONFIGURATION =============
TARGETS = ['192.168.1.255']  # Broadcast address (recommended for multiple devices)
# TARGETS = ['192.168.1.100', '192.168.1.101']  # Or list specific device IPs
PORT = 5555
RATE = 100  # packets per second (recommended: 100)
PID_FILE = '/tmp/espectre_traffic.pid'
# =========================================


def start():
    """Start traffic generator in background (daemon mode)."""
    if os.path.exists(PID_FILE):
        with open(PID_FILE) as f:
            pid = int(f.read().strip())
        try:
            os.kill(pid, 0)
            print(f"Already running (PID {pid})")
            return
        except OSError:
            os.remove(PID_FILE)

    script_path = os.path.abspath(__file__)
    python_exe = sys.executable or "python3"

    with open(os.devnull, 'w') as devnull:
        popen_kwargs = {
            "stdin": devnull,
            "stdout": devnull,
            "stderr": devnull,
        }
        if os.name == "posix":
            popen_kwargs["start_new_session"] = True
        else:
            popen_kwargs["creationflags"] = (
                subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP
            )

        proc = subprocess.Popen([python_exe, script_path, "run"], **popen_kwargs)

    with open(PID_FILE, 'w') as f:
        f.write(str(proc.pid))
    print(f"Started (PID {proc.pid})")


def stop():
    """Stop running traffic generator."""
    if not os.path.exists(PID_FILE):
        print("Not running")
        return

    with open(PID_FILE) as f:
        pid = int(f.read().strip())

    try:
        os.kill(pid, signal.SIGTERM)
        print(f"Stopped (PID {pid})")
    except OSError:
        print("Process not found")

    if os.path.exists(PID_FILE):
        os.remove(PID_FILE)


def status():
    """Check if traffic generator is running."""
    if not os.path.exists(PID_FILE):
        print("Not running")
        return

    with open(PID_FILE) as f:
        pid = int(f.read().strip())

    try:
        os.kill(pid, 0)
        print(f"Running (PID {pid})")
    except OSError:
        print("Not running (stale PID file)")
        os.remove(PID_FILE)


def run():
    """Run traffic generator in foreground (Ctrl+C to stop)."""
    print(f"Sending UDP to {TARGETS}:{PORT} @ {RATE} pps (Ctrl+C to stop)")
    run_loop()


def run_loop():
    """Main packet sending loop."""
    def handle_signal(sig, frame):
        sys.exit(0)

    signal.signal(signal.SIGTERM, handle_signal)
    signal.signal(signal.SIGINT, handle_signal)

    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)

    interval = 1.0 / RATE
    next_time = time.perf_counter()

    try:
        while True:
            for ip in TARGETS:
                s.sendto(b'.', (ip, PORT))
            next_time += interval
            sleep_time = next_time - time.perf_counter()
            if sleep_time > 0:
                time.sleep(sleep_time)
    finally:
        s.close()
        if os.path.exists(PID_FILE):
            os.remove(PID_FILE)


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    cmd = sys.argv[1].lower()
    commands = {'start': start, 'stop': stop, 'status': status, 'run': run}

    if cmd in commands:
        commands[cmd]()
    else:
        print(f"Unknown command: {cmd}")
        print("Use: start, stop, status, or run")
        sys.exit(1)

