#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-only
# Commercial licensing available under separate agreement; see LICENSING.md.
"""
ESPectre - Traffic Generator

Generates UDP traffic to trigger CSI extraction on ESPectre devices
in external traffic mode (port 5555). Works on Linux, macOS, Windows, and
Home Assistant.

Do not target a subnet or limited broadcast address. Those frames usually
arrive as legacy PHY and do not produce HT20 CSI.

Usage:
  python3 espectre_traffic_generator.py start         # Start in background
  python3 espectre_traffic_generator.py stop          # Stop running instance
  python3 espectre_traffic_generator.py status        # Check if running
  python3 espectre_traffic_generator.py run           # Run in foreground (Ctrl+C to stop)

Configuration:
  Edit TARGETS, PORT, RATE below. Unicast each device IP, or use the joined
  multicast group 239.255.0.1. Do not use x.x.x.255.

Streamer collection should use ./espectre collect, not this script. The
Streamer default pacing port is 9999, and several streamers can share the
same multicast group 239.255.0.1.

Home Assistant integration:
  See src/cpp/frontend/esphome/README.md for external traffic mode.

Thanks to: https://github.com/phoenixtechnam

Author: Francesco Pace <francesco.pace@gmail.com>
"""

import ipaddress
import os
import signal
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path

# ============= CONFIGURATION =============
TARGETS = ['192.168.1.100']  # Unicast device IP
# TARGETS = ['192.168.1.100', '192.168.1.101']  # Multiple devices: list each IP
# TARGETS = ['239.255.0.1']  # All devices that joined the default multicast group
PORT = 5555
RATE = 100  # packets per second (recommended: 100)
PID_FILE = Path(tempfile.gettempdir()) / "espectre_traffic.pid"
SENSING_IP_TOS = 46 << 2
# =========================================


def next_send_deadline(previous_deadline, send_started, interval):
    """Keep ordinary pacing phase without scheduling a catch-up packet."""
    if interval <= 0.0:
        return send_started
    if previous_deadline <= 0.0:
        return send_started + interval

    phase_deadline = previous_deadline + interval
    if phase_deadline - send_started < interval / 2.0:
        return send_started + interval
    return phase_deadline


def configure_socket(sock):
    """Configure low-latency unicast or local-link multicast pacing."""
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
    sock.setsockopt(socket.IPPROTO_IP, socket.IP_TOS, SENSING_IP_TOS)
    if any(ipaddress.ip_address(target).is_multicast for target in TARGETS):
        sock.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_TTL, 1)


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

    try:
        PID_FILE.write_text(str(proc.pid), encoding="utf-8")
    except OSError:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
        raise
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

    if RATE <= 0:
        raise ValueError(f"RATE must be > 0, got {RATE}")
    interval = 1.0 / RATE
    next_time = 0.0
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    try:
        configure_socket(s)
        while True:
            if next_time > 0.0:
                sleep_time = next_time - time.perf_counter()
                if sleep_time > 0:
                    time.sleep(sleep_time)

            send_started = time.perf_counter()
            for ip in TARGETS:
                s.sendto(b'.', (ip, PORT))
            next_time = next_send_deadline(next_time, send_started, interval)
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
