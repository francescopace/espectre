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

The ./espectre collect command imports the same ExternalTrafficGenerator
class, persistently selects external mode on one raw-capable device, and
uses --pps as this generator's intentional rate.

Home Assistant integration:
  See src/cpp/frontend/esphome/README.md for external traffic mode.

Thanks to: https://github.com/phoenixtechnam

Author: Francesco Pace <francesco.pace@gmail.com>
"""

import ipaddress
import json
import os
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from pathlib import Path

# ============= CONFIGURATION =============
TARGETS = ['192.168.1.100']  # Unicast device IP
# TARGETS = ['192.168.1.100', '192.168.1.101']  # Multiple devices: list each IP
# TARGETS = ['239.255.0.1']  # All devices that joined the default multicast group
PORT = 5555
RATE = 100  # packets per second (recommended: 100)
_RUNTIME_OWNER = str(os.getuid()) if hasattr(os, "getuid") else "user"
PID_FILE = Path(tempfile.gettempdir()) / f"espectre_traffic_{_RUNTIME_OWNER}.pid"
CONTROL_FILE = Path(tempfile.gettempdir()) / f"espectre_traffic_{_RUNTIME_OWNER}.control"
SENSING_IP_TOS = 46 << 2
# =========================================


def next_send_deadline(previous_deadline, send_started, interval):
    """Keep the requested send-rate phase without scheduling catch-up traffic."""
    if interval <= 0.0:
        return send_started
    if previous_deadline <= 0.0:
        return send_started + interval

    phase_deadline = previous_deadline + interval
    if phase_deadline - send_started < interval / 2.0:
        return send_started + interval
    return phase_deadline


def configure_socket(sock, targets=None, source_ip=None):
    """Configure low-latency unicast or local-link multicast delivery."""
    targets = TARGETS if targets is None else list(targets)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
    sock.setsockopt(socket.IPPROTO_IP, socket.IP_TOS, SENSING_IP_TOS)
    if any(ipaddress.ip_address(target).is_multicast for target in targets):
        sock.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_TTL, 1)
    if source_ip:
        sock.bind((source_ip, 0))


def _atomic_write_text(path, text):
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(text, encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _read_state():
    try:
        payload = json.loads(PID_FILE.read_text(encoding="utf-8"))
        pid = int(payload["pid"])
        token = str(payload["token"])
        if pid <= 0 or not token:
            raise ValueError
        return pid, token
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
        return None


def _control_matches(token):
    try:
        return CONTROL_FILE.read_text(encoding="utf-8").strip() == token
    except OSError:
        return False


def _process_exists(pid):
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def _remove_runtime_files(token):
    if _control_matches(token):
        CONTROL_FILE.unlink(missing_ok=True)
    state = _read_state()
    if state is not None and state[1] == token:
        PID_FILE.unlink(missing_ok=True)


class ExternalTrafficGenerator:
    """Reusable external ESPectre UDP traffic generator."""

    TRAFFIC_MARKER = '👻'
    PAYLOAD = TRAFFIC_MARKER.encode('utf-8')

    def __init__(self, targets, port=PORT, rate_pps=RATE, source_ip=None, control_token=None):
        raw_targets = [targets] if isinstance(targets, str) else list(targets)
        self.targets = []
        for target in raw_targets:
            address = ipaddress.ip_address(str(target).strip())
            if address.version != 4:
                raise ValueError("targets must be IPv4 addresses")
            self.targets.append(str(address))
        if not self.targets:
            raise ValueError("at least one target is required")
        if not 1 <= int(port) <= 65535:
            raise ValueError("port must be in the 1-65535 range")
        if float(rate_pps) <= 0:
            raise ValueError("rate_pps must be greater than zero")
        if source_ip is not None:
            source = ipaddress.ip_address(str(source_ip).strip())
            if source.version != 4:
                raise ValueError("source_ip must be IPv4")
            source_ip = str(source)
        self.port = int(port)
        self.rate_pps = float(rate_pps)
        self.source_ip = source_ip
        self.control_token = control_token
        self.sent_packets = 0
        self.send_errors = 0
        self.sent_by_target = {target: 0 for target in self.targets}
        self.errors_by_target = {target: 0 for target in self.targets}
        self._socket = None
        self._thread = None
        self._stop_event = threading.Event()
        self._lifecycle_lock = threading.RLock()

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, _exc_type, _exc, _traceback):
        self.stop()

    @property
    def running(self):
        with self._lifecycle_lock:
            return self._thread is not None and self._thread.is_alive()

    def start(self):
        with self._lifecycle_lock:
            if self._thread is not None and self._thread.is_alive():
                return
            self._stop_event.clear()
            self._thread = threading.Thread(
                target=self._run,
                name="espectre-external-traffic",
                daemon=True,
            )
            self._thread.start()

    def stop(self):
        with self._lifecycle_lock:
            self._stop_event.set()
            thread = self._thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=2.0)
        with self._lifecycle_lock:
            if self._thread is thread:
                self._thread = None
            if self._socket is not None:
                self._socket.close()
                self._socket = None

    def run_forever(self):
        """Run synchronously until ``stop`` is requested."""
        self._stop_event.clear()
        self._run()

    def _run(self):
        interval = 1.0 / self.rate_pps
        next_time = 0.0
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        with self._lifecycle_lock:
            self._socket = sock
        try:
            configure_socket(sock, self.targets, self.source_ip)
            while not self._stop_event.is_set() and (
                self.control_token is None or _control_matches(self.control_token)
            ):
                if next_time > 0.0:
                    sleep_time = next_time - time.perf_counter()
                    if sleep_time > 0 and self._stop_event.wait(sleep_time):
                        break
                send_started = time.perf_counter()
                for target in self.targets:
                    try:
                        sock.sendto(self.PAYLOAD, (target, self.port))
                        self.sent_packets += 1
                        self.sent_by_target[target] += 1
                    except OSError:
                        self.send_errors += 1
                        self.errors_by_target[target] += 1
                next_time = next_send_deadline(next_time, send_started, interval)
        finally:
            sock.close()
            with self._lifecycle_lock:
                if self._socket is sock:
                    self._socket = None


def start():
    """Start traffic generator in background (daemon mode)."""
    state = _read_state()
    if state is not None:
        pid, token = state
        if _control_matches(token) and _process_exists(pid):
            print(f"Already running (PID {pid})")
            return
        _remove_runtime_files(token)
    else:
        PID_FILE.unlink(missing_ok=True)

    script_path = os.path.abspath(__file__)
    python_exe = sys.executable or "python3"
    token = uuid.uuid4().hex
    _atomic_write_text(CONTROL_FILE, token)

    try:
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

            proc = subprocess.Popen(
                [python_exe, script_path, "run", token],
                **popen_kwargs,
            )
    except OSError:
        _remove_runtime_files(token)
        raise

    try:
        _atomic_write_text(PID_FILE, json.dumps({"pid": proc.pid, "token": token}))
    except OSError:
        _remove_runtime_files(token)
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            pass
        raise
    print(f"Started (PID {proc.pid})")


def stop():
    """Stop running traffic generator."""
    state = _read_state()
    if state is None:
        PID_FILE.unlink(missing_ok=True)
        print("Not running")
        return

    pid, token = state
    if not _control_matches(token):
        PID_FILE.unlink(missing_ok=True)
        print("Not running (stale state file)")
        return

    CONTROL_FILE.unlink(missing_ok=True)
    deadline = time.monotonic() + 5.0
    while _process_exists(pid) and time.monotonic() < deadline:
        time.sleep(0.05)
    PID_FILE.unlink(missing_ok=True)
    if _process_exists(pid):
        print(f"Stop requested (PID {pid})")
    else:
        print(f"Stopped (PID {pid})")


def status():
    """Check if traffic generator is running."""
    state = _read_state()
    if state is None:
        PID_FILE.unlink(missing_ok=True)
        print("Not running")
        return

    pid, token = state
    if _control_matches(token) and _process_exists(pid):
        print(f"Running (PID {pid})")
        return
    _remove_runtime_files(token)
    print("Not running (stale state file)")


def run(token=None):
    """Run traffic generator in foreground (Ctrl+C to stop)."""
    print(f"Sending UDP to {TARGETS}:{PORT} @ {RATE} pps (Ctrl+C to stop)")
    run_loop(token)


def run_loop(token=None):
    """Main packet sending loop."""
    def handle_signal(sig, frame):
        sys.exit(0)

    signal.signal(signal.SIGTERM, handle_signal)
    signal.signal(signal.SIGINT, handle_signal)

    generator = ExternalTrafficGenerator(
        TARGETS,
        port=PORT,
        rate_pps=RATE,
        control_token=token,
    )
    try:
        generator.run_forever()
    finally:
        generator.stop()
        if token is not None:
            _remove_runtime_files(token)


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    cmd = sys.argv[1].lower()
    commands = {'start': start, 'stop': stop, 'status': status}

    if cmd == "run":
        run(sys.argv[2] if len(sys.argv) > 2 else None)
    elif cmd in commands:
        commands[cmd]()
    else:
        print(f"Unknown command: {cmd}")
        print("Use: start, stop, status, or run")
        sys.exit(1)
