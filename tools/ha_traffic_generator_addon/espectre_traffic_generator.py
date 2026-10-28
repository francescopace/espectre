#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-only
# Commercial licensing available under separate agreement; see LICENSING.md.
"""
ESPectre - Traffic Generator

Generates traffic to trigger CSI extraction on ESPectre devices in external
traffic mode: UDP markers on port 5555, or ICMP Echo Requests (ping). Works on Linux,
macOS, Windows, and Home Assistant; ICMP is not available on Windows.

Prefer ICMP in most setups. Each device answers every Echo Request, and that
reply keeps it transmitting while it measures. UDP markers get no reply, and
without steady uplink an empty room can show motion. Use UDP only to save
airtime, when the devices already transmit at a similar steady rate.
ICMP to a multicast group needs firmware 3.1.0 or later.

Do not target a subnet or limited broadcast address. Those frames usually
arrive as legacy PHY and do not produce HT20 CSI.

Usage:
  From tools/ha_traffic_generator_addon/:
  python3 espectre_traffic_generator.py start         # Start in background
  python3 espectre_traffic_generator.py stop          # Stop running instance
  python3 espectre_traffic_generator.py status        # Check if running
  python3 espectre_traffic_generator.py run           # Run in foreground (Ctrl+C to stop)

Configuration:
  Edit TARGETS, PROTOCOL, PORT, RATE, MULTICAST_TTL, and DSCP below. PORT
  applies only to UDP. Unicast each device IP,
  or use the joined multicast group 239.255.0.1. Do not use x.x.x.255.
  MULTICAST_TTL defaults to 8; routed multicast still needs multicast routing.
  Ordinary switches on the same VLAN do not consume TTL.
  DSCP defaults to 46 (Expedited Forwarding); use 0 for Best Effort.

The ./espectre collect command imports the same ExternalTrafficGenerator
class, persistently selects external mode on one raw-capable device, and
uses --pps as this generator's intentional rate.

Home Assistant integration:
  See DOCS.md in this directory for add-on installation and external traffic mode.

Thanks to: https://github.com/phoenixtechnam

Author: Francesco Pace <francesco.pace@gmail.com>
"""

import ipaddress
import json
import os
import select
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
PROTOCOL = 'udp'  # 'udp' markers, or 'icmp' Echo Requests that each device answers
PORT = 5555
RATE = 100  # packets per second (recommended: 100)
MULTICAST_TTL = 8  # Allows up to seven router hops when multicast routing is configured.
DSCP = 46  # Expedited Forwarding; 0 requests Best Effort (valid range: 0-63).
_RUNTIME_OWNER = str(os.getuid()) if hasattr(os, "getuid") else "user"
PID_FILE = Path(tempfile.gettempdir()) / f"espectre_traffic_{_RUNTIME_OWNER}.pid"
CONTROL_FILE = Path(tempfile.gettempdir()) / f"espectre_traffic_{_RUNTIME_OWNER}.control"
# =========================================


PROTOCOLS = ("udp", "icmp")
_ICMP_ECHO_REQUEST = 8
_ICMP_ECHO_REPLY = 0


def icmp_checksum(data):
    """Return the RFC 1071 Internet checksum of data."""
    if len(data) % 2:
        data += b"\0"
    total = sum(int.from_bytes(data[index:index + 2], "big") for index in range(0, len(data), 2))
    while total >> 16:
        total = (total & 0xFFFF) + (total >> 16)
    return ~total & 0xFFFF


def icmp_echo_request(identifier, sequence):
    """Build an ICMP Echo Request without data: the smallest frame the device answers."""
    header = bytes([_ICMP_ECHO_REQUEST, 0, 0, 0]) + (
        (identifier & 0xFFFF).to_bytes(2, "big") + (sequence & 0xFFFF).to_bytes(2, "big"))
    checksum = icmp_checksum(header)
    return header[:2] + checksum.to_bytes(2, "big") + header[4:]


def open_icmp_socket():
    """Open an ICMP socket, unprivileged where the OS allows it, else raw."""
    if os.name == "nt":
        raise OSError("ICMP traffic is not supported on Windows; use UDP")
    try:
        return socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_ICMP), False
    except OSError as unprivileged_error:
        try:
            return socket.socket(socket.AF_INET, socket.SOCK_RAW, socket.IPPROTO_ICMP), True
        except OSError as raw_error:
            raise OSError(
                "ICMP traffic needs an unprivileged ICMP socket (on Linux, allow this group in "
                "net.ipv4.ping_group_range) or a raw socket (root or CAP_NET_RAW); "
                f"unprivileged: {unprivileged_error}; raw: {raw_error}"
            ) from raw_error


def is_echo_reply(packet, identifiers=None):
    """Return whether packet, with or without its IPv4 header, is an Echo Reply to us."""
    if len(packet) >= 20 and packet[0] >> 4 == 4:
        packet = packet[(packet[0] & 0x0F) * 4:]
    if len(packet) < 8 or packet[0] != _ICMP_ECHO_REPLY:
        return False
    return identifiers is None or int.from_bytes(packet[4:6], "big") in identifiers


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


def configure_socket(sock, targets=None, source_ip=None, multicast_ttl=MULTICAST_TTL,
                     dscp=DSCP):
    """Configure DSCP, multicast scope, and an optional source interface."""
    targets = TARGETS if targets is None else list(targets)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
    sock.setsockopt(socket.IPPROTO_IP, socket.IP_TOS, dscp << 2)
    if any(ipaddress.ip_address(target).is_multicast for target in targets):
        sock.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_TTL, multicast_ttl)
        if source_ip:
            sock.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_IF,
                            socket.inet_aton(source_ip))
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
    """Reusable external ESPectre traffic generator: UDP markers or ICMP Echo Requests."""

    TRAFFIC_MARKER = '👻'
    PAYLOAD = TRAFFIC_MARKER.encode('utf-8')

    def __init__(self, targets, port=PORT, rate_pps=RATE, source_ip=None, control_token=None,
                 multicast_ttl=MULTICAST_TTL, dscp=DSCP, protocol=PROTOCOL):
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
        if type(multicast_ttl) is not int or not 1 <= multicast_ttl <= 255:
            raise ValueError("multicast_ttl must be an integer in the 1-255 range")
        if type(dscp) is not int or not 0 <= dscp <= 63:
            raise ValueError("dscp must be an integer in the 0-63 range")
        if protocol not in PROTOCOLS:
            raise ValueError("protocol must be 'udp' or 'icmp'")
        if source_ip is not None:
            source = ipaddress.ip_address(str(source_ip).strip())
            if source.version != 4:
                raise ValueError("source_ip must be IPv4")
            source_ip = str(source)
        self.port = int(port)
        self.rate_pps = float(rate_pps)
        self.source_ip = source_ip
        self.multicast_ttl = multicast_ttl
        self.dscp = dscp
        self.protocol = protocol
        self.control_token = control_token
        self.sent_packets = 0
        self.send_errors = 0
        self.replies_received = 0
        # Why the background thread ended, if it failed; cleared by start().
        self.error = None
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
            self.error = None
            self._thread = threading.Thread(
                target=self._run_thread,
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

    def _run_thread(self):
        try:
            self._run()
        except Exception as error:
            self.error = str(error) or type(error).__name__
            raise

    def _drain_echo_replies(self, sock, identifiers):
        """Count pending Echo Replies without blocking the send cadence."""
        while True:
            try:
                if not select.select([sock], [], [], 0)[0]:
                    return
                packet = sock.recv(2048)
            except (OSError, ValueError):
                return
            if is_echo_reply(packet, identifiers):
                self.replies_received += 1

    def run_forever(self, report_interval=0):
        """Run synchronously, optionally reporting counters every N seconds."""
        self._stop_event.clear()
        self._run(report_interval=report_interval)

    def _run(self, report_interval=0):
        interval = 1.0 / self.rate_pps
        next_time = 0.0
        next_report = time.perf_counter() + report_interval if report_interval > 0 else None
        first_error_reported = False
        last_error = None
        icmp = self.protocol == "icmp"
        raw_icmp = False
        # Raw sockets, and ping sockets on macOS, see every Echo Reply to the
        # host; match ours by identifier.
        identifier = int.from_bytes(os.urandom(2), "big")
        sequence = 0
        if icmp:
            sock, raw_icmp = open_icmp_socket()
        else:
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        with self._lifecycle_lock:
            self._socket = sock
        try:
            configure_socket(sock, self.targets, self.source_ip, self.multicast_ttl, self.dscp)
            while not self._stop_event.is_set() and (
                self.control_token is None or _control_matches(self.control_token)
            ):
                if next_time > 0.0:
                    sleep_time = next_time - time.perf_counter()
                    if sleep_time > 0 and self._stop_event.wait(sleep_time):
                        break
                send_started = time.perf_counter()
                if icmp:
                    identifiers = {identifier}
                    if not raw_icmp:
                        # Linux replaces the identifier of a ping socket with its local port.
                        try:
                            identifiers.add(sock.getsockname()[1])
                        except OSError:
                            pass
                    self._drain_echo_replies(sock, identifiers)
                    payload = icmp_echo_request(identifier, sequence)
                    sequence += 1
                for target in self.targets:
                    try:
                        if icmp:
                            sock.sendto(payload, (target, 0))
                        else:
                            sock.sendto(self.PAYLOAD, (target, self.port))
                        self.sent_packets += 1
                        self.sent_by_target[target] += 1
                    except OSError as error:
                        self.send_errors += 1
                        self.errors_by_target[target] += 1
                        last_error = f"{target}: {error}"
                        if next_report is not None and not first_error_reported:
                            print(f"Failed to send traffic to {last_error}; "
                                  f"further errors summarized every {report_interval:g}s", flush=True)
                            first_error_reported = True
                next_time = next_send_deadline(next_time, send_started, interval)
                if next_report is not None and send_started >= next_report:
                    message = f"Sent {self.sent_packets} packets; send errors: {self.send_errors}"
                    if icmp:
                        message += f"; echo replies: {self.replies_received}"
                    if last_error is not None:
                        message += f"; last error: {last_error}"
                    print(message, flush=True)
                    last_error = None
                    next_report = send_started + report_interval
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


def run(token=None, *, targets=None, port=None, rate_pps=None, source_ip=None,
        multicast_ttl=None, dscp=None, protocol=None):
    """Run traffic generator in foreground (Ctrl+C to stop)."""
    def handle_signal(sig, frame):
        sys.exit(0)

    signal.signal(signal.SIGTERM, handle_signal)
    signal.signal(signal.SIGINT, handle_signal)

    generator = ExternalTrafficGenerator(
        TARGETS if targets is None else targets,
        port=PORT if port is None else port,
        rate_pps=RATE if rate_pps is None else rate_pps,
        source_ip=source_ip,
        control_token=token,
        multicast_ttl=MULTICAST_TTL if multicast_ttl is None else multicast_ttl,
        dscp=DSCP if dscp is None else dscp,
        protocol=PROTOCOL if protocol is None else protocol,
    )
    try:
        scope = (f"; multicast TTL {generator.multicast_ttl}"
                 if any(ipaddress.ip_address(target).is_multicast
                        for target in generator.targets) else "")
        destination = ("ICMP Echo Requests to " + str(generator.targets)
                       if generator.protocol == "icmp"
                       else f"UDP to {generator.targets}:{generator.port}")
        print(f"Sending {destination} "
              f"@ {generator.rate_pps:g} pps per target; DSCP {generator.dscp}"
              f"{scope} (Ctrl+C to stop)", flush=True)
        generator.run_forever(report_interval=60)
    finally:
        generator.stop()
        if token is not None:
            _remove_runtime_files(token)
        print("External CSI traffic generator stopped", flush=True)


def main(argv=None):
    """Dispatch the standalone traffic-generator command."""
    args = list(sys.argv[1:] if argv is None else argv)
    if not args:
        print(__doc__)
        return 1

    cmd = args[0].lower()
    commands = {'start': start, 'stop': stop, 'status': status}

    if cmd in {"-h", "--help", "help"}:
        print(__doc__)
    elif cmd == "run":
        run(args[1] if len(args) > 1 else None)
    elif cmd in commands:
        commands[cmd]()
    else:
        print(f"Unknown command: {cmd}")
        print("Use: start, stop, status, or run")
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
