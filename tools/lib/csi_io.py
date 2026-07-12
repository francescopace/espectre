"""
CSI stream I/O, collection, and dataset loading helpers for tooling.
"""

from __future__ import annotations

import ipaddress
import socket
import struct
import subprocess
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

import numpy as np

from .bootstrap import setup_paths
from .csi_analysis import VarianceDetectorAdapter
from . import dataset_metadata

setup_paths()

try:
    import config
except ImportError:
    import src.config as config

MAGIC_STREAM = 0x4353
STREAM_VERSION = 6
DEFAULT_PORT = 5001
STREAM_FLAG_FIRST_WORD_INVALID = 1 << 0
STREAM_FLAG_WIFI_RX_TS_VALID = 1 << 1
STREAM_FLAG_WIFI_RX_START_TS_NS_VALID = 1 << 2
STREAM_FLAG_CSI_FRESH = 1 << 3
CSI_HEADER_FORMAT = "<HBBBBIHHQQIQBbbQII"
CSI_HEADER_STRUCT = struct.Struct(CSI_HEADER_FORMAT)
MAX_STREAM_DATAGRAM_BYTES = 2048
DEFAULT_SOCKET_RCVBUF_BYTES = 1024 * 1024
DEFAULT_PACING_PORT = 9999
DEFAULT_PACING_INTERVAL_SECONDS = 5.0
DEFAULT_PACING_PAYLOAD = b"ESPE"

CHIP_CODES = {
    0: "unknown",
    1: "ESP32",
    2: "S2",
    3: "S3",
    4: "C3",
    5: "C5",
    6: "C6",
}
def get_default_bind_host() -> str:
    """Determine a safe default bind interface."""
    import os

    env_host = os.getenv("CSI_BIND_HOST", "").strip()
    if env_host:
        return env_host

    probe = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        probe.connect(("8.8.8.8", 80))
        return probe.getsockname()[0]
    except OSError:
        return "127.0.0.1"
    finally:
        probe.close()


def format_device_token(device_id: int) -> str:
    return f"dev{int(device_id):016x}"


def format_device_id_hex(device_id: int) -> str:
    return f"0x{int(device_id):016x}"


@dataclass
class CSIPacket:
    """One CSI packet received via UDP."""

    timestamp: float
    seq_num: int
    num_subcarriers: int
    iq_raw: np.ndarray
    chip: str = "unknown"
    device_id: Optional[int] = None
    device_ticks_us: Optional[int] = None
    wifi_rx_ts_us: Optional[int] = None
    wifi_rx_start_ts_ns: Optional[int] = None
    channel: Optional[int] = None
    rssi_dbm: Optional[int] = None
    noise_floor_dbm: Optional[int] = None
    tx_backpressure_total: Optional[int] = None
    stream_fresh_total: Optional[int] = None
    pacing_rx_total: Optional[int] = None
    # False for older captures where the device could re-send its latest CSI
    # sample. Current streamer firmware emits only fresh CSI records.
    csi_fresh: bool = True
    source_ip: Optional[str] = None
    _iq_complex: Optional[np.ndarray] = None
    _amplitudes: Optional[np.ndarray] = None
    _phases: Optional[np.ndarray] = None

    def _ensure_derived_arrays(self) -> None:
        if self._iq_complex is not None and self._amplitudes is not None and self._phases is not None:
            return
        q_values = self.iq_raw[0::2].astype(np.float32)
        i_values = self.iq_raw[1::2].astype(np.float32)
        iq_complex = i_values + 1j * q_values
        self._iq_complex = iq_complex.astype(np.complex64, copy=False)
        self._amplitudes = np.abs(self._iq_complex)
        self._phases = np.angle(self._iq_complex)

    @property
    def iq_complex(self) -> np.ndarray:
        self._ensure_derived_arrays()
        return self._iq_complex

    @property
    def amplitudes(self) -> np.ndarray:
        self._ensure_derived_arrays()
        return self._amplitudes

    @property
    def phases(self) -> np.ndarray:
        self._ensure_derived_arrays()
        return self._phases


def build_pacing_datagram(*, payload: bytes = DEFAULT_PACING_PAYLOAD) -> bytes:
    """Build one UDP pacing datagram consumed by the streamer firmware."""
    if not payload:
        raise ValueError("payload cannot be empty")
    return bytes(payload)


class UdpPacingSender:
    """Background UDP sender that refreshes the collector pacing path periodically."""

    def __init__(
        self,
        target_host: str | Iterable[str],
        target_port: int = DEFAULT_PACING_PORT,
        source_host: Optional[str] = None,
        interval_s: float = DEFAULT_PACING_INTERVAL_SECONDS,
        payload: bytes = DEFAULT_PACING_PAYLOAD,
    ):
        if target_port <= 0 or target_port > 65535:
            raise ValueError(f"invalid target_port: {target_port}")
        if interval_s <= 0:
            raise ValueError(f"interval_s must be > 0, got {interval_s}")
        raw_targets = [target_host] if isinstance(target_host, str) else list(target_host)
        self.target_hosts: List[str] = []
        self.target_ips: List[ipaddress.IPv4Address] = []
        for raw_target in raw_targets:
            target = str(raw_target).strip()
            if not target:
                continue
            try:
                target_ip = ipaddress.ip_address(target)
            except ValueError as exc:
                raise ValueError(f"invalid target_host: {target}") from exc
            if target_ip.version != 4:
                raise ValueError(f"target_host must be an IPv4 address: {target}")
            self.target_hosts.append(target)
            self.target_ips.append(target_ip)
        if not self.target_hosts:
            raise ValueError("target_host cannot be empty")

        self.target_port = int(target_port)
        self.source_host = str(source_host).strip() if source_host is not None else ""
        self.interval_s = float(interval_s)
        self.payload = build_pacing_datagram(payload=payload)
        if self.source_host:
            try:
                ipaddress.ip_address(self.source_host)
            except ValueError as exc:
                raise ValueError(f"invalid source_host: {self.source_host}") from exc
        self.sent_packets = 0
        self.sock: Optional[socket.socket] = None
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._interval_lock = threading.Lock()

    def start(self) -> None:
        if self._thread is not None:
            return
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        if any(target_ip.is_multicast for target_ip in self.target_ips):
            self.sock.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_TTL, 1)
        if self.source_host:
            self.sock.bind((self.source_host, 0))
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, name="espectre-pacing", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None
        if self.sock is not None:
            self.sock.close()
            self.sock = None

    def set_rate_pps(self, rate_pps: float) -> None:
        rate_value = float(rate_pps)
        if rate_value <= 0:
            raise ValueError(f"rate_pps must be > 0, got {rate_value}")
        with self._interval_lock:
            self.interval_s = 1.0 / rate_value

    def get_rate_pps(self) -> float:
        with self._interval_lock:
            return 1.0 / self.interval_s

    def _send_once(self) -> None:
        try:
            if self.sock is not None:
                for target_host in self.target_hosts:
                    self.sock.sendto(self.payload, (target_host, self.target_port))
        except OSError:
            pass
        self.sent_packets += 1

    def _run(self) -> None:
        next_send_time = time.perf_counter()
        while not self._stop_event.is_set():
            self._send_once()
            with self._interval_lock:
                interval_s = self.interval_s
            next_send_time += interval_s
            sleep_time = next_send_time - time.perf_counter()
            if sleep_time > 0 and self._stop_event.wait(sleep_time):
                break


class AdaptivePacingController:
    """Track firmware backpressure telemetry and optionally adjust pacing."""

    def __init__(
        self,
        *,
        initial_pps: float,
        enabled: bool = False,
        min_pps: Optional[float] = None,
        max_pps: Optional[float] = None,
        additive_step_pps: Optional[float] = None,
        control_window_s: float = 1.0,
        receive_tolerance_ratio: float = 0.05,
    ):
        initial_pps = float(initial_pps)
        min_pps = max(5.0, min(initial_pps, 10.0)) if min_pps is None else max(0.1, float(min_pps))
        max_pps = max(initial_pps * 1.25, min_pps) if max_pps is None else max(float(max_pps), min_pps)
        additive_step_pps = (
            max(1.0, initial_pps * 0.02) if additive_step_pps is None else max(0.1, float(additive_step_pps))
        )
        self.enabled = bool(enabled)
        self.current_pps = initial_pps
        self.target_pps = initial_pps
        self.min_pps = min_pps
        self.max_pps = max_pps
        self.additive_step_pps = additive_step_pps
        self.control_window_s = max(0.1, float(control_window_s))
        self.receive_tolerance_ratio = max(0.0, min(0.5, float(receive_tolerance_ratio)))
        self.last_control_at: Optional[float] = None
        self.backpressure_windows = 0
        self.last_window_backpressure_delta = 0
        self.last_window_backpressure_source: Optional[str] = None
        self.last_window_receive_ratio: Optional[float] = None
        self.last_action = "hold"

    def observe_device(
        self,
        device_state: Dict[str, Any],
        tx_backpressure_total: Optional[int],
        stream_fresh_total: Optional[int] = None,
        pacing_rx_total: Optional[int] = None,
    ) -> None:
        if tx_backpressure_total is not None:
            total = int(tx_backpressure_total)
            previous_total = device_state.get("tx_backpressure_total")
            if previous_total is not None and total >= previous_total:
                device_state["tx_backpressure_window_delta"] = int(
                    device_state.get("tx_backpressure_window_delta", 0) or 0
                ) + (total - previous_total)
            device_state["tx_backpressure_total"] = total

        if stream_fresh_total is not None:
            fresh_total = int(stream_fresh_total)
            previous_fresh_total = device_state.get("stream_fresh_total")
            if previous_fresh_total is not None and fresh_total >= previous_fresh_total:
                device_state["stream_fresh_window_delta"] = int(device_state.get("stream_fresh_window_delta", 0) or 0) + (
                    fresh_total - previous_fresh_total
                )
            device_state["stream_fresh_total"] = fresh_total

        if pacing_rx_total is not None:
            rx_total = int(pacing_rx_total)
            previous_rx_total = device_state.get("pacing_rx_total")
            if previous_rx_total is not None and rx_total >= previous_rx_total:
                device_state["pacing_rx_window_delta"] = int(device_state.get("pacing_rx_window_delta", 0) or 0) + (
                    rx_total - previous_rx_total
                )
            device_state["pacing_rx_total"] = rx_total

    def _apply_pacing_rate(self, new_pps: float, *, action: str, pacing_sender: Any) -> None:
        clamped_pps = max(self.min_pps, min(self.max_pps, float(new_pps)))
        if abs(clamped_pps - self.current_pps) < 1e-9:
            self.last_action = action
            return
        if pacing_sender is None or not hasattr(pacing_sender, "set_rate_pps"):
            self.last_action = "hold"
            return
        pacing_sender.set_rate_pps(clamped_pps)
        self.current_pps = clamped_pps
        self.last_action = action

    def maybe_adjust(self, device_states: Dict[Any, Dict[str, Any]], *, now: float, pacing_sender: Any = None) -> None:
        if self.last_control_at is None:
            self.last_control_at = now
            for device_state in device_states.values():
                device_state["adaptive_prev_packet_total"] = int(device_state.get("packet_count", 0) or 0)
            return
        elapsed_window_s = now - self.last_control_at
        if elapsed_window_s < self.control_window_s:
            return

        worst_delta = 0
        tracked_devices = 0
        worst_sources: List[str] = []
        target_packets = max(self.target_pps * elapsed_window_s, 1.0)
        highest_receive_ratio: Optional[float] = None
        lowest_receive_ratio: Optional[float] = None
        for device_state in device_states.values():
            window_delta = int(device_state.get("tx_backpressure_window_delta", 0) or 0)
            device_state["tx_backpressure_last_delta"] = window_delta
            device_state["tx_backpressure_window_delta"] = 0

            packet_total = int(device_state.get("packet_count", 0) or 0)
            previous_packet_total = device_state.get("adaptive_prev_packet_total")
            receive_delta = 0
            if previous_packet_total is not None and packet_total >= int(previous_packet_total):
                receive_delta = packet_total - int(previous_packet_total)
            device_state["adaptive_prev_packet_total"] = packet_total
            device_state["receive_window_last_delta"] = receive_delta

            fresh_delta = int(device_state.get("stream_fresh_window_delta", 0) or 0)
            pacing_delta = int(device_state.get("pacing_rx_window_delta", 0) or 0)
            device_state["stream_fresh_last_delta"] = fresh_delta
            device_state["pacing_rx_last_delta"] = pacing_delta
            device_state["stream_fresh_window_delta"] = 0
            device_state["pacing_rx_window_delta"] = 0
            tracked_devices += 1
            if window_delta > worst_delta:
                worst_delta = window_delta
                worst_sources = [str(device_state.get("source_ip") or "?")]
            elif window_delta > 0 and window_delta == worst_delta:
                worst_sources.append(str(device_state.get("source_ip") or "?"))

        self.last_control_at = now
        self.last_window_backpressure_delta = worst_delta
        self.last_window_backpressure_source = worst_sources[0] if worst_sources else None
        if tracked_devices > 0:
            for device_state in device_states.values():
                receive_delta = int(device_state.get("receive_window_last_delta", 0) or 0)
                receive_ratio = receive_delta / target_packets
                if highest_receive_ratio is None or receive_ratio > highest_receive_ratio:
                    highest_receive_ratio = receive_ratio
                if lowest_receive_ratio is None or receive_ratio < lowest_receive_ratio:
                    lowest_receive_ratio = receive_ratio
        self.last_window_receive_ratio = lowest_receive_ratio
        if tracked_devices == 0:
            self.last_action = "hold"
            return
        if not self.enabled:
            self.last_action = "hold"
            return

        if worst_delta > 0:
            self.backpressure_windows += 1
            reduce_factor = 0.85 if self.backpressure_windows == 1 else 0.70
            self._apply_pacing_rate(self.current_pps * reduce_factor, action="slowdown", pacing_sender=pacing_sender)
            return

        self.backpressure_windows = 0
        lower_receive_bound = 1.0 - self.receive_tolerance_ratio
        upper_receive_bound = 1.0 + self.receive_tolerance_ratio
        if lowest_receive_ratio is not None and lowest_receive_ratio < lower_receive_bound:
            self._apply_pacing_rate(
                self.current_pps + self.additive_step_pps,
                action="speedup",
                pacing_sender=pacing_sender,
            )
            return
        if highest_receive_ratio is not None and highest_receive_ratio > upper_receive_bound:
            self._apply_pacing_rate(
                self.current_pps - self.additive_step_pps,
                action="trim",
                pacing_sender=pacing_sender,
            )
            return
        self.last_action = "hold"


class CSIReceiver:
    """UDP receiver for CSI data with callback support."""

    def __init__(
        self,
        port: int = DEFAULT_PORT,
        buffer_size: int = 500,
        bind_host: Optional[str] = None,
        socket_rcvbuf_bytes: int = DEFAULT_SOCKET_RCVBUF_BYTES,
        derive_complex: bool = True,
    ):
        self.port = port
        self.buffer_size = buffer_size
        self.bind_host = str(bind_host or get_default_bind_host()).strip()
        self.socket_rcvbuf_bytes = max(int(socket_rcvbuf_bytes), 0)
        self.derive_complex = bool(derive_complex)
        self.effective_socket_rcvbuf_bytes: Optional[int] = None
        if not self.bind_host:
            raise ValueError("bind_host cannot be empty")
        try:
            ipaddress.ip_address(self.bind_host)
        except ValueError as exc:
            raise ValueError(f"Invalid bind_host: {self.bind_host}") from exc

        self.buffer: deque[CSIPacket] = deque(maxlen=buffer_size)
        self.packet_count = 0
        self.dropped_count = 0
        self.last_seq = -1
        self.start_time = 0.0
        self.pps = 0
        self._pps_counter = 0
        self._last_pps_time = 0.0
        self._callbacks: List[Callable[[CSIPacket], None]] = []
        self._buffer_callbacks: List[Tuple[Callable[[deque], None], int]] = []
        self.sock: Optional[socket.socket] = None
        self.running = False

    def add_callback(self, callback: Callable[[CSIPacket], None]) -> None:
        self._callbacks.append(callback)

    def add_buffer_callback(self, callback: Callable[[deque], None], interval: int = 10) -> None:
        self._buffer_callbacks.append((callback, interval))

    def _parse_record(self, data: bytes, offset: int = 0) -> Tuple[Optional[CSIPacket], int]:
        if offset < 0 or len(data) - offset < CSI_HEADER_STRUCT.size:
            return None, offset

        (
            magic,
            version,
            header_len,
            chip_code,
            flags,
            seq_num,
            num_sc,
            csi_len_bytes,
            device_id,
            device_ticks_us,
            wifi_rx_ts_us,
            wifi_rx_start_ts_ns,
            channel,
            rssi_dbm,
            noise_floor_dbm,
            tx_backpressure_total,
            stream_fresh_total,
            pacing_rx_total,
        ) = CSI_HEADER_STRUCT.unpack_from(data, offset)

        if magic != MAGIC_STREAM or version != STREAM_VERSION:
            return None, offset
        if header_len < CSI_HEADER_STRUCT.size:
            return None, offset
        if csi_len_bytes == 0 or csi_len_bytes != num_sc * 2:
            return None, offset

        record_len = header_len + csi_len_bytes
        if len(data) - offset < record_len:
            return None, offset

        iq_raw = np.frombuffer(data, dtype=np.int8, count=csi_len_bytes, offset=offset + header_len).copy()
        iq_complex: Optional[np.ndarray] = None
        amplitudes: Optional[np.ndarray] = None
        phases: Optional[np.ndarray] = None
        if self.derive_complex:
            q_values = iq_raw[0::2].astype(np.float32)
            i_values = iq_raw[1::2].astype(np.float32)
            iq_complex = (i_values + 1j * q_values).astype(np.complex64, copy=False)
            amplitudes = np.abs(iq_complex)
            phases = np.angle(iq_complex)
        packet = CSIPacket(
            timestamp=time.time(),
            seq_num=seq_num,
            num_subcarriers=num_sc,
            iq_raw=iq_raw,
            chip=CHIP_CODES.get(chip_code, "unknown"),
            device_id=device_id or None,
            device_ticks_us=device_ticks_us or None,
            wifi_rx_ts_us=wifi_rx_ts_us if (flags & STREAM_FLAG_WIFI_RX_TS_VALID) else None,
            wifi_rx_start_ts_ns=wifi_rx_start_ts_ns if (flags & STREAM_FLAG_WIFI_RX_START_TS_NS_VALID) else None,
            channel=int(channel),
            rssi_dbm=int(rssi_dbm),
            noise_floor_dbm=int(noise_floor_dbm),
            tx_backpressure_total=int(tx_backpressure_total),
            stream_fresh_total=int(stream_fresh_total),
            pacing_rx_total=int(pacing_rx_total),
            csi_fresh=bool(flags & STREAM_FLAG_CSI_FRESH),
            _iq_complex=iq_complex,
            _amplitudes=amplitudes,
            _phases=phases,
        )
        return packet, offset + record_len

    def _parse_packets(self, data: bytes) -> List[CSIPacket]:
        packets: List[CSIPacket] = []
        offset = 0
        while offset < len(data):
            packet, next_offset = self._parse_record(data, offset)
            if packet is None or next_offset <= offset:
                return []
            packets.append(packet)
            offset = next_offset
        return packets

    def _parse_packet(self, data: bytes) -> Optional[CSIPacket]:
        packets = self._parse_packets(data)
        return packets[0] if len(packets) == 1 else None

    @staticmethod
    def _compute_sequence_gap(previous_seq: int, current_seq: int) -> int:
        expected = (previous_seq + 1) & 0xFFFFFFFF
        delta = (current_seq - expected) & 0xFFFFFFFF
        if delta == 0 or delta >= 0x80000000:
            return 0
        return delta

    def _check_sequence(self, seq_num: int) -> None:
        if self.last_seq >= 0:
            self.dropped_count += self._compute_sequence_gap(self.last_seq, seq_num)
        self.last_seq = seq_num

    def _update_pps(self) -> None:
        current_time = time.time()
        if current_time - self._last_pps_time >= 1.0:
            self.pps = self._pps_counter
            self._pps_counter = 0
            self._last_pps_time = current_time

    def get_buffer_array(self) -> np.ndarray:
        if not self.buffer:
            return np.array([])
        return np.array([packet.iq_complex for packet in self.buffer])

    def get_amplitude_matrix(self) -> np.ndarray:
        if not self.buffer:
            return np.array([])
        return np.array([packet.amplitudes for packet in self.buffer])

    def get_phase_matrix(self) -> np.ndarray:
        if not self.buffer:
            return np.array([])
        return np.array([packet.phases for packet in self.buffer])

    def get_stats(self) -> Dict[str, Any]:
        elapsed = time.time() - self.start_time if self.start_time else 0
        total_expected = self.packet_count + self.dropped_count
        return {
            "packets": self.packet_count,
            "dropped": self.dropped_count,
            "drop_rate": self.dropped_count / max(total_expected, 1) * 100,
            "pps": self.pps,
            "buffer_fill": len(self.buffer),
            "buffer_size": self.buffer_size,
            "elapsed": elapsed,
        }

    def reset_stats(self) -> None:
        self.packet_count = 0
        self.dropped_count = 0
        self.last_seq = -1
        self.start_time = time.time()
        self.pps = 0
        self._pps_counter = 0
        self._last_pps_time = time.time()
        self.buffer.clear()

    def run(self, timeout: float = 0, quiet: bool = False, announce_socket_rcvbuf: bool = False) -> None:
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        if self.socket_rcvbuf_bytes > 0:
            try:
                self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, self.socket_rcvbuf_bytes)
            except OSError:
                pass
        try:
            self.effective_socket_rcvbuf_bytes = int(self.sock.getsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF))
        except OSError:
            self.effective_socket_rcvbuf_bytes = None
        self.sock.bind((self.bind_host, self.port))
        self.sock.settimeout(1.0)

        if not quiet:
            print(f"CSI Receiver listening on {self.bind_host}:{self.port}")
            print(f"Buffer size: {self.buffer_size} packets")
            if self.effective_socket_rcvbuf_bytes is not None:
                print(
                    f"Socket RCVBUF: requested {self.socket_rcvbuf_bytes} bytes, "
                    f"effective {self.effective_socket_rcvbuf_bytes} bytes"
                )
            print("Waiting for data...\n")
        elif announce_socket_rcvbuf and self.effective_socket_rcvbuf_bytes is not None:
            print(
                f"  Socket RCVBUF: requested {self.socket_rcvbuf_bytes} bytes, "
                f"effective {self.effective_socket_rcvbuf_bytes} bytes"
            )
            print()

        self.running = True
        self.start_time = time.time()
        self._last_pps_time = time.time()
        try:
            while self.running:
                if timeout > 0 and time.time() - self.start_time >= timeout:
                    break
                try:
                    data, addr = self.sock.recvfrom(MAX_STREAM_DATAGRAM_BYTES)
                except socket.timeout:
                    self._update_pps()
                    continue

                packets = self._parse_packets(data)
                if not packets:
                    continue
                for packet in packets:
                    packet.source_ip = addr[0]
                    self._check_sequence(packet.seq_num)
                    self.buffer.append(packet)
                    self.packet_count += 1
                    self._pps_counter += 1
                    self._update_pps()
                    for callback in self._callbacks:
                        try:
                            callback(packet)
                        except Exception as exc:
                            print(f"Callback error: {exc}")
                    for callback, interval in self._buffer_callbacks:
                        if self.packet_count % interval == 0:
                            try:
                                callback(self.buffer)
                            except Exception as exc:
                                print(f"Buffer callback error: {exc}")
        except KeyboardInterrupt:
            if not quiet:
                print("\nStopping receiver...")
        finally:
            self.running = False
            if self.sock:
                self.sock.close()

        if not quiet:
            stats = self.get_stats()
            print()
            print("=" * 50)
            print(f'Total packets:  {stats["packets"]}')
            print(f'Dropped:        {stats["dropped"]} ({stats["drop_rate"]:.1f}%)')
            print(f'Duration:       {stats["elapsed"]:.1f}s')
            print(f'Average PPS:    {stats["packets"] / max(stats["elapsed"], 1):.1f}')
            print("=" * 50)

    def stop(self) -> None:
        self.running = False


def get_git_username() -> Optional[str]:
    """Get GitHub username from git config."""
    try:
        result = subprocess.run(["git", "config", "user.name"], capture_output=True, text=True, timeout=2)
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip().lower().replace(" ", "")
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass
    return None


class CSICollector:
    """Collects labeled CSI data for training datasets."""

    FORMAT_VERSION = dataset_metadata.DATASET_FORMAT_VERSION
    READY_STABLE_SECONDS = 3.0
    READY_MV_THRESHOLD = 1.0
    STATUS_REFRESH_SECONDS = 0.2

    def __init__(
        self,
        label: str,
        port: int = DEFAULT_PORT,
        contributor: str = None,
        description: str = None,
        bind_host: Optional[str] = None,
        expected_device_count: Optional[int] = None,
        expected_source_hosts: Optional[List[str]] = None,
    ):
        self.label = label
        self.port = port
        self.bind_host = bind_host
        self.chip = None
        self.contributor = contributor or get_git_username()
        self.description = description
        self.expected_source_hosts = list(dict.fromkeys(expected_source_hosts or []))
        self.expected_device_count = max(1, int(expected_device_count)) if expected_device_count is not None else 1
        self.receiver = CSIReceiver(port=port, buffer_size=2000, bind_host=bind_host, derive_complex=False)
        self._sample_count = 0
        self._ready_detector = self._build_ready_detector()
        self._live_status_line_count = 0

    def _get_label_dir(self) -> Path:
        label_dir = dataset_metadata.DATA_DIR / self.label
        label_dir.mkdir(parents=True, exist_ok=True)
        return label_dir

    def _generate_filename(self, num_subcarriers: int, device_id: int) -> str:
        self._sample_count += 1
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        chip = self.chip or "unknown"
        return f"{self.label}_{chip}_{num_subcarriers}sc_{format_device_token(device_id)}_{timestamp}_{self._sample_count:04d}.npz"

    def _build_default_description(self) -> str:
        return f"HT20 {str(self.label).replace('_', ' ').strip()} sample"

    @staticmethod
    def _require_single_device_id(packets: List[CSIPacket]) -> int:
        missing_device_packets = sum(1 for packet in packets if packet.device_id is None)
        if missing_device_packets:
            raise ValueError(
                f"cannot save sample without device_id metadata ({missing_device_packets} packets missing device_id)"
            )
        device_ids = {int(packet.device_id) for packet in packets if packet.device_id is not None}
        if len(device_ids) != 1:
            raise ValueError(f"cannot save mixed-device sample as one file: found {len(device_ids)} device ids")
        return next(iter(device_ids))

    def save_samples_by_device(self, packets: List[CSIPacket]) -> List[Path]:
        if not packets:
            return []
        packets_by_device: Dict[int, List[CSIPacket]] = {}
        missing_device_packets = 0
        for packet in packets:
            if packet.device_id is None:
                missing_device_packets += 1
                continue
            packets_by_device.setdefault(int(packet.device_id), []).append(packet)
        if missing_device_packets:
            raise ValueError(
                f"cannot save capture window without device_id metadata "
                f"({missing_device_packets} packets missing device_id)"
            )
        saved_files: List[Path] = []
        for device_id in sorted(packets_by_device):
            filepath = self.save_sample(packets_by_device[device_id])
            if filepath is not None:
                saved_files.append(filepath)
        return saved_files

    def save_sample(self, packets: List[CSIPacket]) -> Optional[Path]:
        if not packets:
            return None
        device_id = self._require_single_device_id(packets)
        if packets[0].chip and packets[0].chip != "unknown":
            self.chip = packets[0].chip.lower()

        csi_data = np.array([packet.iq_raw for packet in packets], dtype=np.int8)
        timestamps = np.array([packet.timestamp for packet in packets])
        duration_ms = (timestamps[-1] - timestamps[0]) * 1000 if len(timestamps) > 1 else 0
        sample = {
            "csi_data": csi_data,
            "num_subcarriers": packets[0].num_subcarriers,
            "label": self.label,
            "chip": self.chip or "unknown",
            "collected_at": datetime.now().isoformat(),
            "duration_ms": duration_ms,
            "format_version": self.FORMAT_VERSION,
            "stream_seq_num": np.array([packet.seq_num for packet in packets], dtype=np.uint32),
            "device_id": np.uint64(device_id),
        }

        device_ticks = [packet.device_ticks_us for packet in packets]
        if all(value is not None for value in device_ticks):
            sample["device_ticks_us"] = np.array(device_ticks, dtype=np.uint64)

        def add_optional_array(key: str, values, dtype) -> None:
            if any(value is not None for value in values):
                sample[key] = np.array([0 if value is None else value for value in values], dtype=dtype)

        add_optional_array("wifi_rx_ts_us", [packet.wifi_rx_ts_us for packet in packets], np.uint32)
        add_optional_array("wifi_rx_start_ts_ns", [packet.wifi_rx_start_ts_ns for packet in packets], np.uint64)
        add_optional_array("channel", [packet.channel for packet in packets], np.uint8)
        add_optional_array("rssi_dbm", [packet.rssi_dbm for packet in packets], np.int16)
        add_optional_array("noise_floor_dbm", [packet.noise_floor_dbm for packet in packets], np.int16)

        label_dir = self._get_label_dir()
        filename = self._generate_filename(packets[0].num_subcarriers, device_id)
        filepath = label_dir / filename
        np.savez_compressed(filepath, **sample)

        self._update_dataset_info(
            filename=filename,
            num_subcarriers=packets[0].num_subcarriers,
            num_packets=len(packets),
            duration_ms=duration_ms,
            collected_at=sample["collected_at"],
            description=self.description,
            device_id=device_id,
        )
        return filepath

    def _update_dataset_info(
        self,
        filename: str = None,
        num_subcarriers: int = None,
        num_packets: int = None,
        duration_ms: float = None,
        collected_at: str = None,
        description: str = None,
        device_id: Optional[int] = None,
    ) -> None:
        info = dataset_metadata.load_dataset_info()
        if self.label not in info["labels"]:
            info["labels"][self.label] = {"description": ""}
        info["updated_at"] = datetime.now().isoformat()
        if filename and num_subcarriers:
            info.setdefault("files", {})
            info["files"].setdefault(self.label, [])
            existing_files = [entry["filename"] for entry in info["files"][self.label]]
            if filename not in existing_files:
                file_info = {
                    "filename": filename,
                    "chip": self.chip.upper() if self.chip else "unknown",
                    "subcarriers": num_subcarriers,
                    "contributor": self.contributor or "",
                    "collected_at": collected_at or "",
                    "duration_ms": int(duration_ms) if duration_ms else 0,
                    "num_packets": num_packets or 0,
                    "description": description or self._build_default_description(),
                    "device_id": format_device_id_hex(device_id) if device_id is not None else "",
                }
                info["files"][self.label].append(file_info)
        dataset_metadata.save_dataset_info(info)

    def _drain_udp_backlog(self, max_packets: int = 10000) -> int:
        if self.receiver.sock is None:
            return 0
        drained = 0
        previous_timeout = self.receiver.sock.gettimeout()
        self.receiver.sock.settimeout(0.0)
        try:
            while drained < max_packets:
                try:
                    self.receiver.sock.recvfrom(MAX_STREAM_DATAGRAM_BYTES)
                    drained += 1
                except (BlockingIOError, socket.timeout):
                    break
        finally:
            self.receiver.sock.settimeout(previous_timeout)
        return drained

    def _build_ready_detector(self) -> VarianceDetectorAdapter:
        window_size = int(getattr(config, "SEG_WINDOW_SIZE", 100))
        if window_size < 10:
            window_size = 10
        elif window_size > 200:
            window_size = 200
        return VarianceDetectorAdapter(window_size=window_size, threshold=self.READY_MV_THRESHOLD, track_data=False)

    def _reset_live_status_block(self) -> None:
        self._live_status_line_count = 0

    def _render_live_status_block(self, summary_line: str, detail_lines: List[str], *, inline: Optional[bool] = None) -> None:
        self._live_status_line_count = self._emit_ready_status_block(
            summary_line,
            detail_lines,
            previous_line_count=self._live_status_line_count,
            inline=inline,
        )

    @staticmethod
    def _build_status_bar(ratio: float, width: int = 18) -> str:
        clamped = max(0.0, min(1.0, ratio))
        filled = int(round(clamped * width))
        return "[" + ("#" * filled) + ("-" * (width - filled)) + "]"

    @staticmethod
    def _format_backpressure_text(device_state: Dict[str, Any]) -> str:
        total = device_state.get("tx_backpressure_total")
        if total is None:
            return " | bp:--"
        recent_delta = int(device_state.get("tx_backpressure_last_delta", 0) or 0)
        if recent_delta > 0:
            return f" | bp:active(+{recent_delta})"
        return " | bp:no"

    @staticmethod
    def _supports_inline_terminal(stream: Any = None) -> bool:
        target_stream = sys.stdout if stream is None else stream
        isatty = getattr(target_stream, "isatty", None)
        return bool(callable(isatty) and isatty())

    @staticmethod
    def _emit_ready_status_block(
        summary_line: str,
        detail_lines: List[str],
        *,
        previous_line_count: int = 0,
        stream: Any = None,
        inline: Optional[bool] = None,
    ) -> int:
        target_stream = sys.stdout if stream is None else stream
        use_inline = CSICollector._supports_inline_terminal(target_stream) if inline is None else inline
        lines = [summary_line, *detail_lines]
        if not use_inline:
            for line in lines:
                target_stream.write(f"{line}\n")
            target_stream.flush()
            return len(lines)

        if previous_line_count > 0:
            target_stream.write(f"\x1b[{previous_line_count}F")
        total_lines = max(previous_line_count, len(lines))
        for idx in range(total_lines):
            target_stream.write("\x1b[2K")
            if idx < len(lines):
                target_stream.write(lines[idx])
            target_stream.write("\n")
        target_stream.flush()
        return len(lines)

    @staticmethod
    def _check_sequence_by_device(packet: CSIPacket, last_seq_by_device: Dict[int, int]) -> int:
        if packet.device_id is None:
            return 0
        device_id = int(packet.device_id)
        dropped = 0
        if device_id in last_seq_by_device:
            dropped = CSIReceiver._compute_sequence_gap(last_seq_by_device[device_id], packet.seq_num)
        last_seq_by_device[device_id] = packet.seq_num
        return dropped

    @staticmethod
    def _summarize_ready_devices(
        device_states: Dict[int, Dict[str, Any]],
        *,
        expected_device_count: int,
        warmup_target: int,
        threshold: float,
        now: float,
    ) -> Dict[str, Any]:
        observed_count = len(device_states)
        required_count = max(1, expected_device_count)
        relevant_states = list(device_states.values())
        if observed_count < required_count:
            return {
                "ready": False,
                "status": f"DEVICES {observed_count}/{required_count}",
                "stable_elapsed": 0.0,
                "ready_count": 0,
                "observed_count": observed_count,
                "required_count": required_count,
            }
        warm_states = [state for state in relevant_states if state["processed_packets"] >= warmup_target]
        total_relevant = max(observed_count, required_count)
        if len(warm_states) < observed_count:
            return {
                "ready": False,
                "status": f"WARMUP {len(warm_states)}/{total_relevant}",
                "stable_elapsed": 0.0,
                "ready_count": 0,
                "observed_count": observed_count,
                "required_count": required_count,
            }
        if any(state["current_mv"] > threshold for state in relevant_states):
            ready_count = sum(1 for state in relevant_states if state["current_mv"] <= threshold)
            return {
                "ready": False,
                "status": f"UNSTABLE {ready_count}/{total_relevant}",
                "stable_elapsed": 0.0,
                "ready_count": ready_count,
                "observed_count": observed_count,
                "required_count": required_count,
            }
        stable_elapsed = min(
            max(0.0, now - state["stable_since"]) if state["stable_since"] is not None else 0.0
            for state in relevant_states
        )
        ready = stable_elapsed >= CSICollector.READY_STABLE_SECONDS
        return {
            "ready": ready,
            "status": f"READY {observed_count}/{total_relevant}" if ready else f"STABLE {observed_count}/{total_relevant}",
            "stable_elapsed": stable_elapsed,
            "ready_count": observed_count,
            "observed_count": observed_count,
            "required_count": required_count,
        }

    @staticmethod
    def _format_ready_device_lines(
        device_states: Dict[int, Dict[str, Any]],
        *,
        expected_source_hosts: List[str],
        warmup_target: int,
        threshold: float,
        now: float,
    ) -> List[str]:
        lines: List[str] = []
        seen_ips = {state.get("source_ip") for state in device_states.values() if state.get("source_ip")}
        for expected_ip in expected_source_hosts:
            if expected_ip not in seen_ips:
                lines.append(
                    f"    ip={expected_ip} chip=? ch=-- rssi=--- "
                    f"{CSICollector._build_status_bar(0.0)} "
                    f"mv=--/{threshold:.3f} pps=-- | WAITING | bp:--"
                )
        for device_id in sorted(device_states):
            state = device_states[device_id]
            processed_packets = int(state.get("processed_packets", 0))
            current_mv = float(state.get("current_mv", 0.0))
            stable_since = state.get("stable_since")
            if processed_packets < warmup_target:
                status = f"WARMUP {processed_packets}/{warmup_target}"
                mv_ratio = 0.0
            else:
                stable_value = max(0.0, now - stable_since) if stable_since is not None else 0.0
                mv_ratio = min(current_mv / threshold, 1.0) if threshold > 0 else 0.0
                if current_mv > threshold:
                    status = "UNSTABLE"
                elif stable_value >= CSICollector.READY_STABLE_SECONDS:
                    status = "READY"
                else:
                    status = "STABLE"
            mv_text = "--" if processed_packets < warmup_target else f"{current_mv:.3f}"
            channel = state.get("channel")
            rssi_dbm = state.get("rssi_dbm")
            current_pps = state.get("current_pps")
            lines.append(
                f"    ip={state.get('source_ip', '?')} chip={str(state.get('chip', '?')).upper()} "
                f"ch={'--' if channel is None else f'{int(channel):02d}'} "
                f"rssi={'---' if rssi_dbm is None else str(int(rssi_dbm))} "
                f"{CSICollector._build_status_bar(mv_ratio)} "
                f"mv={mv_text}/{threshold:.3f} pps={'--' if current_pps is None else str(int(current_pps))} "
                f"| {status}{CSICollector._format_backpressure_text(state)}"
            )
        return lines

    def _wait_for_ready_state(self, quiet: bool = False, summary_prefix: str = "  ") -> Dict[int, Dict[str, Any]]:
        return self._wait_for_ready_state_with_pacing(quiet=quiet, summary_prefix=summary_prefix)

    def _wait_for_ready_state_with_pacing(
        self,
        quiet: bool = False,
        summary_prefix: str = "  ",
        *,
        pacing_controller: Optional[AdaptivePacingController] = None,
        pacing_sender: Any = None,
    ) -> Dict[int, Dict[str, Any]]:
        if self.receiver.sock is None:
            raise RuntimeError("Receiver socket is not initialized")
        self.receiver.reset_stats()
        warmup_target = self._ready_detector.window_size
        device_states: Dict[int, Dict[str, Any]] = {}
        last_seq_by_device: Dict[int, int] = {}
        processed_packets = 0
        last_render = 0.0
        last_pps_time = time.monotonic()
        last_pps_count = 0
        current_pps = 0
        current_state = f"DEVICES 0/{self.expected_device_count}"
        use_inline_status = self._supports_inline_terminal()
        while True:
            try:
                data, addr = self.receiver.sock.recvfrom(MAX_STREAM_DATAGRAM_BYTES)
                packets = self.receiver._parse_packets(data)
                if not packets:
                    continue
                for packet in packets:
                    processed_packets += 1
                    self.receiver.packet_count += 1
                    self.receiver.dropped_count += self._check_sequence_by_device(packet, last_seq_by_device)
                    if packet.device_id is None:
                        continue
                    device_id = int(packet.device_id)
                    state = device_states.get(device_id)
                    if state is None:
                        state = {
                            "detector": self._build_ready_detector(),
                            "processed_packets": 0,
                            "stable_since": None,
                            "current_mv": 0.0,
                            "current_pps": 0,
                            "last_pps_count": 0,
                            "source_ip": addr[0],
                            "chip": packet.chip or "unknown",
                            "channel": packet.channel,
                            "rssi_dbm": packet.rssi_dbm,
                            "last_seq": packet.seq_num,
                            "tx_backpressure_total": None,
                            "tx_backpressure_window_delta": 0,
                            "tx_backpressure_last_delta": 0,
                            "stream_fresh_total": None,
                            "stream_fresh_window_delta": 0,
                            "stream_fresh_last_delta": 0,
                            "pacing_rx_total": None,
                            "pacing_rx_window_delta": 0,
                            "pacing_rx_last_delta": 0,
                        }
                        device_states[device_id] = state
                    else:
                        state["source_ip"] = addr[0]
                        if packet.chip and packet.chip != "unknown":
                            state["chip"] = packet.chip
                        if packet.channel is not None:
                            state["channel"] = packet.channel
                        if packet.rssi_dbm is not None:
                            state["rssi_dbm"] = packet.rssi_dbm
                        state["last_seq"] = packet.seq_num
                    if pacing_controller is not None:
                        pacing_controller.observe_device(
                            state,
                            getattr(packet, "tx_backpressure_total", None),
                            getattr(packet, "stream_fresh_total", None),
                            getattr(packet, "pacing_rx_total", None),
                        )
                    state["detector"].process_packet({"csi_data": packet.iq_raw})
                    state["processed_packets"] += 1
                    if state["processed_packets"] >= warmup_target:
                        state["current_mv"] = state["detector"]._context.current_moving_variance
                        now = time.monotonic()
                        if state["current_mv"] <= self.READY_MV_THRESHOLD:
                            if state["stable_since"] is None:
                                state["stable_since"] = now
                        else:
                            state["stable_since"] = None

                now = time.monotonic()
                if pacing_controller is not None:
                    pacing_controller.maybe_adjust(device_states, now=now, pacing_sender=pacing_sender)
                summary = self._summarize_ready_devices(
                    device_states,
                    expected_device_count=self.expected_device_count,
                    warmup_target=warmup_target,
                    threshold=self.READY_MV_THRESHOLD,
                    now=now,
                )
                current_state = summary["status"]
                if summary["ready"]:
                    if not quiet:
                        self._render_live_status_block(
                            (
                                f'{summary_prefix}STATUS: READY {summary["observed_count"]}/{summary["required_count"]} '
                                f'| pps {current_pps} | drop {self.receiver.get_stats()["drop_rate"]:.1f}% '
                            ),
                            self._format_ready_device_lines(
                                device_states,
                                expected_source_hosts=self.expected_source_hosts,
                                warmup_target=warmup_target,
                                threshold=self.READY_MV_THRESHOLD,
                                now=now,
                            ),
                            inline=use_inline_status,
                        )
                    return device_states
                if now - last_pps_time >= 1.0:
                    delta = processed_packets - last_pps_count
                    elapsed = now - last_pps_time
                    current_pps = int(delta / elapsed) if elapsed > 0 else 0
                    for state in device_states.values():
                        device_delta = int(state.get("processed_packets", 0)) - int(state.get("last_pps_count", 0))
                        state["current_pps"] = int(device_delta / elapsed) if elapsed > 0 else 0
                        state["last_pps_count"] = int(state.get("processed_packets", 0))
                    last_pps_time = now
                    last_pps_count = processed_packets
                if (not quiet) and (now - last_render >= self.STATUS_REFRESH_SECONDS):
                    self._render_live_status_block(
                        f"{summary_prefix}STATUS: {current_state} | pps {current_pps} | drop {self.receiver.get_stats()['drop_rate']:.1f}% ",
                        self._format_ready_device_lines(
                            device_states,
                            expected_source_hosts=self.expected_source_hosts,
                            warmup_target=warmup_target,
                            threshold=self.READY_MV_THRESHOLD,
                            now=now,
                        ),
                        inline=use_inline_status,
                    )
                    last_render = now
            except socket.timeout:
                now = time.monotonic()
                if (not quiet) and (now - last_render >= self.STATUS_REFRESH_SECONDS):
                    self._render_live_status_block(
                        f"{summary_prefix}STATUS: NO DATA | pps 0 | drop 0.0%",
                        self._format_ready_device_lines(
                            device_states,
                            expected_source_hosts=self.expected_source_hosts,
                            warmup_target=warmup_target,
                            threshold=self.READY_MV_THRESHOLD,
                            now=now,
                        ),
                        inline=use_inline_status,
                    )
                    last_render = now

    def _collect_with_live_status(
        self,
        duration: float,
        *,
        quiet: bool = False,
        initial_device_states: Optional[Dict[int, Dict[str, Any]]] = None,
        summary_prefix: str = "  ",
        pacing_controller: Optional[AdaptivePacingController] = None,
        pacing_sender: Any = None,
    ) -> List[CSIPacket]:
        if self.receiver.sock is None:
            raise RuntimeError("Receiver socket is not initialized")
        self.receiver.reset_stats()
        packets: List[CSIPacket] = []
        deadline = time.monotonic() + duration
        warmup_target = self._ready_detector.window_size
        device_states: Dict[int, Dict[str, Any]] = dict(initial_device_states or {})
        last_seq_by_device: Dict[int, int] = {
            device_id: int(state["last_seq"])
            for device_id, state in device_states.items()
            if state.get("last_seq") is not None
        }
        processed_packets = 0
        last_render = 0.0
        last_pps_time = time.monotonic()
        last_pps_count = 0
        current_pps = 0
        use_inline_status = self._supports_inline_terminal()
        while time.monotonic() < deadline:
            try:
                data, addr = self.receiver.sock.recvfrom(MAX_STREAM_DATAGRAM_BYTES)
                parsed_packets = self.receiver._parse_packets(data)
                if not parsed_packets:
                    continue
                for packet in parsed_packets:
                    packets.append(packet)
                    processed_packets += 1
                    self.receiver.packet_count += 1
                    self.receiver.dropped_count += self._check_sequence_by_device(packet, last_seq_by_device)
                    if packet.device_id is None:
                        continue
                    device_id = int(packet.device_id)
                    state = device_states.get(device_id)
                    if state is None:
                        state = {
                            "detector": self._build_ready_detector(),
                            "processed_packets": 0,
                            "stable_since": None,
                            "current_mv": 0.0,
                            "current_pps": 0,
                            "last_pps_count": 0,
                            "source_ip": addr[0],
                            "chip": packet.chip or "unknown",
                            "channel": packet.channel,
                            "rssi_dbm": packet.rssi_dbm,
                            "last_seq": packet.seq_num,
                            "tx_backpressure_total": None,
                            "tx_backpressure_window_delta": 0,
                            "tx_backpressure_last_delta": 0,
                            "stream_fresh_total": None,
                            "stream_fresh_window_delta": 0,
                            "stream_fresh_last_delta": 0,
                            "pacing_rx_total": None,
                            "pacing_rx_window_delta": 0,
                            "pacing_rx_last_delta": 0,
                        }
                        device_states[device_id] = state
                    else:
                        state["source_ip"] = addr[0]
                        if packet.chip and packet.chip != "unknown":
                            state["chip"] = packet.chip
                        if packet.channel is not None:
                            state["channel"] = packet.channel
                        if packet.rssi_dbm is not None:
                            state["rssi_dbm"] = packet.rssi_dbm
                        state["last_seq"] = packet.seq_num
                    if pacing_controller is not None:
                        pacing_controller.observe_device(
                            state,
                            getattr(packet, "tx_backpressure_total", None),
                            getattr(packet, "stream_fresh_total", None),
                            getattr(packet, "pacing_rx_total", None),
                        )
                    state["detector"].process_packet({"csi_data": packet.iq_raw})
                    state["processed_packets"] += 1
                    if state["processed_packets"] >= warmup_target:
                        state["current_mv"] = state["detector"]._context.current_moving_variance
                        now = time.monotonic()
                        if state["current_mv"] <= self.READY_MV_THRESHOLD:
                            if state["stable_since"] is None:
                                state["stable_since"] = now
                        else:
                            state["stable_since"] = None
                now = time.monotonic()
                if pacing_controller is not None:
                    pacing_controller.maybe_adjust(device_states, now=now, pacing_sender=pacing_sender)
                if now - last_pps_time >= 1.0:
                    delta = processed_packets - last_pps_count
                    elapsed = now - last_pps_time
                    current_pps = int(delta / elapsed) if elapsed > 0 else 0
                    for state in device_states.values():
                        device_delta = int(state.get("processed_packets", 0)) - int(state.get("last_pps_count", 0))
                        state["current_pps"] = int(device_delta / elapsed) if elapsed > 0 else 0
                        state["last_pps_count"] = int(state.get("processed_packets", 0))
                    last_pps_time = now
                    last_pps_count = processed_packets
                if (not quiet) and (now - last_render >= self.STATUS_REFRESH_SECONDS):
                    elapsed = max(0.0, duration - max(0.0, deadline - now))
                    self._render_live_status_block(
                        f"{summary_prefix}STATUS: RECORDING | elapsed {elapsed:.1f}/{duration:.1f}s | pps {current_pps} | drop {self.receiver.get_stats()['drop_rate']:.1f}% | packets {len(packets)}",
                        self._format_ready_device_lines(
                            device_states,
                            expected_source_hosts=self.expected_source_hosts,
                            warmup_target=warmup_target,
                            threshold=self.READY_MV_THRESHOLD,
                            now=now,
                        ),
                        inline=use_inline_status,
                    )
                    last_render = now
            except socket.timeout:
                now = time.monotonic()
                if (not quiet) and (now - last_render >= self.STATUS_REFRESH_SECONDS):
                    elapsed = max(0.0, duration - max(0.0, deadline - now))
                    self._render_live_status_block(
                        f"{summary_prefix}STATUS: RECORDING | elapsed {elapsed:.1f}/{duration:.1f}s | pps {current_pps} | drop {self.receiver.get_stats()['drop_rate']:.1f}% | packets {len(packets)}",
                        self._format_ready_device_lines(
                            device_states,
                            expected_source_hosts=self.expected_source_hosts,
                            warmup_target=warmup_target,
                            threshold=self.READY_MV_THRESHOLD,
                            now=now,
                        ),
                        inline=use_inline_status,
                    )
                    last_render = now
        return packets

    def collect_timed(
        self,
        duration: float,
        num_samples: int = 1,
        quiet: bool = False,
        *,
        pacing_sender: Any = None,
        adaptive: bool = False,
    ) -> List[Path]:
        saved_files: List[Path] = []
        initial_pacing_pps = 0.0
        if pacing_sender is not None and hasattr(pacing_sender, "get_rate_pps"):
            try:
                initial_pacing_pps = float(pacing_sender.get_rate_pps())
            except (TypeError, ValueError):
                initial_pacing_pps = 0.0
        pacing_controller = AdaptivePacingController(initial_pps=max(1.0, initial_pacing_pps), enabled=adaptive)
        if not quiet:
            print(f'\n{"=" * 60}')
            print(f"  CSI Data Collection: {self.label}")
            print(f'{"=" * 60}')
            print(f"  Duration per sample: {duration}s")
            print(f"  Samples to collect:  {num_samples}")
            print(f"  Ready gate:          implicit ({self.READY_STABLE_SECONDS:.1f}s stable)")
            print(f"  Pacing mode:         {'adaptive' if adaptive else 'fixed'}")
            print(f'{"=" * 60}\n')
        self.receiver.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.receiver.sock.bind((self.receiver.bind_host, self.port))
        self.receiver.sock.settimeout(0.1)
        try:
            for sample_idx in range(num_samples):
                self._reset_live_status_block()
                summary_prefix = f"  Sample {sample_idx + 1}/{num_samples} | "
                self._drain_udp_backlog()
                ready_device_states = self._wait_for_ready_state_with_pacing(
                    quiet=quiet,
                    summary_prefix=summary_prefix,
                    pacing_controller=pacing_controller,
                    pacing_sender=pacing_sender,
                )
                packets = self._collect_with_live_status(
                    duration,
                    quiet=quiet,
                    initial_device_states=ready_device_states,
                    summary_prefix=summary_prefix,
                    pacing_controller=pacing_controller,
                    pacing_sender=pacing_sender,
                )
                sample_files = self.save_samples_by_device(packets)
                if sample_files:
                    saved_files.extend(sample_files)
                    if not quiet:
                        print(f"\r  ✅ Saved {len(sample_files)} device file(s) from {len(packets)} packets")
                        for filepath in sample_files:
                            print(f"     - {filepath.name}")
                elif not quiet:
                    print("\r  ❌ No packets received!")
                self._reset_live_status_block()
        finally:
            if self.receiver.sock:
                self.receiver.sock.close()
        if not quiet:
            print(f'\n{"=" * 60}')
            print(f"  Collection complete: {len(saved_files)} device file(s) saved")
            print(f'{"=" * 60}\n')
        return saved_files

    def collect_interactive(self, num_samples: int = 10, duration: float = 2.0) -> List[Path]:
        saved_files: List[Path] = []
        print(f'\n{"=" * 60}')
        print(f"  CSI Data Collection: {self.label}")
        print(f'{"=" * 60}')
        print(f"  Target samples: {num_samples}")
        print("  Press ENTER to record each sample, Q to quit")
        print(f'{"=" * 60}\n')
        self.receiver.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.receiver.sock.bind((self.receiver.bind_host, self.port))
        self.receiver.sock.settimeout(0.1)
        try:
            sample_idx = 0
            while sample_idx < num_samples:
                try:
                    self._reset_live_status_block()
                    user_input = input(f"\nSample {sample_idx + 1}/{num_samples} - Press ENTER to record (Q to quit): ")
                    if user_input.lower() == "q":
                        print("Collection cancelled.")
                        break
                    print(f"  Recording for {duration} seconds...", end="", flush=True)
                    self._drain_udp_backlog()
                    ready_device_states = self._wait_for_ready_state(quiet=False)
                    packets = self._collect_with_live_status(duration, quiet=False, initial_device_states=ready_device_states)
                    sample_files = self.save_samples_by_device(packets)
                    if sample_files:
                        saved_files.extend(sample_files)
                        print(f"\r  ✅ Saved {len(sample_files)} device file(s) from {len(packets)} packets")
                        for filepath in sample_files:
                            print(f"     - {filepath.name}")
                        sample_idx += 1
                    else:
                        print("\r  ❌ No packets received! Check the streamer firmware and collector IP/port.")
                    self._reset_live_status_block()
                except KeyboardInterrupt:
                    print("\nCollection cancelled.")
                    break
        finally:
            if self.receiver.sock:
                self.receiver.sock.close()
        print(f'\n{"=" * 60}')
        print(f"  Collection complete: {len(saved_files)} device file(s) saved")
        print(f'{"=" * 60}\n')
        return saved_files


def load_npz_as_packets(filepath: Path) -> List[Dict[str, Any]]:
    """Load a ``.npz`` file and convert it to packet dictionaries."""
    data = np.load(filepath, allow_pickle=True)
    if "csi_data" not in data.files:
        raise ValueError(f"No CSI data found in {filepath}")
    csi_array = data["csi_data"]

    label = str(data.get("label", "unknown"))
    num_subcarriers = int(data.get("num_subcarriers", csi_array.shape[1] // 2))
    chip = str(data.get("chip", "unknown"))
    stream_seq_nums = data["stream_seq_num"] if "stream_seq_num" in data.files else None
    device_ticks_us = data["device_ticks_us"] if "device_ticks_us" in data.files else None
    wifi_rx_ts_us = data["wifi_rx_ts_us"] if "wifi_rx_ts_us" in data.files else None
    wifi_rx_start_ts_ns = data["wifi_rx_start_ts_ns"] if "wifi_rx_start_ts_ns" in data.files else None
    device_ids = data["device_id"] if "device_id" in data.files else None
    channels = data["channel"] if "channel" in data.files else None
    rssi_dbm = data["rssi_dbm"] if "rssi_dbm" in data.files else None
    noise_floor_dbm = data["noise_floor_dbm"] if "noise_floor_dbm" in data.files else None

    def optional_scalar(array, index, cast):
        if array is None:
            return None
        if np.ndim(array) == 0:
            return cast(np.asarray(array).item())
        if index >= len(array):
            return None
        return cast(array[index])

    packets = []
    for index in range(len(csi_array)):
        packets.append(
            {
                "csi_data": np.array(csi_array[index], dtype=np.int8),
                "label": label,
                "num_subcarriers": num_subcarriers,
                "chip": chip,
                "stream_seq_num": optional_scalar(stream_seq_nums, index, int),
                "device_ticks_us": optional_scalar(device_ticks_us, index, int),
                "wifi_rx_ts_us": optional_scalar(wifi_rx_ts_us, index, int),
                "wifi_rx_start_ts_ns": optional_scalar(wifi_rx_start_ts_ns, index, int),
                "device_id": optional_scalar(device_ids, index, int),
                "channel": optional_scalar(channels, index, int),
                "rssi_dbm": optional_scalar(rssi_dbm, index, int),
                "noise_floor_dbm": optional_scalar(noise_floor_dbm, index, int),
            }
        )
    return packets


def get_dataset_stats() -> Dict[str, Any]:
    """Proxy dataset statistics through the dataset metadata layer."""
    return dataset_metadata.get_dataset_stats()


def find_static_presence_motion_dataset(
    chip: str = None,
    num_sc: int = 64,
    dataset: str = None,
    prefer_latest: bool = True,
) -> Tuple[Path, Path, str]:
    """Find an explicit static-presence/motion dataset pair from metadata."""
    pair = dataset_metadata.resolve_explicit_pair(
        dataset=dataset,
        chip=chip,
        num_sc=num_sc,
        prefer_latest=prefer_latest,
    )
    return pair.static_presence.path, pair.motion.path, pair.chip


def load_static_presence_and_motion(
    static_presence_file: str = None,
    motion_file: str = None,
    chip: str = "C6",
    dataset: str = None,
    prefer_latest: bool = True,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Load static-presence and motion data from ``.npz`` files."""
    if static_presence_file is None or motion_file is None:
        found_static_presence, found_motion, _ = find_static_presence_motion_dataset(
            chip=chip,
            dataset=dataset,
            prefer_latest=prefer_latest,
        )
        static_presence_file = static_presence_file or found_static_presence
        motion_file = motion_file or found_motion

    static_presence_path = Path(static_presence_file) if isinstance(static_presence_file, str) else static_presence_file
    motion_path = Path(motion_file) if isinstance(motion_file, str) else motion_file
    if not static_presence_path.exists():
        raise FileNotFoundError(
            f"{static_presence_path} not found.\n"
            "Collect data using: ./espectre collect --label static_presence --duration 10"
        )
    if not motion_path.exists():
        raise FileNotFoundError(
            f"{motion_path} not found.\n"
            "Collect data using: ./espectre collect --label motion --duration 10"
        )
    return load_npz_as_packets(static_presence_path), load_npz_as_packets(motion_path)

