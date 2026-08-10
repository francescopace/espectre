"""
ESPectre - CSI I/O

CSI stream I/O, collection, and dataset loading helpers for tooling.

Author: Francesco Pace <francesco.pace@gmail.com>
License: GPLv3
"""

from __future__ import annotations

import ipaddress
import math
import socket
import struct
import subprocess
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from types import MappingProxyType
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Tuple

import numpy as np

from .bootstrap import setup_paths
from . import dataset_metadata
from . import npz_cache

setup_paths()

try:
    import config
except ImportError:
    import src.config as config

try:
    from serial_sequence import SerialSequenceTracker
except ImportError:
    from src.serial_sequence import SerialSequenceTracker

try:
    from device_utils import (
        HT20_CENTERED_ONLY_NULL_BINS,
        HT20_CLASSIC_ONLY_NULL_BINS,
        HT20_CSI_LEN,
        HT20_NUM_SUBCARRIERS,
    )
except ImportError:
    from src.device_utils import (
        HT20_CENTERED_ONLY_NULL_BINS,
        HT20_CLASSIC_ONLY_NULL_BINS,
        HT20_CSI_LEN,
        HT20_NUM_SUBCARRIERS,
    )

try:
    from detector_interface import (
        detector_needs_startup_calibration,
        load_detector_class,
        normalize_detector_algorithm,
    )
    from ml_detector import ML_DEFAULT_THRESHOLD
    from runtime_policy import (
        PacketTimingTracker,
        derive_detector_timing,
        duration_packet_count,
        make_evaluation_cadence,
        nominal_packet_interval_us,
    )
    from threshold import (
        StartupThresholdCalibrator,
        get_detector_auto_factor,
        get_detector_startup_gate,
    )
except ImportError:
    from src.detector_interface import (
        detector_needs_startup_calibration,
        load_detector_class,
        normalize_detector_algorithm,
    )
    from src.ml_detector import ML_DEFAULT_THRESHOLD
    from src.runtime_policy import (
        PacketTimingTracker,
        derive_detector_timing,
        duration_packet_count,
        make_evaluation_cadence,
        nominal_packet_interval_us,
    )
    from src.threshold import (
        StartupThresholdCalibrator,
        get_detector_auto_factor,
        get_detector_startup_gate,
    )

MAGIC_STREAM = 0x4353
STREAM_VERSION = 7
DEFAULT_PORT = 5001
STREAM_FLAG_FIRST_WORD_INVALID = 1 << 0
STREAM_FLAG_WIFI_RX_TS_VALID = 1 << 1
STREAM_FLAG_WIFI_RX_START_TS_NS_VALID = 1 << 2
STREAM_FLAG_CSI_FRESH = 1 << 3
# Firmware older than the capability-based noise-floor read reported this
# sentinel on targets it did not cover. It is far below thermal noise for a
# 20 MHz channel, so it can never be a genuine measurement.
NOISE_FLOOR_INVALID_DBM = -128
CSI_HEADER_FORMAT = "<HBBBBIHHQQIQBbbQIIBBB"
CSI_HEADER_STRUCT = struct.Struct(CSI_HEADER_FORMAT)
MAX_STREAM_DATAGRAM_BYTES = 2048
DEFAULT_SOCKET_RCVBUF_BYTES = 1024 * 1024
DEFAULT_PACING_PORT = 9999
DEFAULT_PACING_INTERVAL_SECONDS = 5.0


def _valid_noise_floor(value: Optional[int]) -> Optional[int]:
    """Return the noise floor, or None when it carries the invalid sentinel."""
    return None if value is None or int(value) == NOISE_FLOOR_INVALID_DBM else int(value)
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
PHY_MODE_CODES = {
    0: "unknown",
    1: "legacy",
    2: "ht",
    3: "vht",
    4: "he-su",
    5: "he-mu",
    6: "he-ersu",
    7: "he-tb",
}
LTF_TYPE_CODES = {
    0: "unknown",
    2: "ht-ltf",
    3: "vht-ltf",
    4: "he-ltf",
}
CHANNEL_WIDTH_CODES = {
    0: "unknown",
    1: "20",
    2: "40",
    3: "80",
    4: "160",
    5: "80+80",
}


def _raise_unsafe_npz_object_array(filepath: Path, key: str | None = None, exc: Exception | None = None):
    """Raise a consistent error for pickle-backed object arrays in dataset NPZs."""
    detail = (
        f"field {key!r} contains a pickle-backed object array"
        if key is not None
        else "contains a pickle-backed object array"
    )
    error = ValueError(
        f"Unsafe NPZ dataset {filepath}: {detail}; only numeric and string arrays are supported"
    )
    if exc is not None:
        raise error from exc
    raise error


def normalize_stored_csi_bin_layout(csi: np.ndarray) -> np.ndarray:
    """Rotate stored classic-order HT20 rows into the centered convention.

    Captures taken before the firmware rotated classic-MAC payloads kept
    Espressif's native ``0~31, -32~-1`` order, while Wi-Fi 6 captures were
    already centered on DC. ``DEFAULT_SUBCARRIERS`` assumes the centered
    convention, so correct the stored rows on load and leave the files alone.

    Identification requires positive evidence in both directions: one guard set
    entirely null across the recording and the other populated in every one of
    its bins. Absence of energy alone is not enough, because sparse or synthetic
    rows are null under both conventions. Recordings whose layout cannot be
    identified are returned untouched.
    """
    if csi.ndim != 2 or csi.shape[1] != HT20_CSI_LEN or csi.shape[0] == 0:
        return csi

    bins = csi.reshape(csi.shape[0], HT20_NUM_SUBCARRIERS, 2)
    classic_populated = bins[:, list(HT20_CLASSIC_ONLY_NULL_BINS), :].any(axis=(0, 2))
    centered_populated = bins[:, list(HT20_CENTERED_ONLY_NULL_BINS), :].any(axis=(0, 2))
    if not classic_populated.any() and centered_populated.all():
        return np.roll(csi, HT20_CSI_LEN // 2, axis=1)
    return csi


def _load_npz_arrays_uncached(filepath: Path) -> Dict[str, np.ndarray]:
    """Materialize a dataset NPZ without enabling pickle-backed object arrays.

    CSI rows are returned in the centered bin convention regardless of which
    ordering the capturing chip used; see ``normalize_stored_csi_bin_layout``.
    """
    arrays: Dict[str, np.ndarray] = {}
    with np.load(filepath, allow_pickle=False) as data:
        for key in data.files:
            try:
                value = np.asarray(data[key])
            except ValueError as exc:
                if "Object arrays cannot be loaded when allow_pickle=False" in str(exc):
                    _raise_unsafe_npz_object_array(filepath, key, exc)
                raise
            if value.dtype.kind == "O":
                _raise_unsafe_npz_object_array(filepath, key)
            arrays[key] = value
    for csi_key in ("csi_data", "csi"):
        if csi_key in arrays:
            arrays[csi_key] = normalize_stored_csi_bin_layout(arrays[csi_key])
    return arrays


def _freeze_npz_mapping(arrays: Dict[str, Any]) -> Mapping[str, Any]:
    """Return one immutable, read-only view of cached NPZ arrays."""
    frozen: Dict[str, Any] = {}
    for key, value in arrays.items():
        if isinstance(value, np.ndarray):
            value.setflags(write=False)
        frozen[key] = value
    return MappingProxyType(frozen)


def _freeze_packet_sequence(
    packets: tuple[Dict[str, Any], ...],
) -> tuple[Mapping[str, Any], ...]:
    """Return one immutable packet-view cache value."""
    frozen_packets = []
    for packet in packets:
        frozen_packet: Dict[str, Any] = {}
        for key, value in packet.items():
            if isinstance(value, np.ndarray):
                value.setflags(write=False)
            frozen_packet[key] = value
        frozen_packets.append(MappingProxyType(frozen_packet))
    return tuple(frozen_packets)


def load_npz_arrays(filepath: Path) -> Mapping[str, Any]:
    """Load one dataset NPZ through the shared in-process array cache."""
    return npz_cache.get_runtime_artifact(
        filepath,
        artifact_name="raw_arrays",
        artifact_version=1,
        builder=lambda: _freeze_npz_mapping(_load_npz_arrays_uncached(filepath)),
    )


def load_npz_sensing_arrays(filepath: Path) -> Mapping[str, Any]:
    """Load the sensing-only NPZ view through the shared in-process cache."""
    return npz_cache.get_runtime_artifact(
        filepath,
        artifact_name="sensing_arrays",
        artifact_version=1,
        builder=lambda: _freeze_npz_mapping(
            filter_npz_arrays_sensing(_load_npz_arrays_uncached(filepath))
        ),
    )


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
    phy_mode: str = "unknown"
    ltf_type: str = "unknown"
    channel_width: str = "unknown"
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


@dataclass(frozen=True)
class SequenceObservation:
    """Result of checking one packet against a wrapping stream sequence."""

    accepted: bool
    missing: int = 0


class PacketSequenceTracker:
    """Track independent wrapping 32-bit packet streams."""

    def __init__(self):
        self._trackers: Dict[Tuple[str, Any], SerialSequenceTracker] = {}

    @staticmethod
    def _stream_key(packet: CSIPacket) -> Tuple[str, Any]:
        if packet.device_id is not None:
            return ("device", int(packet.device_id))
        if packet.source_ip:
            return ("source", str(packet.source_ip))
        return ("receiver", 0)

    def observe(self, packet: CSIPacket) -> SequenceObservation:
        key = self._stream_key(packet)
        tracker = self._trackers.get(key)
        if tracker is None:
            tracker = SerialSequenceTracker()
            self._trackers[key] = tracker
        missing = tracker.observe(packet.seq_num)
        if missing < 0:
            return SequenceObservation(False)
        return SequenceObservation(True, missing=missing)

    def seed_device(self, device_id: int, seq_num: int) -> None:
        key = ("device", int(device_id))
        tracker = self._trackers.get(key)
        if tracker is None:
            tracker = SerialSequenceTracker()
            self._trackers[key] = tracker
        tracker.seed(seq_num)

    def reset(self) -> None:
        self._trackers.clear()


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
    """Track firmware stream telemetry and optionally adjust pacing.

    Two chip-independent control behaviors on top of the requested target:

    - backpressure: significant device TX backpressure lowers pacing in spaced
      multiplicative steps with a floor at 70% of the target, then recovers.
    - delivery targeting: when records delivered to the collector fall short of
      the target while the device converts received pacing into fresh CSI
      cleanly (path loss, e.g. broadcast pacing without retries), pacing rises
      above the target up to ``max_boost_ratio`` to compensate, trims back
      once delivery overshoots, and is revoked outright when the fresh ratio
      degrades while above target. A low fresh ratio means the deficit is on
      the device/AP side (aggregation, rate downshift), where extra pacing
      would not help, so the controller holds instead. Callers enable this
      behavior only for group-addressed pacing via ``boost_allowed``.
    """

    # Single-window delivery ratios carry the binomial noise of a lossy,
    # unacknowledged path; delivery targeting acts on an EMA of the ratio so
    # the pacing rate settles instead of chasing per-window RF variance.
    DEFAULT_RECEIVE_RATIO_EMA_ALPHA = 0.5

    def __init__(
        self,
        *,
        initial_pps: float,
        enabled: bool = False,
        min_pps: Optional[float] = None,
        additive_step_pps: Optional[float] = None,
        control_window_s: float = 1.0,
        max_boost_ratio: float = 1.5,
        boost_allowed: bool = True,
        receive_ratio_ema_alpha: float = DEFAULT_RECEIVE_RATIO_EMA_ALPHA,
    ):
        initial_pps = float(initial_pps)
        min_pps = max(5.0, min(initial_pps, 10.0)) if min_pps is None else max(0.1, float(min_pps))
        additive_step_pps = (
            max(1.0, initial_pps * 0.02) if additive_step_pps is None else max(0.1, float(additive_step_pps))
        )
        receive_ratio_ema_alpha = float(receive_ratio_ema_alpha)
        if not 0.0 < receive_ratio_ema_alpha <= 1.0:
            raise ValueError(f"receive_ratio_ema_alpha must be in the (0, 1] range, got {receive_ratio_ema_alpha}")
        self.enabled = bool(enabled)
        # Unicast pacing is MAC-retransmitted, so a delivery deficit there is
        # almost always device-side (aggregation, rate downshift), where extra
        # pacing does not help; callers enable boosting only for group-addressed
        # (broadcast/multicast) pacing, whose loss is retry-less by design.
        self.boost_allowed = bool(boost_allowed)
        self.current_pps = initial_pps
        self.target_pps = initial_pps
        self.max_pps = initial_pps * max(1.0, float(max_boost_ratio))
        self.min_pps = min_pps
        self.additive_step_pps = additive_step_pps
        self.control_window_s = max(0.1, float(control_window_s))
        self.receive_ratio_ema_alpha = receive_ratio_ema_alpha
        self.last_control_at: Optional[float] = None
        self.backpressure_adjust_at: Optional[float] = None
        self.last_window_backpressure_delta = 0
        self.last_window_backpressure_source: Optional[str] = None
        self.last_window_receive_ratio: Optional[float] = None
        self.smoothed_receive_ratio: Optional[float] = None
        self.last_window_fresh_ratio: Optional[float] = None
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
        lowest_receive_ratio: Optional[float] = None
        lowest_fresh_ratio: Optional[float] = None
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
            if pacing_delta > 0:
                fresh_ratio = min(1.0, fresh_delta / pacing_delta)
                if lowest_fresh_ratio is None or fresh_ratio < lowest_fresh_ratio:
                    lowest_fresh_ratio = fresh_ratio
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
                if lowest_receive_ratio is None or receive_ratio < lowest_receive_ratio:
                    lowest_receive_ratio = receive_ratio
        self.last_window_receive_ratio = lowest_receive_ratio
        if lowest_receive_ratio is not None:
            if self.smoothed_receive_ratio is None:
                self.smoothed_receive_ratio = lowest_receive_ratio
            else:
                alpha = self.receive_ratio_ema_alpha
                self.smoothed_receive_ratio = alpha * lowest_receive_ratio + (1.0 - alpha) * self.smoothed_receive_ratio
        self.last_window_fresh_ratio = lowest_fresh_ratio
        if tracked_devices == 0:
            self.last_action = "hold"
            return
        if not self.enabled:
            self.last_action = "hold"
            return

        backpressure_threshold = max(3, math.ceil(target_packets * 0.05))
        significant_backpressure = worst_delta >= backpressure_threshold
        if significant_backpressure:
            settle_time_s = self.control_window_s * 3.0
            if self.backpressure_adjust_at is not None and now - self.backpressure_adjust_at < settle_time_s:
                self.last_action = "backpressure_hold"
                return
            pacing_floor_pps = max(self.min_pps, self.target_pps * 0.70)
            if self.current_pps <= pacing_floor_pps:
                self.last_action = "backpressure_floor"
                self.backpressure_adjust_at = now
                return
            self._apply_pacing_rate(
                max(pacing_floor_pps, self.current_pps * 0.85),
                action="slowdown",
                pacing_sender=pacing_sender,
            )
            self.backpressure_adjust_at = now
            return

        if self.current_pps < self.target_pps:
            settle_time_s = self.control_window_s * 3.0
            if self.backpressure_adjust_at is not None and now - self.backpressure_adjust_at < settle_time_s:
                self.last_action = "recovery_hold"
                return
            self._apply_pacing_rate(
                min(self.current_pps + self.additive_step_pps, self.target_pps),
                action="speedup",
                pacing_sender=pacing_sender,
            )
            return

        # Delivery targeting: only with a fully healthy device (zero
        # backpressure this window, fresh ratio near 1) and outside the
        # post-backpressure settle window, so it can never fight the
        # protective slowdown or chase device-side CSI deficits.
        if self.boost_allowed and worst_delta == 0 and self.smoothed_receive_ratio is not None:
            settle_time_s = self.control_window_s * 3.0
            settled = self.backpressure_adjust_at is None or now - self.backpressure_adjust_at >= settle_time_s
            fresh_healthy = lowest_fresh_ratio is not None and lowest_fresh_ratio >= 0.90
            fresh_degraded = lowest_fresh_ratio is not None and lowest_fresh_ratio < 0.85
            if settled and fresh_degraded and self.current_pps > self.target_pps:
                # The boost premise no longer holds: above-target pacing can
                # itself starve CSI conversion (for example HT frame
                # aggregation), so fall straight back to the requested target
                # instead of holding the harmful rate. The 0.85/0.90 gap keeps
                # boost and revoke from flapping on one noisy window.
                self._apply_pacing_rate(
                    self.target_pps,
                    action="boost_revoke",
                    pacing_sender=pacing_sender,
                )
                return
            if settled and fresh_healthy:
                receive_ratio = self.smoothed_receive_ratio
                if receive_ratio < 0.95 and self.current_pps < self.max_pps:
                    deficit_pps = (1.0 - receive_ratio) * self.target_pps
                    step_pps = min(max(self.additive_step_pps, deficit_pps * 0.5), self.target_pps * 0.10)
                    self._apply_pacing_rate(
                        self.current_pps + step_pps,
                        action="boost",
                        pacing_sender=pacing_sender,
                    )
                    return
                if receive_ratio > 1.05 and self.current_pps > self.target_pps:
                    surplus_pps = (receive_ratio - 1.0) * self.target_pps
                    step_pps = min(max(self.additive_step_pps, surplus_pps * 0.5), self.target_pps * 0.10)
                    self._apply_pacing_rate(
                        max(self.current_pps - step_pps, self.target_pps),
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
        self.out_of_order_count = 0
        self.last_seq = -1
        self._sequence_tracker = PacketSequenceTracker()
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
            phy_mode_code,
            ltf_type_code,
            channel_width_code,
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
            noise_floor_dbm=_valid_noise_floor(int(noise_floor_dbm)),
            tx_backpressure_total=int(tx_backpressure_total),
            stream_fresh_total=int(stream_fresh_total),
            pacing_rx_total=int(pacing_rx_total),
            phy_mode=PHY_MODE_CODES.get(phy_mode_code, "unknown"),
            ltf_type=LTF_TYPE_CODES.get(ltf_type_code, "unknown"),
            channel_width=CHANNEL_WIDTH_CODES.get(channel_width_code, "unknown"),
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

    def accept_packet(self, packet: CSIPacket) -> bool:
        """Accept a fresh packet and account for loss or reordering."""
        observation = self._sequence_tracker.observe(packet)
        if not observation.accepted:
            self.out_of_order_count += 1
            return False
        self.dropped_count += observation.missing
        self.last_seq = int(packet.seq_num)
        return True

    def _check_sequence(self, seq_num: int) -> None:
        """Preserve legacy single-stream drop accounting for callers."""
        if self.last_seq >= 0:
            self.dropped_count += self._compute_sequence_gap(self.last_seq, seq_num)
        self.last_seq = int(seq_num)

    def seed_device_sequence(self, device_id: int, seq_num: int) -> None:
        """Continue sequence tracking across collector phases."""
        self._sequence_tracker.seed_device(device_id, seq_num)

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
            "out_of_order": self.out_of_order_count,
            "drop_rate": self.dropped_count / max(total_expected, 1) * 100,
            "pps": self.pps,
            "buffer_fill": len(self.buffer),
            "buffer_size": self.buffer_size,
            "elapsed": elapsed,
        }

    def reset_stats(self) -> None:
        self.packet_count = 0
        self.dropped_count = 0
        self.out_of_order_count = 0
        self.last_seq = -1
        self._sequence_tracker.reset()
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
            print("Waiting for data...\n")

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
                    if not self.accept_packet(packet):
                        continue
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


class CollectionDetectorGate:
    """Production detector and startup calibration used by timed collection."""

    @staticmethod
    def default_window_size() -> int:
        return int(
            derive_detector_timing(
                nominal_packet_interval_us(100),
                int(getattr(config, "SEGMENTATION_WINDOW_SIZE_MS", 1000)),
            )["window_packets"]
        )

    @staticmethod
    def initial_threshold(algorithm: str) -> float:
        return 1.0 if detector_needs_startup_calibration(algorithm) else ML_DEFAULT_THRESHOLD

    def __init__(self, algorithm: str):
        self.algorithm = normalize_detector_algorithm(algorithm)
        self.window_size = self.default_window_size()
        self.nominal_interval_us = nominal_packet_interval_us(self.window_size)
        self.timing_tracker = PacketTimingTracker(self.nominal_interval_us)
        self.cadence = make_evaluation_cadence(
            max(1, int(getattr(config, "EVALUATION_INTERVAL_MS", 250))),
        )
        self.needs_calibration = detector_needs_startup_calibration(self.algorithm)
        initial_threshold = self.initial_threshold(self.algorithm)
        self.detector = self._make_detector(initial_threshold)
        self.calibrator = self._make_calibrator()
        self.calibrated = not self.needs_calibration
        self.current_metric = 0.0
        self.current_threshold = float(self.detector.get_threshold())

    def _make_detector(self, threshold):
        detector_class = load_detector_class(self.algorithm)
        return detector_class(
            window_size=self.window_size,
            threshold=threshold,
            enable_lowpass=config.ENABLE_LOWPASS_FILTER,
            lowpass_cutoff=config.LOWPASS_CUTOFF,
            enable_hampel=config.ENABLE_HAMPEL_FILTER,
            hampel_window=config.HAMPEL_WINDOW,
            hampel_threshold=config.HAMPEL_THRESHOLD,
        )

    def _make_calibrator(self):
        calibration_duration_ms = int(
            getattr(config, "CALIBRATION_DURATION_MS", 10_000)
        )
        calibration_packets = duration_packet_count(
            calibration_duration_ms,
            self.nominal_interval_us,
        )
        return (
            StartupThresholdCalibrator(
                calibration_packets,
                auto_factor=get_detector_auto_factor(self.detector),
                gate_enabled=get_detector_startup_gate(self.detector),
            )
            if self.needs_calibration
            else None
        )

    def _resize_for_measured_cadence(self):
        timing_update = self.timing_tracker.resolve_detector_timing_update(
            self.window_size,
            int(getattr(config, "SEGMENTATION_WINDOW_SIZE_MS", 1000)),
        )
        if timing_update is None:
            return
        measured_interval_us = timing_update["interval_us"]
        resolved = timing_update["window_packets"]
        threshold = self.initial_threshold(self.algorithm)
        self.window_size = int(resolved)
        self.nominal_interval_us = int(measured_interval_us)
        self.detector = self._make_detector(threshold)
        self.calibrator = self._make_calibrator()
        self.calibrated = not self.needs_calibration
        self.current_metric = 0.0
        self.current_threshold = float(self.detector.get_threshold())
        self.cadence.reset()

    def process_packet(self, packet) -> None:
        csi_data = getattr(packet, "iq_raw", packet)
        timing_packet = {
            "seq_num": getattr(packet, "seq_num", None),
            "device_ticks_us": getattr(packet, "device_ticks_us", None),
            "wifi_rx_ts_us": getattr(packet, "wifi_rx_ts_us", None),
        }
        timing = self.timing_tracker.observe_packet(timing_packet)
        self._resize_for_measured_cadence()
        if timing["contaminated"]:
            self.detector.reset()
            self.cadence.reset()
            self.timing_tracker.reset()
            timing = self.timing_tracker.observe_packet(timing_packet)
        self.detector.process_packet(csi_data, config.DEFAULT_SUBCARRIERS)
        self.cadence.note_packet(elapsed_us=timing["coverage_us"])
        if not self.cadence.should_evaluate():
            return
        packet_weight = self.cadence.equivalent_packets_since_evaluation(
            self.nominal_interval_us
        )
        self.cadence.after_evaluation()

        metrics = self.detector.update_state()
        self.current_metric = float(metrics.get("motion_metric", self.detector.get_motion_metric()))
        self.current_threshold = float(metrics.get("threshold", self.detector.get_threshold()))

        if self.calibrator is None or self.calibrated:
            return
        self.calibrator.observe_detector(
            self.detector,
            packet_weight=packet_weight,
        )
        if not self.calibrator.is_complete():
            return
        self.calibrated = self.calibrator.is_successful()
        if not self.calibrated:
            return
        threshold, _ = self.calibrator.calculate_threshold()
        self.detector.set_adaptive_threshold(threshold)
        self.detector.reset()
        self.current_metric = 0.0
        self.current_threshold = float(self.detector.get_threshold())

    def is_ready(self) -> bool:
        return self.calibrated and self.detector.is_ready()


class CSICollector:
    """Collects labeled CSI data for training datasets."""

    FORMAT_VERSION = dataset_metadata.DATASET_FORMAT_VERSION
    READY_STABLE_SECONDS = 3.0
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
        expected_device_id: Optional[int] = None,
        detector_algorithm: str = "classic",
    ):
        self.label = label
        self.port = port
        self.bind_host = bind_host
        self.chip = None
        self.contributor = contributor or get_git_username()
        self.description = description
        self.expected_source_hosts = list(dict.fromkeys(expected_source_hosts or []))
        self.expected_device_count = max(1, int(expected_device_count)) if expected_device_count is not None else 1
        self.expected_device_id = int(expected_device_id) if expected_device_id is not None else None
        self.receiver = CSIReceiver(port=port, buffer_size=2000, bind_host=bind_host, derive_complex=False)
        self._sample_count = 0
        self.detector_algorithm = normalize_detector_algorithm(detector_algorithm)
        self._ready_window_size = CollectionDetectorGate.default_window_size()
        self._ready_initial_threshold = CollectionDetectorGate.initial_threshold(self.detector_algorithm)
        self._live_status_line_count = 0
        self._nominal_packet_rate: Optional[int] = None

    def _should_accept_source_ip(self, source_ip: Optional[str]) -> bool:
        """Accept all broadcast/multicast sources, but pin unicast modes to targets."""
        if not self.expected_source_hosts:
            return True

        accept_all_sources = False
        for expected_host in self.expected_source_hosts:
            try:
                expected_ip = ipaddress.ip_address(str(expected_host))
            except ValueError:
                continue
            if expected_ip.is_multicast or int(expected_ip) == 0xFFFFFFFF or str(expected_ip).endswith(".255"):
                accept_all_sources = True
                break
        if accept_all_sources:
            return True
        if source_ip is None:
            return False
        return str(source_ip) in self.expected_source_hosts

    def _validate_expected_device_id(self, packet: CSIPacket, source_ip: Optional[str]) -> None:
        """Reject stale discovery targets whose first stream packets carry a different device_id."""
        if self.expected_device_id is None:
            return
        if packet.device_id is None:
            raise ValueError(
                "discovered target expected "
                f"{format_device_id_hex(self.expected_device_id)}, "
                f"but stream packet from {source_ip or '?'} had no device_id metadata"
            )
        packet_device_id = int(packet.device_id)
        if packet_device_id != self.expected_device_id:
            raise ValueError(
                "discovered target expected "
                f"{format_device_id_hex(self.expected_device_id)}, "
                f"but received {format_device_id_hex(packet_device_id)} from {source_ip or '?'}"
            )

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
        # Fail fast on corrupt dataset_info.json before writing any NPZ files.
        dataset_metadata.load_dataset_info()
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
        interval_us = dataset_metadata.measure_packet_interval_us(packets)
        average_packet_rate = (
            1_000_000.0 / float(interval_us)
            if interval_us > 0
            else dataset_metadata.estimate_average_packet_rate(len(packets), duration_ms)
        )
        nominal_packet_rate = self._nominal_packet_rate
        sample = {
            "csi_data": csi_data,
            "num_subcarriers": packets[0].num_subcarriers,
            "label": self.label,
            "chip": self.chip or "unknown",
            "collected_at": datetime.now().isoformat(),
            "duration_ms": duration_ms,
            "format_version": self.FORMAT_VERSION,
            "stream_seq_num": np.array([packet.seq_num for packet in packets], dtype=np.uint32),
            "phy_mode": np.array([packet.phy_mode for packet in packets]),
            "ltf_type": np.array([packet.ltf_type for packet in packets]),
            "channel_width": np.array([packet.channel_width for packet in packets]),
            "device_id": np.uint64(device_id),
        }

        device_ticks = [packet.device_ticks_us for packet in packets]
        if all(value is not None for value in device_ticks):
            sample["device_ticks_us"] = np.array(device_ticks, dtype=np.uint64)

        def add_optional_array(key: str, values, dtype, missing_value=0) -> None:
            if any(value is not None for value in values):
                sample[key] = np.array(
                    [missing_value if value is None else value for value in values],
                    dtype=dtype,
                )

        add_optional_array("wifi_rx_ts_us", [packet.wifi_rx_ts_us for packet in packets], np.uint32)
        add_optional_array("wifi_rx_start_ts_ns", [packet.wifi_rx_start_ts_ns for packet in packets], np.uint64)
        add_optional_array("channel", [packet.channel for packet in packets], np.uint8)
        add_optional_array("rssi_dbm", [packet.rssi_dbm for packet in packets], np.int16)
        # Keep the sentinel rather than the generic 0 fill: 0 dBm would read as a
        # plausible measurement, NOISE_FLOOR_INVALID_DBM cannot.
        add_optional_array(
            "noise_floor_dbm",
            [packet.noise_floor_dbm for packet in packets],
            np.int16,
            missing_value=NOISE_FLOOR_INVALID_DBM,
        )

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
            average_packet_rate=average_packet_rate,
            nominal_packet_rate=nominal_packet_rate,
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
        average_packet_rate: Optional[float] = None,
        nominal_packet_rate: Optional[int] = None,
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
                if average_packet_rate is not None:
                    file_info["average_packet_rate"] = round(float(average_packet_rate), 3)
                if nominal_packet_rate is not None:
                    file_info["nominal_packet_rate"] = int(nominal_packet_rate)
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

    def _build_ready_detector(self) -> CollectionDetectorGate:
        return CollectionDetectorGate(self.detector_algorithm)

    def _update_device_detector_state(
        self,
        device_states: Dict[int, Dict[str, Any]],
        packet: CSIPacket,
        source_ip: str,
    ) -> Dict[str, Any]:
        """Update the per-device detector state and return it."""
        device_id = int(packet.device_id)
        state = device_states.get(device_id)
        if state is None:
            detector = self._build_ready_detector()
            state = {
                "detector": detector,
                "processed_packets": 0,
                "stable_since": None,
                "current_metric": 0.0,
                "current_threshold": detector.current_threshold,
                "current_pps": 0,
                "last_pps_count": 0,
                "source_ip": source_ip,
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
            state["source_ip"] = source_ip
            if packet.chip and packet.chip != "unknown":
                state["chip"] = packet.chip
            if packet.channel is not None:
                state["channel"] = packet.channel
            if packet.rssi_dbm is not None:
                state["rssi_dbm"] = packet.rssi_dbm
            state["last_seq"] = packet.seq_num

        detector = state["detector"]
        detector.process_packet(packet)
        state["processed_packets"] += 1
        state["current_metric"] = detector.current_metric
        state["current_threshold"] = detector.current_threshold
        if detector.is_ready() and state["current_metric"] <= state["current_threshold"]:
            if state["stable_since"] is None:
                state["stable_since"] = time.monotonic()
        else:
            state["stable_since"] = None
        return state

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
    def _summarize_ready_devices(
        device_states: Dict[int, Dict[str, Any]],
        *,
        expected_device_count: int,
        warmup_target: int,
        threshold: float,
        now: float,
        ready_stable_seconds: float,
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
        warm_states = [
            state
            for state in relevant_states
            if state["processed_packets"] >= warmup_target
            and (
                not hasattr(state.get("detector"), "is_ready")
                or state["detector"].is_ready()
            )
        ]
        total_relevant = max(observed_count, required_count)
        if len(warm_states) < observed_count:
            return {
                "ready": False,
                "status": f"CALIBRATING {len(warm_states)}/{total_relevant}",
                "stable_elapsed": 0.0,
                "ready_count": 0,
                "observed_count": observed_count,
                "required_count": required_count,
            }
        if any(
            state["current_metric"] > state.get("current_threshold", threshold)
            for state in relevant_states
        ):
            ready_count = sum(
                1
                for state in relevant_states
                if state["current_metric"] <= state.get("current_threshold", threshold)
            )
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
        ready = stable_elapsed >= ready_stable_seconds
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
        ready_stable_seconds: float,
    ) -> List[str]:
        lines: List[str] = []
        seen_ips = {state.get("source_ip") for state in device_states.values() if state.get("source_ip")}
        for expected_ip in expected_source_hosts:
            if expected_ip not in seen_ips:
                lines.append(
                    f"    ip={expected_ip} chip=? ch=-- rssi=--- "
                    f"{CSICollector._build_status_bar(0.0)} "
                    f"metric=--/{threshold:.3f} pps=-- | WAITING | bp:--"
                )
        for device_id in sorted(device_states):
            state = device_states[device_id]
            processed_packets = int(state.get("processed_packets", 0))
            current_metric = float(state.get("current_metric", 0.0))
            current_threshold = float(state.get("current_threshold", threshold))
            stable_since = state.get("stable_since")
            detector = state.get("detector")
            detector_ready = (
                not hasattr(detector, "is_ready")
                or detector.is_ready()
            )
            if processed_packets < warmup_target:
                status = f"WARMUP {processed_packets}/{warmup_target}"
                metric_ratio = 0.0
            elif not detector_ready:
                status = "CALIBRATING"
                metric_ratio = 0.0
            else:
                stable_value = max(0.0, now - stable_since) if stable_since is not None else 0.0
                metric_ratio = min(current_metric / current_threshold, 1.0) if current_threshold > 0 else 0.0
                if current_metric > current_threshold:
                    status = "UNSTABLE"
                elif stable_value >= ready_stable_seconds:
                    status = "READY"
                else:
                    status = "STABLE"
            metric_text = "--" if processed_packets < warmup_target or not detector_ready else f"{current_metric:.3f}"
            channel = state.get("channel")
            rssi_dbm = state.get("rssi_dbm")
            current_pps = state.get("current_pps")
            lines.append(
                f"    ip={state.get('source_ip', '?')} chip={str(state.get('chip', '?')).upper()} "
                f"ch={'--' if channel is None else f'{int(channel):02d}'} "
                f"rssi={'---' if rssi_dbm is None else str(int(rssi_dbm))} "
                f"{CSICollector._build_status_bar(metric_ratio)} "
                f"metric={metric_text}/{current_threshold:.3f} pps={'--' if current_pps is None else str(int(current_pps))} "
                f"| {status}{CSICollector._format_backpressure_text(state)}"
            )
        return lines

    def _wait_for_ready_state(self, quiet: bool = False, summary_prefix: str = "  ") -> Dict[int, Dict[str, Any]]:
        return self._wait_for_ready_state_with_pacing(
            quiet=quiet,
            summary_prefix=summary_prefix,
            ready_stable_seconds=self.READY_STABLE_SECONDS,
        )

    def _wait_for_ready_state_with_pacing(
        self,
        quiet: bool = False,
        summary_prefix: str = "  ",
        *,
        pacing_controller: Optional[AdaptivePacingController] = None,
        pacing_sender: Any = None,
        ready_stable_seconds: float = READY_STABLE_SECONDS,
    ) -> Dict[int, Dict[str, Any]]:
        if self.receiver.sock is None:
            raise RuntimeError("Receiver socket is not initialized")
        self.receiver.reset_stats()
        warmup_target = self._ready_window_size
        device_states: Dict[int, Dict[str, Any]] = {}
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
                    packet.source_ip = addr[0]
                    if not self._should_accept_source_ip(packet.source_ip):
                        continue
                    self._validate_expected_device_id(packet, addr[0])
                    if not self.receiver.accept_packet(packet):
                        continue
                    processed_packets += 1
                    self.receiver.packet_count += 1
                    if packet.device_id is None:
                        continue
                    state = self._update_device_detector_state(device_states, packet, addr[0])
                    if pacing_controller is not None:
                        pacing_controller.observe_device(
                            state,
                            getattr(packet, "tx_backpressure_total", None),
                            getattr(packet, "stream_fresh_total", None),
                            getattr(packet, "pacing_rx_total", None),
                        )
                now = time.monotonic()
                if pacing_controller is not None:
                    pacing_controller.maybe_adjust(device_states, now=now, pacing_sender=pacing_sender)
                summary = self._summarize_ready_devices(
                    device_states,
                    expected_device_count=self.expected_device_count,
                    warmup_target=warmup_target,
                    threshold=self._ready_initial_threshold,
                    now=now,
                    ready_stable_seconds=ready_stable_seconds,
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
                                threshold=self._ready_initial_threshold,
                                now=now,
                                ready_stable_seconds=ready_stable_seconds,
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
                            threshold=self._ready_initial_threshold,
                            now=now,
                            ready_stable_seconds=ready_stable_seconds,
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
                            threshold=self._ready_initial_threshold,
                            now=now,
                            ready_stable_seconds=ready_stable_seconds,
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
        ready_stable_seconds: float = READY_STABLE_SECONDS,
    ) -> List[CSIPacket]:
        if self.receiver.sock is None:
            raise RuntimeError("Receiver socket is not initialized")
        self.receiver.reset_stats()
        packets: List[CSIPacket] = []
        deadline = time.monotonic() + duration
        warmup_target = self._ready_window_size
        device_states: Dict[int, Dict[str, Any]] = dict(initial_device_states or {})
        for device_id, state in device_states.items():
            if state.get("last_seq") is not None:
                self.receiver.seed_device_sequence(device_id, int(state["last_seq"]))
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
                    packet.source_ip = addr[0]
                    if not self._should_accept_source_ip(packet.source_ip):
                        continue
                    self._validate_expected_device_id(packet, addr[0])
                    if not self.receiver.accept_packet(packet):
                        continue
                    packets.append(packet)
                    processed_packets += 1
                    self.receiver.packet_count += 1
                    if packet.device_id is None:
                        continue
                    state = self._update_device_detector_state(device_states, packet, addr[0])
                    if pacing_controller is not None:
                        pacing_controller.observe_device(
                            state,
                            getattr(packet, "tx_backpressure_total", None),
                            getattr(packet, "stream_fresh_total", None),
                            getattr(packet, "pacing_rx_total", None),
                        )
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
                            threshold=self._ready_initial_threshold,
                            now=now,
                            ready_stable_seconds=ready_stable_seconds,
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
                            threshold=self._ready_initial_threshold,
                            now=now,
                            ready_stable_seconds=ready_stable_seconds,
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
        adaptive_controller_kwargs: Optional[Dict[str, Any]] = None,
        ready_stable_seconds: float = READY_STABLE_SECONDS,
    ) -> List[Path]:
        saved_files: List[Path] = []
        initial_pacing_pps = 0.0
        if pacing_sender is not None and hasattr(pacing_sender, "get_rate_pps"):
            try:
                initial_pacing_pps = float(pacing_sender.get_rate_pps())
            except (TypeError, ValueError):
                initial_pacing_pps = 0.0
        self._nominal_packet_rate = (
            int(round(initial_pacing_pps)) if initial_pacing_pps > 0.0 else None
        )
        controller_kwargs = dict(adaptive_controller_kwargs or {})
        pacing_controller = AdaptivePacingController(
            initial_pps=max(1.0, initial_pacing_pps),
            enabled=adaptive,
            **controller_kwargs,
        )
        if not quiet:
            print(f'\n{"=" * 60}')
            print(f"  CSI Data Collection: {self.label}")
            print(f'{"=" * 60}')
            print(f"  Duration per sample: {duration}s")
            print(f"  Samples to collect:  {num_samples}")
            if ready_stable_seconds <= 0.0:
                print("  Ready gate:          disabled")
            else:
                print(f"  Ready gate:          implicit ({ready_stable_seconds:.1f}s stable)")
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
                ready_device_states = None
                if ready_stable_seconds > 0.0:
                    ready_device_states = self._wait_for_ready_state_with_pacing(
                        quiet=quiet,
                        summary_prefix=summary_prefix,
                        pacing_controller=pacing_controller,
                        pacing_sender=pacing_sender,
                        ready_stable_seconds=ready_stable_seconds,
                    )
                packets = self._collect_with_live_status(
                    duration,
                    quiet=quiet,
                    initial_device_states=ready_device_states,
                    summary_prefix=summary_prefix,
                    pacing_controller=pacing_controller,
                    pacing_sender=pacing_sender,
                    ready_stable_seconds=ready_stable_seconds,
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
            self._nominal_packet_rate = None
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


FORMAT_ID_UNKNOWN = "unknown"
FORMAT_ID_HT20 = "ht20"
LAYOUT_ID_UNKNOWN = "unknown"
LAYOUT_ID_HT20_64 = "ht20_64"
LAYOUT_ID_HT20_57 = "ht20_57"
LAYOUT_ID_HT20_64_DOUBLE = "ht20_64_double"
LAYOUT_ID_HT20_57_DOUBLE = "ht20_57_double"
PAYLOAD_VIEW_RAW = "raw"
PAYLOAD_VIEW_NORMALIZED = "normalized"
METADATA_SOURCE_EXPLICIT = "explicit"
METADATA_SOURCE_HISTORICAL = "historical_missing_phy"
DISPOSITION_DROP = "drop"
DISPOSITION_SENSE = "sense"
REASON_NONE = "none"
REASON_BAD_LENGTH = "bad_length"
REASON_UNSUPPORTED_PHY = "unsupported_phy"
REASON_UNSUPPORTED_WIDTH = "unsupported_width"
REASON_UNEXPECTED_LTF = "unexpected_ltf"
REASON_UNKNOWN_LAYOUT = "unknown_layout"
REASON_UNNORMALIZED_LAYOUT = "unnormalized_layout"
REASON_MISSING_METADATA = "missing_metadata"
NORMALIZATION_DOUBLE_HT20 = "double_ht20"
NORMALIZATION_HT57_TO_64 = "ht57_to_64"
NORMALIZATION_DOUBLE_HT57_TO_64 = "double_ht57_to_64"


def build_csi_format_assessment(
    *,
    format_id: str = FORMAT_ID_UNKNOWN,
    layout_id: str = LAYOUT_ID_UNKNOWN,
    metadata_source: str = METADATA_SOURCE_EXPLICIT,
    payload_view: str = PAYLOAD_VIEW_RAW,
    disposition: str = DISPOSITION_DROP,
    reason_code: str = REASON_BAD_LENGTH,
    normalization_id: Optional[str] = None,
    raw_len: int = 0,
    raw_num_subcarriers: int = 0,
    normalized_len: int = 0,
    normalized_num_subcarriers: int = 0,
) -> Dict[str, Any]:
    """Build a host-side CSI format assessment mapping."""
    return {
        "format_id": format_id,
        "layout_id": layout_id,
        "metadata_source": metadata_source,
        "payload_view": payload_view,
        "disposition": disposition,
        "reason_code": reason_code,
        "normalization_id": normalization_id,
        "raw_len": int(raw_len),
        "raw_num_subcarriers": int(raw_num_subcarriers),
        "normalized_len": int(normalized_len),
        "normalized_num_subcarriers": int(normalized_num_subcarriers),
    }


def assess_ht20_payload_layout(raw_len: int, raw_num_subcarriers: int) -> Dict[str, Any]:
    """Assess one stored CSI payload layout against the HT20 sensing contract."""
    if raw_len <= 0 or raw_len % 2:
        return build_csi_format_assessment(
            reason_code=REASON_BAD_LENGTH,
            raw_len=raw_len,
            raw_num_subcarriers=raw_num_subcarriers,
        )
    if raw_num_subcarriers != raw_len // 2:
        return build_csi_format_assessment(
            reason_code=REASON_BAD_LENGTH,
            raw_len=raw_len,
            raw_num_subcarriers=raw_num_subcarriers,
        )

    common = {
        "format_id": FORMAT_ID_HT20,
        "raw_len": raw_len,
        "raw_num_subcarriers": raw_num_subcarriers,
        "normalized_len": int(getattr(config, "EXPECTED_CSI_LEN", 128)),
        "normalized_num_subcarriers": int(getattr(config, "NUM_SUBCARRIERS", 64)),
        "disposition": DISPOSITION_SENSE,
        "reason_code": REASON_NONE,
    }
    if raw_len == getattr(config, "EXPECTED_CSI_LEN", 128) and raw_num_subcarriers == getattr(config, "NUM_SUBCARRIERS", 64):
        return build_csi_format_assessment(
            layout_id=LAYOUT_ID_HT20_64,
            payload_view=PAYLOAD_VIEW_RAW,
            **common,
        )
    if raw_len == 114 and raw_num_subcarriers == 57:
        return build_csi_format_assessment(
            layout_id=LAYOUT_ID_HT20_57,
            payload_view=PAYLOAD_VIEW_NORMALIZED,
            normalization_id=NORMALIZATION_HT57_TO_64,
            **common,
        )
    if raw_len == 256 and raw_num_subcarriers == 128:
        return build_csi_format_assessment(
            layout_id=LAYOUT_ID_HT20_64_DOUBLE,
            payload_view=PAYLOAD_VIEW_NORMALIZED,
            normalization_id=NORMALIZATION_DOUBLE_HT20,
            **common,
        )
    if raw_len == 228 and raw_num_subcarriers == 114:
        return build_csi_format_assessment(
            layout_id=LAYOUT_ID_HT20_57_DOUBLE,
            payload_view=PAYLOAD_VIEW_NORMALIZED,
            normalization_id=NORMALIZATION_DOUBLE_HT57_TO_64,
            **common,
        )
    return build_csi_format_assessment(
        format_id=FORMAT_ID_HT20,
        reason_code=REASON_UNKNOWN_LAYOUT,
        raw_len=raw_len,
        raw_num_subcarriers=raw_num_subcarriers,
    )


def assess_ht20_sensing_record(
    phy_mode: Any,
    ltf_type: Any,
    channel_width: Any,
    raw_len: int,
    raw_num_subcarriers: int,
    *,
    allow_historical_missing_metadata: bool,
) -> Dict[str, Any]:
    """Assess whether one host record belongs to the supported sensing view."""
    metadata_missing = (
        phy_mode is None
        and ltf_type is None
        and channel_width is None
    )
    metadata_source = METADATA_SOURCE_HISTORICAL if metadata_missing else METADATA_SOURCE_EXPLICIT
    if metadata_missing and not allow_historical_missing_metadata:
        return build_csi_format_assessment(
            metadata_source=metadata_source,
            reason_code=REASON_MISSING_METADATA,
            raw_len=raw_len,
            raw_num_subcarriers=raw_num_subcarriers,
        )
    resolved_phy = "ht" if metadata_missing else str(phy_mode)
    resolved_ltf = "ht-ltf" if metadata_missing else str(ltf_type)
    resolved_width = "20" if metadata_missing else str(channel_width)
    if resolved_phy != "ht":
        return build_csi_format_assessment(
            metadata_source=metadata_source,
            reason_code=REASON_UNSUPPORTED_PHY,
            raw_len=raw_len,
            raw_num_subcarriers=raw_num_subcarriers,
        )
    if resolved_width != "20":
        return build_csi_format_assessment(
            metadata_source=metadata_source,
            reason_code=REASON_UNSUPPORTED_WIDTH,
            raw_len=raw_len,
            raw_num_subcarriers=raw_num_subcarriers,
        )
    if resolved_ltf != "ht-ltf":
        return build_csi_format_assessment(
            metadata_source=metadata_source,
            reason_code=REASON_UNEXPECTED_LTF,
            raw_len=raw_len,
            raw_num_subcarriers=raw_num_subcarriers,
        )
    layout = assess_ht20_payload_layout(raw_len, raw_num_subcarriers)
    layout["metadata_source"] = metadata_source
    if layout["reason_code"] != REASON_NONE:
        layout["disposition"] = DISPOSITION_DROP
        return layout
    if layout["payload_view"] != PAYLOAD_VIEW_RAW:
        # The layout is a known HT20 variant, but stored host data must
        # already be normalized to the 64-SC grid; report the honest reason
        # instead of pretending the layout is unknown.
        layout["disposition"] = DISPOSITION_DROP
        layout["reason_code"] = REASON_UNNORMALIZED_LAYOUT
        return layout
    return layout


def is_ht20_phy(phy_mode: Any, channel_width: Any, ltf_type: Any = "ht-ltf") -> bool:
    """Return True when metadata matches the HT20 HT-LTF sensing contract."""
    return str(phy_mode) == "ht" and str(channel_width) == "20" and str(ltf_type) == "ht-ltf"


def _packet_phy_mask(
    phy_modes: Optional[np.ndarray],
    ltf_types: Optional[np.ndarray],
    channel_widths: Optional[np.ndarray],
    raw_len: int,
    raw_num_subcarriers: int,
    num_packets: int,
    *,
    predicate,
    allow_historical_missing_metadata: bool = True,
) -> Optional[np.ndarray]:
    """Build a boolean keep-mask for one per-packet PHY predicate."""
    if num_packets <= 0:
        return None
    if phy_modes is None and ltf_types is None and channel_widths is None:
        assessment = assess_ht20_sensing_record(
            None,
            None,
            None,
            raw_len,
            raw_num_subcarriers,
            allow_historical_missing_metadata=allow_historical_missing_metadata,
        )
        if assessment["reason_code"] == REASON_NONE:
            return None
        return np.zeros(num_packets, dtype=bool)

    def _value_at(array: Optional[np.ndarray], index: int, default: str) -> str:
        if array is None:
            return default
        arr = np.asarray(array)
        if arr.ndim == 0:
            return str(arr.item())
        if index >= len(arr):
            return default
        return str(arr[index])

    mask = np.empty(num_packets, dtype=bool)
    for index in range(num_packets):
        assessment = predicate(
            _value_at(phy_modes, index, ""),
            _value_at(ltf_types, index, ""),
            _value_at(channel_widths, index, ""),
            raw_len,
            raw_num_subcarriers,
            allow_historical_missing_metadata=allow_historical_missing_metadata,
        )
        mask[index] = assessment["reason_code"] == REASON_NONE
    return mask


def ht20_packet_mask(
    phy_modes: Optional[np.ndarray],
    ltf_types: Optional[np.ndarray],
    channel_widths: Optional[np.ndarray],
    raw_len: int,
    raw_num_subcarriers: int,
    num_packets: int,
    *,
    allow_historical_missing_metadata: bool = True,
) -> Optional[np.ndarray]:
    """Build a boolean keep-mask for the supported HT20 sensing view."""
    return _packet_phy_mask(
        phy_modes,
        ltf_types,
        channel_widths,
        raw_len,
        raw_num_subcarriers,
        num_packets,
        predicate=assess_ht20_sensing_record,
        allow_historical_missing_metadata=allow_historical_missing_metadata,
    )


def filter_npz_arrays_ht20(arrays: Dict[str, Any]) -> Dict[str, Any]:
    """Return a shallow copy of NPZ arrays with non-HT20 packets removed.

    Scalar metadata is preserved. Per-packet vectors whose leading axis matches
    ``csi_data`` / ``csi`` length are sliced with the HT20 mask. When PHY
    metadata is absent, the input mapping is returned unchanged.
    """
    if "csi_data" in arrays:
        csi_key = "csi_data"
    elif "csi" in arrays:
        csi_key = "csi"
    else:
        return arrays

    csi = np.asarray(arrays[csi_key])
    if csi.ndim == 0:
        return arrays
    num_packets = int(csi.shape[0])
    raw_len = int(csi.shape[1])
    raw_num_subcarriers = int(arrays.get("num_subcarriers", raw_len // 2))
    mask = ht20_packet_mask(
        arrays.get("phy_mode"),
        arrays.get("ltf_type"),
        arrays.get("channel_width"),
        raw_len,
        raw_num_subcarriers,
        num_packets,
    )
    if mask is None or bool(np.all(mask)):
        return arrays

    filtered: Dict[str, Any] = {}
    for key, value in arrays.items():
        arr = np.asarray(value)
        if arr.ndim >= 1 and arr.shape[0] == num_packets:
            filtered[key] = arr[mask]
        else:
            filtered[key] = value
    return filtered


def filter_npz_arrays_sensing(arrays: Dict[str, Any]) -> Dict[str, Any]:
    """Return the sensing-view slice, keeping only HT20/HT-LTF/64-SC packets."""
    if "csi_data" in arrays:
        csi_key = "csi_data"
    elif "csi" in arrays:
        csi_key = "csi"
    else:
        return arrays

    csi = np.asarray(arrays[csi_key])
    if csi.ndim == 0:
        return arrays
    num_packets = int(csi.shape[0])
    raw_len = int(csi.shape[1])
    raw_num_subcarriers = int(arrays.get("num_subcarriers", raw_len // 2))
    mask = ht20_packet_mask(
        arrays.get("phy_mode"),
        arrays.get("ltf_type"),
        arrays.get("channel_width"),
        raw_len,
        raw_num_subcarriers,
        num_packets,
    )
    if mask is None or bool(np.all(mask)):
        return arrays

    filtered: Dict[str, Any] = {}
    for key, value in arrays.items():
        arr = np.asarray(value)
        if arr.ndim >= 1 and arr.shape[0] == num_packets:
            filtered[key] = arr[mask]
        else:
            filtered[key] = value
    return filtered


def _packets_from_npz_arrays(
    data: Dict[str, np.ndarray],
    *,
    keep_all_phy: bool = False,
) -> tuple[Mapping[str, Any], ...]:
    """Convert materialized NPZ arrays into packet dictionaries."""
    if "csi_data" not in data:
        raise ValueError("No CSI data found in materialized NPZ")
    csi_array = data["csi_data"]

    label = str(data.get("label", "unknown"))
    num_subcarriers = int(data.get("num_subcarriers", csi_array.shape[1] // 2))
    chip = str(data.get("chip", "unknown"))
    stream_seq_nums = data["stream_seq_num"] if "stream_seq_num" in data else None
    device_ticks_us = data["device_ticks_us"] if "device_ticks_us" in data else None
    wifi_rx_ts_us = data["wifi_rx_ts_us"] if "wifi_rx_ts_us" in data else None
    wifi_rx_start_ts_ns = data["wifi_rx_start_ts_ns"] if "wifi_rx_start_ts_ns" in data else None
    device_ids = data["device_id"] if "device_id" in data else None
    channels = data["channel"] if "channel" in data else None
    rssi_dbm = data["rssi_dbm"] if "rssi_dbm" in data else None
    noise_floor_dbm = data["noise_floor_dbm"] if "noise_floor_dbm" in data else None
    phy_modes = data["phy_mode"] if "phy_mode" in data else None
    ltf_types = data["ltf_type"] if "ltf_type" in data else None
    channel_widths = data["channel_width"] if "channel_width" in data else None

    def optional_scalar(array, index, cast):
        if array is None:
            return None
        if np.ndim(array) == 0:
            return cast(np.asarray(array).item())
        if index >= len(array):
            return None
        return cast(array[index])

    keep_mask = None
    row_len = int(csi_array.shape[1]) if len(csi_array.shape) >= 2 else 0
    if not keep_all_phy:
        keep_mask = ht20_packet_mask(
            phy_modes,
            ltf_types,
            channel_widths,
            row_len,
            num_subcarriers,
            len(csi_array),
        )

    packets: list[Dict[str, Any]] = []
    for index in range(len(csi_array)):
        if keep_mask is not None and not bool(keep_mask[index]):
            continue
        phy_mode = optional_scalar(phy_modes, index, str)
        ltf_type = optional_scalar(ltf_types, index, str)
        channel_width = optional_scalar(channel_widths, index, str)
        assessment = assess_ht20_sensing_record(
            phy_mode,
            ltf_type,
            channel_width,
            len(csi_array[index]),
            num_subcarriers,
            allow_historical_missing_metadata=True,
        )
        phy_mode = "ht" if phy_mode is None else phy_mode
        ltf_type = "ht-ltf" if ltf_type is None else ltf_type
        channel_width = "20" if channel_width is None else channel_width
        phy_format = f"{phy_mode}{channel_width}" if channel_width != "unknown" else phy_mode
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
                "noise_floor_dbm": _valid_noise_floor(optional_scalar(noise_floor_dbm, index, int)),
                "phy_format": phy_format,
                "phy_mode": phy_mode,
                "ltf_type": ltf_type,
                "channel_width": channel_width,
                "format_id": assessment["format_id"],
                "layout_id": assessment["layout_id"],
                "payload_view": assessment["payload_view"],
                "normalization_id": assessment["normalization_id"],
                "format_reason": assessment["reason_code"],
                "format_metadata_source": assessment["metadata_source"],
            }
        )
    return tuple(packets)


def load_npz_packet_view(
    filepath: Path,
    *,
    keep_all_phy: bool = False,
) -> tuple[Dict[str, Any], ...]:
    """Return a shared read-only packet sequence for one dataset NPZ."""
    artifact_name = "packet_view_all_phy" if keep_all_phy else "packet_view_sensing"
    return npz_cache.get_runtime_artifact(
        filepath,
        artifact_name=artifact_name,
        artifact_version=1,
        parameters={"keep_all_phy": bool(keep_all_phy)},
        builder=lambda: _freeze_packet_sequence(
            _packets_from_npz_arrays(
                _load_npz_arrays_uncached(filepath),
                keep_all_phy=keep_all_phy,
            )
        ),
    )


def load_npz_as_packets(
    filepath: Path,
    *,
    keep_all_phy: bool = False,
) -> List[Dict[str, Any]]:
    """Load a ``.npz`` file and convert it to packet dictionaries.

    By default the sensing view keeps only HT20 + HT-LTF + 64-SC packets. A
    narrow compatibility path still accepts historical captures that lack all
    per-record PHY metadata, but only when their stored layout already matches
    the HT20 sensing contract. Pass ``keep_all_phy=True`` to inspect mixed-PHY
    captures. Gaps left by dropped non-sensing packets are intentional for
    sensing consumers; dataset quality validation uses the same filtered view so
    excessive drops fail continuity.

    Callers that mutate packet dictionaries receive shallow copies so the shared
    packet-view cache remains safe to reuse across tools.
    """
    packets = []
    for packet in load_npz_packet_view(filepath, keep_all_phy=keep_all_phy):
        packet_copy = {}
        for key, value in packet.items():
            packet_copy[key] = value.copy() if isinstance(value, np.ndarray) else value
        packets.append(packet_copy)
    return packets


def load_npz_csi_data(
    filepath: Path,
    *,
    keep_all_phy: bool = False,
) -> np.ndarray:
    """Load only the signed CSI matrix from a ``.npz`` recording.

    Long-recording replays do not consume per-packet transport metadata.  Keep
    that hot path on the compact NumPy matrix instead of expanding every row
    into a metadata dictionary. Non-sensing rows are dropped by default when
    PHY metadata is present. Pass ``keep_all_phy=True`` to keep every row.
    """
    data = load_npz_arrays(filepath)
    if "csi_data" not in data:
        raise ValueError(f"No CSI data found in {filepath}")
    csi = np.asarray(data["csi_data"], dtype=np.int8)
    if keep_all_phy:
        return csi
    raw_num_subcarriers = int(data["num_subcarriers"]) if "num_subcarriers" in data else int(csi.shape[1] // 2)
    mask = ht20_packet_mask(
        data["phy_mode"] if "phy_mode" in data else None,
        data["ltf_type"] if "ltf_type" in data else None,
        data["channel_width"] if "channel_width" in data else None,
        int(csi.shape[1]),
        raw_num_subcarriers,
        len(csi),
    )
    if mask is None:
        return csi
    return csi[mask]


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
