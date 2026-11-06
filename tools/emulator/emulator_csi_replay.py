#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-only
# Commercial licensing available under separate agreement; see LICENSING.md.
"""
ESPectre - Emulator CSI replay

Runs a released ESPHome or Native image in esp-emu and streams a
recorded quiet, motion, quiet scenario into it over UART1. esp-emu gets an
eFuse image whose factory MAC carries the replay marker, which turns on the
firmware's CSI replay; Improv Serial provisions Wi-Fi, and Direct HTTP selects
`external` traffic. Frames are paced on the recorded callback times and carry
the recorded RX timestamps, so the firmware sampler sees the capture's own
cadence. The same stream is replayed through the Python detector, and the
firmware's Direct motion events must agree with it.

Usage:
    .venv/bin/python tools/emulator/emulator_csi_replay.py --chip c3 \\
        --firmware <firmware.factory.bin> --emulator esp-emu
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import socket
import string
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.lib.bootstrap import setup_paths  # noqa: E402

setup_paths()

import tools.lib.performance_report as performance_report  # noqa: E402
from config import (  # noqa: E402
    CALIBRATION_DURATION_MS,
    CSI_TARGET_PPS,
    DEFAULT_SUBCARRIERS,
    EVALUATION_INTERVAL_MS,
    MOTION_OFF_HITS,
    MOTION_ON_HITS,
)
from detector_interface import MotionState  # noqa: E402
from espectre_cli.device_transport import (  # noqa: E402
    DirectClient,
    DirectEvent,
    DirectProtocolError,
    ImprovCommand,
    ImprovFrameParser,
    ImprovPacketType,
    ImprovState,
    encode_improv_rpc,
)
from runtime_motion_policy import RuntimeMotionPolicy  # noqa: E402
from threshold import (  # noqa: E402
    StartupThresholdCalibrator,
    get_detector_auto_factor,
    get_detector_startup_gate,
)
from tools.lib import csi_io  # noqa: E402
from tools.lib.dataset_metadata import build_lightweight_detector  # noqa: E402
from tools.lib.temporal_csi_sampler import minimum_valid_slots  # noqa: E402

# (name, capture label, seconds). The first quiet segment must cover calibration.
SCENARIO = (("quiet", "static_presence", 60), ("motion", "motion", 60), ("quiet", "static_presence", 45))
# Detector window plus the on/off hit filters settle within this after a change.
TRANSITION_GUARD_S = 3.0
QUIET_MAX_MOTION_FRACTION = 0.05
MOTION_MIN_MOTION_FRACTION = 0.80
PARITY_MAX_FRACTION_DELTA = 0.10
# Both sides calibrate on the same evaluations, so only rounding may differ.
PARITY_MAX_THRESHOLD_DELTA = 0.005
DIRECT_DEVICE_PORT = 62587
# The emulator can stall for seconds; Direct still sends an SSE heartbeat every 10 s.
DIRECT_TIMEOUT_S = 30.0
# esp-emu's soft AP exists only inside the emulator.
EMULATOR_WIFI_SSID = "espectre-replay"
PROVISIONING_TIMEOUT_S = 120.0
# Seconds between Improv Wi-Fi settings retries while the station joins.
PROVISIONING_RETRY_S = 20.0
# Factory MAC with the firmware's replay marker (kEmulatorReplayMacPrefix in
# src/cpp/frontend/csi_replay_frame.h). Its locally administered bit keeps it
# off every real chip.
REPLAY_FACTORY_MAC = bytes((0x02, 0x45, 0x53, 0x50, 0x00, 0x01))
# Non-MAC words of esp-emu 0.51.0's default eFuse, by word index in its
# QEMU-compatible image (BLK0 6 words, BLK1 6, BLK2 8, BLK3 8, keys, BLK10).
# They carry the wafer revision and calibration version the default boots as.
# Re-read them with espefuse when ESP_EMU_VERSION in .github/workflows/ci.yml changes.
EFUSE_DEFAULT_WORDS = {
    "c3": {9: 0x000C0000, 16: 0x00000001},
    "c5": {5: 0x00001200, 8: 0x00000110},
    "c6": {9: 0x080C0000},
    "s3": {16: 0x00000001},
}
EFUSE_WORDS = 84
EFUSE_FACTORY_MAC_WORD = 6
# Pipeline counters that show where a packet left the replayed stream.
DIAGNOSTIC_FIELDS = (
    "csi_callbacks_total", "csi_provenance_rejected_total", "csi_accepted_total", "csi_admitted_total",
    "csi_pending_frame_drops_total", "csi_missing_slots_total", "csi_excess_total", "csi_stale_total",
    "csi_out_of_order_total", "loop_max_us",
)
# Snapshot the counters once calibration has settled, and again at the end.
DIAGNOSTICS_AFTER_CALIBRATION_S = 20.0
NOMINAL_GAP_US = 1_000_000 // CSI_TARGET_PPS


@dataclass
class StreamPacket:
    segment: int
    stream_us: int  # recorded callback time on the continuous timeline
    rx_ts_us: int  # recorded MAC timestamp on the continuous timeline
    rssi_dbm: int
    csi: np.ndarray


@dataclass
class Segment:
    name: str
    start_s: float
    end_s: float


@dataclass
class FirmwareRun:
    motion: list[tuple[float, bool]] = field(default_factory=list)
    thresholds: list[tuple[float, float]] = field(default_factory=list)
    faults: list[str] = field(default_factory=list)
    calibrated_at_s: Optional[float] = None
    calibrated_threshold: Optional[float] = None
    send_lateness_us: list[float] = field(default_factory=list)
    diagnostics: dict[str, Any] = field(default_factory=dict)
    event_stream_error: Optional[str] = None


def build_stream(chip: str) -> tuple[list[StreamPacket], list[Segment]]:
    """Concatenate scenario segments on one timeline that keeps recorded spacing."""
    static_path, motion_path, _ = csi_io.find_static_presence_motion_dataset(chip=chip)
    sources = {}
    for label, path in (("static_presence", static_path), ("motion", motion_path)):
        arrays = csi_io.load_npz_arrays(path)
        csi = csi_io.load_npz_csi_data(path)
        if len(csi) != len(arrays["csi_data"]):
            raise ValueError(f"{path.name}: PHY filtering would misalign per-packet metadata")
        sources[label] = (arrays, csi, path.name)

    packets: list[StreamPacket] = []
    segments: list[Segment] = []
    stream_base = 0
    rx_base = 1_000_000  # nonzero: the firmware treats a zero RX timestamp as unknown
    for index, (name, label, seconds) in enumerate(SCENARIO):
        arrays, csi, file_name = sources[label]
        ticks = np.asarray(arrays["device_ticks_us"], dtype=np.int64)
        ticks = ticks - ticks[0]
        rx_rel = (np.asarray(arrays["wifi_rx_ts_us"], dtype=np.int64) - int(arrays["wifi_rx_ts_us"][0])) % (1 << 32)
        rows = np.nonzero(ticks < seconds * 1_000_000)[0]
        if ticks[rows[-1]] < (seconds - 1) * 1_000_000:
            raise ValueError(f"{file_name} is shorter than the {seconds} s {name} segment")
        rssi = np.asarray(arrays["rssi_dbm"], dtype=np.int16)
        for row in rows:
            packets.append(StreamPacket(index, stream_base + int(ticks[row]), rx_base + int(rx_rel[row]),
                                        int(rssi[row]), csi[row]))
        segments.append(Segment(name, stream_base / 1e6, (stream_base + int(ticks[rows[-1]])) / 1e6))
        stream_base += int(ticks[rows[-1]]) + NOMINAL_GAP_US
        rx_base += int(rx_rel[rows[-1]]) + NOMINAL_GAP_US
    return packets, segments


def replay_on_host(packets: list[StreamPacket]) -> tuple[list[tuple[float, bool]], Optional[float], Optional[float]]:
    """Run the Python detector, startup calibration, and hit filter over the stream."""
    detector = build_lightweight_detector()
    window = max(1, int(detector.get_window_size()))
    detector.set_minimum_valid_samples(minimum_valid_slots(window))
    interval_us = NOMINAL_GAP_US
    calibrator = StartupThresholdCalibrator(
        max(1, int(round(CALIBRATION_DURATION_MS * 1000.0 / interval_us))),
        auto_factor=get_detector_auto_factor(detector),
        gate_enabled=get_detector_startup_gate(detector),
    )
    policy = RuntimeMotionPolicy(
        evaluation_interval_ms=EVALUATION_INTERVAL_MS,
        motion_on_hits=MOTION_ON_HITS,
        motion_off_hits=MOTION_OFF_HITS,
    )
    detector.on_startup_calibration_begin()
    calibrated_at_s = None
    calibrated_threshold = None
    states: list[tuple[float, bool]] = []
    replay_packets = [{"csi_data": p.csi, "wifi_rx_ts_us": p.rx_ts_us, "rssi_dbm": p.rssi_dbm} for p in packets]
    packets_since_evaluation = 0
    for admission, should_evaluate, _ in performance_report.temporal_detector_ticks(
        detector, replay_packets, interval_us
    ):
        packet = admission.packet
        packets_since_evaluation += 1
        detector.process_packet(performance_report._packet_csi_data(packet), DEFAULT_SUBCARRIERS,
                                rssi_dbm=packet["rssi_dbm"], timestamp_us=int(packet["wifi_rx_ts_us"]))
        if not should_evaluate:
            continue
        detector.update_state()
        stream_s = packets[admission.packet_index].stream_us / 1e6
        if calibrated_at_s is None:
            if detector.is_ready():
                calibrator.observe_detector(detector, packet_weight=packets_since_evaluation)
            calibrator.extend_if_inconclusive(detector.startup_calibration_conclusive())
            if calibrator.is_complete():
                if calibrator.is_successful():
                    calibrator.calculate_threshold()
                    detector.set_adaptive_threshold(0.0)
                else:
                    detector.on_startup_calibration_abandoned()
                calibrated_at_s = stream_s
                calibrated_threshold = float(detector.get_threshold())
        else:
            effective, _ = policy.apply_state(detector.get_state(), ready=detector.is_ready())
            states.append((stream_s, effective == MotionState.MOTION))
        packets_since_evaluation = 0
    return states, calibrated_at_s, calibrated_threshold


def _encode_frame(packet: StreamPacket) -> bytes:
    body = (bytes([packet.rssi_dbm & 0xFF, len(packet.csi)]) + (packet.rx_ts_us & 0xFFFFFFFF).to_bytes(4, "little")
            + packet.csi.astype(np.int8).tobytes())
    return b"CSIR" + body + bytes([sum(body) & 0xFF])


def write_replay_efuse(chip: str, path: Path) -> None:
    """Write esp-emu's default eFuse for `chip` with the replay marker as factory MAC."""
    words = [0] * EFUSE_WORDS
    for index, value in EFUSE_DEFAULT_WORDS[chip].items():
        words[index] = value
    # esp-emu stores the factory MAC with BLK1 word 0 holding bytes 2-5 and
    # word 1 bytes 0-1, both big-endian.
    words[EFUSE_FACTORY_MAC_WORD] = int.from_bytes(REPLAY_FACTORY_MAC[2:6], "big")
    words[EFUSE_FACTORY_MAC_WORD + 1] = int.from_bytes(REPLAY_FACTORY_MAC[0:2], "big")
    path.write_bytes(b"".join(word.to_bytes(4, "little") for word in words))


def improv_wifi_settings() -> tuple[str, bytes]:
    """Pick a soft AP password whose Improv Wi-Fi settings frame is plain ASCII.

    esp-emu forwards stdin to the console as text, which mangles bytes above
    0x7F, and the frame's checksum byte depends on the credentials.
    """
    # Each added letter shifts the checksum by about a hundred, so short suffixes cover every value.
    for length in range(1, 4):
        for suffix in itertools.product(string.ascii_lowercase, repeat=length):
            password = "replay-" + "".join(suffix)
            frame = encode_improv_rpc(ImprovCommand.WIFI_SETTINGS, (EMULATOR_WIFI_SSID, password))
            if max(frame) < 0x80 and frame.count(b"\n") == 1 and b"\r" not in frame:
                return password, frame
    raise RuntimeError("no ASCII-only Improv Wi-Fi settings frame")


class EmulatorConsole:
    """esp-emu's console: output mirrored to the log, input written to stdin.

    Watches the output for Improv's provisioned state, so the tool does not
    poll Direct HTTP while the station joins: connections queued behind the
    association would exhaust the firmware's sockets.
    """

    def __init__(self, process: subprocess.Popen, log) -> None:
        self._process = process
        self._log = log
        self._improv = ImprovFrameParser()
        self.provisioned = threading.Event()
        self._reader = threading.Thread(target=self._pump, daemon=True)
        self._reader.start()

    def _pump(self) -> None:
        stream = self._process.stdout
        while True:
            chunk = os.read(stream.fileno(), 4096)
            if not chunk:
                return
            self._log.write(chunk)
            self._log.flush()
            if not self.provisioned.is_set():
                for frame in self._improv.feed(chunk):
                    if frame.packet_type == ImprovPacketType.CURRENT_STATE \
                            and frame.data == bytes((ImprovState.PROVISIONED,)):
                        self.provisioned.set()

    def write(self, data: bytes) -> None:
        self._process.stdin.write(data)
        self._process.stdin.flush()

    def join(self, timeout_s: float) -> None:
        self._reader.join(timeout_s)


def provision_wifi(console: EmulatorConsole, frame: bytes) -> None:
    """Send Improv Wi-Fi settings until the firmware reports it is provisioned."""
    deadline = time.monotonic() + PROVISIONING_TIMEOUT_S
    while time.monotonic() < deadline:
        console.write(frame)
        if console.provisioned.wait(min(PROVISIONING_RETRY_S, max(0.0, deadline - time.monotonic()))):
            return
    raise TimeoutError(f"firmware did not join the emulator soft AP within {PROVISIONING_TIMEOUT_S:.0f} s")


def _read_diagnostics(client: DirectClient) -> dict[str, Any]:
    # Each frontend publishes its own catalog; ask only for the counters it has.
    catalog = {field["name"] for field in client.request("get", "diagnostics")["fields"]}
    return client.request("get", "diagnostics", {"fields": [name for name in DIAGNOSTIC_FIELDS if name in catalog]})


def _snapshot_diagnostics(client: DirectClient, run: FirmwareRun, name: str) -> None:
    try:
        run.diagnostics[name] = _read_diagnostics(client)
    except (DirectProtocolError, KeyError, TypeError) as error:
        run.diagnostics[name] = {"error": str(error)}


def _record_events(events: list[DirectEvent], offset_s: float, run: FirmwareRun) -> None:
    """Collect motion, threshold, and fault events on the stream timeline."""
    for event in events:
        now_s = event.host_elapsed_seconds - offset_s
        if now_s < 0:
            continue
        data = event.data
        if event.name == "motion":
            run.motion.append((now_s, data.get("state") == "motion"))
        elif event.name == "sensing":
            run.thresholds.append((now_s, float(data["threshold"])))
            if run.calibrated_at_s is None and not data.get("calibrating", True):
                run.calibrated_at_s = now_s
                run.calibrated_threshold = float(data["threshold"])
        elif event.name == "fault":
            run.faults.append(str(data.get("message")))


def run_firmware(args: argparse.Namespace, packets: list[StreamPacket]) -> FirmwareRun:
    """Boot the firmware in esp-emu, provision it, stream the scenario, and collect Direct events."""
    duration_s = int(packets[-1].stream_us / 1e6 + PROVISIONING_TIMEOUT_S) + 120
    efuse = args.log.with_suffix(".efuse")
    write_replay_efuse(args.chip, efuse)
    password, wifi_frame = improv_wifi_settings()
    command = [args.emulator, "--chip", f"esp32{args.chip}", "--firmware", str(args.firmware),
               "--efuse", str(efuse), "--timeout", f"{duration_s}s",
               "--wifi-ssid", EMULATOR_WIFI_SSID, "--wifi-password", password,
               "--uart1-tcp", f"127.0.0.1:{args.uart_port}",
               "--net", f"user,hostfwd=tcp::{args.direct_port}-:{DIRECT_DEVICE_PORT}"]
    run = FirmwareRun()
    client = DirectClient(f"http://127.0.0.1:{args.direct_port}/espectre/v1", timeout=DIRECT_TIMEOUT_S)
    with open(args.log, "wb") as log, subprocess.Popen(
            command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT) as emulator:
        console = EmulatorConsole(emulator, log)
        try:
            uart = _wait_for(lambda: socket.create_connection(("127.0.0.1", args.uart_port), timeout=2), 30)
            provision_wifi(console, wifi_frame)
            _wait_for(lambda: client.request("get", "health", timeout=2.0)["status"] == "ok" or None, 60,
                      interval_s=2.0)
            # Replayed frames arrive as an external host's ping, so no ICMP identifier is needed.
            client.request("patch", "sensing", {"traffic_generator_mode": "external"})
            # Open the event stream before the first frame so no early event is missed.
            events_opened = time.monotonic()
            client.start_events()
            stream_start = time.monotonic()
            snapshot = threading.Thread(target=_snapshot_diagnostics, args=(client, run, "after_calibration"),
                                        daemon=True)
            for packet in packets:
                due = stream_start + packet.stream_us / 1e6
                delay = due - time.monotonic()
                if delay > 0:
                    time.sleep(delay)
                run.send_lateness_us.append((time.monotonic() - due) * 1e6)
                uart.sendall(_encode_frame(packet))
                if snapshot.ident is None and packet.stream_us >= DIAGNOSTICS_AFTER_CALIBRATION_S * 1e6:
                    snapshot.start()
            time.sleep(TRANSITION_GUARD_S)
            snapshot.join(DIRECT_TIMEOUT_S)
            _snapshot_diagnostics(client, run, "end")
            uart.close()
            try:
                client.stop_events()
            except DirectProtocolError as error:
                run.event_stream_error = str(error)
            _record_events(list(client.events), stream_start - events_opened, run)
        finally:
            client.close()
            emulator.terminate()
            emulator.wait(timeout=30)
            console.join(5.0)
    return run


def _wait_for(probe, timeout_s: float, interval_s: float = 0.25):
    deadline = time.monotonic() + timeout_s
    while True:
        try:
            result = probe()
            if result:
                return result
        except (OSError, DirectProtocolError):
            pass
        if time.monotonic() > deadline:
            raise TimeoutError(f"emulator did not become ready within {timeout_s:.0f} s")
        time.sleep(interval_s)


def motion_fraction(samples: list[tuple[float, bool]], segment: Segment, scored_from_s: float) -> Optional[float]:
    start = max(segment.start_s + TRANSITION_GUARD_S, scored_from_s)
    window = [motion for t, motion in samples if start <= t < segment.end_s]
    return None if not window else float(sum(window)) / len(window)


def evaluate(segments, firmware: FirmwareRun, host_states, host_calibrated_at_s,
             host_calibrated_threshold) -> dict[str, Any]:
    failures: list[str] = []
    if firmware.faults:
        failures.append(f"firmware faults: {firmware.faults}")
    if firmware.event_stream_error:
        failures.append(f"Direct event stream ended early: {firmware.event_stream_error}")
    if firmware.calibrated_at_s is None or firmware.calibrated_at_s >= segments[0].end_s:
        failures.append("firmware calibration did not complete in the first quiet segment")
    if host_calibrated_at_s is None or host_calibrated_at_s >= segments[0].end_s:
        failures.append("host calibration did not complete in the first quiet segment")
    if (firmware.calibrated_threshold is not None and host_calibrated_threshold is not None
            and abs(firmware.calibrated_threshold - host_calibrated_threshold) > PARITY_MAX_THRESHOLD_DELTA):
        failures.append(f"calibrated threshold: firmware {firmware.calibrated_threshold:.4f} "
                        f"vs host {host_calibrated_threshold:.4f}")
    scored_from = max(firmware.calibrated_at_s or 0.0, host_calibrated_at_s or 0.0)
    rows = []
    for segment in segments:
        fw = motion_fraction(firmware.motion, segment, scored_from)
        host = motion_fraction(host_states, segment, scored_from)
        rows.append({"segment": segment.name, "start_s": round(segment.start_s, 1), "end_s": round(segment.end_s, 1),
                     "firmware_motion_fraction": fw, "host_motion_fraction": host})
        if fw is None or host is None:
            failures.append(f"{segment.name} {segment.start_s:.0f}-{segment.end_s:.0f} s has no scored evaluations")
            continue
        if segment.name == "quiet" and fw > QUIET_MAX_MOTION_FRACTION:
            failures.append(f"quiet {segment.start_s:.0f} s: firmware motion fraction {fw:.2f}")
        if segment.name == "motion" and fw < MOTION_MIN_MOTION_FRACTION:
            failures.append(f"motion {segment.start_s:.0f} s: firmware motion fraction {fw:.2f}")
        if abs(fw - host) > PARITY_MAX_FRACTION_DELTA:
            failures.append(f"{segment.name} {segment.start_s:.0f} s: firmware {fw:.2f} vs host {host:.2f}")
    lateness = np.asarray(firmware.send_lateness_us or [0.0])
    return {
        "passed": not failures,
        "failures": failures,
        "segments": rows,
        "firmware_calibrated_at_s": firmware.calibrated_at_s,
        "host_calibrated_at_s": host_calibrated_at_s,
        "firmware_calibrated_threshold": firmware.calibrated_threshold,
        "host_calibrated_threshold": host_calibrated_threshold,
        "firmware_thresholds": firmware.thresholds,
        "firmware_diagnostics": firmware.diagnostics,
        "send_lateness_us": {"p50": float(np.percentile(lateness, 50)), "p99": float(np.percentile(lateness, 99)),
                             "max": float(lateness.max())},
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--chip", required=True, choices=("c3", "c5", "c6", "s3"), help="Replay test firmware chip")
    parser.add_argument("--firmware", required=True, type=Path, help="Merged factory image to run")
    parser.add_argument("--emulator", default="esp-emu", help="esp-emu executable")
    parser.add_argument("--log", type=Path, default=Path("emulator-replay.log"), help="Emulator console log")
    parser.add_argument("--json", type=Path, help="Write the verdict as JSON")
    parser.add_argument("--uart-port", type=int, default=5556, help="Host port for the UART1 bridge")
    parser.add_argument("--direct-port", type=int, default=18690, help="Host port forwarded to Direct HTTP")
    args = parser.parse_args()

    packets, segments = build_stream(args.chip)
    host_states, host_calibrated_at_s, host_calibrated_threshold = replay_on_host(packets)
    firmware = run_firmware(args, packets)
    verdict = evaluate(segments, firmware, host_states, host_calibrated_at_s, host_calibrated_threshold)
    if args.json:
        args.json.write_text(json.dumps(verdict, indent=2) + "\n", encoding="utf-8")
    for row in verdict["segments"]:
        fw, host = row["firmware_motion_fraction"], row["host_motion_fraction"]
        print(f"{row['segment']:6s} {row['start_s']:6.1f}-{row['end_s']:6.1f} s  firmware motion "
              f"{'n/a' if fw is None else f'{fw:.2f}'}  host motion {'n/a' if host is None else f'{host:.2f}'}")
    print(f"calibrated at firmware {verdict['firmware_calibrated_at_s']} s, host {verdict['host_calibrated_at_s']} s; "
          f"threshold firmware {verdict['firmware_calibrated_threshold']}, host {verdict['host_calibrated_threshold']}; "
          f"send lateness p99 {verdict['send_lateness_us']['p99']:.0f} us")
    for name, values in verdict["firmware_diagnostics"].items():
        counters = ", ".join(f"{key} {values[key]}" for key in DIAGNOSTIC_FIELDS if key in values)
        print(f"diagnostics {name}: {counters or values}")
    for failure in verdict["failures"]:
        print(f"FAIL: {failure}")
    return 0 if verdict["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
