# SPDX-License-Identifier: GPL-3.0-only
# Commercial licensing available under separate agreement; see LICENSING.md.
"""
ESPectre - Raw CSI Record Tests

Unit tests for the transport-neutral CSI raw record parser and dataset writer.

Author: Francesco Pace <francesco.pace@gmail.com>
"""

import asyncio
import copy
import io
import json
import queue
import socket
import threading
import time
from types import SimpleNamespace
from urllib.error import HTTPError

import numpy as np
import pytest

from tools.ha_traffic_generator_addon import espectre_traffic_generator
from tools.ha_traffic_generator_addon import traffic as traffic_addon
from tools.ha_traffic_generator_addon import ha_client, panel
from tools.traffic_generator_docker import devices as standalone_devices
from tools.traffic_generator_docker import server as standalone
from tools.lib import dataset_metadata
from espectre_cli.device_transport import DirectProtocolError, DirectRequestError
from tools.lib.csi_io import (
    CSICollector,
    CSIReceiver,
    DirectRawCSIReceiver,
    CSI_HEADER_STRUCT,
    RAW_CSI_FLAG_WIFI_RX_START_TS_NS_VALID,
    RAW_CSI_FLAG_WIFI_RX_TS_VALID,
    RAW_CSI_RECORD_MAGIC,
    RAW_CSI_RECORD_VERSION,
    RAW_CSI_RECORD_VERSION_V8,
    RAW_CSI_HTTP_FRAME_STRUCT,
    RAW_CSI_PROTOCOL_VERSION,
    RAW_CSI_RESPONSE_MAGIC,
    load_npz_arrays,
    load_npz_as_packets,
    load_npz_packet_view,
    normalize_stored_csi_bin_layout,
)

from device_utils import (
    HT20_CENTERED_ONLY_NULL_BINS,
    HT20_CLASSIC_ONLY_NULL_BINS,
)


def _layout_rows(null_bins, num_rows=3):
    """Build CSI rows populated everywhere except one layout's guard bins."""
    rows = np.zeros((num_rows, 128), dtype=np.int8)
    for bin_index in range(64):
        rows[:, bin_index * 2] = 7 + (bin_index % 5)
        rows[:, bin_index * 2 + 1] = -3 - (bin_index % 4)
    for bin_index in null_bins:
        rows[:, bin_index * 2] = 0
        rows[:, bin_index * 2 + 1] = 0
    return rows


def test_stored_classic_layout_is_rotated_to_centered():
    classic = _layout_rows(HT20_CLASSIC_ONLY_NULL_BINS)
    rotated = normalize_stored_csi_bin_layout(classic)
    np.testing.assert_array_equal(rotated, np.roll(classic, 64, axis=1))


def test_stored_centered_layout_is_left_alone():
    centered = _layout_rows(HT20_CENTERED_ONLY_NULL_BINS)
    np.testing.assert_array_equal(normalize_stored_csi_bin_layout(centered), centered)


def test_sparse_rows_are_left_alone():
    """Rows null under both conventions carry no evidence either way."""
    sparse = np.zeros((3, 128), dtype=np.int8)
    sparse[:, :4] = np.array([1, 2, 3, 4], dtype=np.int8)
    np.testing.assert_array_equal(normalize_stored_csi_bin_layout(sparse), sparse)


def test_load_npz_arrays_rotates_classic_recordings(tmp_path):
    filepath = tmp_path / "classic_layout.npz"
    classic = _layout_rows(HT20_CLASSIC_ONLY_NULL_BINS)
    np.savez_compressed(filepath, csi_data=classic, num_subcarriers=64, label="motion", chip="c3")

    loaded = load_npz_arrays(filepath)["csi_data"]
    np.testing.assert_array_equal(loaded, np.roll(classic, 64, axis=1))


def test_load_npz_arrays_returns_read_only_cached_mapping(tmp_path):
    filepath = tmp_path / "read_only_arrays.npz"
    np.savez_compressed(
        filepath,
        csi_data=np.zeros((1, 128), dtype=np.int8),
        num_subcarriers=64,
        label="motion",
        chip="c3",
    )

    arrays = load_npz_arrays(filepath)

    assert arrays["csi_data"].flags.writeable is False
    with pytest.raises(TypeError):
        arrays["extra"] = np.zeros(1, dtype=np.int8)


def test_packet_view_is_read_only_but_packet_loader_returns_writable_copies(tmp_path):
    filepath = tmp_path / "packet_views.npz"
    np.savez_compressed(
        filepath,
        csi_data=np.arange(128, dtype=np.int8).reshape(1, 128),
        num_subcarriers=64,
        label="motion",
        chip="c3",
    )

    packet_view = load_npz_packet_view(filepath)
    assert packet_view[0]["csi_data"].flags.writeable is False
    with pytest.raises(TypeError):
        packet_view[0]["label"] = "changed"

    packets = load_npz_as_packets(filepath)
    assert packets[0]["csi_data"].flags.writeable is True
    packets[0]["csi_data"][0] = -5
    assert load_npz_packet_view(filepath)[0]["csi_data"][0] != -5


def build_packet(
    *,
    seq_num=1,
    chip_code=6,
    flags=0,
    payload=None,
    device_id=0x112233445566,
    device_ticks_us=123456,
    wifi_rx_ts_us=0,
    wifi_rx_start_ts_ns=0,
    channel=6,
    rssi_dbm=-42,
    noise_floor_dbm=-96,
    transport_backpressure_total=0,
    fresh_record_total=0,
    request_accepted_total=0,
    phy_mode=2,
    ltf_type=2,
    channel_width=1,
    version=RAW_CSI_RECORD_VERSION,
):
    payload_values = payload if payload is not None else [1, 2, 3, 4]
    payload = np.array(payload_values, dtype=np.int8).tobytes()
    num_sc = len(payload) // 2
    header = CSI_HEADER_STRUCT.pack(
        RAW_CSI_RECORD_MAGIC,
        version,
        CSI_HEADER_STRUCT.size,
        chip_code,
        flags,
        seq_num,
        num_sc,
        len(payload),
        device_id,
        device_ticks_us,
        wifi_rx_ts_us,
        wifi_rx_start_ts_ns,
        channel,
        rssi_dbm,
        noise_floor_dbm,
        transport_backpressure_total,
        fresh_record_total,
        request_accepted_total,
        phy_mode,
        ltf_type,
        channel_width,
    )
    return header + payload


def test_parse_packet_accepts_transport_neutral_v8_header():
    receiver = CSIReceiver(bind_host='127.0.0.1')
    packet = receiver._parse_packet(
        build_packet(
            seq_num=7,
            payload=[10, 20, -30, 40],
            device_ticks_us=987654,
            channel=11,
            rssi_dbm=-55,
        )
    )

    assert packet is not None
    assert packet.seq_num == 7
    assert packet.num_subcarriers == 2
    assert packet.chip == 'C6'
    assert packet.device_ticks_us == 987654
    assert packet.channel == 11
    assert packet.rssi_dbm == -55
    assert packet.transport_backpressure_total == 0
    assert packet.phy_mode == 'ht'
    assert packet.ltf_type == 'ht-ltf'
    assert packet.channel_width == '20'
    np.testing.assert_array_equal(packet.iq_raw, np.array([10, 20, -30, 40], dtype=np.int8))
    np.testing.assert_allclose(packet.iq_complex, np.array([20 + 10j, 40 - 30j], dtype=np.complex64))


def test_parse_packet_preserves_legacy_lltf_metadata_and_iq():
    payload = [10, -20] * 64
    packet = CSIReceiver(bind_host='127.0.0.1')._parse_packet(
        build_packet(phy_mode=1, ltf_type=1, payload=payload)
    )

    assert packet is not None
    assert packet.record_version == RAW_CSI_RECORD_VERSION
    assert packet.phy_mode == 'legacy'
    assert packet.ltf_type == 'lltf'
    assert packet.iq_raw.tolist() == payload


def test_parse_packet_accepts_v8_transport_neutral_counters_byte_for_byte():
    receiver = CSIReceiver(bind_host="127.0.0.1", derive_complex=False)
    record = build_packet(
        version=RAW_CSI_RECORD_VERSION_V8,
        transport_backpressure_total=11,
        fresh_record_total=22,
        request_accepted_total=33,
        payload=[1, -2, 3, -4],
    )

    packet = receiver._parse_packet(record)

    assert packet is not None
    assert packet.record_version == RAW_CSI_RECORD_VERSION_V8
    assert packet.transport_backpressure_total == 11
    assert packet.fresh_record_total == 22
    assert packet.request_accepted_total == 33
    assert record[: CSI_HEADER_STRUCT.size] == CSI_HEADER_STRUCT.pack(
        RAW_CSI_RECORD_MAGIC,
        RAW_CSI_RECORD_VERSION_V8,
        64,
        6,
        0,
        1,
        2,
        4,
        0x112233445566,
        123456,
        0,
        0,
        6,
        -42,
        -96,
        11,
        22,
        33,
        2,
        2,
        1,
    )


def test_direct_raw_receiver_negotiates_v8_and_feeds_shared_packet_parser():
    session_id = bytes.fromhex("00112233445566778899aabbccddeeff")

    class FakeControl:
        def __init__(self, endpoint, **_kwargs):
            self.endpoint = endpoint
            self.requests = []
            self.closed = False

        def request(self, verb, resource, data=None, **_kwargs):
            self.requests.append((verb, resource, data))
            if resource == "capabilities":
                return {
                    "features": {"csi": True},
                    "csi": {
                        "transport": "http",
                        "protocol_version": RAW_CSI_PROTOCOL_VERSION,
                        "record_version": 8,
                        "frame_prefix_bytes": 60,
                    },
                }
            if resource == "device":
                return {
                    "device_id": "112233445566",
                    "frontend": "native",
                    "firmware": "test",
                    "chip": "esp32c3",
                }
            if resource == "diagnostics":
                return {
                    "raw_csi": {
                        "fresh_record_total": 1,
                        "raw_drop_total": 2,
                        "send_backpressure_total": 3,
                        "stream_sequence": 3,
                    }
                }
            raise AssertionError((verb, resource, data))

        def close(self):
            self.closed = True

    class FakeRawResponse:
        status = 200

        def __init__(self, payload):
            self.chunks = [payload[:17], payload[17:93], payload[93:]]
            self.closed = False

        def read1(self, _size):
            return self.chunks.pop(0)

        def close(self):
            self.closed = True

    class FakeRawConnection:
        def __init__(self):
            self.closed = False
            self.request_args = None
            self.socket_timeouts = []
            self.socket_options = []
            self.sock = self
            record = build_packet(
                version=RAW_CSI_RECORD_VERSION_V8,
                flags=1 << 3,
                transport_backpressure_total=2,
                fresh_record_total=1,
                request_accepted_total=1,
            )
            prefix = RAW_CSI_HTTP_FRAME_STRUCT.pack(
                RAW_CSI_RESPONSE_MAGIC,
                RAW_CSI_PROTOCOL_VERSION,
                8,
                RAW_CSI_HTTP_FRAME_STRUCT.size,
                session_id,
                1,
                len(record),
                0,
                1,
                0,
                2,
            )
            self.response = FakeRawResponse(prefix + record)

        def settimeout(self, timeout):
            self.socket_timeouts.append(timeout)

        def setsockopt(self, level, option, value):
            self.socket_options.append((level, option, value))

        def request(self, method, path, headers):
            self.request_args = (method, path, headers)

        def getresponse(self):
            return self.response

        def close(self):
            self.closed = True

    control = None
    raw = FakeRawConnection()

    def control_factory(*args, **kwargs):
        nonlocal control
        control = FakeControl(*args, **kwargs)
        return control

    receiver = DirectRawCSIReceiver(
        "192.168.1.23:62587",
        control_client_factory=control_factory,
        raw_connection_factory=lambda *_args, **_kwargs: raw,
        derive_complex=False,
    )
    observed = []
    receiver.add_callback(observed.append)

    receiver._open()
    receiver._raw_buffer.extend(receiver._read_raw_chunk())
    receiver._consume_raw_frames()
    assert observed == []
    receiver._raw_buffer.extend(receiver._read_raw_chunk())
    receiver._consume_raw_frames()
    receiver._raw_buffer.extend(receiver._read_raw_chunk())
    receiver._consume_raw_frames()
    detached_packet = copy.copy(observed[0])
    receiver.stop()
    receiver.apply_final_raw_diagnostics(detached_packet)

    assert len(observed) == 1
    assert observed[0].record_version == RAW_CSI_RECORD_VERSION_V8
    assert observed[0].transport == "http"
    assert observed[0].source_ip == "192.168.1.23"
    assert observed[0].raw_stream_sequence == 1
    assert observed[0].raw_final_stream_sequence == 3
    assert observed[0].fresh_record_total == 1
    assert observed[0].raw_drop_total == 2
    assert observed[0].raw_send_backpressure_total == 3
    assert detached_packet.raw_final_stream_sequence == 3
    assert detached_packet.fresh_record_total == 1
    assert detached_packet.raw_drop_total == 2
    assert detached_packet.raw_send_backpressure_total == 3
    assert raw.request_args[0:2] == ("GET", "/espectre/v1/csi")
    assert "Authorization" not in raw.request_args[2]
    assert raw.socket_timeouts == [None]
    assert raw.socket_options == [(socket.SOL_SOCKET, socket.SO_RCVBUF, 1024 * 1024)]
    assert [(request[0], request[1]) for request in control.requests] == [
        ("get", "capabilities"),
        ("get", "device"),
        ("get", "diagnostics"),
    ]


def test_direct_raw_receiver_run_timeout_does_not_wait_for_idle_stream():
    class IdleRawResponse:
        def __init__(self):
            self.closed = threading.Event()

        def read1(self, _size):
            self.closed.wait(timeout=2.0)
            return b""

        def close(self):
            self.closed.set()

    class IdleRawConnection:
        def __init__(self, response):
            self.sock = self
            self._response = response

        def shutdown(self, _mode):
            self._response.closed.set()

        def close(self):
            pass

    response = IdleRawResponse()
    receiver = DirectRawCSIReceiver("192.168.1.23", derive_complex=False)
    receiver._raw_response = response
    receiver._raw_connection = IdleRawConnection(response)
    receiver.running = True
    receiver._open = lambda: None

    started_at = time.monotonic()
    receiver.run(timeout=0.05)
    elapsed = time.monotonic() - started_at
    receiver.stop()

    assert elapsed < 0.5
    assert response.closed.is_set()


@pytest.mark.parametrize("raw_state", [False, True, None, OSError("offline")])
def test_direct_raw_receiver_checks_device_session_after_idle_across_runs(monkeypatch, raw_state):
    clock = [0.0]
    requests = []

    class IdleQueue:
        def get(self, timeout):
            clock[0] += timeout
            raise queue.Empty

    class Control:
        def request(self, method, resource, data, **kwargs):
            requests.append((method, resource, data, kwargs))
            if isinstance(raw_state, Exception):
                raise raw_state
            if raw_state is None:
                return {}
            return {"raw_csi": {"active": raw_state, "send_backpressure_total": 1, "raw_drop_total": 543}}

    receiver = DirectRawCSIReceiver("192.168.1.23", derive_complex=False)
    receiver._control = Control()
    receiver._raw_read_queue = IdleQueue()
    receiver._open = lambda: None
    receiver._ensure_raw_reader = lambda: None
    receiver.running = True
    monkeypatch.setattr(time, "monotonic", lambda: clock[0])

    # The CLI repeatedly runs one-second slices; idle time must survive them.
    for _ in range(7):
        receiver.run(timeout=1.0)
    assert requests == []
    if raw_state is False:
        with pytest.raises(
            RuntimeError,
            match=r"session ended on the device while the stream remained open "
            r"\(send backpressure: 1, dropped records: 543\)",
        ):
            receiver.run(timeout=2.0)
    else:
        receiver.run(timeout=2.0)
        assert receiver.running
    assert len(requests) == 1
    assert requests[0][:3] == ("get", "diagnostics", {"fields": ["raw_csi"]})


@pytest.mark.parametrize("raw_state", [False, True, None, OSError("offline")])
def test_direct_raw_receiver_reports_ended_session_when_stream_closes(raw_state):
    requests = []

    class ClosedQueue:
        def get(self, timeout):
            del timeout
            return None, RuntimeError("Direct raw HTTP stream ended unexpectedly")

    class Control:
        def request(self, method, resource, data, **kwargs):
            requests.append((method, resource, data, kwargs))
            if isinstance(raw_state, Exception):
                raise raw_state
            if raw_state is None:
                return {}
            return {"raw_csi": {"active": raw_state, "send_backpressure_total": 1, "raw_drop_total": 543}}

    receiver = DirectRawCSIReceiver("192.168.1.23", derive_complex=False)
    receiver._control = Control()
    receiver._raw_read_queue = ClosedQueue()
    receiver._raw_idle_check_at = time.monotonic() + 60.0
    receiver._open = lambda: None
    receiver._ensure_raw_reader = lambda: None
    receiver.running = True

    if raw_state is False:
        with pytest.raises(
            RuntimeError,
            match=r"session ended on the device "
            r"\(send backpressure: 1, dropped records: 543\)",
        ) as caught:
            receiver.run(timeout=1.0)
        assert isinstance(caught.value.__cause__, RuntimeError)
        assert "ended unexpectedly" in str(caught.value.__cause__)
    else:
        with pytest.raises(RuntimeError, match="ended unexpectedly"):
            receiver.run(timeout=1.0)
    assert len(requests) == 1
    assert requests[0][:3] == ("get", "diagnostics", {"fields": ["raw_csi"]})


def test_direct_raw_receiver_stop_unblocks_reader_before_closing_response():
    peer, client = socket.socketpair()
    try:
        client.setblocking(True)

        class BlockingResponse:
            def __init__(self):
                self._lock = threading.Lock()
                self.closed = False
                self.read_entered = threading.Event()

            def read1(self, _size):
                with self._lock:
                    self.read_entered.set()
                    try:
                        return client.recv(4096)
                    except OSError:
                        return b""

            def close(self):
                if not self._lock.acquire(timeout=0.3):
                    raise TimeoutError("response close blocked on the raw reader")
                try:
                    self.closed = True
                finally:
                    self._lock.release()

        class BlockingConnection:
            def __init__(self):
                self.sock = client
                self.closed = False

            def close(self):
                self.closed = True

        response = BlockingResponse()
        connection = BlockingConnection()
        receiver = DirectRawCSIReceiver("192.168.1.23", derive_complex=False)
        receiver._raw_response = response
        receiver._raw_connection = connection
        receiver.running = True
        receiver._open = lambda: None
        receiver._ensure_raw_reader()
        assert response.read_entered.wait(1.0)

        started_at = time.monotonic()
        receiver.stop()
        elapsed = time.monotonic() - started_at

        assert elapsed < 1.0
        assert response.closed is True
        assert connection.closed is True
        assert not receiver._raw_reader_is_busy(receiver._raw_reader_thread)
    finally:
        peer.close()
        client.close()


def test_direct_raw_receiver_stop_reads_final_diagnostics_before_closing_stream():
    events = []

    class Control:
        def request(self, verb, resource, *_args, **_kwargs):
            events.append((verb, resource))
            return {"raw_csi": {"raw_drop_total": 0, "fresh_record_total": 7}}

        def close(self):
            events.append("control_close")

    class Connection:
        def __init__(self):
            self.sock = self

        def shutdown(self, _mode):
            events.append("stream_shutdown")

        def close(self):
            events.append("stream_close")

    receiver = DirectRawCSIReceiver("192.168.1.23", derive_complex=False)
    receiver._control = Control()
    receiver._raw_connection = Connection()
    receiver.running = True

    receiver.stop()

    # A closed stream makes the device count its queued records as drops.
    assert events == [("get", "diagnostics"), "stream_shutdown", "stream_close", "control_close"]
    assert receiver.raw_final_fresh_record_total == 7


def test_direct_raw_receiver_rejects_capability_mismatch_without_transport_fallback():
    class IncompatibleControl:
        def __init__(self, *_args, **_kwargs):
            self.closed = False

        def request(self, verb, resource, *_args, **_kwargs):
            assert (verb, resource) == ("get", "capabilities")
            return {"features": {"csi": False}}

        def close(self):
            self.closed = True

    receiver = DirectRawCSIReceiver(
        "192.168.1.24",
        control_client_factory=IncompatibleControl,
    )

    with pytest.raises(RuntimeError, match="does not advertise"):
        receiver._open()
    assert receiver._control is None


def test_direct_raw_receiver_uses_shared_ghost_port_for_bare_targets():
    receiver = DirectRawCSIReceiver("192.168.1.23", derive_complex=False)

    assert receiver.control_endpoint == "http://192.168.1.23:62587/espectre/v1"
    assert receiver.raw_endpoint == "http://192.168.1.23:62587/espectre/v1/csi"


def test_direct_raw_receiver_parses_aggregated_frames_and_rejects_sequence_mismatch():
    session_id = bytes.fromhex("00112233445566778899aabbccddeeff")

    def frame(stream_sequence, record_sequence=None):
        sequence = stream_sequence if record_sequence is None else record_sequence
        record = build_packet(
            version=RAW_CSI_RECORD_VERSION_V8,
            seq_num=sequence,
            request_accepted_total=sequence,
            fresh_record_total=stream_sequence,
        )
        prefix = RAW_CSI_HTTP_FRAME_STRUCT.pack(
            RAW_CSI_RESPONSE_MAGIC,
            RAW_CSI_PROTOCOL_VERSION,
            8,
            RAW_CSI_HTTP_FRAME_STRUCT.size,
            session_id,
            stream_sequence,
            len(record),
            0,
            stream_sequence,
            0,
            0,
        )
        return prefix + record

    receiver = DirectRawCSIReceiver("192.168.1.23", derive_complex=False)
    receiver._session_id = session_id
    receiver._raw_buffer.extend(frame(1) + frame(2))
    receiver._consume_raw_frames()
    assert [packet.raw_stream_sequence for packet in receiver.buffer] == [1, 2]

    mismatched = DirectRawCSIReceiver("192.168.1.23", derive_complex=False)
    mismatched._session_id = session_id
    mismatched._raw_buffer.extend(frame(1, record_sequence=2))
    with pytest.raises(RuntimeError, match="invalid V8 record"):
        mismatched._consume_raw_frames()


def test_parse_packet_reads_phy_metadata():
    receiver = CSIReceiver(bind_host='127.0.0.1')
    packet = receiver._parse_packet(
        build_packet(
            phy_mode=2,
            ltf_type=2,
            channel_width=2,
        )
    )

    assert packet is not None
    assert packet.phy_mode == 'ht'
    assert packet.ltf_type == 'ht-ltf'
    assert packet.channel_width == '40'


@pytest.mark.parametrize("version", [RAW_CSI_RECORD_VERSION - 1, RAW_CSI_RECORD_VERSION + 1])
def test_parse_packet_rejects_unsupported_record_version(version):
    receiver = CSIReceiver(bind_host='127.0.0.1')
    packet_data = build_packet(version=version)

    assert receiver._parse_packet(packet_data) is None


def test_parse_packet_preserves_short_transport_payload():
    receiver = CSIReceiver(bind_host='127.0.0.1')
    payload = []
    for idx in range(12):
        payload.extend([idx + 1, -(idx + 1)])

    packet = receiver._parse_packet(build_packet(seq_num=9, payload=payload))

    assert packet is not None
    assert packet.seq_num == 9
    assert packet.num_subcarriers == 12
    assert packet.iq_raw.shape == (24,)
    np.testing.assert_array_equal(packet.iq_raw, np.array(payload, dtype=np.int8))


def test_parse_packet_derives_complex_data_lazily_when_disabled():
    receiver = CSIReceiver(bind_host='127.0.0.1', derive_complex=False)
    packet = receiver._parse_packet(build_packet(seq_num=8, payload=[10, 20, -30, 40]))

    assert packet is not None
    assert packet._iq_complex is None
    np.testing.assert_allclose(packet.iq_complex, np.array([20 + 10j, 40 - 30j], dtype=np.complex64))
    assert packet._iq_complex is not None


def test_external_traffic_generator_configures_low_latency_multicast(monkeypatch):
    class FakeSocket:
        def __init__(self):
            self.options = []

        def setsockopt(self, level, option, value):
            self.options.append((level, option, value))

    sock = FakeSocket()
    monkeypatch.setattr(espectre_traffic_generator, "TARGETS", ["239.255.0.1"])

    espectre_traffic_generator.configure_socket(sock)

    assert (socket.IPPROTO_IP, socket.IP_TOS, 46 << 2) in sock.options
    assert (socket.IPPROTO_IP, socket.IP_MULTICAST_TTL, 8) in sock.options


def test_external_traffic_generator_uses_canonical_ghost_marker():
    assert espectre_traffic_generator.ExternalTrafficGenerator.TRAFFIC_MARKER == "👻"
    assert espectre_traffic_generator.ExternalTrafficGenerator.PAYLOAD == bytes.fromhex("f09f91bb")


def test_external_traffic_generator_help_exits_successfully(capsys):
    assert espectre_traffic_generator.main(["--help"]) == 0
    output = capsys.readouterr().out
    assert "python3 espectre_traffic_generator.py start" in output
    assert "python3 espectre_traffic_generator.py run" in output


@pytest.mark.parametrize("targets,multicast_ttl", [
    (["127.0.0.1", "127.0.0.2"], 8),
    (["127.0.0.1", "239.255.0.1"], 1),
    (["239.255.0.1", "239.255.0.2"], 255),
])
@pytest.mark.parametrize("dscp", [0, 46, 63])
def test_external_traffic_generator_rates_each_target_and_stops_safely(
    monkeypatch, targets, multicast_ttl, dscp,
):
    sockets = []

    class FakeSocket:
        def __init__(self):
            self.options = []
            self.bound = None
            self.sent = []
            self.closed = False
            sockets.append(self)

        def setsockopt(self, level, option, value):
            self.options.append((level, option, value))

        def bind(self, address):
            self.bound = address

        def sendto(self, payload, destination):
            self.sent.append((payload, destination))
            if len(self.sent) == 4:
                generator._stop_event.set()

        def close(self):
            self.closed = True

    monkeypatch.setattr(espectre_traffic_generator.socket, "socket", lambda *_args: FakeSocket())
    generator = espectre_traffic_generator.ExternalTrafficGenerator(
        targets,
        rate_pps=1000,
        source_ip="127.0.0.3",
        multicast_ttl=multicast_ttl,
        dscp=dscp,
    )

    with generator as active:
        deadline = time.monotonic() + 1.0
        while active.running and time.monotonic() < deadline:
            time.sleep(0.001)

    assert not generator.running
    assert generator.sent_packets == 4
    assert generator.sent_by_target == dict.fromkeys(targets, 2)
    assert generator.errors_by_target == dict.fromkeys(targets, 0)
    assert sockets[0].bound == ("127.0.0.3", 0)
    options = {(level, option): value for level, option, value in sockets[0].options}
    assert options[socket.IPPROTO_IP, socket.IP_TOS] == dscp << 2
    assert (socket.IPPROTO_IP, socket.IP_TTL) not in options  # Preserve the OS unicast TTL.
    if "239.255.0.1" in targets:
        assert options[socket.IPPROTO_IP, socket.IP_MULTICAST_TTL] == multicast_ttl
        assert options[socket.IPPROTO_IP, socket.IP_MULTICAST_IF] == socket.inet_aton("127.0.0.3")
    else:
        assert (socket.IPPROTO_IP, socket.IP_MULTICAST_TTL) not in options
        assert (socket.IPPROTO_IP, socket.IP_MULTICAST_IF) not in options
    assert all(payload == bytes.fromhex("f09f91bb") for payload, _destination in sockets[0].sent)
    assert sockets[0].closed


class _FakeIcmpSocket:
    """ICMP socket double: records sends and returns queued packets once per drain."""

    def __init__(self, replies=()):
        self.options = []
        self.bound = None
        self.sent = []
        self.replies = list(replies)
        self.closed = False
        self.generator = None

    def setsockopt(self, level, option, value):
        self.options.append((level, option, value))

    def bind(self, address):
        self.bound = address

    def getsockname(self):
        # Linux assigns a ping socket's Echo identifier as its local port.
        return ("0.0.0.0", 0x0102)

    def sendto(self, payload, destination):
        self.sent.append((payload, destination))
        if len(self.sent) == 4:
            self.generator._stop_event.set()

    def recv(self, _size):
        if self.replies:
            return self.replies.pop(0)
        raise BlockingIOError

    def close(self):
        self.closed = True


def _echo_reply(identifier, with_ip_header=False):
    reply = bytes([0, 0, 0, 0]) + identifier.to_bytes(2, "big") + bytes(2)
    return (bytes([0x45]) + bytes(19) + reply) if with_ip_header else reply


def _poll_fake_icmp_socket(monkeypatch, fake):
    monkeypatch.setattr(espectre_traffic_generator.select, "select",
                        lambda read, _write, _error, _timeout: (read if fake.replies else [], [], []))


def _run_until_stopped(generator):
    with generator as active:
        deadline = time.monotonic() + 1.0
        while active.running and time.monotonic() < deadline:
            time.sleep(0.001)
    assert not generator.running


def test_external_traffic_generator_sends_empty_echo_requests_and_counts_replies(monkeypatch):
    fake = _FakeIcmpSocket(replies=[_echo_reply(0x0102), _echo_reply(0x0103), b"short"])
    opened = []
    _poll_fake_icmp_socket(monkeypatch, fake)

    def open_socket(*args):
        opened.append(args)
        return fake

    monkeypatch.setattr(espectre_traffic_generator.socket, "socket", open_socket)
    generator = espectre_traffic_generator.ExternalTrafficGenerator(
        ["127.0.0.1", "239.255.0.1"], rate_pps=1000, source_ip="127.0.0.3", protocol="icmp",
    )
    fake.generator = generator

    _run_until_stopped(generator)

    assert opened == [(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_ICMP)]
    assert fake.closed
    assert generator.sent_by_target == {"127.0.0.1": 2, "239.255.0.1": 2}
    # Only the reply carrying the socket's identifier counts.
    assert generator.replies_received == 1
    options = {(level, option): value for level, option, value in fake.options}
    assert options[socket.IPPROTO_IP, socket.IP_MULTICAST_IF] == socket.inet_aton("127.0.0.3")
    sequences = []
    for payload, destination in fake.sent:
        assert destination[1] == 0
        assert len(payload) == 8 and payload[:2] == bytes([8, 0])
        assert espectre_traffic_generator.icmp_checksum(payload) == 0
        sequences.append(int.from_bytes(payload[6:8], "big"))
    assert sequences == [0, 0, 1, 1]


def test_external_traffic_generator_falls_back_to_a_raw_icmp_socket(monkeypatch):
    fake = _FakeIcmpSocket()
    kinds = []
    _poll_fake_icmp_socket(monkeypatch, fake)

    def open_socket(_family, kind, _protocol):
        kinds.append(kind)
        if kind == socket.SOCK_DGRAM:
            raise PermissionError("ping sockets disabled")
        return fake

    monkeypatch.setattr(espectre_traffic_generator.socket, "socket", open_socket)
    generator = espectre_traffic_generator.ExternalTrafficGenerator(
        ["127.0.0.1"], rate_pps=1000, protocol="icmp",
    )
    fake.generator = generator
    original_send = fake.sendto

    def send_and_queue_replies(payload, destination):
        identifier = int.from_bytes(payload[4:6], "big")
        # A raw socket also sees replies to other programs on the host.
        fake.replies[:] = [_echo_reply(identifier, with_ip_header=True),
                           _echo_reply(identifier ^ 0xFFFF, with_ip_header=True)]
        original_send(payload, destination)

    fake.sendto = send_and_queue_replies
    _run_until_stopped(generator)

    assert kinds == [socket.SOCK_DGRAM, socket.SOCK_RAW]
    assert generator.replies_received == 3


def test_external_traffic_generator_explains_unavailable_icmp_sockets(monkeypatch):
    def deny(*_args):
        raise PermissionError("denied")

    monkeypatch.setattr(espectre_traffic_generator.socket, "socket", deny)
    with pytest.raises(OSError, match="ping_group_range"):
        espectre_traffic_generator.open_icmp_socket()


def test_external_traffic_generator_refuses_icmp_on_windows(monkeypatch):
    opened = []
    monkeypatch.setattr(espectre_traffic_generator.socket, "socket", lambda *args: opened.append(args))
    with monkeypatch.context() as patch:
        patch.setattr(espectre_traffic_generator.os, "name", "nt")
        with pytest.raises(OSError, match="not supported on Windows"):
            espectre_traffic_generator.open_icmp_socket()
    assert opened == []


@pytest.mark.parametrize("protocol", ["tcp", "ICMP", "", None, 1])
def test_external_traffic_generator_rejects_unknown_protocol(tmp_path, protocol):
    with pytest.raises(ValueError, match="protocol"):
        espectre_traffic_generator.ExternalTrafficGenerator(["239.255.0.1"], protocol=protocol)
    options_file = tmp_path / "options.json"
    options_file.write_text(json.dumps({
        "targets": ["239.255.0.1"], "port": 5555, "rate_pps": 100, "protocol": protocol,
    }), encoding="utf-8")
    with pytest.raises(ValueError, match="protocol"):
        traffic_addon.load_options(options_file)


@pytest.mark.parametrize("multicast_ttl", [-1, 0, 256, True, 1.5, "8", None])
def test_external_traffic_generator_rejects_invalid_multicast_ttl(tmp_path, multicast_ttl):
    with pytest.raises(ValueError, match="multicast_ttl"):
        espectre_traffic_generator.ExternalTrafficGenerator(
            ["239.255.0.1"], multicast_ttl=multicast_ttl,
        )
    options_file = tmp_path / "options.json"
    options_file.write_text(json.dumps({
        "targets": ["239.255.0.1"], "port": 5555, "rate_pps": 100,
        "multicast_ttl": multicast_ttl,
    }), encoding="utf-8")
    with pytest.raises(ValueError, match="multicast_ttl"):
        traffic_addon.load_options(options_file)


@pytest.mark.parametrize("dscp", [-1, 64, 184, True, False, 1.5, "46", None])
def test_external_traffic_generator_rejects_invalid_dscp(tmp_path, dscp):
    with pytest.raises(ValueError, match="dscp"):
        espectre_traffic_generator.ExternalTrafficGenerator(["239.255.0.1"], dscp=dscp)
    options_file = tmp_path / "options.json"
    options_file.write_text(json.dumps({
        "targets": ["239.255.0.1"], "port": 5555, "rate_pps": 100, "dscp": dscp,
    }), encoding="utf-8")
    with pytest.raises(ValueError, match="dscp"):
        traffic_addon.load_options(options_file)


@pytest.mark.parametrize("overrides", [
    {}, {"multicast_ttl": 1}, {"multicast_ttl": 255},
    {"dscp": 0}, {"dscp": 63}, {"multicast_ttl": 1, "dscp": 0},
    {"protocol": "icmp"}, {"protocol": "udp"},
])
def test_traffic_addon_passes_network_options_to_shared_generator(tmp_path, monkeypatch, overrides):
    options_file = tmp_path / "options.json"
    options_file.write_text(json.dumps({
        "targets": ["239.255.0.1"], "port": 5555, "rate_pps": 100,
        "source_ip": "192.168.1.10", **overrides,
    }), encoding="utf-8")
    options = traffic_addon.load_options(options_file)
    observed = []
    monkeypatch.setattr(traffic_addon, "load_options", lambda: options)
    monkeypatch.setattr(traffic_addon, "run", lambda **kwargs: observed.append(kwargs))

    traffic_addon.main()

    assert observed == [{
        "targets": ["239.255.0.1"], "port": 5555, "rate_pps": 100,
        "source_ip": "192.168.1.10", "multicast_ttl": 8, "dscp": 46, "protocol": "udp",
        **overrides,
    }]


_TRAFFIC_SOURCE_OPTIONS = ["ping", "dns", "dns_tcp", "wifi_raw", "external"]


def _ha_panel_inventory(platform="esphome", version=3, source="ping"):
    devices = [{"id": "sensor-a", "name": "ESPectre kitchen", "name_by_user": "Kitchen",
                "manufacturer": "ESPectre" if platform == "mqtt" else "Espressif", "area_id": "kitchen"}]
    entities, states = [], []
    values = {"source": source, "spacing": "half_slot", "refresh": "unknown",
              "generator": "0", "traffic": "5", "traffic_rx": "100",
              "accepted": "99.5", "occupancy": "98", "rssi": "-54"}
    options = {"source": _TRAFFIC_SOURCE_OPTIONS, "spacing": ha_client.SAMPLE_SPACINGS}
    roles = {"source": ("select", ("csi_traffic_source",)), "spacing": ("select", ("csi_packet_spacing",)),
             **ha_client.ENTITY_ROLES, "rssi": ("sensor", ("wifi_rssi",))}
    for role, (domain, (suffix, *_)) in roles.items():
        name = suffix.replace("_", " ").title()
        if platform == "mqtt":
            unique_id = "espectre_0123456789abcdef_" + suffix
        elif version == 3:
            unique_id = f"aabbccddeeff/0/{domain}/{name}"
        else:
            unique_id = f"aabbccddeeff-{domain}-" + (suffix if version == 1 else name)
        entity_id = domain + ".renamed_" + role
        entities.append({"entity_id": entity_id, "device_id": "sensor-a", "platform": platform,
                         "unique_id": unique_id, "name": "A custom HA name", "disabled_by": None,
                         "entity_category": "diagnostic" if role == "rssi" else None})
        states.append({"entity_id": entity_id, "state": values[role],
                       "attributes": {"options": list(options[role])} if role in options else
                       {"device_class": "signal_strength", "unit_of_measurement": "dBm"} if role == "rssi" else {},
                       "last_updated": "2026-09-19T10:00:00+00:00"})
    return devices, entities, states, [{"area_id": "kitchen", "name": "Kitchen"}]


@pytest.mark.parametrize("platform,version", [("esphome", 1), ("esphome", 2), ("esphome", 3), ("mqtt", 1)])
def test_ha_panel_recognizes_registry_identity_after_ha_renames(platform, version):
    inventory = _ha_panel_inventory(platform, version)
    row, = ha_client.build_inventory(*inventory)
    assert row["name"] == row["area"] == "Kitchen"
    assert row["can_control"] and row["can_refresh"]
    assert row["fields"]["ownership"]["value"] == "internal"
    assert row["fields"]["generator"]["value"] == 0
    assert row["fields"]["traffic"]["value"] == 5
    assert row["fields"]["traffic_rx"]["value"] == 100
    assert row["fields"]["accepted"]["value"] == 99.5
    assert row["fields"]["occupancy"]["value"] == 98
    assert row["fields"]["rssi"]["value"] == -54


@pytest.mark.parametrize("source,ownership,internal_mode", [
    ("dns", "internal", "dns"), ("external", "external", None)])
def test_ha_panel_derives_ownership_from_the_single_traffic_source_select(source, ownership, internal_mode):
    row, = ha_client.build_inventory(*_ha_panel_inventory(source=source))
    assert row["can_control"]
    assert row["fields"]["ownership"]["value"] == ownership
    assert row["fields"]["ownership"]["entity_id"] == "select.renamed_source"
    assert row["internal_mode"] == internal_mode


@pytest.mark.parametrize("manufacturer", ["ESPectre", "https://espectre.dev"])
def test_ha_panel_excludes_supervisor_apps_from_sensor_inventory(manufacturer):
    devices, entities, states, areas = _ha_panel_inventory("mqtt")
    devices[0]["entry_type"] = None
    devices.append({
        "id": "traffic-app", "name": "ESPectre Traffic Generator",
        "name_by_user": "Renamed traffic app", "manufacturer": manufacturer,
        "model": "Home Assistant Add-on", "entry_type": "service",
        "identifiers": [["hassio", "b04e5925_espectre_traffic_generator"]],
    })
    entities.append({
        "entity_id": "sensor.renamed_app_cpu", "device_id": "traffic-app",
        "platform": "hassio", "unique_id": "b04e5925_espectre_traffic_generator_cpu_percent",
        "disabled_by": "integration",
    })

    row, = ha_client.build_inventory(devices, entities, states, areas)

    assert row["id"] == "sensor-a"
    assert row["can_control"] and row["can_refresh"]


@pytest.mark.parametrize("metadata,expected", [
    ({"model": "ESP32-C3"}, "ESP32-C3"),
    ({"model": "ESPectre (esp32s3)"}, "ESP32-S3"),
    ({"model": "ESP32", "hw_version": "ESP32-C6"}, "ESP32-C6"),
    ({"model_id": "ESP32-S2"}, "ESP32-S2"),
    ({"model": "ESP32"}, "ESP32"),
    ({"model": "espectre", "name": "ESP32-C3"}, None),
    ({"model": "ESP32-C3", "hw_version": "ESP32-S3"}, None),
    ({}, None),
])
def test_ha_panel_chip_uses_explicit_hardware_metadata(metadata, expected):
    inventory = _ha_panel_inventory()
    inventory[0][0].update(metadata)
    assert ha_client.build_inventory(*inventory)[0]["chip"] == expected


@pytest.mark.parametrize("url,expected", [
    ("http://192.168.1.42:80", "192.168.1.42"),
    ("https://[2001:db8::42]:443/", "2001:db8::42"),
    ("http://espectre.local", None),
    ("homeassistant://app/esphome", None),
    ("http://[invalid", None),
    (None, None),
])
def test_ha_panel_device_ip_uses_only_literal_registry_address(url, expected):
    inventory = _ha_panel_inventory()
    inventory[0][0]["configuration_url"] = url
    row, = ha_client.build_inventory(*inventory)
    assert row["ip_address"] == expected


def test_ha_panel_dhcp_matches_mac_not_name_and_updates_addresses():
    inventory = _ha_panel_inventory()
    device = inventory[0][0]
    device["connections"] = [["mac", "AA:BB:CC:DD:EE:FF"]]
    device["configuration_url"] = "http://192.168.1.9"
    ha_client.network_addresses(inventory[0], {"aabbccddeeff": "192.168.1.4"})
    assert ha_client.build_inventory(*inventory)[0]["ip_address"] == "192.168.1.4"
    ha_client.network_addresses(inventory[0], {"aabbccddeeff": "192.168.1.5"})
    assert ha_client.build_inventory(*inventory)[0]["ip_address"] == "192.168.1.5"
    ha_client.network_addresses(inventory[0], {"112233445566": "192.168.1.4"})
    assert ha_client.build_inventory(*inventory)[0]["ip_address"] == "192.168.1.9"


@pytest.mark.parametrize("case", ["disabled", "device_disabled", "offline", "ambiguous", "options"])
def test_ha_panel_rejects_unavailable_or_ambiguous_controls(case):
    devices, entities, states, areas = _ha_panel_inventory("mqtt")
    if case == "disabled":
        entities[0]["disabled_by"] = "user"
    elif case == "device_disabled":
        devices[0]["disabled_by"] = "user"
    elif case == "offline":
        states[0]["state"] = "unavailable"
    elif case == "ambiguous":
        entities.append(dict(entities[0], entity_id="select.duplicate"))
        states.append(dict(states[0], entity_id="select.duplicate"))
    else:
        states[0]["attributes"]["options"] = ["ping", "dns", "dns_tcp"]
    row, = ha_client.build_inventory(devices, entities, states, areas)
    assert not row["can_control"]
    assert row["reason"]


def test_ha_panel_recognizes_devices_only_by_traffic_source_options():
    devices, entities, states, areas = _ha_panel_inventory()
    for entity in entities:
        entity["unique_id"] = "unrelated-identifier"
        entity["name"] = "CSI Accepted Rate"
    row, = ha_client.build_inventory(devices, entities, states, areas)
    assert row["can_control"] and not row["can_refresh"]
    assert row["fields"]["accepted"]["status"] == "missing"
    for options in (["ping", "dns"], ["internal", "external"]):
        changed = [dict(states[0], attributes={"options": options}), *states[1:]]
        assert ha_client.build_inventory(devices, entities, changed, areas) == []
    # HA gives disabled entities no state, so a disabled select hides the device.
    assert ha_client.build_inventory(devices, entities, states[1:], areas) == []


_UPSTREAM_ROLES = ["source", "generator", "traffic", "traffic_rx", "accepted", "occupancy"]
_UPSTREAM_RECOMMENDED = ["CSI Traffic Source", "Generator Rate", "Traffic TX Rate", "Traffic RX Rate",
                         "CSI Accepted Rate", "CSI Temporal Occupancy"]
_UPSTREAM_CUSTOM = ["Sorgente", "Generati", "Inviati", "Ricevuti", "CSI utili", "Copertura"]
# Named after the `diagnostics` config keys; only `csi_occupancy` differs from the recommended names.
_UPSTREAM_CONFIG_KEYS = [*_UPSTREAM_RECOMMENDED[:-1], "CSI Occupancy"]


def _upstream_unique_id(domain, name, layout):
    # ESPHome v3: mac/sub-device/domain/name; v1/v2: mac-domain-name with an optional @n.
    return f"a0b1c2d3e4f5/0/{domain}/{name}" if layout == "v3" else f"a0b1c2d3e4f5-{domain}-{name}@1"


def _upstream_inventory(names=_UPSTREAM_RECOMMENDED, options=_TRAFFIC_SOURCE_OPTIONS,
                        diagnostics=True, signals=(("WiFi Signal", "diagnostic"),),
                        layout="v3", refresh=None):
    """An ESPHome device with the upstream `espectre` component: user-chosen names,
    a board model, ESPHome's own `wifi_signal` sensor, and an optional template
    button named `refresh`."""
    devices = [{"id": "upstream-a", "name": "living-room", "manufacturer": "Espressif",
                "model": "esp32-c6-devkitc-1"}]
    values = ["ping", "100.0", "101.0", "3.0", "99.0", "97"]
    entities, states = [], []
    rows = list(zip(_UPSTREAM_ROLES, names, values, strict=True))[:None if diagnostics else 1]
    for role, name, value in rows:
        domain = "select" if role == "source" else "sensor"
        entities.append({"entity_id": f"{domain}.living_room_{role}", "device_id": "upstream-a",
                         "platform": "esphome", "unique_id": _upstream_unique_id(domain, name, layout)})
        states.append({"entity_id": entities[-1]["entity_id"], "state": value,
                       "attributes": {"options": list(options)} if role == "source" else {}})
    if refresh:
        entities.append({"entity_id": "button.living_room_refresh", "device_id": "upstream-a",
                         "platform": "esphome", "unique_id": _upstream_unique_id("button", refresh, layout)})
        states.append({"entity_id": "button.living_room_refresh", "state": "unknown", "attributes": {}})
    for index, (name, category) in enumerate(signals):
        entities.append({"entity_id": f"sensor.living_room_signal_{index}", "device_id": "upstream-a",
                         "platform": "esphome", "unique_id": _upstream_unique_id("sensor", name, layout),
                         "entity_category": category})
        states.append({"entity_id": entities[-1]["entity_id"], "state": str(-61 - index),
                       "attributes": {"device_class": "signal_strength", "unit_of_measurement": "dBm"}})
    return devices, entities, states, []


@pytest.mark.parametrize("names,options,diagnostics,layout,refresh", [
    (_UPSTREAM_RECOMMENDED, _TRAFFIC_SOURCE_OPTIONS, True, "v3", None),
    (_UPSTREAM_CUSTOM, _TRAFFIC_SOURCE_OPTIONS, True, "v3", None),
    (_UPSTREAM_RECOMMENDED, ["ping", "dns", "dns_tcp", "external"], True, "v3", None),  # ESP32-C6
    (_UPSTREAM_RECOMMENDED, _TRAFFIC_SOURCE_OPTIONS, False, "v3", None),
    (_UPSTREAM_CONFIG_KEYS, _TRAFFIC_SOURCE_OPTIONS, True, "v3", None),
    (_UPSTREAM_CONFIG_KEYS, _TRAFFIC_SOURCE_OPTIONS, True, "legacy", None),
    (_UPSTREAM_RECOMMENDED, _TRAFFIC_SOURCE_OPTIONS, True, "v3", "Refresh Diagnostics"),
    (_UPSTREAM_RECOMMENDED, _TRAFFIC_SOURCE_OPTIONS, True, "legacy", "Refresh Diagnostics"),
], ids=["recommended", "custom", "c6", "no_diagnostics", "config_keys", "config_keys_legacy",
        "refresh", "refresh_legacy"])
def test_ha_panel_recognizes_the_upstream_esphome_component(names, options, diagnostics, layout, refresh):
    row, = ha_client.build_inventory(*_upstream_inventory(
        names, options, diagnostics, layout=layout, refresh=refresh))
    assert row["can_control"] and row["reason"] is None and row["chip"] == "ESP32-C6"
    assert row["can_refresh"] == bool(refresh)
    assert row["fields"]["refresh"].get("entity_id") == ("button.living_room_refresh" if refresh else None)
    assert row["fields"]["ownership"]["value"] == "internal" and row["internal_mode"] == "ping"
    assert row["fields"]["rssi"]["value"] == -61
    recognized = diagnostics and names is not _UPSTREAM_CUSTOM
    assert [row["fields"][role]["value"] for role in _UPSTREAM_ROLES[1:]] == (
        [100, 101, 3, 99, 97] if recognized else [None] * 5)


def test_ha_panel_reads_the_packet_spacing_select_of_newer_upstream_versions():
    """ESPHome after 2026.10 adds CSI Packet Spacing and Detection Profile selects."""
    devices, entities, states, areas = _upstream_inventory(refresh="Refresh Diagnostics")
    for role, name, options, value in (
            ("profile", "Detection Profile", ["lightweight", "high_accuracy"], "lightweight"),
            ("spacing", "CSI Packet Spacing", list(ha_client.SAMPLE_SPACINGS), "distinct")):
        entities.append({"entity_id": f"select.living_room_{role}", "device_id": "upstream-a",
                         "platform": "esphome", "unique_id": _upstream_unique_id("select", name, "v3")})
        states.append({"entity_id": entities[-1]["entity_id"], "state": value,
                       "attributes": {"options": options}})
    row, = ha_client.build_inventory(devices, entities, states, areas)
    assert row["can_control"] and row["can_refresh"] and row["can_set_spacing"]
    assert row["fields"]["spacing"]["value"] == "distinct"
    assert row["fields"]["ownership"]["entity_id"] == "select.living_room_source"


def test_ha_panel_drops_a_field_that_two_upstream_entities_claim():
    devices, entities, states, areas = _upstream_inventory()
    occupancy = entities[_UPSTREAM_ROLES.index("occupancy")]
    entities.append(dict(occupancy, entity_id="sensor.living_room_csi_occupancy",
                         unique_id="a0b1c2d3e4f5/0/sensor/CSI Occupancy"))
    states.append({"entity_id": "sensor.living_room_csi_occupancy", "state": "96", "attributes": {}})
    row, = ha_client.build_inventory(devices, entities, states, areas)
    assert row["fields"]["occupancy"]["status"] == "missing"
    assert row["fields"]["accepted"]["value"] == 99


@pytest.mark.parametrize("signals,expected", [
    # A BLE beacon's RSSI shares the type but not the diagnostic category.
    ((("BLE Beacon", None), ("WiFi Signal", "diagnostic"), ("WiFi RSSI", "diagnostic")),
     "sensor.living_room_signal_1"),
    ((("BLE Beacon", None),), None),
])
def test_ha_panel_takes_rssi_from_the_first_diagnostic_signal_strength_sensor(signals, expected):
    row, = ha_client.build_inventory(*_upstream_inventory(signals=signals))
    assert row["fields"]["rssi"].get("entity_id") == expected
    assert row["can_control"]


@pytest.mark.parametrize("value", ["unavailable", "unknown", "nan", "inf", "bad"])
@pytest.mark.parametrize("role", ["generator", "traffic", "traffic_rx", "accepted"])
def test_ha_panel_missing_diagnostics_are_not_zero(value, role):
    inventory = _ha_panel_inventory()
    state = next(state for state in inventory[2] if state["entity_id"] == "sensor.renamed_" + role)
    state["state"] = value
    row, = ha_client.build_inventory(*inventory)
    assert row["fields"][role]["value"] is None


class _PanelHomeAssistantServer:
    """Exercise the actual internal WebSocket protocol without a live HA instance."""

    def __init__(self):
        self.inventory = _ha_panel_inventory()
        self.calls = []
        self.confirm = True
        self.reject = False
        self.subscriptions = {}
        self.connections = 0
        self.emit_diagnostics = False

    async def emit(self, event_type, data):
        for websocket, subscriptions in list(self.subscriptions.items()):
            if not websocket.closed and event_type in subscriptions:
                await websocket.send_json({"id": subscriptions[event_type], "type": "event",
                                           "event": {"event_type": event_type, "data": data}})

    async def websocket(self, request):
        from aiohttp import web
        websocket = web.WebSocketResponse()
        await websocket.prepare(request)
        await websocket.send_json({"type": "auth_required"})
        auth = await websocket.receive_json()
        assert auth == {"type": "auth", "access_token": "test-supervisor-secret"}
        await websocket.send_json({"type": "auth_ok"})
        self.connections += 1
        self.subscriptions[websocket] = {}
        async for message in websocket:
            command = json.loads(message.data)
            self.calls.append(command)
            kind = command["type"]
            if kind == "dhcp/subscribe_discovery":
                await websocket.send_json({"id": command["id"], "type": "result", "success": True, "result": None})
                await websocket.send_json({"id": command["id"], "type": "event", "event": {"add": [
                    {"mac_address": "AA:BB:CC:DD:EE:FF", "ip_address": "192.168.1.4", "hostname": "espectre"}
                ]}})
                continue
            if kind == "subscribe_events":
                self.subscriptions[websocket][command["event_type"]] = command["id"]
                result = None
            elif kind == "call_service":
                if self.reject:
                    await websocket.send_json({"id": command["id"], "type": "result", "success": False,
                                               "error": {"message": "private details"}})
                    continue
                if self.confirm and command["domain"] == "select":
                    target = command["target"]["entity_id"]
                    state = next(state for state in self.inventory[2] if state["entity_id"] == target)
                    state["state"] = command["service_data"]["option"]
                if self.emit_diagnostics and command["domain"] == "button":
                    state = next(state for state in self.inventory[2] if state["entity_id"] == "sensor.renamed_accepted")
                    state["state"] = str(float(state["state"]) + 1)
                    await self.emit("state_changed", {"entity_id": state["entity_id"], "new_state": state})
                result = {"context": {"id": "test"}}
            else:
                index = {"config/device_registry/list": 0, "config/entity_registry/list": 1,
                         "get_states": 2, "config/area_registry/list": 3}[kind]
                result = self.inventory[index]
            await websocket.send_json({"id": command["id"], "type": "result", "success": True, "result": result})
        return websocket


@pytest.mark.parametrize("status,payload,expected", [
    (200, {"data": {"storage_data": {"device_info": {"model": "esp32-c3"}}}}, "ESP32-C3"),
    (200, {"data": {"storage_data": {"device_info": {"model": "esp32-s3"}}}}, "ESP32-S3"),
    (200, {"data": None}, None),
    (200, {"data": {}}, None),
    (401, {}, None),
    (404, {}, None),
])
def test_ha_panel_esphome_model_uses_optional_cached_diagnostics(status, payload, expected):
    async def exercise():
        from aiohttp import web
        from aiohttp.test_utils import TestServer
        fake = _PanelHomeAssistantServer()
        fake.inventory[0][0].update(model="espectre", config_entries=["entry-a"])
        for entity in fake.inventory[1]:
            entity["config_entry_id"] = "entry-a"
        requests = []

        async def diagnostics(request):
            assert request.headers["Authorization"] == "Bearer test-supervisor-secret"
            requests.append(request.path)
            return web.json_response(payload, status=status)

        app = web.Application()
        app.router.add_get("/core/websocket", fake.websocket)
        app.router.add_get("/core/api/diagnostics/config_entry/entry-a", diagnostics)
        async with TestServer(app) as server:
            ha = ha_client.HomeAssistant("test-supervisor-secret",
                                         endpoint=str(server.make_url("/core/websocket")).replace("http:", "ws:"))
            for _ in range(2):
                row, = await ha.snapshot()
                assert row["chip"] == expected
                assert row["can_control"]
            assert len(requests) == 1
            fake.inventory[0][0]["sw_version"] = "new firmware"
            await ha.snapshot()
            assert len(requests) == 2
            assert not any(call["type"] == "call_service" for call in fake.calls)
            fake.inventory[0][0]["model"] = "ESP32-C6"
            assert (await ha.snapshot())[0]["chip"] == "ESP32-C6"
            assert len(requests) == 2
    asyncio.run(exercise())


@pytest.mark.parametrize("outcome", ["confirmed", "unconfirmed", "error"])
def test_ha_panel_actions_use_internal_services_and_check_returned_state(outcome):
    async def exercise():
        from aiohttp import web
        from aiohttp.test_utils import TestServer
        fake = _PanelHomeAssistantServer()
        fake.confirm, fake.reject = outcome == "confirmed", outcome == "error"
        app = web.Application()
        app.router.add_get("/core/websocket", fake.websocket)
        async with TestServer(app) as server:
            ha = ha_client.HomeAssistant("test-supervisor-secret",
                                         endpoint=str(server.make_url("/core/websocket")).replace("http:", "ws:"),
                                         confirmation_timeout=0)
            assert (await ha.snapshot())[0]["can_control"]
            assert all(call["type"] != "call_service" for call in fake.calls)
            data = await ha.act(["sensor-a", "unknown-device"], "external")
            assert [result["status"] for result in data["results"]] == [outcome, "error"]
            writes = [call for call in fake.calls if call["type"] == "call_service"]
            assert len(writes) == 1
            assert writes[0]["target"] == {"entity_id": "select.renamed_source"}
            assert writes[0]["service_data"] == {"option": "external"}
            assert "test-supervisor-secret" not in json.dumps(data)
            assert "private details" not in json.dumps(data)
            if outcome == "confirmed":
                assert (await ha.act(["sensor-a"], "external"))["results"][0]["status"] == "unchanged"
                assert (await ha.act(["sensor-a"], "internal"))["results"][0]["status"] == "confirmed"
    asyncio.run(exercise())


def test_ha_panel_single_source_select_restores_the_previous_internal_packet():
    async def exercise():
        from aiohttp import web
        from aiohttp.test_utils import TestServer
        fake = _PanelHomeAssistantServer()
        fake.inventory = _ha_panel_inventory(source="dns")
        app = web.Application()
        app.router.add_get("/core/websocket", fake.websocket)
        async with TestServer(app) as server:
            ha = ha_client.HomeAssistant("test-supervisor-secret",
                                         endpoint=str(server.make_url("/core/websocket")).replace("http:", "ws:"),
                                         confirmation_timeout=0)
            assert (await ha.act(["sensor-a"], "external"))["results"][0]["status"] == "confirmed"
            assert (await ha.act(["sensor-a"], "internal"))["results"][0]["status"] == "confirmed"
            # Any option the select offers works directly, and `internal` keeps the last internal one.
            assert (await ha.act(["sensor-a"], "wifi_raw"))["results"][0]["option"] == "wifi_raw"
            assert (await ha.act(["sensor-a"], "external"))["results"][0]["status"] == "confirmed"
            restored, = (await ha.act(["sensor-a"], "internal"))["results"]
            assert restored["status"] == "confirmed" and restored["option"] == "wifi_raw"
            rejected, = (await ha.act(["sensor-a"], "flood"))["results"]
            assert rejected["status"] == "error" and rejected["option"] == "flood"
            writes = [call for call in fake.calls if call["type"] == "call_service"]
            assert [call["target"] for call in writes] == [{"entity_id": "select.renamed_source"}] * 5
            assert [call["service_data"]["option"] for call in writes] == [
                "external", "dns", "wifi_raw", "external", "wifi_raw"]
    asyncio.run(exercise())


@pytest.mark.parametrize("case", ["available", "unavailable", "missing", "device_disabled"])
def test_ha_panel_offers_sample_spacing_only_through_a_usable_select(case):
    devices, entities, states, areas = _ha_panel_inventory()
    spacing = next(state for state in states if state["entity_id"] == "select.renamed_spacing")
    if case == "unavailable":
        spacing["state"] = "unavailable"
    elif case == "missing":
        entities = [entity for entity in entities if entity["entity_id"] != "select.renamed_spacing"]
    elif case == "device_disabled":
        devices[0]["disabled_by"] = "user"
    row, = ha_client.build_inventory(devices, entities, states, areas)
    assert row["can_set_spacing"] == (case == "available")
    assert row["fields"]["spacing"]["value"] == ("half_slot" if case in {"available", "device_disabled"} else None)
    # The traffic source select stays the device's control, whatever happens to the spacing select.
    assert row["fields"]["source"]["entity_id"] == "select.renamed_source"


def test_ha_panel_sets_sample_spacing_through_its_select_and_confirms_it():
    async def exercise():
        from aiohttp import web
        from aiohttp.test_utils import TestServer
        fake = _PanelHomeAssistantServer()
        upstream = _upstream_inventory()
        for index in range(3):
            fake.inventory[index].extend(upstream[index])
        app = web.Application()
        app.router.add_get("/core/websocket", fake.websocket)
        async with TestServer(app) as server:
            ha = ha_client.HomeAssistant("test-supervisor-secret",
                                         endpoint=str(server.make_url("/core/websocket")).replace("http:", "ws:"),
                                         confirmation_timeout=0)
            changed, unsupported = (await ha.set_spacing(["sensor-a", "upstream-a"], "distinct"))["results"]
            assert changed["status"] == "confirmed" and changed["option"] == "distinct"
            assert unsupported["status"] == "error"
            assert (await ha.set_spacing(["sensor-a"], "distinct"))["results"][0]["status"] == "unchanged"
            writes = [call for call in fake.calls if call["type"] == "call_service"]
            assert [call["target"] for call in writes] == [{"entity_id": "select.renamed_spacing"}]
            assert writes[0]["service_data"] == {"option": "distinct"}
            source = next(state for state in fake.inventory[2] if state["entity_id"] == "select.renamed_source")
            assert source["state"] == "ping"
    asyncio.run(exercise())


def test_ha_panel_controls_the_upstream_component_without_refresh():
    async def exercise():
        from aiohttp import web
        from aiohttp.test_utils import TestServer
        fake = _PanelHomeAssistantServer()
        fake.inventory = _upstream_inventory(options=["ping", "dns", "dns_tcp", "external"])
        app = web.Application()
        app.router.add_get("/core/websocket", fake.websocket)
        async with TestServer(app) as server:
            ha = ha_client.HomeAssistant("test-supervisor-secret",
                                         endpoint=str(server.make_url("/core/websocket")).replace("http:", "ws:"),
                                         confirmation_timeout=0)
            assert (await ha.act(["upstream-a"], "external"))["results"][0]["status"] == "confirmed"
            assert (await ha.act(["upstream-a"], "internal"))["results"][0]["status"] == "confirmed"
            writes = [call for call in fake.calls if call["type"] == "call_service"]
            assert [call["target"] for call in writes] == [{"entity_id": "select.living_room_source"}] * 2
            assert [call["service_data"] for call in writes] == [{"option": "external"}, {"option": "ping"}]
    asyncio.run(exercise())


def test_ha_panel_http_routes_restrict_ingress_csrf_and_action_scope():
    async def exercise():
        from aiohttp.test_utils import TestClient, TestServer
        class HA:
            calls = []
            async def snapshot(self):
                return ha_client.build_inventory(*_ha_panel_inventory())
            async def act(self, ids, action):
                self.calls.append((ids, action))
                return {"results": []}
            async def set_spacing(self, ids, spacing):
                self.calls.append((ids, "spacing", spacing))
                return {"results": []}
        generator = SimpleNamespace(running=True, targets=["239.255.0.1"], protocol="udp", port=5555,
                                    rate_pps=100, sent_packets=10, send_errors=0,
                                    replies_received=0)
        ha = HA()
        async with TestClient(TestServer(panel.create_app(ha, generator))) as client:
            response = await client.get("/api/status", headers={"X-Forwarded-For": panel.INGRESS_ADDRESS})
            assert response.status == 403
        async with TestClient(TestServer(panel.create_app(ha, generator, allowed_peer="127.0.0.1"))) as client:
            assert (await client.get("/")).status == 200
            assert (await client.get("/static/panel.js")).status == 200
            assert (await client.get("/static/../config.yaml")).status == 404
            response = await client.get("/api/status")
            data = await response.json()
            assert "default-src 'self'" in response.headers["Content-Security-Policy"]
            headers = {"X-ESPectre-CSRF": data["csrf"]}
            payload = {"device_ids": ["sensor-a"], "mode": "external"}
            assert (await client.post("/api/action", json=payload)).status == 403
            for invalid in [[], {}, dict(payload, service="turn_on"), dict(payload, mode="select.option"),
                            dict(payload, mode="x" * 33), dict(payload, mode=""), dict(payload, mode=1),
                            {"device_ids": ["sensor-a"], "action": "external"},
                            dict(payload, device_ids=["sensor-a"] * 33), dict(payload, device_ids=[{}])]:
                assert (await client.post("/api/action", json=invalid, headers=headers)).status == 400
            assert not ha.calls
            generator.running = False
            assert (await client.post("/api/action", json=payload, headers=headers)).status == 409
            generator.running = True
            assert (await client.post("/api/action", json=payload, headers=headers)).status == 200
            assert ha.calls == [(["sensor-a"], "external")]
            spacing = {"device_ids": ["sensor-a"], "spacing": "distinct"}
            assert (await client.post("/api/spacing", json=spacing)).status == 403
            for invalid in [dict(spacing, spacing="tight"), dict(spacing, spacing=["distinct"]),
                            dict(spacing, mode="external"), dict(spacing, device_ids=[])]:
                assert (await client.post("/api/spacing", json=invalid, headers=headers)).status == 400
            # Spacing does not depend on the generator, unlike external traffic.
            generator.running = False
            assert (await client.post("/api/spacing", json=spacing, headers=headers)).status == 200
            assert ha.calls[1:] == [(["sensor-a"], "spacing", "distinct")]
            assert (await client.get("/api/devices")).status == 200
    asyncio.run(exercise())


def test_ha_panel_bulk_keeps_disabled_devices_unchanged():
    async def exercise():
        from aiohttp import web
        from aiohttp.test_utils import TestServer
        fake = _PanelHomeAssistantServer()
        device = dict(fake.inventory[0][0], id="sensor-b", manufacturer="ESPectre")
        fake.inventory[0].append(device)
        entity = dict(fake.inventory[1][0], device_id="sensor-b", entity_id="select.disabled_source", disabled_by="user")
        fake.inventory[1].append(entity)
        app = web.Application()
        app.router.add_get("/core/websocket", fake.websocket)
        async with TestServer(app) as server:
            ha = ha_client.HomeAssistant("test-supervisor-secret",
                                         endpoint=str(server.make_url("/core/websocket")).replace("http:", "ws:"))
            result = await ha.act(["sensor-a", "sensor-b"], "external")
            assert [item["status"] for item in result["results"]] == ["confirmed", "error"]
            writes = [call for call in fake.calls if call["type"] == "call_service"]
            assert len(writes) == 1 and writes[0]["target"] == {"entity_id": "select.renamed_source"}
            assert fake.inventory[1][-1]["disabled_by"] == "user"
    asyncio.run(exercise())


def test_ha_panel_generator_controls_preserve_panel_and_device_modes():
    async def exercise():
        from aiohttp.test_utils import TestClient, TestServer
        class HA:
            async def snapshot(self):
                return ha_client.build_inventory(*_ha_panel_inventory())
            async def act(self, *_args):
                raise AssertionError("Generator controls must not change device modes")
        generator = espectre_traffic_generator.ExternalTrafficGenerator(["127.0.0.1"])
        app = panel.create_app(HA(), generator, allowed_peer="127.0.0.1")
        try:
            async with TestClient(TestServer(app)) as client:
                data = await (await client.get("/api/status")).json()
                headers = {"X-ESPectre-CSRF": data["csrf"]}
                assert (await client.get("/logo.png")).content_type == "image/png"
                assert (await client.post("/api/generator", json={"action": "start"})).status == 403
                for payload in [[], {}, {"action": []}, {"action": "restart"}, {"action": "start", "targets": []}]:
                    assert (await client.post("/api/generator", json=payload, headers=headers)).status == 400
                async with app[panel.ACTION_LOCK]:
                    assert (await client.post("/api/generator", json={"action": "start"}, headers=headers)).status == 409
                for operation in ["start", "start", "stop", "stop", "start", "stop"]:
                    previous_sent = generator.sent_packets
                    response = await client.post("/api/generator", json={"action": operation}, headers=headers)
                    assert response.status == 200
                    assert (await response.json())["generator"]["running"] is (operation == "start")
                    assert not panel.generator_failed(app)
                    assert generator.sent_packets >= previous_sent
                    assert (await client.get("/")).status == 200
                    snapshot = await (await client.get("/api/devices")).json()
                    assert snapshot["devices"][0]["fields"]["ownership"]["value"] == "internal"
                    if operation == "start":
                        await asyncio.sleep(0.03)
                    else:
                        sent = generator.sent_packets
                        await asyncio.sleep(0.03)
                        assert generator.sent_packets == sent
                assert generator.sent_packets > 0
                app[panel.GENERATOR_STATE].expected_running = True
                assert panel.generator_failed(app)
        finally:
            generator.stop()
    asyncio.run(exercise())


def _standalone_settings(**overrides):
    return {**standalone.DEFAULT_SETTINGS, "multicast_ttl": 1, "rate_pps": 1, **overrides}


class _FakeDirectNetwork:
    """Direct HTTP devices keyed by endpoint; an unknown endpoint never answers."""

    def __init__(self):
        self.devices, self.calls = {}, []

    def add(self, ip, device_id, *, mode="ping", operations=("update_sensing",), catalog=None, reject=None,
            chip="esp32c6", features=None):
        self.devices[f"http://{ip}:62587/espectre/v1"] = {
            "info": {"device_id": device_id, "name": f"Sensor {device_id[-1]}", "chip": chip,
                     "frontend": "native", "firmware": "3.1.0"},
            "operations": operations, "reject": reject, "features": features,
            "catalog": tuple(standalone_devices.DIAGNOSTIC_FIELDS.values()) if catalog is None else catalog,
            "sensing": {"traffic_generator_mode": mode, "csi_target_pps": 100, "csi_traffic_udp_port": 5555,
                        "csi_traffic_multicast_group": "239.255.0.1", "csi_sample_spacing": "half_slot"}}

    def client(self, endpoint, timeout):
        network = self

        class Client:
            def negotiate(self):
                return network.request(endpoint, "get", "capabilities", None)

            def request(self, method, resource, data=None):
                return network.request(endpoint, method, resource, data)

            def close(self):
                pass
        return Client()

    def request(self, endpoint, method, resource, data):
        self.calls.append((endpoint, method, resource, data))
        device = self.devices.get(endpoint)
        if device is None:
            raise DirectProtocolError("Direct HTTP GET request failed: timed out") from TimeoutError("timed out")
        if resource == "capabilities":
            capabilities = {"protocol_version": "1.0", "resources": ["sensing"],
                            "operations": [{"name": name} for name in device["operations"]]}
            if device["features"] is not None:
                capabilities["features"] = device["features"]
            return capabilities
        if resource == "device":
            return dict(device["info"])
        if resource == "diagnostics":
            if not data["fields"]:
                return {"fields": [{"name": name} for name in device["catalog"]]}
            if any(name not in device["catalog"] for name in data["fields"]):
                raise DirectRequestError("invalid_params", "unknown diagnostics field")
            return {name: 12.5 for name in data["fields"]}
        if method == "patch":
            if device["reject"]:
                # C++ frontends answer invalid_params with HTTP 400, which urllib raises as HTTPError.
                body = json.dumps({"accepted": False, "code": "invalid_params", "message": device["reject"]})
                raise DirectProtocolError(f"Direct HTTP 400: {body}") from HTTPError(
                    endpoint, 400, "Bad Request", {}, None)
            device["sensing"].update(data)
            return {"accepted": True, "code": "ok", "message": ""}
        return dict(device["sensing"])


async def _standalone_client(controller, token="standalone-test-token", registry=None):
    from aiohttp.test_utils import TestClient, TestServer
    if registry is None:
        registry = standalone_devices.DeviceRegistry(controller.path.with_name("devices.json"),
                                                     client_factory=_FakeDirectNetwork().client,
                                                     discover=lambda: [])
    client = TestClient(TestServer(standalone.create_app(controller, token, registry)))
    await client.start_server()
    return client


def test_standalone_ui_requires_sign_in_csrf_and_throttles_failures(tmp_path):
    async def exercise():
        controller = standalone.Controller(_standalone_settings(), tmp_path / "settings.json")
        client = await _standalone_client(controller)
        try:
            for path in ["/", "/static/app.js", "/static/panel.css"]:
                assert (await client.get(path)).status == 200
            assert (await client.get("/logo.png")).content_type == "image/png"
            assert (await client.get("/static/../server.py")).status in {401, 404}
            assert (await client.get("/api/status")).status == 401
            assert (await client.post("/api/generator", json={"action": "start"})).status == 401
            for _attempt in range(standalone.LOGIN_FAILURES):
                assert (await client.post("/api/login", json={"token": "wrong-token-value"})).status == 401
            assert (await client.post("/api/login", json={"token": "standalone-test-token"})).status == 429
            assert (await client.get("/api/status")).status == 401
        finally:
            await client.close()

        client = await _standalone_client(controller)
        try:
            for invalid in [[], {}, {"token": 1}, {"token": "standalone-test-token", "user": "admin"}]:
                assert (await client.post("/api/login", json=invalid)).status == 400
            response = await client.post("/api/login", json={"token": "standalone-test-token"})
            assert response.status == 200
            cookie = response.cookies[standalone.SESSION_COOKIE]
            assert cookie["httponly"] and cookie["samesite"] == "Strict" and cookie["path"] == "/"
            response = await client.get("/api/status")
            data = await response.json()
            assert "default-src 'self'" in response.headers["Content-Security-Policy"]
            assert data["settings"] == controller.settings and data["generator"]["running"] is False
            assert (await client.post("/api/generator", json={"action": "start"})).status == 403
            headers = {"X-ESPectre-CSRF": data["csrf"]}
            assert (await client.post("/api/generator", data="start", headers=headers)).status == 415
            assert (await client.post("/api/logout", json={}, headers=headers)).status == 200
            assert (await client.get("/api/status")).status == 401
            assert not controller.generator.running
        finally:
            await client.close()
            controller.stop()
    asyncio.run(exercise())


def test_standalone_settings_validate_persist_and_keep_the_run_state(tmp_path):
    async def exercise():
        path = tmp_path / "settings.json"
        controller = standalone.Controller(_standalone_settings(), path)
        client = await _standalone_client(controller)
        try:
            await client.post("/api/login", json={"token": "standalone-test-token"})
            headers = {"X-ESPectre-CSRF": (await (await client.get("/api/status")).json())["csrf"]}
            await client.post("/api/generator", json={"action": "start"}, headers=headers)
            original = controller.generator
            valid = _standalone_settings(targets=["239.255.0.2", "192.168.1.20"], protocol="udp", rate_pps=2, dscp=0)
            missing = dict(valid)
            del missing["dscp"]
            for invalid in [[], missing, dict(valid, extra=1), dict(valid, targets=["2001:db8::1"]),
                            dict(valid, targets=["239.255.0.2", "239.255.0.2"]), dict(valid, targets=[]),
                            dict(valid, targets=[f"192.168.1.{index}" for index in range(1, standalone.MAX_TARGETS + 2)]),
                            dict(valid, rate_pps=0), dict(valid, protocol="tcp"), dict(valid, multicast_ttl="8"),
                            dict(valid, source_ip="239.255.0.9")]:
                assert (await client.post("/api/settings", json=invalid, headers=headers)).status == 400
            assert controller.generator is original and original.running and not path.exists()

            response = await client.post("/api/settings", json=valid, headers=headers)
            assert response.status == 200
            data = await response.json()
            assert data["settings"] == valid and data["generator"]["running"] is True
            assert data["generator"]["targets"] == valid["targets"] and data["generator"]["protocol"] == "udp"
            assert not original.running and controller.generator.running
            assert standalone.load_settings(path) == valid

            await client.post("/api/generator", json={"action": "stop"}, headers=headers)
            response = await client.post("/api/settings", json=dict(valid, rate_pps=3), headers=headers)
            assert (await response.json())["generator"]["running"] is False and not controller.generator.running
        finally:
            await client.close()
            controller.stop()
    asyncio.run(exercise())


@pytest.mark.filterwarnings("ignore::pytest.PytestUnhandledThreadExceptionWarning")
def test_standalone_reports_a_failed_generator_until_settings_fix_it(tmp_path):
    async def exercise():
        # 192.0.2.0/24 is reserved for documentation, so no host owns this source address.
        controller = standalone.Controller(_standalone_settings(protocol="udp", source_ip="192.0.2.55"),
                                           tmp_path / "settings.json")
        client = await _standalone_client(controller)
        try:
            await client.post("/api/login", json={"token": "standalone-test-token"})
            headers = {"X-ESPectre-CSRF": (await (await client.get("/api/status")).json())["csrf"]}
            await client.post("/api/generator", json={"action": "start"}, headers=headers)
            for _attempt in range(100):
                status = (await (await client.get("/api/status")).json())["generator"]
                if status["failed"]:
                    break
                await asyncio.sleep(0.01)
            assert status["failed"] and not status["running"] and status["error"]
            response = await client.post("/api/settings", json=dict(controller.settings, source_ip=""), headers=headers)
            status = (await response.json())["generator"]
            assert status["running"] and not status["failed"] and status["error"] is None
        finally:
            await client.close()
            controller.stop()
    asyncio.run(exercise())


def test_standalone_token_and_settings_files(tmp_path, monkeypatch):
    monkeypatch.delenv("ESPECTRE_UI_TOKEN", raising=False)
    token = standalone.load_token(tmp_path)
    assert len(token) >= standalone.MIN_TOKEN_LENGTH
    assert (tmp_path / "ui_token").stat().st_mode & 0o777 == 0o600
    assert standalone.load_token(tmp_path) == token
    assert standalone.load_token(tmp_path, "configured-token-value") == "configured-token-value"
    with pytest.raises(ValueError, match="token"):
        standalone.load_token(tmp_path, "short")

    path = tmp_path / "settings.json"
    assert standalone.load_settings(path) == standalone.DEFAULT_SETTINGS
    path.write_text(json.dumps({"targets": ["192.168.1.20"], "protocol": "udp"}), encoding="utf-8")
    assert standalone.load_settings(path) == dict(standalone.DEFAULT_SETTINGS, targets=["192.168.1.20"], protocol="udp")
    for invalid in [[], {"targets": ["127.0.0.1"]}]:
        path.write_text(json.dumps(invalid), encoding="utf-8")
        with pytest.raises(ValueError):
            standalone.load_settings(path)


def test_ha_panel_shared_event_feed_reads_once_and_tracks_registry_changes():
    async def exercise():
        from aiohttp import web
        from aiohttp.test_utils import TestServer
        fake = _PanelHomeAssistantServer()
        fake.emit_diagnostics = True
        app = web.Application()
        app.router.add_get("/core/websocket", fake.websocket)
        async with TestServer(app) as server:
            ha = ha_client.HomeAssistant("test-supervisor-secret",
                                         endpoint=str(server.make_url("/core/websocket")).replace("http:", "ws:"))
            feed = ha_client.HomeAssistantFeed(ha)
            async with feed.subscribe() as first, feed.subscribe() as second:
                async def next_rows(queue, predicate):
                    async with asyncio.timeout(4):
                        while True:
                            data = await queue.get()
                            if data["devices"] and predicate(data["devices"]):
                                return data
                await next_rows(first, lambda rows: rows[0]["fields"]["accepted"]["value"] >= 100.5)
                await next_rows(second, lambda rows: rows[0]["fields"]["accepted"]["value"] >= 101.5)
                assert fake.connections == 1
                assert sum(call["type"] == "get_states" for call in fake.calls) == 1
                assert sum(call["type"] == "config/entity_registry/list" for call in fake.calls) == 1
                assert all(call.get("domain") != "select" for call in fake.calls)
                fake.inventory[0][0]["name_by_user"] = "Renamed room"
                await fake.emit("device_registry_updated", {"action": "update", "device_id": "sensor-a"})
                await next_rows(first, lambda rows: rows[0]["name"] == "Renamed room")
                assert sum(call["type"] == "get_states" for call in fake.calls) == 2
                for websocket in list(fake.subscriptions):
                    await websocket.close()
                await next_rows(first, lambda rows: fake.connections == 2 and rows[0]["name"] == "Renamed room")
                assert fake.connections == 2
            count = len(fake.calls)
            await asyncio.sleep(1.1)
            assert len(fake.calls) == count
            assert feed.task is None
    asyncio.run(exercise())


def test_ha_panel_websocket_requires_csrf_and_streams_without_browser_polling():
    async def exercise():
        from aiohttp import web, WSMsgType
        from aiohttp.test_utils import TestClient, TestServer
        fake = _PanelHomeAssistantServer()
        fake.inventory[0][0]["connections"] = [["mac", "aa:bb:cc:dd:ee:ff"]]
        fake.emit_diagnostics = True
        backend = web.Application()
        backend.router.add_get("/core/websocket", fake.websocket)
        async with TestServer(backend) as server:
            ha = ha_client.HomeAssistant("test-supervisor-secret",
                                         endpoint=str(server.make_url("/core/websocket")).replace("http:", "ws:"))
            generator = SimpleNamespace(running=False, targets=["239.255.0.1"], protocol="udp", port=5555,
                                        rate_pps=100, sent_packets=7, send_errors=0,
                                        replies_received=0)
            async with TestClient(TestServer(panel.create_app(ha, generator, allowed_peer="127.0.0.1"))) as client:
                async with client.ws_connect("/api/stream") as socket:
                    await socket.send_json({"csrf": "wrong"})
                    assert (await socket.receive()).type == WSMsgType.CLOSE
                assert fake.connections == 0
                csrf = (await (await client.get("/api/status")).json())["csrf"]
                async with client.ws_connect("/api/stream") as socket:
                    await socket.send_json({"csrf": csrf})
                    async with asyncio.timeout(3):
                        while True:
                            data = await socket.receive_json()
                            assert data["generator"]["sent_packets"] == 7
                            assert "test-supervisor-secret" not in json.dumps(data)
                            if data.get("devices") and data["devices"][0]["fields"]["accepted"]["value"] >= 100.5:
                                assert data["devices"][0]["ip_address"] == "192.168.1.4"
                                assert data["devices"][0]["fields"]["rssi"]["value"] == -54
                                break
                await asyncio.sleep(0.1)
                assert not client.app[panel.STREAMS]
                assert not client.app[panel.FEED].listeners
                assert client.app[panel.FEED].task is None
                async with client.ws_connect("/api/stream") as socket:
                    await socket.send_json({"csrf": csrf})
                    await socket.receive_json()
                    async with asyncio.timeout(2):
                        await panel.close_feed(client.app)
                    assert client.app[panel.FEED].task is None
                    assert not client.app[panel.STREAMS]
    asyncio.run(exercise())


def test_ha_panel_ha_failure_does_not_stop_udp_generator():
    async def exercise():
        from aiohttp.test_utils import TestClient, TestServer
        with espectre_traffic_generator.ExternalTrafficGenerator(["127.0.0.1"], rate_pps=100) as generator:
            app = panel.create_app(ha_client.HomeAssistant(""), generator, allowed_peer="127.0.0.1")
            async with TestClient(TestServer(app)) as client:
                before = generator.sent_packets
                assert (await client.get("/api/devices")).status == 503
                await asyncio.sleep(0.1)
                assert generator.running and generator.sent_packets > before
                assert (await client.get("/api/status")).status == 200
    asyncio.run(exercise())


@pytest.mark.parametrize("report_interval", [0, 1])
def test_external_traffic_generator_bounds_error_logs(monkeypatch, capsys, report_interval):
    attempts = 0
    closed = False

    class FailingSocket:
        def setsockopt(self, *_args):
            pass

        def sendto(self, _payload, _destination):
            nonlocal attempts
            attempts += 1
            if attempts == 250:
                generator._stop_event.set()
            raise OSError("network unreachable")

        def close(self):
            nonlocal closed
            closed = True

    monkeypatch.setattr(espectre_traffic_generator.socket, "socket", lambda *_args: FailingSocket())
    monkeypatch.setattr(espectre_traffic_generator.time, "perf_counter", lambda: attempts / 100)
    generator = espectre_traffic_generator.ExternalTrafficGenerator(["239.255.0.1"])
    monkeypatch.setattr(generator._stop_event, "wait", lambda _timeout: False)

    generator.run_forever(report_interval=report_interval)

    assert attempts == generator.send_errors == 250
    assert generator.sent_packets == 0
    assert closed
    lines = capsys.readouterr().out.splitlines()
    if report_interval:
        assert len(lines) == 3  # One initial error, then one report per simulated second.
        assert "network unreachable" in lines[0]
    else:
        assert lines == []  # Library use must not write over CLI collection output.


def test_background_traffic_stop_never_signals_pid_from_state_file(tmp_path, monkeypatch):
    pid_file = tmp_path / "traffic.pid"
    control_file = tmp_path / "traffic.control"
    token = "owned-session"
    pid_file.write_text('{"pid": 4242, "token": "owned-session"}', encoding="utf-8")
    control_file.write_text(token, encoding="utf-8")
    signals = []

    def fake_kill(pid, signal_number):
        signals.append((pid, signal_number))
        raise ProcessLookupError

    monkeypatch.setattr(espectre_traffic_generator, "PID_FILE", pid_file)
    monkeypatch.setattr(espectre_traffic_generator, "CONTROL_FILE", control_file)
    monkeypatch.setattr(espectre_traffic_generator.os, "kill", fake_kill)

    espectre_traffic_generator.stop()

    assert signals
    assert all(signal_number == 0 for _pid, signal_number in signals)
    assert not pid_file.exists()
    assert not control_file.exists()


def test_background_traffic_start_cleans_control_after_launch_failure(tmp_path, monkeypatch):
    pid_file = tmp_path / "traffic.pid"
    control_file = tmp_path / "traffic.control"
    monkeypatch.setattr(espectre_traffic_generator, "PID_FILE", pid_file)
    monkeypatch.setattr(espectre_traffic_generator, "CONTROL_FILE", control_file)
    monkeypatch.setattr(
        espectre_traffic_generator.subprocess,
        "Popen",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("cannot launch")),
    )

    with pytest.raises(OSError, match="cannot launch"):
        espectre_traffic_generator.start()

    assert not pid_file.exists()
    assert not control_file.exists()


def test_parse_packet_reads_optional_metadata():
    receiver = CSIReceiver(bind_host='127.0.0.1')
    flags = RAW_CSI_FLAG_WIFI_RX_TS_VALID | RAW_CSI_FLAG_WIFI_RX_START_TS_NS_VALID
    packet = receiver._parse_packet(
        build_packet(
            seq_num=42,
            flags=flags,
            wifi_rx_ts_us=5555,
            wifi_rx_start_ts_ns=987654321,
        )
    )

    assert packet is not None
    assert packet.wifi_rx_ts_us == 5555
    assert packet.wifi_rx_start_ts_ns == 987654321


def test_parse_packet_reads_transport_backpressure_total():
    receiver = CSIReceiver(bind_host='127.0.0.1')
    packet = receiver._parse_packet(
        build_packet(
            seq_num=43,
            transport_backpressure_total=17,
        )
    )

    assert packet is not None
    assert packet.transport_backpressure_total == 17


def test_parse_packet_reads_transport_counters():
    receiver = CSIReceiver(bind_host='127.0.0.1')
    receiver._parse_packet(
        build_packet(
            seq_num=44,
            fresh_record_total=123,
            request_accepted_total=140,
        )
    )
def test_receiver_configures_udp_receive_buffer(monkeypatch):
    calls = []

    class FakeSocket:
        def setsockopt(self, level, optname, value):
            calls.append(("setsockopt", level, optname, value))

        def getsockopt(self, level, optname):
            calls.append(("getsockopt", level, optname))
            return 425984

        def bind(self, addr):
            calls.append(("bind", addr))

        def settimeout(self, value):
            calls.append(("settimeout", value))

        def recvfrom(self, _size):
            raise socket.timeout()

        def close(self):
            calls.append(("close",))

    fake_socket = FakeSocket()
    time_values = iter([1000.0, 1000.0, 1002.0])

    monkeypatch.setattr(socket, "socket", lambda *_args, **_kwargs: fake_socket)
    monkeypatch.setattr("tools.lib.csi_io.time.time", lambda: next(time_values))

    receiver = CSIReceiver(bind_host='127.0.0.1', socket_rcvbuf_bytes=262144)
    receiver.run(timeout=1.0, quiet=True)

    assert receiver.effective_socket_rcvbuf_bytes == 425984
    assert ("setsockopt", socket.SOL_SOCKET, socket.SO_RCVBUF, 262144) in calls
    assert ("getsockopt", socket.SOL_SOCKET, socket.SO_RCVBUF) in calls


def test_receiver_filters_out_of_order_packets_before_buffer_and_callbacks(monkeypatch):
    datagrams = [
        build_packet(seq_num=100, device_id=0x1),
        build_packet(seq_num=3, device_id=0x1),
        build_packet(seq_num=101, device_id=0x1),
    ]

    class FakeSocket:
        def setsockopt(self, *_args):
            pass

        def getsockopt(self, *_args):
            return 425984

        def bind(self, _addr):
            pass

        def settimeout(self, _value):
            pass

        def recvfrom(self, _size):
            return datagrams.pop(0), ("192.0.2.1", 5001)

        def close(self):
            pass

    monkeypatch.setattr(socket, "socket", lambda *_args, **_kwargs: FakeSocket())
    receiver = CSIReceiver(bind_host="127.0.0.1")
    observed = []

    def on_packet(packet):
        observed.append(packet.seq_num)
        if packet.seq_num == 101:
            receiver.stop()

    receiver.add_callback(on_packet)
    receiver.run(quiet=True)

    assert observed == [100, 101]
    assert [packet.seq_num for packet in receiver.buffer] == [100, 101]
    assert receiver.packet_count == 2
    assert receiver.out_of_order_count == 1


def test_parse_packets_accepts_multiple_records_in_one_datagram():
    receiver = CSIReceiver(bind_host='127.0.0.1')
    datagram = build_packet(seq_num=10, payload=[1, 2, 3, 4]) + build_packet(seq_num=11, payload=[5, 6, 7, 8])

    packets = receiver._parse_packets(datagram)

    assert len(packets) == 2
    assert packets[0].seq_num == 10
    assert packets[1].seq_num == 11
    np.testing.assert_array_equal(packets[0].iq_raw, np.array([1, 2, 3, 4], dtype=np.int8))
    np.testing.assert_array_equal(packets[1].iq_raw, np.array([5, 6, 7, 8], dtype=np.int8))


def test_parse_packet_rejects_multi_record_datagram():
    receiver = CSIReceiver(bind_host='127.0.0.1')
    datagram = build_packet(seq_num=1) + build_packet(seq_num=2)

    assert receiver._parse_packet(datagram) is None


def test_save_sample_keeps_existing_schema_and_adds_optional_metadata(tmp_path, monkeypatch):
    data_dir = tmp_path / 'data'
    monkeypatch.setattr(dataset_metadata, 'DATA_DIR', data_dir)
    monkeypatch.setattr(dataset_metadata, 'DATASET_INFO_FILE', data_dir / 'dataset_info.json')

    receiver = CSIReceiver(bind_host='127.0.0.1')
    flags = RAW_CSI_FLAG_WIFI_RX_TS_VALID
    packets = [
        receiver._parse_packet(
            build_packet(
                seq_num=100,
                flags=flags,
                payload=[1, 2, 3, 4],
                device_id=0xABCDEF,
                device_ticks_us=1000,
                wifi_rx_ts_us=4000,
                channel=1,
                rssi_dbm=-50,
            )
        ),
        receiver._parse_packet(
            build_packet(
                seq_num=101,
                flags=flags,
                payload=[5, 6, 7, 8],
                device_id=0xABCDEF,
                device_ticks_us=2000,
                wifi_rx_ts_us=5000,
                channel=1,
                rssi_dbm=-49,
            )
        ),
    ]
    for index, packet in enumerate(packets, start=1000):
        packet.transport = 'http'
        packet.transport_target = 'http://192.168.1.23/espectre/v1/csi'
        packet.requested_pps = 200.0
        packet.raw_protocol_version = RAW_CSI_PROTOCOL_VERSION
        packet.record_version = RAW_CSI_RECORD_VERSION_V8
        packet.raw_stream_sequence = index
        packet.frontend = 'esphome'
        packet.firmware_version = '2.0.0-test'
        packet.firmware_identity = 'espectre-esphome-test'
    packets[-1].fresh_record_total = 102
    packets[-1].raw_drop_total = 3
    packets[-1].raw_send_backpressure_total = 4
    packets[-1].raw_final_stream_sequence = 1004

    collector = CSICollector(
        label='static_presence',
        contributor='tester',
        bind_host='127.0.0.1',
        target_pps=100,
    )
    filepath = collector.save_sample(packets)

    assert filepath is not None
    assert filepath.exists()

    data = np.load(filepath, allow_pickle=False)
    assert str(data['label']) == 'static_presence'
    assert str(data['chip']) == 'c6'
    assert int(data['num_subcarriers']) == 2
    assert str(data['format_version']) == '1.2'
    assert int(data['device_id']) == 0xABCDEF
    assert int(data['csi_target_pps']) == 100
    np.testing.assert_array_equal(data['stream_seq_num'], np.array([100, 101], dtype=np.uint32))
    np.testing.assert_array_equal(data['raw_stream_sequence'], np.array([1000, 1001], dtype=np.uint64))
    assert str(data['transport']) == 'http'
    assert str(data['transport_target']) == 'http://192.168.1.23/espectre/v1/csi'
    assert str(data['endpoint']) == 'http://192.168.1.23/espectre/v1/csi'
    assert float(data['observed_pps']) == float(data['effective_pps'])
    assert float(data['requested_pps']) == 200.0
    assert int(data['raw_protocol_version']) == RAW_CSI_PROTOCOL_VERSION
    assert int(data['record_version']) == 8
    assert str(data['frontend']) == 'esphome'
    assert str(data['firmware_version']) == '2.0.0-test'
    assert str(data['firmware_identity']) == 'espectre-esphome-test'
    assert int(data['raw_fresh_record_total']) == 102
    assert int(data['fresh_record_total']) == 102
    assert int(data['raw_drop_total']) == 3
    assert int(data['raw_send_backpressure_total']) == 4
    assert int(data['send_backpressure_total']) == 4
    assert int(data['raw_final_stream_sequence']) == 1004
    np.testing.assert_array_equal(data['phy_mode'], np.array(['ht', 'ht']))
    np.testing.assert_array_equal(data['ltf_type'], np.array(['ht-ltf', 'ht-ltf']))
    np.testing.assert_array_equal(data['channel_width'], np.array(['20', '20']))
    np.testing.assert_array_equal(data['device_ticks_us'], np.array([1000, 2000], dtype=np.uint64))
    np.testing.assert_array_equal(data['wifi_rx_ts_us'], np.array([4000, 5000], dtype=np.uint32))
    np.testing.assert_array_equal(data['csi_data'], np.array([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=np.int8))

    info = dataset_metadata.load_dataset_info()
    assert info['format_version'] == '1.2'
    assert info['files']['static_presence'][0]['filename'] == filepath.name
    assert info['files']['static_presence'][0]['device_id'] == '0000000000abcdef'
    assert info['files']['static_presence'][0]['description'] == 'HT20 static presence sample'
    assert info['files']['static_presence'][0]['nominal_packet_rate'] == 100
    assert info['files']['static_presence'][0]['transport'] == 'http'
    assert info['files']['static_presence'][0]['endpoint'] == 'http://192.168.1.23/espectre/v1/csi'
    assert info['files']['static_presence'][0]['observed_pps'] == info['files']['static_presence'][0]['effective_pps']
    assert info['files']['static_presence'][0]['requested_pps'] == 200.0
    assert info['files']['static_presence'][0]['frontend'] == 'esphome'
    assert info['files']['static_presence'][0]['raw_fresh_record_total'] == 102
    assert info['files']['static_presence'][0]['fresh_record_total'] == 102
    assert info['files']['static_presence'][0]['raw_drop_total'] == 3
    assert info['files']['static_presence'][0]['raw_send_backpressure_total'] == 4
    assert info['files']['static_presence'][0]['send_backpressure_total'] == 4
    assert info['files']['static_presence'][0]['raw_final_stream_sequence'] == 1004
    assert 'dev0000000000abcdef' in filepath.name


def test_save_sample_preserves_short_transport_schema(tmp_path, monkeypatch):
    data_dir = tmp_path / 'data'
    monkeypatch.setattr(dataset_metadata, 'DATA_DIR', data_dir)
    monkeypatch.setattr(dataset_metadata, 'DATASET_INFO_FILE', data_dir / 'dataset_info.json')

    short_payload = []
    for idx in range(12):
        short_payload.extend([idx + 1, 50 + idx])

    receiver = CSIReceiver(bind_host='127.0.0.1')
    packet = receiver._parse_packet(build_packet(seq_num=200, payload=short_payload, device_id=0xABCDEF))

    collector = CSICollector(label='short_transport', contributor='tester', bind_host='127.0.0.1')
    filepath = collector.save_sample([packet])

    assert filepath is not None
    data = np.load(filepath, allow_pickle=False)
    assert int(data['num_subcarriers']) == 12
    assert data['csi_data'].shape == (1, 24)
    np.testing.assert_array_equal(data['csi_data'][0], np.array(short_payload, dtype=np.int8))
    assert '_12sc_' in filepath.name


def test_load_historical_dataset_labels_missing_phy_as_ht20(tmp_path):
    filepath = tmp_path / 'historical_ht20.npz'
    np.savez_compressed(
        filepath,
        csi_data=np.zeros((1, 128), dtype=np.int8),
        num_subcarriers=64,
        label='motion',
        chip='esp32',
    )

    packet = load_npz_as_packets(filepath)[0]

    assert packet['phy_format'] == 'ht20'
    assert packet['phy_mode'] == 'ht'
    assert packet['ltf_type'] == 'ht-ltf'
    assert packet['channel_width'] == '20'
    assert packet['layout_id'] == 'ht20_64'
    assert packet['format_metadata_source'] == 'historical_missing_phy'


def test_load_npz_as_packets_rejects_object_arrays(tmp_path):
    filepath = tmp_path / 'malicious_object_array.npz'
    np.savez_compressed(
        filepath,
        csi_data=np.zeros((1, 128), dtype=np.int8),
        num_subcarriers=64,
        label='motion',
        chip=np.array('esp32', dtype=object),
    )

    with pytest.raises(ValueError, match="Unsafe NPZ dataset"):
        load_npz_as_packets(filepath)


def test_load_dataset_preserves_explicit_ht20_phy_metadata(tmp_path):
    filepath = tmp_path / 'explicit_phy.npz'
    np.savez_compressed(
        filepath,
        csi_data=np.zeros((1, 128), dtype=np.int8),
        num_subcarriers=64,
        label='motion',
        chip='esp32',
        phy_mode=np.array(['ht']),
        ltf_type=np.array(['ht-ltf']),
        channel_width=np.array(['20']),
    )

    packet = load_npz_as_packets(filepath, keep_all_phy=True)[0]

    assert packet['phy_format'] == 'ht20'
    assert packet['phy_mode'] == 'ht'
    assert packet['ltf_type'] == 'ht-ltf'
    assert packet['channel_width'] == '20'
    assert packet['layout_id'] == 'ht20_64'


def test_load_npz_filters_non_ht20_packets_by_default(tmp_path):
    filepath = tmp_path / 'mixed_phy.npz'
    rows = np.zeros((3, 128), dtype=np.int8)
    rows[0, :4] = np.array([1, 2, 3, 4], dtype=np.int8)
    rows[1, :4] = np.array([5, 6, 7, 8], dtype=np.int8)
    rows[2, :4] = np.array([9, 10, 11, 12], dtype=np.int8)
    np.savez_compressed(
        filepath,
        csi_data=rows,
        num_subcarriers=64,
        label='motion',
        chip='esp32',
        phy_mode=np.array(['ht', 'legacy', 'ht']),
        ltf_type=np.array(['ht-ltf', 'unknown', 'ht-ltf']),
        channel_width=np.array(['20', '20', '20']),
        stream_seq_num=np.array([10, 11, 12], dtype=np.uint32),
    )

    packets = load_npz_as_packets(filepath)
    assert len(packets) == 2
    assert [packet['phy_mode'] for packet in packets] == ['ht', 'ht']
    assert [packet['stream_seq_num'] for packet in packets] == [10, 12]
    np.testing.assert_array_equal(packets[0]['csi_data'][:4], np.array([1, 2, 3, 4], dtype=np.int8))
    np.testing.assert_array_equal(packets[1]['csi_data'][:4], np.array([9, 10, 11, 12], dtype=np.int8))

    all_packets = load_npz_as_packets(filepath, keep_all_phy=True)
    assert len(all_packets) == 3
    assert [packet['phy_mode'] for packet in all_packets] == ['ht', 'legacy', 'ht']


def test_load_npz_filters_non_ht_ltf_packets_by_default(tmp_path):
    filepath = tmp_path / 'mixed_ltf.npz'
    np.savez_compressed(
        filepath,
        csi_data=np.zeros((2, 128), dtype=np.int8),
        num_subcarriers=64,
        label='motion',
        chip='esp32',
        phy_mode=np.array(['ht', 'ht']),
        ltf_type=np.array(['ht-ltf', 'vht-ltf']),
        channel_width=np.array(['20', '20']),
    )

    packets = load_npz_as_packets(filepath)
    assert len(packets) == 1
    assert packets[0]['ltf_type'] == 'ht-ltf'


def test_load_npz_filters_non_64sc_historical_rows_by_default(tmp_path):
    filepath = tmp_path / 'historical_short.npz'
    np.savez_compressed(
        filepath,
        csi_data=np.array([[1, 2, 3, 4]], dtype=np.int8),
        num_subcarriers=2,
        label='motion',
        chip='esp32',
    )

    packets = load_npz_as_packets(filepath)
    assert packets == []


def test_load_npz_csi_data_filters_non_ht20_rows(tmp_path):
    from tools.lib.csi_io import load_npz_csi_data

    filepath = tmp_path / 'mixed_phy_matrix.npz'
    rows = np.zeros((2, 128), dtype=np.int8)
    rows[0, :4] = np.array([1, 2, 3, 4], dtype=np.int8)
    rows[1, :4] = np.array([5, 6, 7, 8], dtype=np.int8)
    np.savez_compressed(
        filepath,
        csi_data=rows,
        num_subcarriers=64,
        phy_mode=np.array(['legacy', 'ht']),
        ltf_type=np.array(['unknown', 'ht-ltf']),
        channel_width=np.array(['20', '20']),
    )

    filtered = load_npz_csi_data(filepath)
    assert filtered.shape == (1, 128)
    np.testing.assert_array_equal(filtered[0][:4], np.array([5, 6, 7, 8], dtype=np.int8))

    raw = load_npz_csi_data(filepath, keep_all_phy=True)
    assert raw.shape == (2, 128)


def test_load_npz_csi_data_drops_legacy20_rows_even_for_original_esp32(tmp_path):
    from tools.lib.csi_io import load_npz_csi_data

    filepath = tmp_path / 'esp32_legacy_only_matrix.npz'
    np.savez_compressed(
        filepath,
        csi_data=np.array(
            [
                [1, 2, 3, 4],
                [5, 6, 7, 8],
            ],
            dtype=np.int8,
        ),
        chip=np.array('esp32'),
        phy_mode=np.array(['legacy', 'legacy']),
        channel_width=np.array(['20', '20']),
    )

    filtered = load_npz_csi_data(filepath)
    assert filtered.shape == (0, 4)


def test_load_npz_csi_data_rejects_object_arrays(tmp_path):
    from tools.lib.csi_io import load_npz_csi_data

    filepath = tmp_path / 'malicious_csi_object_array.npz'
    np.savez_compressed(
        filepath,
        csi_data=np.array([object()], dtype=object),
        num_subcarriers=64,
        label='motion',
        chip='esp32',
    )

    with pytest.raises(ValueError, match="Unsafe NPZ dataset"):
        load_npz_csi_data(filepath)


def test_save_samples_by_device_splits_capture_window(tmp_path, monkeypatch):
    data_dir = tmp_path / 'data'
    monkeypatch.setattr(dataset_metadata, 'DATA_DIR', data_dir)
    monkeypatch.setattr(dataset_metadata, 'DATASET_INFO_FILE', data_dir / 'dataset_info.json')

    receiver = CSIReceiver(bind_host='127.0.0.1')
    collector = CSICollector(label='motion', contributor='tester', bind_host='127.0.0.1')
    packets = [
        receiver._parse_packet(build_packet(seq_num=1, payload=[1, 2, 3, 4], device_id=0x10)),
        receiver._parse_packet(build_packet(seq_num=2, payload=[5, 6, 7, 8], device_id=0x20)),
        receiver._parse_packet(build_packet(seq_num=3, payload=[9, 10, 11, 12], device_id=0x10)),
    ]

    saved_paths = collector.save_samples_by_device(packets)

    assert len(saved_paths) == 2
    saved_names = {path.name for path in saved_paths}
    assert any('dev0000000000000010' in name for name in saved_names)
    assert any('dev0000000000000020' in name for name in saved_names)

    info = dataset_metadata.load_dataset_info()
    assert len(info['files']['motion']) == 2
    assert {entry['device_id'] for entry in info['files']['motion']} == {
        '0000000000000010',
        '0000000000000020',
    }


@pytest.mark.parametrize('label', ('../motion', '/tmp/motion', 'room/left', '.', 'motion label'))
def test_collector_rejects_labels_that_are_not_safe_directory_names(label):
    with pytest.raises(ValueError, match='dataset label'):
        CSICollector(label=label, bind_host='127.0.0.1')


def test_save_sample_removes_archive_when_catalog_update_fails(tmp_path, monkeypatch):
    data_dir = tmp_path / 'data'
    monkeypatch.setattr(dataset_metadata, 'DATA_DIR', data_dir)
    monkeypatch.setattr(dataset_metadata, 'DATASET_INFO_FILE', data_dir / 'dataset_info.json')
    receiver = CSIReceiver(bind_host='127.0.0.1')
    collector = CSICollector(label='motion', contributor='tester', bind_host='127.0.0.1')
    packet = receiver._parse_packet(build_packet(seq_num=1, payload=[1, 2, 3, 4], device_id=0x10))
    monkeypatch.setattr(
        collector,
        '_update_dataset_info',
        lambda **_kwargs: (_ for _ in ()).throw(OSError('catalog unavailable')),
    )

    with pytest.raises(OSError, match='catalog unavailable'):
        collector.save_sample([packet])

    assert not list(data_dir.rglob('*.npz'))


def test_multi_device_save_rolls_back_archives_and_catalog(tmp_path, monkeypatch):
    data_dir = tmp_path / 'data'
    monkeypatch.setattr(dataset_metadata, 'DATA_DIR', data_dir)
    monkeypatch.setattr(dataset_metadata, 'DATASET_INFO_FILE', data_dir / 'dataset_info.json')
    receiver = CSIReceiver(bind_host='127.0.0.1')
    collector = CSICollector(label='motion', contributor='tester', bind_host='127.0.0.1')
    packets = [
        receiver._parse_packet(build_packet(seq_num=1, payload=[1, 2, 3, 4], device_id=0x10)),
        receiver._parse_packet(build_packet(seq_num=2, payload=[5, 6, 7, 8], device_id=0x20)),
    ]
    original_info = dataset_metadata.load_dataset_info()
    dataset_metadata.save_dataset_info(original_info)
    real_save_sample = collector.save_sample
    save_calls = 0

    def fail_second_device(device_packets):
        nonlocal save_calls
        save_calls += 1
        if save_calls == 2:
            raise OSError('second device failed')
        return real_save_sample(device_packets)

    monkeypatch.setattr(collector, 'save_sample', fail_second_device)

    with pytest.raises(OSError, match='second device failed'):
        collector.save_samples_by_device(packets)

    assert not list(data_dir.rglob('*.npz'))
    assert dataset_metadata.load_dataset_info() == original_info


def test_save_samples_by_device_rejects_missing_device_id(tmp_path, monkeypatch):
    data_dir = tmp_path / 'data'
    monkeypatch.setattr(dataset_metadata, 'DATA_DIR', data_dir)
    monkeypatch.setattr(dataset_metadata, 'DATASET_INFO_FILE', data_dir / 'dataset_info.json')

    receiver = CSIReceiver(bind_host='127.0.0.1')
    collector = CSICollector(label='motion', contributor='tester', bind_host='127.0.0.1')
    packet = receiver._parse_packet(build_packet(seq_num=7, payload=[1, 2, 3, 4], device_id=0))

    with pytest.raises(ValueError, match='missing device_id'):
        collector.save_samples_by_device([packet])


def test_save_sample_rejects_mixed_device_packets(tmp_path, monkeypatch):
    data_dir = tmp_path / 'data'
    monkeypatch.setattr(dataset_metadata, 'DATA_DIR', data_dir)
    monkeypatch.setattr(dataset_metadata, 'DATASET_INFO_FILE', data_dir / 'dataset_info.json')

    receiver = CSIReceiver(bind_host='127.0.0.1')
    collector = CSICollector(label='motion', contributor='tester', bind_host='127.0.0.1')
    packets = [
        receiver._parse_packet(build_packet(seq_num=1, payload=[1, 2, 3, 4], device_id=0x1)),
        receiver._parse_packet(build_packet(seq_num=2, payload=[5, 6, 7, 8], device_id=0x2)),
    ]

    with pytest.raises(ValueError, match='mixed-device sample'):
        collector.save_sample(packets)


def test_check_sequence_by_device_handles_interleaved_streams():
    receiver = CSIReceiver(bind_host='127.0.0.1')
    packets = [
        receiver._parse_packet(build_packet(seq_num=1, device_id=0x1)),
        receiver._parse_packet(build_packet(seq_num=1, device_id=0x2)),
        receiver._parse_packet(build_packet(seq_num=2, device_id=0x1)),
        receiver._parse_packet(build_packet(seq_num=2, device_id=0x2)),
        receiver._parse_packet(build_packet(seq_num=4, device_id=0x1)),
    ]

    accepted = [receiver.accept_packet(packet) for packet in packets]

    assert accepted == [True, True, True, True, True]
    assert receiver.dropped_count == 1
    assert receiver.out_of_order_count == 0


def test_receiver_rejects_backward_and_duplicate_sequences_without_poisoning_state():
    receiver = CSIReceiver(bind_host='127.0.0.1')
    packets = [
        receiver._parse_packet(build_packet(seq_num=100, device_id=0x1)),
        receiver._parse_packet(build_packet(seq_num=101, device_id=0x1)),
        receiver._parse_packet(build_packet(seq_num=3, device_id=0x1)),
        receiver._parse_packet(build_packet(seq_num=101, device_id=0x1)),
        receiver._parse_packet(build_packet(seq_num=102, device_id=0x1)),
    ]

    accepted = [receiver.accept_packet(packet) for packet in packets]

    assert accepted == [True, True, False, False, True]
    assert receiver.dropped_count == 0
    assert receiver.out_of_order_count == 2


def test_receiver_sequence_filter_accepts_uint32_wrap_and_reset():
    receiver = CSIReceiver(bind_host='127.0.0.1')
    wrapped = [
        receiver._parse_packet(build_packet(seq_num=0xFFFFFFFE, device_id=0x1)),
        receiver._parse_packet(build_packet(seq_num=0xFFFFFFFF, device_id=0x1)),
        receiver._parse_packet(build_packet(seq_num=0, device_id=0x1)),
        receiver._parse_packet(build_packet(seq_num=1, device_id=0x1)),
    ]

    assert all(receiver.accept_packet(packet) for packet in wrapped)
    assert receiver.dropped_count == 0
    assert receiver.out_of_order_count == 0

    receiver.reset_stats()
    restarted = receiver._parse_packet(build_packet(seq_num=0, device_id=0x1))
    assert receiver.accept_packet(restarted)


def test_summarize_ready_devices_waits_for_all_expected_devices():
    now = 100.0
    warmup_target = 10
    threshold = 1.0
    ready_stable_seconds = 3.0

    summary = CSICollector._summarize_ready_devices(
        {
            0x1: {'processed_packets': warmup_target, 'stable_since': now - 4.0, 'current_metric': 0.2},
        },
        expected_device_count=2,
        warmup_target=warmup_target,
        threshold=threshold,
        now=now,
        ready_stable_seconds=ready_stable_seconds,
    )
    assert summary['ready'] is False
    assert summary['status'] == 'DEVICES 1/2'

    summary = CSICollector._summarize_ready_devices(
        {
            0x1: {'processed_packets': warmup_target, 'stable_since': now - 4.0, 'current_metric': 0.2},
            0x2: {'processed_packets': warmup_target, 'stable_since': now - 1.0, 'current_metric': 0.3},
        },
        expected_device_count=2,
        warmup_target=warmup_target,
        threshold=threshold,
        now=now,
        ready_stable_seconds=ready_stable_seconds,
    )
    assert summary['ready'] is False
    assert summary['status'] == 'STABLE 2/2'
    assert summary['stable_elapsed'] == pytest.approx(1.0)

    summary = CSICollector._summarize_ready_devices(
        {
            0x1: {'processed_packets': warmup_target, 'stable_since': now - 4.0, 'current_metric': 0.2},
            0x2: {'processed_packets': warmup_target, 'stable_since': now - 3.5, 'current_metric': 0.3},
        },
        expected_device_count=2,
        warmup_target=warmup_target,
        threshold=threshold,
        now=now,
        ready_stable_seconds=ready_stable_seconds,
    )
    assert summary['ready'] is True
    assert summary['status'] == 'READY 2/2'


def test_format_ready_device_lines_includes_waiting_ip_and_device_details():
    now = 100.0
    ready_stable_seconds = 3.0
    lines = CSICollector._format_ready_device_lines(
        {
            0x1: {
                'processed_packets': 12,
                'stable_since': now - 3.5,
                'current_metric': 0.2,
                'current_pps': 121,
                'source_ip': '192.168.1.17',
                'chip': 'c6',
                'channel': 6,
                'rssi_dbm': -48,
            },
            0x2: {
                'processed_packets': 8,
                'stable_since': None,
                'current_metric': 0.0,
                'current_pps': 87,
                'source_ip': '192.168.1.24',
                'chip': 'c3',
                'channel': 11,
                'rssi_dbm': -63,
            },
        },
        expected_source_hosts=['192.168.1.17', '192.168.1.24', '192.168.1.29'],
        warmup_target=10,
        threshold=1.0,
        now=now,
        ready_stable_seconds=ready_stable_seconds,
    )

    assert any(
        'ip=192.168.1.29' in line
        and 'chip=?' in line
        and 'ch=--' in line
        and 'rssi=---' in line
        and 'pps=--' in line
        and '[------------------]' in line
        and 'WAITING' in line
        and 'stable' not in line
        for line in lines
    )
    assert any(
        'ip=192.168.1.17' in line
        and 'chip=C6' in line
        and 'ch=06' in line
        and 'rssi=-48' in line
        and 'pps=121' in line
        and '[####--------------]' in line
        and 'READY' in line
        and 'stable' not in line
        for line in lines
    )
    assert any(
        'ip=192.168.1.24' in line
        and 'chip=C3' in line
        and 'ch=11' in line
        and 'rssi=-63' in line
        and 'pps=87' in line
        and '[------------------]' in line
        and 'WARMUP 8/10' in line
        and 'stable' not in line
        for line in lines
    )


def test_summarize_ready_devices_reports_unstable_device_count():
    now = 100.0
    warmup_target = 10
    threshold = 1.0
    ready_stable_seconds = 3.0

    summary = CSICollector._summarize_ready_devices(
        {
            0x1: {'processed_packets': warmup_target, 'stable_since': now - 4.0, 'current_metric': 0.2},
            0x2: {'processed_packets': warmup_target, 'stable_since': None, 'current_metric': 2.5},
        },
        expected_device_count=2,
        warmup_target=warmup_target,
        threshold=threshold,
        now=now,
        ready_stable_seconds=ready_stable_seconds,
    )

    assert summary == {
        'ready': False,
        'status': 'UNSTABLE 1/2',
        'stable_elapsed': 0.0,
        'ready_count': 1,
        'observed_count': 2,
        'required_count': 2,
    }


def test_receiver_drop_rate_uses_expected_packet_total():
    receiver = CSIReceiver(bind_host='127.0.0.1')
    receiver.packet_count = 400
    receiver.dropped_count = 4

    stats = receiver.get_stats()

    assert stats['drop_rate'] == pytest.approx(4 / 404 * 100)


class _TTYBuffer(io.StringIO):
    def isatty(self):
        return True


def test_emit_ready_status_block_uses_ansi_when_inline_enabled():
    stream = _TTYBuffer()

    rendered = CSICollector._emit_ready_status_block(
        'summary 1',
        ['detail 1', 'detail 2'],
        stream=stream,
        inline=True,
    )
    assert rendered == 3
    assert stream.getvalue() == '\x1b[2Ksummary 1\n\x1b[2Kdetail 1\n\x1b[2Kdetail 2\n'

    stream.seek(0)
    stream.truncate(0)

    rendered = CSICollector._emit_ready_status_block(
        'summary 2',
        ['detail 3'],
        previous_line_count=3,
        stream=stream,
        inline=True,
    )
    assert rendered == 2
    assert stream.getvalue() == '\x1b[3F\x1b[2Ksummary 2\n\x1b[2Kdetail 3\n\x1b[2K\n'


def test_emit_ready_status_block_falls_back_to_plain_lines():
    stream = io.StringIO()

    rendered = CSICollector._emit_ready_status_block(
        'summary',
        ['detail'],
        previous_line_count=5,
        stream=stream,
        inline=False,
    )

    assert rendered == 2
    assert stream.getvalue() == 'summary\ndetail\n'


def _standalone_registry(tmp_path, network, found=()):
    records = [SimpleNamespace(ip_address=ip, port=62587) for ip in found]
    return standalone_devices.DeviceRegistry(tmp_path / "devices.json", client_factory=network.client,
                                             discover=lambda: list(records))


def test_standalone_devices_are_found_by_mdns_or_address_and_read_only_supported_fields(tmp_path):
    network = _FakeDirectNetwork()
    network.add("192.168.1.20", "aaaaaaaaaaaaaaa1")
    network.add("192.168.1.21", "aaaaaaaaaaaaaaa2", catalog=("generator_pps", "traffic_tx_pps"))
    network.add("192.168.1.22", "aaaaaaaaaaaaaaa3")
    registry = _standalone_registry(tmp_path, network, found=["192.168.1.20", "192.168.1.21", "192.168.1.99"])

    # `wifi_raw` follows the raw-destination capability; 3.0 firmware without it falls back to the chip.
    capabilities = {"features": {"traffic_generator_target_mac": True}}
    assert "wifi_raw" in standalone_devices.supported_modes(capabilities, "esp32c6")
    assert "wifi_raw" not in standalone_devices.supported_modes({"features": {"traffic_generator_target_mac": False}},
                                                                "esp32s3")
    assert "wifi_raw" in standalone_devices.supported_modes({}, "esp32s3")
    assert standalone_devices.supported_modes({}, "esp32c6") == ("ping", "dns", "dns_tcp", "external")
    # The HTTP server rejects some requests in plain text before they reach the application.
    rejected = DirectProtocolError("Direct HTTP 403: origin not allowed")
    rejected.__cause__ = HTTPError("http://192.168.1.20:62587/espectre/v1/sensing", 403, "Forbidden", {}, None)
    assert standalone_devices.describe_error(rejected) == "origin not allowed"
    rejected = DirectProtocolError("Direct HTTP 404: ")
    rejected.__cause__ = HTTPError("http://192.168.1.20:62587/espectre/v1/sensing", 404, "Not Found", {}, None)
    assert standalone_devices.describe_error(rejected) == "The device rejected the request (HTTP 404)."

    assert registry.scan() == 2
    devices = {device["id"]: device for device in registry.snapshot()["devices"]}
    assert set(devices) == {"aaaaaaaaaaaaaaa1", "aaaaaaaaaaaaaaa2"}
    first, second = devices["aaaaaaaaaaaaaaa1"], devices["aaaaaaaaaaaaaaa2"]
    assert first["source"] == "mdns" and first["can_control"] and first["error"] is None
    assert first["sensing"]["traffic_generator_mode"] == "ping"
    assert first["modes"] == ["ping", "dns", "dns_tcp", "external"]
    assert all(value == 12.5 for value in first["fields"].values())
    # Asking for a field outside the catalog would fail the whole diagnostics request.
    assert second["fields"]["generator"] == 12.5 and second["fields"]["rssi"] is None
    assert second["error"] is None

    for invalid in ["sensor.local", "127.0.0.1", "239.255.0.1", "192.168.1.22:0", "2001:db8::1"]:
        with pytest.raises(ValueError):
            registry.add(invalid)
    with pytest.raises(ValueError, match="No ESPectre device"):
        registry.add("192.168.1.98")
    assert not (tmp_path / "devices.json").exists()
    registry.add("192.168.1.22")
    registry.add("192.168.1.20")
    assert json.loads((tmp_path / "devices.json").read_text())["manual"] == ["192.168.1.22", "192.168.1.20"]

    # A device added by address survives a search that misses it, and keeps its source when mDNS finds it.
    reloaded = _standalone_registry(tmp_path, network, found=["192.168.1.20"])
    assert reloaded.scan() == 3
    sources = {device["id"]: device["source"] for device in reloaded.snapshot()["devices"]}
    assert sources == {"aaaaaaaaaaaaaaa1": "manual", "aaaaaaaaaaaaaaa3": "manual"}
    with pytest.raises(KeyError):
        reloaded.remove("unknown")
    reloaded.remove("aaaaaaaaaaaaaaa3")
    assert json.loads((tmp_path / "devices.json").read_text())["manual"] == ["192.168.1.20"]

    del network.devices["http://192.168.1.20:62587/espectre/v1"]
    reloaded.refresh_all()
    lost, = reloaded.snapshot()["devices"]
    assert lost["error"] == "The device did not answer." and not lost["can_control"]


def test_standalone_device_modes_change_each_device_on_its_own(tmp_path):
    async def exercise():
        network = _FakeDirectNetwork()
        network.add("192.168.1.20", "aaaaaaaaaaaaaaa1")
        network.add("192.168.1.21", "aaaaaaaaaaaaaaa2", reject="traffic_generator_mode is invalid")
        network.add("192.168.1.22", "aaaaaaaaaaaaaaa3", operations=())
        network.add("192.168.1.23", "aaaaaaaaaaaaaaa4", mode="external")
        registry = _standalone_registry(tmp_path, network, found=["192.168.1.20", "192.168.1.21",
                                                                   "192.168.1.22", "192.168.1.23"])
        registry.scan()
        # UDP keeps the generator running on hosts without ICMP socket permission, such as CI runners.
        controller = standalone.Controller(_standalone_settings(protocol="udp"), tmp_path / "settings.json")
        client = await _standalone_client(controller, registry=registry)
        try:
            await client.post("/api/login", json={"token": "standalone-test-token"})
            data = await (await client.get("/api/status")).json()
            headers = {"X-ESPectre-CSRF": data["csrf"]}
            assert len(data["devices"]["devices"]) == 4
            ids = ["aaaaaaaaaaaaaaa1", "aaaaaaaaaaaaaaa2", "aaaaaaaaaaaaaaa3", "aaaaaaaaaaaaaaa4"]
            payload = {"device_ids": ids, "mode": "external"}
            assert (await client.post("/api/devices/mode", json=payload)).status == 403
            assert (await client.post("/api/devices/mode", json=payload, headers=headers)).status == 409
            for invalid in [{}, dict(payload, mode="flood"), dict(payload, device_ids=[]),
                            dict(payload, device_ids=ids[:1] * 2), dict(payload, extra=1),
                            dict(payload, device_ids=[f"device-{index}" for index in range(33)])]:
                assert (await client.post("/api/devices/mode", json=invalid, headers=headers)).status == 400
            assert not [call for call in network.calls if call[1] == "patch"]

            await client.post("/api/generator", json={"action": "start"}, headers=headers)
            response = await client.post("/api/devices/mode", json=payload, headers=headers)
            data = await response.json()
            assert [result["status"] for result in data["results"]] == ["confirmed", "error", "error", "unchanged"]
            # An HTTP 400 is an answer: show the device's own message, not a timeout.
            assert data["results"][1]["message"] == "traffic_generator_mode is invalid"
            modes = {device["id"]: device["sensing"]["traffic_generator_mode"] for device in data["devices"]["devices"]}
            assert modes == dict(zip(ids, ["external", "ping", "ping", "external"], strict=True))
            patches = [call for call in network.calls if call[1] == "patch"]
            assert [call[3] for call in patches] == [{"traffic_generator_mode": "external"}] * 2

            # `internal` restores the source each device used before external traffic.
            response = await client.post("/api/devices/mode", json={"device_ids": ids[:1], "mode": "internal"},
                                         headers=headers)
            restored, = (await response.json())["results"]
            assert restored["status"] == "confirmed" and restored["option"] == "ping"

            # A C6 never receives a source it does not offer.
            patch_count = len([call for call in network.calls if call[1] == "patch"])
            response = await client.post("/api/devices/mode", json={"device_ids": ids[:1], "mode": "wifi_raw"},
                                         headers=headers)
            refused, = (await response.json())["results"]
            assert refused["status"] == "error" and refused["message"] == "This device does not offer wifi_raw."
            assert len([call for call in network.calls if call[1] == "patch"]) == patch_count

            assert (await client.post("/api/devices/add", json={"address": "sensor.local"}, headers=headers)).status == 400
            assert (await client.post("/api/devices/remove", json={"device_id": ids[0]}, headers=headers)).status == 404
            assert (await client.post("/api/devices/scan", json={}, headers=headers)).status == 200
            # A finished search leaves no task behind.
            await asyncio.gather(*client.app[standalone.TASKS])
            assert not client.app[standalone.TASKS]
        finally:
            await client.close()
            controller.stop()
    asyncio.run(exercise())


def test_standalone_saved_addresses_stay_listed_and_removable_while_silent(tmp_path):
    network = _FakeDirectNetwork()
    (tmp_path / "devices.json").write_text(json.dumps({"manual": ["192.168.1.11"]}), encoding="utf-8")
    registry = _standalone_registry(tmp_path, network)

    assert registry.scan() == 0
    silent, = registry.snapshot()["devices"]
    assert silent["source"] == "manual" and silent["address"] == "192.168.1.11"
    assert silent["error"] == "The device did not answer." and not silent["can_control"]
    result, = registry.set_mode([silent["id"]], "external")
    assert result["status"] == "error"

    # Once the address answers, its row takes the device's own identity.
    network.add("192.168.1.11", "aaaaaaaaaaaaaaa1")
    registry.refresh_all()
    found, = registry.snapshot()["devices"]
    assert found["id"] == "aaaaaaaaaaaaaaa1" and found["source"] == "manual" and found["error"] is None

    # A known device that stops answering keeps its identity instead of turning back into an address.
    del network.devices["http://192.168.1.11:62587/espectre/v1"]
    registry.scan()
    lost, = registry.snapshot()["devices"]
    assert lost["id"] == "aaaaaaaaaaaaaaa1" and lost["error"] == "The device did not answer."

    registry.remove("aaaaaaaaaaaaaaa1")
    assert registry.snapshot()["devices"] == []
    assert json.loads((tmp_path / "devices.json").read_text())["manual"] == []
    registry.manual = ["192.168.1.12"]
    registry.scan()
    silent, = registry.snapshot()["devices"]
    registry.remove(silent["id"])
    assert registry.manual == [] and registry.snapshot()["devices"] == []


def test_standalone_address_removed_during_a_probe_stays_removed(tmp_path):
    registry = _standalone_registry(tmp_path, _FakeDirectNetwork())
    probe = registry.probe
    for run in (registry.refresh_all, registry.scan):
        registry.probe, registry.manual = probe, ["192.168.1.11"]
        registry.scan()
        silent, = registry.snapshot()["devices"]

        # Remove arrives while the probe that will find nothing is still waiting for an answer.
        def removed_meanwhile(*args, device_id=silent["id"]):
            registry.remove(device_id)
            return probe(*args)
        registry.probe = removed_meanwhile
        run()
        assert registry.manual == [] and registry.snapshot()["devices"] == []


def test_standalone_sample_spacing_changes_only_devices_that_offer_it(tmp_path):
    async def exercise():
        network = _FakeDirectNetwork()
        network.add("192.168.1.20", "aaaaaaaaaaaaaaa1", features={"csi_sample_spacing": True})
        network.add("192.168.1.21", "aaaaaaaaaaaaaaa2", features={"csi_sample_spacing": True},
                    reject="csi_sample_spacing is busy")
        network.add("192.168.1.22", "aaaaaaaaaaaaaaa3")
        registry = _standalone_registry(tmp_path, network, found=["192.168.1.20", "192.168.1.21", "192.168.1.22"])
        registry.scan()
        controller = standalone.Controller(_standalone_settings(), tmp_path / "settings.json")
        client = await _standalone_client(controller, registry=registry)
        try:
            await client.post("/api/login", json={"token": "standalone-test-token"})
            data = await (await client.get("/api/status")).json()
            headers = {"X-ESPectre-CSRF": data["csrf"]}
            supported = {device["id"]: device["spacing_supported"] for device in data["devices"]["devices"]}
            assert supported == {"aaaaaaaaaaaaaaa1": True, "aaaaaaaaaaaaaaa2": True, "aaaaaaaaaaaaaaa3": False}
            ids = sorted(supported)
            for invalid in [{}, {"device_ids": ids, "spacing": "quarter_slot"}, {"device_ids": [], "spacing": "distinct"},
                            {"device_ids": ids, "spacing": "distinct", "mode": "external"}]:
                assert (await client.post("/api/devices/spacing", json=invalid, headers=headers)).status == 400
            # Spacing needs no running generator: it changes admission, not the traffic source.
            response = await client.post("/api/devices/spacing", json={"device_ids": ids, "spacing": "distinct"},
                                         headers=headers)
            data = await response.json()
            assert [result["status"] for result in data["results"]] == ["confirmed", "error", "error"]
            assert data["results"][1]["message"] == "csi_sample_spacing is busy"
            spacing = {device["id"]: device["sensing"]["csi_sample_spacing"] for device in data["devices"]["devices"]}
            assert spacing == dict(zip(ids, ["distinct", "half_slot", "half_slot"], strict=True))
            patches = [call[3] for call in network.calls if call[1] == "patch"]
            assert patches == [{"csi_sample_spacing": "distinct"}] * 2
            again, = (await (await client.post("/api/devices/spacing", json={"device_ids": ids[:1], "spacing": "distinct"},
                                               headers=headers)).json())["results"]
            assert again["status"] == "unchanged"
        finally:
            await client.close()
            controller.stop()
    asyncio.run(exercise())
