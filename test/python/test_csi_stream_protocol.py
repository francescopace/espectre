"""
Unit tests for the unified CSI stream protocol parser and dataset writer.
"""

import io
import socket

import numpy as np
import pytest

from tools.lib import dataset_metadata
from tools.lib.csi_io import (
    AdaptivePacingController,
    UdpPacingSender,
    CSICollector,
    CSIReceiver,
    CSI_HEADER_STRUCT,
    DEFAULT_PACING_INTERVAL_SECONDS,
    DEFAULT_PACING_PAYLOAD,
    MAGIC_STREAM,
    STREAM_FLAG_WIFI_RX_START_TS_NS_VALID,
    STREAM_FLAG_WIFI_RX_TS_VALID,
    STREAM_VERSION,
    build_pacing_datagram,
)


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
    tx_backpressure_total=0,
    stream_fresh_total=0,
    pacing_rx_total=0,
):
    payload_values = payload if payload is not None else [1, 2, 3, 4]
    payload = np.array(payload_values, dtype=np.int8).tobytes()
    num_sc = len(payload) // 2
    header = CSI_HEADER_STRUCT.pack(
        MAGIC_STREAM,
        STREAM_VERSION,
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
        tx_backpressure_total,
        stream_fresh_total,
        pacing_rx_total,
    )
    return header + payload


def test_parse_packet_accepts_unified_stream_header():
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
    assert packet.tx_backpressure_total == 0
    np.testing.assert_array_equal(packet.iq_raw, np.array([10, 20, -30, 40], dtype=np.int8))
    np.testing.assert_allclose(packet.iq_complex, np.array([20 + 10j, 40 - 30j], dtype=np.complex64))


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


def test_build_pacing_datagram_uses_default_pacing_payload():
    datagram = build_pacing_datagram()
    assert datagram == b"ESPE"
    assert datagram == DEFAULT_PACING_PAYLOAD


def test_udp_pacing_sender_emits_udp_pacing_packets():
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(("127.0.0.1", 0))
    sock.settimeout(6.0)

    sender = UdpPacingSender(
        target_host="127.0.0.1",
        target_port=sock.getsockname()[1],
        interval_s=0.05,
    )
    try:
        sender.start()
        first, _addr = sock.recvfrom(64)
        second, _addr = sock.recvfrom(64)
    finally:
        sender.stop()
        sock.close()

    assert first == DEFAULT_PACING_PAYLOAD
    assert second == DEFAULT_PACING_PAYLOAD


def test_udp_pacing_sender_binds_to_requested_source_host():
    rx_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    rx_sock.bind(("127.0.0.1", 0))
    rx_sock.settimeout(1.0)

    sender = UdpPacingSender(
        target_host="127.0.0.1",
        target_port=rx_sock.getsockname()[1],
        source_host="127.0.0.1",
        interval_s=0.05,
    )
    try:
        sender.start()
        _payload, addr = rx_sock.recvfrom(64)
    finally:
        sender.stop()
        rx_sock.close()

    assert addr[0] == "127.0.0.1"


def test_udp_pacing_sender_fans_out_same_datagram_to_multiple_targets():
    class FakeSocket:
        def __init__(self):
            self.calls = []

        def sendto(self, payload, addr):
            self.calls.append((payload, addr))

    sender = UdpPacingSender(
        target_host=["192.168.1.17", "192.168.1.24", "192.168.1.29"],
        target_port=9999,
    )
    sender.sock = FakeSocket()

    original_wait = sender._stop_event.wait

    def stop_after_first_interval(_timeout):
        sender._stop_event.set()
        return True

    sender._stop_event.wait = stop_after_first_interval
    try:
        sender._run()
    finally:
        sender._stop_event.wait = original_wait

    assert len(sender.sock.calls) == 3
    payloads = {payload for payload, _addr in sender.sock.calls}
    assert len(payloads) == 1
    assert [addr for _payload, addr in sender.sock.calls] == [
        ("192.168.1.17", 9999),
        ("192.168.1.24", 9999),
        ("192.168.1.29", 9999),
    ]
    assert next(iter(payloads)) == DEFAULT_PACING_PAYLOAD


def test_udp_pacing_sender_uses_default_interval():
    sender = UdpPacingSender(target_host="192.168.1.17", target_port=9999)

    assert sender.interval_s == DEFAULT_PACING_INTERVAL_SECONDS


def test_udp_pacing_sender_updates_rate_safely():
    sender = UdpPacingSender(target_host="192.168.1.17", target_port=9999, interval_s=0.1)

    sender.set_rate_pps(25)

    assert sender.get_rate_pps() == pytest.approx(25.0)
    assert sender.interval_s == pytest.approx(0.04)


def test_udp_pacing_sender_uses_absolute_deadlines(monkeypatch):
    sender = UdpPacingSender(target_host="192.168.1.17", target_port=9999, interval_s=0.1)
    clock = {"now": 10.0}
    waits = []
    sends = {"count": 0}

    monkeypatch.setattr("tools.lib.csi_io.time.perf_counter", lambda: clock["now"])

    def fake_send_once():
        sends["count"] += 1
        clock["now"] += 0.03

    def fake_wait(timeout):
        waits.append(timeout)
        clock["now"] += timeout
        if len(waits) >= 3:
            sender._stop_event.set()
        return False

    sender._send_once = fake_send_once
    sender._stop_event.wait = fake_wait

    sender._run()

    assert sends["count"] == 3
    assert waits == pytest.approx([0.07, 0.07, 0.07], abs=1e-6)


def test_parse_packet_reads_optional_metadata():
    receiver = CSIReceiver(bind_host='127.0.0.1')
    flags = STREAM_FLAG_WIFI_RX_TS_VALID | STREAM_FLAG_WIFI_RX_START_TS_NS_VALID
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


def test_parse_packet_reads_tx_backpressure_total():
    receiver = CSIReceiver(bind_host='127.0.0.1')
    packet = receiver._parse_packet(
        build_packet(
            seq_num=43,
            tx_backpressure_total=17,
        )
    )

    assert packet is not None
    assert packet.tx_backpressure_total == 17


def test_parse_packet_reads_adaptive_up_counters():
    receiver = CSIReceiver(bind_host='127.0.0.1')
    packet = receiver._parse_packet(
        build_packet(
            seq_num=44,
            stream_fresh_total=123,
            pacing_rx_total=140,
        )
    )

    assert packet is not None
    assert packet.stream_fresh_total == 123
    assert packet.pacing_rx_total == 140


def test_adaptive_pacing_controller_speeds_up_when_device_under_target_without_backpressure():
    class FakePacingSender:
        def __init__(self):
            self.rate_updates = []

        def set_rate_pps(self, rate_pps):
            self.rate_updates.append(float(rate_pps))

    controller = AdaptivePacingController(initial_pps=100.0, enabled=True, control_window_s=1.0)
    sender = FakePacingSender()
    device_state = {"source_ip": "192.168.1.17"}

    controller.observe_device(device_state, 0, 0, 0)
    controller.maybe_adjust({1: device_state}, now=0.0, pacing_sender=sender)

    controller.observe_device(device_state, 0, 80, 80)
    controller.maybe_adjust({1: device_state}, now=1.1, pacing_sender=sender)

    controller.observe_device(device_state, 0, 160, 160)
    controller.maybe_adjust({1: device_state}, now=2.2, pacing_sender=sender)

    assert sender.rate_updates == pytest.approx([102.0])
    assert controller.last_action == "speedup"


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


def test_parse_packet_rejects_legacy_python_header():
    receiver = CSIReceiver(bind_host='127.0.0.1')
    legacy = bytes([0x53, 0x43, 0x04, 0x01, 0x07, 0x40, 0x00]) + bytes(128)
    assert receiver._parse_packet(legacy) is None


def test_save_sample_keeps_existing_schema_and_adds_optional_metadata(tmp_path, monkeypatch):
    data_dir = tmp_path / 'data'
    monkeypatch.setattr(dataset_metadata, 'DATA_DIR', data_dir)
    monkeypatch.setattr(dataset_metadata, 'DATASET_INFO_FILE', data_dir / 'dataset_info.json')

    receiver = CSIReceiver(bind_host='127.0.0.1')
    flags = STREAM_FLAG_WIFI_RX_TS_VALID
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

    collector = CSICollector(label='static_presence', contributor='tester', bind_host='127.0.0.1')
    filepath = collector.save_sample(packets)

    assert filepath is not None
    assert filepath.exists()

    data = np.load(filepath, allow_pickle=True)
    assert str(data['label']) == 'static_presence'
    assert str(data['chip']) == 'c6'
    assert int(data['num_subcarriers']) == 2
    assert str(data['format_version']) == '1.2'
    assert int(data['device_id']) == 0xABCDEF
    np.testing.assert_array_equal(data['stream_seq_num'], np.array([100, 101], dtype=np.uint32))
    np.testing.assert_array_equal(data['device_ticks_us'], np.array([1000, 2000], dtype=np.uint64))
    np.testing.assert_array_equal(data['wifi_rx_ts_us'], np.array([4000, 5000], dtype=np.uint32))
    np.testing.assert_array_equal(data['csi_data'], np.array([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=np.int8))

    info = dataset_metadata.load_dataset_info()
    assert info['format_version'] == '1.2'
    assert info['files']['static_presence'][0]['filename'] == filepath.name
    assert info['files']['static_presence'][0]['device_id'] == '0x0000000000abcdef'
    assert info['files']['static_presence'][0]['description'] == 'HT20 static presence sample'
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
    data = np.load(filepath, allow_pickle=True)
    assert int(data['num_subcarriers']) == 12
    assert data['csi_data'].shape == (1, 24)
    np.testing.assert_array_equal(data['csi_data'][0], np.array(short_payload, dtype=np.int8))
    assert '_12sc_' in filepath.name


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
        '0x0000000000000010',
        '0x0000000000000020',
    }


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

    last_seq_by_device = {}
    drops = [CSICollector._check_sequence_by_device(packet, last_seq_by_device) for packet in packets]

    assert drops == [0, 0, 0, 0, 1]


def test_check_sequence_by_device_ignores_large_backward_jump():
    receiver = CSIReceiver(bind_host='127.0.0.1')
    packets = [
        receiver._parse_packet(build_packet(seq_num=100, device_id=0x1)),
        receiver._parse_packet(build_packet(seq_num=101, device_id=0x1)),
        receiver._parse_packet(build_packet(seq_num=3, device_id=0x1)),
        receiver._parse_packet(build_packet(seq_num=4, device_id=0x1)),
    ]

    last_seq_by_device = {}
    drops = [CSICollector._check_sequence_by_device(packet, last_seq_by_device) for packet in packets]

    assert drops == [0, 0, 0, 0]


def test_summarize_ready_devices_waits_for_all_expected_devices():
    now = 100.0
    warmup_target = 10
    threshold = CSICollector.READY_MV_THRESHOLD

    summary = CSICollector._summarize_ready_devices(
        {
            0x1: {'processed_packets': warmup_target, 'stable_since': now - 4.0, 'current_mv': 0.2},
        },
        expected_device_count=2,
        warmup_target=warmup_target,
        threshold=threshold,
        now=now,
    )
    assert summary['ready'] is False
    assert summary['status'] == 'DEVICES 1/2'

    summary = CSICollector._summarize_ready_devices(
        {
            0x1: {'processed_packets': warmup_target, 'stable_since': now - 4.0, 'current_mv': 0.2},
            0x2: {'processed_packets': warmup_target, 'stable_since': now - 1.0, 'current_mv': 0.3},
        },
        expected_device_count=2,
        warmup_target=warmup_target,
        threshold=threshold,
        now=now,
    )
    assert summary['ready'] is False
    assert summary['status'] == 'STABLE 2/2'
    assert summary['stable_elapsed'] == pytest.approx(1.0)

    summary = CSICollector._summarize_ready_devices(
        {
            0x1: {'processed_packets': warmup_target, 'stable_since': now - 4.0, 'current_mv': 0.2},
            0x2: {'processed_packets': warmup_target, 'stable_since': now - 3.5, 'current_mv': 0.3},
        },
        expected_device_count=2,
        warmup_target=warmup_target,
        threshold=threshold,
        now=now,
    )
    assert summary['ready'] is True
    assert summary['status'] == 'READY 2/2'


def test_format_ready_device_lines_includes_waiting_ip_and_device_details():
    now = 100.0
    lines = CSICollector._format_ready_device_lines(
        {
            0x1: {
                'processed_packets': 12,
                'stable_since': now - 3.5,
                'current_mv': 0.2,
                'current_pps': 121,
                'source_ip': '192.168.1.17',
                'chip': 'c6',
                'channel': 6,
                'rssi_dbm': -48,
            },
            0x2: {
                'processed_packets': 8,
                'stable_since': None,
                'current_mv': 0.0,
                'current_pps': 87,
                'source_ip': '192.168.1.24',
                'chip': 'c3',
                'channel': 11,
                'rssi_dbm': -63,
            },
        },
        expected_source_hosts=['192.168.1.17', '192.168.1.24', '192.168.1.29'],
        warmup_target=10,
        threshold=CSICollector.READY_MV_THRESHOLD,
        now=now,
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


def test_summarize_ready_devices_does_not_expose_global_mv_metrics():
    now = 100.0
    warmup_target = 10
    threshold = CSICollector.READY_MV_THRESHOLD

    summary = CSICollector._summarize_ready_devices(
        {
            0x1: {'processed_packets': warmup_target, 'stable_since': now - 4.0, 'current_mv': 0.2},
            0x2: {'processed_packets': warmup_target, 'stable_since': None, 'current_mv': 2.5},
        },
        expected_device_count=2,
        warmup_target=warmup_target,
        threshold=threshold,
        now=now,
    )

    assert summary['status'] == 'UNSTABLE 1/2'
    assert 'max_mv' not in summary
    assert 'ready_ratio' not in summary


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
