"""
Unit tests for the unified CSI stream protocol parser and dataset writer.
"""

import socket

import numpy as np

import csi_utils
from csi_utils import (
    CSICollector,
    CSIReceiver,
    CSI_HEADER_STRUCT,
    MAGIC_STREAM,
    STIMULUS_HEADER_STRUCT,
    STIMULUS_MAGIC,
    STIMULUS_ROLE_MEASUREMENT,
    STIMULUS_ROLE_REFERENCE,
    STIMULUS_VERSION,
    STREAM_FLAG_GAIN_INFO_VALID,
    STREAM_FLAG_GAIN_LOCKED,
    STREAM_FLAG_REFERENCE_FRAME,
    STREAM_FLAG_STIMULUS_ID_VALID,
    STREAM_FLAG_WIFI_RX_START_TS_NS_VALID,
    STREAM_FLAG_WIFI_RX_TS_VALID,
    STREAM_VERSION,
    StimulusSender,
    build_stimulus_datagram,
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
    stimulus_id=0,
    channel=6,
    rssi_dbm=-42,
    noise_floor_dbm=-96,
    agc_gain=0,
    fft_gain=0,
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
        stimulus_id,
        channel,
        rssi_dbm,
        noise_floor_dbm,
        agc_gain,
        fft_gain,
        0,
    )
    return header + payload


def test_parse_packet_accepts_unified_stream_header():
    receiver = CSIReceiver(bind_host='127.0.0.1')
    packet = receiver._parse_packet(
        build_packet(
            seq_num=7,
            flags=STREAM_FLAG_GAIN_LOCKED,
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
    assert packet.gain_locked is True
    assert packet.device_ticks_us == 987654
    assert packet.channel == 11
    assert packet.rssi_dbm == -55
    np.testing.assert_array_equal(packet.iq_raw, np.array([10, 20, -30, 40], dtype=np.int8))
    np.testing.assert_allclose(packet.iq_complex, np.array([20 + 10j, 40 - 30j], dtype=np.complex64))


def test_build_stimulus_datagram_uses_estm_wire_format():
    datagram = build_stimulus_datagram(0x01020304)
    magic, version, role, stimulus_id = STIMULUS_HEADER_STRUCT.unpack(datagram)

    assert magic == STIMULUS_MAGIC
    assert version == STIMULUS_VERSION
    assert role == STIMULUS_ROLE_MEASUREMENT
    assert stimulus_id == 0x01020304


def test_build_stimulus_datagram_marks_reference_role():
    datagram = build_stimulus_datagram(77, is_reference=True)
    _magic, _version, role, stimulus_id = STIMULUS_HEADER_STRUCT.unpack(datagram)

    assert role == STIMULUS_ROLE_REFERENCE
    assert stimulus_id == 77


def test_stimulus_sender_emits_incrementing_estm_packets():
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(("127.0.0.1", 0))
    sock.settimeout(1.0)

    sender = StimulusSender(
        target_host="127.0.0.1",
        target_port=sock.getsockname()[1],
        rate_pps=200,
        reference_every=2,
        stimulus_id_start=10,
    )
    try:
        sender.start()
        first, _addr = sock.recvfrom(64)
        second, _addr = sock.recvfrom(64)
    finally:
        sender.stop()
        sock.close()

    first_magic, first_version, first_role, first_id = STIMULUS_HEADER_STRUCT.unpack(first)
    second_magic, second_version, second_role, second_id = STIMULUS_HEADER_STRUCT.unpack(second)

    assert first_magic == STIMULUS_MAGIC
    assert second_magic == STIMULUS_MAGIC
    assert first_version == STIMULUS_VERSION
    assert second_version == STIMULUS_VERSION
    assert first_role == STIMULUS_ROLE_MEASUREMENT
    assert second_role == STIMULUS_ROLE_REFERENCE
    assert first_id == 10
    assert second_id == 11


def test_stimulus_sender_binds_to_requested_source_host():
    rx_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    rx_sock.bind(("127.0.0.1", 0))
    rx_sock.settimeout(1.0)

    sender = StimulusSender(
        target_host="127.0.0.1",
        target_port=rx_sock.getsockname()[1],
        rate_pps=50,
        source_host="127.0.0.1",
    )
    try:
        sender.start()
        _payload, addr = rx_sock.recvfrom(64)
    finally:
        sender.stop()
        rx_sock.close()

    assert addr[0] == "127.0.0.1"


def test_parse_packet_reads_optional_metadata():
    receiver = CSIReceiver(bind_host='127.0.0.1')
    flags = (
        STREAM_FLAG_GAIN_LOCKED
        | STREAM_FLAG_WIFI_RX_TS_VALID
        | STREAM_FLAG_WIFI_RX_START_TS_NS_VALID
        | STREAM_FLAG_GAIN_INFO_VALID
        | STREAM_FLAG_STIMULUS_ID_VALID
        | STREAM_FLAG_REFERENCE_FRAME
    )
    packet = receiver._parse_packet(
        build_packet(
            seq_num=42,
            flags=flags,
            wifi_rx_ts_us=5555,
            wifi_rx_start_ts_ns=987654321,
            stimulus_id=1234,
            agc_gain=77,
            fft_gain=-8,
        )
    )

    assert packet is not None
    assert packet.wifi_rx_ts_us == 5555
    assert packet.wifi_rx_start_ts_ns == 987654321
    assert packet.stimulus_id == 1234
    assert packet.is_reference is True
    assert packet.agc_gain == 77
    assert packet.fft_gain == -8


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
    monkeypatch.setattr(csi_utils, 'DATA_DIR', data_dir)
    monkeypatch.setattr(csi_utils, 'DATASET_INFO_FILE', data_dir / 'dataset_info.json')

    receiver = CSIReceiver(bind_host='127.0.0.1')
    flags = (
        STREAM_FLAG_GAIN_LOCKED
        | STREAM_FLAG_WIFI_RX_TS_VALID
        | STREAM_FLAG_GAIN_INFO_VALID
    )
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
                agc_gain=33,
                fft_gain=-3,
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
                agc_gain=34,
                fft_gain=-2,
            )
        ),
    ]

    collector = CSICollector(label='baseline', contributor='tester', bind_host='127.0.0.1')
    filepath = collector.save_sample(packets)

    assert filepath is not None
    assert filepath.exists()

    data = np.load(filepath, allow_pickle=True)
    assert str(data['label']) == 'baseline'
    assert str(data['chip']) == 'c6'
    assert int(data['num_subcarriers']) == 2
    assert bool(data['gain_locked']) is True
    assert str(data['format_version']) == '1.1'
    assert int(data['device_id']) == 0xABCDEF
    np.testing.assert_array_equal(data['stream_seq_num'], np.array([100, 101], dtype=np.uint32))
    np.testing.assert_array_equal(data['device_ticks_us'], np.array([1000, 2000], dtype=np.uint64))
    np.testing.assert_array_equal(data['wifi_rx_ts_us'], np.array([4000, 5000], dtype=np.uint32))
    np.testing.assert_array_equal(data['csi_data'], np.array([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=np.int8))

    info = csi_utils.load_dataset_info()
    assert info['format_version'] == '1.1'
    assert info['files']['baseline'][0]['gain_locked'] is True
    assert info['files']['baseline'][0]['filename'] == filepath.name
