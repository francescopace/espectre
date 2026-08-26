# SPDX-License-Identifier: GPL-3.0-only
# Commercial licensing available under separate agreement; see LICENSING.md.
"""Contracts for the Micro-ESPectre native traffic-generator facade."""

import sys
import time
from unittest.mock import MagicMock

import pytest


mock_network = MagicMock()
mock_network.STA_IF = 0
sys.modules["network"] = mock_network


class MockNativeTraffic:
    def __init__(self):
        self.start_result = True
        self.start_error = None
        self.running = False
        self.start_calls = []
        self.stop_calls = 0
        self.pause_result = True
        self.resume_result = True
        self.reopen_result = True
        self.sent_packets = 0
        self.send_errors = 0

    def start(self, gateway, rate_pps):
        self.start_calls.append((gateway, rate_pps))
        if self.start_error is not None:
            raise self.start_error
        self.running = self.start_result
        return self.start_result

    def stop(self):
        self.stop_calls += 1
        self.running = False

    def pause(self):
        return self.pause_result

    def resume(self):
        return self.resume_result

    def reopen(self):
        return self.reopen_result

    def is_running(self):
        return self.running

    def packet_count(self):
        return self.sent_packets

    def error_count(self):
        return self.send_errors


mock_native_traffic = MagicMock()
mock_native_traffic.TrafficGenerator = MockNativeTraffic
sys.modules["espectre_native_traffic"] = mock_native_traffic

if not hasattr(time, "ticks_ms"):
    time.ticks_ms = lambda: int(time.time() * 1000)

from traffic_generator import (  # noqa: E402
    MODE_PING,
    TRAFFIC_RATE_MAX,
    TRAFFIC_RATE_MIN,
    TrafficGenerator,
)


@pytest.fixture
def mock_wlan():
    wlan = MagicMock()
    wlan.isconnected.return_value = True
    wlan.ifconfig.return_value = (
        "192.168.1.100",
        "255.255.255.0",
        "192.168.1.1",
        "8.8.8.8",
    )
    mock_network.WLAN.return_value = wlan
    mock_network.WLAN.side_effect = None
    return wlan


@pytest.fixture
def traffic_gen():
    return TrafficGenerator()


def test_init_requires_native_backend(traffic_gen):
    assert isinstance(traffic_gen._native_traffic, MockNativeTraffic)
    assert traffic_gen.running is False
    assert traffic_gen.rate_pps == 0
    assert traffic_gen.packet_count == 0
    assert traffic_gen.error_count == 0
    assert traffic_gen.gateway_ip is None
    assert traffic_gen.sock is None
    assert not hasattr(traffic_gen, "_run_sender_task")


def test_start_delegates_ping_to_native_backend(mock_wlan):
    generator = TrafficGenerator()
    backend = generator._native_traffic
    backend.sent_packets = 321
    backend.send_errors = 2

    assert generator.start(100) is True
    assert backend.start_calls == [("192.168.1.1", 100)]
    assert generator.get_packet_count() == 321
    assert generator.get_error_count() == 2
    assert generator.is_running() is True


def test_start_rejects_invalid_or_disabled_rates(traffic_gen):
    assert traffic_gen.start(-1) is False
    assert traffic_gen.start(0) is False
    assert traffic_gen.start(TRAFFIC_RATE_MAX + 1) is False
    assert traffic_gen._native_traffic.start_calls == []


def test_start_rejects_second_active_start(mock_wlan):
    generator = TrafficGenerator()
    assert generator.start(100)
    assert generator.start(100) is False
    assert len(generator._native_traffic.start_calls) == 1


def test_start_fails_without_gateway(traffic_gen, mock_wlan):
    mock_wlan.isconnected.return_value = False

    assert traffic_gen.start(100, max_retries=1, retry_delay=0) is False
    assert traffic_gen._native_traffic.start_calls == []


def test_start_handles_native_failure(mock_wlan):
    rejected = TrafficGenerator()
    rejected._native_traffic.start_result = False
    assert rejected.start(100) is False
    assert rejected.rate_pps == 0
    assert rejected.target_pps == 0

    failed = TrafficGenerator()
    failed._native_traffic.start_error = OSError("native unavailable")
    assert failed.start(100) is False
    assert failed.running is False


def test_gateway_lookup_contract(traffic_gen, mock_wlan):
    assert traffic_gen._get_gateway_ip() == "192.168.1.1"

    mock_wlan.ifconfig.return_value = ("192.168.1.100",)
    assert traffic_gen._get_gateway_ip() is None

    mock_wlan.isconnected.return_value = False
    assert traffic_gen._get_gateway_ip() is None

    mock_network.WLAN.side_effect = OSError("network unavailable")
    assert traffic_gen._get_gateway_ip() is None


def test_mode_validation_and_live_change_guard(traffic_gen):
    with pytest.raises(ValueError, match="Invalid traffic generator mode"):
        traffic_gen.set_mode("udp")
    with pytest.raises(ValueError, match="Invalid traffic generator mode"):
        traffic_gen.set_mode("dns")

    traffic_gen.running = True
    assert traffic_gen.set_mode(MODE_PING) is False
    assert traffic_gen.get_mode() == MODE_PING


def test_pause_resume_and_reopen_delegate_to_native(mock_wlan):
    generator = TrafficGenerator()
    assert generator.start(60)
    assert generator.pause()
    assert generator.paused
    assert generator.resume()
    assert not generator.paused
    assert generator.reopen_socket()

    generator._native_traffic.pause_result = False
    assert generator.pause() is False
    generator._native_traffic.resume_result = False
    assert generator.resume() is False
    generator._native_traffic.reopen_result = False
    assert generator.reopen_socket() is False


def test_stop_retains_native_counters(mock_wlan):
    generator = TrafficGenerator()
    assert generator.start(100)
    generator._native_traffic.sent_packets = 123
    generator._native_traffic.send_errors = 4

    generator.stop()

    assert generator._native_traffic.stop_calls == 1
    assert generator.running is False
    assert generator.get_packet_count() == 123
    assert generator.get_error_count() == 4
    assert generator.get_rate() == 0
    assert generator.get_target_rate() == 0


def test_stopped_lifecycle_calls_are_noops(traffic_gen):
    traffic_gen.stop()
    assert traffic_gen.pause() is False
    assert traffic_gen.resume() is False
    assert traffic_gen.reopen_socket() is False
    assert traffic_gen.is_running() is False


def test_metrics_getters(traffic_gen):
    traffic_gen.actual_pps = 99.5678
    traffic_gen.avg_loop_time_ms = 9.5678
    assert traffic_gen.get_actual_pps() == 99.6
    assert traffic_gen.get_avg_loop_time_ms() == 9.57
    assert TRAFFIC_RATE_MIN == 0
    assert TRAFFIC_RATE_MAX == 1000
