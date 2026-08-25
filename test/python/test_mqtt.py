# SPDX-License-Identifier: GPL-3.0-only
# Commercial licensing available under separate agreement; see LICENSING.md.
"""
ESPectre - MQTT Tests

Unit tests for MQTT communication helpers.

Author: Francesco Pace <francesco.pace@gmail.com>
"""

import pytest
import json
import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

# Mock native MicroPython firmware modules before importing mqtt modules
mock_mqtt_client = MagicMock()
sys.modules['espectre_native_mqtt'] = MagicMock()
sys.modules['espectre_native_mqtt'].MQTTClient = mock_mqtt_client

# Mock network module (MicroPython)
mock_network = MagicMock()
mock_network.MODE_11B = 1
mock_network.MODE_11G = 2
mock_network.MODE_11N = 4
mock_network.MODE_LR = 8
sys.modules['network'] = mock_network

class MockWLAN:
    """Mock WLAN interface for testing"""
    
    def __init__(self, connected=True):
        self._connected = connected
        self._active = True
        
    def active(self):
        return self._active
    
    def isconnected(self):
        return self._connected
    
    def config(self, key):
        configs = {
            'mac': b'\x12\x34\x56\x78\x9a\xbc',
            'channel': 6,
            'protocol': 7  # b/g/n
        }
        return configs.get(key, 0)
    
    def ifconfig(self):
        return ('192.168.1.100', '255.255.255.0', '192.168.1.1', '8.8.8.8')


def test_device_id_matches_native_sha256_pseudonym():
    from mqtt.protocol import _derive_device_id_from_mac

    assert _derive_device_id_from_mac(bytes.fromhex("7c2c6742bbac")) == "3cf79180d3a0aca4"


def test_cpp_and_micropython_command_catalogs_match():
    """Keep the independent C++ and MicroPython command registries aligned."""
    from mqtt.protocol import build_capabilities_payload

    repo_root = Path(__file__).resolve().parents[2]
    build_dir = repo_root / "test" / "cpp" / "build"
    probe = build_dir / "suites" / "espectre_capabilities_probe"
    if not probe.exists():
        subprocess.run(
            ["cmake", "-S", str(repo_root / "test" / "cpp"), "-B", str(build_dir)],
            cwd=repo_root,
            check=True,
        )
        subprocess.run(
            ["cmake", "--build", str(build_dir), "--target", "espectre_capabilities_probe"],
            cwd=repo_root,
            check=True,
        )

    cpp_catalog_text = subprocess.run(
        [str(probe)],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    cpp_catalog = json.loads(cpp_catalog_text)
    micro_catalog = build_capabilities_payload(
        "0000000000000000",
        supports_info=True,
        supports_diagnostics=True,
        supports_device_config=False,
        supports_runtime_threshold=True,
        supports_runtime_motion_hits=True,
        supports_runtime_detector=True,
        supports_manual_recalibration=True,
        supports_traffic_control=True,
        supports_ota=False,
    )

    assert cpp_catalog == micro_catalog
    assert len(cpp_catalog_text.encode("utf-8")) < 4096
    assert len(json.dumps(micro_catalog, separators=(",", ":")).encode("utf-8")) < 4096


class MockConfig:
    """Mock configuration module"""
    MQTT_BROKER = "localhost"
    MQTT_PORT = 1883
    MQTT_USERNAME = "user"
    MQTT_PASSWORD = "pass"
    MQTT_TOPIC_PREFIX = "test/espectre/devices"
    MQTT_HA_DISCOVERY_ENABLED = False
    MQTT_HA_DISCOVERY_PREFIX = "homeassistant"
    MQTT_DEVICE_LABEL = ""
    PUBLISH_INTERVAL_MS = 1000
    EVALUATION_INTERVAL_MS = 250
    MOTION_ON_HITS = 3
    MOTION_OFF_HITS = 3
    DEFAULT_SUBCARRIERS = None


class MockSegmentation:
    """Mock SegmentationContext for testing shared detector context state."""
    STATE_IDLE = 0
    STATE_MOTION = 1
    
    def __init__(self):
        self.window_size = 50
        self.state = self.STATE_IDLE
        self.last_turbulence = 2.5
        self.turbulence_buffer = [0.0] * 50
        self.buffer_index = 0
        self.buffer_count = 0


class MockDetector:
    """Mock IDetector implementation for testing"""
    ALGORITHM = "lightweight"
    
    def __init__(self):
        self._threshold = 1.0
        self._state = 0  # IDLE
        self._motion_metric = 0.5
        # Segmentation-like _context for compatibility
        self._context = MockSegmentation()
    
    def get_name(self):
        return "Lightweight"
    
    def get_threshold(self):
        return self._threshold
    
    def set_threshold(self, threshold):
        if 0.0 <= threshold <= 10.0:
            self._threshold = threshold
            return True
        return False
    
    def get_state(self):
        return self._state
    
    def get_motion_metric(self):
        return self._motion_metric
    
    def reset(self):
        self._state = 0
        self._motion_metric = 0.0


class MockGlobalState:
    """Mock global state for testing"""
    
    def __init__(self):
        self.loop_time_us = 5000  # 5ms
        self.chip_type = 'c6'


class MockRuntimePolicy:
    """Mock runtime motion policy for session-only MQTT updates."""

    def __init__(self, motion_on_hits=3, motion_off_hits=3):
        self.motion_on_hits = motion_on_hits
        self.motion_off_hits = motion_off_hits


class MockTrafficGenerator:
    """Mock Micro traffic generator for session-only traffic-control updates."""

    def __init__(self, mode="ping", running=False):
        self.mode = mode
        self.running = running
        self.set_mode_calls = []
        self.start_calls = []
        self.stop_calls = 0

    def is_running(self):
        return self.running

    def set_mode(self, mode):
        self.set_mode_calls.append(mode)
        self.mode = mode
        return True

    def start(self, rate_pps):
        self.start_calls.append(rate_pps)
        self.running = True
        return True

    def stop(self):
        self.stop_calls += 1
        self.running = False


@pytest.fixture
def mock_mqtt_client_instance():
    """Create a mock MQTT client instance"""
    client = MagicMock()
    client.connect = MagicMock()
    client.publish = MagicMock()
    client.subscribe = MagicMock()
    client.set_callback = MagicMock()
    client.check_msg = MagicMock()
    client.disconnect = MagicMock()
    client.deinit = MagicMock()
    client.set_last_will = MagicMock()
    return client


@pytest.fixture(autouse=True)
def stable_device_identity(monkeypatch):
    """Keep topic-focused tests independent from the identity hash vector."""
    from mqtt import protocol as mqtt_protocol

    monkeypatch.setattr(
        mqtt_protocol,
        "derive_runtime_device_id",
        lambda _wlan: "test-device",
    )


@pytest.fixture
def mock_wlan():
    """Create mock WLAN"""
    return MockWLAN()


@pytest.fixture
def mock_config(default_subcarriers):
    """Create mock config with default subcarriers from conftest"""
    config = MockConfig()
    config.DEFAULT_SUBCARRIERS = default_subcarriers
    return config


@pytest.fixture
def mock_segmentation():
    """Create mock detector"""
    return MockDetector()


@pytest.fixture
def mock_global_state():
    """Create mock global state"""
    return MockGlobalState()


@pytest.fixture
def mock_runtime_policy():
    """Create a mock motion-hit runtime policy."""
    return MockRuntimePolicy()


@pytest.fixture
def mock_traffic_generator():
    """Create a mock traffic generator."""
    return MockTrafficGenerator()


class TestMQTTHandler:
    """Test MQTTHandler class"""
    
    def test_init(self, mock_config, mock_segmentation, mock_wlan):
        """Test handler initialization"""
        from mqtt.handler import MQTTHandler
        
        handler = MQTTHandler(mock_config, mock_segmentation, mock_wlan)
        
        assert handler.config == mock_config
        assert handler.detector == mock_segmentation
        assert handler.wlan == mock_wlan
        assert handler.base_topic == "test/espectre/devices/test-device"
        assert handler.telemetry_topic == "test/espectre/devices/test-device/telemetry"
        assert handler.cmd_topic == "test/espectre/devices/test-device/commands/request"
        assert handler.result_topic == "test/espectre/devices/test-device/commands/result"
        assert handler.capabilities_topic == "test/espectre/devices/test-device/capabilities"
        assert handler.config_topic == "test/espectre/devices/test-device/config"

    @patch('mqtt.handler.MQTTClient')
    def test_initial_connection_failure_schedules_reconnect(
        self,
        mock_client_class,
        mock_config,
        mock_segmentation,
        mock_wlan,
    ):
        """A missing broker at boot must not terminate the sensing runtime."""
        from mqtt.handler import MQTTHandler

        mock_client_class.return_value.connect.side_effect = OSError("offline")
        handler = MQTTHandler(mock_config, mock_segmentation, mock_wlan)

        assert handler.connect() is None
        assert handler.connected is False
        assert handler._next_reconnect_ms != 0

    def test_native_transport_resolves_broker_before_client_start(
        self,
        monkeypatch,
        mock_config,
        mock_segmentation,
        mock_wlan,
    ):
        """The native ESP-IDF client should receive an address resolved by MicroPython."""
        import mqtt.handler as handler_module
        import socket

        client = MagicMock()
        client_factory = MagicMock(return_value=client)
        monkeypatch.setattr(handler_module, "MQTTClient", client_factory)
        monkeypatch.setattr(
            socket,
            "getaddrinfo",
            lambda *_args, **_kwargs: [
                (socket.AF_INET, socket.SOCK_STREAM, 0, "", ("192.0.2.10", 1883))
            ],
        )

        handler = handler_module.MQTTHandler(mock_config, mock_segmentation, mock_wlan)
        handler._connect_client()

        assert client_factory.call_args.args[:2] == ("test-device", "192.0.2.10")
        assert client_factory.call_args.kwargs["last_will_topic"] == handler.status_topic
        assert json.loads(client_factory.call_args.kwargs["last_will_msg"])["online"] is False
        assert client_factory.call_args.kwargs["last_will_retain"] is True
        client.connect.assert_called_once()

    def test_native_transport_owns_runtime_reconnect(
        self,
        monkeypatch,
        mock_config,
        mock_segmentation,
        mock_wlan,
    ):
        """A runtime failure must remain owned by the ESP-IDF client task."""
        import mqtt.handler as handler_module
        import socket

        client = MagicMock()
        client_factory = MagicMock(return_value=client)
        monkeypatch.setattr(handler_module, "MQTTClient", client_factory)
        monkeypatch.setattr(
            socket,
            "getaddrinfo",
            lambda *_args, **_kwargs: [
                (socket.AF_INET, socket.SOCK_STREAM, 0, "", ("192.0.2.10", 1883))
            ],
        )

        handler = handler_module.MQTTHandler(mock_config, mock_segmentation, mock_wlan)
        handler._connect_client()
        client.status.return_value = handler_module.NATIVE_MQTT_FAILED
        assert handler._reconnect_if_due() is False

        client_factory.assert_called_once()
        client.connect.assert_called_once()

    def test_native_transport_finishes_setup_after_async_connect(
        self,
        monkeypatch,
        mock_config,
        mock_segmentation,
        mock_wlan,
    ):
        """Native session setup starts only after ESP-IDF reports connected."""
        import mqtt.handler as handler_module
        import socket

        client = MagicMock()
        client.status.return_value = handler_module.NATIVE_MQTT_CONNECTED
        monkeypatch.setattr(handler_module, "MQTTClient", MagicMock(return_value=client))
        monkeypatch.setattr(
            socket,
            "getaddrinfo",
            lambda *_args, **_kwargs: [
                (socket.AF_INET, socket.SOCK_STREAM, 0, "", ("192.0.2.10", 1883))
            ],
        )

        handler = handler_module.MQTTHandler(mock_config, mock_segmentation, mock_wlan)
        handler._connect_client()

        assert handler.connected is False
        assert handler._reconnect_if_due() is True
        assert handler.connected is True
        client.set_callback.assert_called_once()
        client.subscribe.assert_called_once_with(handler.cmd_topic)
    
    def test_publish_state_idle(self, mock_config, mock_segmentation, mock_wlan, mock_mqtt_client_instance, mock_global_state):
        """Test publishing idle state"""
        from mqtt.handler import MQTTHandler
        
        handler = MQTTHandler(mock_config, mock_segmentation, mock_wlan, mock_global_state)
        handler.client = mock_mqtt_client_instance
        handler.connected = True
        
        handler.publish_state(
            current_variance=0.5,
            current_state=0,  # IDLE
            current_threshold=1.0
        )
        
        # Verify publish was called
        mock_mqtt_client_instance.publish.assert_called_once()
        call_args = mock_mqtt_client_instance.publish.call_args
        topic = call_args[0][0]
        payload = json.loads(call_args[0][1])
        
        assert topic == "test/espectre/devices/test-device/telemetry"
        assert payload['protocol_version'] == '1.0'
        assert payload['device_id'] == 'test-device'
        assert payload['motion_state'] == 'idle'
        assert payload['movement_score'] == 0.5
        assert payload['threshold'] == 1.0
        assert payload['detector'] == 'lightweight'
        assert payload['health']['uptime_s'] >= 0
        assert 'packets_processed' not in payload
        assert 'packets_dropped' not in payload
        assert 'pps' not in payload
    
    def test_publish_state_motion(self, mock_config, mock_segmentation, mock_wlan, mock_mqtt_client_instance, mock_global_state):
        """Test publishing motion state"""
        from mqtt.handler import MQTTHandler
        
        handler = MQTTHandler(mock_config, mock_segmentation, mock_wlan, mock_global_state)
        handler.client = mock_mqtt_client_instance
        handler.connected = True
        
        handler.publish_state(
            current_variance=5.0,
            current_state=1,  # MOTION
            current_threshold=1.0
        )
        
        call_args = mock_mqtt_client_instance.publish.call_args
        payload = json.loads(call_args[0][1])
        
        assert payload['motion_state'] == 'motion'
        assert payload['movement_score'] == 5.0
        assert payload['threshold'] == 1.0
        assert payload['detector'] == 'lightweight'
   
    def test_publish_state_error_handling(self, mock_config, mock_segmentation, mock_wlan, mock_mqtt_client_instance):
        """Test error handling during publish"""
        from mqtt.handler import MQTTHandler
        
        handler = MQTTHandler(mock_config, mock_segmentation, mock_wlan)
        handler.client = mock_mqtt_client_instance
        handler.connected = True
        mock_mqtt_client_instance.publish.side_effect = Exception("Network error")
        
        # Should not raise exception
        handler.publish_state(
            current_variance=0.5,
            current_state=0,
            current_threshold=1.0
        )
    
    def test_check_messages(self, mock_config, mock_segmentation, mock_wlan, mock_mqtt_client_instance):
        """Test checking for incoming messages"""
        from mqtt.handler import MQTTHandler
        
        handler = MQTTHandler(mock_config, mock_segmentation, mock_wlan)
        handler.client = mock_mqtt_client_instance
        handler.connected = True

        handler.check_messages()
        
        mock_mqtt_client_instance.check_msg.assert_called_once()
    
    def test_check_messages_error_handling(self, mock_config, mock_segmentation, mock_wlan, mock_mqtt_client_instance):
        """Test error handling when checking messages"""
        from mqtt.handler import MQTTHandler
        
        handler = MQTTHandler(mock_config, mock_segmentation, mock_wlan)
        handler.client = mock_mqtt_client_instance
        handler.connected = True
        mock_mqtt_client_instance.check_msg.side_effect = Exception("Error")
        
        # Should not raise exception
        handler.check_messages()

        assert handler.connected is False
        assert handler._next_reconnect_ms == 0
        assert handler.client is mock_mqtt_client_instance

    def test_check_messages_resumes_after_native_reconnect(
        self,
        monkeypatch,
        mock_config,
        mock_segmentation,
        mock_wlan,
        mock_mqtt_client_instance,
    ):
        """Polling resumes after the ESP-IDF task reports its reconnect."""
        import mqtt.handler as handler_module

        handler = handler_module.MQTTHandler(mock_config, mock_segmentation, mock_wlan)
        handler.client = mock_mqtt_client_instance
        handler.connected = True
        mock_mqtt_client_instance.check_msg.side_effect = OSError("disconnected")
        handler.check_messages()

        mock_mqtt_client_instance.check_msg.side_effect = None
        mock_mqtt_client_instance.status.return_value = handler_module.NATIVE_MQTT_CONNECTED
        handler.check_messages()

        assert handler.connected is True
        mock_mqtt_client_instance.subscribe.assert_called_once_with(handler.cmd_topic)
        assert mock_mqtt_client_instance.check_msg.call_count == 2

    def test_failed_native_task_start_releases_client(
        self,
        monkeypatch,
        mock_config,
        mock_segmentation,
        mock_wlan,
    ):
        """A client whose ESP-IDF task cannot start must be released."""
        import mqtt.handler as handler_module
        import socket

        failed_client = MagicMock()
        failed_client.connect.side_effect = OSError("offline")
        client_factory = MagicMock(return_value=failed_client)
        monkeypatch.setattr(handler_module, "MQTTClient", client_factory)
        monkeypatch.setattr(
            socket,
            "getaddrinfo",
            lambda *_args, **_kwargs: [
                (socket.AF_INET, socket.SOCK_STREAM, 0, "", ("192.0.2.10", 1883))
            ],
        )
        handler = handler_module.MQTTHandler(mock_config, mock_segmentation, mock_wlan)
        handler._next_reconnect_ms = 0

        assert handler._reconnect_if_due() is False
        failed_client.deinit.assert_called_once()
        assert handler.client is None
        assert handler._next_reconnect_ms != 0
        client_factory.assert_called_once()

    def test_disconnect_deinitializes_native_client(
        self, mock_config, mock_segmentation, mock_wlan
    ):
        """Final shutdown must release the native task, event loop, and buffers."""
        from mqtt.handler import MQTTHandler

        client = MagicMock()
        handler = MQTTHandler(mock_config, mock_segmentation, mock_wlan)
        handler.client = client

        handler.disconnect()

        client.deinit.assert_called_once()
        client.disconnect.assert_not_called()
        assert handler.client is None
    
    def test_disconnect_error_handling(self, mock_config, mock_segmentation, mock_wlan, mock_mqtt_client_instance):
        """Test error handling during disconnect"""
        from mqtt.handler import MQTTHandler
        
        handler = MQTTHandler(mock_config, mock_segmentation, mock_wlan)
        handler.client = mock_mqtt_client_instance
        mock_mqtt_client_instance.deinit.side_effect = Exception("Error")
        
        # Should not raise exception
        handler.disconnect()
    
    def test_disconnect_no_client(self, mock_config, mock_segmentation, mock_wlan):
        """Test disconnect when client is None"""
        from mqtt.handler import MQTTHandler
        
        handler = MQTTHandler(mock_config, mock_segmentation, mock_wlan)
        handler.client = None
        
        # Should not raise exception
        handler.disconnect()

    @patch('mqtt.handler.MQTTClient')
    def test_ha_discovery_connect_and_birth_republish(
        self,
        mock_client_class,
        mock_config,
        mock_segmentation,
        mock_wlan,
        mock_global_state,
        mock_runtime_policy,
        mock_traffic_generator,
    ):
        """HA-enabled sessions should publish discovery and republish on HA birth."""
        from mqtt.handler import MQTTHandler

        mock_config.MQTT_HA_DISCOVERY_ENABLED = True
        client = MagicMock()
        client.connect = MagicMock()
        client.publish = MagicMock()
        client.subscribe = MagicMock()
        client.set_callback = MagicMock()
        client.set_last_will = MagicMock()
        mock_client_class.return_value = client
        client.status.return_value = 2

        handler = MQTTHandler(
            mock_config,
            mock_segmentation,
            mock_wlan,
            mock_global_state,
            runtime_policy=mock_runtime_policy,
            traffic_generator=mock_traffic_generator,
        )
        handler.connect()
        assert handler._reconnect_if_due() is True

        client.set_last_will.assert_called_once_with(
            "test/espectre/devices/test-device/ha/availability", "offline", retain=False
        )
        subscribed_topics = [call.args[0] for call in client.subscribe.call_args_list]
        assert handler.cmd_topic in subscribed_topics
        assert "homeassistant/status" in subscribed_topics
        discovery_calls = [
            call for call in client.publish.call_args_list if call.args[0].startswith("homeassistant/")
        ]
        assert len(discovery_calls) == 31
        assert all(call.kwargs.get("retain") is True for call in discovery_calls)
        discovery_topics = [call.args[0] for call in discovery_calls]
        assert "homeassistant/binary_sensor/micro_test_device_motion_detected/config" in discovery_topics
        assert "homeassistant/sensor/micro_test_device_movement_score/config" in discovery_topics
        intensity_topic = "homeassistant/sensor/micro_test_device_intensity/config"
        assert intensity_topic in discovery_topics
        intensity_payloads = [call.args[1] for call in discovery_calls if call.args[0] == intensity_topic]
        assert intensity_payloads == [""]
        retired_motion_topic = "homeassistant/binary_sensor/micro_test_device_motion/config"
        assert retired_motion_topic in discovery_topics
        retired_motion_payloads = [call.args[1] for call in discovery_calls if call.args[0] == retired_motion_topic]
        assert retired_motion_payloads == [""]
        assert "homeassistant/sensor/micro_test_device_traffic_tx_rate/config" in discovery_topics
        assert "homeassistant/sensor/micro_test_device_csi_missing_slot_rate/config" in discovery_topics
        assert "homeassistant/sensor/micro_test_device_csi_temporal_occupancy/config" in discovery_topics
        assert "homeassistant/button/micro_test_device_refresh_diagnostics/config" in discovery_topics
        assert "homeassistant/number/micro_test_device_threshold/config" in discovery_topics
        assert "homeassistant/number/micro_test_device_motion_on_hits/config" in discovery_topics
        assert "homeassistant/number/micro_test_device_motion_off_hits/config" in discovery_topics
        assert "homeassistant/switch/micro_test_device_trigger_calibration/config" in discovery_topics
        assert "homeassistant/select/micro_test_device_csi_traffic_ownership/config" in discovery_topics
        assert "homeassistant/select/micro_test_device_csi_traffic_source/config" in discovery_topics
        assert discovery_topics.index(
            "homeassistant/select/micro_test_device_csi_traffic_ownership/config"
        ) < discovery_topics.index(
            "homeassistant/select/micro_test_device_csi_traffic_source/config"
        )
        assert discovery_topics.index(
            "homeassistant/select/micro_test_device_csi_traffic_source/config"
        ) < discovery_topics.index("homeassistant/switch/micro_test_device_trigger_calibration/config")
        assert "test/espectre/devices/test-device/ha/threshold/set" in subscribed_topics
        assert "test/espectre/devices/test-device/ha/motion_on_hits/set" in subscribed_topics
        assert "test/espectre/devices/test-device/ha/motion_off_hits/set" in subscribed_topics
        assert "test/espectre/devices/test-device/ha/calibrate/set" in subscribed_topics
        assert "test/espectre/devices/test-device/ha/csi_traffic_mode/set" in subscribed_topics
        assert "test/espectre/devices/test-device/ha/traffic_generator_mode/set" in subscribed_topics
        assert "test/espectre/devices/test-device/ha/diagnostics/set" in subscribed_topics

        client.publish.reset_mock()
        handler._on_message(b"homeassistant/status", b"online")
        republished = [call.args[0] for call in client.publish.call_args_list]
        assert "homeassistant/binary_sensor/micro_test_device_motion_detected/config" in republished
        assert "homeassistant/sensor/micro_test_device_intensity/config" in republished
        assert "homeassistant/binary_sensor/micro_test_device_motion/config" in republished
        assert "homeassistant/sensor/micro_test_device_traffic_tx_rate/config" in republished
        assert "homeassistant/button/micro_test_device_refresh_diagnostics/config" in republished
        assert "homeassistant/number/micro_test_device_threshold/config" in republished
        assert "homeassistant/number/micro_test_device_motion_on_hits/config" in republished
        assert "homeassistant/number/micro_test_device_motion_off_hits/config" in republished
        assert "homeassistant/switch/micro_test_device_trigger_calibration/config" in republished
        assert "homeassistant/select/micro_test_device_csi_traffic_ownership/config" in republished
        assert "homeassistant/select/micro_test_device_csi_traffic_source/config" in republished
        assert "test/espectre/devices/test-device/ha/availability" in republished
        assert "test/espectre/devices/test-device/ha/movement/state" in republished
        assert "test/espectre/devices/test-device/ha/intensity/state" not in republished
        assert "test/espectre/devices/test-device/ha/threshold/state" in republished
        assert "test/espectre/devices/test-device/ha/motion_on_hits/state" in republished
        assert "test/espectre/devices/test-device/ha/motion_off_hits/state" in republished
        assert "test/espectre/devices/test-device/ha/calibrate/state" in republished
        assert "test/espectre/devices/test-device/ha/csi_traffic_mode/state" in republished
        assert "test/espectre/devices/test-device/ha/traffic_generator_mode/state" in republished
        assert "test/espectre/devices/test-device/ha/traffic_tx_rate/state" not in republished

    def test_publish_state_mirrors_ha_topics_when_enabled(
        self,
        mock_config,
        mock_segmentation,
        mock_wlan,
        mock_mqtt_client_instance,
        mock_global_state,
    ):
        """HA-enabled telemetry should mirror movement, not motion edges."""
        from mqtt.handler import MQTTHandler

        mock_config.MQTT_HA_DISCOVERY_ENABLED = True
        handler = MQTTHandler(mock_config, mock_segmentation, mock_wlan, mock_global_state)
        handler.client = mock_mqtt_client_instance
        handler.connected = True

        handler.publish_state(
            current_variance=0.75,
            current_state=1,
            current_threshold=1.0
        )

        published_topics = [call.args[0] for call in mock_mqtt_client_instance.publish.call_args_list]
        assert handler.telemetry_topic in published_topics
        assert "test/espectre/devices/test-device/ha/movement/state" in published_topics
        assert "test/espectre/devices/test-device/ha/motion/state" not in published_topics
        assert "test/espectre/devices/test-device/ha/intensity/state" not in published_topics
        assert "test/espectre/devices/test-device/ha/threshold/state" not in published_topics
        assert "test/espectre/devices/test-device/ha/calibrate/state" not in published_topics

    def test_publish_live_ha_mirrors_telemetry_movement_and_motion_edges(
        self,
        mock_config,
        mock_segmentation,
        mock_wlan,
        mock_mqtt_client_instance,
        mock_global_state,
    ):
        """Live publishes should update telemetry and movement every evaluation and motion only on edges."""
        from mqtt.handler import MQTTHandler

        mock_config.MQTT_HA_DISCOVERY_ENABLED = True
        handler = MQTTHandler(mock_config, mock_segmentation, mock_wlan, mock_global_state)
        handler.client = mock_mqtt_client_instance
        handler.connected = True

        handler.publish_live_ha(0.75, 1, 1.0)
        first_topics = [call.args[0] for call in mock_mqtt_client_instance.publish.call_args_list]
        first_payloads = {
            call.args[0]: call.args[1] for call in mock_mqtt_client_instance.publish.call_args_list
        }
        assert handler.telemetry_topic in first_topics
        assert "test/espectre/devices/test-device/ha/movement/state" in first_topics
        assert first_payloads["test/espectre/devices/test-device/ha/movement/state"] == "0.7500"
        assert "test/espectre/devices/test-device/ha/intensity/state" not in first_topics
        assert "test/espectre/devices/test-device/ha/threshold/state" in first_topics
        assert first_payloads["test/espectre/devices/test-device/ha/threshold/state"] == "1.0000"
        assert "test/espectre/devices/test-device/ha/motion/state" in first_topics
        assert first_payloads["test/espectre/devices/test-device/ha/motion/state"] == "ON"
        assert "test/espectre/devices/test-device/ha/calibrate/state" not in first_topics

        mock_mqtt_client_instance.publish.reset_mock()
        handler.publish_live_ha(1.0, 1, 1.0)
        second_payloads = {
            call.args[0]: call.args[1] for call in mock_mqtt_client_instance.publish.call_args_list
        }
        second_topics = [call.args[0] for call in mock_mqtt_client_instance.publish.call_args_list]
        assert handler.telemetry_topic in second_topics
        assert second_payloads["test/espectre/devices/test-device/ha/movement/state"] == "1.0000"
        assert "test/espectre/devices/test-device/ha/motion/state" not in second_topics
        assert "test/espectre/devices/test-device/ha/threshold/state" not in second_topics
        assert "test/espectre/devices/test-device/ha/intensity/state" not in second_topics

        mock_mqtt_client_instance.publish.reset_mock()
        handler.publish_live_ha(0.1, 0, 1.0)
        third_payloads = {
            call.args[0]: call.args[1] for call in mock_mqtt_client_instance.publish.call_args_list
        }
        assert third_payloads["test/espectre/devices/test-device/ha/movement/state"] == "0.1000"
        assert third_payloads["test/espectre/devices/test-device/ha/motion/state"] == "OFF"

        mock_mqtt_client_instance.publish.reset_mock()
        handler.publish_live_ha(0.1, 0, 0.4)
        fourth_payloads = {
            call.args[0]: call.args[1] for call in mock_mqtt_client_instance.publish.call_args_list
        }
        assert fourth_payloads["test/espectre/devices/test-device/ha/threshold/state"] == "0.4000"

    def test_ha_threshold_command_updates_detector(
        self,
        mock_config,
        mock_segmentation,
        mock_wlan,
        mock_mqtt_client_instance,
        mock_global_state,
    ):
        """HA number commands should write the live detector threshold."""
        from mqtt.handler import MQTTHandler

        mock_config.MQTT_HA_DISCOVERY_ENABLED = True
        handler = MQTTHandler(mock_config, mock_segmentation, mock_wlan, mock_global_state)
        handler.client = mock_mqtt_client_instance
        handler.connected = True

        handler._on_message(
            b"test/espectre/devices/test-device/ha/threshold/set",
            b"0.45",
        )

        assert mock_segmentation.get_threshold() == 0.45
        published = {
            call.args[0]: call.args[1] for call in mock_mqtt_client_instance.publish.call_args_list
        }
        assert published["test/espectre/devices/test-device/ha/threshold/state"] == "0.4500"
        config_payload = json.loads(published["test/espectre/devices/test-device/config"])
        assert config_payload["runtime"]["threshold"] == 0.45

    def test_ha_diagnostics_command_publishes_cached_sample(
        self,
        mock_config,
        mock_segmentation,
        mock_wlan,
        mock_mqtt_client_instance,
        mock_global_state,
    ):
        """HA Refresh Diagnostics should publish the cached CSI/Wi-Fi sample."""
        from mqtt.handler import MQTTHandler

        mock_config.MQTT_HA_DISCOVERY_ENABLED = True
        mock_global_state.latest_diagnostics = {
            "traffic_tx_pps": 100.0,
            "csi_callback_pps": 96.0,
            "csi_accepted_pps": 90.0,
            "csi_admitted_pps": 84.0,
            "csi_filtered_pps": 6.0,
            "csi_missing_slots_pps": 10.0,
            "csi_excess_pps": 6.0,
            "csi_stale_pps": 0.0,
            "csi_out_of_order_pps": 0.0,
            "csi_occupancy": 0.84,
            "wifi_channel": 10,
            "wifi_rssi_dbm": -55,
        }
        handler = MQTTHandler(mock_config, mock_segmentation, mock_wlan, mock_global_state)
        handler.client = mock_mqtt_client_instance
        handler.connected = True

        handler._on_message(
            b"test/espectre/devices/test-device/ha/diagnostics/set",
            b"PRESS",
        )

        published = {
            call.args[0]: call.args[1] for call in mock_mqtt_client_instance.publish.call_args_list
        }
        assert published["test/espectre/devices/test-device/ha/traffic_tx_rate/state"] == "100.0"
        assert published["test/espectre/devices/test-device/ha/csi_callback_rate/state"] == "96.0"
        assert published["test/espectre/devices/test-device/ha/csi_occupancy/state"] == "84.0"
        assert published["test/espectre/devices/test-device/ha/wifi_channel/state"] == "10"
        assert published["test/espectre/devices/test-device/ha/wifi_rssi/state"] == "-55"

    def test_ha_motion_hits_commands_update_runtime_policy(
        self,
        mock_config,
        mock_segmentation,
        mock_wlan,
        mock_mqtt_client_instance,
        mock_global_state,
        mock_runtime_policy,
    ):
        """HA number commands should update the session-only motion-hit policy."""
        from mqtt.handler import MQTTHandler

        mock_config.MQTT_HA_DISCOVERY_ENABLED = True
        handler = MQTTHandler(
            mock_config,
            mock_segmentation,
            mock_wlan,
            mock_global_state,
            runtime_policy=mock_runtime_policy,
        )
        handler.client = mock_mqtt_client_instance
        handler.connected = True

        handler._on_message(
            b"test/espectre/devices/test-device/ha/motion_on_hits/set",
            b"6",
        )
        handler._on_message(
            b"test/espectre/devices/test-device/ha/motion_off_hits/set",
            b"4",
        )

        assert mock_runtime_policy.motion_on_hits == 6
        assert mock_runtime_policy.motion_off_hits == 4
        published = {
            call.args[0]: call.args[1] for call in mock_mqtt_client_instance.publish.call_args_list
        }
        assert published["test/espectre/devices/test-device/ha/motion_on_hits/state"] == "6"
        assert published["test/espectre/devices/test-device/ha/motion_off_hits/state"] == "4"
        config_payload = json.loads(published["test/espectre/devices/test-device/config"])
        assert config_payload["runtime"]["motion_on_hits"] == 6
        assert config_payload["runtime"]["motion_off_hits"] == 4
        motion_on_publishes = [
            call for call in mock_mqtt_client_instance.publish.call_args_list
            if call.args[0] == "test/espectre/devices/test-device/ha/motion_on_hits/state"
        ]
        motion_off_publishes = [
            call for call in mock_mqtt_client_instance.publish.call_args_list
            if call.args[0] == "test/espectre/devices/test-device/ha/motion_off_hits/state"
        ]
        assert len(motion_on_publishes) == 2
        assert len(motion_off_publishes) == 2

    def test_mqtt_set_motion_hits_command_updates_runtime_policy(
        self,
        mock_config,
        mock_segmentation,
        mock_wlan,
        mock_mqtt_client_instance,
        mock_global_state,
        mock_runtime_policy,
        mock_traffic_generator,
    ):
        """Canonical MQTT set_motion_hits should update the session-only policy."""
        from mqtt.commands import MQTTCommands
        from mqtt.handler import MQTTHandler

        mock_config.MQTT_HA_DISCOVERY_ENABLED = True
        handler = MQTTHandler(
            mock_config,
            mock_segmentation,
            mock_wlan,
            mock_global_state,
            runtime_policy=mock_runtime_policy,
            traffic_generator=mock_traffic_generator,
        )
        handler.client = mock_mqtt_client_instance
        handler.connected = True
        handler.cmd_handler = MQTTCommands(
            mock_mqtt_client_instance,
            mock_config,
            mock_segmentation,
            handler.result_topic,
            handler.info_topic,
            mock_wlan,
            mock_global_state,
            runtime_policy=mock_runtime_policy,
            ha_adapter=handler.ha_adapter,
            recalibrate_callback=handler.request_recalibration,
            traffic_control_callback=handler.set_traffic_control,
            traffic_control_supported=True,
        )

        handler._on_message(
            b"test/espectre/devices/test-device/commands/request",
            b'{"command_id":"motion-1","command":"set_motion_hits","motion_on_hits":5,"motion_off_hits":4}',
        )

        assert mock_runtime_policy.motion_on_hits == 5
        assert mock_runtime_policy.motion_off_hits == 4
        published = {
            call.args[0]: call.args[1] for call in mock_mqtt_client_instance.publish.call_args_list
        }
        assert published["test/espectre/devices/test-device/commands/result"]
        config_payload = json.loads(published["test/espectre/devices/test-device/config"])
        assert config_payload["runtime"]["motion_on_hits"] == 5
        assert published["test/espectre/devices/test-device/ha/motion_on_hits/state"] == "5"
        assert published["test/espectre/devices/test-device/ha/motion_off_hits/state"] == "4"

    def test_ha_traffic_control_commands_update_runtime_generator(
        self,
        mock_config,
        mock_segmentation,
        mock_wlan,
        mock_mqtt_client_instance,
        mock_global_state,
        mock_traffic_generator,
    ):
        """HA select commands should update Micro traffic ownership and generator mode."""
        from mqtt.handler import MQTTHandler

        mock_config.MQTT_HA_DISCOVERY_ENABLED = True
        handler = MQTTHandler(
            mock_config,
            mock_segmentation,
            mock_wlan,
            mock_global_state,
            traffic_generator=mock_traffic_generator,
        )
        handler.client = mock_mqtt_client_instance
        handler.connected = True

        handler._on_message(
            b"test/espectre/devices/test-device/ha/traffic_generator_mode/set",
            b"dns",
        )
        handler._on_message(
            b"test/espectre/devices/test-device/ha/csi_traffic_mode/set",
            b"external",
        )

        assert mock_traffic_generator.set_mode_calls[0] == "dns"
        assert mock_traffic_generator.stop_calls >= 1
        assert handler.csi_traffic_mode == "external"
        assert handler.traffic_generator_mode == "dns"
        published = {
            call.args[0]: call.args[1] for call in mock_mqtt_client_instance.publish.call_args_list
        }
        assert published["test/espectre/devices/test-device/ha/csi_traffic_mode/state"] == "external"
        assert published["test/espectre/devices/test-device/ha/traffic_generator_mode/state"] == "dns"
        config_payload = json.loads(published["test/espectre/devices/test-device/config"])
        assert config_payload["runtime"]["csi_traffic_mode"] == "external"
        assert config_payload["runtime"]["traffic_generator_mode"] == "dns"
        csi_mode_publishes = [
            call for call in mock_mqtt_client_instance.publish.call_args_list
            if call.args[0] == "test/espectre/devices/test-device/ha/csi_traffic_mode/state"
        ]
        generator_mode_publishes = [
            call for call in mock_mqtt_client_instance.publish.call_args_list
            if call.args[0] == "test/espectre/devices/test-device/ha/traffic_generator_mode/state"
        ]
        assert len(csi_mode_publishes) == 2
        assert len(generator_mode_publishes) == 2

        mock_mqtt_client_instance.publish.reset_mock()
        handler._on_message(
            b"test/espectre/devices/test-device/ha/csi_traffic_mode/set",
            b"pacing",
        )
        published = [call.args[0] for call in mock_mqtt_client_instance.publish.call_args_list]
        assert "test/espectre/devices/test-device/ha/csi_traffic_mode/state" not in published
        assert handler.csi_traffic_mode == "external"

    def test_mqtt_traffic_control_commands_update_runtime_generator(
        self,
        mock_config,
        mock_segmentation,
        mock_wlan,
        mock_mqtt_client_instance,
        mock_global_state,
        mock_traffic_generator,
    ):
        """Canonical MQTT traffic commands should reconfigure Micro session traffic."""
        from mqtt.commands import MQTTCommands
        from mqtt.handler import MQTTHandler

        mock_config.MQTT_HA_DISCOVERY_ENABLED = True
        handler = MQTTHandler(
            mock_config,
            mock_segmentation,
            mock_wlan,
            mock_global_state,
            traffic_generator=mock_traffic_generator,
        )
        handler.client = mock_mqtt_client_instance
        handler.connected = True
        handler.cmd_handler = MQTTCommands(
            mock_mqtt_client_instance,
            mock_config,
            mock_segmentation,
            handler.result_topic,
            handler.info_topic,
            mock_wlan,
            mock_global_state,
            ha_adapter=handler.ha_adapter,
            recalibrate_callback=handler.request_recalibration,
            traffic_control_callback=handler.set_traffic_control,
            traffic_control_supported=True,
        )

        handler._on_message(
            b"test/espectre/devices/test-device/commands/request",
            b'{"command_id":"traffic-1","command":"set_csi_traffic_mode","csi_traffic_mode":"disabled"}',
        )
        handler._on_message(
            b"test/espectre/devices/test-device/commands/request",
            b'{"command_id":"traffic-2","command":"set_traffic_generator_mode","traffic_generator_mode":"dns"}',
        )

        published = {
            call.args[0]: call.args[1] for call in mock_mqtt_client_instance.publish.call_args_list
        }
        assert published["test/espectre/devices/test-device/commands/result"]
        config_payload = json.loads(published["test/espectre/devices/test-device/config"])
        assert config_payload["runtime"]["csi_traffic_mode"] == "internal"
        assert config_payload["runtime"]["traffic_generator_mode"] == "dns"
        assert published["test/espectre/devices/test-device/ha/csi_traffic_mode/state"] == "internal"
        assert published["test/espectre/devices/test-device/ha/traffic_generator_mode/state"] == "dns"

        mock_mqtt_client_instance.publish.reset_mock()
        handler._on_message(
            b"test/espectre/devices/test-device/commands/request",
            b'{"command_id":"traffic-3","command":"set_csi_traffic_mode","csi_traffic_mode":"pacing"}',
        )
        published_topics = [call.args[0] for call in mock_mqtt_client_instance.publish.call_args_list]
        assert handler.result_topic in published_topics

    def test_ha_calibrate_command_requests_recalibration(
        self,
        mock_config,
        mock_segmentation,
        mock_wlan,
        mock_mqtt_client_instance,
        mock_global_state,
    ):
        """HA Calibrate ON should queue a main-loop recalibration request."""
        from mqtt.handler import MQTTHandler

        mock_config.MQTT_HA_DISCOVERY_ENABLED = True
        handler = MQTTHandler(mock_config, mock_segmentation, mock_wlan, mock_global_state)
        handler.client = mock_mqtt_client_instance
        handler.connected = True

        handler._on_message(
            b"test/espectre/devices/test-device/ha/calibrate/set",
            b"ON",
        )

        published = {
            call.args[0]: call.args[1] for call in mock_mqtt_client_instance.publish.call_args_list
        }
        assert published["test/espectre/devices/test-device/ha/calibrate/state"] == "ON"
        assert handler.take_recalibrate_request() is True
        assert handler.take_recalibrate_request() is False

        mock_mqtt_client_instance.publish.reset_mock()
        handler._on_message(
            b"test/espectre/devices/test-device/ha/calibrate/set",
            b"OFF",
        )
        published = {
            call.args[0]: call.args[1] for call in mock_mqtt_client_instance.publish.call_args_list
        }
        assert published["test/espectre/devices/test-device/ha/calibrate/state"] == "ON"
        assert handler.take_recalibrate_request() is False

        mock_mqtt_client_instance.publish.reset_mock()
        handler.finish_recalibration(0.1, 0.42)
        published = {
            call.args[0]: call.args[1] for call in mock_mqtt_client_instance.publish.call_args_list
        }
        assert published["test/espectre/devices/test-device/ha/calibrate/state"] == "OFF"

    def test_mqtt_recalibrate_command_rejects_when_request_is_pending(
        self,
        mock_config,
        mock_segmentation,
        mock_wlan,
        mock_mqtt_client_instance,
        mock_global_state,
    ):
        """Canonical MQTT recalibrate should reject when one request is already pending."""
        from mqtt.commands import MQTTCommands
        from mqtt.handler import MQTTHandler

        mock_config.MQTT_HA_DISCOVERY_ENABLED = True
        handler = MQTTHandler(mock_config, mock_segmentation, mock_wlan, mock_global_state)
        handler.client = mock_mqtt_client_instance
        handler.connected = True
        handler.cmd_handler = MQTTCommands(
            mock_mqtt_client_instance,
            mock_config,
            mock_segmentation,
            handler.result_topic,
            handler.info_topic,
            mock_wlan,
            mock_global_state,
            ha_adapter=handler.ha_adapter,
            recalibrate_callback=handler.request_recalibration,
        )

        handler._on_message(
            b"test/espectre/devices/test-device/commands/request",
            b'{"command_id":"recal-1","command":"recalibrate"}',
        )
        handler._on_message(
            b"test/espectre/devices/test-device/commands/request",
            b'{"command_id":"recal-2","command":"recalibrate"}',
        )

        accepted_payloads = [
            json.loads(call.args[1])
            for call in mock_mqtt_client_instance.publish.call_args_list
            if call.args[0] == handler.result_topic and json.loads(call.args[1]).get("accepted") is True
        ]
        rejected_payloads = [
            json.loads(call.args[1])
            for call in mock_mqtt_client_instance.publish.call_args_list
            if call.args[0] == handler.result_topic and json.loads(call.args[1]).get("accepted") is False
        ]
        calibrate_publishes = [
            call for call in mock_mqtt_client_instance.publish.call_args_list
            if call.args[0] == "test/espectre/devices/test-device/ha/calibrate/state"
        ]

        assert accepted_payloads[-1]["command_id"] == "recal-1"
        assert rejected_payloads[-1]["command_id"] == "recal-2"
        assert rejected_payloads[-1]["accepted"] is False
        assert handler.take_recalibrate_request() is True
        assert handler.take_recalibrate_request() is False
        assert len(calibrate_publishes) == 2

    def test_publish_info(self, mock_config, mock_segmentation, mock_wlan, mock_mqtt_client_instance):
        """Test publish_info delegates to cmd_handler"""
        from mqtt.handler import MQTTHandler
        
        handler = MQTTHandler(mock_config, mock_segmentation, mock_wlan)
        handler.client = mock_mqtt_client_instance
        handler.cmd_handler = MagicMock()
        
        handler.publish_info()
        
        handler.cmd_handler.cmd_info.assert_called_once()
    
    def test_publish_info_no_handler(self, mock_config, mock_segmentation, mock_wlan):
        """Test publish_info when cmd_handler is None"""
        from mqtt.handler import MQTTHandler
        
        handler = MQTTHandler(mock_config, mock_segmentation, mock_wlan)
        handler.cmd_handler = None
        
        # Should not raise exception
        handler.publish_info()
    
    def test_on_message_callback(self, mock_config, mock_segmentation, mock_wlan, mock_mqtt_client_instance):
        """Test _on_message callback processing"""
        from mqtt.handler import MQTTHandler
        
        handler = MQTTHandler(mock_config, mock_segmentation, mock_wlan)
        handler.client = mock_mqtt_client_instance
        handler.cmd_handler = MagicMock()
        
        # Simulate receiving a message on cmd topic
        topic = b'test/espectre/devices/test-device/commands/request'
        msg = b'{"command": "info"}'
        
        handler._on_message(topic, msg)
        
        handler.cmd_handler.process_command.assert_called_once_with(msg)
    
    def test_on_message_wrong_topic(self, mock_config, mock_segmentation, mock_wlan, mock_mqtt_client_instance):
        """Test _on_message ignores wrong topics"""
        from mqtt.handler import MQTTHandler
        
        handler = MQTTHandler(mock_config, mock_segmentation, mock_wlan)
        handler.client = mock_mqtt_client_instance
        handler.cmd_handler = MagicMock()
        
        # Simulate receiving a message on wrong topic
        topic = b'other/topic'
        msg = b'{"command": "info"}'
        
        handler._on_message(topic, msg)
        
        handler.cmd_handler.process_command.assert_not_called()
    
    def test_on_message_error_handling(self, mock_config, mock_segmentation, mock_wlan, mock_mqtt_client_instance):
        """Test _on_message error handling"""
        from mqtt.handler import MQTTHandler
        
        handler = MQTTHandler(mock_config, mock_segmentation, mock_wlan)
        handler.client = mock_mqtt_client_instance
        handler.cmd_handler = MagicMock()
        handler.cmd_handler.process_command.side_effect = Exception("Error")
        
        # Should not raise exception
        handler._on_message(b'test/espectre/devices/test-device/commands/request', b'{"command": "info"}')


class TestMQTTCommands:
    """Test MQTTCommands class"""
    
    @pytest.fixture
    def commands_instance(self, mock_mqtt_client_instance, mock_config, mock_segmentation, mock_wlan, mock_global_state):
        """Create MQTTCommands instance with all mocks"""
        from mqtt.commands import MQTTCommands
        
        return MQTTCommands(
            mock_mqtt_client_instance,
            mock_config,
            mock_segmentation,
            "test/espectre/devices/test-device/commands/result",
            "test/espectre/devices/test-device/info",
            mock_wlan,
            mock_global_state
        )
    
    def test_send_response_dict(self, commands_instance, mock_mqtt_client_instance):
        """Test sending dict response"""
        commands_instance.send_response({"status": "ok"})
        
        mock_mqtt_client_instance.publish.assert_called_once()
        call_args = mock_mqtt_client_instance.publish.call_args
        assert call_args[0][0] == "test/espectre/devices/test-device/commands/result"
        payload = json.loads(call_args[0][1])
        assert payload['status'] == 'ok'
        assert payload['device_id'] == 'test-device'
        assert payload['accepted'] is True
    
    def test_send_response_string(self, commands_instance, mock_mqtt_client_instance):
        """Test sending string response"""
        commands_instance.send_response("Success")
        
        mock_mqtt_client_instance.publish.assert_called_once()
        call_args = mock_mqtt_client_instance.publish.call_args
        payload = json.loads(call_args[0][1])
        assert payload['message'] == 'Success'
        assert payload['device_id'] == 'test-device'
        assert payload['accepted'] is True
    
    def test_send_response_json_string(self, commands_instance, mock_mqtt_client_instance):
        """Test sending already-valid JSON string"""
        commands_instance.send_response('{"already": "json"}')
        
        mock_mqtt_client_instance.publish.assert_called_once()
        call_args = mock_mqtt_client_instance.publish.call_args
        payload = json.loads(call_args[0][1])
        assert payload['already'] == 'json'
        assert payload['device_id'] == 'test-device'
        assert payload['accepted'] is True
    
    def test_send_response_error_handling(self, commands_instance, mock_mqtt_client_instance):
        """Test error handling when sending response"""
        mock_mqtt_client_instance.publish.side_effect = Exception("Error")
        
        # Should not raise exception
        commands_instance.send_response("test")
    
    def test_cmd_diagnostics(self, commands_instance, mock_mqtt_client_instance):
        """Test diagnostics query."""
        with patch('mqtt.commands.gc') as mock_gc:
            mock_gc.mem_free.return_value = 100000
            payload = commands_instance.cmd_diagnostics()
        
        assert 'uptime' in payload
        assert 'free_memory_kb' in payload
        assert 'loop_time_ms' in payload
        assert payload['traffic_tx_pps'] == 0.0
        assert payload['csi_callback_pps'] == 0.0
        assert payload['csi_accepted_pps'] == 0.0
        assert payload['csi_admitted_pps'] == 0.0
        assert payload['csi_filtered_pps'] == 0.0
        assert payload['csi_missing_slots_pps'] == 0.0
        assert payload['csi_excess_pps'] == 0.0
        assert payload['csi_stale_pps'] == 0.0
        assert payload['csi_out_of_order_pps'] == 0.0
        assert payload['csi_occupancy'] == 0.0
        assert payload['wifi_channel'] == 0
        assert payload['wifi_rssi_dbm'] is None
        assert isinstance(payload['uptime'], int)
        assert 'timestamp' not in payload
        assert 'state' not in payload
        assert 'movement' not in payload
        assert 'threshold' not in payload
        assert 'traffic_generator' not in payload

    def test_cmd_diagnostics_uses_cached_sample(self, commands_instance, mock_mqtt_client_instance, mock_global_state):
        """Test diagnostics query uses the cached CSI/Wi-Fi sample."""
        mock_global_state.current_channel = 10
        mock_global_state.latest_diagnostics = {
            "traffic_tx_pps": 100.0,
            "csi_callback_pps": 96.0,
            "csi_accepted_pps": 90.0,
            "csi_admitted_pps": 84.0,
            "csi_filtered_pps": 6.0,
            "csi_missing_slots_pps": 10.0,
            "csi_excess_pps": 6.0,
            "csi_stale_pps": 0.0,
            "csi_out_of_order_pps": 0.0,
            "csi_occupancy": 0.84,
            "wifi_channel": 10,
            "wifi_rssi_dbm": -55,
        }
        with patch('mqtt.commands.gc') as mock_gc:
            mock_gc.mem_free.return_value = 100000
            payload = commands_instance.cmd_diagnostics()
        assert payload['traffic_tx_pps'] == 100.0
        assert payload['csi_callback_pps'] == 96.0
        assert payload['csi_admitted_pps'] == 84.0
        assert payload['csi_occupancy'] == 0.84
        assert payload['wifi_channel'] == 10
        assert payload['wifi_rssi_dbm'] == -55

    def test_runtime_diagnostics_sampler_derives_five_second_rates(self):
        """Test Micro rate sampling matches the C++ five-second diagnostics contract."""
        from runtime_diagnostics import RuntimeDiagnosticsSampler, collect_runtime_diagnostics_snapshot

        class _Traffic:
            def __init__(self, count):
                self.count = count

            def get_packet_count(self):
                return self.count

        sampler = RuntimeDiagnosticsSampler()
        sampler.reset(
            collect_runtime_diagnostics_snapshot(
                traffic_generator=_Traffic(100),
                callback_total=100,
                accepted_total=90,
                admitted_total=80,
                filtered_total=10,
            ),
            1000,
        )
        sample = sampler.sample(
            collect_runtime_diagnostics_snapshot(
                traffic_generator=_Traffic(600),
                callback_total=580,
                accepted_total=540,
                admitted_total=505,
                filtered_total=40,
                missing_slots_total=25,
                excess_total=15,
                stale_total=5,
                out_of_order_total=10,
                occupancy_slots=82,
                window_slots=100,
                wifi_channel=10,
                rssi_dbm=-55,
            ),
            6000,
        )
        assert sample["traffic_tx_pps"] == 100.0
        assert sample["csi_callback_pps"] == 96.0
        assert sample["csi_accepted_pps"] == 90.0
        assert sample["csi_admitted_pps"] == 85.0
        assert sample["csi_filtered_pps"] == 6.0
        assert sample["csi_missing_slots_pps"] == 5.0
        assert sample["csi_excess_pps"] == 3.0
        assert sample["csi_stale_pps"] == 1.0
        assert sample["csi_out_of_order_pps"] == 2.0
        assert sample["csi_occupancy"] == 0.82
        assert sample["wifi_channel"] == 10
        assert sample["wifi_rssi_dbm"] == -55

    def test_runtime_diagnostics_reads_native_csi_ring_drops(self):
        from runtime_diagnostics import wifi_csi_dropped

        class _Wlan:
            @staticmethod
            def csi_dropped():
                return 37

        assert wifi_csi_dropped(_Wlan()) == 37
        assert wifi_csi_dropped(object()) == 0
    
    def test_cmd_set_threshold_success(self, commands_instance, mock_mqtt_client_instance, mock_segmentation):
        """Test setting detection threshold (session-only, not persisted)"""
        commands_instance.cmd_set_threshold({'threshold': 0.75})
        
        assert mock_segmentation.get_threshold() == 0.75
    
    def test_cmd_set_threshold_missing_threshold(self, commands_instance, mock_mqtt_client_instance):
        """Test threshold command without threshold"""
        commands_instance.cmd_set_threshold({})
        
        call_args = mock_mqtt_client_instance.publish.call_args
        payload = json.loads(call_args[0][1])
        assert 'ERROR' in payload['message']
        assert "Missing 'threshold'" in payload['message']
    
    def test_cmd_set_threshold_out_of_range(self, commands_instance, mock_mqtt_client_instance):
        """Test threshold command with out-of-range value"""
        commands_instance.cmd_set_threshold({'threshold': 100.0})
        
        call_args = mock_mqtt_client_instance.publish.call_args
        payload = json.loads(call_args[0][1])
        assert 'ERROR' in payload['message']

    def test_cmd_set_threshold_below_min(self, commands_instance, mock_mqtt_client_instance):
        """Test threshold command with value below minimum range."""
        commands_instance.cmd_set_threshold({'threshold': -0.1})

        call_args = mock_mqtt_client_instance.publish.call_args
        payload = json.loads(call_args[0][1])
        assert 'ERROR' in payload['message']
    
    def test_cmd_set_threshold_invalid_value(self, commands_instance, mock_mqtt_client_instance):
        """Test threshold command with invalid value"""
        commands_instance.cmd_set_threshold({'threshold': 'invalid'})
        
        call_args = mock_mqtt_client_instance.publish.call_args
        payload = json.loads(call_args[0][1])
        assert 'ERROR' in payload['message']
    
    def test_process_command_info(self, commands_instance, mock_mqtt_client_instance):
        """Test processing info command"""
        with patch.object(commands_instance, 'cmd_info') as mock_info:
            commands_instance.process_command(b'{"command": "info"}')
            mock_info.assert_called_once()
    
    def test_process_command_diagnostics(self, commands_instance, mock_mqtt_client_instance):
        """Test processing diagnostics command."""
        with patch.object(commands_instance, 'cmd_diagnostics', return_value={}) as mock_diagnostics:
            commands_instance.process_command(b'{"command": "diagnostics"}')
            mock_diagnostics.assert_called_once()
    
    def test_process_command_unknown(self, commands_instance, mock_mqtt_client_instance):
        """Test processing unknown command"""
        commands_instance.process_command(b'{"command": "unknown_cmd"}')
        
        call_args = mock_mqtt_client_instance.publish.call_args
        payload = json.loads(call_args[0][1])
        assert 'ERROR' in payload['message']
        assert 'Unknown command' in payload['message']

    def test_process_command_missing_cmd(self, commands_instance, mock_mqtt_client_instance):
        """Test processing command without cmd field"""
        commands_instance.process_command(b'{"value": 123}')
        
        call_args = mock_mqtt_client_instance.publish.call_args
        payload = json.loads(call_args[0][1])
        assert 'ERROR' in payload['message']
    
    def test_process_command_invalid_json(self, commands_instance, mock_mqtt_client_instance):
        """Test processing invalid JSON"""
        commands_instance.process_command(b'invalid json')
        
        call_args = mock_mqtt_client_instance.publish.call_args
        payload = json.loads(call_args[0][1])
        assert 'ERROR' in payload['message']
    
    def test_process_command_string_data(self, commands_instance, mock_mqtt_client_instance):
        """Test processing string data (not bytes)"""
        with patch.object(commands_instance, 'cmd_info') as mock_info:
            commands_instance.process_command('{"command": "info"}')
            mock_info.assert_called_once()
    
    def test_cmd_info(self, commands_instance, mock_mqtt_client_instance):
        """Test info command returns system information"""
        payload = commands_instance.cmd_info()
        assert payload["frontend"] == "micro"
        
        assert 'network' in payload
        assert 'detection' in payload
        assert payload['device_name'] == 'ESPectre C6 device'
        assert payload['device_label'] == ''
        assert not any(key.startswith("supports_") for key in payload)
        assert payload['csi_traffic_mode'] == 'internal'
        assert payload['traffic_mode'] == 'ping'
        assert payload['csi_target_pps'] == 100
        assert payload['evaluation_interval_ms'] == 250
        assert payload['publish_interval_ms'] == 1000
        assert 'device' not in payload
        assert 'mqtt' not in payload
        assert 'subcarriers' not in payload
        assert payload['detection']['algorithm'] == 'lightweight'

    def test_cmd_capabilities_returns_schema_catalog(self, commands_instance, mock_mqtt_client_instance):
        """Test capabilities lists only executable canonical commands."""
        payload = commands_instance.cmd_capabilities()
        names = [item["name"] for item in payload["commands"]]
        assert names == ["capabilities", "info", "status", "config", "diagnostics", "set_threshold"]
        assert len(json.dumps(payload, separators=(",", ":"))) < 4096
    
    def test_cmd_info_with_connected_wlan(self, mock_mqtt_client_instance, mock_config, mock_segmentation, mock_global_state):
        """Test info command with connected WLAN"""
        from mqtt.commands import MQTTCommands
        
        # Create mock WLAN that is active and connected
        mock_wlan = MagicMock()
        mock_wlan.active.return_value = True
        mock_wlan.isconnected.return_value = True
        mock_wlan.config.side_effect = lambda key: {
            'mac': b'\x12\x34\x56\x78\x9a\xbc',
            'channel': 6,
            'protocol': 7
        }.get(key, 0)
        mock_wlan.ifconfig.return_value = ('192.168.1.100', '255.255.255.0', '192.168.1.1', '8.8.8.8')
        
        commands = MQTTCommands(
            mock_mqtt_client_instance,
            mock_config,
            mock_segmentation,
            "test/espectre/devices/test-device/commands/result",
            "test/espectre/devices/test-device/info",
            mock_wlan,
            mock_global_state
        )
        
        payload = commands.cmd_info()
        
        assert payload['device_name'] == 'ESPectre C6 device'
        assert 'ip_address' not in payload['network']
        assert 'mac_address' not in payload['network']
        assert payload['network']['channel']['primary'] == 6
        assert payload['detection']['algorithm'] == 'lightweight'
    
    def test_cmd_info_with_inactive_wlan(self, mock_mqtt_client_instance, mock_config, mock_segmentation, mock_global_state):
        """Test info command with inactive WLAN"""
        from mqtt.commands import MQTTCommands
        
        mock_wlan = MagicMock()
        mock_wlan.active.return_value = False
        
        commands = MQTTCommands(
            mock_mqtt_client_instance,
            mock_config,
            mock_segmentation,
            "test/espectre/devices/test-device/commands/result",
            "test/espectre/devices/test-device/info",
            mock_wlan,
            mock_global_state
        )
        
        payload = commands.cmd_info()
        
        assert payload['device_name'] == 'ESPectre C6 device'
        assert 'ip_address' not in payload['network']
        assert 'mac_address' not in payload['network']
        assert payload['detection']['algorithm'] == 'lightweight'

    def test_cmd_info_uses_detector_algorithm_identifier(self, mock_mqtt_client_instance, mock_config, mock_wlan, mock_global_state):
        """Test info commands publish the canonical detector identifier."""
        from mqtt.commands import MQTTCommands

        class MockNamedDetector(MockDetector):
            ALGORITHM = "lightweight"

            def get_name(self):
                return "A custom display name"

        commands = MQTTCommands(
            mock_mqtt_client_instance,
            mock_config,
            MockNamedDetector(),
            "test/espectre/devices/test-device/commands/result",
            "test/espectre/devices/test-device/info",
            mock_wlan,
            mock_global_state
        )

        payload = commands.cmd_info()
        assert payload['detection']['algorithm'] == 'lightweight'
