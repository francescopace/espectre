# SPDX-License-Identifier: GPL-3.0-only
# Commercial licensing available under separate agreement; see LICENSING.md.
"""
ESPectre - MQTT Tests

Unit tests for MQTT communication helpers.

Author: Francesco Pace <francesco.pace@gmail.com>
"""

import pytest
import json
import sys
from unittest.mock import MagicMock, patch

# Mock MicroPython modules before importing mqtt modules
mock_mqtt_client = MagicMock()
sys.modules['umqtt'] = MagicMock()
sys.modules['umqtt.simple'] = MagicMock()
sys.modules['umqtt.simple'].MQTTClient = mock_mqtt_client

# Mock network module (MicroPython)
mock_network = MagicMock()
mock_network.MODE_11B = 1
mock_network.MODE_11G = 2
mock_network.MODE_11N = 4
mock_network.MODE_LR = 8
sys.modules['network'] = mock_network

# Mock _thread module (MicroPython)
sys.modules['_thread'] = MagicMock()


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


class MockConfig:
    """Mock configuration module"""
    MQTT_CLIENT_ID = "test-device"
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
    client.set_last_will = MagicMock()
    return client


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
        assert handler.accepted_topic == "test/espectre/devices/test-device/commands/accepted"
        assert handler.rejected_topic == "test/espectre/devices/test-device/commands/rejected"

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
        assert handler._next_reconnect_ms != 0

    def test_check_messages_reconnects_after_backoff(
        self,
        monkeypatch,
        mock_config,
        mock_segmentation,
        mock_wlan,
        mock_mqtt_client_instance,
    ):
        """A failed poll should reconnect, resubscribe, and resume polling."""
        import mqtt.handler as handler_module

        handler = handler_module.MQTTHandler(mock_config, mock_segmentation, mock_wlan)
        handler.client = mock_mqtt_client_instance
        handler.connected = True
        mock_mqtt_client_instance.check_msg.side_effect = OSError("disconnected")
        handler.check_messages()

        replacement = MagicMock()
        monkeypatch.setattr(handler_module, "MQTTClient", lambda *args, **kwargs: replacement)
        handler._next_reconnect_ms = 0
        handler.check_messages()

        assert handler.connected is True
        replacement.connect.assert_called_once()
        replacement.subscribe.assert_called_once_with(handler.cmd_topic)
        replacement.check_msg.assert_called_once()
    
    def test_disconnect(self, mock_config, mock_segmentation, mock_wlan, mock_mqtt_client_instance):
        """Test disconnecting from MQTT broker"""
        from mqtt.handler import MQTTHandler
        
        handler = MQTTHandler(mock_config, mock_segmentation, mock_wlan)
        handler.client = mock_mqtt_client_instance
        
        handler.disconnect()
        
        mock_mqtt_client_instance.disconnect.assert_called_once()
    
    def test_disconnect_error_handling(self, mock_config, mock_segmentation, mock_wlan, mock_mqtt_client_instance):
        """Test error handling during disconnect"""
        from mqtt.handler import MQTTHandler
        
        handler = MQTTHandler(mock_config, mock_segmentation, mock_wlan)
        handler.client = mock_mqtt_client_instance
        mock_mqtt_client_instance.disconnect.side_effect = Exception("Error")
        
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

        handler = MQTTHandler(mock_config, mock_segmentation, mock_wlan, mock_global_state)
        handler.connect()

        client.set_last_will.assert_called_once_with(
            "test/espectre/devices/test-device/ha/availability", "offline", retain=False
        )
        subscribed_topics = [call.args[0] for call in client.subscribe.call_args_list]
        assert handler.cmd_topic in subscribed_topics
        assert "homeassistant/status" in subscribed_topics
        discovery_calls = [
            call for call in client.publish.call_args_list if call.args[0].startswith("homeassistant/")
        ]
        assert len(discovery_calls) == 2
        assert all(call.kwargs.get("retain") is True for call in discovery_calls)

        client.publish.reset_mock()
        handler._on_message(b"homeassistant/status", b"online")
        republished = [call.args[0] for call in client.publish.call_args_list]
        assert "homeassistant/binary_sensor/micro_test_device_motion/config" in republished
        assert "test/espectre/devices/test-device/ha/availability" in republished

    def test_publish_state_mirrors_ha_topics_when_enabled(
        self,
        mock_config,
        mock_segmentation,
        mock_wlan,
        mock_mqtt_client_instance,
        mock_global_state,
    ):
        """HA-enabled state publishes should mirror simple HA topics."""
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
        assert "test/espectre/devices/test-device/ha/motion/state" in published_topics
        assert "test/espectre/devices/test-device/ha/movement/state" in published_topics
    
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
            "test/espectre/devices/test-device/commands/accepted",
            "test/espectre/devices/test-device/commands/rejected",
            "test/espectre/devices/test-device/info",
            "test/espectre/devices/test-device/stats",
            mock_wlan,
            mock_global_state
        )
    
    def test_send_response_dict(self, commands_instance, mock_mqtt_client_instance):
        """Test sending dict response"""
        commands_instance.send_response({"status": "ok"})
        
        mock_mqtt_client_instance.publish.assert_called_once()
        call_args = mock_mqtt_client_instance.publish.call_args
        assert call_args[0][0] == "test/espectre/devices/test-device/commands/accepted"
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
    
    def test_cmd_stats(self, commands_instance, mock_mqtt_client_instance):
        """Test stats command"""
        with patch('mqtt.commands.gc') as mock_gc:
            mock_gc.mem_free.return_value = 100000
            
            commands_instance.cmd_stats()
        
        mock_mqtt_client_instance.publish.assert_called_once()
        call_args = mock_mqtt_client_instance.publish.call_args
        payload = json.loads(call_args[0][1])
        
        assert 'uptime' in payload
        assert 'free_memory_kb' in payload
        assert 'loop_time_ms' in payload
        assert isinstance(payload['uptime'], int)
        assert 'timestamp' not in payload
        assert 'state' not in payload
        assert 'movement' not in payload
        assert 'threshold' not in payload
        assert 'traffic_generator' not in payload
    
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
    
    def test_process_command_stats(self, commands_instance, mock_mqtt_client_instance):
        """Test processing stats command"""
        with patch.object(commands_instance, 'cmd_stats') as mock_stats:
            commands_instance.process_command(b'{"command": "stats"}')
            mock_stats.assert_called_once()
    
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
        commands_instance.cmd_info()
        
        mock_mqtt_client_instance.publish.assert_called_once()
        call_args = mock_mqtt_client_instance.publish.call_args
        payload = json.loads(call_args[0][1])
        
        assert 'network' in payload
        assert 'detection' in payload
        assert payload['device_name'] == 'ESPectre C6 device'
        assert payload['device_label'] == ''
        assert payload['supports_info'] is True
        assert payload['supports_stats'] is True
        assert payload['supports_runtime_threshold'] is True
        assert payload['supports_runtime_detector'] is False
        assert payload['supports_ota'] is False
        assert 'device' not in payload
        assert 'mqtt' not in payload
        assert 'subcarriers' not in payload
        assert payload['detection']['algorithm'] == 'lightweight'
    
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
            "test/espectre/devices/test-device/commands/accepted",
            "test/espectre/devices/test-device/commands/rejected",
            "test/espectre/devices/test-device/info",
            "test/espectre/devices/test-device/stats",
            mock_wlan,
            mock_global_state
        )
        
        commands.cmd_info()
        
        call_args = mock_mqtt_client_instance.publish.call_args
        payload = json.loads(call_args[0][1])
        
        assert payload['device_name'] == 'ESPectre C6 device'
        assert payload['network']['ip_address'] == '192.168.1.100'
        assert payload['network']['mac_address'] == '12:34:56:78:9A:BC'
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
            "test/espectre/devices/test-device/commands/accepted",
            "test/espectre/devices/test-device/commands/rejected",
            "test/espectre/devices/test-device/info",
            "test/espectre/devices/test-device/stats",
            mock_wlan,
            mock_global_state
        )
        
        commands.cmd_info()
        
        call_args = mock_mqtt_client_instance.publish.call_args
        payload = json.loads(call_args[0][1])
        
        assert payload['device_name'] == 'ESPectre C6 device'
        assert payload['network']['ip_address'] == ''
        assert payload['network']['mac_address'] == ''
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
            "test/espectre/devices/test-device/commands/accepted",
            "test/espectre/devices/test-device/commands/rejected",
            "test/espectre/devices/test-device/info",
            "test/espectre/devices/test-device/stats",
            mock_wlan,
            mock_global_state
        )

        commands.cmd_info()

        call_args = mock_mqtt_client_instance.publish.call_args
        payload = json.loads(call_args[0][1])
        assert payload['detection']['algorithm'] == 'lightweight'
