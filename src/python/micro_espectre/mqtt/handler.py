"""
Micro-ESPectre - MQTT Handler Module

Handles MQTT communication and command processing.
Manages connection, publishing state updates, and processing remote commands.

Author: Francesco Pace <francesco.pace@gmail.com>
License: GPLv3
"""
import json
import time
from umqtt.simple import MQTTClient

try:
    from src.mqtt.commands import MQTTCommands
    from src.mqtt.home_assistant import HomeAssistantMqttAdapter
except ImportError:
    from mqtt.commands import MQTTCommands
    from mqtt.home_assistant import HomeAssistantMqttAdapter

MQTT_RECONNECT_INITIAL_MS = 1000
MQTT_RECONNECT_MAX_MS = 60000


def _ticks_ms():
    ticks_fn = getattr(time, "ticks_ms", None)
    if ticks_fn is not None:
        return ticks_fn()
    return int(time.time() * 1000)


def _ticks_diff(new, old):
    diff_fn = getattr(time, "ticks_diff", None)
    return diff_fn(new, old) if diff_fn is not None else new - old


def _ticks_add(value, delta):
    add_fn = getattr(time, "ticks_add", None)
    return add_fn(value, delta) if add_fn is not None else value + delta


class MQTTHandler:
    """MQTT handler with publishing and command support"""
    
    def __init__(self, config, detector, wlan, global_state=None):
        """
        Initialize MQTT handler
        
        Args:
            config: Configuration module
            detector: IDetector instance
            wlan: WLAN instance
            global_state: GlobalState instance for accessing loop metrics (optional)
        """
        self.config = config
        self.detector = detector
        self.wlan = wlan
        self.global_state = global_state
        self.client = None
        self.cmd_handler = None
        self.ha_adapter = HomeAssistantMqttAdapter(config, detector, wlan, global_state)
        self.connected = False
        self._stopping = False
        self._next_reconnect_ms = 0
        self._reconnect_backoff_ms = MQTT_RECONNECT_INITIAL_MS
        self.start_time = time.time()
        
        # ESPectre Protocol topics
        topic_prefix = config.MQTT_TOPIC_PREFIX.rstrip('/')
        self.device_id = config.MQTT_CLIENT_ID
        self.base_topic = f"{topic_prefix}/{self.device_id}"
        self.telemetry_topic = f"{self.base_topic}/telemetry"
        self.status_topic = f"{self.base_topic}/status"
        self.info_topic = f"{self.base_topic}/info"
        self.stats_topic = f"{self.base_topic}/stats"
        self.cmd_topic = f"{self.base_topic}/commands/request"
        self.accepted_topic = f"{self.base_topic}/commands/accepted"
        self.rejected_topic = f"{self.base_topic}/commands/rejected"
        
        # Publishing state
        self.last_variance = 0.0
        self.last_state = 0  # STATE_IDLE
        
    def connect(self):
        """Connect to MQTT broker"""
        self._stopping = False
        try:
            self._connect_client()
            return self.client
        except Exception as e:
            print(f"MQTT connection failed: {e}")
            self._mark_disconnected()
            return None

    def _connect_client(self):
        """Create, connect, subscribe, and announce one MQTT client."""
        self.client = MQTTClient(
            self.config.MQTT_CLIENT_ID,
            self.config.MQTT_BROKER,
            port=self.config.MQTT_PORT,
            user=self.config.MQTT_USERNAME,
            password=self.config.MQTT_PASSWORD
        )
        self.ha_adapter.configure_client(self.client)
        
        print('Connecting to MQTT broker...')
        self.client.connect()
        print('MQTT connected')
        
        # Initialize command handler
        self.cmd_handler = MQTTCommands(
            self.client,
            self.config,
            self.detector,
            self.accepted_topic,
            self.rejected_topic,
            self.info_topic,
            self.stats_topic,
            self.wlan,
            self.global_state
        )
        
        # Set callback for incoming messages
        self.client.set_callback(self._on_message)
        
        # Subscribe to command topic
        self.client.subscribe(self.cmd_topic)
        self.ha_adapter.subscribe_topics(self.client)
        self.connected = True
        self._next_reconnect_ms = 0
        self._reconnect_backoff_ms = MQTT_RECONNECT_INITIAL_MS
        self.publish_status(True)
        self.publish_info()
        self.ha_adapter.publish_discovery(self.client)
        self.ha_adapter.publish_state(self.client, self.last_variance, self.last_state)
        if not self.connected:
            raise OSError("MQTT connection lost during session setup")

    def _mark_disconnected(self):
        """Schedule a non-blocking reconnect after a transport failure."""
        if not self.connected and self._next_reconnect_ms:
            return
        self.connected = False
        self._next_reconnect_ms = _ticks_add(
            _ticks_ms(), self._reconnect_backoff_ms
        )
        self._reconnect_backoff_ms = min(
            self._reconnect_backoff_ms * 2,
            MQTT_RECONNECT_MAX_MS,
        )
        if self.client:
            try:
                self.client.disconnect()
            except Exception:
                pass

    def _reconnect_if_due(self):
        """Attempt one reconnect when the backoff deadline has elapsed."""
        if self._stopping or self.connected:
            return self.connected
        now = _ticks_ms()
        if self._next_reconnect_ms and _ticks_diff(now, self._next_reconnect_ms) < 0:
            return False
        try:
            print("Reconnecting to MQTT broker...")
            self._connect_client()
            print("MQTT reconnected")
            return True
        except Exception as e:
            print(f"MQTT reconnect failed: {e}")
            self._next_reconnect_ms = _ticks_add(now, self._reconnect_backoff_ms)
            self._reconnect_backoff_ms = min(
                self._reconnect_backoff_ms * 2,
                MQTT_RECONNECT_MAX_MS,
            )
            return False
    
    def _on_message(self, topic, msg):
        """Callback for incoming MQTT messages"""
        try:
            topic_str = topic.decode('utf-8') if isinstance(topic, bytes) else topic
            
            if topic_str == self.cmd_topic:
                # Process command
                self.cmd_handler.process_command(msg)
            else:
                self.ha_adapter.handle_message(self.client, topic_str, msg)
            
        except Exception as e:
            print(f"Error processing MQTT message: {e}")
    
    def check_messages(self):
        """Check for incoming MQTT messages (non-blocking)"""
        if not self.connected and not self._reconnect_if_due():
            return
        try:
            self.client.check_msg()
        except Exception as e:
            print(f"Error checking MQTT messages: {e}")
            self._mark_disconnected()
    
    def publish_state(self, current_variance, current_state, current_threshold):
        """
        Publish current state to MQTT
        
        Args:
            current_variance: Current motion metric (probability for Classic/ML)
            current_state: Current state (0=IDLE, 1=MOTION)
            current_threshold: Current threshold
        """
        if not self.connected:
            self.ha_adapter.record_state(current_variance, current_state)
            self.last_variance = current_variance
            self.last_state = current_state
            return

        state_str = 'motion' if current_state == 1 else 'idle'
        timestamp_ms = int(time.time() * 1000)
        health = {
            'uptime_s': int(time.time() - self.start_time),
        }
        
        payload = {
            'protocol_version': '1.0',
            'device_id': self.device_id,
            'frontend': 'micro',
            'timestamp_ms': timestamp_ms,
            'motion_state': state_str,
            'movement_score': round(current_variance, 4),
            'threshold': round(current_threshold, 4),
            'detector': self.detector.get_name(),
            'health': health
        }
        
        try:
            self.client.publish(self.telemetry_topic, json.dumps(payload))
            self.ha_adapter.publish_state(self.client, current_variance, current_state)
        except Exception as e:
            print(f"Error publishing to MQTT: {e}")
            self._mark_disconnected()
        
        # Update state
        self.last_variance = current_variance
        self.last_state = current_state
    
    def disconnect(self):
        """Disconnect from MQTT broker"""
        self._stopping = True
        if self.client:
            try:
                self.publish_status(False)
                self.client.disconnect()
                print('MQTT disconnected')
            except Exception as e:
                print(f"Error disconnecting MQTT: {e}")
        self.connected = False
    
    def publish_info(self):
        """Publish system info"""
        if self.cmd_handler:
            self.cmd_handler.cmd_info()

    def publish_status(self, online):
        """Publish live online/offline status."""
        if not self.client:
            return
        payload = {
            "protocol_version": "1.0",
            "device_id": self.device_id,
            "online": bool(online),
            "timestamp_ms": int(time.time() * 1000)
        }
        try:
            self.client.publish(self.status_topic, json.dumps(payload))
            self.ha_adapter.publish_availability(self.client, online)
        except Exception as e:
            print(f"Error publishing MQTT status: {e}")
            if not self._stopping:
                self._mark_disconnected()
