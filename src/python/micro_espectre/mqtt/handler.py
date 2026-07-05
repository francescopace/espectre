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
except ImportError:
    from mqtt.commands import MQTTCommands


class MQTTHandler:
    """MQTT handler with publishing and command support"""
    
    def __init__(self, config, detector, wlan, global_state=None):
        """
        Initialize MQTT handler
        
        Args:
            config: Configuration module
            detector: IDetector instance (MVSDetector or MLDetector)
            wlan: WLAN instance
            global_state: GlobalState instance for accessing loop metrics (optional)
        """
        self.config = config
        self.detector = detector
        self.wlan = wlan
        self.global_state = global_state
        self.client = None
        self.cmd_handler = None
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
        self.client = MQTTClient(
            self.config.MQTT_CLIENT_ID,
            self.config.MQTT_BROKER,
            port=self.config.MQTT_PORT,
            user=self.config.MQTT_USERNAME,
            password=self.config.MQTT_PASSWORD
        )
        
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
        #print(f'Subscribed to: {self.cmd_topic}')
        self.publish_status(True)
        self.publish_info()
        
        return self.client
    
    def _on_message(self, topic, msg):
        """Callback for incoming MQTT messages"""
        try:
            topic_str = topic.decode('utf-8') if isinstance(topic, bytes) else topic
            
            if topic_str == self.cmd_topic:
                # Process command
                self.cmd_handler.process_command(msg)
            
        except Exception as e:
            print(f"Error processing MQTT message: {e}")
    
    def check_messages(self):
        """Check for incoming MQTT messages (non-blocking)"""
        try:
            self.client.check_msg()
        except Exception as e:
            print(f"Error checking MQTT messages: {e}")
    
    def publish_state(self, current_variance, current_state, current_threshold):
        """
        Publish current state to MQTT
        
        Args:
            current_variance: Current moving variance (or probability for ML)
            current_state: Current state (0=IDLE, 1=MOTION)
            current_threshold: Current threshold
        """
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
        except Exception as e:
            print(f"Error publishing to MQTT: {e}")
        
        # Update state
        self.last_variance = current_variance
        self.last_state = current_state
    
    def disconnect(self):
        """Disconnect from MQTT broker"""
        if self.client:
            try:
                self.publish_status(False)
                self.client.disconnect()
                print('MQTT disconnected')
            except Exception as e:
                print(f"Error disconnecting MQTT: {e}")
    
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
        except Exception as e:
            print(f"Error publishing MQTT status: {e}")
