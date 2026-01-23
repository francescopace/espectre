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
from src.mqtt.commands import MQTTCommands


class MQTTHandler:
    """MQTT handler with publishing and command support"""
    
    def __init__(self, config, detector, wlan, traffic_generator=None, band_calibration_func=None, global_state=None):
        """
        Initialize MQTT handler
        
        Args:
            config: Configuration module
            detector: IDetector instance (MVSDetector or PCADetector)
            wlan: WLAN instance
            traffic_generator: TrafficGenerator instance (optional)
            band_calibration_func: Function to run band calibration (optional)
            global_state: GlobalState instance for accessing loop metrics (optional)
        """
        self.config = config
        self.detector = detector
        self.wlan = wlan
        self.traffic_gen = traffic_generator
        self.band_calibration_func = band_calibration_func
        self.global_state = global_state
        self.client = None
        self.cmd_handler = None
        
        # Topics
        self.base_topic = config.MQTT_TOPIC
        self.cmd_topic = f"{config.MQTT_TOPIC}/cmd"
        self.response_topic = f"{config.MQTT_TOPIC}/response"
        
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
            self.response_topic,
            self.wlan,
            self.traffic_gen,
            self.band_calibration_func,
            self.global_state
        )
        
        # Set callback for incoming messages
        self.client.set_callback(self._on_message)
        
        # Subscribe to command topic
        self.client.subscribe(self.cmd_topic)
        #print(f'Subscribed to: {self.cmd_topic}')
        
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
    
    def publish_state(self, current_variance, current_state, current_threshold, 
                     packet_delta, dropped_delta, pps, features=None, confidence=None, triggered=None):
        """
        Publish current state to MQTT
        
        Args:
            current_variance: Current moving variance
            current_state: Current state (0=IDLE, 1=MOTION)
            current_threshold: Current threshold
            packet_delta: Packets processed since last publish
            dropped_delta: Packets dropped since last publish
            pps: Packets per second (calculated server-side with ms precision)
            features: Optional dict of CSI features
            confidence: Optional confidence score (0-1) from multi-feature detector
            triggered: Optional list of triggered feature names
        """
        # Convert state to string format (matching C version)
        state_str = 'motion' if current_state == 1 else 'idle'
        
        payload = {
            'movement': round(current_variance, 4),
            'threshold': round(current_threshold, 4),
            'state': state_str,
            'packets_processed': packet_delta,
            'packets_dropped': dropped_delta,
            'pps': pps,
            'timestamp': time.time()
        }
        
        # Add CSI features if available (publish-time features)
        if features is not None:
            payload['features'] = {
                # Turbulence buffer features
                'entropy_turb': round(features.get('entropy_turb', 0), 3),
                'iqr_turb': round(features.get('iqr_turb', 0), 3),
                'variance_turb': round(features.get('variance_turb', 0), 3),
                # W=1 features (current packet)
                'skewness': round(features.get('skewness', 0), 3),
                'kurtosis': round(features.get('kurtosis', 0), 3)
            }
            
            # Add confidence score if available
            if confidence is not None:
                payload['confidence'] = round(confidence, 3)
            
            # Add triggered features if available
            if triggered is not None:
                payload['triggered'] = triggered
        
        try:
            self.client.publish(self.base_topic, json.dumps(payload))
        except Exception as e:
            print(f"Error publishing to MQTT: {e}")
        
        # Update state
        self.last_variance = current_variance
        self.last_state = current_state
    
    def disconnect(self):
        """Disconnect from MQTT broker"""
        if self.client:
            try:
                self.client.disconnect()
                print('MQTT disconnected')
            except Exception as e:
                print(f"Error disconnecting MQTT: {e}")
    
    def publish_info(self):
        """Publish system info"""
        if self.cmd_handler:
            self.cmd_handler.cmd_info()
