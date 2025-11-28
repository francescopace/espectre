"""
Micro-ESPectre - MQTT Handler Module
Handles MQTT connection, publishing, and command processing

Author: Francesco Pace <francesco.pace@gmail.com>
License: GPLv3
"""
import json
import time
from umqtt.simple import MQTTClient
from src.mqtt.commands import MQTTCommands


class MQTTHandler:
    """MQTT handler with publishing and command support"""
    
    def __init__(self, config, segmentation, wlan, traffic_generator=None, nbvi_calibration_func=None):
        """
        Initialize MQTT handler
        
        Args:
            config: Configuration module
            segmentation: SegmentationContext instance
            wlan: WLAN instance
            traffic_generator: TrafficGenerator instance (optional)
            nbvi_calibration_func: Function to run NBVI calibration (optional)
        """
        self.config = config
        self.seg = segmentation
        self.wlan = wlan
        self.traffic_gen = traffic_generator
        self.nbvi_calibration_func = nbvi_calibration_func
        self.client = None
        self.cmd_handler = None
        
        # Topics
        self.base_topic = config.MQTT_TOPIC
        self.cmd_topic = f"{config.MQTT_TOPIC}/cmd"
        self.response_topic = f"{config.MQTT_TOPIC}/response"
        
        # Publishing state
        self.last_variance = 0.0
        self.last_state = 0  # STATE_IDLE
        self.last_publish_time = 0
        self.packets_published = 0
        self.packets_skipped = 0
        
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
            self.seg,
            self.response_topic,
            self.wlan,
            self.traffic_gen,
            self.nbvi_calibration_func
        )
        
        # Set callback for incoming messages
        self.client.set_callback(self._on_message)
        
        # Subscribe to command topic
        self.client.subscribe(self.cmd_topic)
        print(f'Subscribed to: {self.cmd_topic}')
        
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
    
    def should_publish(self, current_variance, current_state, current_time):
        """Determine if we should publish (smart publishing logic)"""
        if not self.config.SMART_PUBLISHING:
            return True
        
        # Always publish on state change
        if self.last_state != current_state:
            return True
        
        # Publish if variance changed significantly
        if abs(current_variance - self.last_variance) > self.config.DELTA_THRESHOLD:
            return True
        
        # Heartbeat: publish if max interval exceeded
        if time.ticks_diff(current_time, self.last_publish_time) >= self.config.MAX_PUBLISH_INTERVAL_MS:
            return True
        
        return False
    
    def publish_state(self, current_variance, current_state, current_threshold, 
                     packet_delta, dropped_delta, current_time):
        """
        Publish current state to MQTT
        
        Args:
            current_variance: Current moving variance
            current_state: Current state (0=IDLE, 1=MOTION)
            current_threshold: Current threshold
            packet_delta: Packets processed since last publish
            dropped_delta: Packets dropped since last publish
            current_time: Current time in milliseconds
        """
        if self.should_publish(current_variance, current_state, current_time):
            # Convert state to string format (matching C version)
            state_str = 'motion' if current_state == 1 else 'idle'
            
            payload = {
                'movement': round(current_variance, 4),
                'threshold': round(current_threshold, 4),
                'state': state_str,
                'packets_processed': packet_delta,
                'packets_dropped': dropped_delta,
                'timestamp': time.time()
            }
            
            try:
                self.client.publish(self.base_topic, json.dumps(payload))
                self.packets_published += 1
                self.last_publish_time = current_time
            except Exception as e:
                print(f"Error publishing to MQTT: {e}")
        else:
            self.packets_skipped += 1
        
        # Update state
        self.last_variance = current_variance
        self.last_state = current_state
    
    def update_packet_count(self, count, dropped=None):
        """Update packets processed and dropped counters in command handler"""
        if self.cmd_handler:
            self.cmd_handler.update_packet_count(count, dropped)
    
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
    
    def get_stats(self):
        """Get publishing statistics"""
        return {
            'published': self.packets_published,
            'skipped': self.packets_skipped
        }
