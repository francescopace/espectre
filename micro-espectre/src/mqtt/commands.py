"""
Micro-ESPectre - MQTT Commands Module
Handles MQTT command processing (info, stats)

Author: Francesco Pace <francesco.pace@gmail.com>
License: GPLv3
"""
import json
import time
import gc
import network
from src.nvs_storage import NVSStorage


class MQTTCommands:
    """MQTT command processor"""
    
    def __init__(self, mqtt_client, config, segmentation, response_topic, traffic_generator=None):
        """
        Initialize MQTT commands
        
        Args:
            mqtt_client: MQTT client instance
            config: Configuration module
            segmentation: SegmentationContext instance
            response_topic: MQTT topic for responses
            traffic_generator: TrafficGenerator instance (optional)
        """
        self.mqtt = mqtt_client
        self.config = config
        self.seg = segmentation
        self.traffic_gen = traffic_generator
        self.response_topic = response_topic
        self.start_time = time.time()
        self.packets_processed = 0
        
        # CPU usage estimation
        self.last_stats_time = time.ticks_ms()
        self.last_packets_count = 0
        
        # NVS storage for configuration persistence
        self.nvs = NVSStorage()
        
    def send_response(self, message):
        """Send response message to MQTT"""
        try:
            # If message is a dict, convert to JSON
            if isinstance(message, dict):
                message = json.dumps(message)
            else:
                # If message is plain text, check if it's already valid JSON
                try:
                    json.loads(message)
                    # Already valid JSON, send as-is
                except (ValueError, TypeError):
                    # Plain text message, wrap in {"response": "..."}
                    message = json.dumps({"response": message})
            
            self.mqtt.publish(self.response_topic, message)
        except Exception as e:
            print(f"Error sending MQTT response: {e}")
    
    def format_uptime(self, uptime_sec):
        """Format uptime as human-readable string"""
        hours = int(uptime_sec // 3600)
        minutes = int((uptime_sec % 3600) // 60)
        seconds = int(uptime_sec % 60)
        
        if hours > 0:
            return f"{hours}h {minutes}m {seconds}s"
        elif minutes > 0:
            return f"{minutes}m {seconds}s"
        else:
            return f"{seconds}s"
    
    def cmd_info(self):
        """Get system information"""
        # Get WiFi info
        wlan = network.WLAN(network.STA_IF)
        ip_address = "not connected"
        if wlan.isconnected():
            ip_info = wlan.ifconfig()
            ip_address = ip_info[0] if ip_info else "unknown"
        
        # Get traffic generator rate (current runtime value)
        traffic_rate = 0
        if self.traffic_gen:
            traffic_rate = self.traffic_gen.get_rate()
        
        response = {
            "network": {
                "ip_address": ip_address,
                "traffic_generator_rate": traffic_rate
            },
            "mqtt": {
                "base_topic": self.config.MQTT_TOPIC,
                "cmd_topic": f"{self.config.MQTT_TOPIC}/cmd",
                "response_topic": self.response_topic
            },
            "segmentation": {
                "threshold": round(self.seg.threshold, 2),
                "window_size": self.seg.window_size,
                "min_length": self.seg.min_length,
                "max_length": self.seg.max_length
            },
            "options": {
                "smart_publishing_enabled": self.config.SMART_PUBLISHING
            },
            "subcarriers": {
                "indices": self.config.SELECTED_SUBCARRIERS,
                "count": len(self.config.SELECTED_SUBCARRIERS)
            }
        }
        
        self.send_response(response)
        print("📋 Info command processed")
    
    def cmd_stats(self):
        """Get runtime statistics"""
        current_time = time.time()
        uptime_sec = current_time - self.start_time
        
        # Get memory info
        gc.collect()
        free_mem = gc.mem_free()
        alloc_mem = gc.mem_alloc()
        total_mem = free_mem + alloc_mem
        heap_usage = (alloc_mem / total_mem * 100) if total_mem > 0 else 0
        
        # Estimate CPU usage based on packet processing rate
        current_ticks = time.ticks_ms()
        time_delta = time.ticks_diff(current_ticks, self.last_stats_time)
        packet_delta = self.packets_processed - self.last_packets_count
        
        # Estimate: assume ~5ms per packet processing, calculate % of time spent
        if time_delta > 0:
            processing_time = packet_delta * 5  # ms
            cpu_usage = min(100.0, (processing_time / time_delta) * 100)
        else:
            cpu_usage = 0.0
        
        # Update for next call
        self.last_stats_time = current_ticks
        self.last_packets_count = self.packets_processed
        
        # Get current state
        state_str = 'motion' if self.seg.state == self.seg.STATE_MOTION else 'idle'
        
        response = {
            "timestamp": int(current_time),
            "uptime": self.format_uptime(uptime_sec),
            "cpu_usage_percent": round(cpu_usage, 1),
            "heap_usage_percent": round(heap_usage, 1),
            "state": state_str,
            "turbulence": round(self.seg.last_turbulence, 4),
            "movement": round(self.seg.current_moving_variance, 4),
            "threshold": round(self.seg.threshold, 4),
            "packets_processed": self.packets_processed
        }
        
        self.send_response(response)
        print("📊 Stats command processed")
    
    def cmd_segmentation_threshold(self, cmd_obj):
        """Set segmentation threshold"""
        if 'value' not in cmd_obj:
            self.send_response("ERROR: Missing 'value' field")
            return
        
        try:
            threshold = float(cmd_obj['value'])
            
            if threshold <= 0.0 or threshold > 10.0:
                self.send_response("ERROR: Threshold must be between 0.0 and 10.0")
                return
            
            old_threshold = self.seg.threshold
            self.seg.threshold = threshold
            
            # Save to NVS
            self.nvs.save_full_config(self.seg, self.config, self.traffic_gen)
            
            self.send_response(f"Segmentation threshold updated: {old_threshold:.2f} -> {threshold:.2f}")
            print(f"📍 Threshold updated: {old_threshold:.2f} -> {threshold:.2f}")
            
        except ValueError:
            self.send_response("ERROR: Invalid threshold value (must be float)")
    
    def cmd_segmentation_window_size(self, cmd_obj):
        """Set window size"""
        if 'value' not in cmd_obj:
            self.send_response("ERROR: Missing 'value' field")
            return
        
        try:
            window_size = int(cmd_obj['value'])
            
            if window_size < 3 or window_size > 50:
                self.send_response("ERROR: Window size must be between 3 and 50 packets")
                return
            
            old_size = self.seg.window_size
            
            # Update window size and reset buffer
            self.seg.window_size = window_size
            self.seg.turbulence_buffer = [0.0] * window_size
            self.seg.buffer_index = 0
            self.seg.buffer_count = 0
            self.seg.current_moving_variance = 0.0
            
            # Save to NVS
            self.nvs.save_full_config(self.seg, self.config, self.traffic_gen)
            
            reactivity = "more reactive" if window_size < 10 else "more stable"
            self.send_response(f"Window size updated: {old_size} -> {window_size} packets ({window_size/20.0:.2f}s @ 20Hz, {reactivity})")
            print(f"📍 Window size updated: {old_size} -> {window_size} (buffer reset)")
            
        except ValueError:
            self.send_response("ERROR: Invalid window size value (must be integer)")
    
    def cmd_segmentation_min_length(self, cmd_obj):
        """Set minimum segment length"""
        if 'value' not in cmd_obj:
            self.send_response("ERROR: Missing 'value' field")
            return
        
        try:
            min_length = int(cmd_obj['value'])
            
            if min_length < 5 or min_length > 100:
                self.send_response("ERROR: Min length must be between 5 and 100 packets")
                return
            
            old_min = self.seg.min_length
            self.seg.min_length = min_length
            
            # Save to NVS
            self.nvs.save_full_config(self.seg, self.config, self.traffic_gen)
            
            self.send_response(f"Min segment length updated: {old_min} -> {min_length} packets ({min_length/20.0:.2f}s @ 20Hz)")
            print(f"📍 Min length updated: {old_min} -> {min_length}")
            
        except ValueError:
            self.send_response("ERROR: Invalid min length value (must be integer)")
    
    def cmd_segmentation_max_length(self, cmd_obj):
        """Set maximum segment length"""
        if 'value' not in cmd_obj:
            self.send_response("ERROR: Missing 'value' field")
            return
        
        try:
            max_length = int(cmd_obj['value'])
            
            if max_length != 0 and (max_length < 10 or max_length > 200):
                self.send_response("ERROR: Max length must be 0 (no limit) or 10-200 packets")
                return
            
            old_max = self.seg.max_length
            self.seg.max_length = max_length
            
            # Save to NVS
            self.nvs.save_full_config(self.seg, self.config, self.traffic_gen)
            
            if max_length == 0:
                self.send_response(f"Max segment length updated: {old_max} -> no limit")
            else:
                self.send_response(f"Max segment length updated: {old_max} -> {max_length} packets ({max_length/20.0:.2f}s @ 20Hz)")
            print(f"📍 Max length updated: {old_max} -> {max_length}")
            
        except ValueError:
            self.send_response("ERROR: Invalid max length value (must be integer)")
    
    def cmd_subcarrier_selection(self, cmd_obj):
        """Set subcarrier selection"""
        if 'indices' not in cmd_obj:
            self.send_response("ERROR: Missing 'indices' field")
            return
        
        try:
            indices = cmd_obj['indices']
            
            if not isinstance(indices, list):
                self.send_response("ERROR: 'indices' must be an array")
                return
            
            if len(indices) < 1 or len(indices) > 64:
                self.send_response("ERROR: Number of subcarriers must be between 1 and 64")
                return
            
            # Validate all indices
            for idx in indices:
                if not isinstance(idx, int) or idx < 0 or idx > 63:
                    self.send_response(f"ERROR: Subcarrier index {idx} out of range (must be 0-63)")
                    return
            
            # Update configuration
            self.config.SELECTED_SUBCARRIERS = indices
            
            # Save to NVS
            self.nvs.save_full_config(self.seg, self.config, self.traffic_gen)
            
            self.send_response(f"Subcarrier selection updated: {len(indices)} subcarriers")
            print(f"📡 Subcarrier selection updated: {len(indices)} subcarriers")
            
        except Exception as e:
            self.send_response(f"ERROR: Invalid subcarrier selection: {e}")
    
    def cmd_smart_publishing(self, cmd_obj):
        """Enable/disable smart publishing"""
        if 'enabled' not in cmd_obj:
            self.send_response("ERROR: Missing 'enabled' field")
            return
        
        try:
            enabled = bool(cmd_obj['enabled'])
            
            old_state = self.config.SMART_PUBLISHING
            self.config.SMART_PUBLISHING = enabled
            
            # Save to NVS
            self.nvs.save_full_config(self.seg, self.config, self.traffic_gen)
            
            status = "enabled" if enabled else "disabled"
            self.send_response(f"Smart publishing {status}")
            print(f"📡 Smart publishing {status}")
            
        except Exception as e:
            self.send_response(f"ERROR: Invalid enabled value: {e}")
    
    def cmd_factory_reset(self, cmd_obj):
        """Reset all parameters to defaults"""
        print("⚠️  Factory reset requested")
        
        # Reset segmentation to defaults
        self.seg.threshold = 3.0
        self.seg.window_size = 30
        self.seg.min_length = 10
        self.seg.max_length = 60
        
        # Reset buffer
        self.seg.turbulence_buffer = [0.0] * self.seg.window_size
        self.seg.buffer_index = 0
        self.seg.buffer_count = 0
        self.seg.current_moving_variance = 0.0
        
        # Reset state machine
        self.seg.state = self.seg.STATE_IDLE
        self.seg.motion_start_index = 0
        self.seg.motion_length = 0
        self.seg.packet_index = 0
        
        # Reset subcarrier selection to defaults
        self.config.SELECTED_SUBCARRIERS = [47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58]
        
        # Reset smart publishing
        self.config.SMART_PUBLISHING = False
        
        # Stop traffic generator if running
        if self.traffic_gen and self.traffic_gen.is_running():
            self.traffic_gen.stop()
        
        # Erase saved configuration
        self.nvs.erase()
        
        self.send_response("Factory reset complete")
        print("✅ Factory reset complete")
    
    def cmd_traffic_generator_rate(self, cmd_obj):
        """Set traffic generator rate"""
        if not self.traffic_gen:
            self.send_response("ERROR: Traffic generator not available")
            return
        
        if 'value' not in cmd_obj:
            self.send_response("ERROR: Missing 'value' field")
            return
        
        try:
            rate = int(cmd_obj['value'])
            
            if rate < 0 or rate > 50:
                self.send_response("ERROR: Rate must be 0-50 packets/sec (0=disabled, recommended: 15)")
                return
            
            old_rate = self.traffic_gen.get_rate()
            
            if rate == 0:
                # Disable traffic generator
                if self.traffic_gen.is_running():
                    self.traffic_gen.stop()
                    self.send_response(f"Traffic generator disabled (was {old_rate} pps)")
                    print(f"📡 Traffic generator disabled")
                else:
                    self.send_response("Traffic generator already disabled")
            else:
                # Enable or change rate
                if self.traffic_gen.is_running():
                    # Change rate
                    if self.traffic_gen.set_rate(rate):
                        # Save to NVS
                        self.nvs.save_full_config(self.seg, self.config, self.traffic_gen)
                        self.send_response(f"Traffic rate updated: {old_rate} -> {rate} pps")
                        print(f"📡 Traffic rate changed to {rate} pps")
                    else:
                        self.send_response("ERROR: Failed to change traffic rate")
                else:
                    # Start traffic generator
                    if self.traffic_gen.start(rate):
                        # Save to NVS
                        self.nvs.save_full_config(self.seg, self.config, self.traffic_gen)
                        self.send_response(f"Traffic generator enabled ({rate} pps)")
                        print(f"📡 Traffic generator started ({rate} pps)")
                    else:
                        self.send_response("ERROR: Failed to start traffic generator")
                        
        except ValueError:
            self.send_response("ERROR: Invalid rate value (must be integer)")
    
    def process_command(self, data):
        """
        Process incoming MQTT command
        
        Args:
            data: Command data (bytes or string)
        """
        try:
            # Parse JSON command
            if isinstance(data, bytes):
                data = data.decode('utf-8')
            
            cmd_obj = json.loads(data)
            
            if 'cmd' not in cmd_obj:
                self.send_response("ERROR: Missing 'cmd' field")
                return
            
            command = cmd_obj['cmd']
            print(f"Processing MQTT command: {command}")
            
            # Dispatch command
            if command == 'info':
                self.cmd_info()
            elif command == 'stats':
                self.cmd_stats()
            elif command == 'segmentation_threshold':
                self.cmd_segmentation_threshold(cmd_obj)
            elif command == 'segmentation_window_size':
                self.cmd_segmentation_window_size(cmd_obj)
            elif command == 'segmentation_min_length':
                self.cmd_segmentation_min_length(cmd_obj)
            elif command == 'segmentation_max_length':
                self.cmd_segmentation_max_length(cmd_obj)
            elif command == 'subcarrier_selection':
                self.cmd_subcarrier_selection(cmd_obj)
            elif command == 'smart_publishing':
                self.cmd_smart_publishing(cmd_obj)
            elif command == 'factory_reset':
                self.cmd_factory_reset(cmd_obj)
            elif command == 'traffic_generator_rate':
                self.cmd_traffic_generator_rate(cmd_obj)
            else:
                self.send_response(f"ERROR: Unknown command '{command}'")
                
        except Exception as e:
            error_msg = f"ERROR: Command processing failed: {e}"
            print(error_msg)
            self.send_response(error_msg)
    
    def update_packet_count(self, count):
        """Update packets processed counter"""
        self.packets_processed = count
