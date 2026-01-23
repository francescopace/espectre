"""
Micro-ESPectre - MQTT Commands Module

Processes MQTT commands for remote configuration.
Handles system configuration, calibration, and status queries via MQTT.

Author: Francesco Pace <francesco.pace@gmail.com>
License: GPLv3
"""
import json
import time
import gc
import sys
from src.config import (
    TRAFFIC_GENERATOR_RATE,
    SEG_WINDOW_SIZE
)

# Segmentation limits
SEG_WINDOW_SIZE_MIN = 10
SEG_WINDOW_SIZE_MAX = 200
SEG_THRESHOLD_MIN = 0.0
SEG_THRESHOLD_MAX = 10.0

class MQTTCommands:
    """MQTT command processor"""
    
    def __init__(self, mqtt_client, config, detector, response_topic, wlan, traffic_generator=None, band_calibration_func=None, global_state=None):
        """
        Initialize MQTT commands
        
        Args:
            mqtt_client: MQTT client instance
            config: Configuration module
            detector: IDetector instance (MVSDetector or PCADetector)
            response_topic: MQTT topic for responses
            wlan: wlan instance
            traffic_generator: TrafficGenerator instance (optional)
            band_calibration_func: Function to run band calibration (optional)
            global_state: GlobalState instance for accessing loop metrics (optional)
        """
        self.mqtt = mqtt_client
        self.config = config
        self.detector = detector
        self.wlan = wlan
        self.traffic_gen = traffic_generator
        self.band_calibration_func = band_calibration_func
        self.global_state = global_state
        self.response_topic = response_topic
        self.start_time = time.time()
        
        # Check detector type for MVS-specific features
        self._is_mvs = detector.get_name() == "MVS"
        
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
        import network
        
        # Get WiFi info
        ip_address = "not connected"
        mac_address = "unknown"
        channel_primary = 0
        channel_secondary = 0
        bandwidth = "HT20"  # MicroPython ESP32 default
        protocol = "unknown"
        csi_enabled = False
        
        if self.wlan.active():
            # MAC address
            mac_bytes = self.wlan.config('mac')
            mac_address = ':'.join(['%02X' % b for b in mac_bytes])
            
            # IP address
            if self.wlan.isconnected():
                ip_info = self.wlan.ifconfig()
                ip_address = ip_info[0] if ip_info else "unknown"
            
            # WiFi channel
            try:
                channel_primary = self.wlan.config('channel')
                # MicroPython doesn't expose secondary channel directly
                channel_secondary = 0
            except:
                pass
            
            # WiFi protocol - decode bitmask
            try:
                proto_val = self.wlan.config('protocol')
                modes = []
                if proto_val & network.MODE_11B:
                    modes.append('b')
                if proto_val & network.MODE_11G:
                    modes.append('g')
                if proto_val & network.MODE_11N:
                    modes.append('n')
                # 802.11ax (WiFi 6) - bit 6 (64), not exposed as constant in MicroPython
                if proto_val & 64:
                    modes.append('ax')
                if proto_val & network.MODE_LR:
                    modes.append('LR')
                protocol = '802.11' + '/'.join(modes) if modes else 'unknown'
            except:
                pass
            
            # CSI enabled (indicates promiscuous-like mode for CSI capture)
            try:
                csi_enabled = hasattr(self.wlan, 'csi_available')
            except:
                pass
        
        # Get traffic generator rate (current runtime value)
        traffic_rate = 0
        if self.traffic_gen:
            traffic_rate = self.traffic_gen.get_rate()
        
        response = {
            "network": {
                "ip_address": ip_address,
                "mac_address": mac_address,
                "traffic_generator_rate": traffic_rate,
                "channel": {
                    "primary": channel_primary,
                    "secondary": channel_secondary
                },
                "bandwidth": bandwidth,
                "protocol": protocol,
                "csi_enabled": csi_enabled
            },
            "device": {
                "type": sys.platform
            },
            "mqtt": {
                "base_topic": self.config.MQTT_TOPIC,
                "cmd_topic": f"{self.config.MQTT_TOPIC}/cmd",
                "response_topic": self.response_topic
            },
            "detection": {
                "algorithm": self.detector.get_name(),
                "threshold": round(self.detector.get_threshold(), 4),
                "threshold_source": "config" if getattr(self.config, 'SEG_THRESHOLD', None) is not None else "auto",
                "window_size": self.detector._context.window_size if self._is_mvs else "N/A"
            },
            "subcarriers": {
                "indices": getattr(self.config, 'SELECTED_SUBCARRIERS', None) or []
            }
        }
        
        self.send_response(response)
    
    def cmd_stats(self):
        """Get runtime statistics"""
        current_time = time.time()
        uptime_sec = current_time - self.start_time
        
        # Get free memory in KB using gc module (Python heap)
        free_mem_kb = round(gc.mem_free() / 1024, 1)
        
        # Get loop time from global state (in microseconds, convert to ms)
        loop_time_ms = 0
        if self.global_state and hasattr(self.global_state, 'loop_time_us'):
            loop_time_ms = round(self.global_state.loop_time_us / 1000, 2)
        
        # Get current state
        state_str = 'motion' if self.detector.get_state() == 1 else 'idle'
        
        # Get traffic generator stats
        traffic_gen_stats = {}
        if self.traffic_gen:
            traffic_gen_stats = {
                "running": self.traffic_gen.is_running(),
                "target_pps": self.traffic_gen.get_rate(),
                "actual_pps": self.traffic_gen.get_actual_pps(),
                "packets_sent": self.traffic_gen.get_packet_count(),
                "errors": self.traffic_gen.get_error_count(),
                "avg_loop_ms": self.traffic_gen.get_avg_loop_time_ms()
            }
        
        # Get motion metric (turbulence for MVS, jitter for PCA)
        motion_metric = self.detector.get_motion_metric()
        turbulence = self.detector._context.last_turbulence if self._is_mvs else 0.0
        
        response = {
            "timestamp": int(current_time),
            "uptime": self.format_uptime(uptime_sec),
            "free_memory_kb": free_mem_kb,
            "loop_time_ms": loop_time_ms,
            "algorithm": self.detector.get_name(),
            "state": state_str,
            "turbulence": round(turbulence, 4),
            "movement": round(motion_metric, 4),
            "threshold": round(self.detector.get_threshold(), 4),
            "traffic_generator": traffic_gen_stats
        }
        
        self.send_response(response)
    
    def cmd_segmentation_threshold(self, cmd_obj):
        """Set detection threshold (session-only, not persisted)"""
        if 'value' not in cmd_obj:
            self.send_response("ERROR: Missing 'value' field")
            return
        
        try:
            threshold = float(cmd_obj['value'])
            
            if threshold < SEG_THRESHOLD_MIN or threshold > SEG_THRESHOLD_MAX:
                self.send_response(f"ERROR: Threshold must be between {SEG_THRESHOLD_MIN} and {SEG_THRESHOLD_MAX}")
                return
            
            old_threshold = self.detector.get_threshold()
            self.detector.set_threshold(threshold)
            
            # Note: threshold is session-only, adaptive threshold is recalculated on every boot
            
            self.send_response(f"Detection threshold updated: {old_threshold:.4f} -> {threshold:.4f} (session-only)")
            print(f"Threshold updated: {old_threshold:.4f} -> {threshold:.4f} (session-only)")
            
        except ValueError:
            self.send_response("ERROR: Invalid threshold value (must be float)")
    
    def cmd_segmentation_window_size(self, cmd_obj):
        """Set window size (MVS only)"""
        if not self._is_mvs:
            self.send_response("ERROR: Window size adjustment is only available for MVS detector")
            return
        
        if 'value' not in cmd_obj:
            self.send_response("ERROR: Missing 'value' field")
            return
        
        try:
            window_size = int(cmd_obj['value'])
            
            if window_size < SEG_WINDOW_SIZE_MIN or window_size > SEG_WINDOW_SIZE_MAX:
                self.send_response(f"ERROR: Window size must be between {SEG_WINDOW_SIZE_MIN} and {SEG_WINDOW_SIZE_MAX} packets")
                return
            
            ctx = self.detector._context
            old_size = ctx.window_size
            
            # Update window size and reset buffer
            ctx.window_size = window_size
            ctx.turbulence_buffer = [0.0] * window_size
            ctx.buffer_index = 0
            ctx.buffer_count = 0
            ctx.current_moving_variance = 0.0
            
            # Calculate duration using actual traffic rate
            rate = self.traffic_gen.get_rate() if self.traffic_gen else TRAFFIC_GENERATOR_RATE
            duration = window_size / rate if rate > 0 else 0.0
            reactivity = "more reactive" if window_size < (SEG_WINDOW_SIZE_MAX // 2) else "more stable"
            self.send_response(f"Window size updated: {old_size} -> {window_size} packets ({duration:.2f}s @ {rate}Hz, {reactivity}, session-only)")
            print(f"Window size updated: {old_size} -> {window_size} (session-only)")
            
        except ValueError:
            self.send_response("ERROR: Invalid window size value (must be integer)")
    
    def cmd_factory_reset(self, cmd_obj):
        """Reset all parameters to defaults and trigger re-calibration"""
        print("Factory reset requested")
        
        # Reset detector
        self.detector.reset()
        self.detector.set_threshold(1.0 if self._is_mvs else 0.01)
        
        # Reset MVS-specific parameters
        if self._is_mvs:
            ctx = self.detector._context
            ctx.window_size = SEG_WINDOW_SIZE
            ctx.turbulence_buffer = [0.0] * ctx.window_size
            ctx.buffer_index = 0
            ctx.buffer_count = 0
            ctx.current_moving_variance = 0.0

        print("Factory reset complete")
        
        # Run calibration immediately if function provided
        if self.band_calibration_func:
            self.send_response("Factory reset complete. Starting re-calibration...")
            print("Starting re-calibration...")
            
            # Get chip_type from global_state if available
            chip_type = getattr(self.global_state, 'chip_type', None) if self.global_state else None
            
            # Run calibration with detector
            success = self.band_calibration_func(self.wlan, self.detector, self.traffic_gen, chip_type)
            
            if success:
                if self._is_mvs:
                    band = getattr(self.config, 'SELECTED_SUBCARRIERS')
                    self.send_response(f"Re-calibration successful! Band: {band}")
                else:
                    self.send_response(f"Re-calibration successful! Threshold: {self.detector.get_threshold():.4f}")
            else:
                self.send_response(f"Re-calibration failed. Using default settings.")
        else:
            self.send_response(f"Factory reset complete.")
            
        # Send info response with updated configuration
        self.cmd_info()
    
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
            #print(f"Processing MQTT command: {command}")
            
            # Dispatch command
            if command == 'info':
                self.cmd_info()
            elif command == 'stats':
                self.cmd_stats()
            elif command == 'segmentation_threshold':
                self.cmd_segmentation_threshold(cmd_obj)
            elif command == 'segmentation_window_size':
                self.cmd_segmentation_window_size(cmd_obj)
            elif command == 'factory_reset':
                self.cmd_factory_reset(cmd_obj)
            else:
                self.send_response(f"ERROR: Unknown command '{command}'")
                
        except Exception as e:
            error_msg = f"ERROR: Command processing failed: {e}"
            print(error_msg)
            self.send_response(error_msg)
