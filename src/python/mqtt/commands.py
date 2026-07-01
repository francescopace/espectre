"""
Micro-ESPectre - MQTT Commands Module

Processes MQTT commands for remote configuration.
Handles system configuration, startup calibration, and status queries via MQTT.

Author: Francesco Pace <francesco.pace@gmail.com>
License: GPLv3
"""
import json
import time
import gc
import sys

# Threshold limits (unified with MVS/ML detector runtime validation)
SEG_THRESHOLD_MIN = 0.0
SEG_THRESHOLD_MAX = 10.0

class MQTTCommands:
    """MQTT command processor"""
    
    def __init__(self,
                 mqtt_client,
                 config,
                 detector,
                 accepted_topic,
                 rejected_topic,
                 info_topic,
                 stats_topic,
                 wlan,
                 global_state=None):
        """
        Initialize MQTT commands
        
        Args:
            mqtt_client: MQTT client instance
            config: Configuration module
            detector: IDetector instance (MVSDetector or MLDetector)
            accepted_topic: MQTT topic for accepted command responses
            rejected_topic: MQTT topic for rejected command responses
            info_topic: MQTT topic for retained system info
            stats_topic: MQTT topic for runtime stats
            wlan: wlan instance
            global_state: GlobalState instance for accessing loop metrics (optional)
        """
        self.mqtt = mqtt_client
        self.config = config
        self.detector = detector
        self.wlan = wlan
        self.global_state = global_state
        self.accepted_topic = accepted_topic
        self.rejected_topic = rejected_topic
        self.info_topic = info_topic
        self.stats_topic = stats_topic
        self.start_time = time.time()
        
        # Check detector type for MVS-specific features
        self._is_mvs = detector.get_name() == "MVS"
    
    def _get_detection_info(self):
        """Build detection info dict based on detector type."""
        return {"algorithm": self.detector.get_name()}
        
    def send_response(self, message, accepted=True, command_id="", command=""):
        """Send response message to MQTT"""
        try:
            topic = self.accepted_topic if accepted else self.rejected_topic
            # If message is a dict, convert to JSON
            if isinstance(message, dict):
                payload = message
            else:
                # If message is plain text, check if it's already valid JSON
                try:
                    payload = json.loads(message)
                    # Already valid JSON, send as-is
                except (ValueError, TypeError):
                    # Plain text message, wrap in {"response": "..."}
                    payload = {"response": message}

            if isinstance(payload, dict) and "protocol_version" not in payload:
                payload.setdefault("protocol_version", "1.0")
            if isinstance(payload, dict) and command_id:
                payload.setdefault("command_id", command_id)
            if isinstance(payload, dict) and command:
                payload.setdefault("command", command)
            if isinstance(payload, dict) and "accepted" not in payload and (command_id or command):
                payload["accepted"] = bool(accepted)
            
            self.mqtt.publish(topic, json.dumps(payload))
        except Exception as e:
            print(f"Error sending MQTT response: {e}")

    def publish_info_payload(self, payload):
        """Publish retained info payload."""
        try:
            self.mqtt.publish(self.info_topic, json.dumps(payload), retain=True)
        except TypeError:
            self.mqtt.publish(self.info_topic, json.dumps(payload))

    def publish_stats_payload(self, payload):
        """Publish stats payload."""
        self.mqtt.publish(self.stats_topic, json.dumps(payload))
    
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
        ip_address = ""
        mac_address = ""
        channel_primary = 0
        
        if self.wlan.active():
            try:
                mac_bytes = self.wlan.config('mac')
                mac_address = ':'.join(['%02X' % b for b in mac_bytes])
            except Exception:  # pragma: no cover
                pass
            
            if self.wlan.isconnected():
                ip_info = self.wlan.ifconfig()
                ip_address = ip_info[0] if ip_info else ""
            
            try:
                channel_primary = self.wlan.config('channel')
            except Exception:  # pragma: no cover
                pass
        
        response = {
            "protocol_version": "1.0",
            "device_id": self.config.MQTT_CLIENT_ID,
            "device_name": getattr(self.config, "MQTT_DEVICE_LABEL", self.config.MQTT_CLIENT_ID),
            "frontend": "micro",
            "firmware_version": "micropython",
            "chip": getattr(self.global_state, 'chip_type', None) or sys.platform,
            "network": {
                "ip_address": ip_address,
                "mac_address": mac_address,
                "channel": {
                    "primary": channel_primary
                }
            },
            "detection": self._get_detection_info()
        }
        
        self.publish_info_payload(response)
    
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

        response = {
            "protocol_version": "1.0",
            "device_id": self.config.MQTT_CLIENT_ID,
            "timestamp": int(current_time),
            "timestamp_ms": int(current_time * 1000),
            "uptime": self.format_uptime(uptime_sec),
            "free_memory_kb": free_mem_kb,
            "loop_time_ms": loop_time_ms,
        }
        
        self.publish_stats_payload(response)
    
    def cmd_set_threshold(self, cmd_obj):
        """Set detection threshold (session-only, not persisted)"""
        command_id = cmd_obj.get('command_id', '')
        command = cmd_obj.get('command', 'set_threshold')
        if 'threshold' not in cmd_obj:
            self.send_response("ERROR: Missing 'threshold' field", accepted=False, command_id=command_id, command=command)
            return
        
        try:
            threshold = float(cmd_obj['threshold'])
            
            if threshold < SEG_THRESHOLD_MIN or threshold > SEG_THRESHOLD_MAX:
                self.send_response(
                    f"ERROR: Threshold must be between {SEG_THRESHOLD_MIN} and {SEG_THRESHOLD_MAX}",
                    accepted=False,
                    command_id=command_id,
                    command=command
                )
                return
            
            old_threshold = self.detector.get_threshold()
            if not self.detector.set_threshold(threshold):
                self.send_response(
                    f"ERROR: Threshold rejected by detector (allowed range: {SEG_THRESHOLD_MIN}-{SEG_THRESHOLD_MAX})",
                    accepted=False,
                    command_id=command_id,
                    command=command
                )
                return
            
            # Note: threshold is session-only, adaptive threshold is recalculated on every boot
            
            self.send_response(
                f"Detection threshold updated: {old_threshold:.4f} -> {threshold:.4f} (session-only)",
                accepted=True,
                command_id=command_id,
                command=command
            )
            print(f"Threshold updated: {old_threshold:.4f} -> {threshold:.4f} (session-only)")
            
        except ValueError:
            self.send_response(
                "ERROR: Invalid threshold value (must be float)",
                accepted=False,
                command_id=command_id,
                command=command
            )
            return
    
            
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
            
            command_id = cmd_obj.get('command_id', '')
            if 'command' not in cmd_obj:
                self.send_response("ERROR: Missing 'command' field", accepted=False, command_id=command_id)
                return
            
            command = cmd_obj['command']
            #print(f"Processing MQTT command: {command}")
            
            # Dispatch command
            if command == 'info':
                self.cmd_info()
                self.send_response("info published", accepted=True, command_id=command_id, command=command)
            elif command == 'stats':
                self.cmd_stats()
                self.send_response("stats published", accepted=True, command_id=command_id, command=command)
            elif command == 'set_threshold':
                self.cmd_set_threshold(cmd_obj)
            else:
                self.send_response(f"ERROR: Unknown command '{command}'", accepted=False, command_id=command_id, command=command)
                
        except Exception as e:
            error_msg = f"ERROR: Command processing failed: {e}"
            print(error_msg)
            self.send_response(error_msg, accepted=False)
