# SPDX-License-Identifier: GPL-3.0-only
# Commercial licensing available under separate agreement; see LICENSING.md.
"""
Micro-ESPectre - MQTT Commands Module

Processes MQTT commands for remote configuration.
Handles system configuration, startup calibration, and status queries via MQTT.

Author: Francesco Pace <francesco.pace@gmail.com>
"""
import json
import time
import gc
import sys

try:
    from src.detector_interface import get_detector_algorithm
    from src.config import MOTION_HITS_MAX, MOTION_HITS_MIN
    from src.runtime_diagnostics import apply_diagnostics_sample, wifi_rssi_dbm
except ImportError:
    from detector_interface import get_detector_algorithm
    from config import MOTION_HITS_MAX, MOTION_HITS_MIN
    from runtime_diagnostics import apply_diagnostics_sample, wifi_rssi_dbm

# Threshold limits shared by the runtime detectors
THRESHOLD_MIN = 0.0
THRESHOLD_MAX = 1.0


def _protocol_mqtt_commands(
    supports_info=True,
    supports_stats=False,
    supports_runtime_threshold=False,
    supports_runtime_motion_hits=False,
    supports_runtime_detector=False,
    supports_manual_recalibration=False,
    supports_traffic_control=False,
    supports_ble=False,
    supports_ota=False,
):
    """Return the MQTT command names advertised by this frontend."""
    commands = ["commands"]
    if supports_info:
        commands.append("info")
    if supports_stats:
        commands.append("stats")
    if supports_runtime_threshold:
        commands.append("set_threshold")
    if supports_runtime_motion_hits:
        commands.append("set_motion_hits")
    if supports_runtime_detector:
        commands.append("set_detector")
    if supports_manual_recalibration:
        commands.append("recalibrate")
    if supports_traffic_control:
        commands.append("set_csi_traffic_mode")
        commands.append("set_traffic_generator_mode")
    if supports_ble:
        commands.append("set_ble")
    if supports_ota:
        commands.extend(["ota_status", "ota_check", "ota_start"])
    return commands


def _is_ascii_alnum(char):
    """Return whether one character is an ASCII letter or digit."""
    code = ord(char)
    return (
        48 <= code <= 57
        or 65 <= code <= 90
        or 97 <= code <= 122
    )


def _threshold_bounds_for_detector(detector):
    """Return the accepted threshold range for the active detector."""
    return THRESHOLD_MIN, THRESHOLD_MAX


def _normalize_chip_label(chip):
    """Normalize chip identifiers to the shared short labels used by firmware."""
    if not chip:
        return "UNK"
    normalized = ''.join(ch for ch in str(chip).upper() if _is_ascii_alnum(ch))
    if normalized == "ESP32C3":
        return "C3"
    if normalized == "ESP32C5":
        return "C5"
    if normalized == "ESP32C6":
        return "C6"
    if normalized == "ESP32S3":
        return "S3"
    if normalized == "ESP32":
        return "ESP32"
    return normalized or "UNK"


def _protocol_device_name(device_id, chip):
    """Build the immutable ESPectre device name from chip and device_id."""
    compact_id = ''.join(
        ch for ch in str(device_id).lower() if _is_ascii_alnum(ch)
    )
    suffix = compact_id[-6:] if compact_id else "000000"
    return "ESPectre {} {}".format(_normalize_chip_label(chip), suffix)


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
                 global_state=None,
                 runtime_policy=None,
                 ha_adapter=None,
                 recalibrate_callback=None,
                 traffic_control_callback=None,
                 traffic_control_supported=False,
                 catalog_topic=None):
        """
        Initialize MQTT commands
        
        Args:
            mqtt_client: MQTT client instance
            config: Configuration module
            detector: IDetector instance
            accepted_topic: MQTT topic for accepted command responses
            rejected_topic: MQTT topic for rejected command responses
            info_topic: MQTT topic for live system info
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
        if catalog_topic:
            self.catalog_topic = catalog_topic
        elif info_topic.endswith("/info"):
            self.catalog_topic = info_topic[: -len("/info")] + "/commands/catalog"
        else:
            self.catalog_topic = ""
        self.start_time = time.time()
        self.runtime_policy = runtime_policy
        self.ha_adapter = ha_adapter
        self.recalibrate_callback = recalibrate_callback
        self.traffic_control_callback = traffic_control_callback
        self.traffic_control_supported = bool(traffic_control_supported)
        
    def _get_detection_info(self):
        """Build detection info dict based on detector type."""
        return {"algorithm": get_detector_algorithm(self.detector)}
        
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
                    # Plain text message, wrap in protocol command-result form.
                    payload = {"message": message}

            if isinstance(payload, dict):
                if "response" in payload and "message" not in payload:
                    payload["message"] = payload.pop("response")
                payload.setdefault("protocol_version", "1.0")
                payload.setdefault("device_id", self.config.MQTT_CLIENT_ID)
                if command_id:
                    payload.setdefault("command_id", command_id)
                if command:
                    payload.setdefault("command", command)
                payload.setdefault("accepted", bool(accepted))
            
            self.mqtt.publish(topic, json.dumps(payload))
        except Exception as e:
            print(f"Error sending MQTT response: {e}")

    def publish_info_payload(self, payload):
        """Publish retained info so MQTT discovery sees the current frontend."""
        self.mqtt.publish(self.info_topic, json.dumps(payload), retain=True)

    def publish_stats_payload(self, payload):
        """Publish stats payload."""
        self.mqtt.publish(self.stats_topic, json.dumps(payload))
    
    def cmd_info(self):
        """Get system information"""
        ip_address = ""
        mac_address = ""
        channel_primary = 0
        chip = getattr(self.global_state, 'chip_type', None) or sys.platform
        
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
            "device_name": _protocol_device_name(self.config.MQTT_CLIENT_ID, chip),
            "device_label": getattr(self.config, "MQTT_DEVICE_LABEL", ""),
            "frontend": "micro",
            "firmware_version": "micropython",
            "chip": chip,
            "supports_info": True,
            "supports_stats": True,
            "supports_runtime_threshold": True,
            "supports_runtime_motion_hits": self.runtime_policy is not None,
            "supports_runtime_detector": False,
            "supports_manual_recalibration": callable(self.recalibrate_callback),
            "supports_traffic_control": self.traffic_control_supported,
            "supports_ota": False,
            "supports_ble": False,
            "network": {
                "ip_address": ip_address,
                "mac_address": mac_address,
                "channel": {
                    "primary": channel_primary
                }
            },
            "detection": self._get_detection_info(),
            "csi_traffic_mode": getattr(self.ha_adapter, "_last_csi_traffic_mode", "internal"),
            "traffic_mode": getattr(self.ha_adapter, "_last_traffic_generator_mode", "ping"),
            "csi_target_pps": max(1, int(getattr(self.config, "CSI_TARGET_PPS", 100))),
        }
        
        self.publish_info_payload(response)

    def cmd_commands(self):
        """Publish the MQTT command catalog for this frontend."""
        payload = {
            "protocol_version": "1.0",
            "device_id": self.config.MQTT_CLIENT_ID,
            "commands": _protocol_mqtt_commands(
                supports_info=True,
                supports_stats=True,
                supports_runtime_threshold=True,
                supports_runtime_motion_hits=self.runtime_policy is not None,
                supports_runtime_detector=False,
                supports_manual_recalibration=callable(self.recalibrate_callback),
                supports_traffic_control=self.traffic_control_supported,
                supports_ble=False,
                supports_ota=False,
            ),
        }
        if self.catalog_topic:
            self.mqtt.publish(self.catalog_topic, json.dumps(payload))
    
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
            "timestamp_ms": int(current_time * 1000),
            "uptime": int(uptime_sec),
            "free_memory_kb": free_mem_kb,
            "loop_time_ms": loop_time_ms,
        }
        wifi_channel = 0
        cached = None
        if self.global_state is not None:
            wifi_channel = int(getattr(self.global_state, "current_channel", 0) or 0)
            cached = getattr(self.global_state, "latest_diagnostics", None)
        apply_diagnostics_sample(
            response,
            cached,
            wifi_channel=wifi_channel,
            rssi_dbm=wifi_rssi_dbm(self.wlan),
        )

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
            
            threshold_min, threshold_max = _threshold_bounds_for_detector(self.detector)

            if threshold < threshold_min or threshold > threshold_max:
                self.send_response(
                    f"ERROR: Threshold must be between {threshold_min} and {threshold_max}",
                    accepted=False,
                    command_id=command_id,
                    command=command
                )
                return
            
            old_threshold = self.detector.get_threshold()
            if not self.detector.set_threshold(threshold):
                self.send_response(
                    f"ERROR: Threshold rejected by detector (allowed range: {threshold_min}-{threshold_max})",
                    accepted=False,
                    command_id=command_id,
                    command=command
                )
                return
            
            # Note: threshold is session-only, startup threshold is recalculated on every boot
            
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

    def cmd_set_motion_hits(self, cmd_obj):
        """Set runtime motion-hit debounce counts (session-only)."""
        command_id = cmd_obj.get('command_id', '')
        command = cmd_obj.get('command', 'set_motion_hits')
        if self.runtime_policy is None:
            self.send_response("ERROR: Motion hit updates are unsupported", accepted=False, command_id=command_id, command=command)
            return
        if 'motion_on_hits' not in cmd_obj or 'motion_off_hits' not in cmd_obj:
            self.send_response(
                "ERROR: Missing motion hit fields (accepted: motion_on_hits and motion_off_hits in {}-{})".format(
                    MOTION_HITS_MIN,
                    MOTION_HITS_MAX,
                ),
                accepted=False,
                command_id=command_id,
                command=command,
            )
            return
        try:
            motion_on_hits = int(cmd_obj['motion_on_hits'])
            motion_off_hits = int(cmd_obj['motion_off_hits'])
        except (TypeError, ValueError):
            self.send_response("ERROR: Invalid motion hit value", accepted=False, command_id=command_id, command=command)
            return
        if not (MOTION_HITS_MIN <= motion_on_hits <= MOTION_HITS_MAX) or not (MOTION_HITS_MIN <= motion_off_hits <= MOTION_HITS_MAX):
            self.send_response(
                "ERROR: Motion hits must be between {} and {}".format(MOTION_HITS_MIN, MOTION_HITS_MAX),
                accepted=False,
                command_id=command_id,
                command=command,
            )
            return
        self.runtime_policy.motion_on_hits = motion_on_hits
        self.runtime_policy.motion_off_hits = motion_off_hits
        if self.ha_adapter is not None:
            self.ha_adapter.set_motion_hits(motion_on_hits, motion_off_hits)
            self.ha_adapter.publish_motion_hits(self.mqtt, motion_on_hits, motion_off_hits, force=True)
        self.send_response(
            "Motion hits updated: on={} off={} (session-only)".format(motion_on_hits, motion_off_hits),
            accepted=True,
            command_id=command_id,
            command=command,
        )
        self.cmd_info()

    def cmd_recalibrate(self, cmd_obj):
        """Queue a recalibration request for the main loop."""
        command_id = cmd_obj.get('command_id', '')
        command = cmd_obj.get('command', 'recalibrate')
        if not callable(self.recalibrate_callback):
            self.send_response("ERROR: Recalibration is unsupported", accepted=False, command_id=command_id, command=command)
            return
        if not self.recalibrate_callback():
            self.send_response(
                "ERROR: Recalibration already pending or active",
                accepted=False,
                command_id=command_id,
                command=command,
            )
            return
        self.send_response("recalibration started", accepted=True, command_id=command_id, command=command)

    def cmd_set_csi_traffic_mode(self, cmd_obj):
        """Set the live CSI traffic ownership mode (session-only)."""
        command_id = cmd_obj.get('command_id', '')
        command = cmd_obj.get('command', 'set_csi_traffic_mode')
        mode = str(cmd_obj.get('csi_traffic_mode', '')).strip().lower()
        if mode not in ("internal", "external", "pacing", "disabled"):
            self.send_response(
                "ERROR: Invalid CSI traffic mode (accepted: internal, external, pacing, and disabled)",
                accepted=False,
                command_id=command_id,
                command=command,
            )
            return
        callback = self.traffic_control_callback
        if not self.traffic_control_supported or not callable(callback):
            self.send_response("ERROR: Traffic control is unsupported", accepted=False, command_id=command_id, command=command)
            return
        generator_mode = getattr(self.ha_adapter, "_last_traffic_generator_mode", "ping")
        if not callback(mode, generator_mode):
            message = "ERROR: CSI traffic mode unsupported" if mode == "pacing" else "ERROR: CSI traffic mode rejected"
            self.send_response(message, accepted=False, command_id=command_id, command=command)
            return
        self.send_response(
            "CSI traffic mode updated: {} (session-only)".format(mode),
            accepted=True,
            command_id=command_id,
            command=command,
        )
        self.cmd_info()

    def cmd_set_traffic_generator_mode(self, cmd_obj):
        """Set the live internal traffic generator packet type (session-only)."""
        command_id = cmd_obj.get('command_id', '')
        command = cmd_obj.get('command', 'set_traffic_generator_mode')
        mode = str(cmd_obj.get('traffic_generator_mode', '')).strip().lower()
        if mode not in ("ping", "dns"):
            self.send_response(
                "ERROR: Invalid traffic generator mode (accepted: ping and dns)",
                accepted=False,
                command_id=command_id,
                command=command,
            )
            return
        callback = self.traffic_control_callback
        if not self.traffic_control_supported or not callable(callback):
            self.send_response("ERROR: Traffic control is unsupported", accepted=False, command_id=command_id, command=command)
            return
        csi_mode = getattr(self.ha_adapter, "_last_csi_traffic_mode", "internal")
        if not callback(csi_mode, mode):
            self.send_response("ERROR: Traffic generator mode rejected", accepted=False, command_id=command_id, command=command)
            return
        self.send_response(
            "Traffic generator mode updated: {} (session-only)".format(mode),
            accepted=True,
            command_id=command_id,
            command=command,
        )
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
            elif command == 'commands':
                self.cmd_commands()
                self.send_response("commands published", accepted=True, command_id=command_id, command=command)
            elif command == 'stats':
                self.cmd_stats()
                self.send_response("stats published", accepted=True, command_id=command_id, command=command)
            elif command == 'set_threshold':
                self.cmd_set_threshold(cmd_obj)
            elif command == 'set_motion_hits':
                self.cmd_set_motion_hits(cmd_obj)
            elif command == 'set_csi_traffic_mode':
                self.cmd_set_csi_traffic_mode(cmd_obj)
            elif command == 'set_traffic_generator_mode':
                self.cmd_set_traffic_generator_mode(cmd_obj)
            elif command == 'recalibrate':
                self.cmd_recalibrate(cmd_obj)
            else:
                self.send_response(f"ERROR: Unknown command '{command}'", accepted=False, command_id=command_id, command=command)
                
        except Exception as e:
            error_msg = f"ERROR: Command processing failed: {e}"
            print(error_msg)
            self.send_response(error_msg, accepted=False)
