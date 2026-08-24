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

try:
    from src.detector_interface import get_detector_algorithm
    from src.config import MOTION_HITS_MAX, MOTION_HITS_MIN
    from src.runtime_diagnostics import apply_diagnostics_sample, wifi_rssi_dbm
    import src.mqtt.protocol as mqtt_protocol
    from src.mqtt.protocol import (
        THRESHOLD_MAX,
        THRESHOLD_MIN,
        build_capabilities_payload,
        build_info_payload,
    )
except ImportError:
    from detector_interface import get_detector_algorithm
    from config import MOTION_HITS_MAX, MOTION_HITS_MIN
    from runtime_diagnostics import apply_diagnostics_sample, wifi_rssi_dbm
    import mqtt.protocol as mqtt_protocol
    from mqtt.protocol import (
        THRESHOLD_MAX,
        THRESHOLD_MIN,
        build_capabilities_payload,
        build_info_payload,
    )


def _threshold_bounds_for_detector(detector):
    """Return the accepted threshold range for the active detector."""
    return THRESHOLD_MIN, THRESHOLD_MAX


class MQTTCommands:
    """MQTT command processor"""
    
    def __init__(self,
                 mqtt_client,
                 config,
                 detector,
                 result_topic,
                 info_topic,
                 wlan,
                 global_state=None,
                 runtime_policy=None,
                 ha_adapter=None,
                 recalibrate_callback=None,
                 traffic_control_callback=None,
                 traffic_control_supported=False,
                 catalog_topic=None,
                 device_id=None):
        """
        Initialize MQTT commands
        
        Args:
            mqtt_client: MQTT client instance
            config: Configuration module
            detector: IDetector instance
            result_topic: MQTT topic for correlated command results
            info_topic: MQTT topic for live system info
            wlan: wlan instance
            global_state: GlobalState instance for accessing loop metrics (optional)
        """
        self.mqtt = mqtt_client
        self.config = config
        self.device_id = device_id or mqtt_protocol.derive_runtime_device_id(wlan)
        self.detector = detector
        self.wlan = wlan
        self.global_state = global_state
        self.result_topic = result_topic
        self.info_topic = info_topic
        self.config_topic = info_topic[: -len("/info")] + "/config" if info_topic.endswith("/info") else ""
        self.status_topic = info_topic[: -len("/info")] + "/status" if info_topic.endswith("/info") else ""
        if catalog_topic:
            self.catalog_topic = catalog_topic
        elif info_topic.endswith("/info"):
            self.catalog_topic = info_topic[: -len("/info")] + "/capabilities"
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
        
    def send_response(self, message, accepted=True, command_id="", command="", code=None, data=None):
        """Send response message to MQTT"""
        try:
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
                payload.setdefault("device_id", self.device_id)
                if command_id:
                    payload.setdefault("command_id", command_id)
                if command:
                    payload.setdefault("command", command)
                payload.setdefault("accepted", bool(accepted))
                payload.setdefault("code", code or ("ok" if accepted else "invalid_params"))
                if data is not None:
                    payload["data"] = data
            
            self.mqtt.publish(self.result_topic, json.dumps(payload))
            if accepted and command.startswith("set_") and self.config_topic:
                self.mqtt.publish(self.config_topic, json.dumps(self.build_config_payload()), retain=True)
            elif accepted and command == "recalibrate" and self.status_topic:
                self.mqtt.publish(self.status_topic, json.dumps(self.build_status_payload()), retain=True)
        except Exception as e:
            print(f"Error sending MQTT response: {e}")

    def publish_info_payload(self, payload):
        """Publish retained info so MQTT discovery sees the current frontend."""
        self.mqtt.publish(self.info_topic, json.dumps(payload), retain=True)

    def cmd_info(self):
        """Get system information"""
        return build_info_payload(
            self.config,
            get_detector_algorithm(self.detector),
            self.wlan,
            self.global_state,
            self.runtime_policy,
            self.ha_adapter,
            self.recalibrate_callback,
            self.traffic_control_supported,
            device_id=self.device_id,
        )

    def cmd_capabilities(self):
        """Return the filtered canonical command catalog."""
        return build_capabilities_payload(
            self.device_id,
            supports_info=True,
            supports_diagnostics=True,
            supports_device_config=False,
            supports_runtime_threshold=True,
            supports_runtime_motion_hits=self.runtime_policy is not None,
            supports_runtime_detector=False,
            supports_manual_recalibration=callable(self.recalibrate_callback),
            supports_traffic_control=self.traffic_control_supported,
            supports_ota=False,
        )
    
    def cmd_diagnostics(self):
        """Return runtime diagnostics."""
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
            "device_id": self.device_id,
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

        return response

    def build_status_payload(self):
        return {
            "protocol_version": "1.0",
            "device_id": self.device_id,
            "online": True,
            "sensing_enabled": True,
            "ready_to_publish": bool(getattr(self.global_state, "ready_to_publish", False)),
            "calibrating": bool(self.ha_adapter is not None and self.ha_adapter.is_calibrating()),
        }

    def build_config_payload(self):
        runtime = {
            "threshold": float(self.detector.get_threshold()),
            "detector": get_detector_algorithm(self.detector),
            "csi_traffic_mode": getattr(self.ha_adapter, "_last_csi_traffic_mode", "internal"),
            "traffic_generator_mode": getattr(self.ha_adapter, "_last_traffic_generator_mode", "ping"),
        }
        if self.runtime_policy is not None:
            runtime["motion_on_hits"] = int(self.runtime_policy.motion_on_hits)
            runtime["motion_off_hits"] = int(self.runtime_policy.motion_off_hits)
        return {"protocol_version": "1.0", "device_id": self.device_id, "runtime": runtime}
    
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
                    command=command,
                    code="unavailable",
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

    def cmd_set_motion_hits(self, cmd_obj):
        """Set runtime motion-hit debounce counts (session-only)."""
        command_id = cmd_obj.get('command_id', '')
        command = cmd_obj.get('command', 'set_motion_hits')
        if self.runtime_policy is None:
            self.send_response("ERROR: Motion hit updates are unsupported", accepted=False, command_id=command_id, command=command, code="unsupported")
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

    def cmd_recalibrate(self, cmd_obj):
        """Queue a recalibration request for the main loop."""
        command_id = cmd_obj.get('command_id', '')
        command = cmd_obj.get('command', 'recalibrate')
        if not callable(self.recalibrate_callback):
            self.send_response("ERROR: Recalibration is unsupported", accepted=False, command_id=command_id, command=command, code="unsupported")
            return
        if not self.recalibrate_callback():
            self.send_response(
                "ERROR: Recalibration already pending or active",
                accepted=False,
                command_id=command_id,
                command=command,
                code="busy",
            )
            return
        self.send_response("recalibration started", accepted=True, command_id=command_id, command=command)

    def cmd_set_csi_traffic_mode(self, cmd_obj):
        """Set the live CSI traffic ownership mode (session-only)."""
        command_id = cmd_obj.get('command_id', '')
        command = cmd_obj.get('command', 'set_csi_traffic_mode')
        mode = str(cmd_obj.get('csi_traffic_mode', '')).strip().lower()
        if mode not in ("internal", "external", "disabled"):
            self.send_response(
                "ERROR: Invalid CSI traffic mode (accepted: internal, external, and disabled)",
                accepted=False,
                command_id=command_id,
                command=command,
            )
            return
        callback = self.traffic_control_callback
        if not self.traffic_control_supported or not callable(callback):
            self.send_response("ERROR: Traffic control is unsupported", accepted=False, command_id=command_id, command=command, code="unsupported")
            return
        generator_mode = getattr(self.ha_adapter, "_last_traffic_generator_mode", "ping")
        if not callback(mode, generator_mode):
            self.send_response("ERROR: CSI traffic mode rejected", accepted=False, command_id=command_id, command=command, code="unavailable")
            return
        self.send_response(
            "CSI traffic mode updated: {} (session-only)".format(mode),
            accepted=True,
            command_id=command_id,
            command=command,
        )

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
            self.send_response("ERROR: Traffic control is unsupported", accepted=False, command_id=command_id, command=command, code="unsupported")
            return
        csi_mode = getattr(self.ha_adapter, "_last_csi_traffic_mode", "internal")
        if not callback(csi_mode, mode):
            self.send_response("ERROR: Traffic generator mode rejected", accepted=False, command_id=command_id, command=command, code="unavailable")
            return
        self.send_response(
            "Traffic generator mode updated: {} (session-only)".format(mode),
            accepted=True,
            command_id=command_id,
            command=command,
        )
    
    @staticmethod
    def _validate_params(cmd_obj, descriptor):
        """Validate the canonical JSON Schema subset without CPython-only dependencies."""
        schema = descriptor["params"]
        properties = schema.get("properties", {})
        allowed = {"protocol_version", "command_id", "command"}
        allowed.update(properties)
        if schema.get("additionalProperties") is False:
            unknown = [key for key in cmd_obj if key not in allowed]
            if unknown:
                return False, "unknown command parameter"
        for name in schema.get("required", []):
            if name not in cmd_obj:
                return False, "missing required parameter: {}".format(name)
        for name, value_schema in properties.items():
            if name not in cmd_obj:
                continue
            value = cmd_obj[name]
            expected = value_schema.get("type")
            valid_type = (
                (expected == "boolean" and isinstance(value, bool))
                or (expected == "string" and isinstance(value, str))
                or (expected == "integer" and isinstance(value, int) and not isinstance(value, bool))
                or (expected == "number" and isinstance(value, (int, float)) and not isinstance(value, bool))
            )
            if not valid_type:
                return False, "invalid parameter type: {}".format(name)
            if "enum" in value_schema and value not in value_schema["enum"]:
                return False, "invalid parameter value: {}".format(name)
            if "minimum" in value_schema and value < value_schema["minimum"]:
                return False, "parameter below minimum: {}".format(name)
            if "maximum" in value_schema and value > value_schema["maximum"]:
                return False, "parameter above maximum: {}".format(name)
        return True, ""

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
            descriptors = {item["name"]: item for item in self.cmd_capabilities()["commands"]}
            descriptor = descriptors.get(command)
            if descriptor is None:
                self.send_response(f"ERROR: Unknown command '{command}'", accepted=False, command_id=command_id, command=command, code="unsupported")
                return
            valid, validation_message = self._validate_params(cmd_obj, descriptor)
            if not valid:
                self.send_response(validation_message, accepted=False, command_id=command_id, command=command, code="invalid_params")
                return
            #print(f"Processing MQTT command: {command}")
            
            query_handlers = {
                'capabilities': self.cmd_capabilities,
                'info': self.cmd_info,
                'status': self.build_status_payload,
                'config': self.build_config_payload,
                'diagnostics': self.cmd_diagnostics,
            }
            if command in query_handlers:
                self.send_response(
                    "{} returned".format(command),
                    accepted=True,
                    command_id=command_id,
                    command=command,
                    data=query_handlers[command](),
                )
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
                self.send_response("ERROR: Command has no dispatcher", accepted=False, command_id=command_id, command=command, code="internal_error")
                
        except Exception as e:
            error_msg = f"ERROR: Command processing failed: {e}"
            print(error_msg)
            self.send_response(error_msg, accepted=False, code="internal_error")
