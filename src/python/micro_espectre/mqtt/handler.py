# SPDX-License-Identifier: GPL-3.0-only
# Commercial licensing available under separate agreement; see LICENSING.md.
"""
Micro-ESPectre - MQTT Handler Module

Handles MQTT communication and command processing.
Manages connection, publishing state updates, and processing remote commands.

Author: Francesco Pace <francesco.pace@gmail.com>
"""
import json
import time

from espectre_native_mqtt import MQTTClient

try:
    from src.detector_interface import get_detector_algorithm
    import src.mqtt.protocol as mqtt_protocol
except ImportError:
    from detector_interface import get_detector_algorithm
    import mqtt.protocol as mqtt_protocol

MQTT_RECONNECT_INITIAL_MS = 1000
MQTT_RECONNECT_MAX_MS = 60000
NATIVE_MQTT_IDLE = 0
NATIVE_MQTT_CONNECTING = 1
NATIVE_MQTT_CONNECTED = 2
NATIVE_MQTT_FAILED = 3


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


class _DisabledHomeAssistantAdapter:
    """Allocation-light HA adapter used on constrained runtimes."""

    enabled = False

    def __init__(self, config):
        self._last_csi_traffic_mode = (
            "internal" if getattr(config, "TRAFFIC_GENERATOR_ENABLED", True) else "external"
        )
        self._last_traffic_generator_mode = str(
            getattr(config, "TRAFFIC_GENERATOR_MODE", "ping")
        ).lower()

    @staticmethod
    def _noop(*_args, **_kwargs):
        return None

    def __getattr__(self, _name):
        return self._noop

    def set_traffic_control(self, csi_traffic_mode, traffic_generator_mode):
        self._last_csi_traffic_mode = str(csi_traffic_mode).lower()
        self._last_traffic_generator_mode = str(traffic_generator_mode).lower()


class MQTTHandler:
    """MQTT handler with publishing and command support"""

    def __init__(self, config, detector, wlan, global_state=None, runtime_policy=None, traffic_generator=None):
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
        self.runtime_policy = runtime_policy
        self.traffic_generator = traffic_generator
        self.device_id = mqtt_protocol.derive_runtime_device_id(wlan)
        self.csi_target_pps = max(1, int(getattr(config, "CSI_TARGET_PPS", 100)))
        self.csi_traffic_mode = "internal" if getattr(config, "TRAFFIC_GENERATOR_ENABLED", True) else "external"
        self.traffic_generator_mode = str(getattr(config, "TRAFFIC_GENERATOR_MODE", "ping")).lower()
        if getattr(config, "MQTT_HA_DISCOVERY_ENABLED", False):
            try:
                from src.mqtt.home_assistant import HomeAssistantMqttAdapter
            except ImportError:
                from mqtt.home_assistant import HomeAssistantMqttAdapter
            self.ha_adapter = HomeAssistantMqttAdapter(
                config,
                detector,
                wlan,
                global_state,
                device_id=self.device_id,
            )
        else:
            self.ha_adapter = _DisabledHomeAssistantAdapter(config)
        self.ha_adapter.set_calibrate_handler(self.request_recalibration)
        self.ha_adapter.set_motion_hits_handler(self.set_motion_hits)
        self.ha_adapter.set_traffic_control_handler(self.set_traffic_control)
        if runtime_policy is not None:
            self.ha_adapter.set_motion_hits(runtime_policy.motion_on_hits, runtime_policy.motion_off_hits)
        self.ha_adapter.set_traffic_control(self.csi_traffic_mode, self.traffic_generator_mode)
        self.connected = False
        self._stopping = False
        self._recalibrate_requested = False
        self._next_reconnect_ms = 0
        self._reconnect_backoff_ms = MQTT_RECONNECT_INITIAL_MS
        self._has_connected_once = False
        self.start_time = time.time()

        # ESPectre Protocol topics
        topic_prefix = config.MQTT_TOPIC_PREFIX.rstrip('/')
        self.base_topic = f"{topic_prefix}/{self.device_id}"
        self.telemetry_topic = f"{self.base_topic}/telemetry"
        self.status_topic = f"{self.base_topic}/status"
        self.info_topic = f"{self.base_topic}/info"
        self.stats_topic = f"{self.base_topic}/stats"
        self.catalog_topic = f"{self.base_topic}/commands/catalog"
        self.cmd_topic = f"{self.base_topic}/commands/request"
        self.accepted_topic = f"{self.base_topic}/commands/accepted"
        self.rejected_topic = f"{self.base_topic}/commands/rejected"

        # Publishing state
        self.last_variance = 0.0
        self.last_state = 0  # STATE_IDLE
        self.last_threshold = 0.0
        get_threshold = getattr(detector, "get_threshold", None)
        if callable(get_threshold):
            try:
                self.last_threshold = float(get_threshold())
            except (TypeError, ValueError):
                self.last_threshold = 0.0

    def set_runtime_policy(self, runtime_policy):
        """Attach the live motion policy once the main loop creates it."""
        self.runtime_policy = runtime_policy
        if runtime_policy is not None:
            self.ha_adapter.set_motion_hits(runtime_policy.motion_on_hits, runtime_policy.motion_off_hits)
        if self.cmd_handler is not None:
            self.cmd_handler.runtime_policy = runtime_policy
        if self.connected:
            self.publish_info()

    def _start_internal_traffic(self):
        """Ensure the local traffic generator is running with the selected mode."""
        if self.traffic_generator is None:
            return False
        if self.traffic_generator.is_running():
            self.traffic_generator.stop()
            deadline = _ticks_add(_ticks_ms(), 1000)
            sleep_ms = getattr(time, "sleep_ms", None)
            while (
                getattr(self.traffic_generator, "sock", None) is not None
                and _ticks_diff(deadline, _ticks_ms()) > 0
            ):
                if sleep_ms is not None:
                    sleep_ms(10)
                else:
                    time.sleep(0.01)
        if not self.traffic_generator.set_mode(self.traffic_generator_mode):
            return False
        return bool(self.traffic_generator.start(self.csi_target_pps))

    def set_traffic_control(self, csi_traffic_mode, traffic_generator_mode):
        """Apply one session-only traffic ownership and generator update."""
        csi_mode = str(csi_traffic_mode).lower()
        generator_mode = str(traffic_generator_mode).lower()
        if csi_mode == "pacing":
            return False
        if csi_mode not in ("internal", "external", "disabled"):
            return False
        if generator_mode not in ("ping", "dns"):
            return False
        if self.traffic_generator is None:
            return False

        previous_csi_mode = self.csi_traffic_mode
        previous_generator_mode = self.traffic_generator_mode
        self.csi_traffic_mode = csi_mode
        self.traffic_generator_mode = generator_mode

        if csi_mode == "internal":
            if not self._start_internal_traffic():
                self.csi_traffic_mode = previous_csi_mode
                self.traffic_generator_mode = previous_generator_mode
                return False
        else:
            self.traffic_generator.stop()

        self.ha_adapter.set_traffic_control(self.csi_traffic_mode, self.traffic_generator_mode)
        if self.connected:
            self.ha_adapter.publish_traffic_control(
                self.client,
                self.csi_traffic_mode,
                self.traffic_generator_mode,
                force=True,
            )
        return True

    def connect(self):
        """Connect to MQTT broker"""
        self._stopping = False
        try:
            self._connect_client()
            return self.client
        except Exception as e:
            print(f"MQTT connection failed: {e}")
            self._discard_client()
            self._schedule_connect_retry()
            return None

    def _create_command_handler(self):
        """Load the command processor only while a command or info request runs."""
        try:
            from src.mqtt.commands import MQTTCommands
        except ImportError:
            from mqtt.commands import MQTTCommands

        handler = MQTTCommands(
            self.client,
            self.config,
            self.detector,
            self.accepted_topic,
            self.rejected_topic,
            self.info_topic,
            self.stats_topic,
            self.wlan,
            self.global_state,
            runtime_policy=self.runtime_policy,
            ha_adapter=self.ha_adapter,
            recalibrate_callback=self.request_recalibration,
            traffic_control_callback=self.set_traffic_control,
            traffic_control_supported=self.traffic_generator is not None,
            catalog_topic=self.catalog_topic,
            device_id=self.device_id,
        )
        handler.start_time = self.start_time
        return handler

    @staticmethod
    def _release_command_module():
        """Release command bytecode after transient use on MicroPython."""
        try:
            import gc
            import sys

            if getattr(sys.implementation, "name", "") != "micropython":
                return
            sys.modules.pop("src.mqtt.commands", None)
            sys.modules.pop("mqtt.commands", None)
            gc.collect()
        except (AttributeError, ImportError):
            pass

    def _prepare_client(self):
        """Allocate and configure the transport without starting its task."""
        if self.client is not None:
            return self.client
        broker = self.config.MQTT_BROKER
        # MicroPython's socket resolver adds `.local` mDNS handling that the
        # ESP-IDF MQTT task does not provide. Resolve once before the native
        # task starts, then release the transient result objects.
        import socket
        address_info = socket.getaddrinfo(
            broker,
            self.config.MQTT_PORT,
            socket.AF_INET,
            socket.SOCK_STREAM,
        )
        broker = address_info[0][-1][0]
        del address_info
        client_args = {
            "port": self.config.MQTT_PORT,
            "user": self.config.MQTT_USERNAME,
            "password": self.config.MQTT_PASSWORD,
            "last_will_topic": self.status_topic,
            "last_will_msg": json.dumps(self._status_payload(False)),
            "last_will_retain": True,
        }
        self.client = MQTTClient(self.device_id, broker, **client_args)
        self.ha_adapter.configure_client(self.client)
        return self.client

    def prepare(self):
        """Reserve transport memory before detector calibration fragments heap."""
        try:
            return self._prepare_client()
        except Exception as e:
            print(f"MQTT preparation failed: {e}")
            self.client = None
            return None

    def _connect_client(self):
        """Connect, subscribe, and announce one MQTT client."""
        self._prepare_client()

        print('Connecting to MQTT broker...')
        self.client.connect()
        # ESP-IDF owns the socket and reports completion through status(). Do
        # not block the MicroPython VM while the broker is unavailable.
        self._next_reconnect_ms = 0

    def _finish_client_setup(self):
        """Subscribe and publish the session snapshot after transport connect."""
        if self.connected:
            return False
        is_reconnect = self._has_connected_once
        print('MQTT connected')

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
        self.ha_adapter.publish_snapshot(
            self.client, self.last_variance, self.last_state, self.last_threshold
        )
        if not self.connected:
            raise OSError("MQTT connection lost during session setup")
        self._has_connected_once = True
        return is_reconnect

    def _mark_disconnected(self):
        """Let the native ESP-IDF task recover a transport failure."""
        self.connected = False

    def _schedule_connect_retry(self):
        """Back off after failing to start a native MQTT task."""
        self.connected = False
        self._next_reconnect_ms = _ticks_add(
            _ticks_ms(), self._reconnect_backoff_ms
        )
        self._reconnect_backoff_ms = min(
            self._reconnect_backoff_ms * 2,
            MQTT_RECONNECT_MAX_MS,
        )

    def _discard_client(self):
        """Release a client whose native task could not be started."""
        if self.client:
            try:
                self.client.deinit()
            except Exception:
                pass
        self.client = None

    def _reconnect_if_due(self):
        """Attempt one reconnect when the backoff deadline has elapsed."""
        if self._stopping or self.connected:
            return self.connected
        now = _ticks_ms()

        if self.client is not None:
            try:
                native_status = self.client.status()
                if native_status == NATIVE_MQTT_CONNECTED:
                    if self._finish_client_setup():
                        print("MQTT reconnected")
                    return True
                if native_status == NATIVE_MQTT_CONNECTING:
                    return False
                if native_status == NATIVE_MQTT_FAILED:
                    # The native ESP-IDF transport retries in its own task.
                    return False
            except Exception as e:
                print(f"MQTT status failed: {e}")
                self._mark_disconnected()
                return False

        if self._next_reconnect_ms and _ticks_diff(now, self._next_reconnect_ms) < 0:
            return False
        try:
            print("Reconnecting to MQTT broker...")
            self._connect_client()
            return False
        except Exception as e:
            print(f"MQTT reconnect failed: {e}")
            self._discard_client()
            self._schedule_connect_retry()
            return False

    def _on_message(self, topic, msg):
        """Callback for incoming MQTT messages"""
        try:
            topic_str = topic.decode('utf-8') if isinstance(topic, bytes) else topic

            if topic_str == self.cmd_topic:
                transient_handler = self.cmd_handler is None
                command_handler = self.cmd_handler or self._create_command_handler()
                try:
                    command_handler.process_command(msg)
                    self._sync_ha_threshold_from_detector()
                finally:
                    if transient_handler:
                        del command_handler
                        self._release_command_module()
            else:
                self.ha_adapter.handle_message(self.client, topic_str, msg)
                getter = getattr(self.detector, "get_threshold", None)
                if callable(getter):
                    try:
                        self.last_threshold = float(getter())
                    except (TypeError, ValueError):
                        pass

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
            current_variance: Current motion metric on the shared probability scale
            current_state: Current state (0=IDLE, 1=MOTION)
            current_threshold: Current threshold
        """
        if not self.connected:
            self.ha_adapter.record_state(current_variance, current_state, current_threshold)
            self.last_variance = current_variance
            self.last_state = current_state
            self.last_threshold = current_threshold
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
            'detector': get_detector_algorithm(self.detector),
            'health': health
        }

        try:
            self.client.publish(self.telemetry_topic, json.dumps(payload))
            self.ha_adapter.publish_movement(self.client, current_variance)
            self.ha_adapter.record_state(current_variance, current_state, current_threshold)
        except Exception as e:
            print(f"Error publishing to MQTT: {e}")
            self._mark_disconnected()

        # Update state
        self.last_variance = current_variance
        self.last_state = current_state
        self.last_threshold = current_threshold

    def publish_live_ha(self, current_variance, current_state, current_threshold):
        """Publish telemetry and HA movement every evaluation; motion only on edges."""
        self.publish_state(current_variance, current_state, current_threshold)
        if not self.connected:
            return
        try:
            self.ha_adapter.publish_threshold(self.client, current_threshold)
            self.ha_adapter.publish_motion(self.client, current_state)
        except Exception as e:
            print(f"Error publishing HA live state: {e}")
            self._mark_disconnected()

    def _sync_ha_threshold_from_detector(self):
        """Publish the HA threshold entity when a protocol command changes it."""
        getter = getattr(self.detector, "get_threshold", None)
        if not callable(getter):
            return
        try:
            threshold = float(getter())
        except (TypeError, ValueError):
            return
        if threshold == self.last_threshold:
            return
        self.last_threshold = threshold
        if self.connected:
            self.ha_adapter.publish_threshold(self.client, threshold, force=True)

    def request_recalibration(self):
        """Queue a main-loop recalibration from the HA Calibrate switch."""
        if self._recalibrate_requested or self.ha_adapter.is_calibrating():
            self.publish_calibrate_state(True)
            return False
        self._recalibrate_requested = True
        self.publish_calibrate_state(True)
        return True

    def take_recalibrate_request(self):
        """Consume one pending HA recalibration request for the main loop."""
        requested = self._recalibrate_requested
        self._recalibrate_requested = False
        return requested

    def publish_calibrate_state(self, calibrating):
        """Publish the HA Calibrate switch, or cache the state while offline."""
        if not self.connected:
            self.ha_adapter.set_calibrating(calibrating)
            return
        self.ha_adapter.publish_calibrate(self.client, calibrating, force=True)

    def set_motion_hits(self, motion_on_hits, motion_off_hits):
        """Apply one session-only motion-hit update."""
        if self.runtime_policy is None:
            return False
        self.runtime_policy.motion_on_hits = int(motion_on_hits)
        self.runtime_policy.motion_off_hits = int(motion_off_hits)
        self.ha_adapter.set_motion_hits(self.runtime_policy.motion_on_hits, self.runtime_policy.motion_off_hits)
        if self.connected:
            self.ha_adapter.publish_motion_hits(
                self.client,
                self.runtime_policy.motion_on_hits,
                self.runtime_policy.motion_off_hits,
                force=True,
            )
        return True

    def finish_recalibration(self, movement_score, threshold):
        """Publish HA Calibrate OFF and refresh threshold-linked entities."""
        self.last_variance = float(movement_score)
        self.last_state = 0
        self.last_threshold = float(threshold)
        self.ha_adapter.record_state(self.last_variance, self.last_state, self.last_threshold)
        if not self.connected:
            self.ha_adapter.set_calibrating(False)
            return
        self.ha_adapter.publish_calibrate(self.client, False, force=True)
        self.ha_adapter.publish_threshold(self.client, self.last_threshold, force=True)
        if self.runtime_policy is not None:
            self.ha_adapter.publish_motion_hits(
                self.client,
                self.runtime_policy.motion_on_hits,
                self.runtime_policy.motion_off_hits,
                force=True,
            )
        self.ha_adapter.publish_motion(self.client, self.last_state, force=True)

    def disconnect(self):
        """Disconnect from MQTT broker"""
        self._stopping = True
        if self.client:
            try:
                self.publish_status(False)
                self.client.deinit()
                print('MQTT disconnected')
            except Exception as e:
                print(f"Error disconnecting MQTT: {e}")
        self.client = None
        self.connected = False

    def publish_info(self):
        """Publish system info"""
        if self.cmd_handler:
            self.cmd_handler.cmd_info()
            return
        if self.client is None:
            return
        try:
            from src.mqtt.protocol import build_info_payload
        except ImportError:
            from mqtt.protocol import build_info_payload
        payload = build_info_payload(
            self.config,
            get_detector_algorithm(self.detector),
            self.wlan,
            self.global_state,
            self.runtime_policy,
            self.ha_adapter,
            self.request_recalibration,
            self.traffic_generator is not None,
        )
        self.client.publish(self.info_topic, json.dumps(payload), retain=True)

    def publish_status(self, online):
        """Publish live online/offline status."""
        if not self.client:
            return
        payload = self._status_payload(online)
        try:
            self.client.publish(self.status_topic, json.dumps(payload), retain=True)
            self.ha_adapter.publish_availability(self.client, online)
        except Exception as e:
            print(f"Error publishing MQTT status: {e}")
            if not self._stopping:
                self._mark_disconnected()

    def _status_payload(self, online):
        """Build the shared retained availability payload."""
        return {
            "protocol_version": "1.0",
            "device_id": self.device_id,
            "online": bool(online),
            "timestamp_ms": int(time.time() * 1000)
        }
