# SPDX-License-Identifier: GPL-3.0-only
# Commercial licensing available under separate agreement; see LICENSING.md.
"""
Micro-ESPectre - Home Assistant MQTT adapter.

Publishes Home Assistant MQTT Discovery and entity-shaped state topics while
preserving the canonical ESPectre protocol topics.

Author: Francesco Pace <francesco.pace@gmail.com>
"""

try:
    from src.config import MOTION_HITS_MAX, MOTION_HITS_MIN
    import src.mqtt.protocol as mqtt_protocol
    from src.mqtt.protocol import THRESHOLD_MAX, THRESHOLD_MIN, _normalize_chip_label, _protocol_device_name
except ImportError:
    from config import MOTION_HITS_MAX, MOTION_HITS_MIN
    import mqtt.protocol as mqtt_protocol
    from mqtt.protocol import THRESHOLD_MAX, THRESHOLD_MIN, _normalize_chip_label, _protocol_device_name


def _sanitize_identifier(value):
    """Return a Home Assistant-safe object identifier token."""
    chars = []
    for ch in str(value):
        codepoint = ord(ch)
        if 48 <= codepoint <= 57 or 97 <= codepoint <= 122:
            chars.append(ch)
        elif 65 <= codepoint <= 90:
            chars.append(chr(codepoint + 32))
        else:
            chars.append("_")
    return "".join(chars)


_DIAGNOSTIC_SENSORS = (
    ("Traffic TX Rate", "traffic_tx_rate", "traffic_tx_pps", "pkt/s", "mdi:upload-network", None, None, True, None, False),
    ("CSI Callback Rate", "csi_callback_rate", "csi_callback_pps", "pkt/s", "mdi:access-point", None, None, True, None, False),
    ("CSI Accepted Rate", "csi_accepted_rate", "csi_accepted_pps", "pkt/s", "mdi:check-network", None, None, True, None, False),
    ("CSI Admitted Rate", "csi_admitted_rate", "csi_admitted_pps", "pkt/s", "mdi:timeline-check-outline", None, None, True, None, False),
    ("CSI Filtered Rate", "csi_filtered_rate", "csi_filtered_pps", "pkt/s", "mdi:filter-outline", None, None, True, None, False),
    ("CSI Missing Slot Rate", "csi_missing_rate", "csi_missing_slots_pps", "slot/s", "mdi:timeline-minus-outline", "csi_missing_slot_rate", None, True, None, False),
    ("CSI Excess Rate", "csi_excess_rate", "csi_excess_pps", "pkt/s", "mdi:timeline-plus-outline", None, None, True, None, False),
    ("CSI Stale Rate", "csi_stale_rate", "csi_stale_pps", "pkt/s", "mdi:timer-sand", None, None, True, None, False),
    ("CSI Out-of-order Rate", "csi_out_of_order_rate", "csi_out_of_order_pps", "pkt/s", "mdi:swap-vertical", None, None, True, None, False),
    ("CSI Temporal Occupancy", "csi_occupancy", "csi_occupancy", "%", "mdi:view-grid-outline", "csi_temporal_occupancy", None, True, 100.0, False),
    ("WiFi Channel", "wifi_channel", "wifi_channel", None, "mdi:wifi-marker", None, None, False, None, True),
    ("WiFi RSSI", "wifi_rssi", "wifi_rssi_dbm", "dBm", None, None, "signal_strength", True, None, True),
)

_RETIRED_DISCOVERY = (
    ("sensor", "intensity"),
    ("binary_sensor", "motion"),
    ("sensor", "movement"),
    ("switch", "calibrate"),
    ("select", "detector"),
    ("select", "csi_traffic_mode"),
    ("select", "traffic_generator_mode"),
    ("button", "diagnostics"),
    ("sensor", "csi_missing_rate"),
    ("sensor", "csi_occupancy"),
)


class HomeAssistantMqttAdapter:
    """Home Assistant MQTT Discovery companion for Micro-ESPectre."""

    BIRTH_TOPIC = "homeassistant/status"

    def __init__(self, config, detector, wlan, global_state=None, device_id=None):
        self.config = config
        self.detector = detector
        self.wlan = wlan
        self.global_state = global_state
        self.enabled = bool(getattr(config, "MQTT_HA_DISCOVERY_ENABLED", False))
        self.discovery_prefix = getattr(config, "MQTT_HA_DISCOVERY_PREFIX", "homeassistant").rstrip("/")
        topic_prefix = config.MQTT_TOPIC_PREFIX.rstrip("/")
        self.device_id = device_id or mqtt_protocol.derive_runtime_device_id(wlan)
        self._state_prefix = "{}/{}/ha".format(topic_prefix, self.device_id)
        object_prefix = _sanitize_identifier("micro_{}".format(self.device_id))
        self.ha_object_prefix = object_prefix
        self._last_state = 0
        self._last_variance = 0.0
        self._last_threshold = 0.0
        self._last_motion_on_hits = int(getattr(config, "MOTION_ON_HITS", 4))
        self._last_motion_off_hits = int(getattr(config, "MOTION_OFF_HITS", 3))
        self._last_published_motion = None
        self._last_published_threshold = None
        self._calibrating = False
        self._calibrate_handler = None
        self._threshold_handler = None
        self._motion_hits_handler = None
        self._traffic_control_handler = None
        self._last_csi_traffic_mode = "internal" if getattr(config, "TRAFFIC_GENERATOR_ENABLED", True) else "external"
        self._last_traffic_generator_mode = str(getattr(config, "TRAFFIC_GENERATOR_MODE", "ping")).lower()

    def _topic(self, suffix):
        return "{}/{}".format(self._state_prefix, suffix)

    def _object_id(self, suffix):
        return "{}_{}".format(self.ha_object_prefix, suffix)

    def _device_name(self):
        chip = getattr(self.global_state, "chip_type", None) or "esp32"
        return getattr(
            self.config,
            "MQTT_DEVICE_LABEL",
            "",
        ) or _protocol_device_name(self.device_id, chip)

    def _device_block(self):
        chip = getattr(self.global_state, "chip_type", None) or "esp32"
        return {
            "identifiers": [self.device_id],
            "name": self._device_name(),
            "manufacturer": "ESPectre",
            "model": "ESPectre Micro ({})".format(_normalize_chip_label(chip)),
            "sw_version": "micropython",
        }

    def _publish_json(self, client, topic, payload, retain=False):
        import gc
        import json
        import time

        encoded = json.dumps(payload)
        client.publish(topic, encoded, retain=retain)
        del encoded
        gc.collect()
        sleep_ms = getattr(time, "sleep_ms", None)
        if sleep_ms is not None:
            sleep_ms(20)
        else:
            time.sleep(0.02)

    def configure_client(self, client):
        """Attach HA-specific client settings before connect."""
        if not self.enabled or not hasattr(client, "set_last_will"):
            return
        client.set_last_will(self._topic("availability"), "offline", retain=False)

    def subscribe_topics(self, client):
        """Subscribe the client to HA lifecycle and command topics."""
        if not self.enabled:
            return
        client.subscribe(self.BIRTH_TOPIC)
        client.subscribe(self._topic("threshold/set"))
        client.subscribe(self._topic("motion_on_hits/set"))
        client.subscribe(self._topic("motion_off_hits/set"))
        client.subscribe(self._topic("calibrate/set"))
        client.subscribe(self._topic("csi_traffic_mode/set"))
        client.subscribe(self._topic("traffic_generator_mode/set"))
        client.subscribe(self._topic("diagnostics/set"))

    def set_calibrate_handler(self, handler):
        """Install the callback used by the HA Calibrate switch."""
        self._calibrate_handler = handler

    def set_threshold_handler(self, handler):
        """Install the callback used by the HA threshold number."""
        self._threshold_handler = handler

    def set_motion_hits_handler(self, handler):
        """Install the callback used by the HA motion-hit numbers."""
        self._motion_hits_handler = handler

    def set_traffic_control_handler(self, handler):
        """Install the callback used by the HA traffic-control selects."""
        self._traffic_control_handler = handler

    def record_state(self, movement_score, motion_state, threshold=None):
        """Cache the latest runtime state for HA birth republish."""
        self._last_variance = float(movement_score)
        self._last_state = int(motion_state)
        if threshold is not None:
            try:
                self._last_threshold = float(threshold)
            except (TypeError, ValueError):
                self._last_threshold = 0.0

    def set_motion_hits(self, motion_on_hits, motion_off_hits):
        """Cache the current motion-hit configuration."""
        self._last_motion_on_hits = int(motion_on_hits)
        self._last_motion_off_hits = int(motion_off_hits)

    def set_traffic_control(self, csi_traffic_mode, traffic_generator_mode):
        """Cache the current traffic-control state."""
        self._last_csi_traffic_mode = str(csi_traffic_mode).lower()
        self._last_traffic_generator_mode = str(traffic_generator_mode).lower()

    def publish_discovery(self, client):
        """Publish retained HA discovery payloads."""
        if not self.enabled:
            return
        base = {
            "availability_topic": self._topic("availability"),
            "payload_available": "online",
            "payload_not_available": "offline",
            "device": self._device_block(),
        }
        motion_payload = dict(base)
        motion_payload.update(
            {
                "name": "Motion Detected",
                "unique_id": self._object_id("motion_detected"),
                "object_id": self._object_id("motion_detected"),
                "state_topic": self._topic("motion/state"),
                "payload_on": "ON",
                "payload_off": "OFF",
                "device_class": "motion",
            }
        )
        self._publish_json(
            client,
            "{}/binary_sensor/{}/config".format(self.discovery_prefix, self._object_id("motion_detected")),
            motion_payload,
            retain=True,
        )
        del motion_payload
        movement_payload = dict(base)
        movement_payload.update(
            {
                "name": "Movement Score",
                "unique_id": self._object_id("movement_score"),
                "object_id": self._object_id("movement_score"),
                "state_topic": self._topic("movement/state"),
                "state_class": "measurement",
                "icon": "mdi:sine-wave",
            }
        )
        self._publish_json(
            client,
            "{}/sensor/{}/config".format(self.discovery_prefix, self._object_id("movement_score")),
            movement_payload,
            retain=True,
        )
        del movement_payload
        for (
            name,
            key,
            _sample_key,
            unit,
            icon,
            object_suffix,
            device_class,
            measurement,
            _scale,
            _integer,
        ) in _DIAGNOSTIC_SENSORS:
            object_id = "{}_{}".format(self.ha_object_prefix, object_suffix or key)
            payload = dict(base)
            payload.update(
                {
                    "name": name,
                    "unique_id": object_id,
                    "object_id": object_id,
                    "state_topic": self._topic("{}/state".format(key)),
                    "entity_category": "diagnostic",
                }
            )
            if unit:
                payload["unit_of_measurement"] = unit
            if measurement:
                payload["state_class"] = "measurement"
            if device_class:
                payload["device_class"] = device_class
            if icon:
                payload["icon"] = icon
            self._publish_json(
                client,
                "{}/sensor/{}/config".format(self.discovery_prefix, object_id),
                payload,
                retain=True,
            )
            del payload
        diagnostics_payload = dict(base)
        diagnostics_payload.update(
            {
                "name": "Refresh Diagnostics",
                "unique_id": self._object_id("refresh_diagnostics"),
                "object_id": self._object_id("refresh_diagnostics"),
                "command_topic": self._topic("diagnostics/set"),
                "payload_press": "PRESS",
                "entity_category": "diagnostic",
                "icon": "mdi:refresh",
            }
        )
        self._publish_json(
            client,
            "{}/button/{}/config".format(self.discovery_prefix, self._object_id("refresh_diagnostics")),
            diagnostics_payload,
            retain=True,
        )
        del diagnostics_payload
        threshold_payload = dict(base)
        threshold_payload.update(
            {
                "name": "Threshold",
                "unique_id": self._object_id("threshold"),
                "object_id": self._object_id("threshold"),
                "state_topic": self._topic("threshold/state"),
                "command_topic": self._topic("threshold/set"),
                "min": THRESHOLD_MIN,
                "max": THRESHOLD_MAX,
                "step": 0.01,
                "mode": "box",
                "entity_category": "config",
                "icon": "mdi:pulse",
            }
        )
        self._publish_json(
            client,
            "{}/number/{}/config".format(self.discovery_prefix, self._object_id("threshold")),
            threshold_payload,
            retain=True,
        )
        del threshold_payload
        motion_on_hits_payload = dict(base)
        motion_on_hits_payload.update(
            {
                "name": "Motion On Hits",
                "unique_id": self._object_id("motion_on_hits"),
                "object_id": self._object_id("motion_on_hits"),
                "state_topic": self._topic("motion_on_hits/state"),
                "command_topic": self._topic("motion_on_hits/set"),
                "min": MOTION_HITS_MIN,
                "max": MOTION_HITS_MAX,
                "step": 1,
                "mode": "box",
                "entity_category": "config",
                "icon": "mdi:motion-play-outline",
            }
        )
        self._publish_json(
            client,
            "{}/number/{}/config".format(self.discovery_prefix, self._object_id("motion_on_hits")),
            motion_on_hits_payload,
            retain=True,
        )
        del motion_on_hits_payload
        motion_off_hits_payload = dict(base)
        motion_off_hits_payload.update(
            {
                "name": "Motion Off Hits",
                "unique_id": self._object_id("motion_off_hits"),
                "object_id": self._object_id("motion_off_hits"),
                "state_topic": self._topic("motion_off_hits/state"),
                "command_topic": self._topic("motion_off_hits/set"),
                "min": MOTION_HITS_MIN,
                "max": MOTION_HITS_MAX,
                "step": 1,
                "mode": "box",
                "entity_category": "config",
                "icon": "mdi:motion-pause-outline",
            }
        )
        self._publish_json(
            client,
            "{}/number/{}/config".format(self.discovery_prefix, self._object_id("motion_off_hits")),
            motion_off_hits_payload,
            retain=True,
        )
        del motion_off_hits_payload
        csi_traffic_mode_payload = dict(base)
        csi_traffic_mode_payload.update(
            {
                "name": "CSI Traffic Ownership",
                "unique_id": self._object_id("csi_traffic_ownership"),
                "object_id": self._object_id("csi_traffic_ownership"),
                "state_topic": self._topic("csi_traffic_mode/state"),
                "command_topic": self._topic("csi_traffic_mode/set"),
                "options": ["internal", "external"],
                "entity_category": "config",
                "icon": "mdi:wifi-cog",
            }
        )
        self._publish_json(
            client,
            "{}/select/{}/config".format(self.discovery_prefix, self._object_id("csi_traffic_ownership")),
            csi_traffic_mode_payload,
            retain=True,
        )
        del csi_traffic_mode_payload
        traffic_generator_mode_payload = dict(base)
        traffic_generator_mode_payload.update(
            {
                "name": "CSI Traffic Source",
                "unique_id": self._object_id("csi_traffic_source"),
                "object_id": self._object_id("csi_traffic_source"),
                "state_topic": self._topic("traffic_generator_mode/state"),
                "command_topic": self._topic("traffic_generator_mode/set"),
                "options": ["ping", "dns"],
                "entity_category": "config",
                "icon": "mdi:swap-horizontal",
            }
        )
        self._publish_json(
            client,
            "{}/select/{}/config".format(self.discovery_prefix, self._object_id("csi_traffic_source")),
            traffic_generator_mode_payload,
            retain=True,
        )
        del traffic_generator_mode_payload
        calibrate_payload = dict(base)
        calibrate_payload.update(
            {
                "name": "Trigger Calibration",
                "unique_id": self._object_id("trigger_calibration"),
                "object_id": self._object_id("trigger_calibration"),
                "state_topic": self._topic("calibrate/state"),
                "command_topic": self._topic("calibrate/set"),
                "payload_on": "ON",
                "payload_off": "OFF",
                "entity_category": "config",
                "icon": "mdi:refresh",
            }
        )
        self._publish_json(
            client,
            "{}/switch/{}/config".format(self.discovery_prefix, self._object_id("trigger_calibration")),
            calibrate_payload,
            retain=True,
        )
        del calibrate_payload
        for component, suffix in _RETIRED_DISCOVERY:
            client.publish(
                "{}/{}/{}_{}/config".format(self.discovery_prefix, component, self.ha_object_prefix, suffix),
                "",
                retain=True,
            )
        import gc
        gc.collect()

    def publish_availability(self, client, online):
        """Publish plain-text HA availability updates."""
        if not self.enabled:
            return
        client.publish(self._topic("availability"), "online" if online else "offline", retain=False)

    def publish_motion(self, client, motion_state, force=False):
        """Publish the HA motion binary sensor on filtered edges, or on demand."""
        if not self.enabled:
            return
        state = int(motion_state)
        if not force and state == self._last_published_motion:
            return
        self._last_published_motion = state
        self._last_state = state
        client.publish(self._topic("motion/state"), "ON" if state == 1 else "OFF", retain=False)

    def publish_movement(self, client, movement_score):
        """Publish the HA movement-score sensor."""
        if not self.enabled:
            return
        self._last_variance = float(movement_score)
        client.publish(self._topic("movement/state"), "{:.4f}".format(self._last_variance), retain=False)

    def publish_threshold(self, client, threshold, force=False):
        """Publish the HA threshold number on change, or on demand."""
        if not self.enabled:
            return
        try:
            value = float(threshold)
        except (TypeError, ValueError):
            return
        if not force and value == self._last_published_threshold:
            return
        self._last_published_threshold = value
        self._last_threshold = value
        client.publish(self._topic("threshold/state"), "{:.4f}".format(value), retain=False)

    def publish_motion_hits(self, client, motion_on_hits, motion_off_hits, force=False):
        """Publish the HA motion-hit numbers."""
        if not self.enabled:
            return
        try:
            motion_on = int(motion_on_hits)
            motion_off = int(motion_off_hits)
        except (TypeError, ValueError):
            return
        changed = motion_on != self._last_motion_on_hits or motion_off != self._last_motion_off_hits
        self.set_motion_hits(motion_on, motion_off)
        if not force and not changed:
            return
        client.publish(self._topic("motion_on_hits/state"), str(motion_on), retain=False)
        client.publish(self._topic("motion_off_hits/state"), str(motion_off), retain=False)

    def publish_traffic_control(self, client, csi_traffic_mode, traffic_generator_mode, force=False):
        """Publish the HA traffic-control selects."""
        if not self.enabled:
            return
        csi_mode = str(csi_traffic_mode).lower()
        generator_mode = str(traffic_generator_mode).lower()
        changed = (
            csi_mode != self._last_csi_traffic_mode
            or generator_mode != self._last_traffic_generator_mode
        )
        self.set_traffic_control(csi_mode, generator_mode)
        if not force and not changed:
            return
        client.publish(self._topic("csi_traffic_mode/state"), csi_mode, retain=False)
        client.publish(self._topic("traffic_generator_mode/state"), generator_mode, retain=False)

    def is_calibrating(self):
        """Return whether the HA Calibrate switch currently reports ON."""
        return self._calibrating

    def set_calibrating(self, calibrating):
        """Cache the HA Calibrate switch state without publishing."""
        self._calibrating = bool(calibrating)

    def publish_calibrate(self, client, calibrating, force=False):
        """Publish the HA Calibrate switch state."""
        if not self.enabled:
            return
        state = bool(calibrating)
        if not force and state == self._calibrating:
            return
        self._calibrating = state
        client.publish(self._topic("calibrate/state"), "ON" if state else "OFF", retain=False)

    def publish_diagnostics(self, client, sample=None):
        """Publish cached CSI/Wi-Fi diagnostic sensors on demand."""
        if not self.enabled:
            return
        if sample is None:
            sample = getattr(self.global_state, "latest_diagnostics", None) or {}
        for (
            _name,
            key,
            sample_key,
            _unit,
            _icon,
            _object_suffix,
            _device_class,
            _measurement,
            scale,
            integer,
        ) in _DIAGNOSTIC_SENSORS:
            value = sample.get(sample_key)
            if value is None:
                continue
            try:
                number = float(value)
            except (TypeError, ValueError):
                continue
            if scale:
                number = number * float(scale)
            state_topic = self._topic("{}/state".format(key))
            if integer:
                client.publish(state_topic, str(int(number)), retain=False)
            else:
                client.publish(state_topic, "{:.1f}".format(number), retain=False)

    def apply_diagnostics_command(self, client, payload):
        """Publish the cached diagnostic sample when Home Assistant presses Refresh Diagnostics."""
        del payload
        self.publish_diagnostics(client)
        return True

    def publish_snapshot(self, client, movement_score, motion_state, threshold):
        """Publish all HA entity states for connect and Home Assistant birth."""
        self.record_state(movement_score, motion_state, threshold)
        if not self.enabled:
            return
        self.publish_motion(client, motion_state, force=True)
        self.publish_movement(client, movement_score)
        self.publish_threshold(client, threshold, force=True)
        self.publish_motion_hits(client, self._last_motion_on_hits, self._last_motion_off_hits, force=True)
        self.publish_calibrate(client, self._calibrating, force=True)
        self.publish_traffic_control(
            client,
            self._last_csi_traffic_mode,
            self._last_traffic_generator_mode,
            force=True,
        )

    def apply_threshold_command(self, client, payload):
        """Apply a Home Assistant number command to the live detector threshold."""
        if not self.enabled:
            return False
        if isinstance(payload, bytes):
            payload = payload.decode("utf-8", "ignore")
        try:
            threshold = float(str(payload).strip())
        except (TypeError, ValueError):
            return False
        if threshold < THRESHOLD_MIN or threshold > THRESHOLD_MAX:
            return False
        handler = self._threshold_handler
        if callable(handler):
            return bool(handler(threshold))
        setter = getattr(self.detector, "set_threshold", None)
        if not callable(setter) or not setter(threshold):
            return False
        self.publish_threshold(client, threshold, force=True)
        return True

    def apply_calibrate_command(self, client, payload):
        """Apply a Home Assistant switch command to the live recalibration request."""
        if not self.enabled:
            return False
        if isinstance(payload, bytes):
            payload = payload.decode("utf-8", "ignore")
        token = str(payload).strip().lower()
        if token == "off":
            self.publish_calibrate(client, self._calibrating, force=True)
            return True
        if token != "on":
            return False
        if self._calibrating:
            self.publish_calibrate(client, True, force=True)
            return True
        handler = self._calibrate_handler
        if callable(handler):
            handler()
        else:
            self.publish_calibrate(client, True)
        return True

    def _parse_motion_hits_value(self, payload):
        if isinstance(payload, bytes):
            payload = payload.decode("utf-8", "ignore")
        try:
            value = int(str(payload).strip())
        except (TypeError, ValueError):
            return None
        if value < MOTION_HITS_MIN or value > MOTION_HITS_MAX:
            return None
        return value

    def apply_motion_hits_command(self, client, payload, motion_on):
        """Apply one HA number command to the live motion-hit policy."""
        if not self.enabled:
            return False
        value = self._parse_motion_hits_value(payload)
        if value is None:
            return False
        motion_on_hits = value if motion_on else self._last_motion_on_hits
        motion_off_hits = self._last_motion_off_hits if motion_on else value
        handler = self._motion_hits_handler
        if callable(handler):
            return bool(handler(motion_on_hits, motion_off_hits))
        self.set_motion_hits(motion_on_hits, motion_off_hits)
        self.publish_motion_hits(client, motion_on_hits, motion_off_hits, force=True)
        return True

    def apply_traffic_control_command(self, client, payload, csi_traffic_mode):
        """Apply one HA select command to the live traffic-control state."""
        if not self.enabled:
            return False
        if isinstance(payload, bytes):
            payload = payload.decode("utf-8", "ignore")
        value = str(payload).strip().lower()
        if csi_traffic_mode:
            if value not in ("internal", "external"):
                return False
            csi_mode = value
            generator_mode = self._last_traffic_generator_mode
        else:
            if value not in ("ping", "dns"):
                return False
            csi_mode = self._last_csi_traffic_mode
            generator_mode = value
        handler = self._traffic_control_handler
        if callable(handler):
            return bool(handler(csi_mode, generator_mode))
        self.set_traffic_control(csi_mode, generator_mode)
        self.publish_traffic_control(client, csi_mode, generator_mode, force=True)
        return True

    def handle_message(self, client, topic, payload):
        """Handle subscribed HA lifecycle and command messages."""
        if not self.enabled:
            return False
        if topic == self._topic("threshold/set"):
            return self.apply_threshold_command(client, payload)
        if topic == self._topic("motion_on_hits/set"):
            return self.apply_motion_hits_command(client, payload, True)
        if topic == self._topic("motion_off_hits/set"):
            return self.apply_motion_hits_command(client, payload, False)
        if topic == self._topic("calibrate/set"):
            return self.apply_calibrate_command(client, payload)
        if topic == self._topic("csi_traffic_mode/set"):
            return self.apply_traffic_control_command(client, payload, True)
        if topic == self._topic("traffic_generator_mode/set"):
            return self.apply_traffic_control_command(client, payload, False)
        if topic == self._topic("diagnostics/set"):
            return self.apply_diagnostics_command(client, payload)
        if topic != self.BIRTH_TOPIC:
            return False
        if isinstance(payload, bytes):
            payload = payload.decode("utf-8", "ignore")
        if str(payload).strip().lower() != "online":
            return False
        self.publish_discovery(client)
        self.publish_availability(client, True)
        self.publish_snapshot(client, self._last_variance, self._last_state, self._last_threshold)
        return True
