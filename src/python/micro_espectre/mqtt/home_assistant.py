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
    from src.mqtt.commands import THRESHOLD_MAX, THRESHOLD_MIN, _normalize_chip_label, _protocol_device_name
except ImportError:
    from config import MOTION_HITS_MAX, MOTION_HITS_MIN
    from mqtt.commands import THRESHOLD_MAX, THRESHOLD_MIN, _normalize_chip_label, _protocol_device_name


def _sanitize_identifier(value):
    """Return a Home Assistant-safe object identifier token."""
    chars = []
    for ch in str(value):
        if ch.isalnum():
            chars.append(ch.lower())
        else:
            chars.append("_")
    return "".join(chars)


_DIAGNOSTIC_SENSORS = (
    {"name": "Traffic TX Rate", "key": "traffic_tx_rate", "sample_key": "traffic_tx_pps", "unit": "pkt/s", "icon": "mdi:upload-network"},
    {"name": "CSI Callback Rate", "key": "csi_callback_rate", "sample_key": "csi_callback_pps", "unit": "pkt/s", "icon": "mdi:access-point"},
    {"name": "CSI Accepted Rate", "key": "csi_accepted_rate", "sample_key": "csi_accepted_pps", "unit": "pkt/s", "icon": "mdi:check-network"},
    {"name": "CSI Admitted Rate", "key": "csi_admitted_rate", "sample_key": "csi_admitted_pps", "unit": "pkt/s", "icon": "mdi:timeline-check-outline"},
    {"name": "CSI Filtered Rate", "key": "csi_filtered_rate", "sample_key": "csi_filtered_pps", "unit": "pkt/s", "icon": "mdi:filter-outline"},
    {
        "name": "CSI Missing Slot Rate",
        "key": "csi_missing_rate",
        "object_suffix": "csi_missing_slot_rate",
        "sample_key": "csi_missing_slots_pps",
        "unit": "slot/s",
        "icon": "mdi:timeline-minus-outline",
    },
    {"name": "CSI Excess Rate", "key": "csi_excess_rate", "sample_key": "csi_excess_pps", "unit": "pkt/s", "icon": "mdi:timeline-plus-outline"},
    {"name": "CSI Stale Rate", "key": "csi_stale_rate", "sample_key": "csi_stale_pps", "unit": "pkt/s", "icon": "mdi:timer-sand"},
    {"name": "CSI Out-of-order Rate", "key": "csi_out_of_order_rate", "sample_key": "csi_out_of_order_pps", "unit": "pkt/s", "icon": "mdi:swap-vertical"},
    {
        "name": "CSI Temporal Occupancy",
        "key": "csi_occupancy",
        "object_suffix": "csi_temporal_occupancy",
        "sample_key": "csi_occupancy",
        "unit": "%",
        "icon": "mdi:view-grid-outline",
        "scale": 100.0,
    },
    {"name": "WiFi Channel", "key": "wifi_channel", "sample_key": "wifi_channel", "icon": "mdi:wifi-marker", "measurement": False, "integer": True},
    {"name": "WiFi RSSI", "key": "wifi_rssi", "sample_key": "wifi_rssi_dbm", "unit": "dBm", "device_class": "signal_strength", "integer": True},
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

    def __init__(self, config, detector, wlan, global_state=None):
        self.config = config
        self.detector = detector
        self.wlan = wlan
        self.global_state = global_state
        self.enabled = bool(getattr(config, "MQTT_HA_DISCOVERY_ENABLED", False))
        self.discovery_prefix = getattr(config, "MQTT_HA_DISCOVERY_PREFIX", "homeassistant").rstrip("/")
        topic_prefix = config.MQTT_TOPIC_PREFIX.rstrip("/")
        self.device_id = config.MQTT_CLIENT_ID
        state_prefix = "{}/{}/ha".format(topic_prefix, self.device_id)
        self.availability_topic = "{}/availability".format(state_prefix)
        self.motion_state_topic = "{}/motion/state".format(state_prefix)
        self.movement_state_topic = "{}/movement/state".format(state_prefix)
        self.threshold_state_topic = "{}/threshold/state".format(state_prefix)
        self.threshold_command_topic = "{}/threshold/set".format(state_prefix)
        self.motion_on_hits_state_topic = "{}/motion_on_hits/state".format(state_prefix)
        self.motion_on_hits_command_topic = "{}/motion_on_hits/set".format(state_prefix)
        self.motion_off_hits_state_topic = "{}/motion_off_hits/state".format(state_prefix)
        self.motion_off_hits_command_topic = "{}/motion_off_hits/set".format(state_prefix)
        self.calibrate_state_topic = "{}/calibrate/state".format(state_prefix)
        self.calibrate_command_topic = "{}/calibrate/set".format(state_prefix)
        self.csi_traffic_mode_state_topic = "{}/csi_traffic_mode/state".format(state_prefix)
        self.csi_traffic_mode_command_topic = "{}/csi_traffic_mode/set".format(state_prefix)
        self.traffic_generator_mode_state_topic = "{}/traffic_generator_mode/state".format(state_prefix)
        self.traffic_generator_mode_command_topic = "{}/traffic_generator_mode/set".format(state_prefix)
        self.diagnostics_command_topic = "{}/diagnostics/set".format(state_prefix)
        object_prefix = _sanitize_identifier("micro_{}".format(self.device_id))
        self.ha_object_prefix = object_prefix
        self.motion_object_id = "{}_motion_detected".format(object_prefix)
        self.movement_object_id = "{}_movement_score".format(object_prefix)
        self.threshold_object_id = "{}_threshold".format(object_prefix)
        self.motion_on_hits_object_id = "{}_motion_on_hits".format(object_prefix)
        self.motion_off_hits_object_id = "{}_motion_off_hits".format(object_prefix)
        self.calibrate_object_id = "{}_trigger_calibration".format(object_prefix)
        self.csi_traffic_mode_object_id = "{}_csi_traffic_ownership".format(object_prefix)
        self.traffic_generator_mode_object_id = "{}_csi_traffic_source".format(object_prefix)
        self.diagnostics_object_id = "{}_refresh_diagnostics".format(object_prefix)
        self.diagnostic_sensors = []
        for spec in _DIAGNOSTIC_SENSORS:
            object_suffix = spec.get("object_suffix", spec["key"])
            sensor = {
                "name": spec["name"],
                "key": spec["key"],
                "sample_key": spec["sample_key"],
                "object_id": "{}_{}".format(object_prefix, object_suffix),
                "state_topic": "{}/{}/state".format(state_prefix, spec["key"]),
                "unit": spec.get("unit"),
                "icon": spec.get("icon"),
                "device_class": spec.get("device_class"),
                "measurement": spec.get("measurement", True),
                "scale": spec.get("scale"),
                "integer": spec.get("integer", False),
            }
            self.diagnostic_sensors.append(sensor)
        self._last_state = 0
        self._last_variance = 0.0
        self._last_threshold = 0.0
        self._last_motion_on_hits = int(getattr(config, "MOTION_ON_HITS", 4))
        self._last_motion_off_hits = int(getattr(config, "MOTION_OFF_HITS", 3))
        self._last_published_motion = None
        self._last_published_threshold = None
        self._calibrating = False
        self._calibrate_handler = None
        self._motion_hits_handler = None
        self._traffic_control_handler = None
        self._last_csi_traffic_mode = "internal" if getattr(config, "TRAFFIC_GENERATOR_ENABLED", True) else "external"
        self._last_traffic_generator_mode = str(getattr(config, "TRAFFIC_GENERATOR_MODE", "ping")).lower()

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
        import json

        client.publish(topic, json.dumps(payload), retain=retain)

    def configure_client(self, client):
        """Attach HA-specific client settings before connect."""
        if not self.enabled or not hasattr(client, "set_last_will"):
            return
        client.set_last_will(self.availability_topic, "offline", retain=False)

    def subscribe_topics(self, client):
        """Subscribe the client to HA lifecycle and command topics."""
        if not self.enabled:
            return
        client.subscribe(self.BIRTH_TOPIC)
        client.subscribe(self.threshold_command_topic)
        client.subscribe(self.motion_on_hits_command_topic)
        client.subscribe(self.motion_off_hits_command_topic)
        client.subscribe(self.calibrate_command_topic)
        client.subscribe(self.csi_traffic_mode_command_topic)
        client.subscribe(self.traffic_generator_mode_command_topic)
        client.subscribe(self.diagnostics_command_topic)

    def set_calibrate_handler(self, handler):
        """Install the callback used by the HA Calibrate switch."""
        self._calibrate_handler = handler

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
            "availability_topic": self.availability_topic,
            "payload_available": "online",
            "payload_not_available": "offline",
            "device": self._device_block(),
        }
        motion_payload = dict(base)
        motion_payload.update(
            {
                "name": "Motion Detected",
                "unique_id": self.motion_object_id,
                "object_id": self.motion_object_id,
                "state_topic": self.motion_state_topic,
                "payload_on": "ON",
                "payload_off": "OFF",
                "device_class": "motion",
            }
        )
        movement_payload = dict(base)
        movement_payload.update(
            {
                "name": "Movement Score",
                "unique_id": self.movement_object_id,
                "object_id": self.movement_object_id,
                "state_topic": self.movement_state_topic,
                "state_class": "measurement",
                "icon": "mdi:sine-wave",
            }
        )
        self._publish_json(
            client,
            "{}/binary_sensor/{}/config".format(self.discovery_prefix, self.motion_object_id),
            motion_payload,
            retain=True,
        )
        self._publish_json(
            client,
            "{}/sensor/{}/config".format(self.discovery_prefix, self.movement_object_id),
            movement_payload,
            retain=True,
        )
        for sensor in self.diagnostic_sensors:
            payload = dict(base)
            payload.update(
                {
                    "name": sensor["name"],
                    "unique_id": sensor["object_id"],
                    "object_id": sensor["object_id"],
                    "state_topic": sensor["state_topic"],
                    "entity_category": "diagnostic",
                }
            )
            if sensor["unit"]:
                payload["unit_of_measurement"] = sensor["unit"]
            if sensor["measurement"]:
                payload["state_class"] = "measurement"
            if sensor["device_class"]:
                payload["device_class"] = sensor["device_class"]
            if sensor["icon"]:
                payload["icon"] = sensor["icon"]
            self._publish_json(
                client,
                "{}/sensor/{}/config".format(self.discovery_prefix, sensor["object_id"]),
                payload,
                retain=True,
            )
        diagnostics_payload = dict(base)
        diagnostics_payload.update(
            {
                "name": "Refresh Diagnostics",
                "unique_id": self.diagnostics_object_id,
                "object_id": self.diagnostics_object_id,
                "command_topic": self.diagnostics_command_topic,
                "payload_press": "PRESS",
                "entity_category": "diagnostic",
                "icon": "mdi:refresh",
            }
        )
        self._publish_json(
            client,
            "{}/button/{}/config".format(self.discovery_prefix, self.diagnostics_object_id),
            diagnostics_payload,
            retain=True,
        )
        threshold_payload = dict(base)
        threshold_payload.update(
            {
                "name": "Threshold",
                "unique_id": self.threshold_object_id,
                "object_id": self.threshold_object_id,
                "state_topic": self.threshold_state_topic,
                "command_topic": self.threshold_command_topic,
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
            "{}/number/{}/config".format(self.discovery_prefix, self.threshold_object_id),
            threshold_payload,
            retain=True,
        )
        motion_on_hits_payload = dict(base)
        motion_on_hits_payload.update(
            {
                "name": "Motion On Hits",
                "unique_id": self.motion_on_hits_object_id,
                "object_id": self.motion_on_hits_object_id,
                "state_topic": self.motion_on_hits_state_topic,
                "command_topic": self.motion_on_hits_command_topic,
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
            "{}/number/{}/config".format(self.discovery_prefix, self.motion_on_hits_object_id),
            motion_on_hits_payload,
            retain=True,
        )
        motion_off_hits_payload = dict(base)
        motion_off_hits_payload.update(
            {
                "name": "Motion Off Hits",
                "unique_id": self.motion_off_hits_object_id,
                "object_id": self.motion_off_hits_object_id,
                "state_topic": self.motion_off_hits_state_topic,
                "command_topic": self.motion_off_hits_command_topic,
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
            "{}/number/{}/config".format(self.discovery_prefix, self.motion_off_hits_object_id),
            motion_off_hits_payload,
            retain=True,
        )
        csi_traffic_mode_payload = dict(base)
        csi_traffic_mode_payload.update(
            {
                "name": "CSI Traffic Ownership",
                "unique_id": self.csi_traffic_mode_object_id,
                "object_id": self.csi_traffic_mode_object_id,
                "state_topic": self.csi_traffic_mode_state_topic,
                "command_topic": self.csi_traffic_mode_command_topic,
                "options": ["internal", "external", "disabled"],
                "entity_category": "config",
                "icon": "mdi:wifi-cog",
            }
        )
        self._publish_json(
            client,
            "{}/select/{}/config".format(self.discovery_prefix, self.csi_traffic_mode_object_id),
            csi_traffic_mode_payload,
            retain=True,
        )
        traffic_generator_mode_payload = dict(base)
        traffic_generator_mode_payload.update(
            {
                "name": "CSI Traffic Source",
                "unique_id": self.traffic_generator_mode_object_id,
                "object_id": self.traffic_generator_mode_object_id,
                "state_topic": self.traffic_generator_mode_state_topic,
                "command_topic": self.traffic_generator_mode_command_topic,
                "options": ["ping", "dns"],
                "entity_category": "config",
                "icon": "mdi:swap-horizontal",
            }
        )
        self._publish_json(
            client,
            "{}/select/{}/config".format(self.discovery_prefix, self.traffic_generator_mode_object_id),
            traffic_generator_mode_payload,
            retain=True,
        )
        calibrate_payload = dict(base)
        calibrate_payload.update(
            {
                "name": "Trigger Calibration",
                "unique_id": self.calibrate_object_id,
                "object_id": self.calibrate_object_id,
                "state_topic": self.calibrate_state_topic,
                "command_topic": self.calibrate_command_topic,
                "payload_on": "ON",
                "payload_off": "OFF",
                "entity_category": "config",
                "icon": "mdi:refresh",
            }
        )
        self._publish_json(
            client,
            "{}/switch/{}/config".format(self.discovery_prefix, self.calibrate_object_id),
            calibrate_payload,
            retain=True,
        )
        for component, suffix in _RETIRED_DISCOVERY:
            client.publish(
                "{}/{}/{}_{}/config".format(self.discovery_prefix, component, self.ha_object_prefix, suffix),
                "",
                retain=True,
            )

    def publish_availability(self, client, online):
        """Publish plain-text HA availability updates."""
        if not self.enabled:
            return
        client.publish(self.availability_topic, "online" if online else "offline", retain=False)

    def publish_motion(self, client, motion_state, force=False):
        """Publish the HA motion binary sensor on filtered edges, or on demand."""
        if not self.enabled:
            return
        state = int(motion_state)
        if not force and state == self._last_published_motion:
            return
        self._last_published_motion = state
        self._last_state = state
        client.publish(self.motion_state_topic, "ON" if state == 1 else "OFF", retain=False)

    def publish_movement(self, client, movement_score):
        """Publish the HA movement-score sensor."""
        if not self.enabled:
            return
        self._last_variance = float(movement_score)
        client.publish(self.movement_state_topic, "{:.4f}".format(self._last_variance), retain=False)

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
        client.publish(self.threshold_state_topic, "{:.4f}".format(value), retain=False)

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
        client.publish(self.motion_on_hits_state_topic, str(motion_on), retain=False)
        client.publish(self.motion_off_hits_state_topic, str(motion_off), retain=False)

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
        client.publish(self.csi_traffic_mode_state_topic, csi_mode, retain=False)
        client.publish(self.traffic_generator_mode_state_topic, generator_mode, retain=False)

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
        client.publish(self.calibrate_state_topic, "ON" if state else "OFF", retain=False)

    def publish_diagnostics(self, client, sample=None):
        """Publish cached CSI/Wi-Fi diagnostic sensors on demand."""
        if not self.enabled:
            return
        if sample is None:
            sample = getattr(self.global_state, "latest_diagnostics", None) or {}
        for sensor in self.diagnostic_sensors:
            value = sample.get(sensor["sample_key"])
            if value is None:
                continue
            try:
                number = float(value)
            except (TypeError, ValueError):
                continue
            scale = sensor.get("scale")
            if scale:
                number = number * float(scale)
            if sensor.get("integer"):
                client.publish(sensor["state_topic"], str(int(number)), retain=False)
            else:
                client.publish(sensor["state_topic"], "{:.1f}".format(number), retain=False)

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
            if value not in ("internal", "external", "disabled"):
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
        if topic == self.threshold_command_topic:
            return self.apply_threshold_command(client, payload)
        if topic == self.motion_on_hits_command_topic:
            return self.apply_motion_hits_command(client, payload, True)
        if topic == self.motion_off_hits_command_topic:
            return self.apply_motion_hits_command(client, payload, False)
        if topic == self.calibrate_command_topic:
            return self.apply_calibrate_command(client, payload)
        if topic == self.csi_traffic_mode_command_topic:
            return self.apply_traffic_control_command(client, payload, True)
        if topic == self.traffic_generator_mode_command_topic:
            return self.apply_traffic_control_command(client, payload, False)
        if topic == self.diagnostics_command_topic:
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
