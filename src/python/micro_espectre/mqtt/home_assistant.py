# SPDX-License-Identifier: GPL-3.0-only
# Commercial licensing available under separate agreement; see LICENSING.md.
"""
Micro-ESPectre - Home Assistant MQTT adapter.

Publishes Home Assistant MQTT Discovery and entity-shaped state topics while
preserving the canonical ESPectre protocol topics.

Author: Francesco Pace <francesco.pace@gmail.com>
"""

try:
    from src.mqtt.commands import _normalize_chip_label, _protocol_device_name
except ImportError:
    from mqtt.commands import _normalize_chip_label, _protocol_device_name


def _sanitize_identifier(value):
    """Return a Home Assistant-safe object identifier token."""
    chars = []
    for ch in str(value):
        if ch.isalnum():
            chars.append(ch.lower())
        else:
            chars.append("_")
    return "".join(chars)


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
        object_prefix = _sanitize_identifier("micro_{}".format(self.device_id))
        self.motion_object_id = "{}_motion".format(object_prefix)
        self.movement_object_id = "{}_movement".format(object_prefix)
        self._last_state = 0
        self._last_variance = 0.0

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
        """Subscribe the client to HA lifecycle topics."""
        if not self.enabled:
            return
        client.subscribe(self.BIRTH_TOPIC)

    def record_state(self, movement_score, motion_state):
        """Cache the latest runtime state for HA birth republish."""
        self._last_variance = float(movement_score)
        self._last_state = int(motion_state)

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

    def publish_availability(self, client, online):
        """Publish plain-text HA availability updates."""
        if not self.enabled:
            return
        client.publish(self.availability_topic, "online" if online else "offline", retain=False)

    def publish_state(self, client, movement_score, motion_state):
        """Publish plain HA entity state values."""
        if not self.enabled:
            return
        self.record_state(movement_score, motion_state)
        client.publish(self.motion_state_topic, "ON" if motion_state == 1 else "OFF", retain=False)
        client.publish(self.movement_state_topic, "{:.4f}".format(float(movement_score)), retain=False)

    def handle_message(self, client, topic, payload):
        """Handle subscribed HA lifecycle messages."""
        if not self.enabled or topic != self.BIRTH_TOPIC:
            return False
        if isinstance(payload, bytes):
            payload = payload.decode("utf-8", "ignore")
        if str(payload).strip().lower() != "online":
            return False
        self.publish_discovery(client)
        self.publish_availability(client, True)
        self.publish_state(client, self._last_variance, self._last_state)
        return True
