# SPDX-License-Identifier: GPL-3.0-only
# Commercial licensing available under separate agreement; see LICENSING.md.
"""Minimal Direct HTTP facade for the Micro-ESPectre sensing runtime."""

import json
import time

import espectre_native_direct as native_direct

try:
    from src.protocol import (
        DNS_SD_TXT_SCHEMA_VERSION,
        PROTOCOL_VERSION,
        build_capabilities_payload,
        build_diagnostics_payload,
        build_info_payload,
        build_status_payload,
        build_telemetry_payload,
        derive_runtime_device_id,
    )
except ImportError:
    from protocol import (
        DNS_SD_TXT_SCHEMA_VERSION,
        PROTOCOL_VERSION,
        build_capabilities_payload,
        build_diagnostics_payload,
        build_info_payload,
        build_status_payload,
        build_telemetry_payload,
        derive_runtime_device_id,
    )


DIRECT_HTTP_PORT = 62587


def _json(payload):
    return json.dumps(payload)


def _bssid_text(value):
    if not isinstance(value, (bytes, bytearray)) or len(value) < 6:
        return ""
    return ":".join("%02X" % byte for byte in value[:6])


class DirectApi:
    """Publish read-only protocol snapshots and sensing events."""

    def __init__(self, config, wlan, detector, global_state, runtime_policy, traffic_generator):
        self.config = config
        self.wlan = wlan
        self.detector = detector
        self.global_state = global_state
        self.runtime_policy = runtime_policy
        self.traffic_generator = traffic_generator
        self.device_id = derive_runtime_device_id(wlan)
        self.started_ms = time.ticks_ms()
        self.started = False

    def _device_hostname(self):
        return "espectre-micro-" + self.device_id[-6:]

    def _info(self):
        csi_traffic_mode = "internal" if self.traffic_generator.is_running() else "external"
        return build_info_payload(
            self.config,
            "lightweight",
            self.wlan,
            global_state=self.global_state,
            device_id=self.device_id,
            csi_traffic_mode=csi_traffic_mode,
        )

    def _status(self, now_ms=None):
        if now_ms is None:
            now_ms = time.ticks_ms()
        return build_status_payload(
            self.device_id,
            True,
            now_ms,
            sensing_enabled=True,
            ready_to_publish=self.detector.is_ready(),
            calibrating=bool(self.global_state.calibration_mode),
            wifi_connected=self.wlan.isconnected(),
        )

    def _config(self):
        try:
            ssid = self.wlan.config("ssid") or ""
        except Exception:
            ssid = ""
        try:
            bssid = _bssid_text(self.wlan.config("bssid"))
        except Exception:
            bssid = ""
        try:
            rssi_dbm = self.wlan.status("rssi")
        except Exception:
            rssi_dbm = None
        try:
            channel = self.wlan.config("channel")
        except Exception:
            channel = self.global_state.current_channel
        return {
            "protocol_version": PROTOCOL_VERSION,
            "device_id": self.device_id,
            "device": {"device_label": getattr(self.config, "DEVICE_LABEL", "")},
            "wifi": {
                "configured": bool(getattr(self.config, "WIFI_SSID", "")),
                "connected": self.wlan.isconnected(),
                "ssid": ssid,
                "bssid": bssid,
                "band": "2.4GHz",
                "channel": channel,
                "rssi_dbm": rssi_dbm,
            },
            "runtime": {
                "threshold": self.detector.get_threshold(),
                "detector": "lightweight",
                "motion_on_hits": self.runtime_policy.motion_on_hits,
                "motion_off_hits": self.runtime_policy.motion_off_hits,
                "csi_traffic_mode": "internal" if self.traffic_generator.is_running() else "external",
                "traffic_generator_mode": "ping",
                "csi_target_pps": max(1, int(getattr(self.config, "CSI_TARGET_PPS", 100))),
            },
        }

    def _diagnostics(self, now_ms=None, measurements=None):
        if now_ms is None:
            now_ms = time.ticks_ms()
        uptime_ms = max(0, time.ticks_diff(now_ms, self.started_ms))
        return build_diagnostics_payload(
            self.device_id,
            now_ms,
            uptime_ms // 1000,
            measurements,
        )

    def start(self):
        """Start the native bounded HTTP server and mDNS advertisement."""
        if self.started:
            return
        capabilities = build_capabilities_payload(self.device_id)
        info = self._info()
        native_direct.start(
            port=DIRECT_HTTP_PORT,
            hostname=self._device_hostname(),
            instance=info["device_name"],
            device_id=self.device_id,
            chip=str(info["chip"]),
            protocol_version=PROTOCOL_VERSION,
            dns_sd_schema_version=DNS_SD_TXT_SCHEMA_VERSION,
            capabilities=_json(capabilities),
            info=_json(info),
            config=_json(self._config()),
            status=_json(self._status()),
            diagnostics=_json(self._diagnostics()),
        )
        self.started = True

    def refresh_snapshots(self, now_ms=None, diagnostics=None):
        """Refresh the read-only status and diagnostics snapshots."""
        if not self.started:
            return
        if now_ms is None:
            now_ms = time.ticks_ms()
        native_direct.update_status(_json(self._status(now_ms)))
        native_direct.update_diagnostics(_json(self._diagnostics(now_ms, diagnostics)))

    def publish_telemetry(self, movement_score, motion_state, threshold, now_ms=None):
        """Emit one canonical telemetry event after a detector evaluation."""
        if not self.started or not native_direct.has_event_client():
            return
        if now_ms is None:
            now_ms = time.ticks_ms()
        uptime_ms = max(0, time.ticks_diff(now_ms, self.started_ms))
        payload = build_telemetry_payload(
            self.device_id,
            "micro",
            now_ms,
            "motion" if motion_state else "idle",
            movement_score,
            threshold,
            "lightweight",
            uptime_ms // 1000,
        )
        native_direct.publish("telemetry", _json(payload))

    def stop(self):
        """Stop Direct HTTP and remove its DNS-SD service."""
        if self.started:
            native_direct.stop()
            self.started = False
