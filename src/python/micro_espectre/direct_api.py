# SPDX-License-Identifier: GPL-3.0-only
# Commercial licensing available under separate agreement; see LICENSING.md.
"""Minimal Direct HTTP facade for the Micro-ESPectre sensing runtime."""

import gc
import errno
import time

import espectre_native_direct as native_direct

try:
    from src.protocol import (
        DNS_SD_TXT_SCHEMA_VERSION,
        DIAGNOSTIC_FIELDS,
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
        DIAGNOSTIC_FIELDS,
        PROTOCOL_VERSION,
        build_capabilities_payload,
        build_diagnostics_payload,
        build_info_payload,
        build_status_payload,
        build_telemetry_payload,
        derive_runtime_device_id,
    )


DIRECT_HTTP_PORT = 62587
ERRNO_EBUSY = getattr(errno, "EBUSY", 16)

def _bssid_text(value):
    if not isinstance(value, (bytes, bytearray)) or len(value) < 6:
        return ""
    return ":".join("%02X" % byte for byte in value[:6])


def _wifi_band(channel):
    try:
        channel = int(channel)
    except (TypeError, ValueError):
        return ""
    if channel <= 0:
        return ""
    return "2g" if channel <= 14 else "5g"


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
        self.firmware_version = native_direct.firmware_version()
        self.chip = native_direct.chip_target()
        self.started_ms = time.ticks_ms()
        self._uptime_last_ms = self.started_ms
        self._uptime_ms = 0
        self.started = False
        self._status_payload = build_status_payload(
            self.device_id,
            True,
            self.started_ms,
            sensing_enabled=True,
            ready_to_publish=False,
            calibrating=bool(self.global_state.calibration_mode),
            wifi_connected=self.wlan.isconnected(),
        )
        self._diagnostics_payload = build_diagnostics_payload(
            self.device_id,
            self.started_ms,
            0,
        )
        self._direct_http_diagnostics = {
            "event_clients": 0,
            "event_client_limit": 1,
            "queue_capacity": 1,
            "queued_messages": 0,
            "accepted_connections": 0,
            "rejected_connections": 0,
            "malformed_requests": 0,
            "oversized_requests": 0,
            "rate_limited_requests": 0,
            "dropped_telemetry_events": 0,
            "send_failures": 0,
        }
        self._diagnostics_payload["direct_http"] = self._direct_http_diagnostics
        self._telemetry_payload = build_telemetry_payload(
            self.device_id,
            "micro",
            self.started_ms,
            "idle",
            0.0,
            self.detector.get_threshold(),
            "lightweight",
            0,
        )

    def _uptime_seconds(self, now_ms):
        """Accumulate uptime across MicroPython's wrapping tick counter."""
        elapsed_ms = time.ticks_diff(now_ms, self._uptime_last_ms)
        if elapsed_ms > 0:
            self._uptime_ms += elapsed_ms
            self._uptime_last_ms = now_ms
        return self._uptime_ms // 1000

    def _device_hostname(self):
        return "espectre-" + self.device_id

    def _info(self):
        csi_traffic_mode = "internal" if self.traffic_generator.is_running() else "external"
        return build_info_payload(
            self.config,
            "lightweight",
            self.wlan,
            global_state=self.global_state,
            device_id=self.device_id,
            csi_traffic_mode=csi_traffic_mode,
            firmware_version=self.firmware_version,
            chip=self.chip,
        )

    def _status(self, now_ms=None):
        if now_ms is None:
            now_ms = time.ticks_ms()
        payload = self._status_payload
        payload["timestamp_ms"] = now_ms
        payload["ready_to_publish"] = self.detector.is_ready()
        payload["calibrating"] = bool(self.global_state.calibration_mode)
        payload["wifi_connected"] = self.wlan.isconnected()
        return payload

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
                "band": _wifi_band(channel),
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
        payload = self._diagnostics_payload
        payload["timestamp_ms"] = int(now_ms)
        payload["uptime"] = self._uptime_seconds(now_ms)
        if isinstance(measurements, dict):
            for key in DIAGNOSTIC_FIELDS:
                if key in measurements:
                    payload[key] = measurements[key]
        if self.started:
            native_direct.diagnostics(self._direct_http_diagnostics)
        return payload

    def start(self):
        """Start the native bounded HTTP server and mDNS advertisement."""
        if self.started:
            return
        capabilities = build_capabilities_payload(self.device_id)
        info = self._info()
        arguments = dict(
            port=DIRECT_HTTP_PORT,
            hostname=self._device_hostname(),
            instance=info["device_name"],
            device_id=self.device_id,
            chip=str(info["chip"]),
            firmware_version=info["firmware_version"],
            protocol_version=PROTOCOL_VERSION,
            dns_sd_schema_version=DNS_SD_TXT_SCHEMA_VERSION,
            capabilities=capabilities,
            info=info,
            config=self._config(),
            status=self._status(),
            diagnostics=self._diagnostics(),
        )
        max_attempts = 3
        for attempt in range(max_attempts):
            try:
                native_direct.start(**arguments)
                break
            except OSError as exc:
                error_number = exc.args[0] if exc.args else None
                if error_number != ERRNO_EBUSY or attempt + 1 == max_attempts:
                    raise
                gc.collect()
                time.sleep_ms(100)
        self.started = True

    def refresh_status(self, now_ms=None):
        """Refresh the lightweight read-only status snapshot."""
        if not self.started:
            return
        if now_ms is None:
            now_ms = time.ticks_ms()
        native_direct.update_status(self._status(now_ms))

    def refresh_diagnostics(self, now_ms=None, diagnostics=None):
        """Refresh the full read-only diagnostics snapshot."""
        if not self.started:
            return
        if now_ms is None:
            now_ms = time.ticks_ms()
        # The native module encodes these mappings directly into its bounded
        # snapshot buffers.  Avoiding json.dumps here removes the largest
        # contiguous allocation from the periodic C3 diagnostics pass.
        native_direct.update_diagnostics(self._diagnostics(now_ms, diagnostics))

    def refresh_config(self):
        """Refresh configuration after a successful runtime recalibration."""
        if self.started:
            native_direct.update_config(self._config())

    def refresh_snapshots(self, now_ms=None, diagnostics=None):
        """Refresh status and diagnostics for compatibility with callers."""
        self.refresh_diagnostics(now_ms, diagnostics)
        self.refresh_status(now_ms)

    def take_recalibration_request(self):
        """Claim one queued Direct recalibration request for the main loop."""
        if not self.started:
            return False
        return bool(native_direct.take_recalibration_request())

    def complete_recalibration(self):
        """Allow the Direct worker to accept another recalibration request."""
        if self.started:
            native_direct.complete_recalibration()

    def publish_telemetry(self, movement_score, motion_state, threshold, now_ms=None):
        """Emit one canonical telemetry event after a detector evaluation."""
        if not self.started or not native_direct.has_event_client():
            return
        if now_ms is None:
            now_ms = time.ticks_ms()
        payload = self._telemetry_payload
        payload["timestamp_ms"] = now_ms
        payload["motion_state"] = "motion" if motion_state else "idle"
        payload["movement_score"] = movement_score
        payload["threshold"] = threshold
        payload["health"]["uptime_s"] = self._uptime_seconds(now_ms)
        native_direct.publish("telemetry", payload)

    def stop(self):
        """Stop Direct HTTP and remove its DNS-SD service."""
        if self.started:
            native_direct.stop()
            self.started = False
