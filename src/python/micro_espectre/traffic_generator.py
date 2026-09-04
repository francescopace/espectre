# SPDX-License-Identifier: GPL-3.0-only
# Commercial licensing available under separate agreement; see LICENSING.md.
"""Micro-ESPectre native Wi-Fi traffic-generator facade."""

import time
import network

from espectre_native_traffic import TrafficGenerator as _NativeTrafficGenerator

try:
    from src.console_output import print_log
except ImportError:
    from console_output import print_log


TRAFFIC_RATE_MIN = 0          # Minimum rate (0=disabled)
TRAFFIC_RATE_MAX = 1000       # Maximum rate (packets per second)
MODE_PING = "ping"
MODE_DNS = "dns"
MODE_DNS_TCP = "dns_tcp"
TRAFFIC_MODES = (MODE_PING, MODE_DNS, MODE_DNS_TCP)


class TrafficGenerator:
    """Drive the shared firmware-native sensing traffic generator."""

    def __init__(self, mode=MODE_PING):
        self.running = False
        self.paused = False
        self.rate_pps = 0
        self.target_pps = 0
        self.packet_count = 0
        self.error_count = 0
        self.gateway_ip = None
        # Retained for compatibility with lifecycle code that previously
        # waited for a Python-owned socket to close.
        self.sock = None
        self.mode = self._normalize_mode(mode)
        self._native_traffic = _NativeTrafficGenerator()
        self.start_time = 0
        self.avg_loop_time_ms = 0
        self.actual_pps = 0

    @staticmethod
    def _normalize_mode(mode):
        """Validate and normalize the traffic-generator mode."""
        mode = (mode or MODE_PING).lower()
        if mode not in TRAFFIC_MODES:
            raise ValueError(f"Invalid traffic generator mode: {mode}")
        return mode

    def set_mode(self, mode):
        """Set traffic-generation mode while stopped."""
        if self.running:
            print_log("WARN", "Cannot change traffic generator mode while running")
            return False
        self.mode = self._normalize_mode(mode)
        return True

    @staticmethod
    def _get_gateway_ip():
        """Return the station gateway address, if Wi-Fi is connected."""
        try:
            wlan = network.WLAN(network.STA_IF)
            if not wlan.isconnected():
                return None
            ip_info = wlan.ifconfig()
            return ip_info[2] if len(ip_info) >= 3 else None
        except Exception as exc:
            print_log("ERROR", "Failed to get gateway IP: {}".format(exc))
            return None

    def start(self, rate_pps, max_retries=3, retry_delay=2, mode=None):
        """Start firmware-native sensing traffic at a fixed packet cadence."""
        if self.running:
            print_log("WARN", "Traffic generator already running")
            return False
        if rate_pps == 0:
            self.rate_pps = 0
            self.target_pps = 0
            return False
        if mode is not None:
            self.mode = self._normalize_mode(mode)
        if rate_pps < TRAFFIC_RATE_MIN or rate_pps > TRAFFIC_RATE_MAX:
            print_log(
                "ERROR",
                "Invalid rate: %s (must be %s-%s packets/sec)"
                % (rate_pps, TRAFFIC_RATE_MIN, TRAFFIC_RATE_MAX),
            )
            return False

        for attempt in range(1, max_retries + 1):
            self.gateway_ip = self._get_gateway_ip()
            if self.gateway_ip:
                break
            print_log(
                "WARN",
                "Failed to get gateway IP (attempt {}/{})".format(attempt, max_retries),
            )
            if attempt < max_retries:
                time.sleep(retry_delay)
        if not self.gateway_ip:
            print_log(
                "ERROR",
                "Could not get gateway IP after {} attempts".format(max_retries),
            )
            return False

        self.packet_count = 0
        self.error_count = 0
        self.actual_pps = 0
        self.avg_loop_time_ms = 0
        self.target_pps = rate_pps
        self.rate_pps = rate_pps
        self.start_time = time.ticks_ms()
        self.paused = False
        try:
            self.running = bool(
                self._native_traffic.start(self.gateway_ip, rate_pps, self.mode)
            )
        except Exception as exc:
            print_log("ERROR", "Failed to start native traffic generator: {}".format(exc))
            self.running = False
        if not self.running:
            self.rate_pps = 0
            self.target_pps = 0
        return self.running

    def stop(self):
        """Stop the native task and retain its final counters."""
        if not self.running:
            return
        try:
            self.packet_count = int(self._native_traffic.packet_count())
            self.error_count = int(self._native_traffic.error_count())
            self._native_traffic.stop()
        finally:
            self.running = False
            self.paused = False
            self.rate_pps = 0
            self.target_pps = 0

    def pause(self):
        """Pause sends without releasing the native socket."""
        if not self.running or not self._native_traffic.pause():
            return False
        self.paused = True
        return True

    def resume(self):
        """Resume sends on the retained native socket."""
        if not self.running or not self._native_traffic.resume():
            return False
        self.paused = False
        return True

    def is_running(self):
        """Return whether the native traffic task is active."""
        if self.running:
            self.running = bool(self._native_traffic.is_running())
        return self.running

    def get_packet_count(self):
        """Return the number of successful sends."""
        if self.running:
            return int(self._native_traffic.packet_count())
        return self.packet_count

    def get_rate(self):
        return self.rate_pps

    def get_target_rate(self):
        return self.target_pps

    def get_mode(self):
        return self.mode

    def get_actual_pps(self):
        return round(self.actual_pps, 1)

    def get_error_count(self):
        if self.running:
            return int(self._native_traffic.error_count())
        return self.error_count

    def get_avg_loop_time_ms(self):
        return round(self.avg_loop_time_ms, 2)
