# SPDX-License-Identifier: GPL-3.0-only
# Commercial licensing available under separate agreement; see LICENSING.md.
"""
Micro-ESPectre - CSI pacing health monitor

Passive telemetry for sustained CSI callback deficits under pacing on the
original ESP32. Send-rate control lives on the host collector only.

Author: Francesco Pace <francesco.pace@gmail.com>
"""


class CsiPacingHealthMonitor:
    """Report sustained CSI callback deficits under pacing on the original ESP32.

    Passive telemetry only: instrumented runs showed capture is never wedged
    when callbacks stall (every collapse was AP-side), so the rearm action was
    removed per the 2026-07-20 broadcast-pacing ADR. Stalls are logged so the
    condition stays observable.
    """

    SAMPLE_INTERVAL_MS = 2000
    STALL_LOG_COOLDOWN_MS = 10000
    MIN_PACING_PACKETS = 20
    MIN_CALLBACK_PERCENT = 50
    LOW_SUPPLY_WINDOWS_BEFORE_REPORT = 2

    ACTION_NONE = "none"
    ACTION_STALL_REPORTED = "stall_reported"

    def __init__(self, enabled=False):
        self.enabled = bool(enabled)
        self.prev_pacing_total = 0
        self.prev_callback_total = 0
        self.last_sample_ms = 0
        self.last_stall_log_ms = 0
        self.low_supply_windows = 0
        self.baseline_valid = False

    def reset(self):
        self.prev_pacing_total = 0
        self.prev_callback_total = 0
        self.last_sample_ms = 0
        self.last_stall_log_ms = 0
        self.low_supply_windows = 0
        self.baseline_valid = False

    def maintain(self, pacing_total, callback_total, now_ms):
        if not self.enabled:
            return self.ACTION_NONE

        pacing_total = int(pacing_total)
        callback_total = int(callback_total)
        now_ms = int(now_ms)

        if not self.baseline_valid:
            self.prev_pacing_total = pacing_total
            self.prev_callback_total = callback_total
            self.last_sample_ms = now_ms
            self.baseline_valid = True
            return self.ACTION_NONE

        if now_ms - self.last_sample_ms < self.SAMPLE_INTERVAL_MS:
            return self.ACTION_NONE

        pacing_delta = (
            pacing_total - self.prev_pacing_total
            if pacing_total >= self.prev_pacing_total
            else pacing_total
        )
        callback_delta = (
            callback_total - self.prev_callback_total
            if callback_total >= self.prev_callback_total
            else callback_total
        )
        self.prev_pacing_total = pacing_total
        self.prev_callback_total = callback_total
        self.last_sample_ms = now_ms

        if (
            pacing_delta < self.MIN_PACING_PACKETS
            or callback_delta * 100 >= pacing_delta * self.MIN_CALLBACK_PERCENT
        ):
            self.low_supply_windows = 0
            return self.ACTION_NONE

        if self.low_supply_windows < self.LOW_SUPPLY_WINDOWS_BEFORE_REPORT:
            self.low_supply_windows += 1
        if self.low_supply_windows < self.LOW_SUPPLY_WINDOWS_BEFORE_REPORT:
            return self.ACTION_NONE
        if (
            self.last_stall_log_ms != 0
            and now_ms - self.last_stall_log_ms < self.STALL_LOG_COOLDOWN_MS
        ):
            return self.ACTION_NONE

        print(
            "CSI callback supply stalled: callbacks=%d pacing=%d"
            % (callback_delta, pacing_delta)
        )
        self.last_stall_log_ms = now_ms
        self.low_supply_windows = 0
        return self.ACTION_STALL_REPORTED
