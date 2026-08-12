# SPDX-License-Identifier: GPL-3.0-only
# Commercial licensing available under separate agreement; see LICENSING.md.
"""
Micro-ESPectre - Traffic Rate Controller

Adapts CSI traffic-generator send pacing from observed valid CSI load and
local socket backpressure. Mirrors the C++ TrafficRateController policy.

Author: Francesco Pace <francesco.pace@gmail.com>
"""


class TrafficRateController:
    """Shared adaptive pacing policy for the MicroPython traffic generator."""

    CONTROL_WINDOW_US = 2000000
    MIN_RATE_PPS = 5
    # No absolute ceiling: what the adaptive loop may chase is already bounded
    # relative to the configured target by MAX_RATE_NUMERATOR/DENOMINATOR, and
    # the real ceiling is the hardware.
    MAX_RATE_NUMERATOR = 5
    MAX_RATE_DENOMINATOR = 4
    TOLERANCE_PERCENT = 5
    SEVERE_DEFICIT_PERCENT = 50
    PACING_FLOOR_PERCENT = 70
    REDUCTION_PERCENT = 15
    BACKPRESSURE_PERCENT = 5
    MIN_BACKPRESSURE_EVENTS = 3
    OVERSUPPLY_WINDOWS_BEFORE_REDUCTION = 2
    ADJUSTMENT_SETTLE_US = CONTROL_WINDOW_US * 3

    def __init__(self):
        self.target_pps = 0
        self.current_pps = 0
        self.observed_pps = 0
        self.previous_accepted_csi_total = 0
        self.previous_send_success_total = 0
        self.previous_send_error_total = 0
        self.previous_observation_us = 0
        self.last_adjustment_us = 0
        self.oversupply_windows = 0
        self.adaptive_enabled = True

    def init(self, target_pps, adaptive_enabled):
        self.target_pps = int(target_pps)
        self.current_pps = int(target_pps)
        self.observed_pps = 0
        self.previous_accepted_csi_total = 0
        self.previous_send_success_total = 0
        self.previous_send_error_total = 0
        self.previous_observation_us = 0
        self.last_adjustment_us = 0
        self.oversupply_windows = 0
        self.adaptive_enabled = bool(adaptive_enabled)

    def observe(self, accepted_csi_total, send_success_total, send_error_total, now_us):
        accepted_csi_total = int(accepted_csi_total)
        send_success_total = int(send_success_total)
        send_error_total = int(send_error_total)
        now_us = int(now_us)

        if (
            self.previous_observation_us == 0
            or accepted_csi_total < self.previous_accepted_csi_total
            or send_success_total < self.previous_send_success_total
            or send_error_total < self.previous_send_error_total
        ):
            self.previous_accepted_csi_total = accepted_csi_total
            self.previous_send_success_total = send_success_total
            self.previous_send_error_total = send_error_total
            self.previous_observation_us = now_us
            return False

        elapsed_us = now_us - self.previous_observation_us
        if elapsed_us < self.CONTROL_WINDOW_US:
            return False

        accepted_delta = accepted_csi_total - self.previous_accepted_csi_total
        send_success_delta = send_success_total - self.previous_send_success_total
        send_error_delta = send_error_total - self.previous_send_error_total
        self.observed_pps = (accepted_delta * 1000000) // elapsed_us
        self.previous_accepted_csi_total = accepted_csi_total
        self.previous_send_success_total = send_success_total
        self.previous_send_error_total = send_error_total
        self.previous_observation_us = now_us

        if not self.adaptive_enabled or self.target_pps == 0:
            return False

        lower_bound = (self.target_pps * (100 - self.TOLERANCE_PERCENT)) // 100
        upper_bound = (self.target_pps * (100 + self.TOLERANCE_PERCENT) + 99) // 100
        minimum_rate = max(
            min(self.MIN_RATE_PPS, self.target_pps),
            (self.target_pps * self.PACING_FLOOR_PERCENT + 99) // 100,
        )
        maximum_rate = max(
            minimum_rate,
            (
                (self.target_pps * self.MAX_RATE_NUMERATOR + self.MAX_RATE_DENOMINATOR - 1)
                // self.MAX_RATE_DENOMINATOR
            ),
        )
        next_rate = self.current_pps
        send_attempt_delta = send_success_delta + send_error_delta
        backpressure_threshold = max(
            self.MIN_BACKPRESSURE_EVENTS,
            (send_attempt_delta * self.BACKPRESSURE_PERCENT + 99) // 100,
        )
        significant_backpressure = send_error_delta >= backpressure_threshold
        settling = (
            self.last_adjustment_us != 0
            and now_us - self.last_adjustment_us < self.ADJUSTMENT_SETTLE_US
        )
        additive_step = max(1, (self.target_pps * 2 + 99) // 100)

        if significant_backpressure:
            self.oversupply_windows = 0
            if not settling:
                next_rate = max(
                    minimum_rate,
                    (self.current_pps * (100 - self.REDUCTION_PERCENT)) // 100,
                )
        elif self.observed_pps * 100 < self.target_pps * self.SEVERE_DEFICIT_PERCENT:
            self.oversupply_windows = 0
        elif self.observed_pps > upper_bound:
            if self.oversupply_windows < self.OVERSUPPLY_WINDOWS_BEFORE_REDUCTION:
                self.oversupply_windows += 1
            if (
                not settling
                and self.oversupply_windows >= self.OVERSUPPLY_WINDOWS_BEFORE_REDUCTION
            ):
                next_rate = max(
                    minimum_rate,
                    (self.current_pps * (100 - self.REDUCTION_PERCENT)) // 100,
                )
                self.oversupply_windows = 0
        elif self.observed_pps < lower_bound:
            self.oversupply_windows = 0
            if not settling:
                next_rate = min(maximum_rate, self.current_pps + additive_step)
        else:
            self.oversupply_windows = 0
            if not settling and self.current_pps < self.target_pps:
                next_rate = min(self.target_pps, self.current_pps + additive_step)
            elif not settling and self.current_pps > self.target_pps:
                next_rate = max(self.target_pps, self.current_pps - additive_step)

        if next_rate == self.current_pps:
            return False
        self.current_pps = next_rate
        self.last_adjustment_us = now_us
        return True


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
