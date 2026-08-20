# SPDX-License-Identifier: GPL-3.0-only
# Commercial licensing available under separate agreement; see LICENSING.md.
"""
Micro-ESPectre - Main Application

Main entry point for the Micro-ESPectre Wi-Fi CSI runtime.

Author: Francesco Pace <francesco.pace@gmail.com>
"""
import time
import gc
import os
import src.config as config
from src.config import NUM_SUBCARRIERS, EXPECTED_CSI_LEN
from src.device_utils import (
    CsiPayloadNormalizationState,
    CsiFrameTimestampFilter,
    DETECTOR_RESET_DROP_STREAK,
    DISPOSITION_SENSE,
    NORMALIZATION_DOUBLE_HT20,
    NORMALIZATION_DOUBLE_HT57_TO_64,
    NORMALIZATION_HT57_TO_64,
    assess_ht20_sensing_frame,
    normalize_ht20_csi_payload,
    csi_read_frame,
)
from src.detector_interface import (
    detector_needs_startup_calibration,
    get_detector_algorithm,
    get_detector_label,
    load_detector_class,
    normalize_detector_algorithm,
)
from src.runtime_motion_policy import RuntimeMotionPolicy
from src.wifi_bootstrap import cleanup_wifi, connect_wifi

HIGH_ACCURACY_DEFAULT_THRESHOLD = 0.5

# Global state for calibration mode and performance metrics
class GlobalState:
    def __init__(self):
        self.calibration_mode = False  # Flag to suspend main loop during calibration
        self.loop_time_us = 0  # Last loop iteration time in microseconds
        self.chip_type = None  # Detected chip type (S3, C6, etc.)
        self.current_channel = 0  # Track WiFi channel for change detection
        self.latest_diagnostics = None  # Cached MQTT stats CSI/Wi-Fi sample


g_state = GlobalState()


def print_heap(label):
    """Print a compact heap snapshot for boot/runtime profiling."""
    gc.collect()
    print(f"[MEM] {label}: free={gc.mem_free()} alloc={gc.mem_alloc()}")


def create_detector(detection_algorithm, window_packets):
    """
    Create the configured detector instance from the shared registry.

    The runtime keeps one common detector contract:
    - canonical algorithm key via `ALGORITHM`
    - shared `motion_metric` field in update_state()
    """
    try:
        detector_class = load_detector_class(detection_algorithm)
    except ValueError:
        raise ValueError(f"Unsupported DETECTION_ALGORITHM: {detection_algorithm}")

    threshold = 1.0 if detector_needs_startup_calibration(detection_algorithm) else HIGH_ACCURACY_DEFAULT_THRESHOLD

    print(f'Detection algorithm: {get_detector_label(detection_algorithm)}')
    return detector_class(
        window_size=window_packets,
        threshold=threshold,
        enable_lowpass=config.ENABLE_LOWPASS_FILTER,
        lowpass_cutoff=config.LOWPASS_CUTOFF,
        enable_hampel=config.ENABLE_HAMPEL_FILTER,
        hampel_window=config.HAMPEL_WINDOW,
        hampel_threshold=config.HAMPEL_THRESHOLD,
    )


def detector_uses_startup_calibration(detector):
    """Return True when the detector needs quiet-room startup calibration."""
    return detector_needs_startup_calibration(get_detector_algorithm(detector))


def run_startup_calibration(wlan, detector, traffic_gen):
    """
    Run startup calibration with fixed subcarriers.

    Args:
        wlan: WLAN instance
        detector: IDetector instance
        traffic_gen: TrafficGenerator instance
    Returns:
        bool: True if startup calibration completed
    """
    detector_name = detector.get_name()
    if not detector_uses_startup_calibration(detector):
        g_state.calibration_mode = True
        detector.set_threshold(HIGH_ACCURACY_DEFAULT_THRESHOLD)

        print('')
        print('='*60)
        print('High Accuracy Quick Boot')
        print('='*60)
        print(f'Free memory: {gc.mem_free()} bytes')

        print('')
        print('='*60)
        print('High Accuracy Quick Boot Complete!')
        print(f'   Subcarriers: {list(config.DEFAULT_SUBCARRIERS)}')
        print(f'   Threshold: {detector.get_threshold():.2f} (0-1 probability score)')
        print('   Startup path: AGC-active normalized pipeline')
        print('='*60)
        print('')

        g_state.calibration_mode = False
        return True
    g_state.calibration_mode = True

    gc.collect()
    detector.reset()

    print('')
    print('='*60)
    print('Startup Threshold Calibration')
    print('='*60)
    print(f'Free memory: {gc.mem_free()} bytes')
    print('Calibration: stay quiet first, then one short motion is OK.')

    from src.threshold import (
        StartupThresholdCalibrator,
        get_detector_auto_factor,
        get_detector_startup_gate,
    )
    from src.temporal_csi_sampler import (
        TemporalCsiSampler,
        minimum_valid_slots,
        temporal_window_slots,
    )

    detector_window_packets = detector.get_window_size()
    target_pps = max(1, int(getattr(config, 'CSI_TARGET_PPS', 100)))
    calibration_target_packets = temporal_window_slots(
        target_pps,
        getattr(config, 'CALIBRATION_DURATION_MS', 10_000),
    )
    calibration_tracker = StartupThresholdCalibrator(
        calibration_target_packets,
        auto_factor=get_detector_auto_factor(detector),
        gate_enabled=get_detector_startup_gate(detector),
    )
    begin_calibration = getattr(detector, "on_startup_calibration_begin", None)
    if callable(begin_calibration):
        begin_calibration()
    evaluation_interval_ms = max(1, int(getattr(config, 'EVALUATION_INTERVAL_MS', 250)))
    # Calibration evaluates on the same cadence steady-state detection does.
    # Mirrors the C++ EvaluationCadence shared by CsiPipeline and its
    # calibration interceptor.
    calibration_cadence = RuntimeMotionPolicy(
        evaluation_interval_ms=evaluation_interval_ms,
        segmentation_window_size_ms=getattr(config, 'SEGMENTATION_WINDOW_SIZE_MS', 1000),
    )
    temporal_sampler = TemporalCsiSampler(
        target_pps,
        getattr(config, 'SEGMENTATION_WINDOW_SIZE_MS', 1000),
    )
    set_minimum_valid = getattr(detector, "set_minimum_valid_samples", None)
    if callable(set_minimum_valid):
        set_minimum_valid(minimum_valid_slots(temporal_sampler.window_slots))

    print('')
    print('-'*60)
    print(f'{detector_name} Threshold Bootstrap ({calibration_target_packets} packets, evaluate every {evaluation_interval_ms} ms) [HT20: {NUM_SUBCARRIERS} SC]')
    print('-'*60)

    max_timeout_ms = 15000
    filtered_count = 0
    accepted_packet_count = 0
    calibration_progress = 0
    packets_since_evaluation = 0
    next_progress_report = 100
    last_packet_time = time.ticks_ms()
    rate_report_time = last_packet_time
    rate_previous_accepted = 0
    rate_previous_admitted = 0
    rate_previous_tx = traffic_gen.get_packet_count()
    get_dropped = getattr(wlan, "csi_dropped", None)
    rate_previous_dropped = int(get_dropped()) if callable(get_dropped) else 0
    collapse_logged = False
    remap_logged = False
    ht57_remap_buffer = bytearray(EXPECTED_CSI_LEN)
    normalization_state = CsiPayloadNormalizationState()
    frame_timestamp_filter = CsiFrameTimestampFilter()
    frame_result = None
    pending_csi_data = bytearray(EXPECTED_CSI_LEN)
    emitted_csi_data = bytearray(EXPECTED_CSI_LEN)
    pending_timestamp_us = 0
    # Reused per-frame assessment mapping: keeps this loop allocation-free.
    assessment_result = {}
    while not calibration_tracker.is_complete():
        frame = csi_read_frame(wlan, frame_result)
        if frame:
            frame_result = frame
            assessment = assess_ht20_sensing_frame(
                frame, frame[5], expected_len=EXPECTED_CSI_LEN, out=assessment_result
            )
            if assessment["disposition"] != DISPOSITION_SENSE:
                filtered_count += 1
                if filtered_count % 100 == 1:
                    print(
                        f"[WARN] Filtered {filtered_count} packets before calibration "
                        "(reason={}, len={})".format(
                            assessment["reason_code"], assessment["raw_len"]
                        )
                    )
                del frame
                time.sleep_ms(1)
                if time.ticks_diff(time.ticks_ms(), last_packet_time) >= max_timeout_ms:
                    print(f"Timeout waiting for valid CSI packets (collected {calibration_progress}/{calibration_target_packets})")
                    print("Startup calibration aborted")
                    detector.reset()
                    g_state.calibration_mode = False
                    return False
                continue

            csi_data, _, remap_tag = normalize_ht20_csi_payload(
                frame[5], EXPECTED_CSI_LEN,
                remap_buffer=ht57_remap_buffer,
                assessment=assessment,
                state=normalization_state,
            )
            if csi_data is None:
                filtered_count += 1
                del frame
                time.sleep_ms(1)
                if time.ticks_diff(time.ticks_ms(), last_packet_time) >= max_timeout_ms:
                    print(f"Timeout waiting for valid CSI packets (collected {calibration_progress}/{calibration_target_packets})")
                    print("Startup calibration aborted")
                    detector.reset()
                    g_state.calibration_mode = False
                    return False
                continue

            if not frame_timestamp_filter.accept(frame):
                del frame
                time.sleep_ms(1)
                continue

            if remap_tag in (NORMALIZATION_DOUBLE_HT20, NORMALIZATION_DOUBLE_HT57_TO_64) and not collapse_logged:
                print("[INFO] CSI double-length collapse active: 256->128 and/or 228->114")
                collapse_logged = True
            if remap_tag in (NORMALIZATION_HT57_TO_64, NORMALIZATION_DOUBLE_HT57_TO_64) and not remap_logged:
                print("[INFO] CSI remap active: 57->64 SC (left_pad=4, right_pad=3)")
                remap_logged = True
            del frame
            current_timestamp_us = frame_result[4]
            accepted_packet_count += 1
            emitted = temporal_sampler.admit(current_timestamp_us)
            if emitted:
                emitted_csi_data[:] = pending_csi_data
                emitted_timestamp_us = pending_timestamp_us
            if temporal_sampler.selected_current:
                pending_csi_data[:] = csi_data
                pending_timestamp_us = current_timestamp_us
            if emitted:
                if temporal_sampler.reset_required:
                    detector.reset()
                    calibration_cadence.reset()
                    calibration_tracker = StartupThresholdCalibrator(
                        calibration_target_packets,
                        auto_factor=get_detector_auto_factor(detector),
                        gate_enabled=get_detector_startup_gate(detector),
                    )
                    begin_calibration = getattr(detector, "on_startup_calibration_begin", None)
                    if callable(begin_calibration):
                        begin_calibration()
                    packets_since_evaluation = 0
                if temporal_sampler.missing_slots_before:
                    advance_missing = getattr(detector, "advance_missing_slots", None)
                    if callable(advance_missing):
                        advance_missing(temporal_sampler.missing_slots_before)
                detector.process_packet(
                    emitted_csi_data,
                    config.DEFAULT_SUBCARRIERS,
                    timestamp_us=emitted_timestamp_us,
                )
                packets_since_evaluation += 1
                last_packet_time = time.ticks_ms()
                calibration_cadence.note_arrival(emitted_timestamp_us)
            if temporal_sampler.gap_reset_required:
                detector.reset()
                calibration_cadence.reset()
                calibration_tracker = StartupThresholdCalibrator(
                    calibration_target_packets,
                    auto_factor=get_detector_auto_factor(detector),
                    gate_enabled=get_detector_startup_gate(detector),
                )
                begin_calibration = getattr(detector, "on_startup_calibration_begin", None)
                if callable(begin_calibration):
                    begin_calibration()
                packets_since_evaluation = 0
            if not emitted:
                time.sleep_ms(1)
                continue

            if not calibration_cadence.should_evaluate():
                continue
            calibration_cadence.after_evaluation()

            detector.update_state()
            if detector.is_ready():
                calibration_tracker.observe_detector(
                    detector,
                    packet_weight=packets_since_evaluation,
                )
            packets_since_evaluation = 0
            calibration_progress = calibration_tracker.packet_count
            if calibration_progress >= next_progress_report:
                current_time = time.ticks_ms()
                elapsed_ms = max(1, time.ticks_diff(current_time, rate_report_time))
                rate_scale = 1000.0 / elapsed_ms
                admitted_total = temporal_sampler.accepted_packets
                tx_total = traffic_gen.get_packet_count()
                dropped_total = int(get_dropped()) if callable(get_dropped) else 0
                admitted_pps = (admitted_total - rate_previous_admitted) * rate_scale
                accepted_pps = (accepted_packet_count - rate_previous_accepted) * rate_scale
                tx_pps = (tx_total - rate_previous_tx) * rate_scale
                dropped_pps = (dropped_total - rate_previous_dropped) * rate_scale
                current_mv = detector.get_motion_metric() if detector.is_ready() else None
                movement_text = current_mv if current_mv is not None else "--"
                print(
                    "Calibration {}/{} | mvmt:{} thr:{:.6f} | "
                    "csi:{:.1f}/{:.1f}pps tx:{:.1f}pps drop:{:.1f}pps".format(
                        calibration_progress,
                        calibration_target_packets,
                        movement_text,
                        detector.get_threshold(),
                        admitted_pps,
                        accepted_pps,
                        tx_pps,
                        dropped_pps,
                    )
                )
                rate_report_time = current_time
                rate_previous_accepted = accepted_packet_count
                rate_previous_admitted = admitted_total
                rate_previous_tx = tx_total
                rate_previous_dropped = dropped_total
                while next_progress_report <= calibration_progress:
                    next_progress_report += 100
        else:
            time.sleep_ms(1)
            if time.ticks_diff(time.ticks_ms(), last_packet_time) >= max_timeout_ms:
                print(f"Timeout waiting for CSI packets (collected {calibration_progress}/{calibration_target_packets})")
                print("Startup calibration aborted")
                detector.reset()
                g_state.calibration_mode = False
                return False

    gc.collect()
    success = calibration_tracker.is_successful()
    if success:
        startup_threshold, threshold_formula = calibration_tracker.calculate_threshold()
        detector.set_adaptive_threshold(startup_threshold)
        startup_threshold = detector.get_threshold()
        threshold_source = f"automatic ({threshold_formula})"
        print(f'Startup threshold: {startup_threshold:.4f} ({threshold_source})')

        detector.reset()

        print('')
        print('='*60)
        print(f'{detector_name} Startup Calibration Complete!')
        print(f'   Subcarriers: {list(config.DEFAULT_SUBCARRIERS)}')
        print(f'   Threshold: {detector.get_threshold():.4f} ({threshold_source})')
        print('='*60)
        print('')
    else:
        print('')
        print('='*60)
        print(f'{detector_name} Startup Calibration Failed')
        print(f'   Keeping threshold: {detector.get_threshold():.4f}')
        print(f'   Subcarriers: {list(config.DEFAULT_SUBCARRIERS)}')
        print('='*60)
        print('')

    g_state.calibration_mode = False
    return success


def maybe_run_ha_recalibration(mqtt_handler, wlan, detector, traffic_gen, runtime_policy, temporal_sampler):
    """Run one HA-requested recalibration from the main loop, never from MQTT."""
    if mqtt_handler is None or not mqtt_handler.take_recalibrate_request():
        return False
    success = run_startup_calibration(wlan, detector, traffic_gen)
    if not success:
        print('[WARN] Recalibration failed; keeping current threshold')
    runtime_policy.reset()
    temporal_sampler.reset()
    motion_metric = detector.get_motion_metric() if detector.is_ready() else 0.0
    mqtt_handler.finish_recalibration(motion_metric, detector.get_threshold())
    return True


def get_chip_type():
    """Extract short chip type from os.uname().machine."""
    machine = os.uname().machine.upper()
    # Check for specific variants first
    for variant in ['S3', 'C3', 'C5', 'C6']:
        if variant in machine:
            return variant
    # Fallback to ESP32 base
    if 'ESP32' in machine:
        return 'ESP32'
    return machine


def restart_traffic_generator(traffic_gen):
    """Restart the traffic generator after calibration-sensitive work completes."""
    if not traffic_gen or not getattr(config, 'TRAFFIC_GENERATOR_ENABLED', True):
        return

    time.sleep(1)  # Give WiFi/MQTT stack time to settle before reopening raw socket.
    gc.collect()
    target_pps = max(1, int(getattr(config, 'CSI_TARGET_PPS', 100)))
    if not traffic_gen.start(target_pps):
        print("Warning: Failed to restart traffic generator, retrying...")
        time.sleep(2)
        gc.collect()
        traffic_gen.start(target_pps)


def main(wlan=None):
    """Main application loop"""
    print('Micro-ESPectre starting...')
    print_heap('boot')

    # Detect chip type
    g_state.chip_type = get_chip_type()
    print(f'Detected chip: {g_state.chip_type}')

    # Connect to WiFi
    if wlan is None:
        wlan = connect_wifi()
    print_heap('after_connect_wifi')

    # Detector capacity is fixed by the configured temporal grid. Measured
    # delivery rate is diagnostic only and never reconstructs the detector.
    detection_algorithm = normalize_detector_algorithm(
        getattr(config, 'DETECTION_ALGORITHM', 'lightweight')
    )
    from src.temporal_csi_sampler import (
        TemporalCsiSampler,
        minimum_valid_slots,
        temporal_window_slots,
    )
    target_pps = max(1, int(getattr(config, 'CSI_TARGET_PPS', 100)))
    detector_window_packets = temporal_window_slots(
        target_pps,
        getattr(config, 'SEGMENTATION_WINDOW_SIZE_MS', 1000),
    )

    # Initialize and start traffic generator (target CSI rate from config.py)
    gc.collect()  # Free memory before creating socket
    traffic_mode = getattr(config, 'TRAFFIC_GENERATOR_MODE', 'ping')
    from src.traffic_generator import TrafficGenerator
    traffic_gen = TrafficGenerator(mode=traffic_mode)
    print_heap('after_traffic_gen_init')
    observed_interval_us = None
    if getattr(config, 'TRAFFIC_GENERATOR_ENABLED', True):
        if not traffic_gen.start(target_pps):
            print("FATAL: Traffic generator failed to start - CSI will not work")
            print("Check WiFi connection and gateway availability")
            import machine
            time.sleep(5)
            machine.reset()  # Reboot and retry

        print(f'Traffic generator started ({traffic_mode}, target={target_pps} CSI pps)')
        print_heap('after_traffic_gen_start')

        # Verify CSI packets are flowing with retry logic
        max_tg_retries = 3
        for tg_attempt in range(max_tg_retries):
            time.sleep(2)  # Wait for traffic to start generating CSI packets

            print('Waiting for CSI packets...')
            csi_received = 0
            csi_timestamps = []
            frame_result = None
            for _ in range(100):  # Max 100 attempts (~5 seconds)
                frame = csi_read_frame(wlan, frame_result)
                if frame:
                    frame_result = frame
                    csi_received += 1
                    csi_timestamps.append(int(frame[4]))
                    if csi_received >= 17:
                        break
                time.sleep(0.05)

            if csi_received >= 17:
                deltas = []
                for previous, current in zip(csi_timestamps, csi_timestamps[1:]):
                    delta = (current - previous) % (1 << 32)
                    if 0 < delta < (1 << 31):
                        deltas.append(delta)
                if deltas:
                    observed_interval_us = max(1, sum(deltas) // len(deltas))
                break  # Success

            if tg_attempt < max_tg_retries - 1:
                print(f'WARNING: Only {csi_received} CSI packets - restarting TG (attempt {tg_attempt + 2}/{max_tg_retries})')
                traffic_gen.stop()
                time.sleep(1)
                traffic_gen.start(target_pps)
            else:
                print(f'FATAL: No CSI packets after {max_tg_retries} attempts - cannot operate without traffic')
                print('Please check WiFi connection and retry')
                import sys
                sys.exit(1)
        print_heap('after_csi_flow_check')

    else:
        print('Waiting for external CSI packets...')
        csi_timestamps = []
        frame_result = None
        for _ in range(100):
            frame = csi_read_frame(wlan, frame_result)
            if frame:
                frame_result = frame
                csi_timestamps.append(int(frame[4]))
                if len(csi_timestamps) >= 17:
                    break
            time.sleep(0.05)
        deltas = []
        for previous, current in zip(csi_timestamps, csi_timestamps[1:]):
            delta = (current - previous) % (1 << 32)
            if 0 < delta < (1 << 31):
                deltas.append(delta)
        if not deltas:
            raise RuntimeError('External CSI traffic did not provide advancing timestamps')
        observed_interval_us = max(1, sum(deltas) // len(deltas))

    detector = create_detector(detection_algorithm, detector_window_packets)
    set_minimum_valid = getattr(detector, "set_minimum_valid_samples", None)
    if callable(set_minimum_valid):
        set_minimum_valid(minimum_valid_slots(detector_window_packets))
    print(f'Detector window: {detector_window_packets} samples for {getattr(config, "SEGMENTATION_WINDOW_SIZE_MS", 1000)} ms')
    print_heap('after_detector_init')

    # Detector allocation can fragment the original ESP32 heap enough that the
    # socket opened during the flow probe starts returning ENOMEM. Reopen it
    # against the post-allocation heap before calibration begins.
    if getattr(config, 'TRAFFIC_GENERATOR_ENABLED', True):
        traffic_gen.stop()
        time.sleep(1)
        gc.collect()
        if not traffic_gen.start(target_pps):
            cleanup_wifi(wlan)
            raise RuntimeError("Traffic generator restart failed after detector initialization")
        time.sleep(1)

    runtime_policy = RuntimeMotionPolicy(
        evaluation_interval_ms=getattr(config, 'EVALUATION_INTERVAL_MS', 250),
        motion_on_hits=getattr(config, 'MOTION_ON_HITS', 4),
        motion_off_hits=getattr(config, 'MOTION_OFF_HITS', 3),
        segmentation_window_size_ms=getattr(config, 'SEGMENTATION_WINDOW_SIZE_MS', 1000),
    )
    mqtt_enabled = getattr(config, 'MQTT_ENABLED', True)
    mqtt_handler = None
    if mqtt_enabled:
        # Reserve the transport object and native buffers before calibration
        # fragments the MicroPython and ESP-IDF heaps. The MQTT task itself is
        # still started only after sensing has calibrated and stabilized.
        from src.mqtt.handler import MQTTHandler
        mqtt_handler = MQTTHandler(
            config,
            detector,
            wlan,
            g_state,
            runtime_policy=runtime_policy,
            traffic_generator=traffic_gen,
        )
        mqtt_handler.prepare()
        print_heap('after_mqtt_handler_init')

    calibration_ok = run_startup_calibration(
        wlan,
        detector,
        traffic_gen,
    )
    if not calibration_ok:
        if traffic_gen.is_running():
            traffic_gen.stop()
        cleanup_wifi(wlan)
        raise RuntimeError("Startup calibration failed")
    print_heap('after_calibration')

    # The calibration helper is large and is only needed again on an explicit
    # recalibration request. Release it before importing MQTT and HA discovery;
    # run_startup_calibration() will load it again on demand.
    import sys
    sys.modules.pop("src.threshold", None)
    # Retain the already allocated traffic socket. Reopening a raw socket after
    # MQTT setup fragments the original ESP32 heap and fails with ENOMEM.
    gc.collect()

    if mqtt_handler is not None:
        traffic_paused_for_mqtt = traffic_gen.pause()
        if traffic_paused_for_mqtt:
            time.sleep_ms(50)
        mqtt_handler.connect()
        print_heap('after_mqtt_connect')
        if traffic_paused_for_mqtt:
            traffic_gen.resume()

    else:
        print('MQTT disabled')

    from src.branding import ASCII_BANNER
    from src.console_output import format_detection_publish_line
    from src.runtime_diagnostics import (
        RuntimeDebugTelemetry,
        RuntimeDiagnosticsSampler,
        collect_runtime_diagnostics_snapshot,
        empty_diagnostics_sample,
        wifi_csi_dropped,
        wifi_rssi_dbm,
    )
    if getattr(config, 'TRAFFIC_GENERATOR_ENABLED', True) and not traffic_gen.is_running():
        restart_traffic_generator(traffic_gen)

    print('')
    print(ASCII_BANNER)
    print('')

    # Force garbage collection before main loop
    gc.collect()
    print(f'Free memory before main loop: {gc.mem_free()} bytes')

    # Main CSI processing loop with integrated MQTT publishing
    processed_packet_count = 0
    callback_packet_count = 0
    filtered_count = 0  # Packets with wrong SC count
    last_publish_time = time.ticks_ms()
    collapse_logged = False
    remap_logged = False
    ht57_remap_buffer = bytearray(EXPECTED_CSI_LEN)
    normalization_state = CsiPayloadNormalizationState()
    frame_timestamp_filter = CsiFrameTimestampFilter()
    out_of_order_count = 0
    frame_result = None
    format_drop_streak = 0
    last_normalization_id = None
    # Reused per-frame assessment mapping: keeps the hot loop allocation-free.
    assessment_result = {}
    publish_interval_ms = max(1, int(getattr(config, 'PUBLISH_INTERVAL_MS', 1000)))
    temporal_sampler = TemporalCsiSampler(
        target_pps,
        getattr(config, 'SEGMENTATION_WINDOW_SIZE_MS', 1000),
    )
    diagnostics_sampler = RuntimeDiagnosticsSampler()
    debug_telemetry_enabled = bool(getattr(config, 'DEBUG_TELEMETRY', False))
    debug_telemetry = RuntimeDebugTelemetry(enabled=True) if debug_telemetry_enabled else None
    diagnostics_sampler.reset(
        collect_runtime_diagnostics_snapshot(
            traffic_generator=traffic_gen,
            callback_total=wifi_csi_dropped(wlan),
            accepted_total=0,
            admitted_total=0,
            filtered_total=0,
            missing_slots_total=0,
            excess_total=0,
            stale_total=0,
            out_of_order_total=0,
            occupancy_slots=0,
            window_slots=temporal_sampler.window_slots,
            wifi_channel=g_state.current_channel,
            rssi_dbm=wifi_rssi_dbm(wlan),
        ),
        time.ticks_ms(),
    )
    g_state.latest_diagnostics = empty_diagnostics_sample(
        wifi_channel=g_state.current_channel,
        wifi_rssi_dbm=wifi_rssi_dbm(wlan),
    )
    pending_csi_data = bytearray(EXPECTED_CSI_LEN)
    emitted_csi_data = bytearray(EXPECTED_CSI_LEN)
    pending_timestamp_us = 0
    latest_motion_metric = 0.0
    latest_threshold = detector.get_threshold()
    latest_effective_state = runtime_policy.effective_state

    try:
        while True:
            loop_start = time.ticks_us()

            # Suspend main loop during calibration
            if g_state.calibration_mode:
                time.sleep_ms(1000) # Sleep for 1 second to yield CPU
                continue

            if maybe_run_ha_recalibration(
                mqtt_handler, wlan, detector, traffic_gen, runtime_policy, temporal_sampler
            ):
                latest_motion_metric = mqtt_handler.last_variance
                latest_threshold = mqtt_handler.last_threshold
                latest_effective_state = runtime_policy.effective_state
                continue

            current_time = time.ticks_ms()
            time_delta = time.ticks_diff(current_time, last_publish_time)
            if time_delta >= publish_interval_ms:
                g_state.latest_diagnostics = diagnostics_sampler.sample(
                    collect_runtime_diagnostics_snapshot(
                        traffic_generator=traffic_gen,
                        callback_total=(
                            callback_packet_count + wifi_csi_dropped(wlan)
                        ),
                        accepted_total=processed_packet_count,
                        admitted_total=temporal_sampler.accepted_packets,
                        filtered_total=filtered_count + out_of_order_count,
                        missing_slots_total=temporal_sampler.missing_slots,
                        excess_total=temporal_sampler.excess_packets,
                        stale_total=temporal_sampler.stale_packets,
                        out_of_order_total=temporal_sampler.out_of_order_packets,
                        occupancy_slots=temporal_sampler.occupancy_slots,
                        window_slots=temporal_sampler.window_slots,
                        wifi_channel=g_state.current_channel,
                        rssi_dbm=wifi_rssi_dbm(wlan),
                    ),
                    current_time,
                )
                status_line = format_detection_publish_line(
                    diagnostics=g_state.latest_diagnostics,
                    motion_metric=latest_motion_metric,
                    threshold=latest_threshold,
                    effective_state=latest_effective_state,
                )
                if debug_telemetry_enabled:
                    status_line = f"I ({current_time}) micro_espectre: {status_line}"
                print(status_line)
                if debug_telemetry_enabled:
                    assert debug_telemetry is not None
                    debug_line = debug_telemetry.format_if_due(current_time, gc.mem_free())
                    if debug_line is not None:
                        print(f"D ({current_time}) micro_espectre: {debug_line}")
                if mqtt_handler is not None:
                    mqtt_handler.publish_live_ha(
                        latest_motion_metric,
                        latest_effective_state,
                        latest_threshold,
                    )
                    mqtt_handler.check_messages()
                    if maybe_run_ha_recalibration(
                        mqtt_handler, wlan, detector, traffic_gen, runtime_policy, temporal_sampler
                    ):
                        latest_motion_metric = mqtt_handler.last_variance
                        latest_threshold = mqtt_handler.last_threshold
                        latest_effective_state = runtime_policy.effective_state
                        last_publish_time = current_time
                        continue
                last_publish_time = current_time

            frame = csi_read_frame(wlan, frame_result)

            if frame:
                frame_result = frame
                callback_packet_count += 1
                assessment = assess_ht20_sensing_frame(
                    frame, frame[5], expected_len=EXPECTED_CSI_LEN, out=assessment_result
                )
                if assessment["disposition"] != DISPOSITION_SENSE:
                    filtered_count += 1
                    format_drop_streak += 1
                    if filtered_count % 100 == 1:
                        print(
                            f"[WARN] Filtered {filtered_count} packets before detection "
                            "(reason={}, len={})".format(
                                assessment["reason_code"], assessment["raw_len"]
                            )
                        )
                    del frame
                    g_state.loop_time_us = time.ticks_diff(time.ticks_us(), loop_start)
                    if debug_telemetry_enabled:
                        assert debug_telemetry is not None
                        debug_telemetry.record_loop_duration(g_state.loop_time_us)
                    time.sleep_us(100)
                    continue

                csi_data, _, remap_tag = normalize_ht20_csi_payload(
                    frame[5], EXPECTED_CSI_LEN,
                    remap_buffer=ht57_remap_buffer,
                    assessment=assessment,
                    state=normalization_state,
                )
                if csi_data is None:
                    filtered_count += 1
                    format_drop_streak += 1
                    del frame
                    g_state.loop_time_us = time.ticks_diff(time.ticks_us(), loop_start)
                    if debug_telemetry_enabled:
                        assert debug_telemetry is not None
                        debug_telemetry.record_loop_duration(g_state.loop_time_us)
                    time.sleep_us(100)
                    continue

                if not frame_timestamp_filter.accept(frame):
                    out_of_order_count += 1
                    if out_of_order_count % 100 == 1:
                        print(f"[WARN] Filtered {out_of_order_count} duplicate or out-of-order CSI frames")
                    del frame
                    g_state.loop_time_us = time.ticks_diff(time.ticks_us(), loop_start)
                    if debug_telemetry_enabled:
                        assert debug_telemetry is not None
                        debug_telemetry.record_loop_duration(g_state.loop_time_us)
                    time.sleep_us(100)
                    continue

                should_reset_detector = (
                    format_drop_streak >= DETECTOR_RESET_DROP_STREAK
                    or (
                        last_normalization_id is not None
                        and assessment["normalization_id"] != last_normalization_id
                    )
                )
                format_drop_streak = 0
                if should_reset_detector:
                    print("[WARN] CSI format stream changed after incompatible packets, resetting detection buffer")
                    detector.reset()
                    runtime_policy.reset()
                    frame_timestamp_filter.reset()
                last_normalization_id = assessment["normalization_id"]

                if remap_tag in (NORMALIZATION_DOUBLE_HT20, NORMALIZATION_DOUBLE_HT57_TO_64) and not collapse_logged:
                    print("[INFO] CSI double-length collapse active: 256->128 and/or 228->114")
                    collapse_logged = True
                if remap_tag in (NORMALIZATION_HT57_TO_64, NORMALIZATION_DOUBLE_HT57_TO_64) and not remap_logged:
                    print("[INFO] CSI remap active: 57->64 SC (left_pad=4, right_pad=3)")
                    remap_logged = True
                packet_channel = frame[1]

                del frame

                processed_packet_count += 1

                if g_state.current_channel != 0 and packet_channel != g_state.current_channel:
                    print(f"[WARN] WiFi channel changed: {g_state.current_channel} -> {packet_channel}, resetting detection buffer")
                    detector.reset()
                    runtime_policy.reset()
                    temporal_sampler.clear_history()
                    normalization_state.reset()
                g_state.current_channel = packet_channel

                current_timestamp_us = frame_result[4]
                emitted = temporal_sampler.admit(current_timestamp_us)
                if emitted:
                    emitted_csi_data[:] = pending_csi_data
                    emitted_timestamp_us = pending_timestamp_us
                if temporal_sampler.selected_current:
                    pending_csi_data[:] = csi_data
                    pending_timestamp_us = current_timestamp_us
                if emitted:
                    if temporal_sampler.reset_required:
                        detector.reset()
                        runtime_policy.reset()
                        latest_motion_metric = 0.0
                        latest_effective_state = runtime_policy.effective_state
                    if temporal_sampler.missing_slots_before:
                        advance_missing = getattr(detector, "advance_missing_slots", None)
                        if callable(advance_missing):
                            advance_missing(temporal_sampler.missing_slots_before)

                    detector.process_packet(
                        emitted_csi_data,
                        config.DEFAULT_SUBCARRIERS,
                        timestamp_us=emitted_timestamp_us,
                    )
                    runtime_policy.note_arrival(emitted_timestamp_us)
                if temporal_sampler.gap_reset_required:
                    detector.reset()
                    runtime_policy.reset()
                    latest_motion_metric = 0.0
                    latest_effective_state = runtime_policy.effective_state
                if not emitted:
                    g_state.loop_time_us = time.ticks_diff(time.ticks_us(), loop_start)
                    if debug_telemetry_enabled:
                        assert debug_telemetry is not None
                        debug_telemetry.record_loop_duration(g_state.loop_time_us)
                    time.sleep_us(100)
                    continue

                if runtime_policy.should_evaluate():
                    detection_start = time.ticks_us() if debug_telemetry_enabled else None
                    metrics = detector.update_state()
                    if detection_start is not None:
                        assert debug_telemetry is not None
                        debug_telemetry.record_detection_duration(
                            time.ticks_diff(time.ticks_us(), detection_start),
                        )
                    effective_state, _ = runtime_policy.apply_state(metrics['state'])
                    runtime_policy.after_evaluation()
                    latest_motion_metric = metrics.get('motion_metric', 0.0)
                    latest_threshold = metrics['threshold']
                    latest_effective_state = effective_state

                # Update loop time metric
                g_state.loop_time_us = time.ticks_diff(time.ticks_us(), loop_start)
                if debug_telemetry_enabled:
                    assert debug_telemetry is not None
                    debug_telemetry.record_loop_duration(g_state.loop_time_us)

                time.sleep_us(100)
            else:
                # Update loop time metric (idle iteration)
                g_state.loop_time_us = time.ticks_diff(time.ticks_us(), loop_start)
                if debug_telemetry_enabled:
                    assert debug_telemetry is not None
                    debug_telemetry.record_loop_duration(g_state.loop_time_us)

                time.sleep_us(100)

    except KeyboardInterrupt:
        print('\n\nStopping...')

    finally:
        print('Cleaning up...')
        if mqtt_handler is not None:
            mqtt_handler.disconnect()
        if traffic_gen.is_running():
            traffic_gen.stop()
        cleanup_wifi(wlan)

if __name__ == '__main__':
    main()
