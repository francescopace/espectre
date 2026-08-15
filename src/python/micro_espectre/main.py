# SPDX-License-Identifier: GPL-3.0-only
# Commercial licensing available under separate agreement; see LICENSING.md.
"""
Micro-ESPectre - Main Application

Main entry point for the Micro-ESPectre Wi-Fi CSI runtime.

Author: Francesco Pace <francesco.pace@gmail.com>
"""
import network
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
from src.branding import ASCII_BANNER
from src.console_output import format_calibration_status_line, format_detection_publish_line
from src.detector_interface import (
    detector_needs_startup_calibration,
    get_detector_algorithm,
    get_detector_label,
    load_detector_class,
    normalize_detector_algorithm,
)
from src.traffic_rate_controller import CsiPacingHealthMonitor

HIGH_ACCURACY_DEFAULT_THRESHOLD = 0.5

# Global state for calibration mode and performance metrics
class GlobalState:
    def __init__(self):
        self.calibration_mode = False  # Flag to suspend main loop during calibration
        self.loop_time_us = 0  # Last loop iteration time in microseconds
        self.chip_type = None  # Detected chip type (S3, C6, etc.)
        self.current_channel = 0  # Track WiFi channel for change detection


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


def cleanup_wifi(wlan):
    """
    Force cleanup of WiFi/CSI state.
    
    Handles stale state from previous interrupted runs (e.g., Ctrl+C without proper cleanup).
    Safe to call even if WiFi/CSI is not active.
    
    Args:
        wlan: WLAN instance
    """
    if not wlan.active():
        return
    
    print("Forcing WiFi/CSI cleanup...")
    
    # Disable CSI first (may fail if not enabled, that's ok)
    try:
        wlan.csi_disable()
    except Exception:
        pass
    
    # Disconnect if connected
    if wlan.isconnected():
        wlan.disconnect()
    
    # Deactivate interface
    wlan.active(False)
    time.sleep(1)  # Wait for hardware to settle


def print_wifi_status(wlan):
    """Print WiFi connection status with configuration details."""
    ip = wlan.ifconfig()[0]
    
    # Protocol decode (HT20 only: 802.11b/g/n)
    PROTOCOL_NAMES = {
        network.MODE_11B: 'b',
        network.MODE_11G: 'g', 
        network.MODE_11N: 'n',
    }
    
    proto_val = wlan.config('protocol')
    modes = [name for bit, name in PROTOCOL_NAMES.items() if proto_val & bit]
    protocol_str = '802.11' + '/'.join(modes) if modes else f'0x{proto_val:02x}'
    
    # Bandwidth decode (HT20 only)
    bw_str = 'HT20' if wlan.config('bandwidth') == wlan.BW_HT20 else 'unknown'
    
    # Promiscuous
    prom_str = 'ON' if wlan.config('promiscuous') else 'OFF'
    
    print(f"WiFi connected - IP: {ip}, Protocol: {protocol_str}, Bandwidth: {bw_str}, Promiscuous: {prom_str}")

def connect_wifi():
    """Connect to WiFi"""
    
    print("Activating WiFi interface...")
    
    gc.collect()
    wlan = network.WLAN(network.STA_IF)
    
    # Force cleanup of any stale state from previous interrupted run
    cleanup_wifi(wlan)
    
    wlan.active(True)    
    if not wlan.active():
        raise Exception("WiFi failed to activate")
    
    # Wait for hardware initialization
    time.sleep(2)

    # Dual-band targets (e.g. ESP32-C5/C6): force 2.4GHz for stable CSI capture.
    try:
        wlan.config(band_mode=wlan.BAND_MODE_2G_ONLY)
    except Exception:
        # Legacy/single-band firmware may not expose band_mode.
        pass
        
    # Configure WiFi protocol
    # Force WiFi 4 (802.11b/g/n) only to get 64 subcarriers
    wlan.config(protocol=network.MODE_11B | network.MODE_11G | network.MODE_11N)
    wlan.config(bandwidth=wlan.BW_HT20)          # HT20 for stable CSI
    wlan.config(promiscuous=False)               # CSI from connected AP only
    
    # Connect (optionally locked to a specific BSSID)
    bssid_hex = getattr(config, 'WIFI_BSSID', None)
    bssid = None
    if bssid_hex:
        # Accept "AABBCCDDEEFF" or "AA:BB:CC:DD:EE:FF"
        bssid_clean = bssid_hex.replace(':', '').replace('-', '')
        if len(bssid_clean) == 12:
            bssid = bytes.fromhex(bssid_clean)
    bssid_info = f" (BSSID: {bssid_hex})" if bssid else ""
    print(f"Connecting to WiFi{bssid_info}...")
    wlan.connect(config.WIFI_SSID, config.WIFI_PASSWORD, bssid=bssid)
    
    # Wait for connection
    timeout = 30
    while not wlan.isconnected() and timeout > 0:
        time.sleep(1)
        timeout -= 1
    
    if wlan.isconnected():
        print_wifi_status(wlan)
        # Disable power management
        wlan.config(pm=wlan.PM_NONE)
        # Match the standalone smoke test: enable CSI only after the link is up.
        wlan.csi_enable(buffer_size=config.CSI_BUFFER_SIZE)
        # Stabilization
        time.sleep(1)
        return wlan
    else:
        raise Exception("Connection timeout")


def run_startup_calibration(wlan, detector, traffic_gen, packet_interval_us=None):
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
    from src.runtime_policy import RuntimeMotionPolicy
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
    calibration_progress = 0
    packets_since_evaluation = 0
    next_progress_report = 100
    dropped_at_calibration_start = wlan.csi_dropped()
    last_progress_time = time.ticks_ms()
    last_packet_time = last_progress_time
    last_progress_count = 0
    collapse_logged = False
    remap_logged = False
    ht57_remap_buffer = bytearray(EXPECTED_CSI_LEN)
    normalization_state = CsiPayloadNormalizationState()
    frame_timestamp_filter = CsiFrameTimestampFilter()
    frame_result = None
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
                        f"(reason={assessment['reason_code']}, len={assessment['raw_len']})"
                    )
                del frame
                time.sleep_us(100)
                if time.ticks_diff(time.ticks_ms(), last_packet_time) >= max_timeout_ms:
                    print(f"Timeout waiting for valid CSI packets (collected {calibration_progress}/{calibration_target_packets})")
                    print("Startup calibration aborted")
                    detector.reset()
                    g_state.calibration_mode = False
                    return False
                continue

            csi_data, raw_len, remap_tag = normalize_ht20_csi_payload(
                frame[5], EXPECTED_CSI_LEN,
                remap_buffer=ht57_remap_buffer,
                assessment=assessment,
                state=normalization_state,
            )
            if csi_data is None:
                filtered_count += 1
                del frame
                time.sleep_us(100)
                if time.ticks_diff(time.ticks_ms(), last_packet_time) >= max_timeout_ms:
                    print(f"Timeout waiting for valid CSI packets (collected {calibration_progress}/{calibration_target_packets})")
                    print("Startup calibration aborted")
                    detector.reset()
                    g_state.calibration_mode = False
                    return False
                continue

            if not frame_timestamp_filter.accept(frame):
                del frame
                time.sleep_us(100)
                continue

            if remap_tag in (NORMALIZATION_DOUBLE_HT20, NORMALIZATION_DOUBLE_HT57_TO_64) and not collapse_logged:
                print("[INFO] CSI double-length collapse active: 256->128 and/or 228->114")
                collapse_logged = True
            if remap_tag in (NORMALIZATION_HT57_TO_64, NORMALIZATION_DOUBLE_HT57_TO_64) and not remap_logged:
                print("[INFO] CSI remap active: 57->64 SC (left_pad=4, right_pad=3)")
                remap_logged = True
            del frame
            if not temporal_sampler.admit(frame_result[4]):
                time.sleep_us(100)
                continue
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
                csi_data,
                config.DEFAULT_SUBCARRIERS,
                timestamp_us=frame_result[4],
            )
            packets_since_evaluation += 1
            last_packet_time = time.ticks_ms()

            # frame[4] is the Wi-Fi RX timestamp, the same source the steady
            # loop and the C++ pipeline read.
            calibration_cadence.note_arrival(frame_result[4])

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
                elapsed = time.ticks_diff(current_time, last_progress_time)
                packets_delta = calibration_progress - last_progress_count
                pps = int((packets_delta * 1000) / elapsed) if elapsed > 0 else 0
                dropped = max(
                    0,
                    wlan.csi_dropped() - dropped_at_calibration_start,
                )
                tg_pps = traffic_gen.get_actual_pps()
                current_mv = detector.get_motion_metric() if detector.is_ready() else None
                print(
                    format_calibration_status_line(
                        progress=min(1.0, calibration_progress / calibration_target_packets),
                        pps=pps,
                        motion_metric=current_mv,
                        calibration_packets=calibration_progress,
                        calibration_target_packets=calibration_target_packets,
                        effective_state_label=getattr(
                            calibration_tracker,
                            "get_phase_label",
                            lambda: "CALIBRATING",
                        )(),
                    )
                    + f" | TG:{tg_pps} drop:{dropped}"
                )
                last_progress_time = current_time
                last_progress_count = calibration_progress
                while next_progress_report <= calibration_progress:
                    next_progress_report += 100
        else:
            time.sleep_us(100)
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


def _traffic_adaptive_enabled():
    """Return the configured adaptive-pacing flag."""
    return bool(getattr(config, "TRAFFIC_GENERATOR_ADAPTIVE", False))


def _maintain_traffic_and_csi_health(
    traffic_gen,
    csi_health,
    *,
    accepted_csi_total,
    callback_total,
    now_us,
):
    """Adapt send pacing and report sustained original ESP32 CSI stalls."""
    if traffic_gen is None or not traffic_gen.is_running():
        return
    traffic_gen.observe_accepted_csi(accepted_csi_total, now_us=now_us)
    csi_health.maintain(
        traffic_gen.get_packet_count(),
        callback_total,
        time.ticks_ms(),
    )


def restart_traffic_generator(traffic_gen):
    """Restart the traffic generator after calibration-sensitive work completes."""
    if not traffic_gen or not getattr(config, 'TRAFFIC_GENERATOR_ENABLED', True):
        return

    time.sleep(1)  # Give WiFi/MQTT stack time to settle before reopening raw socket.
    gc.collect()
    adaptive = _traffic_adaptive_enabled()
    target_pps = max(1, int(getattr(config, 'CSI_TARGET_PPS', 100)))
    if not traffic_gen.start(target_pps, adaptive=adaptive):
        print("Warning: Failed to restart traffic generator, retrying...")
        time.sleep(2)
        gc.collect()
        traffic_gen.start(target_pps, adaptive=adaptive)


def main():
    """Main application loop"""
    print('Micro-ESPectre starting...')
    print_heap('boot')
    
    # Detect chip type
    g_state.chip_type = get_chip_type()
    print(f'Detected chip: {g_state.chip_type}')
    
    # Connect to WiFi
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
    traffic_adaptive = _traffic_adaptive_enabled()
    from src.traffic_generator import TrafficGenerator
    traffic_gen = TrafficGenerator(mode=traffic_mode, adaptive=traffic_adaptive)
    print_heap('after_traffic_gen_init')
    observed_interval_us = None
    if getattr(config, 'TRAFFIC_GENERATOR_ENABLED', True):
        if not traffic_gen.start(target_pps, adaptive=traffic_adaptive):
            print("FATAL: Traffic generator failed to start - CSI will not work")
            print("Check WiFi connection and gateway availability")
            import machine
            time.sleep(5)
            machine.reset()  # Reboot and retry
        
        print(
            f'Traffic generator started ({traffic_mode}, target={target_pps} CSI pps, '
            f'adaptive={"on" if traffic_adaptive else "off"})'
        )
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
                traffic_gen.start(target_pps, adaptive=traffic_adaptive)
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
    
    calibration_ok = run_startup_calibration(
        wlan,
        detector,
        traffic_gen,
        packet_interval_us=observed_interval_us,
    )
    if not calibration_ok:
        if traffic_gen.is_running():
            traffic_gen.stop()
        cleanup_wifi(wlan)
        raise RuntimeError("Startup calibration failed")
    print_heap('after_calibration')
    
    mqtt_enabled = getattr(config, 'MQTT_ENABLED', True)
    mqtt_handler = None
    if mqtt_enabled:
        # Initialize MQTT ESPectre Protocol and runtime metrics.
        from src.mqtt.handler import MQTTHandler
        mqtt_handler = MQTTHandler(config, detector, wlan, g_state)
        print_heap('after_mqtt_handler_init')
        mqtt_handler.connect()
        print_heap('after_mqtt_connect')
        
    else:
        print('MQTT disabled')

    if getattr(config, 'TRAFFIC_GENERATOR_ENABLED', True) and not traffic_gen.is_running():
        restart_traffic_generator(traffic_gen)
    
    print('')
    print(ASCII_BANNER)
    print('')
    
    # Force garbage collection before main loop
    gc.collect()
    print(f'Free memory before main loop: {gc.mem_free()} bytes')
    
    # Main CSI processing loop with integrated MQTT publishing
    publish_counter = 0
    processed_packet_count = 0
    callback_packet_count = 0
    mqtt_poll_counter = 0
    mqtt_poll_interval = max(
        1,
        int(round(
            getattr(config, 'EVALUATION_INTERVAL_MS', 250)
            * max(1, traffic_gen.get_target_rate())
            / 1000.0
        )),
    )
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
    dropped_at_main_loop_start = wlan.csi_dropped()
    csi_health = CsiPacingHealthMonitor(enabled=(g_state.chip_type == 'ESP32'))
    
    publish_interval_ms = max(1, int(getattr(config, 'PUBLISH_INTERVAL_MS', 1000)))
    from src.runtime_policy import RuntimeMotionPolicy
    runtime_policy = RuntimeMotionPolicy(
        evaluation_interval_ms=getattr(config, 'EVALUATION_INTERVAL_MS', 250),
        motion_on_hits=getattr(config, 'MOTION_ON_HITS', 4),
        motion_off_hits=getattr(config, 'MOTION_OFF_HITS', 3),
        segmentation_window_size_ms=getattr(config, 'SEGMENTATION_WINDOW_SIZE_MS', 1000),
    )
    temporal_sampler = TemporalCsiSampler(
        target_pps,
        getattr(config, 'SEGMENTATION_WINDOW_SIZE_MS', 1000),
    )
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

            current_time = time.ticks_ms()
            time_delta = time.ticks_diff(current_time, last_publish_time)
            if time_delta >= publish_interval_ms:
                pps = int((publish_counter * 1000) / time_delta) if time_delta > 0 else 0
                progress = (
                    latest_motion_metric / latest_threshold
                    if latest_threshold > 0
                    else 0
                )
                print(
                    format_detection_publish_line(
                        packet_count=processed_packet_count,
                        dropped_count=max(
                            0,
                            wlan.csi_dropped() - dropped_at_main_loop_start,
                        ),
                        pps=pps,
                        motion_metric=latest_motion_metric,
                        threshold=latest_threshold,
                        effective_state=latest_effective_state,
                        progress=progress,
                    )
                )
                if mqtt_handler is not None:
                    mqtt_handler.publish_state(
                        latest_motion_metric,
                        latest_effective_state,
                        latest_threshold,
                    )
                publish_counter = 0
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
                            f"(reason={assessment['reason_code']}, len={assessment['raw_len']})"
                        )
                    del frame
                    _maintain_traffic_and_csi_health(
                        traffic_gen,
                        csi_health,
                        accepted_csi_total=processed_packet_count,
                        callback_total=callback_packet_count,
                        now_us=loop_start,
                    )
                    g_state.loop_time_us = time.ticks_diff(time.ticks_us(), loop_start)
                    time.sleep_us(100)
                    continue

                csi_data, raw_len, remap_tag = normalize_ht20_csi_payload(
                    frame[5], EXPECTED_CSI_LEN,
                    remap_buffer=ht57_remap_buffer,
                    assessment=assessment,
                    state=normalization_state,
                )
                if csi_data is None:
                    filtered_count += 1
                    format_drop_streak += 1
                    del frame
                    _maintain_traffic_and_csi_health(
                        traffic_gen,
                        csi_health,
                        accepted_csi_total=processed_packet_count,
                        callback_total=callback_packet_count,
                        now_us=loop_start,
                    )
                    g_state.loop_time_us = time.ticks_diff(time.ticks_us(), loop_start)
                    time.sleep_us(100)
                    continue

                if not frame_timestamp_filter.accept(frame):
                    out_of_order_count += 1
                    if out_of_order_count % 100 == 1:
                        print(f"[WARN] Filtered {out_of_order_count} duplicate or out-of-order CSI frames")
                    del frame
                    _maintain_traffic_and_csi_health(
                        traffic_gen,
                        csi_health,
                        accepted_csi_total=processed_packet_count,
                        callback_total=callback_packet_count,
                        now_us=loop_start,
                    )
                    g_state.loop_time_us = time.ticks_diff(time.ticks_us(), loop_start)
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
                _maintain_traffic_and_csi_health(
                    traffic_gen,
                    csi_health,
                    accepted_csi_total=processed_packet_count,
                    callback_total=callback_packet_count,
                    now_us=loop_start,
                )

                # Poll MQTT on the same cadence as detector evaluation. This keeps
                # command latency below 250 ms at 100 pps without adding socket
                # work between evaluations.
                if mqtt_handler is not None:
                    mqtt_poll_counter += 1
                    if mqtt_poll_counter >= mqtt_poll_interval:
                        mqtt_handler.check_messages()
                        mqtt_poll_counter = 0
                
                publish_counter += 1
                if g_state.current_channel != 0 and packet_channel != g_state.current_channel:
                    print(f"[WARN] WiFi channel changed: {g_state.current_channel} -> {packet_channel}, resetting detection buffer")
                    detector.reset()
                    runtime_policy.reset()
                    temporal_sampler.clear_history()
                    normalization_state.reset()
                g_state.current_channel = packet_channel

                if not temporal_sampler.admit(frame_result[4]):
                    g_state.loop_time_us = time.ticks_diff(time.ticks_us(), loop_start)
                    time.sleep_us(100)
                    continue
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
                    csi_data,
                    config.DEFAULT_SUBCARRIERS,
                    timestamp_us=frame_result[4],
                )
                runtime_policy.note_arrival(frame_result[4])

                if runtime_policy.should_evaluate():
                    metrics = detector.update_state()
                    effective_state, _ = runtime_policy.apply_state(metrics['state'])
                    runtime_policy.after_evaluation()
                    latest_motion_metric = metrics.get('motion_metric', 0.0)
                    latest_threshold = metrics['threshold']
                    latest_effective_state = effective_state

                # Update loop time metric
                g_state.loop_time_us = time.ticks_diff(time.ticks_us(), loop_start)
                
                time.sleep_us(100)
            else:
                _maintain_traffic_and_csi_health(
                    traffic_gen,
                    csi_health,
                    accepted_csi_total=processed_packet_count,
                    callback_total=callback_packet_count,
                    now_us=loop_start,
                )
                # Update loop time metric (idle iteration)
                g_state.loop_time_us = time.ticks_diff(time.ticks_us(), loop_start)
                
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
