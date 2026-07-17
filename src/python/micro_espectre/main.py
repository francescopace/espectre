"""
Micro-ESPectre - Main Application

Main entry point for the Micro-ESPectre Wi-Fi CSI runtime.

Author: Francesco Pace <francesco.pace@gmail.com>
License: GPLv3
"""
import network
import time
import gc
import os
import src.config as config
from src.config import NUM_SUBCARRIERS, EXPECTED_CSI_LEN
from src.device_utils import (
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

ML_DEFAULT_THRESHOLD = 0.5

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


def create_detector(detection_algorithm):
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

    threshold = 1.0 if detector_needs_startup_calibration(detection_algorithm) else ML_DEFAULT_THRESHOLD

    print(f'Detection algorithm: {get_detector_label(detection_algorithm)}')
    return detector_class(
        window_size=config.SEG_WINDOW_SIZE,
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
        detector.set_threshold(ML_DEFAULT_THRESHOLD)

        print('')
        print('='*60)
        print('ML Quick Boot')
        print('='*60)
        print(f'Free memory: {gc.mem_free()} bytes')
        
        print('')
        print('='*60)
        print('ML Quick Boot Complete!')
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

    calibration_tracker = StartupThresholdCalibrator(
        config.CALIBRATION_BUFFER_SIZE,
        auto_factor=get_detector_auto_factor(detector),
        gate_enabled=get_detector_startup_gate(detector),
    )
    begin_calibration = getattr(detector, "on_startup_calibration_begin", None)
    if callable(begin_calibration):
        begin_calibration()
    evaluation_interval = max(1, int(getattr(config, 'EVALUATION_INTERVAL', 25)))

    print('')
    print('-'*60)
    print(f'{detector_name} Threshold Bootstrap ({config.CALIBRATION_BUFFER_SIZE} packets, evaluate every {evaluation_interval}) [HT20: {NUM_SUBCARRIERS} SC]')
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
    frame_result = None
    
    while not calibration_tracker.is_complete():
        frame = csi_read_frame(wlan, frame_result)
        if frame:
            frame_result = frame
            csi_data, raw_len, remap_tag = normalize_ht20_csi_payload(
                frame[5], EXPECTED_CSI_LEN, remap_buffer=ht57_remap_buffer
            )

            if csi_data is None:
                filtered_count += 1
                if filtered_count % 100 == 1:
                    print(f"[WARN] Filtered {filtered_count} packets with wrong SC count (got {raw_len} bytes, expected {EXPECTED_CSI_LEN})")
                del frame
                time.sleep_us(100)
                if time.ticks_diff(time.ticks_ms(), last_packet_time) >= max_timeout_ms:
                    print(f"Timeout waiting for valid CSI packets (collected {calibration_progress}/{config.CALIBRATION_BUFFER_SIZE})")
                    print("Startup calibration aborted")
                    detector.reset()
                    g_state.calibration_mode = False
                    return False
                continue

            if remap_tag in ('double_ht20', 'double_ht57_and_remap') and not collapse_logged:
                print("[INFO] CSI double-length collapse active: 256->128 and/or 228->114")
                collapse_logged = True
            if remap_tag in ('ht57_to_64', 'double_ht57_and_remap') and not remap_logged:
                print("[INFO] CSI remap active: 57->64 SC (left_pad=4, right_pad=3)")
                remap_logged = True
            del frame
            detector.process_packet(csi_data, config.DEFAULT_SUBCARRIERS)
            packets_since_evaluation += 1
            last_packet_time = time.ticks_ms()

            if packets_since_evaluation < evaluation_interval:
                continue

            detector.update_state()
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
                        progress=min(1.0, calibration_progress / config.CALIBRATION_BUFFER_SIZE),
                        pps=pps,
                        motion_metric=current_mv,
                        calibration_packets=calibration_progress,
                        calibration_target_packets=config.CALIBRATION_BUFFER_SIZE,
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
                print(f"Timeout waiting for CSI packets (collected {calibration_progress}/{config.CALIBRATION_BUFFER_SIZE})")
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


def restart_traffic_generator(traffic_gen):
    """Restart the traffic generator after calibration-sensitive work completes."""
    if not traffic_gen or not config.TRAFFIC_GENERATOR_RATE:
        return

    time.sleep(1)  # Give WiFi/MQTT stack time to settle before reopening raw socket.
    gc.collect()
    if not traffic_gen.start(config.TRAFFIC_GENERATOR_RATE):
        print("Warning: Failed to restart traffic generator, retrying...")
        time.sleep(2)
        gc.collect()
        traffic_gen.start(config.TRAFFIC_GENERATOR_RATE)


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
    
    # Initialize detector based on configured algorithm
    detection_algorithm = normalize_detector_algorithm(
        getattr(config, 'DETECTION_ALGORITHM', 'classic')
    )
    detector = create_detector(detection_algorithm)
    print_heap('after_detector_init')
    
    # Initialize and start traffic generator (rate is static from config.py)
    gc.collect()  # Free memory before creating socket
    traffic_mode = getattr(config, 'TRAFFIC_GENERATOR_MODE', 'ping')
    from src.traffic_generator import TrafficGenerator
    traffic_gen = TrafficGenerator(mode=traffic_mode)
    print_heap('after_traffic_gen_init')
    if config.TRAFFIC_GENERATOR_RATE > 0:
        if not traffic_gen.start(config.TRAFFIC_GENERATOR_RATE):
            print("FATAL: Traffic generator failed to start - CSI will not work")
            print("Check WiFi connection and gateway availability")
            import machine
            time.sleep(5)
            machine.reset()  # Reboot and retry
        
        print(f'Traffic generator started ({traffic_mode}, {config.TRAFFIC_GENERATOR_RATE} pps)')
        print_heap('after_traffic_gen_start')
        
        # Verify CSI packets are flowing with retry logic
        max_tg_retries = 3
        for tg_attempt in range(max_tg_retries):
            time.sleep(2)  # Wait for traffic to start generating CSI packets
            
            print('Waiting for CSI packets...')
            csi_received = 0
            frame_result = None
            for _ in range(100):  # Max 100 attempts (~5 seconds)
                frame = csi_read_frame(wlan, frame_result)
                if frame:
                    frame_result = frame
                    csi_received += 1
                    if csi_received >= 10:
                        break
                time.sleep(0.05)
            
            if csi_received >= 10:
                break  # Success
            
            if tg_attempt < max_tg_retries - 1:
                print(f'WARNING: Only {csi_received} CSI packets - restarting TG (attempt {tg_attempt + 2}/{max_tg_retries})')
                traffic_gen.stop()
                time.sleep(1)
                traffic_gen.start(config.TRAFFIC_GENERATOR_RATE)
            else:
                print(f'FATAL: No CSI packets after {max_tg_retries} attempts - cannot operate without traffic')
                print('Please check WiFi connection and retry')
                import sys
                sys.exit(1)
        print_heap('after_csi_flow_check')
    
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

    if config.TRAFFIC_GENERATOR_RATE > 0 and not traffic_gen.is_running():
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
    mqtt_poll_counter = 0
    mqtt_poll_interval = max(1, int(getattr(config, 'EVALUATION_INTERVAL', 25)))
    filtered_count = 0  # Packets with wrong SC count
    last_publish_time = time.ticks_ms()
    collapse_logged = False
    remap_logged = False
    ht57_remap_buffer = bytearray(EXPECTED_CSI_LEN)
    frame_result = None
    dropped_at_main_loop_start = wlan.csi_dropped()
    
    publish_rate = getattr(config, 'PUBLISH_INTERVAL', None)
    if publish_rate is None:
        publish_rate = traffic_gen.get_rate() if traffic_gen.is_running() else 100
    from src.runtime_policy import RuntimeMotionPolicy
    runtime_policy = RuntimeMotionPolicy(
        evaluation_interval=getattr(config, 'EVALUATION_INTERVAL', 25),
        motion_on_hits=getattr(config, 'MOTION_ON_HITS', 3),
        motion_off_hits=getattr(config, 'MOTION_OFF_HITS', 3),
    )
       
    try:
        while True:
            loop_start = time.ticks_us()
            
            # Suspend main loop during calibration
            if g_state.calibration_mode:
                time.sleep_ms(1000) # Sleep for 1 second to yield CPU
                continue
            
            frame = csi_read_frame(wlan, frame_result)
            
            if frame:
                frame_result = frame
                csi_data, raw_len, remap_tag = normalize_ht20_csi_payload(
                    frame[5], EXPECTED_CSI_LEN, remap_buffer=ht57_remap_buffer
                )

                if csi_data is None:
                    filtered_count += 1
                    if filtered_count % 100 == 1:
                        print(f"[WARN] Filtered {filtered_count} packets with wrong SC count (got {raw_len} bytes, expected {EXPECTED_CSI_LEN})")
                    del frame
                    g_state.loop_time_us = time.ticks_diff(time.ticks_us(), loop_start)
                    time.sleep_us(100)
                    continue

                if remap_tag in ('double_ht20', 'double_ht57_and_remap') and not collapse_logged:
                    print("[INFO] CSI double-length collapse active: 256->128 and/or 228->114")
                    collapse_logged = True
                if remap_tag in ('ht57_to_64', 'double_ht57_and_remap') and not remap_logged:
                    print("[INFO] CSI remap active: 57->64 SC (left_pad=4, right_pad=3)")
                    remap_logged = True
                packet_channel = frame[1]
                
                del frame
                
                # Process packet through detector interface
                detector.process_packet(csi_data, config.DEFAULT_SUBCARRIERS)
                processed_packet_count += 1

                # Poll MQTT on the same cadence as detector evaluation. This keeps
                # command latency below 250 ms at 100 pps without adding socket
                # work between evaluations.
                if mqtt_handler is not None:
                    mqtt_poll_counter += 1
                    if mqtt_poll_counter >= mqtt_poll_interval:
                        mqtt_handler.check_messages()
                        mqtt_poll_counter = 0
                
                publish_counter += 1
                runtime_policy.note_packet()
                should_publish = publish_counter >= publish_rate
                
                if runtime_policy.should_evaluate(should_publish):
                    # Detect WiFi channel changes (AP may switch channels automatically)
                    # Channel changes cause CSI spikes that trigger false motion detection
                    if g_state.current_channel != 0 and packet_channel != g_state.current_channel:
                        print(f"[WARN] WiFi channel changed: {g_state.current_channel} -> {packet_channel}, resetting detection buffer")
                        detector.reset()
                        runtime_policy.reset()
                    g_state.current_channel = packet_channel
                    
                    metrics = detector.update_state()
                    effective_state, _ = runtime_policy.apply_state(metrics['state'])
                    runtime_policy.after_evaluation()

                    if should_publish:
                        current_time = time.ticks_ms()
                        time_delta = time.ticks_diff(current_time, last_publish_time)
                        
                        # Calculate packets per second
                        pps = int((publish_counter * 1000) / time_delta) if time_delta > 0 else 0
                        
                        motion_metric = metrics.get('motion_metric', 0.0)
                        threshold = metrics['threshold']
                        progress = motion_metric / threshold if threshold > 0 else 0
                        print(
                            format_detection_publish_line(
                                packet_count=processed_packet_count,
                                dropped_count=max(
                                    0,
                                    wlan.csi_dropped() - dropped_at_main_loop_start,
                                ),
                                pps=pps,
                                motion_metric=motion_metric,
                                threshold=threshold,
                                effective_state=effective_state,
                                progress=progress,
                            )
                        )
                        
                        if mqtt_handler is not None:
                            mqtt_handler.publish_state(
                                motion_metric,
                                effective_state,
                                threshold
                            )
                        publish_counter = 0
                        last_publish_time = current_time

                # Update loop time metric
                g_state.loop_time_us = time.ticks_diff(time.ticks_us(), loop_start)
                
                time.sleep_us(100)
            else:
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
