"""
Micro-ESPectre - Main Application

Motion detection using WiFi CSI with MVS algorithm.
Main entry point for the Micro-ESPectre system running on ESP32-C6.

Author: Francesco Pace <francesco.pace@gmail.com>
License: GPLv3
"""
import network
import time
import gc
import os
from src.mvs_detector import MVSDetector
from src.ml_detector import MLDetector
from src.mqtt.handler import MQTTHandler
from src.traffic_generator import TrafficGenerator
import src.config as config

# Default subcarriers (used if not configured or for fallback in case of error)
DEFAULT_SUBCARRIERS = [11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22]

# Gain lock configuration
GAIN_LOCK_PACKETS = 300  # ~3 seconds at 100 Hz

# Import HT20 constants from config
from src.config import NUM_SUBCARRIERS, EXPECTED_CSI_LEN, SEG_THRESHOLD
from src.utils import calculate_gain_compensation, to_signed_int8, calculate_median

# Global state for calibration mode and performance metrics
class GlobalState:
    def __init__(self):
        self.calibration_mode = False  # Flag to suspend main loop during calibration
        self.loop_time_us = 0  # Last loop iteration time in microseconds
        self.chip_type = None  # Detected chip type (S3, C6, etc.)
        self.current_channel = 0  # Track WiFi channel for change detection
        # Gain compensation state (when gain lock is skipped or disabled)
        self.needs_compensation = False
        self.baseline_agc = 0  # uint8
        self.baseline_fft = 0  # int8 (signed)


g_state = GlobalState()

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
    except:
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
    
    print(f"Activating WiFi interface...")
    
    gc.collect()
    wlan = network.WLAN(network.STA_IF)
    
    # Force cleanup of any stale state from previous interrupted run
    cleanup_wifi(wlan)
    
    wlan.active(True)    
    if not wlan.active():
        raise Exception("WiFi failed to activate")
    
    # Wait for hardware initialization
    time.sleep(2)
        
    # Configure WiFi protocol
    # Force WiFi 4 (802.11b/g/n) only to get 64 subcarriers
    wlan.config(protocol=network.MODE_11B | network.MODE_11G | network.MODE_11N)
    wlan.config(bandwidth=wlan.BW_HT20)          # HT20 for stable CSI
    wlan.config(promiscuous=False)               # CSI from connected AP only
    
    # Enable CSI after WiFi is stable
    wlan.csi_enable(buffer_size=config.CSI_BUFFER_SIZE)
    
    # Connect
    print(f"Connecting to WiFi...")
    wlan.connect(config.WIFI_SSID, config.WIFI_PASSWORD)
    
    # Wait for connection
    timeout = 30
    while not wlan.isconnected() and timeout > 0:
        time.sleep(1)
        timeout -= 1
    
    if wlan.isconnected():
        print_wifi_status(wlan)
        # Disable power management
        wlan.config(pm=wlan.PM_NONE)
        # Stabilization
        time.sleep(1)
        return wlan
    else:
        raise Exception("Connection timeout")


def format_progress_bar(score, threshold, width=20, is_probability=False):
    """Format progress bar for console output.
    
    For MVS: score = metric/threshold, threshold_pos at 75% (15/20)
    For ML: score = probability, threshold_pos at threshold (e.g., 50% for 0.5)
    """
    if is_probability:
        # ML mode: threshold is a probability (0-1), show it at its actual position
        threshold_pos = int(threshold * width)
        filled = int(score * width)
    else:
        # MVS mode: score is already normalized (metric/threshold)
        threshold_pos = 15  # 75% position
        filled = int(score * threshold_pos)
    
    filled = max(0, min(filled, width))
    
    bar = '['
    for i in range(width):
        if i == threshold_pos:
            bar += '|'
        elif i < filled:
            bar += '█'
        else:
            bar += '░'
    bar += ']'
    
    percent = int(score * 100)
    return f"{bar} {percent}%"


def run_gain_lock(wlan):
    """
    Run gain lock calibration phase (ESP32-S3, C3, C5, C6 only)
    
    Collects AGC/FFT gain values from first packets and locks them
    to stabilize CSI amplitudes for consistent motion detection.
    Uses median calculation for robustness against outliers.
    
    HT20 only: 64 subcarriers.
    
    Respects config.GAIN_LOCK_MODE:
    - "auto": Lock gain, but skip if signal too strong (AGC < MIN_SAFE_AGC)
    - "enabled": Always force gain lock
    - "disabled": Collect baseline for compensation, but no lock
    
    Args:
        wlan: WLAN instance with CSI enabled
        
    Returns:
        tuple: (agc_gain, fft_gain, needs_compensation) where:
            - needs_compensation=True if gain lock was skipped/disabled
    """
    # Check configuration mode
    mode = getattr(config, 'GAIN_LOCK_MODE', 'auto').lower()
    min_safe_agc = getattr(config, 'GAIN_LOCK_MIN_SAFE_AGC', 30)
    
    # Check platform support
    gain_lock_supported = hasattr(wlan, 'csi_gain_lock_supported') and wlan.csi_gain_lock_supported()
    
    if not gain_lock_supported:
        print(f"Gain lock: Not supported on this platform")
        print(f"  HT20 mode: {NUM_SUBCARRIERS} subcarriers")
        return None, None, False
    
    print('')
    print('-'*60)
    print(f'Gain Lock Calibration (~3 seconds) [mode: {mode}]')
    print('-'*60)
    
    # Collect samples for median calculation
    agc_samples = []
    fft_samples = []
    count = 0
    
    while count < GAIN_LOCK_PACKETS:
        frame = wlan.csi_read()
        if frame:
            # frame[22] = agc_gain (uint8), frame[23] = fft_gain (int8 as uint8)
            agc_samples.append(frame[22])
            fft_samples.append(to_signed_int8(frame[23]))
            
            del frame  # Free memory immediately
            count += 1
            
            # Progress every 25% (with GC to prevent ENOMEM)
            if count == GAIN_LOCK_PACKETS // 4:
                gc.collect()
                print(f"  Gain calibration 25% ({count}/{GAIN_LOCK_PACKETS} packets)")
            elif count == GAIN_LOCK_PACKETS // 2:
                gc.collect()
                print(f"  Gain calibration 50% ({count}/{GAIN_LOCK_PACKETS} packets)")
            elif count == (GAIN_LOCK_PACKETS * 3) // 4:
                gc.collect()
                print(f"  Gain calibration 75% ({count}/{GAIN_LOCK_PACKETS} packets)")
    
    # Calculate medians (more robust than mean against outliers)
    median_agc = calculate_median(agc_samples)
    median_fft = calculate_median(fft_samples)
    
    print(f"  HT20 mode: {NUM_SUBCARRIERS} subcarriers")
    
    # Handle different modes
    if mode == 'disabled':
        # DISABLED mode: baseline collected for compensation, but no gain lock
        print(f"Gain baseline: AGC={median_agc}, FFT={median_fft} (compensation enabled, no lock)")
        return median_agc, median_fft, True
    
    # In auto mode, skip gain lock if signal is too strong
    if mode == 'auto' and median_agc < min_safe_agc:
        print(f"WARNING: Signal too strong (AGC={median_agc} < {min_safe_agc}) - skipping gain lock")
        print(f"         Move sensor 2-3 meters from AP for optimal performance")
        print(f"         Compensation enabled for baseline: AGC={median_agc}, FFT={median_fft}")
        return median_agc, median_fft, True
    
    # Lock the gain values
    wlan.csi_force_gain(median_agc, median_fft)
    print(f"Gain locked: AGC={median_agc}, FFT={median_fft} (median of {GAIN_LOCK_PACKETS} packets)")
    
    return median_agc, median_fft, False


def run_band_calibration(wlan, detector, traffic_gen, chip_type=None):
    """
    Run band calibration with selected algorithm (with gain lock phase first)
    
    Args:
        wlan: WLAN instance
        detector: IDetector instance (MVSDetector or MLDetector)
        traffic_gen: TrafficGenerator instance
        chip_type: Chip type ('C5', 'C6', 'S3', etc.) for subcarrier filtering
    
    Returns:
        bool: True if calibration successful
    """
    # Determine calibration type based on detector
    detector_name = detector.get_name()
    is_ml = detector_name == "ML"
    
    if is_ml:
        # ML uses fixed subcarriers - no band calibration needed
        # Only gain lock is required for stable CSI amplitudes
        print("ML detector: Using fixed subcarriers (no band calibration needed)")
        
        # Set calibration mode for gain lock
        g_state.calibration_mode = True
        
        print('')
        print('='*60)
        print('ML Quick Boot - Gain Lock Only')
        print('='*60)
        print(f'Free memory: {gc.mem_free()} bytes')
        print('Please remain still for gain lock...')
        
        # Phase 1: Gain Lock only (~3 seconds)
        agc, fft, needs_comp = run_gain_lock(wlan)
        
        # Save baseline for compensation if needed
        if agc is not None and fft is not None:
            g_state.needs_compensation = needs_comp
            g_state.baseline_agc = agc
            g_state.baseline_fft = fft
        
        if needs_comp:
            print("Note: Proceeding without gain lock (compensation enabled)")
        
        # Use ML-specific subcarriers (must match model training)
        from src.ml_detector import ML_SUBCARRIERS
        config.SELECTED_SUBCARRIERS = ML_SUBCARRIERS
        
        print('')
        print('='*60)
        print('ML Quick Boot Complete!')
        print(f'   Subcarriers: {config.SELECTED_SUBCARRIERS}')
        print(f'   Threshold: 0.5 (probability)')
        print(f'   Total boot time: ~3 seconds (gain lock only)')
        print('='*60)
        print('')
        
        g_state.calibration_mode = False
        return True
    else:
        # Get configured algorithm for MVS
        algorithm = getattr(config, 'CALIBRATION_ALGORITHM', 'nbvi').lower()
        if algorithm == 'nbvi':
            from src.nbvi_calibrator import NBVICalibrator, cleanup_buffer_file
        else:
            from src.p95_calibrator import P95Calibrator, cleanup_buffer_file
    
    # Set calibration mode to suspend main loop
    g_state.calibration_mode = True
    
    # Aggressive garbage collection before allocating calibration buffer
    gc.collect()
    
    # Clean up any leftover files from previous interrupted runs
    cleanup_buffer_file()
    
    print('')
    print('='*60)
    print('Two-Phase Calibration Starting')
    print('='*60)
    print(f'Free memory: {gc.mem_free()} bytes')
    print(f'Algorithm: {algorithm.upper()}')
    print('Please remain still for calibration...')
    
    # Phase 1: Gain Lock (~3 seconds)
    # Stabilizes AGC/FFT before calibration to ensure clean data
    agc, fft, needs_comp = run_gain_lock(wlan)
    
    # Save baseline for compensation if needed
    if agc is not None and fft is not None:
        g_state.needs_compensation = needs_comp
        g_state.baseline_agc = agc
        g_state.baseline_fft = fft
    
    if needs_comp:
        print("Note: Proceeding with band calibration without gain lock (compensation enabled)")
    
    print('')
    print('-'*60)
    print(f'Band Calibration (~7 seconds) [HT20: {NUM_SUBCARRIERS} SC]')
    print('-'*60)
    
    # Initialize calibrator based on algorithm
    if algorithm == 'nbvi':
        calibrator = NBVICalibrator(buffer_size=config.CALIBRATION_BUFFER_SIZE)
    else:
        calibrator = P95Calibrator(buffer_size=config.CALIBRATION_BUFFER_SIZE)
    
    # Collect packets for calibration (now with stable gain)
    calibration_progress = 0
    timeout_counter = 0
    max_timeout = 15000  # 15 seconds
    packets_read = 0
    last_progress_time = time.ticks_ms()
    last_progress_count = 0
    
    while calibration_progress < config.CALIBRATION_BUFFER_SIZE:
        frame = wlan.csi_read()
        packets_read += 1
        
        if frame:
            # HT20: 64 SC × 2 bytes = 128 bytes
            csi_data = frame[5][:EXPECTED_CSI_LEN]
            del frame  # Free memory immediately
            calibration_progress = calibrator.add_packet(csi_data)
            timeout_counter = 0  # Reset timeout on successful read
            
            # Print progress every 100 packets with pps
            if calibration_progress % 100 == 0:
                current_time = time.ticks_ms()
                elapsed = time.ticks_diff(current_time, last_progress_time)
                packets_delta = calibration_progress - last_progress_count
                pps = int((packets_delta * 1000) / elapsed) if elapsed > 0 else 0
                dropped = wlan.csi_dropped()
                tg_pps = traffic_gen.get_actual_pps()
                print(f"Collecting {calibration_progress}/{config.CALIBRATION_BUFFER_SIZE} packets... (pps:{pps}, TG:{tg_pps}, drop:{dropped})")
                last_progress_time = current_time
                last_progress_count = calibration_progress
        else:
            time.sleep_us(100)
            timeout_counter += 1
            
            if timeout_counter >= max_timeout:
                print(f"Timeout waiting for CSI packets (collected {calibration_progress}/{config.CALIBRATION_BUFFER_SIZE})")
                print("Calibration aborted - using default band")
                return False
    
    gc.collect()  # Free any temporary objects before calibration
    
    # Run calibration (both algorithms now return adaptive_threshold)
    success = False
    config.SELECTED_SUBCARRIERS = DEFAULT_SUBCARRIERS
    
    # Stop traffic generator during band evaluation to free memory
    tg_was_running = traffic_gen.is_running()
    if tg_was_running:
        traffic_gen.stop()
        gc.collect()
    
    try:
        # Calibrator returns: calibrate() -> (band, values)
        # band = selected subcarriers, values = mv_values
        selected_band, cal_values = calibrator.calibrate()
        
        if selected_band and len(selected_band) == 12:
            config.SELECTED_SUBCARRIERS = selected_band
            
            if is_ml:
                # ML: subcarriers only, threshold stays at 0.5
                threshold_source = "fixed (0.5)"
                success = True
                
                print('')
                print('='*60)
                print('ML Subcarrier Calibration Successful!')
                print(f'   Algorithm: {algorithm.upper()} (subcarrier selection only)')
                print(f'   Selected band: {selected_band}')
                print(f'   Threshold: {detector.get_threshold():.2f} ({threshold_source})')
                print('='*60)
                print('')
            else:
                # MVS: apply adaptive threshold from MV values
                from src.threshold import calculate_adaptive_threshold
                
                if isinstance(SEG_THRESHOLD, str):
                    # "auto" or "min" mode - calculate adaptive threshold
                    adaptive_threshold, percentile, factor, pxx = calculate_adaptive_threshold(cal_values, SEG_THRESHOLD)
                    detector.set_adaptive_threshold(adaptive_threshold)
                    threshold_source = f"{SEG_THRESHOLD} (P{percentile}x{factor})"
                    print(f'Adaptive threshold: {adaptive_threshold:.4f} ({threshold_source})')
                else:
                    # Numeric value - use fixed manual threshold
                    adaptive_threshold, _, _, _ = calculate_adaptive_threshold(cal_values, "auto")
                    detector.set_threshold(float(SEG_THRESHOLD))
                    threshold_source = "manual"
                    print(f'Manual threshold: {SEG_THRESHOLD:.2f} (adaptive would be: {adaptive_threshold:.4f})')
                
                success = True
                
                print('')
                print('='*60)
                print('Subcarrier Calibration Successful!')
                print(f'   Algorithm: {algorithm.upper()}')
                print(f'   Selected band: {selected_band}')
                print(f'   Threshold: {detector.get_threshold():.4f} ({threshold_source})')
                print('='*60)
                print('')
        else:
            # Calibration failed - keep default
            print('')
            print('='*60)
            print('Subcarrier Calibration Failed')
            print(f'   Using default band: {config.SELECTED_SUBCARRIERS}')
            print('='*60)
            print('')
    
    except Exception as e:
        print(f"Error during calibration: {e}")
        print(f"Using default band: {config.SELECTED_SUBCARRIERS}")
    
    # Free calibrator memory explicitly
    calibrator.free_buffer()
    calibrator = None
    gc.collect()
    
    # Restart traffic generator if it was running
    if tg_was_running:
        time.sleep(1)  # Wait for network stack to stabilize
        if not traffic_gen.start(config.TRAFFIC_GENERATOR_RATE):
            print("Warning: Failed to restart traffic generator, retrying...")
            time.sleep(2)
            traffic_gen.start(config.TRAFFIC_GENERATOR_RATE)
    
    # Resume main loop
    g_state.calibration_mode = False
    
    return success

def get_chip_type():
    """Extract short chip type from os.uname().machine."""
    machine = os.uname().machine.upper()
    # Check for specific variants first
    for variant in ['S3', 'S2', 'C3', 'C5', 'C6']:
        if variant in machine:
            return variant
    # Fallback to ESP32 base
    if 'ESP32' in machine:
        return 'ESP32'
    return machine


def main():
    """Main application loop"""
    print('Micro-ESPectre starting...')
    
    # Detect chip type
    g_state.chip_type = get_chip_type()
    print(f'Detected chip: {g_state.chip_type}')
    
    # Connect to WiFi
    wlan = connect_wifi()
    
    # Initialize detector based on configured algorithm
    detection_algorithm = getattr(config, 'DETECTION_ALGORITHM', 'mvs').lower()
    initial_threshold = getattr(config, 'SEG_THRESHOLD', 1.0)
    
    if detection_algorithm == 'ml':
        print(f'Detection algorithm: ML (Neural Network)')
        detector = MLDetector(
            window_size=config.SEG_WINDOW_SIZE,
            threshold=0.5,  # Probability threshold
            enable_lowpass=config.ENABLE_LOWPASS_FILTER,
            lowpass_cutoff=config.LOWPASS_CUTOFF,
            enable_hampel=config.ENABLE_HAMPEL_FILTER,
            hampel_window=config.HAMPEL_WINDOW,
            hampel_threshold=config.HAMPEL_THRESHOLD
        )
    else:
        print(f'Detection algorithm: MVS (Moving Variance Segmentation)')
        detector = MVSDetector(
            window_size=config.SEG_WINDOW_SIZE,
            threshold=initial_threshold if isinstance(initial_threshold, (int, float)) else 1.0,
            enable_lowpass=config.ENABLE_LOWPASS_FILTER,
            lowpass_cutoff=config.LOWPASS_CUTOFF,
            enable_hampel=config.ENABLE_HAMPEL_FILTER,
            hampel_window=config.HAMPEL_WINDOW,
            hampel_threshold=config.HAMPEL_THRESHOLD
        )
    
    # Initialize and start traffic generator (rate is static from config.py)
    gc.collect()  # Free memory before creating socket
    traffic_gen = TrafficGenerator()
    if config.TRAFFIC_GENERATOR_RATE > 0:
        if not traffic_gen.start(config.TRAFFIC_GENERATOR_RATE):
            print("FATAL: Traffic generator failed to start - CSI will not work")
            print("Check WiFi connection and gateway availability")
            import machine
            time.sleep(5)
            machine.reset()  # Reboot and retry
        
        print(f'Traffic generator started ({config.TRAFFIC_GENERATOR_RATE} pps)')
        
        # Verify CSI packets are flowing with retry logic
        max_tg_retries = 3
        for tg_attempt in range(max_tg_retries):
            time.sleep(2)  # Wait for traffic to start generating CSI packets
            
            print('Waiting for CSI packets...')
            csi_received = 0
            for _ in range(100):  # Max 100 attempts (~5 seconds)
                frame = wlan.csi_read()
                if frame:
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
                print(f'WARNING: Only {csi_received} CSI packets after {max_tg_retries} attempts - TG may not be working')
    
    # P95 Auto-Calibration at boot if subcarriers not configured
    # Handle case where SELECTED_SUBCARRIERS is None, empty, or not defined (commented out)
    current_subcarriers = getattr(config, 'SELECTED_SUBCARRIERS', None)
    needs_calibration = not current_subcarriers
    
    if needs_calibration:
        # Set default fallback before calibration
        run_band_calibration(wlan, detector, traffic_gen, g_state.chip_type)
    else:
        print(f'Using configured subcarriers: {config.SELECTED_SUBCARRIERS}')
    
    # Initialize MQTT (pass calibration function for factory_reset and global state for metrics)
    mqtt_handler = MQTTHandler(config, detector, wlan, traffic_gen, run_band_calibration, g_state)
    mqtt_handler.connect()
    
    # Publish info after boot (always, to show current configuration)
    #print('Publishing system info...')
    mqtt_handler.publish_info()
    
    print('')
    print('  __  __ _                    _____ ____  ____            _            ')
    print(' |  \\/  (_) ___ _ __ ___     | ____/ ___||  _ \\ ___  ___| |_ _ __ ___ ')
    print(' | |\\/| | |/ __| \'__/ _ \\ __ |  _| \\___ \\| |_) / _ \\/ __| __| \'__/ _ \\')
    print(' | |  | | | (__| | | (_) |__|| |___ ___) |  __/  __/ (__| |_| | |  __/')
    print(' |_|  |_|_|\\___|_|  \\___/    |_____|____/|_|   \\___|\\___|\\__|_|  \\___|')
    print('')
    print(' Motion detection system based on Wi-Fi spectrum analysis')
    print('')
    
    # Force garbage collection before main loop
    gc.collect()
    print(f'Free memory before main loop: {gc.mem_free()} bytes')
    
    # Main CSI processing loop with integrated MQTT publishing
    publish_counter = 0
    last_dropped = 0
    filtered_count = 0  # Packets with wrong SC count
    last_publish_time = time.ticks_ms()
    
    # Calculate optimal sleep based on traffic rate
    publish_rate = traffic_gen.get_rate() if traffic_gen.is_running() else 100
       
    try:
        while True:
            loop_start = time.ticks_us()
            
            # Suspend main loop during calibration
            if g_state.calibration_mode:
                time.sleep_ms(1000) # Sleep for 1 second to yield CPU
                continue
            
            # Check MQTT messages (non-blocking)
            mqtt_handler.check_messages()
            
            frame = wlan.csi_read()
            
            if frame:
                # Filter packets by expected CSI length (HT20: 128 bytes)
                if len(frame[5]) != EXPECTED_CSI_LEN:
                    filtered_count += 1
                    # Log warning every 100 filtered packets
                    if filtered_count % 100 == 1:
                        print(f"[WARN] Filtered {filtered_count} packets with wrong SC count (got {len(frame[5])} bytes, expected {EXPECTED_CSI_LEN})")
                    del frame
                    continue
                
                # Extract data and free frame immediately to save memory
                csi_data = frame[5][:EXPECTED_CSI_LEN]
                packet_channel = frame[1]
                
                # Apply gain compensation if needed (gain lock skipped or disabled)
                if g_state.needs_compensation:
                    current_agc = frame[22]
                    current_fft = to_signed_int8(frame[23])
                    compensation = calculate_gain_compensation(
                        g_state.baseline_agc, g_state.baseline_fft,
                        current_agc, current_fft
                    )
                    detector.set_gain_compensation(compensation)
                
                del frame
                
                # Process packet through detector interface
                detector.process_packet(csi_data, config.SELECTED_SUBCARRIERS)
                
                publish_counter += 1
                
                # Publish every N packets (where N = publish_rate)
                if publish_counter >= publish_rate:
                    # Detect WiFi channel changes (AP may switch channels automatically)
                    # Channel changes cause CSI spikes that trigger false motion detection
                    if g_state.current_channel != 0 and packet_channel != g_state.current_channel:
                        print(f"[WARN] WiFi channel changed: {g_state.current_channel} -> {packet_channel}, resetting detection buffer")
                        detector.reset()
                    g_state.current_channel = packet_channel
                    
                    # Update state (lazy evaluation)
                    metrics = detector.update_state()
                    current_time = time.ticks_ms()
                    time_delta = time.ticks_diff(current_time, last_publish_time)
                    
                    # Calculate packets per second
                    pps = int((publish_counter * 1000) / time_delta) if time_delta > 0 else 0
                    
                    dropped = wlan.csi_dropped()
                    dropped_delta = dropped - last_dropped
                    last_dropped = dropped
                    
                    state_str = 'MOTION' if metrics['state'] == 1 else 'IDLE'
                    motion_metric = metrics.get('moving_variance', metrics.get('jitter', metrics.get('probability', 0)))
                    threshold = metrics['threshold']
                    is_ml = 'probability' in metrics
                    # For ML, probability and threshold are both 0-1, so progress = probability
                    if is_ml:
                        progress = motion_metric  # probability is already 0-1
                    else:
                        progress = motion_metric / threshold if threshold > 0 else 0
                    progress_bar = format_progress_bar(progress, threshold, is_probability=is_ml)
                    print(f"{progress_bar} | pkts:{publish_counter} drop:{dropped_delta} pps:{pps} | "
                          f"mvmt:{motion_metric:.4f} thr:{threshold:.4f} | {state_str}")
                    
                    mqtt_handler.publish_state(
                        motion_metric,
                        metrics['state'],
                        threshold,
                        publish_counter,
                        dropped_delta,
                        pps
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
        mqtt_handler.disconnect()        
        if traffic_gen.is_running():
            traffic_gen.stop()
        cleanup_wifi(wlan)

if __name__ == '__main__':
    main()
