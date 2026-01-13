"""
Micro-ESPectre - Main Application

Motion detection using WiFi CSI and MVS segmentation.
Main entry point for the Micro-ESPectre system running on ESP32-C6.

Author: Francesco Pace <francesco.pace@gmail.com>
License: GPLv3
"""
import network
import time
import gc
import os
from src.segmentation import SegmentationContext
from src.mqtt.handler import MQTTHandler
from src.traffic_generator import TrafficGenerator
from src.nvs_storage import NVSStorage
from src.band_calibrator import BandCalibrator, cleanup_buffer_file
import src.config as config

# Default subcarriers (used if not configured or for fallback in case of error)
DEFAULT_SUBCARRIERS = [11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22]

# Gain lock configuration
GAIN_LOCK_PACKETS = 300  # ~3 seconds at 100 Hz

# Global state for calibration mode and performance metrics
class GlobalState:
    def __init__(self):
        self.calibration_mode = False  # Flag to suspend main loop during calibration
        self.loop_time_us = 0  # Last loop iteration time in microseconds
        self.chip_type = None  # Detected chip type (S3, C6, etc.)
        self.current_channel = 0  # Track WiFi channel for change detection
        self.expected_subcarriers = 64  # Expected SC count from gain lock (64, 128, 256)


g_state = GlobalState()

def print_wifi_status(wlan):
    """Print WiFi connection status with configuration details."""
    ip = wlan.ifconfig()[0]
    
    # Protocol decode using exposed constants
    PROTOCOL_NAMES = {
        network.MODE_11B: 'b',
        network.MODE_11G: 'g', 
        network.MODE_11N: 'n',
    }
    # Add 11ax only if available (ESP32-C5/C6)
    if hasattr(network, 'MODE_11AX'):
        PROTOCOL_NAMES[network.MODE_11AX] = 'ax'
    
    proto_val = wlan.config('protocol')
    modes = [name for bit, name in PROTOCOL_NAMES.items() if proto_val & bit]
    protocol_str = '802.11' + '/'.join(modes) if modes else f'0x{proto_val:02x}'
    
    # Bandwidth decode
    BW_NAMES = {wlan.BW_HT20: 'HT20', wlan.BW_HT40: 'HT40'}
    bw_str = BW_NAMES.get(wlan.config('bandwidth'), 'unknown')
    
    # Promiscuous
    prom_str = 'ON' if wlan.config('promiscuous') else 'OFF'
    
    print(f"WiFi connected - IP: {ip}, Protocol: {protocol_str}, Bandwidth: {bw_str}, Promiscuous: {prom_str}")

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
    
    # Protocol decode using exposed constants
    PROTOCOL_NAMES = {
        network.MODE_11B: 'b',
        network.MODE_11G: 'g', 
        network.MODE_11N: 'n',
    }
    # Add 11ax only if available (ESP32-C5/C6)
    if hasattr(network, 'MODE_11AX'):
        PROTOCOL_NAMES[network.MODE_11AX] = 'ax'
    
    proto_val = wlan.config('protocol')
    modes = [name for bit, name in PROTOCOL_NAMES.items() if proto_val & bit]
    protocol_str = '802.11' + '/'.join(modes) if modes else f'0x{proto_val:02x}'
    
    # Bandwidth decode
    BW_NAMES = {wlan.BW_HT20: 'HT20', wlan.BW_HT40: 'HT40'}
    bw_str = BW_NAMES.get(wlan.config('bandwidth'), 'unknown')
    
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


def format_progress_bar(score, threshold, width=20):
    """Format progress bar for console output"""
    threshold_pos = 15
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
    Also counts subcarrier types to determine the dominant type for the session.
    
    Respects config.GAIN_LOCK_MODE:
    - "auto": Lock gain, but skip if signal too strong (AGC < MIN_SAFE_AGC)
    - "enabled": Always force gain lock
    - "disabled": Never lock gain
    
    Args:
        wlan: WLAN instance with CSI enabled
        
    Returns:
        tuple: (agc_gain, fft_gain, skipped, dominant_sc) where:
            - skipped=True if gain lock was skipped
            - dominant_sc = dominant subcarrier count (64, 128, or 256)
    """
    # Check configuration mode
    mode = getattr(config, 'GAIN_LOCK_MODE', 'auto').lower()
    min_safe_agc = getattr(config, 'GAIN_LOCK_MIN_SAFE_AGC', 30)
    
    # Skip gain lock if disabled or not supported
    gain_lock_supported = hasattr(wlan, 'csi_gain_lock_supported') and wlan.csi_gain_lock_supported()
    
    if mode == 'disabled' or not gain_lock_supported:
        reason = "Disabled by configuration" if mode == 'disabled' else "Not supported on this platform"
        print(f"Gain lock: {reason}")
        # Still need to determine subcarrier count from a few packets
        print("  Detecting subcarrier count...")
        sc_counts = {64: 0, 128: 0, 256: 0}
        scan_count = 0
        while scan_count < 50:
            frame = wlan.csi_read()
            if frame:
                num_sc = len(frame[5]) // 2
                if num_sc in sc_counts:
                    sc_counts[num_sc] += 1
                scan_count += 1
                del frame  # Free memory immediately
        dominant_sc = max(sc_counts, key=sc_counts.get) if any(sc_counts.values()) else 64
        print(f"  Detected: 64SC={sc_counts[64]}, 128SC={sc_counts[128]}, 256SC={sc_counts[256]} -> using {dominant_sc}")
        return None, None, False, dominant_sc
    
    print('')
    print('-'*60)
    print(f'Gain Lock Calibration (~3 seconds) [mode: {mode}]')
    print('-'*60)
    
    agc_sum = 0
    fft_sum = 0
    count = 0
    
    # Count subcarrier types (bytes / 2 = subcarriers)
    sc_counts = {64: 0, 128: 0, 256: 0}
    
    while count < GAIN_LOCK_PACKETS:
        frame = wlan.csi_read()
        if frame:
            # frame[22] = agc_gain, frame[23] = fft_gain
            agc_sum += frame[22]
            fft_sum += frame[23]
            
            # Count subcarrier type from CSI data length (avoid storing reference)
            num_sc = len(frame[5]) // 2
            if num_sc in sc_counts:
                sc_counts[num_sc] += 1
            
            del frame  # Free memory immediately
            count += 1
            
            # Progress every 25% (with GC to prevent ENOMEM)
            if count == GAIN_LOCK_PACKETS // 4:
                gc.collect()
                print(f"  Gain calibration 25%: AGC~{agc_sum // count}, FFT~{fft_sum // count}")
            elif count == GAIN_LOCK_PACKETS // 2:
                gc.collect()
                print(f"  Gain calibration 50%: AGC~{agc_sum // count}, FFT~{fft_sum // count}")
            elif count == (GAIN_LOCK_PACKETS * 3) // 4:
                gc.collect()
                print(f"  Gain calibration 75%: AGC~{agc_sum // count}, FFT~{fft_sum // count}")
    
    # Calculate averages
    avg_agc = agc_sum // GAIN_LOCK_PACKETS
    avg_fft = fft_sum // GAIN_LOCK_PACKETS
    
    # Determine dominant subcarrier count
    dominant_sc = max(sc_counts, key=sc_counts.get)
    print(f"  Subcarrier stats: 64SC={sc_counts[64]}, 128SC={sc_counts[128]}, 256SC={sc_counts[256]} -> using {dominant_sc}")
    
    # In auto mode, skip gain lock if signal is too strong
    if mode == 'auto' and avg_agc < min_safe_agc:
        print(f"WARNING: Signal too strong (AGC={avg_agc} < {min_safe_agc}) - skipping gain lock")
        print(f"         Move sensor 2-3 meters from AP for optimal performance")
        return avg_agc, avg_fft, True, dominant_sc
    
    # Lock the gain values
    wlan.csi_force_gain(avg_agc, avg_fft)
    print(f"Gain locked: AGC={avg_agc}, FFT={avg_fft} (after {GAIN_LOCK_PACKETS} packets)")
    
    return avg_agc, avg_fft, False, dominant_sc


def run_band_calibration(wlan, nvs, seg, traffic_gen, chip_type=None):
    """
    Run P95 band calibration (with gain lock phase first)
    
    Args:
        wlan: WLAN instance
        nvs: NVSStorage instance
        seg: SegmentationContext instance
        traffic_gen: TrafficGenerator instance
        chip_type: Chip type ('C5', 'C6', 'S3', etc.) for subcarrier filtering
    
    Returns:
        bool: True if calibration successful
    """
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
    print('Please remain still for calibration...')
    
    # Phase 1: Gain Lock (~3 seconds)
    # Stabilizes AGC/FFT before calibration to ensure clean data
    # Also determines dominant subcarrier count for the session
    agc, fft, skipped, dominant_sc = run_gain_lock(wlan)
    
    # Save dominant SC for main loop packet filtering
    g_state.expected_subcarriers = dominant_sc
    
    if skipped:
        print("Note: Proceeding with band calibration without gain lock")
    
    print('')
    print('-'*60)
    print(f'Band Calibration (~7 seconds) [expecting {dominant_sc} SC]')
    print('-'*60)
    
    # Initialize band calibrator with expected subcarrier count
    # Uses P95 moving variance optimization for optimal band selection
    band_calibrator = BandCalibrator(
        buffer_size=config.CALIBRATION_BUFFER_SIZE,
        expected_subcarriers=dominant_sc
    )
    
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
            # Truncate to expected SC count to save memory
            csi_data = frame[5][:g_state.expected_subcarriers * 2]
            del frame  # Free memory immediately
            calibration_progress = band_calibrator.add_packet(csi_data)
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
    
    # Calibrate using P95 moving variance approach
    success = False
    config.SELECTED_SUBCARRIERS = DEFAULT_SUBCARRIERS
    
    # Stop traffic generator during band evaluation to free memory
    tg_was_running = traffic_gen.is_running()
    if tg_was_running:
        traffic_gen.stop()
        gc.collect()
    
    try:
        selected_band, adaptive_threshold = band_calibrator.calibrate()
        
        if selected_band and len(selected_band) == 12:
            # Calibration successful
            config.SELECTED_SUBCARRIERS = selected_band
            # Apply adaptive threshold to segmentation context
            seg.set_adaptive_threshold(adaptive_threshold)
            success = True
            
            print('')
            print('='*60)
            print('Subcarrier Calibration Successful!')
            print(f'   Selected band: {selected_band}')
            print(f'   Adaptive threshold: {adaptive_threshold:.4f}')
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
    band_calibrator.free_buffer()
    band_calibrator = None
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

def main():
    """Main application loop"""
    print('Micro-ESPectre starting...')
    
    # Detect chip type and get CSI scale
    g_state.chip_type = os.uname().machine  # Save for MQTT factory_reset
    print(f'Detected chip: {g_state.chip_type}')
    
    # Connect to WiFi
    wlan = connect_wifi()
    
    # Initialize segmentation with full configuration
    seg = SegmentationContext(
        window_size=config.SEG_WINDOW_SIZE,
        threshold=1.0,  # Default, overwritten by adaptive threshold after calibration
        enable_lowpass=config.ENABLE_LOWPASS_FILTER,
        lowpass_cutoff=config.LOWPASS_CUTOFF,
        enable_hampel=config.ENABLE_HAMPEL_FILTER,
        hampel_window=config.HAMPEL_WINDOW,
        hampel_threshold=config.HAMPEL_THRESHOLD,
        enable_features=config.ENABLE_FEATURES
    )
    
    # Load saved configuration (segmentation parameters only)
    nvs = NVSStorage()
    saved_config = nvs.load_and_apply(seg)
    
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
        time.sleep(2)  # Wait for traffic to start generating CSI packets
        
        # Verify CSI packets are flowing before proceeding
        print('Waiting for CSI packets...')
        csi_received = 0
        for _ in range(100):  # Max 100 attempts (~5 seconds)
            frame = wlan.csi_read()
            if frame:
                csi_received += 1
                if csi_received >= 10:
                    break
            time.sleep(0.05)
        
        if csi_received < 10:
            print(f'WARNING: Only {csi_received} CSI packets received - TG may not be working')
    
    # P95 Auto-Calibration at boot if subcarriers not configured
    # Handle case where SELECTED_SUBCARRIERS is None, empty, or not defined (commented out)
    current_subcarriers = getattr(config, 'SELECTED_SUBCARRIERS', None)
    needs_calibration = not current_subcarriers
    
    if needs_calibration:
        # Set default fallback before calibration
        run_band_calibration(wlan, nvs, seg, traffic_gen, g_state.chip_type)
    else:
        print(f'Using configured subcarriers: {config.SELECTED_SUBCARRIERS}')
    
    # Initialize MQTT (pass calibration function for factory_reset and global state for metrics)
    mqtt_handler = MQTTHandler(config, seg, wlan, traffic_gen, run_band_calibration, g_state)
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
    print(' Motion detection system based on Wi-Fi spectre analysis')
    print('')
    
    # Force garbage collection before main loop
    gc.collect()
    print(f'Free memory before main loop: {gc.mem_free()} bytes')
    
    # Main CSI processing loop with integrated MQTT publishing
    publish_counter = 0
    last_dropped = 0
    last_publish_time = time.ticks_ms()
    
    # Calculate optimal sleep based on traffic rate
    publish_rate = traffic_gen.get_rate() if traffic_gen.is_running() else 100
    
    # Pre-calculate payload size (SC count * 2 bytes per I/Q pair)
    payload_size = g_state.expected_subcarriers * 2
       
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
                # Filter packets by expected subcarrier count BEFORE truncating
                packet_sc = len(frame[5]) // 2
                if packet_sc != g_state.expected_subcarriers:
                    del frame
                    continue  # Skip packets with wrong SC count
                
                # Extract data and free frame immediately to save memory
                csi_data = frame[5][:payload_size]
                packet_channel = frame[1]
                del frame
                
                turbulence = seg.calculate_spatial_turbulence(csi_data, config.SELECTED_SUBCARRIERS)
                seg.add_turbulence(turbulence)
                
                publish_counter += 1
                
                # Publish every N packets (where N = publish_rate)
                if publish_counter >= publish_rate:
                    # Detect WiFi channel changes (AP may switch channels automatically)
                    # Channel changes cause CSI spikes that trigger false motion detection
                    if g_state.current_channel != 0 and packet_channel != g_state.current_channel:
                        print(f"[WARN] WiFi channel changed: {g_state.current_channel} -> {packet_channel}, resetting detection buffer")
                        seg.reset(full=True)
                    g_state.current_channel = packet_channel
                    
                    # Calculate variance and update state (lazy evaluation)
                    metrics = seg.update_state()
                    current_time = time.ticks_ms()
                    time_delta = time.ticks_diff(current_time, last_publish_time)
                    
                    # Calculate packets per second
                    pps = int((publish_counter * 1000) / time_delta) if time_delta > 0 else 0
                    
                    dropped = wlan.csi_dropped()
                    dropped_delta = dropped - last_dropped
                    last_dropped = dropped
                    
                    state_str = 'MOTION' if metrics['state'] == 1 else 'IDLE'
                    progress = metrics['moving_variance'] / metrics['threshold'] if metrics['threshold'] > 0 else 0
                    progress_bar = format_progress_bar(progress, metrics['threshold'])
                    print(f"{progress_bar} | pkts:{publish_counter} drop:{dropped_delta} pps:{pps} | "
                          f"mvmt:{metrics['moving_variance']:.4f} thr:{metrics['threshold']:.4f} | {state_str}")
                    
                    # Compute features at publish time (not per-packet)
                    features = None
                    confidence = None
                    triggered = None
                    if seg.features_ready():
                        features = seg.compute_features()
                        confidence, triggered = seg.compute_confidence(features)
                    
                    mqtt_handler.publish_state(
                        metrics['moving_variance'],
                        metrics['state'],
                        metrics['threshold'],
                        publish_counter,
                        dropped_delta,
                        pps,
                        features,
                        confidence,
                        triggered
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
