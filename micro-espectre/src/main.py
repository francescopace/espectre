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
from src.nbvi_calibrator import NBVICalibrator
import src.config as config

# Default subcarriers (used if not configured or for fallback in case of error)
DEFAULT_SUBCARRIERS = [11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22]

# Gain lock configuration
GAIN_LOCK_PACKETS = 300  # ~3 seconds at 100 Hz

# Try to import local configuration (overrides config.py defaults)
try:
    from src.config_local import WIFI_SSID, WIFI_PASSWORD
except ImportError:
    WIFI_SSID = config.WIFI_SSID
    WIFI_PASSWORD = config.WIFI_PASSWORD

# Try to import MQTT settings from config_local (optional)
try:
    from src.config_local import MQTT_BROKER, MQTT_PORT, MQTT_USERNAME, MQTT_PASSWORD
    config.MQTT_BROKER = MQTT_BROKER
    config.MQTT_PORT = MQTT_PORT
    config.MQTT_USERNAME = MQTT_USERNAME
    config.MQTT_PASSWORD = MQTT_PASSWORD
except ImportError:
    pass  # Use defaults from config.py


# Global state for calibration mode and performance metrics
class GlobalState:
    def __init__(self):
        self.calibration_mode = False  # Flag to suspend main loop during calibration
        self.loop_time_us = 0  # Last loop iteration time in microseconds
        self.chip_type = None  # Detected chip type (S3, C6, etc.)
        self.current_channel = 0  # Track WiFi channel for change detection


g_state = GlobalState()


def connect_wifi():
    """Connect to WiFi"""
    
    print(f"Connecting to WiFi...")
    
    gc.collect()
    wlan = network.WLAN(network.STA_IF)
    
    # Initialize WiFi
    if wlan.active():
        wlan.active(False)
    wlan.active(True)
    
    if not wlan.active():
        raise Exception("WiFi failed to activate")
    
    # Disconnect if already connected
    if wlan.isconnected():
        wlan.disconnect()
    
    # Wait for hardware initialization
    time.sleep(2)  
    
    # Connect
    wlan.connect(WIFI_SSID, WIFI_PASSWORD)
    
    # Wait for connection
    timeout = 30
    while not wlan.isconnected() and timeout > 0:
        time.sleep(1)
        timeout -= 1
    
    if wlan.isconnected():
        print(f"WiFi connected - IP: {wlan.ifconfig()[0]}")
        time.sleep(1)  # Stabilization
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
    
    Args:
        wlan: WLAN instance with CSI enabled
        
    Returns:
        tuple: (agc_gain, fft_gain) or (None, None) if not supported
    """
    # Check if gain lock is supported on this platform
    if not hasattr(wlan, 'csi_gain_lock_supported') or not wlan.csi_gain_lock_supported():
        print("Gain lock: Not supported on this platform (skipping)")
        return None, None
    
    print('')
    print('-'*60)
    print('Gain Lock Calibration (~3 seconds)')
    print('-'*60)
    
    agc_sum = 0
    fft_sum = 0
    count = 0
    
    while count < GAIN_LOCK_PACKETS:
        frame = wlan.csi_read()
        if frame:
            # frame[22] = agc_gain, frame[23] = fft_gain
            agc_sum += frame[22]
            fft_sum += frame[23]
            count += 1
            
            # Progress every 25%
            if count == GAIN_LOCK_PACKETS // 4:
                print(f"  Gain calibration 25%: AGC~{agc_sum // count}, FFT~{fft_sum // count}")
            elif count == GAIN_LOCK_PACKETS // 2:
                print(f"  Gain calibration 50%: AGC~{agc_sum // count}, FFT~{fft_sum // count}")
            elif count == (GAIN_LOCK_PACKETS * 3) // 4:
                print(f"  Gain calibration 75%: AGC~{agc_sum // count}, FFT~{fft_sum // count}")
    
    # Calculate averages
    avg_agc = agc_sum // GAIN_LOCK_PACKETS
    avg_fft = fft_sum // GAIN_LOCK_PACKETS
    
    # Lock the gain values
    wlan.csi_force_gain(avg_agc, avg_fft)
    print(f"Gain locked: AGC={avg_agc}, FFT={avg_fft} (after {GAIN_LOCK_PACKETS} packets)")
    
    return avg_agc, avg_fft


def run_nbvi_calibration(wlan, nvs, seg, traffic_gen, chip_type=None):
    """
    Run NBVI calibration (with gain lock phase first)
    
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
    
    print('')
    print('='*60)
    print('Two-Phase Calibration Starting')
    print('='*60)
    print(f'Free memory: {gc.mem_free()} bytes')
    print('Please remain still for calibration...')
    
    # Phase 1: Gain Lock (~3 seconds)
    # Stabilizes AGC/FFT before NBVI to ensure clean data
    run_gain_lock(wlan)
    
    print('')
    print('-'*60)
    print('NBVI Subcarrier Selection (~7 seconds)')
    print('-'*60)
    
    # Initialize Subcarrier calibrator with chip-specific subcarrier filtering
    # Normalization: attenuate if baseline > 0.25, otherwise no scaling
    nbvi_calibrator = NBVICalibrator(
        buffer_size=config.NBVI_BUFFER_SIZE,
        percentile=config.NBVI_PERCENTILE,
        alpha=config.NBVI_ALPHA,
        min_spacing=config.NBVI_MIN_SPACING,
        chip_type=chip_type
    )
    
    # Collect packets for calibration (now with stable gain)
    calibration_progress = 0
    timeout_counter = 0
    max_timeout = 15000  # 15 seconds
    packets_read = 0
    last_progress_time = time.ticks_ms()
    last_progress_count = 0
    
    while calibration_progress < config.NBVI_BUFFER_SIZE:
        frame = wlan.csi_read()
        packets_read += 1
        
        if frame:
            csi_data = frame[5][:128]  # frame[5] = data (tuple API)
            calibration_progress = nbvi_calibrator.add_packet(csi_data)
            timeout_counter = 0  # Reset timeout on successful read
            
            # Print progress every 100 packets with pps
            if calibration_progress % 100 == 0:
                current_time = time.ticks_ms()
                elapsed = time.ticks_diff(current_time, last_progress_time)
                packets_delta = calibration_progress - last_progress_count
                pps = int((packets_delta * 1000) / elapsed) if elapsed > 0 else 0
                dropped = wlan.csi_dropped()
                tg_pps = traffic_gen.get_actual_pps()
                print(f"Collecting {calibration_progress}/{config.NBVI_BUFFER_SIZE} packets... (pps:{pps}, TG:{tg_pps}, drop:{dropped})")
                last_progress_time = current_time
                last_progress_count = calibration_progress
        else:
            time.sleep_us(100)
            timeout_counter += 1
            
            # Debug: print if stuck
            if timeout_counter % 1000 == 0:
                # Check if traffic generator is still running
                tg_running = traffic_gen.is_running()
                tg_count = traffic_gen.get_packet_count()
                dropped = wlan.csi_dropped()
                print(f"Waiting for CSI... ({timeout_counter/1000:.0f}s, progress: {calibration_progress}, TG running: {tg_running}, TG packets: {tg_count}, dropped: {dropped})")
            
            if timeout_counter >= max_timeout:
                print(f"Timeout waiting for CSI packets (collected {calibration_progress}/{config.NBVI_BUFFER_SIZE})")
                print("Calibration aborted - using default band")
                return False
    
    gc.collect()  # Free any temporary objects before calibration
    
    # Calibrate using percentile-based approach
    success = False
    config.SELECTED_SUBCARRIERS = DEFAULT_SUBCARRIERS
    config.NORMALIZATION_SCALE = 1.0
    try:
        selected_band, normalization_scale = nbvi_calibrator.calibrate(config.SELECTED_SUBCARRIERS)
        
        if selected_band and len(selected_band) == 12:
            # Calibration successful
            config.SELECTED_SUBCARRIERS = selected_band
            config.NORMALIZATION_SCALE = normalization_scale
            # Apply normalization scale to segmentation context
            seg.set_normalization_scale(normalization_scale)
            success = True
            
            print('')
            print('='*60)
            print('Subcarrier Calibration Successful!')
            print(f'   Selected band: {selected_band}')
            print(f'   Normalization scale: {normalization_scale:.3f}')
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
    nbvi_calibrator.free_buffer()
    nbvi_calibrator = None
    gc.collect()
    
    # Resume main loop
    g_state.calibration_mode = False
    
    return success

def get_csi_scale(chip_type):
    """
    Get CSI scale value based on chip type.
    
    Legacy platforms (ESP32, S2, S3, C3) need manual scaling (shift=4) to match
    the C++ ESPectre behavior. Modern platforms (C5, C6) use auto-scaling.
    
    Returns:
        int or None: Scale value (4 for legacy platforms, None for modern)
    """
    # Normalize chip type string for comparison
    chip_lower = chip_type.lower() if chip_type else ""
    
    # Modern platforms (C5, C6): use auto-scaling (val_scale_cfg=0)
    if 'esp32c5' in chip_lower or 'esp32c6' in chip_lower:
        return None  # Auto-scaling
    
    # Legacy platforms (ESP32, S2, S3, C3): use manual scaling
    # Note: shift works as LEFT shift on amplitudes (multiply by 2^shift)
    # Variance scales quadratically: shift=4 → amplitudes ×16, variance ×256
    # shift=4 provides good dynamic range for motion detection
    return 4


def main():
    """Main application loop"""
    print('Micro-ESPectre starting...')
    
    # Detect chip type and get CSI scale
    g_state.chip_type = os.uname().machine  # Save for MQTT factory_reset
    print(f'Detected chip: {g_state.chip_type}')
    
    # Connect to WiFi
    wlan = connect_wifi()
    
    # Configure and enable CSI with platform-specific scaling
    # Legacy platforms (ESP32, S2, S3, C3): scale=4 (divide by 16)
    # Modern platforms (C5, C6): scale=None (auto-scaling)
    csi_scale = get_csi_scale(g_state.chip_type)
    wlan.csi_enable(buffer_size=config.CSI_BUFFER_SIZE, scale=csi_scale)
    if csi_scale is not None:
        print(f'CSI scale: {csi_scale} (manual scaling, divide by {2**csi_scale})')
    
    # Initialize segmentation with full configuration
    seg = SegmentationContext(
        window_size=config.SEG_WINDOW_SIZE,
        threshold=config.SEG_THRESHOLD,
        enable_lowpass=config.ENABLE_LOWPASS_FILTER,
        lowpass_cutoff=config.LOWPASS_CUTOFF,
        enable_hampel=config.ENABLE_HAMPEL_FILTER,
        hampel_window=config.HAMPEL_WINDOW,
        hampel_threshold=config.HAMPEL_THRESHOLD,
        enable_features=config.ENABLE_FEATURES,
        normalization_scale=config.NORMALIZATION_SCALE
    )
    
    # Load saved configuration (segmentation parameters only)
    nvs = NVSStorage()
    saved_config = nvs.load_and_apply(seg)
    
    # Initialize and start traffic generator (rate is static from config.py)
    traffic_gen = TrafficGenerator()
    if config.TRAFFIC_GENERATOR_RATE > 0:
        if not traffic_gen.start(config.TRAFFIC_GENERATOR_RATE):
            print("FATAL: Traffic generator failed to start - CSI will not work")
            print("Check WiFi connection and gateway availability")
            import machine
            time.sleep(5)
            machine.reset()  # Reboot and retry
        
        print(f'Traffic generator started ({config.TRAFFIC_GENERATOR_RATE} pps)')
        time.sleep(1)  # Wait for traffic to start generating CSI packets
    
    # NBVI Auto-Calibration at boot if subcarriers not configured
    # Handle case where SELECTED_SUBCARRIERS is None, empty, or not defined (commented out)
    current_subcarriers = getattr(config, 'SELECTED_SUBCARRIERS', None)
    needs_calibration = not current_subcarriers
    
    if needs_calibration:
        # Set default fallback before calibration
        run_nbvi_calibration(wlan, nvs, seg, traffic_gen, g_state.chip_type)
    else:
        print(f'Using configured subcarriers: {config.SELECTED_SUBCARRIERS}')
    
    # Initialize MQTT (pass calibration function for factory_reset and global state for metrics)
    mqtt_handler = MQTTHandler(config, seg, wlan, traffic_gen, run_nbvi_calibration, g_state)
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
                # ESP32-C6 provides 512 bytes but we use only first 128 bytes
                # This avoids corrupted data in the extended buffer
                csi_data = frame[5][:128]  # frame[5] = data (tuple API)
                turbulence = seg.calculate_spatial_turbulence(csi_data, config.SELECTED_SUBCARRIERS)
                seg.add_turbulence(turbulence)
                
                publish_counter += 1
                
                # Publish every N packets (where N = publish_rate)
                if publish_counter >= publish_rate:
                    # Detect WiFi channel changes (AP may switch channels automatically)
                    # Channel changes cause CSI spikes that trigger false motion detection
                    # Check only at publish time to reduce overhead
                    # frame[1] = channel (from CSI packet metadata)
                    packet_channel = frame[1]
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
        wlan.csi_disable()
        mqtt_handler.disconnect()        
        if traffic_gen.is_running():
            traffic_gen.stop()

if __name__ == '__main__':
    main()
