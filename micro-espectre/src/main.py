"""
Micro-ESPectre - Main Application
Motion detection using WiFi CSI and MVS segmentation

Author: Francesco Pace <francesco.pace@gmail.com>
License: GPLv3
"""
import network
import time
import _thread
from src.segmentation import SegmentationContext
from src.mqtt.handler import MQTTHandler
from src.traffic_generator import TrafficGenerator
from src.nvs_storage import NVSStorage
import src.config as config

# Try to import local configuration (overrides config.py defaults)
try:
    from config_local import WIFI_SSID, WIFI_PASSWORD, MQTT_BROKER, MQTT_PORT, MQTT_USERNAME, MQTT_PASSWORD
    # Override config with local settings
    config.MQTT_BROKER = MQTT_BROKER
    config.MQTT_PORT = MQTT_PORT
    config.MQTT_USERNAME = MQTT_USERNAME
    config.MQTT_PASSWORD = MQTT_PASSWORD
except ImportError:
    WIFI_SSID = config.WIFI_SSID
    WIFI_PASSWORD = config.WIFI_PASSWORD


# Global state (shared between threads)
class GlobalState:
    def __init__(self):
        self.packets_processed = 0
        self.current_state = 0  # STATE_IDLE
        self.current_variance = 0.0
        self.current_threshold = 0.0
        self.lock = _thread.allocate_lock()


g_state = GlobalState()


def connect_wifi():
    """Connect to WiFi"""
    
    # Initialize WiFi in station mode
    wlan = network.WLAN(network.STA_IF)
    wlan.active(True)
    
    print("WiFi initialized")
    mac = wlan.config('mac')
    print("MAC address: " + ':'.join('%02x' % b for b in mac))
    
    # Configure WiFi BEFORE connecting (critical for ESP32-C6 CSI)
    print("Configuring WiFi for CSI...")
    wlan.config(pm=wlan.PM_NONE)  # Disable power save
    # Note: protocol and bandwidth are set automatically by MicroPython
    
    time.sleep(1)  # Small delay to ensure settings take effect
    
    # Connect to WiFi (REQUIRED for CSI)
    print("Connecting to WiFi...")
    wlan.connect(WIFI_SSID, WIFI_PASSWORD)

    print("Checking WiFi connection...")
    
    # Wait for connection
    timeout = 10
    while not wlan.isconnected() and timeout > 0:
        time.sleep(0.5)
        timeout -= 0.5
    
    if not wlan.isconnected():
        raise Exception("Failed to connect to WiFi! CSI requires WiFi connection to work.")
    
    print("WiFi connected to: " + WIFI_SSID)
    
    # Wait for WiFi to be fully ready (critical for ESP32-C6)
    print("Waiting for WiFi to stabilize...")
    time.sleep(2)
    
    return wlan


def format_progress_bar(score, threshold, width=20):
    """Format progress bar for console output"""
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


def mqtt_publish_task(mqtt_handler, wlan):
    """MQTT publish task"""
    last_packets_processed = 0
    last_dropped = 0
    
    while True:
        try:
            time.sleep(1.0)  # Publish interval: 1 second
            
            current_time = time.ticks_ms()
            
            # Check for incoming MQTT commands
            mqtt_handler.check_messages()
            
            # Read global state with lock
            with g_state.lock:
                current_state = g_state.current_state
                current_variance = g_state.current_variance
                current_threshold = g_state.current_threshold
                packets_processed = g_state.packets_processed
            
            # Calculate packet delta (packets processed since last cycle)
            packet_delta = packets_processed - last_packets_processed
            last_packets_processed = packets_processed
            
            # Update packet count in command handler
            mqtt_handler.update_packet_count(packets_processed)
            
            # Check for dropped CSI packets
            dropped = wlan.csi.dropped()
            dropped_delta = dropped - last_dropped
            last_dropped = dropped
            
            # Logging
            state_str = 'MOTION' if current_state == 1 else 'IDLE'
            
            # Calculate progress (variance / threshold)
            progress = current_variance / current_threshold if current_threshold > 0 else 0
            progress_bar = format_progress_bar(progress, 1.0)
            
            # Log with dropped packets if any
            if dropped_delta > 0:
                print(f"📊 {progress_bar} | pkts:{packet_delta} dropped:{dropped_delta} "
                      f"mvmt:{current_variance:.4f} thr:{current_threshold:.4f} | "
                      f"{state_str}")
            else:
                print(f"📊 {progress_bar} | pkts:{packet_delta} "
                      f"mvmt:{current_variance:.4f} thr:{current_threshold:.4f} | "
                      f"{state_str}")
            
            # Publish to MQTT using handler
            mqtt_handler.publish_state(
                current_variance,
                current_state,
                current_threshold,
                packet_delta,
                current_time
            )
            
        except Exception as e:
            print(f"MQTT publish task error: {e}")
            time.sleep(1)


def main():
    """Main application loop"""
    print('Micro-ESPectre starting...')
    
    # Connect to WiFi
    wlan = connect_wifi()
    
    # Configure CSI
    print("Configuring CSI...")
    wlan.csi.config(buffer_size=config.CSI_BUFFER_SIZE)
    
    # Enable CSI
    print("Enabling CSI...")
    wlan.csi.enable()
    
    # Initialize segmentation
    print('Initializing segmentation...')
    seg = SegmentationContext(
        window_size=config.SEG_WINDOW_SIZE,
        min_length=config.SEG_MIN_LENGTH,
        max_length=config.SEG_MAX_LENGTH,
        threshold=config.SEG_THRESHOLD
    )
    
    # Load saved configuration if exists
    print('Loading saved configuration...')
    nvs = NVSStorage()
    saved_config = nvs.load_and_apply(seg, config)
    
    # Initialize traffic generator
    print('Initializing traffic generator...')
    traffic_gen = TrafficGenerator()
    
    # Determine traffic generator rate (saved config or default)
    traffic_rate = config.TRAFFIC_GENERATOR_RATE
    if saved_config and "traffic_generator" in saved_config:
        traffic_rate = saved_config["traffic_generator"].get("rate", traffic_rate)
    
    # Start traffic generator if configured
    if traffic_rate > 0:
        if traffic_gen.start(traffic_rate):
            print(f'✅ Traffic generator started ({traffic_rate} pps)')
        else:
            print('⚠️  Failed to start traffic generator')
    
    # Initialize MQTT handler with command support
    print('Initializing MQTT handler...')
    mqtt_handler = MQTTHandler(config, seg, traffic_gen)
    mqtt = mqtt_handler.connect()
    
    # Start MQTT publish task in separate thread (like C version)
    print("Starting MQTT publish task...")
    _thread.start_new_thread(mqtt_publish_task, (mqtt_handler, wlan))
    
    print('')
    print('╔═══════════════════════════════════════════════════════════╗')
    print('║                   🛜  Micro - ESPectre 👻                  ║')
    print('║                                                           ║')
    print('║                Wi-Fi motion detection system              ║')
    print('║          based on Channel State Information (CSI)         ║')
    print('║                                                           ║')
    print('╠═══════════════════════════════════════════════════════════╣')
    print('║                                                           ║')
    print('║             System Ready - Monitoring Active              ║')
    print('║               Detecting the invisible... 👁️                ║')
    print('║                                                           ║')
    print('╚═══════════════════════════════════════════════════════════╝')
    print('')
    
    try:
        while True:
            # Read CSI frame
            frame = wlan.csi.read()
            
            if frame:
                # Calculate turbulence and update segmentation
                turbulence = seg.calculate_spatial_turbulence(frame['data'], config.SELECTED_SUBCARRIERS)
                segment_completed = seg.add_turbulence(turbulence)
                
                # Get current metrics
                metrics = seg.get_metrics()
                
                # Update global state with lock
                with g_state.lock:
                    g_state.packets_processed += 1
                    g_state.current_state = metrics['state']
                    g_state.current_variance = metrics['moving_variance']
                    g_state.current_threshold = metrics['threshold']
            
            else:
                # No frame available, small delay
                time.sleep_ms(10)
    
    except KeyboardInterrupt:
        print('\n\nStopping...')
    
    finally:
        # Cleanup
        wlan.csi.disable()
        mqtt_handler.disconnect()
        
        # Stop traffic generator if running
        if traffic_gen.is_running():
            traffic_gen.stop()
        
        # Print statistics
        stats = mqtt_handler.get_stats()
        print('\n' + '='*60)
        print('Statistics:')
        print(f'  Packets processed: {g_state.packets_processed}')
        print(f'  Dropped packets: {wlan.csi.dropped()}')
        print(f'  MQTT published: {stats["published"]}, skipped: {stats["skipped"]}')
        if traffic_gen.get_packet_count() > 0:
            print(f'  Traffic generated: {traffic_gen.get_packet_count()} packets')
        print('='*60)


if __name__ == '__main__':
    main()
