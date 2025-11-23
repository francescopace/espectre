"""
Micro-ESPectre - Debug CSI Data
Analyze raw CSI values to understand scaling

Author: Francesco Pace <francesco.pace@gmail.com>
License: GPLv3
"""
import network
import time
import math

# Try to import credentials
try:
    from config_local import WIFI_SSID, WIFI_PASSWORD
except ImportError:
    WIFI_SSID = "YourSSID"
    WIFI_PASSWORD = "YourPassword"


def analyze_csi_frame(frame):
    """Analyze a single CSI frame"""
    csi_data = frame['data']
    
    # Extract I/Q values
    i_values = []
    q_values = []
    amplitudes = []
    
    for i in range(0, len(csi_data), 2):
        if i + 1 < len(csi_data):
            I = csi_data[i]
            Q = csi_data[i + 1]
            i_values.append(I)
            q_values.append(Q)
            
            amplitude = math.sqrt(I * I + Q * Q)
            amplitudes.append(amplitude)
    
    # Statistics
    i_min = min(i_values)
    i_max = max(i_values)
    i_mean = sum(i_values) / len(i_values)
    
    q_min = min(q_values)
    q_max = max(q_values)
    q_mean = sum(q_values) / len(q_values)
    
    amp_min = min(amplitudes)
    amp_max = max(amplitudes)
    amp_mean = sum(amplitudes) / len(amplitudes)
    amp_std = math.sqrt(sum((x - amp_mean) ** 2 for x in amplitudes) / len(amplitudes))
    
    return {
        'num_subcarriers': len(amplitudes),
        'i_range': (i_min, i_max),
        'i_mean': i_mean,
        'q_range': (q_min, q_max),
        'q_mean': q_mean,
        'amp_range': (amp_min, amp_max),
        'amp_mean': amp_mean,
        'amp_std': amp_std,
    }


def main():
    print('Micro-ESPectre - CSI Data Debug')
    print('='*60)
    
    # Connect WiFi
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
    
    # Connect to WiFi (REQUIRED for CSI)
    print("Connecting to WiFi...")
    wlan.connect(WIFI_SSID, WIFI_PASSWORD)
    
    # Wait for connection
    timeout = 10
    while not wlan.isconnected() and timeout > 0:
        time.sleep(0.5)
        timeout -= 0.5
    
    if not wlan.isconnected():
        print("ERROR: Failed to connect to WiFi!")
        print("CSI requires WiFi connection to work.")
        return
    
    print("WiFi connected to: " + WIFI_SSID)
    print()
    
    # Wait for WiFi to be fully ready (critical for ESP32-C6)
    print("Waiting for WiFi to stabilize...")
    time.sleep(2)
    
    # Configure and enable CSI
    print('Configuring CSI...')
    wlan.csi.config(buffer_size=64)
    wlan.csi.enable()
    print('CSI enabled\n')
    
    print('Analyzing 10 CSI frames...')
    print('='*60)
    
    frames_analyzed = 0
    
    try:
        while frames_analyzed < 10:
            frame = wlan.csi.read()
            
            if frame:
                frames_analyzed += 1
                stats = analyze_csi_frame(frame)
                
                print(f'\nFrame {frames_analyzed}:')
                print(f'  Subcarriers: {stats["num_subcarriers"]}')
                print(f'  I values: min={stats["i_range"][0]}, max={stats["i_range"][1]}, mean={stats["i_mean"]:.2f}')
                print(f'  Q values: min={stats["q_range"][0]}, max={stats["q_range"][1]}, mean={stats["q_mean"]:.2f}')
                print(f'  Amplitudes: min={stats["amp_range"][0]:.2f}, max={stats["amp_range"][1]:.2f}')
                print(f'  Amplitude mean: {stats["amp_mean"]:.2f}')
                print(f'  Amplitude std: {stats["amp_std"]:.2f}')
                print(f'  RSSI: {frame["rssi"]} dBm')
            else:
                time.sleep_ms(10)
    
    except KeyboardInterrupt:
        print('\n\nStopped by user')
    
    finally:
        wlan.csi.disable()
        print('\n' + '='*60)
        print('Analysis complete')
        print('='*60)


if __name__ == '__main__':
    main()
