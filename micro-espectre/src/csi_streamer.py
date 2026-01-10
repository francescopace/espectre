"""
Micro-ESPectre - CSI UDP Streamer

Streams raw CSI I/Q data via UDP for real-time processing.

Packet format:
  Header (6 bytes):
    - Magic: 0x4353 ("CS") - 2 bytes
    - Chip type: 1 byte (0=unknown, 1=ESP32, 2=S2, 3=S3, 4=C3, 5=C5, 6=C6)
    - Sequence number: 1 byte (0-255, wrapping)
    - Num subcarriers: 2 bytes (uint16, little-endian)
  Payload (N × 2 bytes):
    - I0, Q0, I1, Q1, ... (int8 each)

Examples:
  - 64 SC:  6 + 128 = 134 bytes
  - 128 SC: 6 + 256 = 262 bytes
  - 256 SC: 6 + 512 = 518 bytes

Usage:
    ./me stream --ip 192.168.1.100

Author: Francesco Pace <francesco.pace@gmail.com>
License: GPLv3
"""
import socket
import struct
import time
import gc
import os
import src.config as config
from src.traffic_generator import TrafficGenerator
from src.main import connect_wifi, cleanup_wifi, run_gain_lock

# Streaming configuration
STREAM_PORT = 5001
MAGIC_STREAM = 0x4353  # "CS" in little-endian

# Chip type codes (must match receiver)
CHIP_UNKNOWN = 0
CHIP_ESP32 = 1
CHIP_S2 = 2
CHIP_S3 = 3
CHIP_C3 = 4
CHIP_C5 = 5
CHIP_C6 = 6


def detect_chip_code():
    """Detect chip type and return code for protocol"""
    machine = os.uname().machine.upper()
    if 'ESP32-C6' in machine or 'ESP32C6' in machine:
        return CHIP_C6
    elif 'ESP32-C5' in machine or 'ESP32C5' in machine:
        return CHIP_C5
    elif 'ESP32-C3' in machine or 'ESP32C3' in machine:
        return CHIP_C3
    elif 'ESP32-S3' in machine or 'ESP32S3' in machine:
        return CHIP_S3
    elif 'ESP32-S2' in machine or 'ESP32S2' in machine:
        return CHIP_S2
    elif 'ESP32' in machine:
        return CHIP_ESP32
    return CHIP_UNKNOWN


def stream_csi(dest_ip, duration_sec=0):
    """
    Stream raw CSI I/Q data via UDP.
    
    Args:
        dest_ip: Destination IP address
        duration_sec: Duration in seconds (0 = infinite)
    """
    duration_sec = int(duration_sec)
    
    print('')
    print('=' * 60)
    print('  CSI UDP Streamer')
    print('=' * 60)
    
    # Connect WiFi (also enables CSI)
    wlan = connect_wifi()
    chip_type = os.uname().machine
    chip_code = detect_chip_code()
    print(f'Chip: {chip_type} (code: {chip_code})')
    
    # Start traffic generator
    traffic_gen = TrafficGenerator()
    traffic_gen_started = False
    if config.TRAFFIC_GENERATOR_RATE > 0:
        if traffic_gen.start(config.TRAFFIC_GENERATOR_RATE):
            traffic_gen_started = True
            print(f'Traffic generator: {config.TRAFFIC_GENERATOR_RATE} pps')
        time.sleep(1)
    
    # Create UDP socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    dest_addr = (dest_ip, STREAM_PORT)
    
    # Phase 1: Gain lock (stabilizes AGC/FFT and determines dominant SC count)
    agc_gain, fft_gain, skipped, num_sc = run_gain_lock(wlan)
    
    # Packet format: <magic><chip><seq><num_sc_u16><payload>
    packet_format = f'<HBBH{num_sc * 2}b'
    packet_size = struct.calcsize(packet_format)
    
    print('')
    print(f'Streaming to: {dest_ip}:{STREAM_PORT}')
    print(f'Subcarriers:  {num_sc}')
    print(f'Packet size:  {packet_size} bytes')
    duration_str = "infinite" if duration_sec == 0 else str(duration_sec) + "s"
    print(f'Duration:     {duration_str}')
    print('')
    print('Press Ctrl+C to stop')
    print('=' * 60)
    print('')
    
    # Streaming loop
    start_time = time.ticks_ms()
    packet_count = 0
    filtered_count = 0
    seq_num = 0
    last_progress_time = start_time
    last_progress_count = 0
    
    try:
        while True:
            # Check duration
            if duration_sec > 0:
                elapsed = time.ticks_diff(time.ticks_ms(), start_time) / 1000
                if elapsed >= duration_sec:
                    break
            
            frame = wlan.csi_read()
            if frame:
                csi_data = frame[5]
                
                # Filter packets by expected subcarrier count
                packet_sc = len(csi_data) // 2
                if packet_sc != num_sc:
                    filtered_count += 1
                    continue
                
                # Extract I/Q values
                iq_values = [int(v) for v in csi_data]
                
                # Build and send packet
                try:
                    packet = struct.pack(packet_format, MAGIC_STREAM, chip_code, seq_num, num_sc, *iq_values)
                    sock.sendto(packet, dest_addr)
                    packet_count += 1
                    seq_num = (seq_num + 1) & 0xFF
                except Exception:
                    pass
                
                # Progress every 100 packets
                if packet_count % 100 == 0:
                    current_time = time.ticks_ms()
                    elapsed_block = time.ticks_diff(current_time, last_progress_time)
                    delta = packet_count - last_progress_count
                    pps = int((delta * 1000) / elapsed_block) if elapsed_block > 0 else 0
                    
                    filter_str = f' | filtered: {filtered_count}' if filtered_count > 0 else ''
                    print(f'Sent {packet_count} pkts | {pps} pps | seq: {seq_num}{filter_str}')
                    
                    last_progress_time = current_time
                    last_progress_count = packet_count
                    gc.collect()
            else:
                time.sleep_us(100)
    
    except KeyboardInterrupt:
        print('\n\nStreaming stopped by user')
    
    finally:
        print('Cleaning up...')
        sock.close()
        if traffic_gen_started and traffic_gen.is_running():
            traffic_gen.stop()
        cleanup_wifi(wlan)
    
    elapsed = time.ticks_diff(time.ticks_ms(), start_time) / 1000
    avg_pps = packet_count / elapsed if elapsed > 0 else 0
    print(f'\nTotal: {packet_count} packets in {elapsed:.1f}s ({avg_pps:.1f} pps avg)')
