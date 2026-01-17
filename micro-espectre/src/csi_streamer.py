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

HT20 only: 64 subcarriers, packet size = 6 + 128 = 134 bytes

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
from src.config import NUM_SUBCARRIERS, EXPECTED_CSI_LEN
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
    
    # Phase 1: Gain lock (stabilizes AGC/FFT)
    # Do this BEFORE creating streaming socket to avoid ENOMEM
    gc.collect()
    agc_gain, fft_gain, skipped = run_gain_lock(wlan)
    
    # Create UDP socket for streaming (after gain lock to reduce memory pressure)
    gc.collect()
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    dest_addr = (dest_ip, STREAM_PORT)
    
    # Packet format: <magic><chip><seq><num_sc_u16><payload>
    # Pre-allocate packet buffer to avoid memory allocation in loop
    header_size = 6  # magic(2) + chip(1) + seq(1) + num_sc(2)
    payload_size = EXPECTED_CSI_LEN  # 64 SC × 2 bytes
    packet_size = header_size + payload_size
    
    # Pre-allocate bytearray (reused every iteration)
    packet_buf = bytearray(packet_size)
    # Write static header fields (magic, chip, num_sc) - only seq changes
    packet_buf[0] = MAGIC_STREAM & 0xFF
    packet_buf[1] = (MAGIC_STREAM >> 8) & 0xFF
    packet_buf[2] = chip_code
    # packet_buf[3] = seq_num (updated in loop)
    packet_buf[4] = NUM_SUBCARRIERS & 0xFF
    packet_buf[5] = (NUM_SUBCARRIERS >> 8) & 0xFF
    
    print('')
    print(f'Streaming to: {dest_ip}:{STREAM_PORT}')
    print(f'Subcarriers:  {NUM_SUBCARRIERS} (HT20)')
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
                # Filter packets by expected CSI length (HT20: 128 bytes)
                if len(frame[5]) != EXPECTED_CSI_LEN:
                    filtered_count += 1
                    # Log warning every 100 filtered packets
                    if filtered_count % 100 == 1:
                        print(f'[WARN] Filtered {filtered_count} packets with wrong SC count (got {len(frame[5])} bytes)')
                    del frame
                    continue
                
                # Extract CSI data (HT20: 128 bytes)
                csi_data = frame[5][:EXPECTED_CSI_LEN]
                del frame
                
                # Build and send packet using pre-allocated buffer (zero allocation)
                try:
                    # Update seq_num in header
                    packet_buf[3] = seq_num
                    # Copy CSI data into payload section
                    packet_buf[header_size:] = csi_data
                    # Send packet
                    sock.sendto(packet_buf, dest_addr)
                    packet_count += 1
                    seq_num = (seq_num + 1) & 0xFF
                except Exception:
                    pass
                
                # GC every 50 packets to prevent ENOMEM
                if packet_count % 50 == 0:
                    gc.collect()
                
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
