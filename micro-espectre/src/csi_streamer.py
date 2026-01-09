"""
Micro-ESPectre - CSI UDP Streamer

Streams raw CSI I/Q data via UDP for real-time processing.

Packet format v2 (for 256 SC support):
  Header (6 bytes):
    - Magic: 0x4353 ("CS") - 2 bytes
    - Version: 1 byte (0x02 for v2)
    - Sequence number: 1 byte (0-255, wrapping)
    - Num subcarriers: 2 bytes (uint16, little-endian)
  Payload (N × 2 bytes):
    - I0, Q0, I1, Q1, ... (int8 each)

Packet format v1 (legacy, max 255 SC):
  Header (4 bytes):
    - Magic: 0x4353 ("CS") - 2 bytes
    - Num subcarriers: 1 byte
    - Sequence number: 1 byte (0-255, wrapping)
  Payload (N × 2 bytes):
    - I0, Q0, I1, Q1, ... (int8 each)

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
PROTOCOL_VERSION = 0x02  # v2 supports up to 65535 subcarriers


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
    print(f'Chip: {chip_type}')
    
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
    
    # Use v2 protocol format: <magic><version><seq><num_sc_u16><payload>
    packet_format = f'<HBBH{num_sc * 2}b'
    packet_size = struct.calcsize(packet_format)
    
    print('')
    print(f'Streaming to: {dest_ip}:{STREAM_PORT}')
    print(f'Protocol:     v2 (supports 256 SC)')
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
                
                # Build and send packet (v2 format)
                try:
                    packet = struct.pack(packet_format, MAGIC_STREAM, PROTOCOL_VERSION, seq_num, num_sc, *iq_values)
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
