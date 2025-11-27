"""
Micro-ESPectre - CSI Data Collector with Metadata
Collects CSI packets with full metadata (RSSI, noise_floor, SNR, etc.)

Usage:
    1. Deploy to ESP32: ampy --port /dev/ttyUSB0 put data_collector.py
    2. Run baseline collection: import data_collector; data_collector.collect_baseline()
    3. Run movement collection: import data_collector; data_collector.collect_movement()
    4. Download files: ampy --port /dev/ttyUSB0 get baseline_data.jsonl
                       ampy --port /dev/ttyUSB0 get movement_data.jsonl

Author: Francesco Pace <francesco.pace@gmail.com>
License: GPLv3
"""

import time
import struct
import gc
import src.config as config
from src.traffic_generator import TrafficGenerator
from src.main import connect_wifi

# Collection parameters
MAX_PACKETS = 1000

# Binary format structure (little-endian):
# Header: 4 bytes magic number (0x43534944 = "CSID")
# Per packet: 
#   - timestamp_ms: I (4 bytes)
#   - packet_id: H (2 bytes)
#   - rssi: b (1 byte signed)
#   - noise_floor: b (1 byte signed)
#   - channel: B (1 byte)
#   - mcs: B (1 byte)
#   - sig_mode: B (1 byte)
#   - cwb: B (1 byte)
#   - csi_data: 128b (128 bytes signed)
# Total per packet: 140 bytes (vs ~280 bytes JSON)
PACKET_FORMAT = '<IHbbBBBB128b'  # Little-endian
PACKET_SIZE = struct.calcsize(PACKET_FORMAT)
MAGIC_NUMBER = 0x43534944  # "CSID" in hex

def collect_csi_packets(wlan, label, max_packets=MAX_PACKETS):
    
    # Aggressive garbage collection before starting
    gc.collect()
    
    # Start traffic generator to ensure continuous CSI packets
    traffic_gen = TrafficGenerator()
    traffic_rate = config.TRAFFIC_GENERATOR_RATE
    traffic_gen_started = False
    
    if traffic_rate > 0:
        if traffic_gen.start(traffic_rate):
            traffic_gen_started = True

    print()
    print(f"Collection will start in...\n")
    for i in range(5, 0, -1):
        time.sleep(1)
        print(f"{i} seconds ...")
    print()
    
    wlan.csi_enable(buffer_size=config.CSI_BUFFER_SIZE)
    
    # Open binary file for buffered writing
    filename = f"{label.lower()}_data.bin"
    file_handle = open(filename, 'wb')
    
    # Write header: magic number + label
    label_byte = 0 if label == 'BASELINE' else 1
    file_handle.write(struct.pack('<IB', MAGIC_NUMBER, label_byte))
    
    start_time = time.ticks_ms()
    packet_count = 0
    last_progress = 0
    last_progress_time = start_time
    
    try:
        while packet_count < max_packets:
            frame = wlan.csi_read()
            if frame:
                # Extract values with defaults
                timestamp_ms = time.ticks_ms()
                rssi = frame.get('rssi', -100)
                noise_floor = frame.get('noise_floor', -95)
                channel = frame.get('channel', 0)
                mcs = frame.get('mcs', 0)
                sig_mode = frame.get('sig_mode', 0)
                cwb = frame.get('cwb', 0)
                csi_data = frame['data'][:128]
                
                # Pack and write directly
                packed_data = struct.pack(PACKET_FORMAT,
                    timestamp_ms, packet_count, rssi, noise_floor,
                    channel, mcs, sig_mode, cwb, *csi_data)
                file_handle.write(packed_data)
                
                packet_count += 1
                
                progress = (packet_count * 100) // max_packets
                if progress >= last_progress + 10:
                    current_time = time.ticks_ms()
                    block_time = time.ticks_diff(current_time, last_progress_time) / 1000
                    dropped = wlan.csi_dropped()
                    print(f"Progress: {progress}% ({packet_count}/{max_packets}) - {block_time:.1f}s for last 100 pkts | dropped: {dropped}")
                    last_progress = progress
                    last_progress_time = current_time
                
                # Garbage collection every 100 packets
                if packet_count % 100 == 0:
                    gc.collect()
            else:
                time.sleep_us(100)
    
    except KeyboardInterrupt:
        print(f"\n⏸️ Collection stopped by user at {packet_count} packets")
    finally:
        # Close file
        file_handle.close()
        
        wlan.csi_disable()
        # Stop traffic generator if it was started
        if traffic_gen_started:
            try:
                if traffic_gen.is_running():
                    traffic_gen.stop()
            except Exception as e:
                print(f"⚠️  Error stopping traffic generator: {e}")
    
    elapsed_total = time.ticks_diff(time.ticks_ms(), start_time) / 1000
    total_dropped = wlan.csi_dropped()
    print(f"\n✅ Collection complete: {packet_count} packets in {elapsed_total:.1f}s")
    print(f"   Rate: {packet_count/elapsed_total:.1f} packets/sec")
    print(f"   Dropped: {total_dropped} packets")
    print(f"   File: {filename}\n")
    
    return packet_count

def collect_baseline():
    """Collect BASELINE data (no movement)"""
    print("\n╔═══════════════════════════════════════════════════════╗")
    print("║          BASELINE DATA COLLECTION                     ║")
    print("╚═══════════════════════════════════════════════════════╝")
    wlan = connect_wifi()
    print("\n⚠️  IMPORTANT: Leave the room or stay completely still!")
    packet_count = collect_csi_packets(wlan, 'BASELINE', MAX_PACKETS)
    print(f"✅ Collected {packet_count} packets")
    print("📥 File ready: baseline_data.bin\n")
    return packet_count

def collect_movement():
    """Collect MOVEMENT data (continuous movement)"""
    print("\n╔═══════════════════════════════════════════════════════╗")
    print("║          MOVEMENT DATA COLLECTION                     ║")
    print("╚═══════════════════════════════════════════════════════╝")
    wlan = connect_wifi()
    print("\n⚠️  IMPORTANT: Move continuously during collection!")
    print("   (Walk around, wave arms, etc.)")
    packet_count = collect_csi_packets(wlan, 'MOVEMENT', MAX_PACKETS)
    print(f"✅ Collected {packet_count} packets")
    print("📥 File ready: movement_data.bin\n")
    return packet_count
