"""
MVS Utilities - Common module for loading CSI data and MVS detection

Author: Francesco Pace <francesco.pace@gmail.com>
License: GPLv3
"""

import struct
import numpy as np
import math
from pathlib import Path

# Binary format (must match data_collector.py)
PACKET_FORMAT = '<IHbbBBBB128b'
PACKET_SIZE = struct.calcsize(PACKET_FORMAT)
MAGIC_NUMBER = 0x43534944  # "CSID"

# Fixed file paths
BASELINE_FILE = Path('baseline_data.bin')
MOVEMENT_FILE = Path('movement_data.bin')

def load_binary_data(filename):
    """
    Load CSI data from binary file
    
    Args:
        filename: Path to binary file
    
    Returns:
        list: Packets with CSI data and metadata
    """
    packets = []
    
    with open(filename, 'rb') as f:
        # Read header
        header = f.read(5)
        if len(header) < 5:
            raise ValueError("Invalid file: header too short")
        
        magic, label_byte = struct.unpack('<IB', header)
        if magic != MAGIC_NUMBER:
            raise ValueError(f"Invalid file: wrong magic number {magic:08x}")
        
        label = 'BASELINE' if label_byte == 0 else 'MOVEMENT'
        
        # Read packets
        while True:
            data = f.read(PACKET_SIZE)
            if len(data) < PACKET_SIZE:
                break
            
            values = struct.unpack(PACKET_FORMAT, data)
            packet = {
                'timestamp_ms': values[0],
                'packet_id': values[1],
                'rssi': values[2],
                'noise_floor': values[3],
                'snr': float(values[2] - values[3]),
                'channel': values[4],
                'mcs': values[5],
                'sig_mode': values[6],
                'cwb': values[7],
                'csi_data': np.array(values[8:], dtype=np.int8),
                'label': label
            }
            packets.append(packet)
    
    return packets

def load_baseline_and_movement():
    """
    Load both baseline and movement data files
    
    Returns:
        tuple: (baseline_packets, movement_packets)
    """
    if not BASELINE_FILE.exists():
        raise FileNotFoundError(f"{BASELINE_FILE} not found. Run: ./deploy.sh --collect-baseline")
    if not MOVEMENT_FILE.exists():
        raise FileNotFoundError(f"{MOVEMENT_FILE} not found. Run: ./deploy.sh --collect-movement")
    
    baseline_packets = load_binary_data(BASELINE_FILE)
    movement_packets = load_binary_data(MOVEMENT_FILE)
    
    return baseline_packets, movement_packets


# ============================================================================
# MVS Detection Functions
# ============================================================================

def calculate_variance_two_pass(values):
    """
    Calculate variance using two-pass algorithm (numerically stable)
    
    Two-pass algorithm: variance = sum((x - mean)^2) / n
    More stable than single-pass E[X²] - E[X]² for float32 arithmetic.
    
    This is the centralized variance function used by all MVS components.
    Matches the implementation in csi_processor.c and segmentation.py.
    
    Args:
        values: List or array of float values
    
    Returns:
        float: Variance (0.0 if empty)
    """
    n = len(values)
    if n == 0:
        return 0.0
    
    # First pass: calculate mean
    mean = sum(values) / n
    
    # Second pass: calculate variance
    variance = sum((x - mean) ** 2 for x in values) / n
    
    return variance


def calculate_spatial_turbulence(csi_data, selected_subcarriers):
    """
    Calculate spatial turbulence (standard deviation of amplitudes)
    
    Uses two-pass variance for numerical stability.
    This matches the implementation in segmentation.py and csi_processor.c.
    
    Args:
        csi_data: CSI data array (I/Q pairs)
        selected_subcarriers: List of subcarrier indices to use
    
    Returns:
        float: Spatial turbulence value (standard deviation of amplitudes)
    """
    # Calculate amplitudes for selected subcarriers
    amplitudes = []
    for sc_idx in selected_subcarriers:
        i = sc_idx * 2
        if i + 1 < len(csi_data):
            I = float(csi_data[i])
            Q = float(csi_data[i + 1])
            amplitudes.append(math.sqrt(I*I + Q*Q))
    
    if len(amplitudes) < 2:
        return 0.0
    
    # Use two-pass variance for numerical stability
    variance = calculate_variance_two_pass(amplitudes)
    
    return math.sqrt(variance)


class MVSDetector:
    """
    Streaming MVS (Moving Variance of Spatial turbulence) detector
    
    This implementation uses CORRECTED TP/FP counting:
    - TP (True Positives): Number of packets in MOTION state during movement data
    - FP (False Positives): Number of packets in MOTION state during baseline data
    
    This reflects the actual "green area" in visualization plots.
    """
    
    def __init__(self, window_size, threshold, selected_subcarriers, track_data=False):
        """
        Initialize MVS detector
        
        Args:
            window_size: Size of the sliding window for variance calculation
            threshold: Threshold for motion detection
            selected_subcarriers: List of subcarrier indices to use
            track_data: If True, track moving variance and state history for plotting
        """
        self.window_size = window_size
        self.threshold = threshold
        self.selected_subcarriers = selected_subcarriers
        self.track_data = track_data
        
        self.turbulence_buffer = []
        self.state = 'IDLE'
        self.motion_packet_count = 0  # Count packets in MOTION state
        
        if track_data:
            self.moving_var_history = []
            self.state_history = []
    
    def process_packet(self, csi_data):
        """
        Process a single CSI packet
        
        Args:
            csi_data: CSI data array (I/Q pairs)
        """
        turb = calculate_spatial_turbulence(csi_data, self.selected_subcarriers)
        self.turbulence_buffer.append(turb)
        
        if len(self.turbulence_buffer) > self.window_size:
            self.turbulence_buffer.pop(0)
        
        if len(self.turbulence_buffer) == self.window_size:
            # Use centralized two-pass variance calculation
            # This matches the implementation in segmentation.py and csi_processor.c
            moving_var = calculate_variance_two_pass(self.turbulence_buffer)
            
            if self.track_data:
                self.moving_var_history.append(moving_var)
                self.state_history.append(self.state)
            
            # State machine
            if self.state == 'IDLE':
                if moving_var > self.threshold:
                    self.state = 'MOTION'
            else:  # MOTION
                if moving_var < self.threshold:
                    self.state = 'IDLE'
            
            # Count packets in MOTION state (this is the "green area")
            if self.state == 'MOTION':
                self.motion_packet_count += 1
    
    def reset(self):
        """Reset detector state"""
        self.turbulence_buffer = []
        self.state = 'IDLE'
        self.motion_packet_count = 0
        if self.track_data:
            self.moving_var_history = []
            self.state_history = []
    
    def get_motion_count(self):
        """Get number of packets detected as motion"""
        return self.motion_packet_count


def test_mvs_configuration(baseline_packets, movement_packets, 
                          subcarriers, threshold, window_size):
    """
    Test MVS configuration and return FP, TP counts
    
    Args:
        baseline_packets: List of baseline packets
        movement_packets: List of movement packets
        subcarriers: List of subcarrier indices to use
        threshold: Motion detection threshold
        window_size: Sliding window size
    
    Returns:
        tuple: (fp, tp, score)
            - fp: False positives (motion packets in baseline)
            - tp: True positives (motion packets in movement)
            - score: Overall score (prioritizes FP=0)
    """
    # Test on baseline (FP = packets incorrectly classified as motion)
    detector = MVSDetector(window_size, threshold, subcarriers)
    for pkt in baseline_packets:
        detector.process_packet(pkt['csi_data'])
    fp = detector.get_motion_count()
    
    # Test on movement (TP = packets correctly classified as motion)
    detector.reset()
    for pkt in movement_packets:
        detector.process_packet(pkt['csi_data'])
    tp = detector.get_motion_count()
    
    # Calculate score (prioritize FP=0)
    if tp == 0:
        score = -1000  # Useless detector
    elif fp == 0:
        score = 1000 + tp  # Bonus for zero false positives
    else:
        score = tp - fp * 100  # Heavy penalty for any false positive
    
    return fp, tp, score
