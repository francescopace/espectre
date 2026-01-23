"""
CSI Utilities - Common module for CSI data handling

Provides:
  - UDP reception (CSIReceiver)
  - Data collection (CSICollector)
  - Dataset management (load, save, stats)
  - MVS detection (MVSDetector)

Author: Francesco Pace <francesco.pace@gmail.com>
License: GPLv3
"""

import socket
import struct
import subprocess
import time
import json
import numpy as np
import math
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, List, Optional, Dict, Any, Tuple

# ============================================================================
# Constants
# ============================================================================

# Default subcarrier band for analysis (shared across all tools)
DEFAULT_SUBCARRIERS = [11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22]

# UDP Protocol constants
MAGIC_STREAM = 0x4353  # "CS" in little-endian
DEFAULT_PORT = 5001

# Dataset paths (shared between tools and tests)
DATA_DIR = Path(__file__).parent.parent / 'data'
DATASET_INFO_FILE = DATA_DIR / 'dataset_info.json'



# ============================================================================
# Data Structures
# ============================================================================

@dataclass
class CSIPacket:
    """Represents a single CSI packet received via UDP"""
    timestamp: float          # Reception timestamp (seconds since epoch)
    seq_num: int             # Sequence number (0-255)
    num_subcarriers: int     # Number of subcarriers
    iq_raw: np.ndarray       # Raw I/Q values as int8 array [I0,Q0,I1,Q1,...]
    iq_complex: np.ndarray   # Complex representation [I0+jQ0, I1+jQ1, ...]
    amplitudes: np.ndarray   # Amplitude per subcarrier
    phases: np.ndarray       # Phase per subcarrier (radians)
    chip: str = 'unknown'    # Chip type (e.g., 'C6', 'S3', 'ESP32')


# Chip code to name mapping (must match streamer)
CHIP_CODES = {
    0: 'unknown',
    1: 'ESP32',
    2: 'S2',
    3: 'S3',
    4: 'C3',
    5: 'C5',
    6: 'C6',
}


# ============================================================================
# UDP Reception
# ============================================================================

class CSIReceiver:
    """
    UDP receiver for CSI data with callback support.
    
    Provides a foundation for building CSI processing pipelines.
    
    Usage:
        receiver = CSIReceiver(port=5001)
        receiver.add_callback(my_callback)
        receiver.run()
    """
    
    def __init__(self, port: int = DEFAULT_PORT, buffer_size: int = 500):
        """
        Initialize CSI receiver.
        
        Args:
            port: UDP port to listen on
            buffer_size: Circular buffer size (packets)
        """
        self.port = port
        self.buffer_size = buffer_size
        
        # Packet buffer (circular)
        self.buffer: deque[CSIPacket] = deque(maxlen=buffer_size)
        
        # Statistics
        self.packet_count = 0
        self.dropped_count = 0
        self.last_seq = -1
        self.start_time = 0.0
        self.pps = 0
        self._pps_counter = 0
        self._last_pps_time = 0.0
        
        # Callbacks
        self._callbacks: List[Callable[[CSIPacket], None]] = []
        self._buffer_callbacks: List[Tuple[Callable[[deque], None], int]] = []
        
        # Socket
        self.sock: Optional[socket.socket] = None
        self.running = False
    
    def add_callback(self, callback: Callable[[CSIPacket], None]):
        """
        Add callback for each received packet.
        
        Args:
            callback: Function that receives CSIPacket
        """
        self._callbacks.append(callback)
    
    def add_buffer_callback(self, callback: Callable[[deque], None], interval: int = 10):
        """
        Add callback that receives the full buffer periodically.
        
        Args:
            callback: Function that receives the packet buffer
            interval: Call every N packets
        """
        self._buffer_callbacks.append((callback, interval))
    
    def _parse_packet(self, data: bytes) -> Optional[CSIPacket]:
        """Parse raw UDP data into CSIPacket
        
        Packet format (6 byte header):
            <magic:2><chip:1><seq:1><num_sc:2><payload>
        """
        if len(data) < 6:
            return None
        
        # Parse header
        magic, chip_code, seq_num, num_sc = struct.unpack('<HBBH', data[:6])
        
        if magic != MAGIC_STREAM:
            return None
        
        chip = CHIP_CODES.get(chip_code, 'unknown')
        header_size = 6
        
        # Parse I/Q data
        iq_size = num_sc * 2
        if len(data) < header_size + iq_size:
            return None
        
        iq_raw = np.array(
            struct.unpack(f'<{iq_size}b', data[header_size:header_size+iq_size]),
            dtype=np.int8
        )
        
        # Espressif CSI format: [Imaginary, Real, ...] per subcarrier
        Q = iq_raw[0::2].astype(np.float32)  # Imaginary first (even indices)
        I = iq_raw[1::2].astype(np.float32)  # Real second (odd indices)
        iq_complex = I + 1j * Q
        
        # Calculate amplitude and phase
        amplitudes = np.abs(iq_complex)
        phases = np.angle(iq_complex)
        
        return CSIPacket(
            timestamp=time.time(),
            seq_num=seq_num,
            num_subcarriers=num_sc,
            iq_raw=iq_raw,
            iq_complex=iq_complex,
            amplitudes=amplitudes,
            phases=phases,
            chip=chip
        )
    
    def _check_sequence(self, seq_num: int):
        """Track sequence numbers and detect drops"""
        if self.last_seq >= 0:
            expected = (self.last_seq + 1) & 0xFF
            if seq_num != expected:
                # Calculate dropped packets (handling wrap-around)
                if seq_num > expected:
                    dropped = seq_num - expected
                else:
                    dropped = (256 - expected) + seq_num
                self.dropped_count += dropped
        self.last_seq = seq_num
    
    def _update_pps(self):
        """Update packets per second calculation"""
        current_time = time.time()
        if current_time - self._last_pps_time >= 1.0:
            self.pps = self._pps_counter
            self._pps_counter = 0
            self._last_pps_time = current_time
    
    def get_buffer_array(self) -> np.ndarray:
        """
        Get buffer as numpy array for batch processing.
        
        Returns:
            Array of shape (num_packets, num_subcarriers) with complex values
        """
        if not self.buffer:
            return np.array([])
        
        return np.array([p.iq_complex for p in self.buffer])
    
    def get_amplitude_matrix(self) -> np.ndarray:
        """
        Get amplitude matrix for analysis.
        
        Returns:
            Array of shape (num_packets, num_subcarriers) with amplitudes
        """
        if not self.buffer:
            return np.array([])
        
        return np.array([p.amplitudes for p in self.buffer])
    
    def get_phase_matrix(self) -> np.ndarray:
        """
        Get phase matrix for analysis.
        
        Returns:
            Array of shape (num_packets, num_subcarriers) with phases
        """
        if not self.buffer:
            return np.array([])
        
        return np.array([p.phases for p in self.buffer])
    
    def get_stats(self) -> Dict[str, Any]:
        """Get current statistics"""
        elapsed = time.time() - self.start_time if self.start_time else 0
        return {
            'packets': self.packet_count,
            'dropped': self.dropped_count,
            'drop_rate': self.dropped_count / max(self.packet_count, 1) * 100,
            'pps': self.pps,
            'buffer_fill': len(self.buffer),
            'buffer_size': self.buffer_size,
            'elapsed': elapsed
        }
    
    def reset_stats(self):
        """Reset statistics for new collection"""
        self.packet_count = 0
        self.dropped_count = 0
        self.last_seq = -1
        self.start_time = time.time()
        self.pps = 0
        self._pps_counter = 0
        self._last_pps_time = time.time()
        self.buffer.clear()
    
    def run(self, timeout: float = 0, quiet: bool = False):
        """
        Start receiving packets (blocking).
        
        Args:
            timeout: Stop after N seconds (0 = infinite)
            quiet: Suppress output messages
        """
        # Create socket
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind(('0.0.0.0', self.port))
        self.sock.settimeout(1.0)  # 1 second timeout for graceful shutdown
        
        if not quiet:
            print(f'CSI Receiver listening on UDP port {self.port}')
            print(f'Buffer size: {self.buffer_size} packets')
            print('Waiting for data...')
            print()
        
        self.running = True
        self.start_time = time.time()
        self._last_pps_time = time.time()
        
        try:
            while self.running:
                # Check timeout
                if timeout > 0:
                    if time.time() - self.start_time >= timeout:
                        break
                
                try:
                    data, addr = self.sock.recvfrom(1024)  # 134 bytes for 64 SC (HT20)
                except socket.timeout:
                    self._update_pps()
                    continue
                
                # Parse packet
                packet = self._parse_packet(data)
                if packet is None:
                    continue
                
                # Track sequence
                self._check_sequence(packet.seq_num)
                
                # Add to buffer
                self.buffer.append(packet)
                self.packet_count += 1
                self._pps_counter += 1
                
                # Update PPS
                self._update_pps()
                
                # Call packet callbacks
                for callback in self._callbacks:
                    try:
                        callback(packet)
                    except Exception as e:
                        print(f'Callback error: {e}')
                
                # Call buffer callbacks
                for callback, interval in self._buffer_callbacks:
                    if self.packet_count % interval == 0:
                        try:
                            callback(self.buffer)
                        except Exception as e:
                            print(f'Buffer callback error: {e}')
        
        except KeyboardInterrupt:
            if not quiet:
                print('\nStopping receiver...')
        
        finally:
            self.running = False
            if self.sock:
                self.sock.close()
        
        # Print final stats
        if not quiet:
            stats = self.get_stats()
            print()
            print('=' * 50)
            print(f'Total packets:  {stats["packets"]}')
            print(f'Dropped:        {stats["dropped"]} ({stats["drop_rate"]:.1f}%)')
            print(f'Duration:       {stats["elapsed"]:.1f}s')
            print(f'Average PPS:    {stats["packets"] / max(stats["elapsed"], 1):.1f}')
            print('=' * 50)
    
    def stop(self):
        """Stop the receiver"""
        self.running = False


# ============================================================================
# Data Collection
# ============================================================================

def get_git_username() -> Optional[str]:
    """Get GitHub username from git config (user.name or user.email prefix)"""
    try:
        result = subprocess.run(
            ['git', 'config', 'user.name'],
            capture_output=True, text=True, timeout=2
        )
        if result.returncode == 0 and result.stdout.strip():
            # Convert "Francesco Pace" -> "francescopace" (lowercase, no spaces)
            return result.stdout.strip().lower().replace(' ', '')
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass
    return None


class CSICollector:
    """
    Collects labeled CSI data for training datasets.
    
    Supports both interactive (keyboard-triggered) and timed collection modes.
    
    Usage:
        collector = CSICollector(label='wave')
        collector.collect_timed(duration=3.0, num_samples=10)
    """
    
    # File format version - increment when format changes
    FORMAT_VERSION = '1.0'
    
    def __init__(self, label: str, port: int = DEFAULT_PORT, contributor: str = None):
        """
        Initialize collector.
        
        Args:
            label: Label for collected samples (e.g., 'wave', 'baseline')
            port: UDP port for CSI receiver
            contributor: GitHub username of the contributor (auto-detected from git if not provided)
        """
        self.label = label
        self.port = port
        self.chip = None  # Auto-detected from CSI packets
        self.contributor = contributor or get_git_username()
        
        self.receiver = CSIReceiver(port=port, buffer_size=2000)
        self._recording = False
        self._recorded_packets: List[CSIPacket] = []
        self._sample_count = 0
    
    def _get_label_dir(self) -> Path:
        """Get directory for this label, create if needed"""
        label_dir = DATA_DIR / self.label
        label_dir.mkdir(parents=True, exist_ok=True)
        return label_dir
    
    def _generate_filename(self, num_subcarriers: int) -> str:
        """Generate filename with format: {label}_{chip}_{num_sc}sc_{timestamp}.npz"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        chip = self.chip or 'unknown'
        return f'{self.label}_{chip}_{num_subcarriers}sc_{timestamp}.npz'
    
    def save_sample(self, packets: List[CSIPacket]) -> Optional[Path]:
        """
        Save collected packets as a sample.
        
        Args:
            packets: List of CSIPacket objects
        
        Returns:
            Path to saved file, or None if no packets
        
        File format (unified, compact):
            csi_data: int8[N, num_sc*2] - Raw I/Q values
            num_subcarriers: int - Number of subcarriers (64 for HT20)
            label: str - Label name
            label_id: int - Numeric label ID
            chip: str - Chip type (C6, S3, etc.)
            collected_at: str - ISO timestamp
            duration_ms: float - Total duration
            format_version: str - Format version
        """
        if not packets:
            return None
        
        # Auto-detect chip from first packet (v2 protocol)
        if packets[0].chip and packets[0].chip != 'unknown':
            self.chip = packets[0].chip.lower()
        
        # Extract I/Q raw data (compact int8 format)
        csi_data = np.array([p.iq_raw for p in packets], dtype=np.int8)
        
        # Calculate duration from timestamps
        timestamps = np.array([p.timestamp for p in packets])
        duration_ms = (timestamps[-1] - timestamps[0]) * 1000 if len(timestamps) > 1 else 0
        
        # Get label ID from dataset info
        label_id = self._get_or_create_label_id()
        
        # Build sample dict (unified format)
        sample = {
            # CSI data (essential)
            'csi_data': csi_data,
            'num_subcarriers': packets[0].num_subcarriers,
            
            # Label (ground truth)
            'label': self.label,
            'label_id': label_id,
            
            # Context
            'chip': self.chip or 'unknown',
            
            # Metadata
            'collected_at': datetime.now().isoformat(),
            'duration_ms': duration_ms,
            'format_version': self.FORMAT_VERSION,
        }
        
        # Save file
        label_dir = self._get_label_dir()
        num_subcarriers = packets[0].num_subcarriers
        filename = self._generate_filename(num_subcarriers)
        filepath = label_dir / filename
        
        np.savez_compressed(filepath, **sample)
        
        # Update dataset info with file details
        self._update_dataset_info(
            filename=filename,
            num_subcarriers=num_subcarriers,
            num_packets=len(packets),
            duration_ms=duration_ms,
            collected_at=sample['collected_at']
        )
        
        return filepath
    
    def _get_or_create_label_id(self) -> int:
        """Get or create label ID from dataset info"""
        info = load_dataset_info()
        
        if self.label in info['labels']:
            return info['labels'][self.label]['id']
        
        # Create new label ID (max existing + 1)
        existing_ids = [l['id'] for l in info['labels'].values()]
        new_id = max(existing_ids) + 1 if existing_ids else 0
        
        info['labels'][self.label] = {
            'id': new_id,
            'samples': 0,
            'description': ''
        }
        save_dataset_info(info)
        
        return new_id
    
    def _update_dataset_info(self, filename: str = None, num_subcarriers: int = None,
                                num_packets: int = None, duration_ms: float = None,
                                collected_at: str = None):
        """Update dataset info with current sample counts and file details"""
        info = load_dataset_info()
        
        # Count samples for this label
        label_dir = self._get_label_dir()
        sample_count = len(list(label_dir.glob('*.npz')))
        
        if self.label not in info['labels']:
            info['labels'][self.label] = {
                'id': len(info['labels']),
                'description': ''
            }
        
        info['updated_at'] = datetime.now().isoformat()
        
        # Track file details if provided
        if filename and num_subcarriers:
            info.setdefault('files', {})
            info['files'].setdefault(self.label, [])
            
            # Check if file already exists in list
            existing_files = [f['filename'] for f in info['files'][self.label]]
            if filename not in existing_files:
                file_info = {
                    'filename': filename,
                    'chip': self.chip.upper() if self.chip else 'unknown',
                    'subcarriers': num_subcarriers,
                    'contributor': self.contributor or '',
                    'collected_at': collected_at or '',
                    'duration_ms': int(duration_ms) if duration_ms else 0,
                    'num_packets': num_packets or 0,
                    'description': ''
                }
                info['files'][self.label].append(file_info)
        
        save_dataset_info(info)
    
    def collect_timed(self, duration: float, num_samples: int = 1,
                     countdown: int = 3, quiet: bool = False) -> List[Path]:
        """
        Collect samples with fixed duration.
        
        Args:
            duration: Duration per sample in seconds
            num_samples: Number of samples to collect
            countdown: Countdown seconds before each sample
            quiet: Suppress output
        
        Returns:
            List of paths to saved samples
        """
        saved_files = []
        
        if not quiet:
            print(f'\n{"=" * 60}')
            print(f'  CSI Data Collection: {self.label}')
            print(f'{"=" * 60}')
            print(f'  Duration per sample: {duration}s')
            print(f'  Samples to collect:  {num_samples}')
            print(f'{"=" * 60}\n')
        
        # Create socket once
        self.receiver.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.receiver.sock.bind(('0.0.0.0', self.port))
        self.receiver.sock.settimeout(0.1)
        
        try:
            for sample_idx in range(num_samples):
                if not quiet:
                    print(f'\nSample {sample_idx + 1}/{num_samples}')
                    
                    # Countdown
                    for i in range(countdown, 0, -1):
                        print(f'  Starting in {i}...', end='\r')
                        time.sleep(1)
                    
                    print(f'  ▶ RECORDING...', end='', flush=True)
                
                # Reset and collect
                self.receiver.reset_stats()
                packets = []
                start_time = time.time()
                
                while time.time() - start_time < duration:
                    try:
                        data, addr = self.receiver.sock.recvfrom(1024)  # 134 bytes for 64 SC (HT20)
                        packet = self.receiver._parse_packet(data)
                        if packet:
                            packets.append(packet)
                            self.receiver._check_sequence(packet.seq_num)
                    except socket.timeout:
                        continue
                
                # Save sample
                filepath = self.save_sample(packets)
                
                if filepath:
                    saved_files.append(filepath)
                    if not quiet:
                        print(f'\r  ✅ Saved: {filepath.name} ({len(packets)} packets)')
                else:
                    if not quiet:
                        print(f'\r  ❌ No packets received!')
        
        finally:
            if self.receiver.sock:
                self.receiver.sock.close()
        
        if not quiet:
            print(f'\n{"=" * 60}')
            print(f'  Collection complete: {len(saved_files)}/{num_samples} samples saved')
            print(f'{"=" * 60}\n')
        
        return saved_files
    
    def collect_interactive(self, num_samples: int = 10) -> List[Path]:
        """
        Collect samples with keyboard control.
        
        Press SPACE to start/stop recording, ENTER to save, R to retry, Q to quit.
        
        Args:
            num_samples: Target number of samples
        
        Returns:
            List of paths to saved samples
        """
        # This requires terminal input handling
        # For simplicity, use timed collection with prompts
        saved_files = []
        
        print(f'\n{"=" * 60}')
        print(f'  CSI Data Collection: {self.label}')
        print(f'{"=" * 60}')
        print(f'  Target samples: {num_samples}')
        print(f'  Press ENTER to record each sample, Q to quit')
        print(f'{"=" * 60}\n')
        
        # Create socket once
        self.receiver.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.receiver.sock.bind(('0.0.0.0', self.port))
        self.receiver.sock.settimeout(0.1)
        
        try:
            sample_idx = 0
            while sample_idx < num_samples:
                try:
                    user_input = input(f'\nSample {sample_idx + 1}/{num_samples} - Press ENTER to record (Q to quit): ')
                    
                    if user_input.lower() == 'q':
                        print('Collection cancelled.')
                        break
                    
                    print('  Recording for 2 seconds...', end='', flush=True)
                    
                    # Collect for 2 seconds
                    self.receiver.reset_stats()
                    packets = []
                    start_time = time.time()
                    
                    while time.time() - start_time < 2.0:
                        try:
                            data, addr = self.receiver.sock.recvfrom(1024)  # 134 bytes for 64 SC (HT20)
                            packet = self.receiver._parse_packet(data)
                            if packet:
                                packets.append(packet)
                        except socket.timeout:
                            continue
                    
                    # Save sample
                    filepath = self.save_sample(packets)
                    
                    if filepath:
                        saved_files.append(filepath)
                        print(f'\r  ✅ Saved: {filepath.name} ({len(packets)} packets)')
                        sample_idx += 1
                    else:
                        print(f'\r  ❌ No packets received! Check ESP32 streaming.')
                        
                except KeyboardInterrupt:
                    print('\nCollection cancelled.')
                    break
        
        finally:
            if self.receiver.sock:
                self.receiver.sock.close()
        
        print(f'\n{"=" * 60}')
        print(f'  Collection complete: {len(saved_files)} samples saved')
        print(f'{"=" * 60}\n')
        
        return saved_files


# ============================================================================
# Dataset Management
# ============================================================================

def load_dataset_info() -> Dict[str, Any]:
    """Load or create dataset info"""
    if DATASET_INFO_FILE.exists():
        with open(DATASET_INFO_FILE, 'r') as f:
            return json.load(f)
    
    # Create default info
    return {
        'format_version': CSICollector.FORMAT_VERSION,
        'created_at': datetime.now().isoformat(),
        'updated_at': datetime.now().isoformat(),
        'labels': {},
        'files': {},
        'contributors': [],
        'environments': []
    }


def save_dataset_info(info: Dict[str, Any]):
    """Save dataset info"""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(DATASET_INFO_FILE, 'w') as f:
        json.dump(info, f, indent=2)


def get_dataset_stats() -> Dict[str, Any]:
    """Get dataset statistics by scanning directories"""
    info = load_dataset_info()
    stats = {
        'labels': {},
        'total_samples': 0,
        'total_packets': 0,
        'labels_count': 0
    }
    
    if not DATA_DIR.exists():
        return stats
    
    # Scan label directories
    for label_dir in DATA_DIR.iterdir():
        if label_dir.is_dir() and not label_dir.name.startswith('.'):
            label = label_dir.name
            samples = list(label_dir.glob('*.npz'))
            
            if samples:
                # Count packets in first sample to get average
                try:
                    sample = np.load(samples[0])
                    avg_packets = sample['num_packets']
                except:
                    avg_packets = 0
                
                stats['labels'][label] = {
                    'samples': len(samples),
                    'id': info.get('labels', {}).get(label, {}).get('id', -1)
                }
                stats['total_samples'] += len(samples)
                stats['labels_count'] += 1
    
    return stats


def load_samples(label: str = None) -> List[Dict[str, Any]]:
    """
    Load samples from dataset.
    
    Args:
        label: Label to load (None = all labels)
    
    Returns:
        List of sample dicts with numpy arrays
    """
    samples = []
    
    if label:
        label_dirs = [DATA_DIR / label]
    else:
        label_dirs = [d for d in DATA_DIR.iterdir() if d.is_dir() and not d.name.startswith('.')]
    
    for label_dir in label_dirs:
        if not label_dir.exists():
            continue
        
        for sample_file in label_dir.glob('*.npz'):
            try:
                data = np.load(sample_file, allow_pickle=True)
                sample = {key: data[key] for key in data.files}
                # Convert numpy strings to Python strings
                for key in ['label', 'subject', 'environment', 'notes', 'collected_at', 'format_version']:
                    if key in sample:
                        sample[key] = str(sample[key])
                samples.append(sample)
            except Exception as e:
                print(f'Error loading {sample_file}: {e}')
    
    return samples


def load_samples_as_arrays(labels: List[str] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load samples as X, y arrays for ML training.
    
    Args:
        labels: List of labels to load (None = all)
    
    Returns:
        Tuple (X, y) where X is array of amplitudes, y is array of label IDs
    """
    all_samples = load_samples()
    
    if labels:
        all_samples = [s for s in all_samples if s['label'] in labels]
    
    if not all_samples:
        return np.array([]), np.array([])
    
    X = [s['amplitudes'] for s in all_samples]
    y = [s['label_id'] for s in all_samples]
    
    return X, np.array(y)


# ============================================================================
# Data Loading Functions
# ============================================================================

def load_npz_as_packets(filepath: Path) -> List[Dict[str, Any]]:
    """
    Load .npz file and convert to list of packet dicts.
    
    Supports:
    - Unified format: csi_data (int8), num_subcarriers, label, chip, etc.
    - Legacy format with iq_raw: converts to csi_data
    
    Args:
        filepath: Path to .npz file
    
    Returns:
        list: Packets with CSI data and metadata
    """
    data = np.load(filepath, allow_pickle=True)
    
    # Get CSI data (unified format uses 'csi_data', legacy may use 'iq_raw')
    if 'csi_data' in data.files:
        csi_array = data['csi_data']
    elif 'iq_raw' in data.files:
        csi_array = data['iq_raw']
    else:
        raise ValueError(f"No CSI data found in {filepath}")
    
    # Get metadata
    label = str(data.get('label', 'unknown'))
    num_subcarriers = int(data.get('num_subcarriers', csi_array.shape[1] // 2))
    chip = str(data.get('chip', 'unknown'))
    
    # Build packet list
    packets = []
    for i in range(len(csi_array)):
        packets.append({
            'csi_data': np.array(csi_array[i], dtype=np.int8),
            'label': label,
            'num_subcarriers': num_subcarriers,
            'chip': chip
        })
    
    return packets


def find_dataset(chip: str = None, num_sc: int = 64) -> Tuple[Path, Path, str]:
    """
    Find the most recent baseline and movement dataset files.
    
    Args:
        chip: Chip type (C6, S3, etc.) or None to find any chip
        num_sc: Number of subcarriers (default: 64 for HT20)
    
    Returns:
        tuple: (baseline_path, movement_path, chip_name)
    
    Raises:
        FileNotFoundError: If no matching files found
    """
    baseline_dir = DATA_DIR / 'baseline'
    movement_dir = DATA_DIR / 'movement'
    
    # Build search pattern
    if chip:
        chip_lower = chip.lower()
        baseline_pattern = f'baseline_{chip_lower}_{num_sc}sc_*.npz'
        movement_pattern = f'movement_{chip_lower}_{num_sc}sc_*.npz'
    else:
        baseline_pattern = f'*_{num_sc}sc_*.npz'
        movement_pattern = f'*_{num_sc}sc_*.npz'
    
    baseline_files = list(baseline_dir.glob(baseline_pattern))
    movement_files = list(movement_dir.glob(movement_pattern))
    
    chip_desc = f"{chip} ({num_sc} SC)" if chip else f"{num_sc} SC"
    
    if not baseline_files:
        raise FileNotFoundError(
            f"No baseline file found for {chip_desc} in {baseline_dir}\n"
            f"Collect data using: ./me collect --label baseline --duration 10"
        )
    if not movement_files:
        raise FileNotFoundError(
            f"No movement file found for {chip_desc} in {movement_dir}\n"
            f"Collect data using: ./me collect --label movement --duration 10"
        )
    
    # Use the most recent file (sorted by name, which includes timestamp)
    baseline_file = sorted(baseline_files)[-1]
    movement_file = sorted(movement_files)[-1]
    
    # Extract chip name from filename (e.g., baseline_c6_64sc_... -> C6)
    chip_name = baseline_file.stem.split('_')[1].upper()
    
    return baseline_file, movement_file, chip_name


def load_baseline_and_movement(
    baseline_file: str = None,
    movement_file: str = None,
    chip: str = 'C6'
) -> Tuple[List[Dict], List[Dict]]:
    """
    Load baseline and movement data from .npz files
    
    Args:
        baseline_file: Path to baseline data file (optional, auto-finds if not specified)
        movement_file: Path to movement data file (optional, auto-finds if not specified)
        chip: Chip type for auto-discovery (default: C6)
    
    Returns:
        tuple: (baseline_packets, movement_packets)
    """
    # Auto-find files if not specified
    if baseline_file is None or movement_file is None:
        found_baseline, found_movement, _ = find_dataset(chip=chip)
        if baseline_file is None:
            baseline_file = found_baseline
        if movement_file is None:
            movement_file = found_movement
    
    # Convert to Path if string
    baseline_path = Path(baseline_file) if isinstance(baseline_file, str) else baseline_file
    movement_path = Path(movement_file) if isinstance(movement_file, str) else movement_file
    
    if not baseline_path.exists():
        raise FileNotFoundError(
            f"{baseline_path} not found.\n"
            f"Collect data using: ./me collect --label baseline --duration 10"
        )
    if not movement_path.exists():
        raise FileNotFoundError(
            f"{movement_path} not found.\n"
            f"Collect data using: ./me collect --label movement --duration 10"
        )
    
    baseline_packets = load_npz_as_packets(baseline_path)
    movement_packets = load_npz_as_packets(movement_path)
    
    return baseline_packets, movement_packets


# ============================================================================
# MVS Detection - Uses src/segmentation.py (single source of truth)
# ============================================================================

# Add src directory to path for imports
import sys as _sys
_src_path = str(Path(__file__).parent.parent / 'src')
if _src_path not in _sys.path:
    _sys.path.append(_src_path)

# Add micro-espectre directory to path so 'from src.config import' works in band_calibrator
_micro_espectre_path = str(Path(__file__).parent.parent)
if _micro_espectre_path not in _sys.path:
    _sys.path.insert(0, _micro_espectre_path)

# Import SegmentationContext from src/segmentation.py
from segmentation import SegmentationContext

# Import filters from src/filters.py (for scripts that need it directly)
from filters import HampelFilter

# Import feature calculation functions from src/features.py
from features import calc_skewness, calc_kurtosis

# Import calibrators from src (band selection algorithms)
from p95_calibrator import P95Calibrator, BandCalibrator  # BandCalibrator is alias for backward compat
from nbvi_calibrator import NBVICalibrator

# Import detectors from src (IDetector interface and implementations)
from detector_interface import IDetector, MotionState
from mvs_detector import MVSDetector as MVSDetectorNew
from pca_detector import PCADetector


# ============================================================================
# Utility Functions (delegate to SegmentationContext static methods)
# ============================================================================

def calculate_spatial_turbulence(csi_data, selected_subcarriers) -> float:
    """
    Calculate spatial turbulence (standard deviation of amplitudes)
    
    Delegates to SegmentationContext.compute_spatial_turbulence (static method).
    
    Args:
        csi_data: CSI data array (I/Q pairs)
        selected_subcarriers: List of subcarrier indices to use
    
    Returns:
        float: Spatial turbulence value (standard deviation of amplitudes)
    """
    turbulence, _ = SegmentationContext.compute_spatial_turbulence(csi_data, selected_subcarriers)
    return turbulence


def calculate_variance_two_pass(values) -> float:
    """
    Calculate variance using two-pass algorithm (numerically stable)
    
    Delegates to SegmentationContext.compute_variance_two_pass (static method).
    
    Args:
        values: List or array of float values
    
    Returns:
        float: Variance (0.0 if empty)
    """
    return SegmentationContext.compute_variance_two_pass(values)


class MVSDetector:
    """
    Streaming MVS (Moving Variance of Spatial turbulence) detector
    
    Wrapper around SegmentationContext for backward compatibility with analysis scripts.
    Provides the same interface as the original MVSDetector while using the
    production implementation from src/segmentation.py.
    """
    
    def __init__(self, window_size: int, threshold: float, 
                 selected_subcarriers: List[int], track_data: bool = False,
                 enable_hampel: bool = False, hampel_window: int = 7,
                 hampel_threshold: float = 4.0,
                 enable_lowpass: bool = False, lowpass_cutoff: float = 11.0):
        """
        Initialize MVS detector
        
        Args:
            window_size: Size of the sliding window for variance calculation
            threshold: Threshold for motion detection
            selected_subcarriers: List of subcarrier indices to use
            track_data: If True, track moving variance and state history
            enable_hampel: Enable Hampel filter for outlier removal
            hampel_window: Hampel filter window size
            hampel_threshold: Hampel filter MAD threshold
            enable_lowpass: Enable low-pass filter for noise reduction
            lowpass_cutoff: Low-pass filter cutoff frequency in Hz
        """
        self.window_size = window_size
        self.threshold = threshold
        self.selected_subcarriers = selected_subcarriers
        self.track_data = track_data
        
        # Use production SegmentationContext
        self._context = SegmentationContext(
            window_size=window_size,
            threshold=threshold,
            enable_hampel=enable_hampel,
            hampel_window=hampel_window,
            hampel_threshold=hampel_threshold,
            enable_lowpass=enable_lowpass,
            lowpass_cutoff=lowpass_cutoff
        )
        
        self.state = 'IDLE'
        self.motion_packet_count = 0
        
        # Expose turbulence_buffer for subclasses (e.g., HampelMVSDetector)
        self.turbulence_buffer: List[float] = []
        
        if track_data:
            self.moving_var_history: List[float] = []
            self.state_history: List[str] = []
    
    def process_packet(self, csi_data):
        """
        Process a single CSI packet
        
        Args:
            csi_data: CSI data array (I/Q pairs)
        """
        # Calculate turbulence using SegmentationContext method
        turb = self._context.calculate_spatial_turbulence(csi_data, self.selected_subcarriers)
        
        # Add to segmentation context
        self._context.add_turbulence(turb)
        
        # Lazy evaluation: must call update_state() to calculate variance and update state
        self._context.update_state()
        
        # Map state from SegmentationContext to string
        new_state = 'MOTION' if self._context.state == SegmentationContext.STATE_MOTION else 'IDLE'
        
        if self.track_data:
            self.moving_var_history.append(self._context.current_moving_variance)
            self.state_history.append(self.state)
        
        self.state = new_state
        
        if self.state == 'MOTION':
            self.motion_packet_count += 1
    
    def reset(self):
        """Reset detector state (full reset, including buffer)"""
        self._context.reset(full=True)
        self.state = 'IDLE'
        self.motion_packet_count = 0
        self.turbulence_buffer = []
        if self.track_data:
            self.moving_var_history = []
            self.state_history = []
    
    def get_motion_count(self) -> int:
        """Get number of packets detected as motion"""
        return self.motion_packet_count


def test_mvs_configuration(baseline_packets, movement_packets, 
                          subcarriers, threshold, window_size) -> Tuple[int, int, float]:
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
    """
    # Test on baseline (FP)
    detector = MVSDetector(window_size, threshold, subcarriers)
    for pkt in baseline_packets:
        detector.process_packet(pkt['csi_data'])
    fp = detector.get_motion_count()
    
    # Test on movement (TP)
    detector.reset()
    for pkt in movement_packets:
        detector.process_packet(pkt['csi_data'])
    tp = detector.get_motion_count()
    
    # Calculate score
    if tp == 0:
        score = -1000.0
    elif fp == 0:
        score = 1000.0 + tp
    else:
        score = tp - fp * 100
    
    return fp, tp, score
