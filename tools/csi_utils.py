"""
CSI Utilities - Common module for CSI data handling

Provides:
  - UDP reception (CSIReceiver)
  - Data collection (CSICollector)
  - Dataset management (load, save, stats)
  - MVS detection (MVSDetector)
  - Path setup for all tools (setup_paths)

Author: Francesco Pace <francesco.pace@gmail.com>
License: GPLv3
"""

import socket
import struct
import subprocess
import sys
import threading
import time
import ipaddress
import json
import numpy as np
import math
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, Iterable, List, Optional, Dict, Any, Tuple

from repo_paths import data_dir, python_src_dir, repo_root, tools_dir


# ============================================================================
# Path Setup (called once at module import)
# ============================================================================

def setup_paths():
    """
    Add repository-root-relative source directories to sys.path.
    
    This allows tools to import from src/ and config.py.
    Safe to call multiple times (checks for duplicates).
    """
    src_path = str(python_src_dir())
    current_tools_path = str(tools_dir())
    root_path = str(repo_root())

    if src_path not in sys.path:
        sys.path.insert(0, src_path)
    if current_tools_path not in sys.path:
        sys.path.insert(0, current_tools_path)
    if root_path not in sys.path:
        sys.path.insert(0, root_path)


# Auto-setup paths when this module is imported
setup_paths()
try:
    import src.config as config
except ImportError:
    import config

# ============================================================================
# Constants
# ============================================================================

# UDP Protocol constants
MAGIC_STREAM = 0x4353  # "CS" in little-endian
STREAM_VERSION = 3
DEFAULT_PORT = 5001
STREAM_FLAG_FIRST_WORD_INVALID = 1 << 0
STREAM_FLAG_WIFI_RX_TS_VALID = 1 << 1
STREAM_FLAG_WIFI_RX_START_TS_NS_VALID = 1 << 2
STREAM_FLAG_STIMULUS_ID_VALID = 1 << 3
STREAM_FLAG_REFERENCE_FRAME = 1 << 4
CSI_HEADER_FORMAT = '<HBBBBIHHQQIQIBbb'
CSI_HEADER_STRUCT = struct.Struct(CSI_HEADER_FORMAT)
MAX_STREAM_DATAGRAM_BYTES = 2048
STIMULUS_MAGIC = b'ESTM'
STIMULUS_VERSION = 1
STIMULUS_ROLE_MEASUREMENT = 0
STIMULUS_ROLE_REFERENCE = 1
DEFAULT_STIMULUS_PORT = 9999
DEFAULT_STIMULUS_RATE_PPS = 100
STIMULUS_HEADER_STRUCT = struct.Struct('>4sBBI')


def get_default_bind_host() -> str:
    """
    Determine a safe default bind interface (single host address, no wildcard).

    Priority:
    1. CSI_BIND_HOST env var if set
    2. Primary outbound IPv4 detected via UDP connect trick
    3. Loopback as final fallback
    """
    import os

    env_host = os.getenv('CSI_BIND_HOST', '').strip()
    if env_host:
        return env_host

    probe = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        probe.connect(('8.8.8.8', 80))
        return probe.getsockname()[0]
    except OSError:
        return '127.0.0.1'
    finally:
        probe.close()

# Dataset paths (shared between tools and tests)
DATA_DIR = data_dir()
DATASET_INFO_FILE = DATA_DIR / 'dataset_info.json'


def format_device_token(device_id: int) -> str:
    """Return a stable ASCII token used in filenames and metadata."""
    return f'dev{int(device_id):016x}'


def format_device_id_hex(device_id: int) -> str:
    """Return the canonical hexadecimal device identifier string."""
    return f'0x{int(device_id):016x}'



# ============================================================================
# Data Structures
# ============================================================================

@dataclass
class CSIPacket:
    """Represents a single CSI packet received via UDP"""
    timestamp: float          # Reception timestamp (seconds since epoch)
    seq_num: int             # Sequence number (uint32)
    num_subcarriers: int     # Number of subcarriers
    iq_raw: np.ndarray       # Raw I/Q values as int8 array [Q0,I0,Q1,I1,...] (Espressif format)
    iq_complex: np.ndarray   # Complex representation [I0+jQ0, I1+jQ1, ...]
    amplitudes: np.ndarray   # Amplitude per subcarrier
    phases: np.ndarray       # Phase per subcarrier (radians)
    chip: str = 'unknown'    # Chip type (e.g., 'C6', 'S3', 'ESP32')
    device_id: Optional[int] = None
    device_ticks_us: Optional[int] = None
    wifi_rx_ts_us: Optional[int] = None
    wifi_rx_start_ts_ns: Optional[int] = None
    stimulus_id: Optional[int] = None
    is_reference: bool = False
    channel: Optional[int] = None
    rssi_dbm: Optional[int] = None
    noise_floor_dbm: Optional[int] = None
    source_ip: Optional[str] = None


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
# Stimulus Generation
# ============================================================================


def build_stimulus_datagram(stimulus_id: int, *, is_reference: bool = False) -> bytes:
    """Build one ESTM datagram consumed by the streamer firmware."""
    if stimulus_id < 0 or stimulus_id > 0xFFFFFFFF:
        raise ValueError(f'stimulus_id out of range: {stimulus_id}')
    role = STIMULUS_ROLE_REFERENCE if is_reference else STIMULUS_ROLE_MEASUREMENT
    return STIMULUS_HEADER_STRUCT.pack(STIMULUS_MAGIC, STIMULUS_VERSION, role, stimulus_id)


class StimulusSender:
    """Background UDP sender that drives streamer-side CSI stimulus."""

    def __init__(
        self,
        target_host: str | Iterable[str],
        target_port: int = DEFAULT_STIMULUS_PORT,
        rate_pps: int = DEFAULT_STIMULUS_RATE_PPS,
        reference_every: int = 0,
        stimulus_id_start: int = 1,
        source_host: Optional[str] = None,
    ):
        if target_port <= 0 or target_port > 65535:
            raise ValueError(f'invalid target_port: {target_port}')
        if rate_pps <= 0:
            raise ValueError(f'rate_pps must be > 0, got {rate_pps}')
        if reference_every < 0:
            raise ValueError(f'reference_every must be >= 0, got {reference_every}')
        if stimulus_id_start < 0 or stimulus_id_start > 0xFFFFFFFF:
            raise ValueError(f'invalid stimulus_id_start: {stimulus_id_start}')

        raw_targets = [target_host] if isinstance(target_host, str) else list(target_host)
        self.target_hosts: List[str] = []
        self.target_ips: List[ipaddress.IPv4Address] = []
        for raw_target in raw_targets:
            target = str(raw_target).strip()
            if not target:
                continue
            try:
                target_ip = ipaddress.ip_address(target)
            except ValueError as exc:
                raise ValueError(f'invalid target_host: {target}') from exc
            if target_ip.version != 4:
                raise ValueError(f'target_host must be an IPv4 address: {target}')
            self.target_hosts.append(target)
            self.target_ips.append(target_ip)
        if not self.target_hosts:
            raise ValueError('target_host cannot be empty')
        self.target_port = int(target_port)
        self.rate_pps = int(rate_pps)
        self.reference_every = int(reference_every)
        self.next_stimulus_id = int(stimulus_id_start)
        self.source_host = str(source_host).strip() if source_host is not None else ''
        if self.source_host:
            try:
                ipaddress.ip_address(self.source_host)
            except ValueError as exc:
                raise ValueError(f'invalid source_host: {self.source_host}') from exc
        self.sent_packets = 0
        self.sock: Optional[socket.socket] = None
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

    def start(self) -> None:
        """Start sending ESTM packets in the background."""
        if self._thread is not None:
            return

        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        if any(target_ip.is_multicast for target_ip in self.target_ips):
            self.sock.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_TTL, 1)
        if self.source_host:
            self.sock.bind((self.source_host, 0))
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, name='espectre-stimulus', daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Stop the background stimulus sender."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None
        if self.sock is not None:
            self.sock.close()
            self.sock = None

    def _run(self) -> None:
        interval_s = 1.0 / float(self.rate_pps)
        next_deadline = time.monotonic()
        while not self._stop_event.is_set():
            packet_index = self.sent_packets + 1
            is_reference = self.reference_every > 0 and (packet_index % self.reference_every) == 0
            payload = build_stimulus_datagram(self.next_stimulus_id, is_reference=is_reference)
            try:
                if self.sock is not None:
                    for target_host in self.target_hosts:
                        self.sock.sendto(payload, (target_host, self.target_port))
            except OSError:
                pass

            self.sent_packets += 1
            self.next_stimulus_id = (self.next_stimulus_id + 1) & 0xFFFFFFFF
            next_deadline += interval_s
            sleep_s = max(0.0, next_deadline - time.monotonic())
            self._stop_event.wait(sleep_s)


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
    
    def __init__(
        self,
        port: int = DEFAULT_PORT,
        buffer_size: int = 500,
        bind_host: Optional[str] = None
    ):
        """
        Initialize CSI receiver.
        
        Args:
            port: UDP port to listen on
            buffer_size: Circular buffer size (packets)
            bind_host: Local interface IP to bind UDP socket
        """
        self.port = port
        self.buffer_size = buffer_size
        resolved_bind_host = bind_host or get_default_bind_host()
        self.bind_host = str(resolved_bind_host).strip()
        if not self.bind_host:
            raise ValueError('bind_host cannot be empty')
        try:
            ipaddress.ip_address(self.bind_host)
        except ValueError as exc:
            raise ValueError(f'Invalid bind_host: {self.bind_host}') from exc
        
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
    
    def _parse_record(self, data: bytes, offset: int = 0) -> Tuple[Optional[CSIPacket], int]:
        """Parse one CSI record from a datagram starting at offset.
        
        Packet format (version 3):
            <magic:2><version:1><header_len:1><chip:1><flags:1>
            <seq:4><num_sc:2><csi_len:2><device_id:8><device_ticks_us:8>
            <wifi_rx_ts_us:4><wifi_rx_start_ts_ns:8><stimulus_id:4>
            <channel:1><rssi_dbm:1><noise_floor_dbm:1>
            <payload>
        """
        if offset < 0 or len(data) - offset < CSI_HEADER_STRUCT.size:
            return None, offset

        (
            magic,
            version,
            header_len,
            chip_code,
            flags,
            seq_num,
            num_sc,
            csi_len_bytes,
            device_id,
            device_ticks_us,
            wifi_rx_ts_us,
            wifi_rx_start_ts_ns,
            stimulus_id,
            channel,
            rssi_dbm,
            noise_floor_dbm,
        ) = CSI_HEADER_STRUCT.unpack_from(data, offset)

        if magic != MAGIC_STREAM or version != STREAM_VERSION:
            return None, offset

        if header_len < CSI_HEADER_STRUCT.size:
            return None, offset
        if csi_len_bytes == 0 or csi_len_bytes != num_sc * 2:
            return None, offset

        record_len = header_len + csi_len_bytes
        if len(data) - offset < record_len:
            return None, offset

        chip = CHIP_CODES.get(chip_code, 'unknown')

        iq_raw = np.array(
            struct.unpack(
                f'<{csi_len_bytes}b',
                data[offset + header_len:offset + header_len + csi_len_bytes]
            ),
            dtype=np.int8
        )

        # Espressif CSI format: [Imaginary, Real, ...] per subcarrier
        Q = iq_raw[0::2].astype(np.float32)  # Imaginary first (even indices)
        I = iq_raw[1::2].astype(np.float32)  # Real second (odd indices)
        iq_complex = I + 1j * Q
        
        # Calculate amplitude and phase
        amplitudes = np.abs(iq_complex)
        phases = np.angle(iq_complex)
        
        packet = CSIPacket(
            timestamp=time.time(),
            seq_num=seq_num,
            num_subcarriers=num_sc,
            iq_raw=iq_raw,
            iq_complex=iq_complex,
            amplitudes=amplitudes,
            phases=phases,
            chip=chip,
            device_id=device_id or None,
            device_ticks_us=device_ticks_us or None,
            wifi_rx_ts_us=wifi_rx_ts_us if (flags & STREAM_FLAG_WIFI_RX_TS_VALID) else None,
            wifi_rx_start_ts_ns=wifi_rx_start_ts_ns if (flags & STREAM_FLAG_WIFI_RX_START_TS_NS_VALID) else None,
            stimulus_id=stimulus_id if (flags & STREAM_FLAG_STIMULUS_ID_VALID) else None,
            is_reference=bool(flags & STREAM_FLAG_REFERENCE_FRAME),
            channel=int(channel),
            rssi_dbm=int(rssi_dbm),
            noise_floor_dbm=int(noise_floor_dbm),
        )
        return packet, offset + record_len

    def _parse_packets(self, data: bytes) -> List[CSIPacket]:
        """Parse one or more concatenated CSI stream records from a UDP datagram."""
        packets: List[CSIPacket] = []
        offset = 0
        while offset < len(data):
            packet, next_offset = self._parse_record(data, offset)
            if packet is None or next_offset <= offset:
                return []
            packets.append(packet)
            offset = next_offset
        return packets

    def _parse_packet(self, data: bytes) -> Optional[CSIPacket]:
        """Parse a datagram that contains exactly one CSI stream record."""
        packets = self._parse_packets(data)
        if len(packets) != 1:
            return None
        return packets[0]

    @staticmethod
    def _compute_sequence_gap(previous_seq: int, current_seq: int) -> int:
        """Return the forward gap between two uint32 sequence numbers.

        Small forward deltas, including wrap-around, are counted as packet loss.
        Large modular deltas are treated as out-of-order packets or sender resets
        and do not contribute to the drop counter.
        """
        expected = (previous_seq + 1) & 0xFFFFFFFF
        delta = (current_seq - expected) & 0xFFFFFFFF
        if delta == 0:
            return 0
        if delta >= 0x80000000:
            return 0
        return delta
    
    def _check_sequence(self, seq_num: int):
        """Track sequence numbers and detect drops"""
        if self.last_seq >= 0:
            self.dropped_count += self._compute_sequence_gap(self.last_seq, seq_num)
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
        total_expected = self.packet_count + self.dropped_count
        return {
            'packets': self.packet_count,
            'dropped': self.dropped_count,
            'drop_rate': self.dropped_count / max(total_expected, 1) * 100,
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
        self.sock.bind((self.bind_host, self.port))
        self.sock.settimeout(1.0)  # 1 second timeout for graceful shutdown
        
        if not quiet:
            print(f'CSI Receiver listening on {self.bind_host}:{self.port}')
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
                    data, addr = self.sock.recvfrom(MAX_STREAM_DATAGRAM_BYTES)
                except socket.timeout:
                    self._update_pps()
                    continue
                
                packets = self._parse_packets(data)
                if not packets:
                    continue

                for packet in packets:
                    packet.source_ip = addr[0]
                    self._check_sequence(packet.seq_num)
                    self.buffer.append(packet)
                    self.packet_count += 1
                    self._pps_counter += 1
                    self._update_pps()

                    for callback in self._callbacks:
                        try:
                            callback(packet)
                        except Exception as e:
                            print(f'Callback error: {e}')

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
    FORMAT_VERSION = '1.1'
    # Implicit readiness gate before each sample recording
    READY_STABLE_SECONDS = 3.0
    READY_MV_THRESHOLD = 1.0
    STATUS_REFRESH_SECONDS = 0.2
    
    def __init__(
        self,
        label: str,
        port: int = DEFAULT_PORT,
        contributor: str = None,
        description: str = None,
        bind_host: Optional[str] = None,
        expected_device_count: Optional[int] = None,
        expected_source_hosts: Optional[List[str]] = None,
    ):
        """
        Initialize collector.
        
        Args:
            label: Label for collected samples (e.g., 'wave', 'static_presence')
            port: UDP port for CSI receiver
            contributor: GitHub username of the contributor (auto-detected from git if not provided)
            description: Optional description for the collected samples
            bind_host: Local interface IP to bind UDP socket
            expected_device_count: Number of devices expected to participate in the session
            expected_source_hosts: Expected streamer IPs for readiness logging
        """
        self.label = label
        self.port = port
        self.bind_host = bind_host
        self.chip = None  # Auto-detected from CSI packets
        self.contributor = contributor or get_git_username()
        self.description = description
        self.expected_source_hosts = list(dict.fromkeys(expected_source_hosts or []))
        self.expected_device_count = max(1, int(expected_device_count)) if expected_device_count is not None else 1
        
        self.receiver = CSIReceiver(port=port, buffer_size=2000, bind_host=bind_host)
        self._recording = False
        self._recorded_packets: List[CSIPacket] = []
        self._sample_count = 0
        self._ready_detector = self._build_ready_detector()
        self._live_status_line_count = 0
    
    def _get_label_dir(self) -> Path:
        """Get directory for this label, create if needed"""
        label_dir = DATA_DIR / self.label
        label_dir.mkdir(parents=True, exist_ok=True)
        return label_dir
    
    def _generate_filename(self, num_subcarriers: int, device_id: int) -> str:
        """Generate a collision-safe per-device filename."""
        self._sample_count += 1
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        chip = self.chip or 'unknown'
        device_token = format_device_token(device_id)
        return f'{self.label}_{chip}_{num_subcarriers}sc_{device_token}_{timestamp}_{self._sample_count:04d}.npz'

    @staticmethod
    def _require_single_device_id(packets: List[CSIPacket]) -> int:
        """Validate that one device and only one device is present."""
        missing_device_packets = sum(1 for packet in packets if packet.device_id is None)
        if missing_device_packets:
            raise ValueError(
                f'cannot save sample without device_id metadata ({missing_device_packets} packets missing device_id)'
            )

        device_ids = {int(packet.device_id) for packet in packets if packet.device_id is not None}
        if len(device_ids) != 1:
            raise ValueError(f'cannot save mixed-device sample as one file: found {len(device_ids)} device ids')
        return next(iter(device_ids))

    def save_samples_by_device(self, packets: List[CSIPacket]) -> List[Path]:
        """Split a mixed capture window into one file per device."""
        if not packets:
            return []

        packets_by_device: Dict[int, List[CSIPacket]] = {}
        missing_device_packets = 0
        for packet in packets:
            if packet.device_id is None:
                missing_device_packets += 1
                continue
            packets_by_device.setdefault(int(packet.device_id), []).append(packet)

        if missing_device_packets:
            raise ValueError(
                f'cannot save capture window without device_id metadata '
                f'({missing_device_packets} packets missing device_id)'
            )

        saved_files: List[Path] = []
        for device_id in sorted(packets_by_device):
            filepath = self.save_sample(packets_by_device[device_id])
            if filepath is not None:
                saved_files.append(filepath)
        return saved_files
    
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
            chip: str - Chip type (C6, S3, etc.)
            collected_at: str - ISO timestamp
            duration_ms: float - Total duration
            format_version: str - Format version
            optional stream metadata arrays - Device timestamps and RF metadata
        """
        if not packets:
            return None
        
        device_id = self._require_single_device_id(packets)

        # Auto-detect chip from first packet (v2 protocol)
        if packets[0].chip and packets[0].chip != 'unknown':
            self.chip = packets[0].chip.lower()
        
        # Extract I/Q raw data (compact int8 format)
        csi_data = np.array([p.iq_raw for p in packets], dtype=np.int8)
        
        # Calculate duration from timestamps
        timestamps = np.array([p.timestamp for p in packets])
        duration_ms = (timestamps[-1] - timestamps[0]) * 1000 if len(timestamps) > 1 else 0
        
        # Build sample dict (unified format)
        sample = {
            # CSI data (essential)
            'csi_data': csi_data,
            'num_subcarriers': packets[0].num_subcarriers,
            
            # Label (ground truth)
            'label': self.label,
            
            # Context
            'chip': self.chip or 'unknown',
            
            # Metadata
            'collected_at': datetime.now().isoformat(),
            'duration_ms': duration_ms,
            'format_version': self.FORMAT_VERSION,
        }
        
        # Packet-level metadata for replay, synchronization, and realtime fusion.
        sample['stream_seq_num'] = np.array([p.seq_num for p in packets], dtype=np.uint32)

        device_ticks = [p.device_ticks_us for p in packets]
        if all(value is not None for value in device_ticks):
            sample['device_ticks_us'] = np.array(device_ticks, dtype=np.uint64)

        sample['device_id'] = np.uint64(device_id)

        def add_optional_array(key: str, values, dtype) -> None:
            if any(value is not None for value in values):
                sample[key] = np.array([0 if value is None else value for value in values], dtype=dtype)

        add_optional_array('wifi_rx_ts_us', [p.wifi_rx_ts_us for p in packets], np.uint32)
        add_optional_array('wifi_rx_start_ts_ns', [p.wifi_rx_start_ts_ns for p in packets], np.uint64)
        add_optional_array('stimulus_id', [p.stimulus_id for p in packets], np.uint32)
        add_optional_array('is_reference', [1 if p.is_reference else 0 for p in packets], np.uint8)
        add_optional_array('channel', [p.channel for p in packets], np.uint8)
        add_optional_array('rssi_dbm', [p.rssi_dbm for p in packets], np.int16)
        add_optional_array('noise_floor_dbm', [p.noise_floor_dbm for p in packets], np.int16)
        
        # Save file
        label_dir = self._get_label_dir()
        num_subcarriers = packets[0].num_subcarriers
        filename = self._generate_filename(num_subcarriers, device_id)
        filepath = label_dir / filename
        
        np.savez_compressed(filepath, **sample)
        
        # Update dataset info with file details
        self._update_dataset_info(
            filename=filename,
            num_subcarriers=num_subcarriers,
            num_packets=len(packets),
            duration_ms=duration_ms,
            collected_at=sample['collected_at'],
            description=self.description,
            device_id=device_id
        )
        
        return filepath
    
    def _update_dataset_info(self, filename: str = None, num_subcarriers: int = None,
                                num_packets: int = None, duration_ms: float = None,
                                collected_at: str = None, description: str = None,
                                device_id: Optional[int] = None):
        """Update dataset info with current sample counts and file details"""
        info = load_dataset_info()
        
        # Count samples for this label
        label_dir = self._get_label_dir()
        sample_count = len(list(label_dir.glob('*.npz')))
        
        if self.label not in info['labels']:
            info['labels'][self.label] = {
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
                if not description:
                    description = f'HT20 {self.label}, AGC-active normalized pipeline'
                
                file_info = {
                    'filename': filename,
                    'chip': self.chip.upper() if self.chip else 'unknown',
                    'subcarriers': num_subcarriers,
                    'contributor': self.contributor or '',
                    'collected_at': collected_at or '',
                    'duration_ms': int(duration_ms) if duration_ms else 0,
                    'num_packets': num_packets or 0,
                    'description': description,
                    'device_id': format_device_id_hex(device_id) if device_id is not None else '',
                }
                info['files'][self.label].append(file_info)
        
        save_dataset_info(info)

    def _drain_udp_backlog(self, max_packets: int = 10000) -> int:
        """
        Drain queued UDP packets to align sample start with current time.

        When `collect_timed()` waits during countdown, packets can accumulate in
        the OS socket buffer. Without draining, the next sample may include old
        packets (pre-countdown), inflating packet count and breaking duration
        coherence against streamer PPS.

        Args:
            max_packets: Safety cap to avoid infinite loops

        Returns:
            int: Number of drained packets
        """
        if self.receiver.sock is None:
            return 0

        drained = 0
        previous_timeout = self.receiver.sock.gettimeout()
        self.receiver.sock.settimeout(0.0)  # non-blocking drain
        try:
            while drained < max_packets:
                try:
                    self.receiver.sock.recvfrom(MAX_STREAM_DATAGRAM_BYTES)
                    drained += 1
                except (BlockingIOError, socket.timeout):
                    break
        finally:
            self.receiver.sock.settimeout(previous_timeout)
        return drained

    def _build_ready_detector(self) -> "MVSDetector":
        """
        Build a lightweight MVS detector used only as pre-recording gate.

        Uses the unified default subcarriers to provide a stable and model-aligned
        readiness indicator before each sample acquisition.
        """
        window_size = int(getattr(config, 'SEG_WINDOW_SIZE', 100))
        if window_size < 10:
            window_size = 10
        elif window_size > 200:
            window_size = 200

        return MVSDetector(
            window_size=window_size,
            threshold=self.READY_MV_THRESHOLD,
            track_data=False
        )

    def _reset_live_status_block(self) -> None:
        """Forget the currently rendered inline status block."""
        self._live_status_line_count = 0

    def _render_live_status_block(
        self,
        summary_line: str,
        detail_lines: List[str],
        *,
        inline: Optional[bool] = None,
    ) -> None:
        """Render and remember the current inline status block size."""
        self._live_status_line_count = self._emit_ready_status_block(
            summary_line,
            detail_lines,
            previous_line_count=self._live_status_line_count,
            inline=inline,
        )

    @staticmethod
    def _build_status_bar(ratio: float, width: int = 18) -> str:
        """Build a compact ASCII progress bar for terminal status."""
        clamped = max(0.0, min(1.0, ratio))
        filled = int(round(clamped * width))
        return '[' + ('#' * filled) + ('-' * (width - filled)) + ']'

    @staticmethod
    def _supports_inline_terminal(stream: Any = None) -> bool:
        """Return True when the output stream supports inline ANSI refresh."""
        target_stream = sys.stdout if stream is None else stream
        isatty = getattr(target_stream, 'isatty', None)
        return bool(callable(isatty) and isatty())

    @staticmethod
    def _emit_ready_status_block(
        summary_line: str,
        detail_lines: List[str],
        *,
        previous_line_count: int = 0,
        stream: Any = None,
        inline: Optional[bool] = None,
    ) -> int:
        """Render the readiness summary and detail lines, optionally in place."""
        target_stream = sys.stdout if stream is None else stream
        use_inline = CSICollector._supports_inline_terminal(target_stream) if inline is None else inline
        lines = [summary_line, *detail_lines]

        if not use_inline:
            for line in lines:
                target_stream.write(f'{line}\n')
            target_stream.flush()
            return len(lines)

        if previous_line_count > 0:
            target_stream.write(f'\x1b[{previous_line_count}F')

        total_lines = max(previous_line_count, len(lines))
        for idx in range(total_lines):
            target_stream.write('\x1b[2K')
            if idx < len(lines):
                target_stream.write(lines[idx])
            target_stream.write('\n')

        target_stream.flush()
        return len(lines)

    @staticmethod
    def _check_sequence_by_device(packet: CSIPacket, last_seq_by_device: Dict[int, int]) -> int:
        """Track per-device sequence gaps and return detected drops."""
        if packet.device_id is None:
            return 0
        device_id = int(packet.device_id)
        dropped = 0
        if device_id in last_seq_by_device:
            dropped = CSIReceiver._compute_sequence_gap(last_seq_by_device[device_id], packet.seq_num)
        last_seq_by_device[device_id] = packet.seq_num
        return dropped

    @staticmethod
    def _summarize_ready_devices(
        device_states: Dict[int, Dict[str, Any]],
        *,
        expected_device_count: int,
        warmup_target: int,
        threshold: float,
        now: float,
    ) -> Dict[str, Any]:
        """Summarize multi-device readiness for status rendering and gating."""
        observed_count = len(device_states)
        required_count = max(1, expected_device_count)
        relevant_states = list(device_states.values())

        if observed_count < required_count:
            return {
                'ready': False,
                'status': f'DEVICES {observed_count}/{required_count}',
                'stable_elapsed': 0.0,
                'ready_count': 0,
                'observed_count': observed_count,
                'required_count': required_count,
            }

        warm_states = [state for state in relevant_states if state['processed_packets'] >= warmup_target]
        total_relevant = max(observed_count, required_count)
        if len(warm_states) < observed_count:
            return {
                'ready': False,
                'status': f'WARMUP {len(warm_states)}/{total_relevant}',
                'stable_elapsed': 0.0,
                'ready_count': 0,
                'observed_count': observed_count,
                'required_count': required_count,
            }

        if any(state['current_mv'] > threshold for state in relevant_states):
            ready_count = sum(1 for state in relevant_states if state['current_mv'] <= threshold)
            return {
                'ready': False,
                'status': f'UNSTABLE {ready_count}/{total_relevant}',
                'stable_elapsed': 0.0,
                'ready_count': ready_count,
                'observed_count': observed_count,
                'required_count': required_count,
            }

        stable_elapsed = min(
            max(0.0, now - state['stable_since']) if state['stable_since'] is not None else 0.0
            for state in relevant_states
        )
        ready = stable_elapsed >= CSICollector.READY_STABLE_SECONDS
        return {
            'ready': ready,
            'status': f'READY {observed_count}/{total_relevant}' if ready else f'STABLE {observed_count}/{total_relevant}',
            'stable_elapsed': stable_elapsed,
            'ready_count': observed_count,
            'observed_count': observed_count,
            'required_count': required_count,
        }

    @staticmethod
    def _format_ready_device_lines(
        device_states: Dict[int, Dict[str, Any]],
        *,
        expected_source_hosts: List[str],
        warmup_target: int,
        threshold: float,
        now: float,
    ) -> List[str]:
        """Build detailed per-device readiness lines for terminal logging."""
        lines: List[str] = []
        seen_ips = {state.get('source_ip') for state in device_states.values() if state.get('source_ip')}

        for expected_ip in expected_source_hosts:
            if expected_ip not in seen_ips:
                lines.append(
                    f'    ip={expected_ip} chip=? ch=-- rssi=--- '
                    f'{CSICollector._build_status_bar(0.0)} '
                    f'mv=--/{threshold:.3f} pps=-- '
                    f'| WAITING'
                )

        for device_id in sorted(device_states):
            state = device_states[device_id]
            source_ip = state.get('source_ip', '?')
            chip = str(state.get('chip', '?')).upper()
            channel = state.get('channel')
            rssi_dbm = state.get('rssi_dbm')
            processed_packets = int(state.get('processed_packets', 0))
            current_mv = float(state.get('current_mv', 0.0))
            current_pps = state.get('current_pps')
            stable_since = state.get('stable_since')

            if processed_packets < warmup_target:
                status = f'WARMUP {processed_packets}/{warmup_target}'
                stable_value = 0.0
                mv_ratio = 0.0
            else:
                stable_value = max(0.0, now - stable_since) if stable_since is not None else 0.0
                mv_ratio = min(current_mv / threshold, 1.0) if threshold > 0 else 0.0
                if current_mv > threshold:
                    status = 'UNSTABLE'
                    stable_value = 0.0
                elif stable_value >= CSICollector.READY_STABLE_SECONDS:
                    status = 'READY'
                    stable_value = CSICollector.READY_STABLE_SECONDS
                else:
                    status = 'STABLE'

            mv_text = '--' if processed_packets < warmup_target else f'{current_mv:.3f}'
            channel_text = '--' if channel is None else f'{int(channel):02d}'
            rssi_text = '---' if rssi_dbm is None else str(int(rssi_dbm))
            pps_text = '--' if current_pps is None else str(int(current_pps))
            lines.append(
                f'    ip={source_ip} chip={chip} ch={channel_text} rssi={rssi_text} '
                f'{CSICollector._build_status_bar(mv_ratio)} '
                f'mv={mv_text}/{threshold:.3f} pps={pps_text} '
                f'| {status}'
            )

        return lines

    def _wait_for_ready_state(
        self,
        quiet: bool = False,
        summary_prefix: str = '  ',
    ) -> Dict[int, Dict[str, Any]]:
        """
        Wait until environment is stable before recording.

        Ready condition:
        - moving variance <= READY_MV_THRESHOLD
        - condition remains true for READY_STABLE_SECONDS continuously
        """
        if self.receiver.sock is None:
            raise RuntimeError('Receiver socket is not initialized')

        self.receiver.reset_stats()
        warmup_target = self._ready_detector.window_size
        device_states: Dict[int, Dict[str, Any]] = {}
        last_seq_by_device: Dict[int, int] = {}
        processed_packets = 0
        last_render = 0.0
        last_pps_time = time.monotonic()
        last_pps_count = 0
        current_pps = 0
        current_state = f'DEVICES 0/{self.expected_device_count}'
        detail_render_interval = self.STATUS_REFRESH_SECONDS
        use_inline_status = self._supports_inline_terminal()
        while True:
            try:
                data, addr = self.receiver.sock.recvfrom(MAX_STREAM_DATAGRAM_BYTES)
                packets = self.receiver._parse_packets(data)
                if not packets:
                    continue

                for packet in packets:
                    processed_packets += 1
                    self.receiver.packet_count += 1
                    self.receiver.dropped_count += self._check_sequence_by_device(packet, last_seq_by_device)

                    if packet.device_id is None:
                        continue

                    device_id = int(packet.device_id)
                    state = device_states.get(device_id)
                    if state is None:
                        state = {
                            'detector': self._build_ready_detector(),
                            'processed_packets': 0,
                            'stable_since': None,
                            'current_mv': 0.0,
                            'current_pps': 0,
                            'last_pps_count': 0,
                            'source_ip': addr[0],
                            'chip': packet.chip or 'unknown',
                            'channel': packet.channel,
                            'rssi_dbm': packet.rssi_dbm,
                            'last_seq': packet.seq_num,
                        }
                        device_states[device_id] = state
                    else:
                        state['source_ip'] = addr[0]
                        if packet.chip and packet.chip != 'unknown':
                            state['chip'] = packet.chip
                        if packet.channel is not None:
                            state['channel'] = packet.channel
                        if packet.rssi_dbm is not None:
                            state['rssi_dbm'] = packet.rssi_dbm
                        state['last_seq'] = packet.seq_num

                    packet_dict = {'csi_data': packet.iq_raw}
                    state['detector'].process_packet(packet_dict)
                    state['processed_packets'] += 1

                    if state['processed_packets'] >= warmup_target:
                        state['current_mv'] = state['detector']._context.current_moving_variance
                        now = time.monotonic()
                        if state['current_mv'] <= self.READY_MV_THRESHOLD:
                            if state['stable_since'] is None:
                                state['stable_since'] = now
                        else:
                            state['stable_since'] = None

                now = time.monotonic()
                summary = self._summarize_ready_devices(
                    device_states,
                    expected_device_count=self.expected_device_count,
                    warmup_target=warmup_target,
                    threshold=self.READY_MV_THRESHOLD,
                    now=now,
                )
                current_state = summary['status']
                if summary['ready']:
                    if not quiet:
                        detail_lines = self._format_ready_device_lines(
                            device_states,
                            expected_source_hosts=self.expected_source_hosts,
                            warmup_target=warmup_target,
                            threshold=self.READY_MV_THRESHOLD,
                            now=now,
                        )
                        summary_line = (
                            f'{summary_prefix}STATUS: READY {summary["observed_count"]}/{summary["required_count"]} '
                            + f'| pps {current_pps} '
                            + f'| drop {self.receiver.get_stats()["drop_rate"]:.1f}% '
                        )
                        self._render_live_status_block(
                            summary_line,
                            detail_lines,
                            inline=use_inline_status,
                        )
                    return device_states

                if now - last_pps_time >= 1.0:
                    delta = processed_packets - last_pps_count
                    elapsed = now - last_pps_time
                    current_pps = int(delta / elapsed) if elapsed > 0 else 0
                    for state in device_states.values():
                        device_delta = int(state.get('processed_packets', 0)) - int(state.get('last_pps_count', 0))
                        state['current_pps'] = int(device_delta / elapsed) if elapsed > 0 else 0
                        state['last_pps_count'] = int(state.get('processed_packets', 0))
                    last_pps_time = now
                    last_pps_count = processed_packets

                if (not quiet) and (now - last_render >= detail_render_interval):
                    drop_rate = self.receiver.get_stats()['drop_rate']
                    detail_lines = self._format_ready_device_lines(
                        device_states,
                        expected_source_hosts=self.expected_source_hosts,
                        warmup_target=warmup_target,
                        threshold=self.READY_MV_THRESHOLD,
                        now=now,
                    )
                    summary_line = (
                        f'{summary_prefix}STATUS: {current_state} '
                        + f'| pps {current_pps} '
                        + f'| drop {drop_rate:.1f}% '
                    )
                    self._render_live_status_block(
                        summary_line,
                        detail_lines,
                        inline=use_inline_status,
                    )
                    last_render = now

            except socket.timeout:
                now = time.monotonic()
                if (not quiet) and (now - last_render >= detail_render_interval):
                    detail_lines = self._format_ready_device_lines(
                        device_states,
                        expected_source_hosts=self.expected_source_hosts,
                        warmup_target=warmup_target,
                        threshold=self.READY_MV_THRESHOLD,
                        now=now,
                    )
                    summary_line = f'{summary_prefix}STATUS: NO DATA | pps 0 | drop 0.0%'
                    self._render_live_status_block(
                        summary_line,
                        detail_lines,
                        inline=use_inline_status,
                    )
                    last_render = now
                continue

    def _collect_with_live_status(
        self,
        duration: float,
        *,
        quiet: bool = False,
        initial_device_states: Optional[Dict[int, Dict[str, Any]]] = None,
        summary_prefix: str = '  ',
    ) -> List[CSIPacket]:
        """Collect a timed window while keeping the per-device status block live."""
        if self.receiver.sock is None:
            raise RuntimeError('Receiver socket is not initialized')

        self.receiver.reset_stats()
        packets: List[CSIPacket] = []
        deadline = time.monotonic() + duration
        warmup_target = self._ready_detector.window_size
        device_states: Dict[int, Dict[str, Any]] = dict(initial_device_states or {})
        last_seq_by_device: Dict[int, int] = {
            device_id: int(state['last_seq'])
            for device_id, state in device_states.items()
            if state.get('last_seq') is not None
        }
        processed_packets = 0
        last_render = 0.0
        last_pps_time = time.monotonic()
        last_pps_count = 0
        current_pps = 0
        use_inline_status = self._supports_inline_terminal()

        while time.monotonic() < deadline:
            try:
                data, addr = self.receiver.sock.recvfrom(MAX_STREAM_DATAGRAM_BYTES)
                parsed_packets = self.receiver._parse_packets(data)
                if not parsed_packets:
                    continue

                for packet in parsed_packets:
                    packets.append(packet)
                    processed_packets += 1
                    self.receiver.packet_count += 1
                    self.receiver.dropped_count += self._check_sequence_by_device(packet, last_seq_by_device)

                    if packet.device_id is None:
                        continue

                    device_id = int(packet.device_id)
                    state = device_states.get(device_id)
                    if state is None:
                        state = {
                            'detector': self._build_ready_detector(),
                            'processed_packets': 0,
                            'stable_since': None,
                            'current_mv': 0.0,
                            'current_pps': 0,
                            'last_pps_count': 0,
                            'source_ip': addr[0],
                            'chip': packet.chip or 'unknown',
                            'channel': packet.channel,
                            'rssi_dbm': packet.rssi_dbm,
                            'last_seq': packet.seq_num,
                        }
                        device_states[device_id] = state
                    else:
                        state['source_ip'] = addr[0]
                        if packet.chip and packet.chip != 'unknown':
                            state['chip'] = packet.chip
                        if packet.channel is not None:
                            state['channel'] = packet.channel
                        if packet.rssi_dbm is not None:
                            state['rssi_dbm'] = packet.rssi_dbm
                        state['last_seq'] = packet.seq_num

                    packet_dict = {'csi_data': packet.iq_raw}
                    state['detector'].process_packet(packet_dict)
                    state['processed_packets'] += 1

                    if state['processed_packets'] >= warmup_target:
                        state['current_mv'] = state['detector']._context.current_moving_variance
                        now = time.monotonic()
                        if state['current_mv'] <= self.READY_MV_THRESHOLD:
                            if state['stable_since'] is None:
                                state['stable_since'] = now
                        else:
                            state['stable_since'] = None

                now = time.monotonic()
                if now - last_pps_time >= 1.0:
                    delta = processed_packets - last_pps_count
                    elapsed = now - last_pps_time
                    current_pps = int(delta / elapsed) if elapsed > 0 else 0
                    for state in device_states.values():
                        device_delta = int(state.get('processed_packets', 0)) - int(state.get('last_pps_count', 0))
                        state['current_pps'] = int(device_delta / elapsed) if elapsed > 0 else 0
                        state['last_pps_count'] = int(state.get('processed_packets', 0))
                    last_pps_time = now
                    last_pps_count = processed_packets

                if (not quiet) and (now - last_render >= self.STATUS_REFRESH_SECONDS):
                    detail_lines = self._format_ready_device_lines(
                        device_states,
                        expected_source_hosts=self.expected_source_hosts,
                        warmup_target=warmup_target,
                        threshold=self.READY_MV_THRESHOLD,
                        now=now,
                    )
                    elapsed = max(0.0, duration - max(0.0, deadline - now))
                    stats = self.receiver.get_stats()
                    summary_line = (
                        f'{summary_prefix}STATUS: RECORDING '
                        + f'| elapsed {elapsed:.1f}/{duration:.1f}s '
                        + f'| pps {current_pps} '
                        + f'| drop {stats["drop_rate"]:.1f}% '
                        + f'| packets {len(packets)}'
                    )
                    self._render_live_status_block(
                        summary_line,
                        detail_lines,
                        inline=use_inline_status,
                    )
                    last_render = now
            except socket.timeout:
                now = time.monotonic()
                if (not quiet) and (now - last_render >= self.STATUS_REFRESH_SECONDS):
                    detail_lines = self._format_ready_device_lines(
                        device_states,
                        expected_source_hosts=self.expected_source_hosts,
                        warmup_target=warmup_target,
                        threshold=self.READY_MV_THRESHOLD,
                        now=now,
                    )
                    elapsed = max(0.0, duration - max(0.0, deadline - now))
                    stats = self.receiver.get_stats()
                    summary_line = (
                        f'{summary_prefix}STATUS: RECORDING '
                        + f'| elapsed {elapsed:.1f}/{duration:.1f}s '
                        + f'| pps {current_pps} '
                        + f'| drop {stats["drop_rate"]:.1f}% '
                        + f'| packets {len(packets)}'
                    )
                    self._render_live_status_block(
                        summary_line,
                        detail_lines,
                        inline=use_inline_status,
                    )
                    last_render = now
                continue

        return packets

    def collect_timed(self, duration: float, num_samples: int = 1, quiet: bool = False) -> List[Path]:
        """
        Collect samples with fixed duration.
        
        Args:
            duration: Duration per sample in seconds
            num_samples: Number of samples to collect
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
            print(f'  Ready gate:          implicit ({self.READY_STABLE_SECONDS:.1f}s stable)')
            print(f'{"=" * 60}\n')
        
        # Create socket once
        self.receiver.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.receiver.sock.bind((self.receiver.bind_host, self.port))
        self.receiver.sock.settimeout(0.1)
        
        try:
            for sample_idx in range(num_samples):
                self._reset_live_status_block()
                summary_prefix = f'  Sample {sample_idx + 1}/{num_samples} | '

                # Flush packets that accumulated during countdown/idle time.
                self._drain_udp_backlog()
                ready_device_states = self._wait_for_ready_state(
                    quiet=quiet,
                    summary_prefix=summary_prefix,
                )

                packets = self._collect_with_live_status(
                    duration,
                    quiet=quiet,
                    initial_device_states=ready_device_states,
                    summary_prefix=summary_prefix,
                )
                
                # Save sample
                sample_files = self.save_samples_by_device(packets)

                if sample_files:
                    saved_files.extend(sample_files)
                    if not quiet:
                        print(
                            f'\r  ✅ Saved {len(sample_files)} device file(s) '
                            f'from {len(packets)} packets'
                        )
                        for filepath in sample_files:
                            print(f'     - {filepath.name}')
                else:
                    if not quiet:
                        print(f'\r  ❌ No packets received!')
                self._reset_live_status_block()
        
        finally:
            if self.receiver.sock:
                self.receiver.sock.close()
        
        if not quiet:
            print(f'\n{"=" * 60}')
            print(f'  Collection complete: {len(saved_files)} device file(s) saved')
            print(f'{"=" * 60}\n')
        
        return saved_files
    
    def collect_interactive(self, num_samples: int = 10, duration: float = 2.0) -> List[Path]:
        """
        Collect samples with keyboard control.
        
        Press SPACE to start/stop recording, ENTER to save, R to retry, Q to quit.
        
        Args:
            num_samples: Target number of samples
            duration: Duration per sample in seconds
        
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
        self.receiver.sock.bind((self.receiver.bind_host, self.port))
        self.receiver.sock.settimeout(0.1)
        
        try:
            sample_idx = 0
            while sample_idx < num_samples:
                try:
                    self._reset_live_status_block()
                    user_input = input(f'\nSample {sample_idx + 1}/{num_samples} - Press ENTER to record (Q to quit): ')
                    
                    if user_input.lower() == 'q':
                        print('Collection cancelled.')
                        break
                    
                    print(f'  Recording for {duration} seconds...', end='', flush=True)

                    # Flush packets that accumulated while waiting for user input.
                    self._drain_udp_backlog()
                    ready_device_states = self._wait_for_ready_state(quiet=False)
                    packets = self._collect_with_live_status(
                        duration,
                        quiet=False,
                        initial_device_states=ready_device_states,
                    )
                    
                    # Save sample
                    sample_files = self.save_samples_by_device(packets)

                    if sample_files:
                        saved_files.extend(sample_files)
                        print(
                            f'\r  ✅ Saved {len(sample_files)} device file(s) '
                            f'from {len(packets)} packets'
                        )
                        for filepath in sample_files:
                            print(f'     - {filepath.name}')
                        sample_idx += 1
                    else:
                        print(f'\r  ❌ No packets received! Check the streamer firmware and collector IP/port.')
                    self._reset_live_status_block()
                        
                except KeyboardInterrupt:
                    print('\nCollection cancelled.')
                    break
        
        finally:
            if self.receiver.sock:
                self.receiver.sock.close()
        
        print(f'\n{"=" * 60}')
        print(f'  Collection complete: {len(saved_files)} device file(s) saved')
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
                except Exception:
                    avg_packets = 0
                
                stats['labels'][label] = {
                    'samples': len(samples),
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
    
    stream_seq_nums = data['stream_seq_num'] if 'stream_seq_num' in data.files else None
    device_ticks_us = data['device_ticks_us'] if 'device_ticks_us' in data.files else None
    wifi_rx_ts_us = data['wifi_rx_ts_us'] if 'wifi_rx_ts_us' in data.files else None
    wifi_rx_start_ts_ns = data['wifi_rx_start_ts_ns'] if 'wifi_rx_start_ts_ns' in data.files else None
    stimulus_ids = data['stimulus_id'] if 'stimulus_id' in data.files else None
    is_reference = data['is_reference'] if 'is_reference' in data.files else None
    device_ids = data['device_id'] if 'device_id' in data.files else None
    channels = data['channel'] if 'channel' in data.files else None
    rssi_dbm = data['rssi_dbm'] if 'rssi_dbm' in data.files else None
    noise_floor_dbm = data['noise_floor_dbm'] if 'noise_floor_dbm' in data.files else None

    def optional_scalar(array, index, cast):
        if array is None:
            return None
        if np.ndim(array) == 0:
            return cast(np.asarray(array).item())
        if index >= len(array):
            return None
        return cast(array[index])

    # Build packet list
    packets = []
    for i in range(len(csi_array)):
        packets.append({
            'csi_data': np.array(csi_array[i], dtype=np.int8),
            'label': label,
            'num_subcarriers': num_subcarriers,
            'chip': chip,
            'stream_seq_num': optional_scalar(stream_seq_nums, i, int),
            'device_ticks_us': optional_scalar(device_ticks_us, i, int),
            'wifi_rx_ts_us': optional_scalar(wifi_rx_ts_us, i, int),
            'wifi_rx_start_ts_ns': optional_scalar(wifi_rx_start_ts_ns, i, int),
            'stimulus_id': optional_scalar(stimulus_ids, i, int),
            'is_reference': bool(optional_scalar(is_reference, i, int) or 0),
            'device_id': optional_scalar(device_ids, i, int),
            'channel': optional_scalar(channels, i, int),
            'rssi_dbm': optional_scalar(rssi_dbm, i, int),
            'noise_floor_dbm': optional_scalar(noise_floor_dbm, i, int),
        })
    
    return packets


def find_static_presence_motion_dataset(chip: str = None, num_sc: int = 64) -> Tuple[Path, Path, str]:
    """
    Find static-presence and motion dataset files with nearest timestamps.
    
    Args:
        chip: Chip type (C6, S3, etc.) or None to find any chip
        num_sc: Number of subcarriers (default: 64 for HT20)
    
    Returns:
        tuple: (static_presence_path, motion_path, chip_name)
    
    Raises:
        FileNotFoundError: If no matching files found
    """
    static_presence_dir = DATA_DIR / 'static_presence'
    motion_dir = DATA_DIR / 'motion'
    
    # Build search pattern
    if chip:
        chip_lower = chip.lower()
        static_presence_pattern = f'static_presence_{chip_lower}_{num_sc}sc_*.npz'
        motion_pattern = f'motion_{chip_lower}_{num_sc}sc_*.npz'
    else:
        static_presence_pattern = f'*_{num_sc}sc_*.npz'
        motion_pattern = f'*_{num_sc}sc_*.npz'
    
    static_presence_files = list(static_presence_dir.glob(static_presence_pattern))
    motion_files = list(motion_dir.glob(motion_pattern))
    
    chip_desc = f"{chip} ({num_sc} SC)" if chip else f"{num_sc} SC"
    
    if not static_presence_files:
        raise FileNotFoundError(
            f"No static-presence file found for {chip_desc} in {static_presence_dir}\n"
            f"Collect data using: ./espectre collect --label static_presence --duration 10"
        )
    if not motion_files:
        raise FileNotFoundError(
            f"No motion file found for {chip_desc} in {motion_dir}\n"
            f"Collect data using: ./espectre collect --label motion --duration 10"
        )
    
    # Prefer nearest static-presence/motion pair from dataset_info metadata, so
    # Python tests match C++ csi_test_data.h pairing policy.
    static_presence_file = None
    motion_file = None
    try:
        info = load_dataset_info()
        files_section = info.get('files', {})
        static_presence_meta = files_section.get('static_presence', [])
        motion_meta = files_section.get('motion', [])

        def _meta_matches(entry: Dict[str, Any], label_chip: Optional[str]) -> bool:
            if int(entry.get('subcarriers', 0)) != int(num_sc):
                return False
            if label_chip is None:
                return True
            return str(entry.get('chip', '')).upper() == label_chip.upper()

        def _parse_ts(value: Any) -> Optional[datetime]:
            if not value:
                return None
            try:
                # Supports both naive and timezone-aware ISO strings.
                return datetime.fromisoformat(str(value))
            except ValueError:
                return None

        selected_chip = chip.upper() if chip else None
        static_presence_candidates = []
        motion_candidates = []
        for entry in static_presence_meta:
            if _meta_matches(entry, selected_chip):
                ts = _parse_ts(entry.get('collected_at'))
                filename = entry.get('filename')
                if ts and filename:
                    candidate = static_presence_dir / str(filename)
                    if candidate.exists():
                        static_presence_candidates.append((ts, candidate))
        for entry in motion_meta:
            if _meta_matches(entry, selected_chip):
                ts = _parse_ts(entry.get('collected_at'))
                filename = entry.get('filename')
                if ts and filename:
                    candidate = motion_dir / str(filename)
                    if candidate.exists():
                        motion_candidates.append((ts, candidate))

        best_delta = None
        for b_ts, b_path in static_presence_candidates:
            for m_ts, m_path in motion_candidates:
                delta = abs((m_ts - b_ts).total_seconds())
                if best_delta is None or delta < best_delta:
                    best_delta = delta
                    static_presence_file = b_path
                    motion_file = m_path
    except Exception:
        # Keep backward-compatible fallback below.
        static_presence_file = None
        motion_file = None

    # Fallback: use the most recent files by filename timestamp.
    if static_presence_file is None or motion_file is None:
        static_presence_file = sorted(static_presence_files)[-1]
        motion_file = sorted(motion_files)[-1]
    
    # Extract chip name from filename (e.g., static_presence_c6_64sc_... -> C6).
    parts = static_presence_file.stem.split('_')
    chip_name = parts[2].upper() if len(parts) >= 3 else 'UNKNOWN'
    
    return static_presence_file, motion_file, chip_name


def load_static_presence_and_motion(
    static_presence_file: str = None,
    motion_file: str = None,
    chip: str = 'C6'
) -> Tuple[List[Dict], List[Dict]]:
    """
    Load static-presence and motion data from .npz files.
    
    Args:
        static_presence_file: Path to static-presence data file (optional, auto-finds if not specified)
        motion_file: Path to motion data file (optional, auto-finds if not specified)
        chip: Chip type for auto-discovery (default: C6)
    
    Returns:
        tuple: (static_presence_packets, motion_packets)
    """
    # Auto-find files if not specified
    if static_presence_file is None or motion_file is None:
        found_static_presence, found_motion, _ = find_static_presence_motion_dataset(chip=chip)
        if static_presence_file is None:
            static_presence_file = found_static_presence
        if motion_file is None:
            motion_file = found_motion
    
    # Convert to Path if string
    static_presence_path = Path(static_presence_file) if isinstance(static_presence_file, str) else static_presence_file
    motion_path = Path(motion_file) if isinstance(motion_file, str) else motion_file
    
    if not static_presence_path.exists():
        raise FileNotFoundError(
            f"{static_presence_path} not found.\n"
            f"Collect data using: ./espectre collect --label static_presence --duration 10"
        )
    if not motion_path.exists():
        raise FileNotFoundError(
            f"{motion_path} not found.\n"
            f"Collect data using: ./espectre collect --label motion --duration 10"
        )
    
    static_presence_packets = load_npz_as_packets(static_presence_path)
    motion_packets = load_npz_as_packets(motion_path)
    
    return static_presence_packets, motion_packets


# ============================================================================
# MVS Detection - Uses src/segmentation.py (single source of truth)
# ============================================================================

# Add src directory to path for imports
import sys as _sys
_src_path = str(python_src_dir())
if _src_path not in _sys.path:
    _sys.path.append(_src_path)

# Add repo root to path so transition helpers remain importable during reorg
_repo_root = str(repo_root())
if _repo_root not in _sys.path:
    _sys.path.insert(0, _repo_root)

# Import SegmentationContext from src/segmentation.py
from segmentation import SegmentationContext

# Import filters from src/filters.py (for scripts that need it directly)
from filters import HampelFilter

# Import feature calculation functions from src/features.py
from features import calc_skewness

# Import detectors from src (IDetector interface and implementations)
from detector_interface import IDetector, MotionState
from mvs_detector import MVSDetector as MVSDetectorNew


# ============================================================================
# Utility Functions (delegate to SegmentationContext static methods)
# ============================================================================

def calculate_spatial_turbulence(csi_data,
                                 selected_subcarriers=None) -> float:
    """
    Calculate spatial turbulence from CSI data using the normalized AGC-active path.
    
    Args:
        csi_data: CSI data array (I/Q pairs)
        selected_subcarriers: Optional explicit subcarrier list. When omitted,
            the fixed production defaults from config.DEFAULT_SUBCARRIERS are used.
    Returns:
        float: Spatial turbulence value
    """
    band = config.DEFAULT_SUBCARRIERS if selected_subcarriers is None else selected_subcarriers
    turbulence, _ = SegmentationContext.compute_spatial_turbulence(
        csi_data, band
    )
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
                 selected_subcarriers=None,
                 track_data: bool = False,
                 enable_hampel: bool = True, hampel_window: int = config.HAMPEL_WINDOW,
                 hampel_threshold: float = config.HAMPEL_THRESHOLD,
                 enable_lowpass: bool = False, lowpass_cutoff: float = 11.0):
        """
        Initialize MVS detector
        
        Args:
            window_size: Size of the sliding window for variance calculation
            threshold: Threshold for motion detection
            selected_subcarriers: Optional explicit subcarrier band to use
            track_data: If True, track moving variance and state history
            enable_hampel: Enable Hampel filter for outlier removal
            hampel_window: Hampel filter window size
            hampel_threshold: Hampel filter MAD threshold
            enable_lowpass: Enable low-pass filter for noise reduction
            lowpass_cutoff: Low-pass filter cutoff frequency in Hz
        """
        self.window_size = window_size
        self.threshold = threshold
        self.fixed_subcarriers = (
            config.DEFAULT_SUBCARRIERS if selected_subcarriers is None else list(selected_subcarriers)
        )
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
    
    def process_packet(self, packet_or_csi):
        """
        Process a single CSI packet
        
        Args:
            packet_or_csi: Either packet dict with {'csi_data'} or CSI array
        """
        if isinstance(packet_or_csi, dict):
            csi_data = packet_or_csi['csi_data']
        else:
            csi_data = packet_or_csi
        
        # Calculate turbulence using SegmentationContext method
        turb = self._context.calculate_spatial_turbulence(csi_data, self.fixed_subcarriers)
        
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


def test_mvs_configuration(static_presence_packets, motion_packets,
                          threshold, window_size) -> Tuple[int, int, float]:
    """
    Test MVS configuration and return FP, TP counts
    
    Args:
        static_presence_packets: List of static-presence packets
        motion_packets: List of motion packets
        threshold: Motion detection threshold
        window_size: Sliding window size
    
    Returns:
        tuple: (fp, tp, score)
    """
    num_static_presence = len(static_presence_packets)
    num_motion = len(motion_packets)

    # Test on static presence (FP)
    detector = MVSDetector(window_size, threshold)
    for pkt in static_presence_packets:
        detector.process_packet(pkt)
    fp = detector.get_motion_count()

    # Keep the turbulence buffer warm across static_presence -> motion to
    # match real performance tests and runtime behavior. Reset only motion counter.
    detector.motion_packet_count = 0

    # Test on motion (TP)
    for pkt in motion_packets:
        detector.process_packet(pkt)
    tp = detector.get_motion_count()

    fn = max(0, num_motion - tp)
    recall = (tp / num_motion * 100.0) if num_motion > 0 else 0.0
    precision = (tp / (tp + fp) * 100.0) if (tp + fp) > 0 else 0.0
    fp_rate = (fp / num_static_presence * 100.0) if num_static_presence > 0 else 100.0
    f1_score = 0.0
    if (precision + recall) > 0.0:
        f1_score = 2.0 * precision * recall / (precision + recall)

    # Match performance objectives:
    # - primary: satisfy recall/FP constraints
    # - secondary: maximize F1 among valid candidates
    recall_target = 95.0
    fp_target = 10.0
    fn_rate = (fn / num_motion * 100.0) if num_motion > 0 else 100.0

    if recall >= recall_target and fp_rate <= fp_target:
        score = 1_000_000.0 + f1_score * 100.0 - fp_rate
    elif recall >= recall_target:
        score = 100_000.0 - (fp_rate - fp_target) * 1_000.0 + f1_score * 10.0
    else:
        score = (
            -1_000_000.0
            - (recall_target - recall) * 2_000.0
            - fn_rate * 200.0
            - fp_rate * 20.0
            + precision
        )

    return fp, tp, score
