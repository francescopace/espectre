"""
Calibrator Interface and Base Class for Calibration Algorithms

Defines the common interface and shared file-based storage logic
for subcarrier selection algorithms (NBVI, P95).

Author: Francesco Pace <francesco.pace@gmail.com>
License: GPLv3
"""

import gc
import os

# Note: MicroPython doesn't have abc module, so we use a simple base class
# with NotImplementedError for abstract methods


class ICalibrator:
    """
    Abstract interface for calibration algorithms.
    
    All calibrators must implement this interface to be used
    interchangeably in the main application.
    """
    
    def add_packet(self, csi_data):
        """
        Add a CSI packet to the calibration buffer.
        
        Args:
            csi_data: Raw CSI data (I/Q pairs as bytes or bytearray)
        
        Returns:
            int: Number of packets added (0 if rejected, 1 if added)
        """
        raise NotImplementedError("Subclasses must implement add_packet()")
    
    def calibrate(self):
        """
        Run the calibration algorithm.
        
        Returns:
            tuple: (selected_band, mv_values)
                - selected_band: List of selected subcarrier indices, or None on failure
                - mv_values: List of moving variance values for threshold calculation
        """
        raise NotImplementedError("Subclasses must implement calibrate()")
    
    def free_buffer(self):
        """
        Free the calibration buffer and clean up resources.
        """
        raise NotImplementedError("Subclasses must implement free_buffer()")
    
    def get_packet_count(self):
        """
        Get the number of packets currently in the buffer.
        
        Returns:
            int: Number of packets collected
        """
        raise NotImplementedError("Subclasses must implement get_packet_count()")
    
    def is_buffer_full(self):
        """
        Check if the buffer has collected enough packets.
        
        Returns:
            bool: True if buffer is full and ready for calibration
        """
        raise NotImplementedError("Subclasses must implement is_buffer_full()")


class BaseCalibrator(ICalibrator):
    """
    Base calibrator with shared file-based storage logic.
    
    Handles common initialization, file I/O, packet counting,
    and cleanup. Subclasses implement add_packet() and calibrate().
    """
    
    def __init__(self, buffer_size, buffer_file):
        """
        Initialize base calibrator.
        
        Args:
            buffer_size: Number of packets to collect
            buffer_file: Path to the file-based buffer
        """
        self.buffer_size = buffer_size
        self._buffer_file = buffer_file
        self._packet_count = 0
        self._filtered_count = 0
        self._file = None
        self._initialized = False
        
        # Remove old buffer file if exists
        try:
            os.remove(buffer_file)
        except OSError:
            pass
        
        # Open file for writing
        self._file = open(buffer_file, 'wb')
    
    def _prepare_for_reading(self):
        """Close write mode and reopen for reading."""
        if self._file:
            self._file.flush()
            self._file.close()
            gc.collect()
        self._file = open(self._buffer_file, 'rb')
    
    def free_buffer(self):
        """Free resources after calibration is complete."""
        if self._file:
            self._file.close()
            self._file = None
        
        try:
            os.remove(self._buffer_file)
        except OSError:
            pass
    
    def get_packet_count(self):
        """Get the number of packets currently in the buffer."""
        return self._packet_count
    
    def is_buffer_full(self):
        """Check if the buffer has collected enough packets."""
        return self._packet_count >= self.buffer_size
