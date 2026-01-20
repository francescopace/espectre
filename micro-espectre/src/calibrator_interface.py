"""
Calibrator Interface - Abstract Base Class for Calibration Algorithms

Defines the common interface for subcarrier selection algorithms.
Allows polymorphic use of different calibration strategies (NBVI, P95).

Author: Francesco Pace <francesco.pace@gmail.com>
License: GPLv3
"""

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
