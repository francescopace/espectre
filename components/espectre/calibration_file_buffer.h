/*
 * ESPectre - Calibration File Buffer
 * 
 * File-based buffer for calibration data storage on SPIFFS.
 * Used by NBVI calibrator for file-based calibration data storage.
 * 
 * Stores CSI magnitude data as uint8 (max CSI magnitude ~181 fits in 1 byte).
 * Uses SPIFFS to avoid RAM limitations on ESP32.
 * 
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * License: GPLv3
 */

#pragma once

#include "base_detector.h"  // For CALIBRATION_DEFAULT_BUFFER_SIZE
#include "utils.h"
#include <cstdint>
#include <cstdio>
#include <vector>

namespace esphome {
namespace espectre {

/**
 * Calibration File Buffer
 * 
 * Manages file-based storage of CSI magnitude data during calibration.
 * Handles SPIFFS mounting, file I/O, progress tracking, and data retrieval.
 * 
 * Used by NBVICalibrator via composition.
 */
class CalibrationFileBuffer {
 public:
  /**
   * Initialize the file buffer
   * 
   * @param buffer_path Path for the calibration data file
   * @param buffer_size Number of packets to collect
   */
  void init(const char* buffer_path, uint16_t buffer_size = CALIBRATION_DEFAULT_BUFFER_SIZE);
  
  /**
   * Reset buffer state for a new calibration cycle
   */
  void reset();
  
  /**
   * Remove old file and open a new one for writing
   * 
   * @return true if file opened successfully
   */
  bool open_for_writing();
  
  /**
   * Open buffer file for reading
   * 
   * @return true if file opened successfully
   */
  bool open_for_reading();
  
  /**
   * Close the buffer file
   */
  void close();
  
  /**
   * Remove the buffer file
   * 
   * Tries remove() first, falls back to truncation if SPIFFS
   * doesn't support file deletion.
   */
  void remove_file();
  
  /**
   * Write a CSI packet to the buffer file as uint8 magnitudes
   * 
   * Validates HT20 format (64 subcarriers), calculates magnitudes
   * from I/Q data, writes to file, and logs progress.
   * 
   * @param csi_data Raw CSI data (I/Q pairs)
   * @param csi_len Length of CSI data in bytes (expected: 128 for HT20)
   * @return true if buffer is now full
   */
  bool write_packet(const int8_t* csi_data, size_t csi_len);
  
  /**
   * Read packet data from the buffer file
   * 
   * @param start_idx Starting packet index
   * @param window_size Number of packets to read
   * @return Vector of magnitude data (window_size * HT20_NUM_SUBCARRIERS bytes)
   */
  std::vector<uint8_t> read_window(uint16_t start_idx, uint16_t window_size);
  
  // State accessors
  uint16_t get_count() const { return buffer_count_; }
  uint16_t get_size() const { return buffer_size_; }
  void set_size(uint16_t size) { buffer_size_ = size; }
  bool is_full() const { return buffer_count_ >= buffer_size_; }
  bool is_open() const { return buffer_file_ != nullptr; }
  
 private:
  bool ensure_spiffs_mounted_();
  
  FILE* buffer_file_{nullptr};
  uint16_t buffer_count_{0};
  uint16_t buffer_size_{CALIBRATION_DEFAULT_BUFFER_SIZE};
  const char* buffer_path_{nullptr};
  uint8_t last_progress_{0};
};

}  // namespace espectre
}  // namespace esphome
