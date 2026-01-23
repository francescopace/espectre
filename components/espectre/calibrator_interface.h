/*
 * ESPectre - Calibrator Interface
 * 
 * Abstract interface for calibration algorithms.
 * Allows polymorphic use of different calibration strategies (NBVI, P95, PCA).
 * 
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * License: GPLv3
 */

#pragma once

#include "esp_err.h"
#include <cstddef>
#include <cstdint>
#include <functional>
#include <vector>

namespace esphome {
namespace espectre {

// Forward declarations
class CSIManager;

/**
 * Abstract interface for calibration algorithms
 * 
 * All calibrators must implement this interface to be used
 * interchangeably in the ESPectre component.
 */
class ICalibrator {
 public:
  // Callback type for calibration results
  // Parameters: band, size, cal_values (algorithm-specific baseline values), success
  // For MVS (NBVI/P95): cal_values = moving variance values
  // For PCA: cal_values = correlation values
  // Adaptive threshold is calculated externally using threshold.h
  using result_callback_t = std::function<void(const uint8_t* band, uint8_t size, 
                                               const std::vector<float>& cal_values, bool success)>;
  
  // Callback type for collection complete notification
  // Called when all packets have been collected, before processing starts.
  using collection_complete_callback_t = std::function<void()>;
  
  virtual ~ICalibrator() = default;
  
  /**
   * Initialize calibrator
   * 
   * @param csi_manager CSI manager instance
   */
  virtual void init(CSIManager* csi_manager) = 0;
  
  /**
   * Set callback for collection complete notification
   * Called when packet collection is done, before processing starts.
   * 
   * @param callback Callback function
   */
  virtual void set_collection_complete_callback(collection_complete_callback_t callback) = 0;
  
  /**
   * Start calibration process
   * 
   * @param current_band Current/fallback subcarrier selection
   * @param current_band_size Size of current band
   * @param callback Callback to invoke with results
   * @return ESP_OK on success
   */
  virtual esp_err_t start_calibration(const uint8_t* current_band,
                                      uint8_t current_band_size,
                                      result_callback_t callback) = 0;
  
  /**
   * Add CSI packet to calibration buffer
   * 
   * @param csi_data Raw CSI data (I/Q pairs)
   * @param csi_len Length of CSI data
   * @return true if buffer is full and calibration should proceed
   */
  virtual bool add_packet(const int8_t* csi_data, size_t csi_len) = 0;
  
  /**
   * Check if calibration is in progress
   * 
   * @return true if calibrating
   */
  virtual bool is_calibrating() const = 0;
  
};

}  // namespace espectre
}  // namespace esphome
