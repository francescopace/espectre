/*
 * ESPectre - Base Calibrator
 * 
 * Abstract base class implementing the common calibration lifecycle.
 * Uses Template Method pattern: subclasses override run_calibration_()
 * to provide their specific calibration algorithm.
 * 
 * Common lifecycle (handled by this class):
 * 1. start_calibration() - validate, open file, set calibration mode
 * 2. add_packet() - write CSI to file via CalibrationFileBuffer
 * 3. on_collection_complete_() - close file, launch processing task
 * 4. calibration_task_() - open for reading, run algorithm, cleanup
 * 5. finish_calibration_() - notify callback with results
 * 
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * License: GPLv3
 */

#pragma once

#include "calibrator_interface.h"
#include "calibration_file_buffer.h"
#include "utils.h"
#include "esp_err.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include <cstddef>
#include <cstdint>
#include <functional>
#include <vector>

namespace esphome {
namespace espectre {

// Forward declarations
class CSIManager;

/**
 * Base Calibrator
 * 
 * Implements common calibration lifecycle shared by all calibrators.
 * Subclasses provide the specific calibration algorithm via run_calibration_().
 */
class BaseCalibrator : public ICalibrator {
 public:
  using result_callback_t = ICalibrator::result_callback_t;
  using collection_complete_callback_t = ICalibrator::collection_complete_callback_t;
  
  /**
   * Initialize calibrator with custom buffer path
   * 
   * @param csi_manager CSI manager instance
   * @param buffer_path Path for temporary calibration file
   */
  void init(CSIManager* csi_manager, const char* buffer_path);
  
  /**
   * Start automatic calibration
   * 
   * Common lifecycle: validate -> open file -> set calibration mode.
   * 
   * @param current_band Current subcarrier selection (for baseline/fallback)
   * @param current_band_size Size of current band
   * @param callback Callback to invoke with results
   * @return ESP_OK on success
   */
  esp_err_t start_calibration(const uint8_t* current_band,
                              uint8_t current_band_size,
                              result_callback_t callback) override;
  
  /**
   * Add CSI packet to calibration buffer
   * 
   * Delegates to CalibrationFileBuffer. Triggers processing when full.
   * 
   * @param csi_data Raw CSI data (I/Q pairs)
   * @param csi_len Length of CSI data
   * @return true if buffer is full
   */
  bool add_packet(const int8_t* csi_data, size_t csi_len) override;
  
  bool is_calibrating() const override { return calibrating_; }
  
  void set_collection_complete_callback(collection_complete_callback_t callback) override {
    collection_complete_callback_ = callback;
  }
  
  void set_buffer_size(uint16_t size) { file_buffer_.set_size(size); }
  uint16_t get_buffer_size() const { return file_buffer_.get_size(); }
  
 protected:
  // ======================================================================
  // Template methods - must be overridden by subclasses
  // ======================================================================
  
  /** Return the log TAG for this calibrator (e.g. "NBVICalibrator") */
  virtual const char* get_tag_() const = 0;
  
  /** Return the FreeRTOS task name (max 16 chars) */
  virtual const char* get_task_name_() const = 0;
  
  /** Run the calibration algorithm. File buffer is open for reading. */
  virtual esp_err_t run_calibration_() = 0;
  
  // ======================================================================
  // Shared state accessible to subclasses
  // ======================================================================
  
  CSIManager* csi_manager_{nullptr};
  CalibrationFileBuffer file_buffer_;
  std::vector<uint8_t> current_band_;
  
  // Results (set by subclass in run_calibration_)
  uint8_t selected_band_[12]{};
  uint8_t selected_band_size_{0};
  std::vector<float> mv_values_;
  
  // Shared constants
  static constexpr uint16_t MVS_WINDOW_SIZE = 50;
  static constexpr float MVS_THRESHOLD = 1.0f;
  static constexpr float NULL_SUBCARRIER_THRESHOLD = 1.0f;
  
 private:
  void on_collection_complete_();
  static void calibration_task_wrapper_(void* arg);
  void finish_calibration_(bool success);
  
  bool calibrating_{false};
  result_callback_t result_callback_;
  collection_complete_callback_t collection_complete_callback_;
  TaskHandle_t calibration_task_handle_{nullptr};
};

}  // namespace espectre
}  // namespace esphome
