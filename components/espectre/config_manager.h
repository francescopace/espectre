/*
 * ESPectre - Configuration Manager
 * 
 * Manages persistent configuration storage using ESPHome preferences.
 * Handles loading, saving, and validation of configuration parameters.
 * 
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * License: GPLv3
 */

#pragma once

#include "esphome/core/preferences.h"
#include <cstdint>

namespace esphome {
namespace espectre {

/**
 * Configuration structure
 * 
 * Stores all configurable parameters for ESPectre.
 * Persisted to flash using ESPHome preferences.
 * Fields ordered by size to minimize struct padding.
 * 
 * Note: Changes to this struct require updating the preference hash
 * in espectre.cpp to avoid loading stale data.
 */
struct ESpectreConfig {
  // 4-byte fields first
  float segmentation_threshold;
  float hampel_threshold;
  float normalization_scale;        // CSI amplitude normalization factor (default: 1.0)
  uint32_t traffic_generator_rate;
  // 2-byte fields
  uint16_t segmentation_window_size;
  // 1-byte fields last
  uint8_t hampel_window;
  bool hampel_enabled;
};

/**
 * Configuration Manager
 * 
 * Manages persistent configuration storage.
 * Provides load/save operations with validation.
 */
class ConfigurationManager {
 public:
  /**
   * Initialize configuration manager
   * 
   * @param pref ESPHome preference object
   */
  void init(ESPPreferenceObject pref) { pref_ = pref; }
  
  /**
   * Load configuration from preferences
   * 
   * @param config Output configuration structure
   * @return true if loaded successfully, false if no saved config
   */
  bool load(ESpectreConfig& config);
  
  /**
   * Save configuration to preferences
   * 
   * @param config Configuration to save
   */
  void save(const ESpectreConfig& config);
  
 private:
  ESPPreferenceObject pref_;
};

}  // namespace espectre
}  // namespace esphome
