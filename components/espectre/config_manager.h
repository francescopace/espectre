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
 */
struct ESpectreConfig {
  float segmentation_threshold;
  uint16_t segmentation_window_size;
  uint32_t traffic_generator_rate;
  bool hampel_enabled;
  uint8_t hampel_window;
  float hampel_threshold;
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
