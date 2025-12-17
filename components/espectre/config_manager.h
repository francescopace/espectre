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
 * Stores runtime-configurable parameters for ESPectre.
 * Persisted to flash using ESPHome preferences.
 * 
 * Currently only segmentation_threshold is persisted, as it's the only
 * parameter adjustable at runtime via the Home Assistant slider.
 * 
 * All other settings come from YAML (compile-time) or are recalculated
 * at boot (e.g., normalization_scale during NBVI calibration).
 * 
 * Note: Changes to this struct require updating the preference hash
 * in espectre.cpp to avoid loading stale data.
 */
struct ESpectreConfig {
  float segmentation_threshold;     // Motion detection threshold (adjustable via HA)
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
