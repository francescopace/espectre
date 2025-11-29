#pragma once

// Mock ESPHome Sensor for PlatformIO tests

#include <string>
#include <cstdint>

namespace esphome {
namespace sensor {

// Mock Sensor class
class Sensor {
public:
    void publish_state(float state) {}
    
    void set_name(const std::string& name) {}
    std::string get_name() const { return ""; }
    
    void set_unit_of_measurement(const std::string& unit) {}
    void set_icon(const std::string& icon) {}
    void set_accuracy_decimals(uint8_t decimals) {}
    
    void set_device_class(const std::string& device_class) {}
    void set_state_class(const std::string& state_class) {}
    
    float get_state() const { return 0.0f; }
    bool has_state() const { return false; }
    
    void add_on_state_callback(void (*callback)(float)) {}
};

} // namespace sensor
} // namespace esphome
