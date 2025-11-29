#pragma once

// Mock ESPHome BinarySensor for PlatformIO tests

#include <string>

namespace esphome {
namespace binary_sensor {

// Mock BinarySensor class
class BinarySensor {
public:
    void publish_state(bool state) {}
    
    void set_name(const std::string& name) {}
    std::string get_name() const { return ""; }
    
    void set_device_class(const std::string& device_class) {}
    
    bool get_state() const { return false; }
    bool has_state() const { return false; }
    
    void add_on_state_callback(void (*callback)(bool)) {}
};

} // namespace binary_sensor
} // namespace esphome
