#pragma once

// Mock ESPHome BinarySensor for PlatformIO tests

#include <string>

namespace esphome {
namespace binary_sensor {

// Mock BinarySensor class
class BinarySensor {
public:
    void publish_state(bool state) {
        state_ = state;
        has_state_ = true;
        publish_count_++;
    }
    
    void set_name(const std::string& name) {}
    std::string get_name() const { return ""; }
    
    void set_device_class(const std::string& device_class) {}
    
    bool get_state() const { return state_; }
    bool has_state() const { return has_state_; }
    unsigned int get_publish_count() const { return publish_count_; }
    
    void add_on_state_callback(void (*callback)(bool)) {}

private:
    bool state_{false};
    bool has_state_{false};
    unsigned int publish_count_{0};
};

} // namespace binary_sensor
} // namespace esphome
