#pragma once

// Mock ESPHome Sensor for PlatformIO tests

#include <string>
#include <cstdint>

namespace esphome {
namespace sensor {

// Mock Sensor class
class Sensor {
public:
    void publish_state(float state) {
        state_ = state;
        has_state_ = true;
        publish_count_++;
    }
    
    void set_name(const std::string& name) {}
    std::string get_name() const { return ""; }
    
    void set_unit_of_measurement(const std::string& unit) {}
    void set_icon(const std::string& icon) {}
    void set_accuracy_decimals(uint8_t decimals) {}
    
    void set_device_class(const std::string& device_class) {}
    void set_state_class(const std::string& state_class) {}
    
    float get_state() const { return state_; }
    bool has_state() const { return has_state_; }
    unsigned int get_publish_count() const { return publish_count_; }
    
    void add_on_state_callback(void (*callback)(float)) {}

private:
    float state_{0.0f};
    bool has_state_{false};
    unsigned int publish_count_{0};
};

} // namespace sensor
} // namespace esphome
