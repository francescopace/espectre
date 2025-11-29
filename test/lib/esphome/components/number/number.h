#pragma once

// Mock ESPHome Number for PlatformIO tests

#include <string>
#include <cstdint>

// Mock LOG_NUMBER macro
#define LOG_NUMBER(tag, name, obj) do {} while(0)

namespace esphome {
namespace number {

// Mock Number class
class Number {
public:
    void publish_state(float state) { state_ = state; }
    
    void set_name(const std::string& name) {}
    std::string get_name() const { return ""; }
    
    void set_unit_of_measurement(const std::string& unit) {}
    void set_icon(const std::string& icon) {}
    
    float get_state() const { return state_; }
    bool has_state() const { return true; }
    
    void set_min_value(float min) { min_ = min; }
    void set_max_value(float max) { max_ = max; }
    void set_step(float step) { step_ = step; }
    
    float get_min_value() const { return min_; }
    float get_max_value() const { return max_; }
    float get_step() const { return step_; }

protected:
    virtual void control(float value) { state_ = value; }
    
    float state_{0.0f};
    float min_{0.0f};
    float max_{100.0f};
    float step_{1.0f};
};

// Mock NumberTraits
class NumberTraits {
public:
    void set_min_value(float min) {}
    void set_max_value(float max) {}
    void set_step(float step) {}
};

} // namespace number
} // namespace esphome

