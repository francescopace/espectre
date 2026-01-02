#pragma once

// Mock ESPHome Switch for PlatformIO tests

#include <string>

// Mock LOG_SWITCH macro
#define LOG_SWITCH(tag, name, obj) do {} while(0)

namespace esphome {
namespace switch_ {

// Mock Switch class
class Switch {
public:
    void set_name(const std::string& name) {}
    std::string get_name() const { return ""; }
    
    void set_icon(const std::string& icon) {}
    
    // Get current state
    bool state{false};
    
    // Publish state to Home Assistant
    void publish_state(bool new_state) { state = new_state; }
    
    // Turn on/off
    void turn_on() { write_state(true); }
    void turn_off() { write_state(false); }

protected:
    virtual void write_state(bool state) { this->state = state; }
};

} // namespace switch_
} // namespace esphome

