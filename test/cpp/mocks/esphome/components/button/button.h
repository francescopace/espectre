#pragma once

// Mock ESPHome Button for PlatformIO tests

#include <string>

// Mock LOG_BUTTON macro
#define LOG_BUTTON(tag, name, obj) do {} while(0)

namespace esphome {
namespace button {

// Mock Button class
class Button {
public:
    void set_name(const std::string& name) {}
    std::string get_name() const { return ""; }
    
    void set_icon(const std::string& icon) {}
    
    // Trigger press action
    void press() { press_action(); }

protected:
    virtual void press_action() {}
};

} // namespace button
} // namespace esphome

