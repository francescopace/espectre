#pragma once

// Mock ESPHome Component for PlatformIO tests

#include <cstdint>

namespace esphome {

// Setup priority constants
namespace setup_priority {
    constexpr float HARDWARE = 1000.0f;
    constexpr float DATA = 800.0f;
    constexpr float LATE = 600.0f;
    constexpr float AFTER_WIFI = 250.0f;
    constexpr float AFTER_CONNECTION = 100.0f;
    constexpr float PROCESSOR = -100.0f;
}

// Mock Component base class
class Component {
public:
    virtual ~Component() = default;
    
    virtual void setup() {}
    virtual void loop() {}
    virtual void dump_config() {}
    
    virtual float get_setup_priority() const { return 0.0f; }
    virtual float get_loop_priority() const { return 0.0f; }
    
    void set_setup_priority(float priority) {}
    
    bool is_failed() const { return false; }
    void mark_failed() {}
    
    void set_timeout(uint32_t timeout, void (*func)()) {}
    void set_interval(uint32_t interval, void (*func)()) {}
};

// Mock PollingComponent
class PollingComponent : public Component {
public:
    virtual void update() {}
    
    void set_update_interval(uint32_t update_interval) {}
    uint32_t get_update_interval() const { return 0; }
};

} // namespace esphome
