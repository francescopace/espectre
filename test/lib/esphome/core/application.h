#pragma once

// Mock ESPHome Application for PlatformIO tests

#include "component.h"
#include <vector>

namespace esphome {

// Mock App class
class App {
public:
    void register_component(Component* comp) {}
    void setup() {}
    void loop() {}
    
    uint32_t get_loop_component_start_time() const { return 0; }
    
    static App& get_instance() {
        static App instance;
        return instance;
    }
};

// Global App instance
extern App App;

} // namespace esphome
