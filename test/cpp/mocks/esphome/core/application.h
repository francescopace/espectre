/*
 * ESPectre - Mock application.h
 *
 * Host-side mock of application.h for native C++ tests.
 *
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * SPDX-License-Identifier: GPL-3.0-only
 * Commercial licensing available under separate agreement; see LICENSING.md.
 */
#pragma once

// Mock ESPHome Application for host tests

#include "component.h"
#include <string>
#include <vector>

namespace esphome {

// Mock App class
class App {
public:
    void register_component(Component* comp) {}
    void setup() {}
    void loop() {}
    
    uint32_t get_loop_component_start_time() const { return 0; }
    std::string get_name() const { return "espectre"; }
    
    static App& get_instance() {
        static App instance;
        return instance;
    }
};

// Global App instance
inline App App;

} // namespace esphome
