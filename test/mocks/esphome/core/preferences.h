#pragma once

// Mock ESPHome Preferences for PlatformIO tests

#include <string>
#include <cstddef>
#include <cstdint>

namespace esphome {

// Mock preference object
class ESPPreferenceObject {
public:
    bool save(const void* data, size_t len) { return true; }
    bool load(void* data, size_t len) { return false; }
    
    // Template methods for convenience
    template<typename T>
    bool save(const T* data) {
        return save(data, sizeof(T));
    }
    
    template<typename T>
    bool load(T* data) {
        return load(data, sizeof(T));
    }
};

// Mock preferences manager
class ESPPreferences {
public:
    ESPPreferenceObject make_preference(const std::string& key) {
        return ESPPreferenceObject();
    }
    
    ESPPreferenceObject make_preference(const std::string& key, bool has_hash) {
        return ESPPreferenceObject();
    }
    
    template<typename T>
    ESPPreferenceObject make_preference(uint32_t hash) {
        return ESPPreferenceObject();
    }
    
    bool sync() { return true; }
    bool reset() { return true; }
};

// FNV-1 hash function (mock)
inline uint32_t fnv1_hash(const std::string& str) {
    uint32_t hash = 2166136261UL;
    for (char c : str) {
        hash ^= c;
        hash *= 16777619UL;
    }
    return hash;
}

// Global preferences instance (mock)
extern ESPPreferences* global_preferences;

} // namespace esphome
