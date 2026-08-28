/*
 * ESPectre - Mock preferences.h
 *
 * Host-side mock of preferences.h for native C++ tests.
 *
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * SPDX-License-Identifier: GPL-3.0-only
 * Commercial licensing available under separate agreement; see LICENSING.md.
 */
#pragma once

// Mock ESPHome Preferences for host tests

#include <cstring>
#include <map>
#include <string>
#include <cstddef>
#include <cstdint>

namespace esphome {

inline std::map<uint32_t, std::string> g_esphome_preference_store;
inline bool g_esphome_preference_save_success = true;

inline void reset_preference_store() {
    g_esphome_preference_store.clear();
    g_esphome_preference_save_success = true;
}

inline uint32_t fnv1_hash(const std::string& str) {
    uint32_t hash = 2166136261UL;
    for (char c : str) {
        hash ^= static_cast<unsigned char>(c);
        hash *= 16777619UL;
    }
    return hash;
}

class ESPPreferenceObject {
public:
    uint32_t hash{0};

    bool save(const void* data, size_t len) {
        if (!g_esphome_preference_save_success) {
            return false;
        }
        g_esphome_preference_store[hash].assign(static_cast<const char *>(data), len);
        return true;
    }
    bool load(void* data, size_t len) {
        const auto it = g_esphome_preference_store.find(hash);
        if (it == g_esphome_preference_store.end() || it->second.size() != len) {
            return false;
        }
        std::memcpy(data, it->second.data(), len);
        return true;
    }

    template<typename T>
    bool save(const T* data) {
        return save(data, sizeof(T));
    }

    template<typename T>
    bool load(T* data) {
        return load(data, sizeof(T));
    }
};

class ESPPreferences {
public:
    ESPPreferenceObject make_preference(const std::string& key) {
        ESPPreferenceObject object;
        object.hash = fnv1_hash(key);
        return object;
    }

    ESPPreferenceObject make_preference(const std::string& key, bool has_hash) {
        (void) has_hash;
        return make_preference(key);
    }

    template<typename T>
    ESPPreferenceObject make_preference(uint32_t hash) {
        ESPPreferenceObject object;
        object.hash = hash;
        return object;
    }

    bool sync() { return true; }
    bool reset() { return true; }
};

inline ESPPreferences* global_preferences = nullptr;

} // namespace esphome
