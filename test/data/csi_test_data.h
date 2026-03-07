/*
 * ESPectre - CSI Test Data Loader
 * 
 * Loads real CSI data from NPZ files for C++ tests using cnpy library.
 * Provides the same interface as the old static arrays for backward compatibility.
 * 
 * Usage:
 *   #include "csi_test_data.h"
 *   
 *   // In test setup:
 *   csi_test_data::load();
 *   
 *   // Access data (same interface as before):
 *   const int8_t** baseline_packets = csi_test_data::baseline_packets();
 *   const int8_t** movement_packets = csi_test_data::movement_packets();
 *   int num_baseline = csi_test_data::num_baseline();
 *   int num_movement = csi_test_data::num_movement();
 * 
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * License: GPLv3
 */

#ifndef CSI_TEST_DATA_H
#define CSI_TEST_DATA_H

// ============================================================================
// Test Data Files (HT20: 64 subcarriers only)
// ============================================================================
// C3 dataset
#define BASELINE_C3_64SC  "../micro-espectre/data/baseline/baseline_c3_64sc_20260214_185025.npz"
#define MOVEMENT_C3_64SC  "../micro-espectre/data/movement/movement_c3_64sc_20260214_185048.npz"
// C6 dataset (validated)
#define BASELINE_C6_64SC  "../micro-espectre/data/baseline/baseline_c6_64sc_20251212_142443.npz"
#define MOVEMENT_C6_64SC  "../micro-espectre/data/movement/movement_c6_64sc_20251212_142443.npz"
// ESP32 (Base) dataset - control set (excluded from ML training)
#define BASELINE_ESP32_64SC  "../micro-espectre/data/baseline/baseline_esp32_64sc_20260214_183059.npz"
#define MOVEMENT_ESP32_64SC  "../micro-espectre/data/movement/movement_esp32_64sc_20260214_183141.npz"
// S3 dataset
#define BASELINE_S3_64SC  "../micro-espectre/data/baseline/baseline_s3_64sc_20260117_222606.npz"
#define MOVEMENT_S3_64SC  "../micro-espectre/data/movement/movement_s3_64sc_20260117_222626.npz"

// Include cnpy implementation (with ZIP64 support)
#include "cnpy.cpp"

#include <vector>
#include <string>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <ctime>
#include <cmath>
#include <fstream>
#include <unordered_map>
#include <ArduinoJson.h>

namespace csi_test_data {

// ============================================================================
// NPZ Loading
// ============================================================================

/**
 * CSI data loaded from NPZ file
 */
struct CsiData {
    std::vector<std::vector<int8_t>> packets;  // [num_packets][packet_size]
    int num_packets;
    int packet_size;      // bytes per packet (num_subcarriers * 2)
    int num_subcarriers;
    bool gain_locked;      // From NPZ 'gain_locked' field; false if not present
    bool has_gain_locked;  // Whether 'gain_locked' was found in the NPZ
};

/**
 * Load CSI data from NPZ file
 */
inline CsiData load_npz(const std::string& filepath) {
    CsiData result;
    
    cnpy::npz_t npz = cnpy::npz_load(filepath);
    
    if (npz.find("csi_data") == npz.end()) {
        throw std::runtime_error("NPZ file missing 'csi_data' field: " + filepath);
    }
    
    cnpy::NpyArray& csi_arr = npz["csi_data"];
    
    if (csi_arr.shape.size() != 2) {
        throw std::runtime_error("csi_data should be 2D array");
    }
    
    result.num_packets = static_cast<int>(csi_arr.shape[0]);
    result.packet_size = static_cast<int>(csi_arr.shape[1]);
    result.num_subcarriers = result.packet_size / 2;
    
    // Load num_subcarriers if available
    if (npz.find("num_subcarriers") != npz.end()) {
        cnpy::NpyArray& ns_arr = npz["num_subcarriers"];
        if (ns_arr.word_size == 8) {
            result.num_subcarriers = static_cast<int>(*ns_arr.data<int64_t>());
        } else if (ns_arr.word_size == 4) {
            result.num_subcarriers = static_cast<int>(*ns_arr.data<int32_t>());
        }
    }

    // Load gain_locked if available (saved as numpy bool -> uint8/bool, word_size=1)
    result.gain_locked = false;
    result.has_gain_locked = false;
    if (npz.find("gain_locked") != npz.end()) {
        cnpy::NpyArray& gl_arr = npz["gain_locked"];
        if (gl_arr.word_size == 1) {
            result.gain_locked = (*gl_arr.data<uint8_t>() != 0);
            result.has_gain_locked = true;
        }
    }
    
    // Copy data into packets vector
    const int8_t* data = csi_arr.data<int8_t>();
    result.packets.resize(result.num_packets);
    
    for (int i = 0; i < result.num_packets; i++) {
        result.packets[i].resize(result.packet_size);
        for (int j = 0; j < result.packet_size; j++) {
            result.packets[i][j] = data[i * result.packet_size + j];
        }
    }
    
    return result;
}

/**
 * Build array of packet pointers for compatibility with existing tests
 */
inline std::vector<const int8_t*> get_packet_pointers(const CsiData& csi_data) {
    std::vector<const int8_t*> ptrs(csi_data.num_packets);
    for (int i = 0; i < csi_data.num_packets; i++) {
        ptrs[i] = csi_data.packets[i].data();
    }
    return ptrs;
}


// ============================================================================
// Dataset Configuration
// ============================================================================

enum class ChipType {
    C3,    // Uses forced subcarriers [20-31] - auto-calibration skipped per-test
    C6,
    ESP32, // Control set (excluded from ML training)
    S3
};

inline const char* chip_name(ChipType chip) {
    switch (chip) {
        case ChipType::C3: return "C3";
        case ChipType::C6: return "C6";
        case ChipType::ESP32: return "ESP32";
        case ChipType::S3: return "S3";
        default: return "Unknown";
    }
}

inline const char* baseline_file_for_chip(ChipType chip) {
    switch (chip) {
        case ChipType::C3: return BASELINE_C3_64SC;
        case ChipType::C6: return BASELINE_C6_64SC;
        case ChipType::ESP32: return BASELINE_ESP32_64SC;
        case ChipType::S3: return BASELINE_S3_64SC;
        default: return nullptr;
    }
}

inline const char* movement_file_for_chip(ChipType chip) {
    switch (chip) {
        case ChipType::C3: return MOVEMENT_C3_64SC;
        case ChipType::C6: return MOVEMENT_C6_64SC;
        case ChipType::ESP32: return MOVEMENT_ESP32_64SC;
        case ChipType::S3: return MOVEMENT_S3_64SC;
        default: return nullptr;
    }
}

/**
 * Check if a chip type should be skipped in tests.
 * Returns skip reason or nullptr if chip should run.
 * 
 * Note: C3 runs with forced subcarriers [20-31]. Only auto-calibration
 * tests are skipped per-test (not at chip level).
 */
inline const char* chip_skip_reason(ChipType chip) {
    switch (chip) {
        default: return nullptr;
    }
}

// ============================================================================
// Global Data Storage
// ============================================================================

// Skip first N packets from baseline to remove gain lock stabilization noise.
// These packets are recorded during radio warm-up and inflate calibration thresholds.
static constexpr int GAIN_LOCK_SKIP = 300;

static CsiData g_baseline_data;
static CsiData g_movement_data;
static std::vector<const int8_t*> g_baseline_ptrs;
static std::vector<const int8_t*> g_movement_ptrs;
static bool g_loaded = false;
static ChipType g_current_chip = ChipType::C6;
static bool g_tuning_cache_loaded = false;
struct BaselineTuningEntry {
    std::vector<uint8_t> subcarriers;
    std::string collected_at;
    std::string pair_movement_file;
};
static std::unordered_map<std::string, BaselineTuningEntry> g_tuning_by_baseline;
static std::unordered_map<std::string, std::string> g_movement_collected_at;

inline std::string filename_from_path(const char* filepath) {
    std::string path(filepath ? filepath : "");
    size_t slash = path.find_last_of("/\\");
    return (slash == std::string::npos) ? path : path.substr(slash + 1);
}

inline bool load_tuning_cache() {
    if (g_tuning_cache_loaded) {
        return true;
    }

    const std::string dataset_info_path = "../micro-espectre/data/dataset_info.json";
    std::ifstream in(dataset_info_path);
    if (!in.is_open()) {
        std::fprintf(stderr, "[CSI Test Data] ERROR: Cannot open %s\n", dataset_info_path.c_str());
        return false;
    }

    DynamicJsonDocument doc(128 * 1024);
    auto err = deserializeJson(doc, in);
    if (err) {
        std::fprintf(stderr, "[CSI Test Data] ERROR: Failed parsing dataset_info.json: %s\n", err.c_str());
        return false;
    }

    JsonArray baseline_entries = doc["files"]["baseline"].as<JsonArray>();
    for (JsonObject entry : baseline_entries) {
        const char* filename = entry["filename"];
        JsonArray arr = entry["optimal_subcarriers_gridsearch"].as<JsonArray>();
        if (filename == nullptr || arr.isNull() || arr.size() != 12) {
            continue;
        }

        std::vector<uint8_t> band;
        band.reserve(12);
        bool valid = true;
        for (JsonVariant sc : arr) {
            if (!sc.is<int>()) {
                valid = false;
                break;
            }
            int value = sc.as<int>();
            if (value < 0 || value > 255) {
                valid = false;
                break;
            }
            band.push_back(static_cast<uint8_t>(value));
        }
        if (valid && band.size() == 12) {
            BaselineTuningEntry tuning{};
            tuning.subcarriers = band;
            const char* collected_at = entry["collected_at"];
            const char* pair_movement = entry["optimal_pair_movement_file"];
            if (collected_at != nullptr) {
                tuning.collected_at = collected_at;
            }
            if (pair_movement != nullptr) {
                tuning.pair_movement_file = pair_movement;
            }
            g_tuning_by_baseline[filename] = tuning;
        }
    }

    JsonArray movement_entries = doc["files"]["movement"].as<JsonArray>();
    for (JsonObject entry : movement_entries) {
        const char* filename = entry["filename"];
        const char* collected_at = entry["collected_at"];
        if (filename != nullptr && collected_at != nullptr) {
            g_movement_collected_at[filename] = collected_at;
        }
    }

    g_tuning_cache_loaded = true;
    return true;
}

inline bool current_optimal_subcarriers(std::vector<uint8_t>& out_band) {
    if (!load_tuning_cache()) {
        return false;
    }
    const char* baseline_file = baseline_file_for_chip(g_current_chip);
    if (baseline_file == nullptr) {
        return false;
    }
    const std::string key = filename_from_path(baseline_file);
    auto it = g_tuning_by_baseline.find(key);
    if (it == g_tuning_by_baseline.end()) {
        return false;
    }
    out_band = it->second.subcarriers;
    return true;
}

/**
 * Remove first N packets from a CsiData struct (in-place).
 */
inline void skip_packets(CsiData& data, int skip) {
    if (skip <= 0 || skip >= data.num_packets) return;
    data.packets.erase(data.packets.begin(), data.packets.begin() + skip);
    data.num_packets = static_cast<int>(data.packets.size());
}

/**
 * Load CSI test data from NPZ files for a specific chip.
 * Baseline data has the first GAIN_LOCK_SKIP packets removed (radio warm-up noise).
 * @param chip Chip type (C3, C6, ESP32, or S3)
 */
inline bool load(ChipType chip = ChipType::C6) {
    // If already loaded with same chip, skip
    if (g_loaded && chip == g_current_chip) return true;
    
    const char* baseline_file = baseline_file_for_chip(chip);
    const char* movement_file = movement_file_for_chip(chip);
    if (baseline_file == nullptr || movement_file == nullptr) {
        std::fprintf(stderr, "[CSI Test Data] ERROR: Unknown chip type in load()\n");
        return false;
    }
    
    try {
        printf("\n[CSI Test Data] Loading %s 64 SC dataset (HT20)...\n", chip_name(chip));
        printf("[CSI Test Data] Baseline: %s\n", baseline_file);
        g_baseline_data = load_npz(baseline_file);
        int raw_count = g_baseline_data.num_packets;
        skip_packets(g_baseline_data, GAIN_LOCK_SKIP);
        g_baseline_ptrs = get_packet_pointers(g_baseline_data);
        printf("[CSI Test Data] Loaded %d baseline packets (%d bytes each, skipped first %d)\n", 
               g_baseline_data.num_packets, g_baseline_data.packet_size, raw_count - g_baseline_data.num_packets);
        
        printf("[CSI Test Data] Movement: %s\n", movement_file);
        g_movement_data = load_npz(movement_file);
        g_movement_ptrs = get_packet_pointers(g_movement_data);
        printf("[CSI Test Data] Loaded %d movement packets (%d bytes each)\n", 
               g_movement_data.num_packets, g_movement_data.packet_size);
        
        g_loaded = true;
        g_current_chip = chip;
        return true;
        
    } catch (const std::exception& e) {
        printf("[CSI Test Data] ERROR: Failed to load NPZ files: %s\n", e.what());
        return false;
    }
}

/**
 * Switch to a different dataset.
 * Forces reload even if already loaded.
 */
inline bool switch_dataset(ChipType chip) {
    g_loaded = false;  // Force reload
    return load(chip);
}

/**
 * Get list of available chip configurations for parametrized testing.
 * Note: Some chips are skipped (check chip_skip_reason()).
 */
inline std::vector<ChipType> get_available_chips() {
    return {ChipType::C3, ChipType::C6, ChipType::ESP32, ChipType::S3};
}

// ============================================================================
// Accessors (compatible with old static array interface)
// ============================================================================

inline bool is_loaded() { return g_loaded; }
inline const int8_t** baseline_packets() { return g_baseline_ptrs.data(); }
inline const int8_t** movement_packets() { return g_movement_ptrs.data(); }
inline int num_baseline() { return g_baseline_data.num_packets; }
inline int num_movement() { return g_movement_data.num_packets; }
inline int num_subcarriers() { return g_baseline_data.num_subcarriers; }
inline int packet_size() { return g_baseline_data.packet_size; }
inline ChipType current_chip() { return g_current_chip; }

/**
 * Whether the baseline dataset was collected with gain lock enabled.
 * Returns false (use CV normalization) when 'gain_locked' field is absent from NPZ.
 */
inline bool baseline_gain_locked() { return g_baseline_data.gain_locked; }

/**
 * Whether 'gain_locked' metadata was found in the baseline NPZ file.
 * If false, callers should fall back to chip-based heuristics.
 */
inline bool baseline_gain_locked_known() { return g_baseline_data.has_gain_locked; }

inline bool parse_iso8601_datetime(const std::string& text, std::tm& out_tm) {
    // Expected examples:
    // 2025-12-12T14:24:43.381306
    // 2026-03-07T19:01:52.250007+00:00
    if (text.size() < 19) {
        return false;
    }
    int y = 0, mo = 0, d = 0, hh = 0, mm = 0, ss = 0;
    int matched = std::sscanf(text.c_str(), "%4d-%2d-%2dT%2d:%2d:%2d",
                              &y, &mo, &d, &hh, &mm, &ss);
    if (matched != 6) {
        return false;
    }
    std::tm tm_val{};
    tm_val.tm_year = y - 1900;
    tm_val.tm_mon = mo - 1;
    tm_val.tm_mday = d;
    tm_val.tm_hour = hh;
    tm_val.tm_min = mm;
    tm_val.tm_sec = ss;
    out_tm = tm_val;
    return true;
}

inline bool current_pair_delta_seconds(double& out_delta_sec) {
    if (!load_tuning_cache()) {
        return false;
    }

    const char* baseline_file = baseline_file_for_chip(g_current_chip);
    if (baseline_file == nullptr) {
        return false;
    }
    const std::string baseline_name = filename_from_path(baseline_file);
    auto bit = g_tuning_by_baseline.find(baseline_name);
    if (bit == g_tuning_by_baseline.end()) {
        return false;
    }
    const std::string& movement_name = bit->second.pair_movement_file;
    if (movement_name.empty()) {
        return false;
    }
    auto mit = g_movement_collected_at.find(movement_name);
    if (mit == g_movement_collected_at.end()) {
        return false;
    }

    std::tm btm{}, mtm{};
    if (!parse_iso8601_datetime(bit->second.collected_at, btm) ||
        !parse_iso8601_datetime(mit->second, mtm)) {
        return false;
    }

    std::time_t bt = std::mktime(&btm);
    std::time_t mt = std::mktime(&mtm);
    if (bt == static_cast<std::time_t>(-1) || mt == static_cast<std::time_t>(-1)) {
        return false;
    }

    out_delta_sec = std::difftime(mt, bt);
    return true;
}

inline bool is_temporally_paired_30m() {
    double delta_sec = 0.0;
    if (!current_pair_delta_seconds(delta_sec)) {
        return false;
    }
    return std::fabs(delta_sec) <= (30.0 * 60.0);
}

} // namespace csi_test_data

#endif // CSI_TEST_DATA_H
