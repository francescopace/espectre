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

// Include cnpy declarations; implementation is linked from espectre_test_support.
#include "cnpy.h"

#include <array>
#include <vector>
#include <string>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <ctime>
#include <cmath>
#include <fstream>
#include <cstring>
#include <regex>
#include <unordered_map>
#include <ArduinoJson.h>
#include "utils.h"

using namespace ArduinoJson;

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
    C5,
    C6,
    ESP32, // Control set (excluded from ML training)
    S3
};

static constexpr size_t CHIP_COUNT = 5;

inline int chip_index(ChipType chip) {
    switch (chip) {
        case ChipType::C3: return 0;
        case ChipType::C5: return 1;
        case ChipType::C6: return 2;
        case ChipType::ESP32: return 3;
        case ChipType::S3: return 4;
        default: return -1;
    }
}

inline bool chip_from_string(const char* text, ChipType& out_chip) {
    if (text == nullptr) {
        return false;
    }
    if (std::strcmp(text, "C3") == 0) {
        out_chip = ChipType::C3;
        return true;
    }
    if (std::strcmp(text, "C5") == 0) {
        out_chip = ChipType::C5;
        return true;
    }
    if (std::strcmp(text, "C6") == 0) {
        out_chip = ChipType::C6;
        return true;
    }
    if (std::strcmp(text, "ESP32") == 0) {
        out_chip = ChipType::ESP32;
        return true;
    }
    if (std::strcmp(text, "S3") == 0) {
        out_chip = ChipType::S3;
        return true;
    }
    return false;
}

inline const char* chip_name(ChipType chip) {
    switch (chip) {
        case ChipType::C3: return "C3";
        case ChipType::C5: return "C5";
        case ChipType::C6: return "C6";
        case ChipType::ESP32: return "ESP32";
        case ChipType::S3: return "S3";
        default: return "Unknown";
    }
}

inline bool load_tuning_cache();
inline const char* baseline_file_for_chip(ChipType chip);
inline const char* movement_file_for_chip(ChipType chip);
inline std::vector<ChipType> get_available_chips();
inline bool parse_iso8601_datetime(const std::string& text, std::tm& out_tm);
inline bool parse_iso8601_epoch_seconds(const std::string& text, double& out_epoch_seconds);

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

enum class DatasetMode {
    StandardPair,
    LongRecording
};

static CsiData g_baseline_data;
static CsiData g_movement_data;
static std::vector<const int8_t*> g_baseline_ptrs;
static std::vector<const int8_t*> g_movement_ptrs;
static bool g_loaded = false;
static ChipType g_current_chip = ChipType::C6;
static DatasetMode g_dataset_mode = DatasetMode::StandardPair;
static bool g_tuning_cache_loaded = false;
static bool g_long_recording_cache_loaded = false;
struct ChipDatasetSelection {
    std::string baseline_filename;
    std::string movement_filename;
    std::string baseline_path;
    std::string movement_path;
    std::string baseline_collected_at;
    std::string movement_collected_at;
    bool valid = false;
};
static std::array<ChipDatasetSelection, CHIP_COUNT> g_selected_by_chip;

struct LongRecordingSelection {
    std::string filename;
    std::string path;
    std::string collected_at;
    int motion_start_packet = 0;
    int num_packets = 0;
    bool gain_locked = false;
    bool valid = false;
};
static std::array<LongRecordingSelection, CHIP_COUNT> g_long_selected_by_chip;

inline bool extract_motion_start_from_description(const std::string& description, int& out_motion_start) {
    static const std::regex kMotionStartPattern(
        "motion\\s+starts\\s+at\\s+packet(?:\\s+index)?(?:\\s+n\\.)?\\s+(\\d+)",
        std::regex_constants::icase);
    std::smatch match;
    if (!std::regex_search(description, match, kMotionStartPattern) || match.size() < 2) {
        return false;
    }
    out_motion_start = std::atoi(match[1].str().c_str());
    return out_motion_start > 0;
}

inline CsiData slice_packets(const CsiData& source, int start_idx, int end_idx) {
    CsiData result;
    const int clamped_start = std::max(0, start_idx);
    const int clamped_end = std::min(end_idx, source.num_packets);
    if (clamped_start >= clamped_end) {
        result.num_packets = 0;
        result.packet_size = source.packet_size;
        result.num_subcarriers = source.num_subcarriers;
        result.gain_locked = source.gain_locked;
        result.has_gain_locked = source.has_gain_locked;
        return result;
    }

    result.packet_size = source.packet_size;
    result.num_subcarriers = source.num_subcarriers;
    result.gain_locked = source.gain_locked;
    result.has_gain_locked = source.has_gain_locked;
    result.packets.assign(source.packets.begin() + clamped_start, source.packets.begin() + clamped_end);
    result.num_packets = static_cast<int>(result.packets.size());
    return result;
}

inline bool load_tuning_cache() {
    if (g_tuning_cache_loaded) {
        return true;
    }

    const std::string dataset_info_path = "../../data/dataset_info.json";
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
    struct LatestFile {
        std::string filename;
        std::string path;
        std::string collected_at;
        bool valid = false;
    };
    std::array<LatestFile, CHIP_COUNT> latest_baseline{};
    std::array<LatestFile, CHIP_COUNT> latest_movement{};
    std::array<std::vector<LatestFile>, CHIP_COUNT> baseline_candidates{};
    std::array<std::vector<LatestFile>, CHIP_COUNT> movement_candidates{};

    for (JsonObject entry : baseline_entries) {
        const char* filename = entry["filename"];
        const char* chip_text = entry["chip"];
        int subcarriers = entry["subcarriers"] | 0;
        const char* collected_at = entry["collected_at"];
        if (filename == nullptr || chip_text == nullptr || collected_at == nullptr) {
            continue;
        }

        ChipType chip{};
        if (!chip_from_string(chip_text, chip)) {
            continue;
        }
        const int idx = chip_index(chip);
        if (idx < 0) {
            continue;
        }

        // Keep the latest baseline per chip for robust fallback pairing.
        if (subcarriers == 64) {
            LatestFile candidate{};
            candidate.filename = filename;
            candidate.path = std::string("../../data/baseline/") + filename;
            candidate.collected_at = collected_at;
            candidate.valid = true;
            baseline_candidates[idx].push_back(candidate);

            LatestFile& latest = latest_baseline[idx];
            const std::string ts(collected_at);
            if (!latest.valid || ts > latest.collected_at) {
                latest.filename = filename;
                latest.path = std::string("../../data/baseline/") + filename;
                latest.collected_at = ts;
                latest.valid = true;
            }
        }

    }

    JsonArray movement_entries = doc["files"]["movement"].as<JsonArray>();
    for (JsonObject entry : movement_entries) {
        const char* filename = entry["filename"];
        const char* collected_at = entry["collected_at"];
        const char* chip_text = entry["chip"];
        int subcarriers = entry["subcarriers"] | 0;
        ChipType chip{};
        if (filename != nullptr && chip_from_string(chip_text, chip)) {
            if (subcarriers == 64 && collected_at != nullptr) {
                const int idx = chip_index(chip);
                if (idx >= 0) {
                    LatestFile& latest = latest_movement[idx];
                    const std::string ts(collected_at);
                    LatestFile candidate{};
                    candidate.filename = filename;
                    candidate.path = std::string("../../data/movement/") + filename;
                    candidate.collected_at = ts;
                    candidate.valid = true;
                    movement_candidates[idx].push_back(candidate);

                    if (!latest.valid || ts > latest.collected_at) {
                        latest.filename = filename;
                        latest.path = std::string("../../data/movement/") + filename;
                        latest.collected_at = ts;
                        latest.valid = true;
                    }
                }
            }
        }
    }

    for (auto& selected : g_selected_by_chip) {
        selected = ChipDatasetSelection{};
    }

    // Select one 64SC baseline/movement pair per chip using nearest timestamps.
    auto parse_epoch = [](const std::string& ts, double& out_epoch_seconds) -> bool {
        return parse_iso8601_epoch_seconds(ts, out_epoch_seconds);
    };

    for (ChipType chip : get_available_chips()) {
        const int idx = chip_index(chip);
        if (idx < 0) {
            continue;
        }

        if (baseline_candidates[idx].empty() || movement_candidates[idx].empty()) {
            continue;
        }

        LatestFile best_baseline{};
        LatestFile best_movement{};
        bool found_nearest_pair = false;
        double best_delta = 1e100;
        for (const auto& b : baseline_candidates[idx]) {
            double b_epoch = 0.0;
            if (!parse_epoch(b.collected_at, b_epoch)) {
                continue;
            }
            for (const auto& m : movement_candidates[idx]) {
                double m_epoch = 0.0;
                if (!parse_epoch(m.collected_at, m_epoch)) {
                    continue;
                }
                const double delta = std::fabs(m_epoch - b_epoch);
                if (!found_nearest_pair || delta < best_delta) {
                    best_delta = delta;
                    best_baseline = b;
                    best_movement = m;
                    found_nearest_pair = true;
                }
            }
        }

        ChipDatasetSelection& selected = g_selected_by_chip[idx];
        if (found_nearest_pair) {
            selected.baseline_filename = best_baseline.filename;
            selected.movement_filename = best_movement.filename;
            selected.baseline_path = best_baseline.path;
            selected.movement_path = best_movement.path;
            selected.baseline_collected_at = best_baseline.collected_at;
            selected.movement_collected_at = best_movement.collected_at;
        } else {
            if (!latest_baseline[idx].valid || !latest_movement[idx].valid) {
                continue;
            }
            selected.baseline_filename = latest_baseline[idx].filename;
            selected.movement_filename = latest_movement[idx].filename;
            selected.baseline_path = latest_baseline[idx].path;
            selected.movement_path = latest_movement[idx].path;
            selected.baseline_collected_at = latest_baseline[idx].collected_at;
            selected.movement_collected_at = latest_movement[idx].collected_at;
        }
        selected.valid = true;
    }

    for (ChipType chip : get_available_chips()) {
        const int idx = chip_index(chip);
        if (idx < 0 || !g_selected_by_chip[idx].valid) {
            std::fprintf(stderr,
                "[CSI Test Data] ERROR: Missing 64SC baseline/movement datasets for chip %s\n",
                chip_name(chip));
            return false;
        }
    }

    g_tuning_cache_loaded = true;
    return true;
}

inline bool load_long_recording_cache() {
    if (g_long_recording_cache_loaded) {
        return true;
    }

    const std::string dataset_info_path = "../../data/dataset_info.json";
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

    for (auto& selected : g_long_selected_by_chip) {
        selected = LongRecordingSelection{};
    }

    JsonArray test_entries = doc["files"]["test"].as<JsonArray>();
    for (JsonObject entry : test_entries) {
        const char* filename = entry["filename"];
        const char* chip_text = entry["chip"];
        const char* collected_at = entry["collected_at"];
        const char* description = entry["description"];
        const int subcarriers = entry["subcarriers"] | 0;
        const int num_packets = entry["num_packets"] | 0;
        const bool gain_locked = entry["gain_locked"] | false;
        if (filename == nullptr || chip_text == nullptr || collected_at == nullptr || subcarriers != 64) {
            continue;
        }

        ChipType chip{};
        if (!chip_from_string(chip_text, chip)) {
            continue;
        }
        const int idx = chip_index(chip);
        if (idx < 0) {
            continue;
        }

        int motion_start_packet = 0;
        if (description == nullptr || !extract_motion_start_from_description(description, motion_start_packet)) {
            motion_start_packet = num_packets / 2;
        }

        if (num_packets <= 1 || motion_start_packet <= 0 || motion_start_packet >= num_packets) {
            continue;
        }

        LongRecordingSelection candidate{};
        candidate.filename = filename;
        candidate.path = std::string("../../data/test/") + filename;
        candidate.collected_at = collected_at;
        candidate.motion_start_packet = motion_start_packet;
        candidate.num_packets = num_packets;
        candidate.gain_locked = gain_locked;
        candidate.valid = true;

        LongRecordingSelection& selected = g_long_selected_by_chip[idx];
        if (!selected.valid || candidate.collected_at > selected.collected_at) {
            selected = candidate;
        }
    }

    for (ChipType chip : get_available_chips()) {
        const int idx = chip_index(chip);
        if (idx < 0) {
            continue;
        }
        if (!g_long_selected_by_chip[idx].valid) {
            continue;
        }
    }

    g_long_recording_cache_loaded = true;
    return true;
}

inline const char* baseline_file_for_chip(ChipType chip) {
    if (!load_tuning_cache()) {
        return nullptr;
    }
    const int idx = chip_index(chip);
    if (idx < 0 || !g_selected_by_chip[idx].valid) {
        return nullptr;
    }
    return g_selected_by_chip[idx].baseline_path.c_str();
}

inline const char* movement_file_for_chip(ChipType chip) {
    if (!load_tuning_cache()) {
        return nullptr;
    }
    const int idx = chip_index(chip);
    if (idx < 0 || !g_selected_by_chip[idx].valid) {
        return nullptr;
    }
    return g_selected_by_chip[idx].movement_path.c_str();
}

inline const char* long_recording_file_for_chip(ChipType chip) {
    if (!load_long_recording_cache()) {
        return nullptr;
    }
    const int idx = chip_index(chip);
    if (idx < 0 || !g_long_selected_by_chip[idx].valid) {
        return nullptr;
    }
    return g_long_selected_by_chip[idx].path.c_str();
}

inline int long_recording_motion_start_for_chip(ChipType chip) {
    if (!load_long_recording_cache()) {
        return 0;
    }
    const int idx = chip_index(chip);
    if (idx < 0 || !g_long_selected_by_chip[idx].valid) {
        return 0;
    }
    return g_long_selected_by_chip[idx].motion_start_packet;
}

inline const char* long_recording_name_for_chip(ChipType chip) {
    if (!load_long_recording_cache()) {
        return nullptr;
    }
    const int idx = chip_index(chip);
    if (idx < 0 || !g_long_selected_by_chip[idx].valid) {
        return nullptr;
    }
    return g_long_selected_by_chip[idx].filename.c_str();
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
    if (g_loaded && chip == g_current_chip && g_dataset_mode == DatasetMode::StandardPair) return true;
    
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
        g_dataset_mode = DatasetMode::StandardPair;
        return true;
        
    } catch (const std::exception& e) {
        printf("[CSI Test Data] ERROR: Failed to load NPZ files: %s\n", e.what());
        return false;
    }
}

inline bool load_long_recording(ChipType chip = ChipType::C6) {
    if (g_loaded && chip == g_current_chip && g_dataset_mode == DatasetMode::LongRecording) return true;

    const char* long_recording_file = long_recording_file_for_chip(chip);
    const int motion_start_packet = long_recording_motion_start_for_chip(chip);
    if (long_recording_file == nullptr || motion_start_packet <= 0) {
        std::fprintf(stderr, "[CSI Test Data] ERROR: Missing long recording metadata for chip %s\n", chip_name(chip));
        return false;
    }

    try {
        printf("\n[CSI Test Data] Loading %s long recording dataset...\n", chip_name(chip));
        printf("[CSI Test Data] Test: %s\n", long_recording_file);
        CsiData full_data = load_npz(long_recording_file);
        if (motion_start_packet >= full_data.num_packets) {
            std::fprintf(stderr,
                         "[CSI Test Data] ERROR: Invalid motion_start_packet=%d for %s (%d packets)\n",
                         motion_start_packet, long_recording_file, full_data.num_packets);
            return false;
        }

        g_baseline_data = slice_packets(full_data, 0, motion_start_packet);
        g_movement_data = slice_packets(full_data, motion_start_packet, full_data.num_packets);
        g_baseline_ptrs = get_packet_pointers(g_baseline_data);
        g_movement_ptrs = get_packet_pointers(g_movement_data);

        printf("[CSI Test Data] Split at packet %d -> baseline=%d, movement=%d (%d bytes each)\n",
               motion_start_packet, g_baseline_data.num_packets, g_movement_data.num_packets,
               g_baseline_data.packet_size);

        g_loaded = true;
        g_current_chip = chip;
        g_dataset_mode = DatasetMode::LongRecording;
        return true;

    } catch (const std::exception& e) {
        printf("[CSI Test Data] ERROR: Failed to load long NPZ file: %s\n", e.what());
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

inline bool switch_long_recording_dataset(ChipType chip) {
    g_loaded = false;
    return load_long_recording(chip);
}

inline std::vector<ChipType> get_available_long_recording_chips() {
    std::vector<ChipType> chips;
    if (!load_long_recording_cache()) {
        return chips;
    }
    for (ChipType chip : get_available_chips()) {
        const int idx = chip_index(chip);
        if (idx >= 0 && g_long_selected_by_chip[idx].valid) {
            chips.push_back(chip);
        }
    }
    return chips;
}

/**
 * Get list of available chip configurations for parametrized testing.
 * Note: Some chips are skipped (check chip_skip_reason()).
 */
inline std::vector<ChipType> get_available_chips() {
    return {ChipType::C3, ChipType::C5, ChipType::C6, ChipType::ESP32, ChipType::S3};
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
inline bool is_long_recording_mode() { return g_dataset_mode == DatasetMode::LongRecording; }
inline const char* current_long_recording_name() {
    return is_long_recording_mode() ? long_recording_name_for_chip(g_current_chip) : nullptr;
}
inline int current_motion_start_packet() {
    return is_long_recording_mode() ? long_recording_motion_start_for_chip(g_current_chip) : 0;
}

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

inline bool parse_iso8601_epoch_seconds(const std::string& text, double& out_epoch_seconds) {
    std::tm tm_val{};
    if (!parse_iso8601_datetime(text, tm_val)) {
        return false;
    }

    std::time_t epoch = std::mktime(&tm_val);
    if (epoch == static_cast<std::time_t>(-1)) {
        return false;
    }

    double fractional_seconds = 0.0;
    const size_t frac_pos = text.find('.', 19);
    if (frac_pos != std::string::npos) {
        size_t frac_end = frac_pos + 1;
        while (frac_end < text.size() && text[frac_end] >= '0' && text[frac_end] <= '9') {
            frac_end++;
        }
        if (frac_end > frac_pos + 1) {
            const std::string frac_digits = text.substr(frac_pos + 1, frac_end - frac_pos - 1);
            fractional_seconds = std::strtod(("0." + frac_digits).c_str(), nullptr);
        }
    }

    out_epoch_seconds = static_cast<double>(epoch) + fractional_seconds;
    return true;
}

inline bool current_pair_delta_seconds(double& out_delta_sec) {
    if (!load_tuning_cache()) {
        return false;
    }
    const int idx = chip_index(g_current_chip);
    if (idx < 0 || !g_selected_by_chip[idx].valid) {
        return false;
    }
    const ChipDatasetSelection& selected = g_selected_by_chip[idx];
    if (selected.baseline_collected_at.empty() || selected.movement_collected_at.empty()) {
        return false;
    }

    double bt = 0.0;
    double mt = 0.0;
    if (!parse_iso8601_epoch_seconds(selected.baseline_collected_at, bt) ||
        !parse_iso8601_epoch_seconds(selected.movement_collected_at, mt)) {
        return false;
    }

    out_delta_sec = mt - bt;
    return true;
}

inline bool is_temporally_paired() {
    double delta_sec = 0.0;
    if (!current_pair_delta_seconds(delta_sec)) {
        return false;
    }
    return std::fabs(delta_sec) <= (60.0);
}

} // namespace csi_test_data

#endif // CSI_TEST_DATA_H
