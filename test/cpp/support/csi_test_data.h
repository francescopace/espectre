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
 *   const int8_t** static_presence_packets = csi_test_data::static_presence_packets();
 *   const int8_t** motion_packets = csi_test_data::motion_packets();
 *   int num_static_presence = csi_test_data::num_static_presence();
 *   int num_motion = csi_test_data::num_motion();
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
#include <algorithm>
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
inline const char* static_presence_file_for_chip(ChipType chip);
inline const char* motion_file_for_chip(ChipType chip);
inline std::vector<ChipType> get_supported_chips();
inline std::vector<ChipType> get_available_chips();
inline int get_available_pair_count();
inline ChipType pair_chip(int pair_index);
inline const char* pair_label(int pair_index);
inline bool switch_dataset_pair(int pair_index);
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

enum class DatasetMode {
    StandardPair,
    LongRecording
};

static CsiData g_static_presence_data;
static CsiData g_motion_data;
static std::vector<const int8_t*> g_static_presence_ptrs;
static std::vector<const int8_t*> g_motion_ptrs;
static bool g_loaded = false;
static ChipType g_current_chip = ChipType::C6;
static DatasetMode g_dataset_mode = DatasetMode::StandardPair;
static bool g_tuning_cache_loaded = false;
static bool g_long_recording_cache_loaded = false;
struct ChipDatasetSelection {
    ChipType chip = ChipType::C6;
    std::string static_presence_filename;
    std::string motion_filename;
    std::string static_presence_path;
    std::string motion_path;
    std::string environment;
    bool valid = false;
};
static std::array<ChipDatasetSelection, CHIP_COUNT> g_selected_by_chip;
static std::vector<ChipDatasetSelection> g_pair_selections;
static int g_current_pair_index = -1;

struct LongRecordingSelection {
    std::string filename;
    std::string path;
    std::string collected_at;
    int motion_start_packet = 0;
    int num_packets = 0;
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
        return result;
    }

    result.packet_size = source.packet_size;
    result.num_subcarriers = source.num_subcarriers;
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

    struct PairFile {
        std::string filename;
        std::string path;
        std::string environment;
        std::string optimal_pair_motion_file;
        bool valid = false;
    };
    std::array<std::vector<PairFile>, CHIP_COUNT> static_presence_candidates{};
    std::unordered_map<std::string, PairFile> motion_by_filename;

    JsonArray static_presence_entries = doc["files"]["static_presence"].as<JsonArray>();
    for (JsonObject entry : static_presence_entries) {
        const char* filename = entry["filename"];
        const char* chip_text = entry["chip"];
        int subcarriers = entry["subcarriers"] | 0;
        const char* environment = entry["environment"] | "";
        const char* optimal_pair_motion_file = entry["optimal_pair_motion_file"];
        if (filename == nullptr || chip_text == nullptr || optimal_pair_motion_file == nullptr) {
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

        if (subcarriers == 64) {
            PairFile candidate{};
            candidate.filename = filename;
            candidate.path = std::string("../../data/static_presence/") + filename;
            candidate.environment = environment;
            candidate.optimal_pair_motion_file = optimal_pair_motion_file;
            candidate.valid = true;
            static_presence_candidates[idx].push_back(candidate);
        }

    }

    JsonArray motion_entries = doc["files"]["motion"].as<JsonArray>();
    for (JsonObject entry : motion_entries) {
        const char* filename = entry["filename"];
        const char* chip_text = entry["chip"];
        int subcarriers = entry["subcarriers"] | 0;
        ChipType chip{};
        if (filename != nullptr && chip_from_string(chip_text, chip)) {
            if (subcarriers == 64) {
                const int idx = chip_index(chip);
                if (idx >= 0) {
                    PairFile candidate{};
                    candidate.filename = filename;
                    candidate.path = std::string("../../data/motion/") + filename;
                    candidate.environment = entry["environment"] | "";
                    candidate.valid = true;
                    motion_by_filename[candidate.filename] = candidate;
                }
            }
        }
    }

    for (auto& selected : g_selected_by_chip) {
        selected = ChipDatasetSelection{};
    }
    g_pair_selections.clear();
    g_current_pair_index = -1;

    // Load every explicit static-presence/motion pair from dataset_info.json.
    for (ChipType chip : get_supported_chips()) {
        const int idx = chip_index(chip);
        if (idx < 0) {
            continue;
        }

        for (const auto& static_presence : static_presence_candidates[idx]) {
            auto motion_it = motion_by_filename.find(static_presence.optimal_pair_motion_file);
            if (motion_it == motion_by_filename.end()) {
                continue;
            }

            const PairFile& motion = motion_it->second;
            ChipDatasetSelection selected{};
            selected.chip = chip;
            selected.static_presence_filename = static_presence.filename;
            selected.motion_filename = motion.filename;
            selected.static_presence_path = static_presence.path;
            selected.motion_path = motion.path;
            selected.environment = static_presence.environment;
            selected.valid = true;
            g_pair_selections.push_back(selected);

            ChipDatasetSelection& selected_by_chip = g_selected_by_chip[idx];
            if (!selected_by_chip.valid) {
                selected_by_chip = selected;
            }
        }
    }

    std::sort(g_pair_selections.begin(), g_pair_selections.end(),
              [](const ChipDatasetSelection& a, const ChipDatasetSelection& b) {
                  const int a_idx = chip_index(a.chip);
                  const int b_idx = chip_index(b.chip);
                  if (a_idx != b_idx) {
                      return a_idx < b_idx;
                  }
                  if (a.environment != b.environment) {
                      return a.environment < b.environment;
                  }
                  return a.static_presence_filename < b.static_presence_filename;
              });

    if (g_pair_selections.empty()) {
        std::fprintf(stderr,
            "[CSI Test Data] ERROR: No complete 64SC static-presence/motion dataset pairs found\n");
        return false;
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
        candidate.valid = true;

        LongRecordingSelection& selected = g_long_selected_by_chip[idx];
        if (!selected.valid || candidate.collected_at > selected.collected_at) {
            selected = candidate;
        }
    }

    for (ChipType chip : get_supported_chips()) {
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

inline const char* static_presence_file_for_chip(ChipType chip) {
    if (!load_tuning_cache()) {
        return nullptr;
    }
    const int idx = chip_index(chip);
    if (idx < 0 || !g_selected_by_chip[idx].valid) {
        return nullptr;
    }
    return g_selected_by_chip[idx].static_presence_path.c_str();
}

inline const char* motion_file_for_chip(ChipType chip) {
    if (!load_tuning_cache()) {
        return nullptr;
    }
    const int idx = chip_index(chip);
    if (idx < 0 || !g_selected_by_chip[idx].valid) {
        return nullptr;
    }
    return g_selected_by_chip[idx].motion_path.c_str();
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
 * Load CSI test data from NPZ files for a specific chip.
 * Static-presence data is loaded from packet 0 so threshold calibration matches
 * live startup behavior.
 * @param chip Chip type (C3, C6, ESP32, or S3)
 */
inline bool load(ChipType chip = ChipType::C6) {
    // If already loaded with same chip, skip
    if (g_loaded && chip == g_current_chip && g_dataset_mode == DatasetMode::StandardPair &&
        g_current_pair_index < 0) {
        return true;
    }
    
    const char* static_presence_file = static_presence_file_for_chip(chip);
    const char* motion_file = motion_file_for_chip(chip);
    if (static_presence_file == nullptr || motion_file == nullptr) {
        std::fprintf(stderr, "[CSI Test Data] ERROR: Unknown chip type in load()\n");
        return false;
    }
    
    try {
        printf("\n[CSI Test Data] Loading %s 64 SC dataset (HT20)...\n", chip_name(chip));
        printf("[CSI Test Data] Static presence: %s\n", static_presence_file);
        g_static_presence_data = load_npz(static_presence_file);
        g_static_presence_ptrs = get_packet_pointers(g_static_presence_data);
        printf("[CSI Test Data] Loaded %d static-presence packets (%d bytes each, from packet 0)\n",
               g_static_presence_data.num_packets, g_static_presence_data.packet_size);
        
        printf("[CSI Test Data] Motion: %s\n", motion_file);
        g_motion_data = load_npz(motion_file);
        g_motion_ptrs = get_packet_pointers(g_motion_data);
        printf("[CSI Test Data] Loaded %d motion packets (%d bytes each)\n", 
               g_motion_data.num_packets, g_motion_data.packet_size);
        
        g_loaded = true;
        g_current_chip = chip;
        g_dataset_mode = DatasetMode::StandardPair;
        g_current_pair_index = -1;
        return true;
        
    } catch (const std::exception& e) {
        printf("[CSI Test Data] ERROR: Failed to load NPZ files: %s\n", e.what());
        return false;
    }
}

inline bool load_pair(int pair_index) {
    if (!load_tuning_cache()) {
        return false;
    }
    if (pair_index < 0 || pair_index >= static_cast<int>(g_pair_selections.size())) {
        std::fprintf(stderr, "[CSI Test Data] ERROR: Invalid pair index %d\n", pair_index);
        return false;
    }
    if (g_loaded && pair_index == g_current_pair_index && g_dataset_mode == DatasetMode::StandardPair) {
        return true;
    }

    const ChipDatasetSelection& selected = g_pair_selections[pair_index];
    try {
        printf("\n[CSI Test Data] Loading %s 64 SC pair #%d (%s)...\n",
               chip_name(selected.chip), pair_index, selected.environment.c_str());
        printf("[CSI Test Data] Static presence: %s\n", selected.static_presence_path.c_str());
        g_static_presence_data = load_npz(selected.static_presence_path);
        g_static_presence_ptrs = get_packet_pointers(g_static_presence_data);
        printf("[CSI Test Data] Loaded %d static-presence packets (%d bytes each, from packet 0)\n",
               g_static_presence_data.num_packets, g_static_presence_data.packet_size);

        printf("[CSI Test Data] Motion: %s\n", selected.motion_path.c_str());
        g_motion_data = load_npz(selected.motion_path);
        g_motion_ptrs = get_packet_pointers(g_motion_data);
        printf("[CSI Test Data] Loaded %d motion packets (%d bytes each)\n",
               g_motion_data.num_packets, g_motion_data.packet_size);

        g_loaded = true;
        g_current_chip = selected.chip;
        g_dataset_mode = DatasetMode::StandardPair;
        g_current_pair_index = pair_index;
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

        g_static_presence_data = slice_packets(full_data, 0, motion_start_packet);
        g_motion_data = slice_packets(full_data, motion_start_packet, full_data.num_packets);
        g_static_presence_ptrs = get_packet_pointers(g_static_presence_data);
        g_motion_ptrs = get_packet_pointers(g_motion_data);

        printf("[CSI Test Data] Split at packet %d -> static_presence=%d, motion=%d (%d bytes each)\n",
               motion_start_packet, g_static_presence_data.num_packets, g_motion_data.num_packets,
               g_static_presence_data.packet_size);

        g_loaded = true;
        g_current_chip = chip;
        g_dataset_mode = DatasetMode::LongRecording;
        g_current_pair_index = -1;
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

inline bool switch_dataset_pair(int pair_index) {
    g_loaded = false;
    return load_pair(pair_index);
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
    for (ChipType chip : get_supported_chips()) {
        const int idx = chip_index(chip);
        if (idx >= 0 && g_long_selected_by_chip[idx].valid) {
            chips.push_back(chip);
        }
    }
    return chips;
}

/**
 * Get list of supported chip configurations.
 */
inline std::vector<ChipType> get_supported_chips() {
    return {ChipType::C3, ChipType::C5, ChipType::C6, ChipType::ESP32, ChipType::S3};
}

/**
 * Get list of chip configurations with complete static-presence/motion pairs.
 * Note: Some chips are skipped (check chip_skip_reason()).
 */
inline std::vector<ChipType> get_available_chips() {
    std::vector<ChipType> chips;
    if (!load_tuning_cache()) {
        return chips;
    }
    for (ChipType chip : get_supported_chips()) {
        const int idx = chip_index(chip);
        if (idx >= 0 && g_selected_by_chip[idx].valid) {
            chips.push_back(chip);
        }
    }
    return chips;
}

inline int get_available_pair_count() {
    if (!load_tuning_cache()) {
        return 0;
    }
    return static_cast<int>(g_pair_selections.size());
}

inline ChipType pair_chip(int pair_index) {
    if (!load_tuning_cache() || pair_index < 0 ||
        pair_index >= static_cast<int>(g_pair_selections.size())) {
        return ChipType::C6;
    }
    return g_pair_selections[pair_index].chip;
}

inline const char* pair_label(int pair_index) {
    static std::string label;
    if (!load_tuning_cache() || pair_index < 0 ||
        pair_index >= static_cast<int>(g_pair_selections.size())) {
        label = "unknown_pair";
        return label.c_str();
    }

    const ChipDatasetSelection& selected = g_pair_selections[pair_index];
    label = std::string(chip_name(selected.chip)) + ":" + selected.environment + ":" +
            selected.static_presence_filename;
    return label.c_str();
}

inline int current_pair_index() {
    return g_current_pair_index;
}

inline const char* current_pair_label() {
    return pair_label(g_current_pair_index);
}

// ============================================================================
// Accessors (compatible with old static array interface)
// ============================================================================

inline bool is_loaded() { return g_loaded; }
inline const int8_t** static_presence_packets() { return g_static_presence_ptrs.data(); }
inline const int8_t** motion_packets() { return g_motion_ptrs.data(); }
inline int num_static_presence() { return g_static_presence_data.num_packets; }
inline int num_motion() { return g_motion_data.num_packets; }
inline int num_subcarriers() { return g_static_presence_data.num_subcarriers; }
inline int packet_size() { return g_static_presence_data.packet_size; }
inline ChipType current_chip() { return g_current_chip; }
inline bool is_long_recording_mode() { return g_dataset_mode == DatasetMode::LongRecording; }
inline const char* current_long_recording_name() {
    return is_long_recording_mode() ? long_recording_name_for_chip(g_current_chip) : nullptr;
}
inline int current_motion_start_packet() {
    return is_long_recording_mode() ? long_recording_motion_start_for_chip(g_current_chip) : 0;
}

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

} // namespace csi_test_data

#endif // CSI_TEST_DATA_H
