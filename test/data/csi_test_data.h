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
// C6 dataset (old, validated)
#define BASELINE_C6_64SC  "../micro-espectre/data/baseline/baseline_c6_64sc_20251212_142443.npz"
#define MOVEMENT_C6_64SC  "../micro-espectre/data/movement/movement_c6_64sc_20251212_142443.npz"
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
    C6,
    S3
};

inline const char* chip_name(ChipType chip) {
    switch (chip) {
        case ChipType::C6: return "C6";
        case ChipType::S3: return "S3";
        default: return "Unknown";
    }
}

// ============================================================================
// Global Data Storage
// ============================================================================

static CsiData g_baseline_data;
static CsiData g_movement_data;
static std::vector<const int8_t*> g_baseline_ptrs;
static std::vector<const int8_t*> g_movement_ptrs;
static bool g_loaded = false;
static ChipType g_current_chip = ChipType::C6;

/**
 * Load CSI test data from NPZ files for a specific chip.
 * @param chip Chip type (C6 or S3)
 */
inline bool load(ChipType chip = ChipType::C6) {
    // If already loaded with same chip, skip
    if (g_loaded && chip == g_current_chip) return true;
    
    const char* baseline_file = nullptr;
    const char* movement_file = nullptr;
    
    switch (chip) {
        case ChipType::C6:
            baseline_file = BASELINE_C6_64SC;
            movement_file = MOVEMENT_C6_64SC;
            break;
        case ChipType::S3:
            baseline_file = BASELINE_S3_64SC;
            movement_file = MOVEMENT_S3_64SC;
            break;
    }
    
    try {
        printf("\n[CSI Test Data] Loading %s 64 SC dataset (HT20)...\n", chip_name(chip));
        printf("[CSI Test Data] Baseline: %s\n", baseline_file);
        g_baseline_data = load_npz(baseline_file);
        g_baseline_ptrs = get_packet_pointers(g_baseline_data);
        printf("[CSI Test Data] Loaded %d baseline packets (%d bytes each)\n", 
               g_baseline_data.num_packets, g_baseline_data.packet_size);
        
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
 */
inline std::vector<ChipType> get_available_chips() {
    return {ChipType::C6, ChipType::S3};
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

} // namespace csi_test_data

#endif // CSI_TEST_DATA_H
