/*
 * ESPectre - Calibration File Storage Unit Tests
 *
 * Unit tests for file-based magnitude storage in P95Calibrator.
 * Tests write/read operations, data integrity, and cleanup.
 *
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * License: GPLv3
 */

#include <unity.h>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <vector>
#include "esphome/core/log.h"

#if defined(ESP_PLATFORM)
#include "esp_spiffs.h"
#endif

static const char *TAG = "test_file_storage";

// Constants matching P95Calibrator
static constexpr uint8_t NUM_SUBCARRIERS = 64;

// Test buffer file path - different for native vs ESP32
#if defined(ESP_PLATFORM)
static const char* TEST_BUFFER_FILE = "/spiffs/test_buffer.bin";
static bool spiffs_mounted = false;
#else
static const char* TEST_BUFFER_FILE = "/tmp/test_buffer.bin";
#endif

#if defined(ESP_PLATFORM)
static void init_spiffs(void) {
    if (spiffs_mounted) return;
    
    esp_vfs_spiffs_conf_t conf = {
        .base_path = "/spiffs",
        .partition_label = NULL,
        .max_files = 5,
        .format_if_mount_failed = true
    };
    
    esp_err_t ret = esp_vfs_spiffs_register(&conf);
    if (ret == ESP_OK) {
        spiffs_mounted = true;
        ESP_LOGI(TAG, "SPIFFS mounted successfully");
    } else {
        ESP_LOGE(TAG, "Failed to mount SPIFFS: %s", esp_err_to_name(ret));
    }
}
#endif

void setUp(void) {
#if defined(ESP_PLATFORM)
    init_spiffs();
#endif
    // Remove test file before each test
    remove(TEST_BUFFER_FILE);
}

void tearDown(void) {
    // Cleanup after each test
    remove(TEST_BUFFER_FILE);
}

// ============================================================================
// FILE WRITE TESTS
// ============================================================================

void test_file_write_single_packet(void) {
    FILE* f = fopen(TEST_BUFFER_FILE, "wb");
    TEST_ASSERT_NOT_NULL(f);
    
    // Create test magnitudes (64 bytes)
    uint8_t magnitudes[NUM_SUBCARRIERS];
    for (uint8_t i = 0; i < NUM_SUBCARRIERS; i++) {
        magnitudes[i] = i;  // 0, 1, 2, ..., 63
    }
    
    size_t written = fwrite(magnitudes, 1, NUM_SUBCARRIERS, f);
    TEST_ASSERT_EQUAL(NUM_SUBCARRIERS, written);
    
    fclose(f);
    
    // Verify file size
    f = fopen(TEST_BUFFER_FILE, "rb");
    TEST_ASSERT_NOT_NULL(f);
    fseek(f, 0, SEEK_END);
    long size = ftell(f);
    TEST_ASSERT_EQUAL(NUM_SUBCARRIERS, size);
    fclose(f);
}

void test_file_write_multiple_packets(void) {
    FILE* f = fopen(TEST_BUFFER_FILE, "wb");
    TEST_ASSERT_NOT_NULL(f);
    
    uint16_t num_packets = 100;
    uint8_t magnitudes[NUM_SUBCARRIERS];
    
    for (uint16_t p = 0; p < num_packets; p++) {
        // Different pattern for each packet
        for (uint8_t i = 0; i < NUM_SUBCARRIERS; i++) {
            magnitudes[i] = (p + i) % 256;
        }
        size_t written = fwrite(magnitudes, 1, NUM_SUBCARRIERS, f);
        TEST_ASSERT_EQUAL(NUM_SUBCARRIERS, written);
    }
    
    fclose(f);
    
    // Verify file size
    f = fopen(TEST_BUFFER_FILE, "rb");
    TEST_ASSERT_NOT_NULL(f);
    fseek(f, 0, SEEK_END);
    long size = ftell(f);
    TEST_ASSERT_EQUAL(num_packets * NUM_SUBCARRIERS, size);
    fclose(f);
}

// ============================================================================
// FILE READ TESTS
// ============================================================================

void test_file_read_single_packet(void) {
    // Write test data
    FILE* f = fopen(TEST_BUFFER_FILE, "wb");
    TEST_ASSERT_NOT_NULL(f);
    
    uint8_t write_data[NUM_SUBCARRIERS];
    for (uint8_t i = 0; i < NUM_SUBCARRIERS; i++) {
        write_data[i] = i * 2;  // 0, 2, 4, ..., 126
    }
    fwrite(write_data, 1, NUM_SUBCARRIERS, f);
    fclose(f);
    
    // Read back
    f = fopen(TEST_BUFFER_FILE, "rb");
    TEST_ASSERT_NOT_NULL(f);
    
    uint8_t read_data[NUM_SUBCARRIERS];
    size_t read = fread(read_data, 1, NUM_SUBCARRIERS, f);
    TEST_ASSERT_EQUAL(NUM_SUBCARRIERS, read);
    
    // Verify data integrity
    TEST_ASSERT_EQUAL_UINT8_ARRAY(write_data, read_data, NUM_SUBCARRIERS);
    
    fclose(f);
}

void test_file_read_window_from_middle(void) {
    // Write 10 packets
    FILE* f = fopen(TEST_BUFFER_FILE, "wb");
    TEST_ASSERT_NOT_NULL(f);
    
    uint16_t num_packets = 10;
    for (uint16_t p = 0; p < num_packets; p++) {
        uint8_t magnitudes[NUM_SUBCARRIERS];
        for (uint8_t i = 0; i < NUM_SUBCARRIERS; i++) {
            magnitudes[i] = p;  // All bytes in packet p have value p
        }
        fwrite(magnitudes, 1, NUM_SUBCARRIERS, f);
    }
    fclose(f);
    
    // Read window of 3 packets starting at packet 5
    f = fopen(TEST_BUFFER_FILE, "rb");
    TEST_ASSERT_NOT_NULL(f);
    
    uint16_t start_idx = 5;
    uint16_t window_size = 3;
    
    fseek(f, start_idx * NUM_SUBCARRIERS, SEEK_SET);
    
    uint8_t window_data[3 * NUM_SUBCARRIERS];
    size_t read = fread(window_data, 1, window_size * NUM_SUBCARRIERS, f);
    TEST_ASSERT_EQUAL(window_size * NUM_SUBCARRIERS, read);
    
    // Verify: packet 5 should have all 5s, packet 6 all 6s, packet 7 all 7s
    for (uint16_t p = 0; p < window_size; p++) {
        for (uint8_t i = 0; i < NUM_SUBCARRIERS; i++) {
            uint8_t expected = start_idx + p;
            uint8_t actual = window_data[p * NUM_SUBCARRIERS + i];
            TEST_ASSERT_EQUAL_UINT8(expected, actual);
        }
    }
    
    fclose(f);
}

// ============================================================================
// MAGNITUDE CONVERSION TESTS
// ============================================================================

void test_magnitude_uint8_conversion(void) {
    // Test that float magnitudes are correctly converted to uint8
    // Simulating calculate_magnitude_ behavior
    
    struct TestCase {
        int8_t I;
        int8_t Q;
        uint8_t expected_mag;  // floor(sqrt(I² + Q²))
    };
    
    TestCase cases[] = {
        {0, 0, 0},
        {10, 0, 10},
        {0, 10, 10},
        {3, 4, 5},       // 3² + 4² = 25, sqrt = 5
        {6, 8, 10},      // 6² + 8² = 100, sqrt = 10
        {127, 0, 127},   // Max single component
        {90, 90, 127},   // sqrt(8100 + 8100) ≈ 127.3
        {127, 127, 179}, // sqrt(16129 + 16129) ≈ 179.6 → 179
    };
    
    for (size_t i = 0; i < sizeof(cases) / sizeof(cases[0]); i++) {
        float I = static_cast<float>(cases[i].I);
        float Q = static_cast<float>(cases[i].Q);
        float mag = std::sqrt(I * I + Q * Q);
        uint8_t uint8_mag = static_cast<uint8_t>(std::min(mag, 255.0f));
        
        ESP_LOGI(TAG, "I=%d Q=%d -> mag=%.2f -> uint8=%d (expected %d)",
                 cases[i].I, cases[i].Q, mag, uint8_mag, cases[i].expected_mag);
        
        TEST_ASSERT_EQUAL_UINT8(cases[i].expected_mag, uint8_mag);
    }
}

void test_magnitude_clamping_to_255(void) {
    // Edge case: magnitude > 255 should be clamped
    // This can happen with max I/Q values: sqrt(127² + 127²) ≈ 179.6
    // But let's test the clamping logic anyway
    
    float large_mag = 300.0f;
    uint8_t clamped = static_cast<uint8_t>(std::min(large_mag, 255.0f));
    TEST_ASSERT_EQUAL_UINT8(255, clamped);
}

// ============================================================================
// DATA INTEGRITY TESTS
// ============================================================================

void test_subcarrier_extraction_from_window(void) {
    // Write 5 packets with known pattern
    FILE* f = fopen(TEST_BUFFER_FILE, "wb");
    TEST_ASSERT_NOT_NULL(f);
    
    uint16_t num_packets = 5;
    for (uint16_t p = 0; p < num_packets; p++) {
        uint8_t magnitudes[NUM_SUBCARRIERS];
        for (uint8_t sc = 0; sc < NUM_SUBCARRIERS; sc++) {
            // Pattern: magnitude = packet_idx * 10 + subcarrier
            magnitudes[sc] = p * 10 + sc;
        }
        fwrite(magnitudes, 1, NUM_SUBCARRIERS, f);
    }
    fclose(f);
    
    // Read all packets
    f = fopen(TEST_BUFFER_FILE, "rb");
    TEST_ASSERT_NOT_NULL(f);
    
    std::vector<uint8_t> all_data(num_packets * NUM_SUBCARRIERS);
    fread(all_data.data(), 1, all_data.size(), f);
    fclose(f);
    
    // Extract subcarrier 10 across all packets
    uint8_t target_sc = 10;
    std::vector<float> sc_magnitudes(num_packets);
    
    for (uint16_t p = 0; p < num_packets; p++) {
        sc_magnitudes[p] = static_cast<float>(all_data[p * NUM_SUBCARRIERS + target_sc]);
    }
    
    // Verify: subcarrier 10 should have values 10, 20, 30, 40, 50
    TEST_ASSERT_EQUAL_FLOAT(10.0f, sc_magnitudes[0]);
    TEST_ASSERT_EQUAL_FLOAT(20.0f, sc_magnitudes[1]);
    TEST_ASSERT_EQUAL_FLOAT(30.0f, sc_magnitudes[2]);
    TEST_ASSERT_EQUAL_FLOAT(40.0f, sc_magnitudes[3]);
    TEST_ASSERT_EQUAL_FLOAT(50.0f, sc_magnitudes[4]);
}

void test_large_buffer_write_read(void) {
    // Test with realistic buffer size (500 packets)
    uint16_t buffer_size = 500;
    
    // Write
    FILE* f = fopen(TEST_BUFFER_FILE, "wb");
    TEST_ASSERT_NOT_NULL(f);
    
    uint8_t magnitudes[NUM_SUBCARRIERS];
    for (uint16_t p = 0; p < buffer_size; p++) {
        for (uint8_t i = 0; i < NUM_SUBCARRIERS; i++) {
            magnitudes[i] = (p * i) % 256;
        }
        fwrite(magnitudes, 1, NUM_SUBCARRIERS, f);
        
        // Flush periodically like add_packet does
        if (p % 100 == 0) {
            fflush(f);
        }
    }
    fclose(f);
    
    // Verify file size
    f = fopen(TEST_BUFFER_FILE, "rb");
    TEST_ASSERT_NOT_NULL(f);
    fseek(f, 0, SEEK_END);
    long size = ftell(f);
    TEST_ASSERT_EQUAL(buffer_size * NUM_SUBCARRIERS, size);
    
    // Read random window and verify
    uint16_t start = 250;
    uint16_t window = 100;
    fseek(f, start * NUM_SUBCARRIERS, SEEK_SET);
    
    std::vector<uint8_t> window_data(window * NUM_SUBCARRIERS);
    size_t read = fread(window_data.data(), 1, window_data.size(), f);
    TEST_ASSERT_EQUAL(window_data.size(), read);
    
    // Verify first packet in window (packet 250)
    for (uint8_t i = 0; i < NUM_SUBCARRIERS; i++) {
        uint8_t expected = (start * i) % 256;
        TEST_ASSERT_EQUAL_UINT8(expected, window_data[i]);
    }
    
    fclose(f);
    
    ESP_LOGI(TAG, "Large buffer test passed: %d packets, %ld bytes",
             buffer_size, size);
}

// ============================================================================
// CLEANUP TESTS
// ============================================================================

void test_file_removal(void) {
    // Create file
    FILE* f = fopen(TEST_BUFFER_FILE, "wb");
    TEST_ASSERT_NOT_NULL(f);
    fwrite("test", 1, 4, f);
    fclose(f);
    
    // Verify exists
    f = fopen(TEST_BUFFER_FILE, "rb");
    TEST_ASSERT_NOT_NULL(f);
    fclose(f);
    
    // Remove
    int result = remove(TEST_BUFFER_FILE);
    TEST_ASSERT_EQUAL(0, result);
    
    // Verify removed
    f = fopen(TEST_BUFFER_FILE, "rb");
    TEST_ASSERT_NULL(f);
}

// ============================================================================
// ENTRY POINT
// ============================================================================

int process(void) {
    UNITY_BEGIN();
    
    // File write tests
    RUN_TEST(test_file_write_single_packet);
    RUN_TEST(test_file_write_multiple_packets);
    
    // File read tests
    RUN_TEST(test_file_read_single_packet);
    RUN_TEST(test_file_read_window_from_middle);
    
    // Magnitude conversion tests
    RUN_TEST(test_magnitude_uint8_conversion);
    RUN_TEST(test_magnitude_clamping_to_255);
    
    // Data integrity tests
    RUN_TEST(test_subcarrier_extraction_from_window);
    RUN_TEST(test_large_buffer_write_read);
    
    // Cleanup tests
    RUN_TEST(test_file_removal);
    
    return UNITY_END();
}

#if defined(ESP_PLATFORM)
extern "C" void app_main(void) { process(); }
#else
int main(int argc, char **argv) { return process(); }
#endif

