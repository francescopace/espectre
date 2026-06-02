/*
 * ESPectre - Calibration File Storage Unit Tests
 *
 * Unit tests for the runtime calibration file buffer implementation.
 */

#include "test_harness.h"

#include <cstdio>
#include <cstdint>
#include <vector>

#include "calibration_file_buffer.h"
#include "utils.h"

using namespace esphome::espectre;

namespace {

static const char *const TEST_BUFFER_FILE = "/tmp/test_buffer.bin";

std::vector<int8_t> make_packet(int8_t i_value, int8_t q_value) {
    std::vector<int8_t> packet(HT20_NUM_SUBCARRIERS * 2);
    for (uint16_t sc = 0; sc < HT20_NUM_SUBCARRIERS; ++sc) {
        packet[sc * 2] = q_value;
        packet[sc * 2 + 1] = i_value;
    }
    return packet;
}

std::vector<uint8_t> read_all_bytes(const char *path) {
    FILE *file = fopen(path, "rb");
    TEST_ASSERT_NOT_NULL(file);

    fseek(file, 0, SEEK_END);
    long size = ftell(file);
    TEST_ASSERT_TRUE(size >= 0);
    rewind(file);

    std::vector<uint8_t> data(static_cast<std::size_t>(size));
    const size_t bytes_read = fread(data.data(), 1, data.size(), file);
    TEST_ASSERT_EQUAL(data.size(), bytes_read);
    fclose(file);
    return data;
}

}  // namespace

void setUp(void) { remove(TEST_BUFFER_FILE); }

void tearDown(void) { remove(TEST_BUFFER_FILE); }

void test_calibration_file_buffer_open_close_cycle(void) {
    CalibrationFileBuffer buffer;
    buffer.init(TEST_BUFFER_FILE, 4);

    TEST_ASSERT_TRUE(buffer.open_for_writing());
    TEST_ASSERT_TRUE(buffer.is_open());
    TEST_ASSERT_EQUAL(4, buffer.get_size());

    buffer.close();
    TEST_ASSERT_FALSE(buffer.is_open());
}

void test_calibration_file_buffer_write_packet_persists_runtime_format(void) {
    CalibrationFileBuffer buffer;
    buffer.init(TEST_BUFFER_FILE, 1);
    TEST_ASSERT_TRUE(buffer.open_for_writing());

    const auto packet = make_packet(3, 4);
    const bool full = buffer.write_packet(packet.data(), packet.size());
    TEST_ASSERT_TRUE(full);
    TEST_ASSERT_EQUAL(1, buffer.get_count());

    buffer.close();

    const std::vector<uint8_t> data = read_all_bytes(TEST_BUFFER_FILE);
    TEST_ASSERT_EQUAL(HT20_NUM_SUBCARRIERS, data.size());

    TEST_ASSERT_EQUAL_UINT8(0, data[0]);
    TEST_ASSERT_EQUAL_UINT8(0, data[HT20_GUARD_BAND_LOW - 1]);
    TEST_ASSERT_EQUAL_UINT8(5, data[HT20_GUARD_BAND_LOW]);
    TEST_ASSERT_EQUAL_UINT8(5, data[HT20_DC_SUBCARRIER - 1]);
    TEST_ASSERT_EQUAL_UINT8(0, data[HT20_DC_SUBCARRIER]);
    TEST_ASSERT_EQUAL_UINT8(5, data[HT20_DC_SUBCARRIER + 1]);
    TEST_ASSERT_EQUAL_UINT8(5, data[HT20_GUARD_BAND_HIGH]);
    TEST_ASSERT_EQUAL_UINT8(0, data[HT20_GUARD_BAND_HIGH + 1]);
}

void test_calibration_file_buffer_tracks_full_state(void) {
    CalibrationFileBuffer buffer;
    buffer.init(TEST_BUFFER_FILE, 2);
    TEST_ASSERT_TRUE(buffer.open_for_writing());

    const auto packet = make_packet(6, 8);

    TEST_ASSERT_FALSE(buffer.write_packet(packet.data(), packet.size()));
    TEST_ASSERT_EQUAL(1, buffer.get_count());
    TEST_ASSERT_FALSE(buffer.is_full());

    TEST_ASSERT_TRUE(buffer.write_packet(packet.data(), packet.size()));
    TEST_ASSERT_EQUAL(2, buffer.get_count());
    TEST_ASSERT_TRUE(buffer.is_full());

    buffer.close();
}

void test_calibration_file_buffer_rejects_invalid_packet_length(void) {
    CalibrationFileBuffer buffer;
    buffer.init(TEST_BUFFER_FILE, 1);
    TEST_ASSERT_TRUE(buffer.open_for_writing());

    const std::vector<int8_t> short_packet(32, 1);
    TEST_ASSERT_FALSE(buffer.write_packet(short_packet.data(), short_packet.size()));
    TEST_ASSERT_EQUAL(0, buffer.get_count());

    buffer.close();

    const std::vector<uint8_t> data = read_all_bytes(TEST_BUFFER_FILE);
    TEST_ASSERT_EQUAL(0, data.size());
}

void test_calibration_file_buffer_read_window_returns_requested_packets(void) {
    CalibrationFileBuffer buffer;
    buffer.init(TEST_BUFFER_FILE, 4);
    TEST_ASSERT_TRUE(buffer.open_for_writing());

    for (int i = 0; i < 4; ++i) {
        const auto packet = make_packet(static_cast<int8_t>(i + 1), 0);
        buffer.write_packet(packet.data(), packet.size());
    }
    buffer.close();

    TEST_ASSERT_TRUE(buffer.open_for_reading());
    const std::vector<uint8_t> window = buffer.read_window(1, 2);
    buffer.close();

    TEST_ASSERT_EQUAL(static_cast<size_t>(HT20_NUM_SUBCARRIERS * 2), window.size());
    TEST_ASSERT_EQUAL_UINT8(2, window[HT20_GUARD_BAND_LOW]);
    TEST_ASSERT_EQUAL_UINT8(0, window[HT20_DC_SUBCARRIER]);
    TEST_ASSERT_EQUAL_UINT8(3, window[HT20_NUM_SUBCARRIERS + HT20_GUARD_BAND_LOW]);
}

void test_calibration_file_buffer_remove_file_deletes_buffer(void) {
    CalibrationFileBuffer buffer;
    buffer.init(TEST_BUFFER_FILE, 1);
    TEST_ASSERT_TRUE(buffer.open_for_writing());

    const auto packet = make_packet(10, 0);
    buffer.write_packet(packet.data(), packet.size());
    buffer.close();

    FILE *before_remove = fopen(TEST_BUFFER_FILE, "rb");
    TEST_ASSERT_NOT_NULL(before_remove);
    fclose(before_remove);

    buffer.remove_file();

    FILE *after_remove = fopen(TEST_BUFFER_FILE, "rb");
    TEST_ASSERT_NULL(after_remove);
}

int process(void) {
    UNITY_BEGIN();
    RUN_TEST(test_calibration_file_buffer_open_close_cycle);
    RUN_TEST(test_calibration_file_buffer_write_packet_persists_runtime_format);
    RUN_TEST(test_calibration_file_buffer_tracks_full_state);
    RUN_TEST(test_calibration_file_buffer_rejects_invalid_packet_length);
    RUN_TEST(test_calibration_file_buffer_read_window_returns_requested_packets);
    RUN_TEST(test_calibration_file_buffer_remove_file_deletes_buffer);
    return UNITY_END();
}

#if defined(ESP_PLATFORM)
extern "C" void app_main(void) { process(); }
#else
int main(int argc, char **argv) { return process(); }
#endif

