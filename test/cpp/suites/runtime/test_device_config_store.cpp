/*
 * ESPectre - Device Config Store and CSI Payload Normalizer Tests
 *
 * Covers NVS-backed runtime config persistence and HT20 CSI payload
 * normalization helpers.
 *
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * License: GPLv3
 */
#include "test_harness.h"

#include <cstring>

#include "csi_payload_normalizer.h"
#include "device_config_store.h"
#include "runtime_detector_store.h"
#include "nvs.h"
#include "csi_format.h"

using namespace espectre;

void setUp(void) { nvs_mock_reset(); }

void tearDown(void) {}

void test_runtime_detector_store_round_trips_and_validates_values(void) {
  DetectionAlgorithm algorithm = DetectionAlgorithm::CLASSIC;
  bool has_saved_value = true;
  TEST_ASSERT_EQUAL(ESP_OK, load_runtime_detection_algorithm(&algorithm, &has_saved_value));
  TEST_ASSERT_FALSE(has_saved_value);
  TEST_ASSERT_EQUAL(ESP_ERR_INVALID_ARG, load_runtime_detection_algorithm(nullptr, &has_saved_value));
  TEST_ASSERT_EQUAL(ESP_ERR_INVALID_ARG, save_runtime_detection_algorithm(static_cast<DetectionAlgorithm>(99)));

  TEST_ASSERT_EQUAL(ESP_OK, save_runtime_detection_algorithm(DetectionAlgorithm::ML));
  TEST_ASSERT_EQUAL(ESP_OK, load_runtime_detection_algorithm(&algorithm, &has_saved_value));
  TEST_ASSERT_TRUE(has_saved_value);
  TEST_ASSERT_TRUE(algorithm == DetectionAlgorithm::ML);

  nvs_mock_put_str("detector", "pca");
  TEST_ASSERT_EQUAL(ESP_ERR_INVALID_STATE, load_runtime_detection_algorithm(&algorithm, &has_saved_value));
}

void test_wifi_config_store_handles_missing_namespace_and_invalid_args(void) {
  TEST_ASSERT_EQUAL(ESP_ERR_INVALID_ARG, load_stored_wifi_config(nullptr));

  StoredWifiConfig config;
  nvs_mock_set_open_result(ESP_ERR_NVS_NOT_FOUND);
  TEST_ASSERT_EQUAL(ESP_OK, load_stored_wifi_config(&config));
  TEST_ASSERT_FALSE(config.has_saved_config);
  TEST_ASSERT_TRUE(config.ssid.empty());

  nvs_mock_set_open_result(ESP_FAIL);
  TEST_ASSERT_EQUAL(ESP_FAIL, load_stored_wifi_config(&config));

  nvs_mock_set_open_result(ESP_ERR_NVS_NOT_FOUND);
  TEST_ASSERT_EQUAL(ESP_OK, clear_stored_wifi_config());
}

void test_wifi_config_store_round_trips_and_clears_saved_values(void) {
  StoredWifiConfig stored;
  stored.ssid = "LabSSID";
  stored.password = "top-secret";
  stored.bssid = "aa:bb:cc:dd:ee:ff";
  stored.channel = 11;

  TEST_ASSERT_EQUAL(ESP_OK, save_stored_wifi_config(stored));

  StoredWifiConfig loaded;
  TEST_ASSERT_EQUAL(ESP_OK, load_stored_wifi_config(&loaded));
  TEST_ASSERT_TRUE(loaded.has_saved_config);
  TEST_ASSERT_EQUAL_STRING("LabSSID", loaded.ssid.c_str());
  TEST_ASSERT_EQUAL_STRING("top-secret", loaded.password.c_str());
  TEST_ASSERT_EQUAL_STRING("aa:bb:cc:dd:ee:ff", loaded.bssid.c_str());
  TEST_ASSERT_EQUAL_UINT8(11, loaded.channel);

  TEST_ASSERT_EQUAL(ESP_OK, clear_stored_wifi_config());
  TEST_ASSERT_EQUAL(ESP_OK, load_stored_wifi_config(&loaded));
  TEST_ASSERT_FALSE(loaded.has_saved_config);
  TEST_ASSERT_TRUE(loaded.ssid.empty());
  TEST_ASSERT_EQUAL_UINT8(0, loaded.channel);
}

void test_wifi_config_store_marks_saved_when_only_ssid_exists(void) {
  nvs_mock_put_str("wifi_ssid", "SSIDOnly");

  StoredWifiConfig loaded;
  TEST_ASSERT_EQUAL(ESP_OK, load_stored_wifi_config(&loaded));

  TEST_ASSERT_TRUE(loaded.has_saved_config);
  TEST_ASSERT_EQUAL_STRING("SSIDOnly", loaded.ssid.c_str());
  TEST_ASSERT_TRUE(loaded.password.empty());
  TEST_ASSERT_TRUE(loaded.bssid.empty());
  TEST_ASSERT_EQUAL_UINT8(0, loaded.channel);
}

void test_device_config_store_handles_missing_namespace_and_invalid_args(void) {
  TEST_ASSERT_EQUAL(ESP_ERR_INVALID_ARG, load_stored_device_config(nullptr, nullptr));

  EspectreDeviceConfig config;
  bool has_saved_config = true;
  nvs_mock_set_open_result(ESP_ERR_NVS_NOT_FOUND);
  TEST_ASSERT_EQUAL(ESP_OK, load_stored_device_config(&config, &has_saved_config));
  TEST_ASSERT_FALSE(has_saved_config);

  nvs_mock_set_open_result(ESP_FAIL);
  TEST_ASSERT_EQUAL(ESP_FAIL, load_stored_device_config(&config, &has_saved_config));

  nvs_mock_set_open_result(ESP_ERR_NVS_NOT_FOUND);
  TEST_ASSERT_EQUAL(ESP_OK, clear_stored_device_config());
}

void test_device_config_store_round_trips_current_fields(void) {
  EspectreDeviceConfig stored;
  stored.device_label = "Office Node";
  stored.mqtt_host = "mqtt.local";
  stored.mqtt_port = 2883;
  stored.mqtt_username = "user";
  stored.mqtt_password = "pass";
  stored.topic_prefix = "custom/topic";

  TEST_ASSERT_EQUAL(ESP_OK, save_stored_device_config(stored));

  EspectreDeviceConfig loaded;
  bool has_saved_config = false;
  TEST_ASSERT_EQUAL(ESP_OK, load_stored_device_config(&loaded, &has_saved_config));

  TEST_ASSERT_TRUE(has_saved_config);
  TEST_ASSERT_EQUAL_STRING("Office Node", loaded.device_label.c_str());
  TEST_ASSERT_EQUAL_STRING("mqtt.local", loaded.mqtt_host.c_str());
  TEST_ASSERT_EQUAL(2883, loaded.mqtt_port);
  TEST_ASSERT_EQUAL_STRING("user", loaded.mqtt_username.c_str());
  TEST_ASSERT_EQUAL_STRING("pass", loaded.mqtt_password.c_str());
  TEST_ASSERT_EQUAL_STRING("custom/topic", loaded.topic_prefix.c_str());
}

void test_device_config_store_applies_defaults_without_legacy_fields(void) {
  nvs_mock_put_str("mqtt_host", "broker.local");
  nvs_mock_put_u16("mqtt_port", 0);

  EspectreDeviceConfig loaded;
  bool has_saved_config = false;
  TEST_ASSERT_EQUAL(ESP_OK, load_stored_device_config(&loaded, &has_saved_config));

  TEST_ASSERT_TRUE(has_saved_config);
  TEST_ASSERT_EQUAL_STRING(ESPECTRE_DEFAULT_DEVICE_LABEL, loaded.device_label.c_str());
  TEST_ASSERT_EQUAL_STRING("broker.local", loaded.mqtt_host.c_str());
  TEST_ASSERT_EQUAL(1883, loaded.mqtt_port);
  TEST_ASSERT_EQUAL_STRING(ESPECTRE_TOPIC_PREFIX, loaded.topic_prefix.c_str());
}

void test_device_config_store_reports_absence_when_no_fields_are_saved(void) {
  EspectreDeviceConfig loaded;
  bool has_saved_config = true;

  TEST_ASSERT_EQUAL(ESP_OK, load_stored_device_config(&loaded, &has_saved_config));
  TEST_ASSERT_FALSE(has_saved_config);
}

void test_device_config_store_clear_removes_all_current_keys(void) {
  nvs_mock_put_str("device_label", "Label");
  nvs_mock_put_str("mqtt_host", "mqtt.local");
  nvs_mock_put_u16("mqtt_port", 1884);
  nvs_mock_put_str("mqtt_user", "user");
  nvs_mock_put_str("mqtt_pass", "pass");
  nvs_mock_put_str("topic_prefix", "custom/topic");

  TEST_ASSERT_EQUAL(ESP_OK, clear_stored_device_config());

  EspectreDeviceConfig loaded;
  bool has_saved_config = true;
  TEST_ASSERT_EQUAL(ESP_OK, load_stored_device_config(&loaded, &has_saved_config));
  TEST_ASSERT_FALSE(has_saved_config);
}

void test_normalize_ht20_csi_payload_handles_supported_lengths(void) {
  int8_t input[HT20_CSI_LEN] = {0};
  int8_t short_input[HT20_CSI_LEN_SHORT] = {0};
  int8_t remap[HT20_CSI_LEN] = {0};
  for (size_t i = 0; i < HT20_CSI_LEN; ++i) {
    input[i] = static_cast<int8_t>(i);
    if (i < HT20_CSI_LEN_SHORT) {
      short_input[i] = static_cast<int8_t>(i + 1);
    }
  }

  const auto exact = normalize_ht20_csi_payload(input, HT20_CSI_LEN, remap, sizeof(remap));
  TEST_ASSERT_TRUE(exact.valid());
  TEST_ASSERT_TRUE(exact.data == input);
  TEST_ASSERT_EQUAL(HT20_CSI_LEN, static_cast<int>(exact.len));
  TEST_ASSERT_TRUE(exact.tag == NormalizedCSIPayloadTag::NONE);

  const auto doubled = normalize_ht20_csi_payload(input, HT20_CSI_LEN_DOUBLE, remap, sizeof(remap));
  TEST_ASSERT_TRUE(doubled.valid());
  TEST_ASSERT_TRUE(doubled.data == input);
  TEST_ASSERT_EQUAL(HT20_CSI_LEN, static_cast<int>(doubled.len));
  TEST_ASSERT_TRUE(doubled.tag == NormalizedCSIPayloadTag::DOUBLE_HT20);

  const auto remapped = normalize_ht20_csi_payload(short_input, HT20_CSI_LEN_SHORT, remap, sizeof(remap));
  TEST_ASSERT_TRUE(remapped.valid());
  TEST_ASSERT_TRUE(remapped.data == remap);
  TEST_ASSERT_EQUAL(HT20_CSI_LEN, static_cast<int>(remapped.len));
  TEST_ASSERT_TRUE(remapped.tag == NormalizedCSIPayloadTag::HT57_TO_64);
  TEST_ASSERT_EQUAL_INT8(0, remap[0]);
  TEST_ASSERT_EQUAL_INT8(1, remap[HT20_CSI_LEN_SHORT_LEFT_PAD]);
  TEST_ASSERT_EQUAL_INT8(static_cast<int8_t>(HT20_CSI_LEN_SHORT), remap[HT20_CSI_LEN_SHORT_LEFT_PAD + HT20_CSI_LEN_SHORT - 1]);

  const auto remapped_double =
      normalize_ht20_csi_payload(short_input, HT20_CSI_LEN_SHORT_DOUBLE, remap, sizeof(remap));
  TEST_ASSERT_TRUE(remapped_double.valid());
  TEST_ASSERT_TRUE(remapped_double.tag == NormalizedCSIPayloadTag::DOUBLE_HT57_TO_64);
}

void test_normalize_ht20_csi_payload_rejects_invalid_inputs_and_renders_tags(void) {
  int8_t input[HT20_CSI_LEN_SHORT] = {0};
  int8_t remap[HT20_CSI_LEN] = {0};

  const auto null_payload = normalize_ht20_csi_payload(nullptr, HT20_CSI_LEN, remap, sizeof(remap));
  TEST_ASSERT_FALSE(null_payload.valid());

  const auto bad_length = normalize_ht20_csi_payload(input, 42, remap, sizeof(remap));
  TEST_ASSERT_FALSE(bad_length.valid());

  const auto missing_buffer = normalize_ht20_csi_payload(input, HT20_CSI_LEN_SHORT, nullptr, sizeof(remap));
  TEST_ASSERT_FALSE(missing_buffer.valid());

  const auto short_buffer = normalize_ht20_csi_payload(input, HT20_CSI_LEN_SHORT, remap, HT20_CSI_LEN_SHORT);
  TEST_ASSERT_FALSE(short_buffer.valid());

  TEST_ASSERT_EQUAL_STRING("none", normalized_csi_payload_tag_to_string(NormalizedCSIPayloadTag::NONE));
  TEST_ASSERT_EQUAL_STRING("double_ht20", normalized_csi_payload_tag_to_string(NormalizedCSIPayloadTag::DOUBLE_HT20));
  TEST_ASSERT_EQUAL_STRING("ht57_to_64", normalized_csi_payload_tag_to_string(NormalizedCSIPayloadTag::HT57_TO_64));
  TEST_ASSERT_EQUAL_STRING("double_ht57_to_64",
                           normalized_csi_payload_tag_to_string(NormalizedCSIPayloadTag::DOUBLE_HT57_TO_64));
  TEST_ASSERT_EQUAL_STRING("unknown", normalized_csi_payload_tag_to_string(static_cast<NormalizedCSIPayloadTag>(99)));
}

int process(void) {
  UNITY_BEGIN();
  RUN_TEST(test_wifi_config_store_handles_missing_namespace_and_invalid_args);
  RUN_TEST(test_wifi_config_store_round_trips_and_clears_saved_values);
  RUN_TEST(test_wifi_config_store_marks_saved_when_only_ssid_exists);
  RUN_TEST(test_device_config_store_handles_missing_namespace_and_invalid_args);
  RUN_TEST(test_device_config_store_round_trips_current_fields);
  RUN_TEST(test_device_config_store_applies_defaults_without_legacy_fields);
  RUN_TEST(test_device_config_store_reports_absence_when_no_fields_are_saved);
  RUN_TEST(test_device_config_store_clear_removes_all_current_keys);
  RUN_TEST(test_runtime_detector_store_round_trips_and_validates_values);
  RUN_TEST(test_normalize_ht20_csi_payload_handles_supported_lengths);
  RUN_TEST(test_normalize_ht20_csi_payload_rejects_invalid_inputs_and_renders_tags);
  return UNITY_END();
}

#if defined(ESP_PLATFORM)
extern "C" void app_main(void) { process(); }
#else
int main(int argc, char **argv) {
  (void) argc;
  (void) argv;
  return process();
}
#endif
