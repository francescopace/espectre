/*
 * ESPectre - SensorPublisher Unit Tests
 *
 * Tests split publishing of the motion binary sensor and movement metric.
 */

#include "test_harness.h"
#include "sensor_publisher.h"
#include "esphome/components/binary_sensor/binary_sensor.h"
#include "esphome/components/sensor/sensor.h"

using namespace esphome::espectre;

void test_sensor_publisher_publish_motion_binary_only(void) {
    SensorPublisher publisher;
    esphome::binary_sensor::BinarySensor binary_sensor;
    esphome::sensor::Sensor movement_sensor;

    publisher.set_motion_binary_sensor(&binary_sensor);
    publisher.set_movement_sensor(&movement_sensor);
    publisher.publish_motion_binary(MotionState::MOTION);

    TEST_ASSERT_TRUE(binary_sensor.has_state());
    TEST_ASSERT_TRUE(binary_sensor.get_state());
    TEST_ASSERT_EQUAL(1, binary_sensor.get_publish_count());
    TEST_ASSERT_FALSE(movement_sensor.has_state());
    TEST_ASSERT_EQUAL(0, movement_sensor.get_publish_count());
}

void test_sensor_publisher_publish_movement_metric_only(void) {
    SensorPublisher publisher;
    esphome::binary_sensor::BinarySensor binary_sensor;
    esphome::sensor::Sensor movement_sensor;

    publisher.set_motion_binary_sensor(&binary_sensor);
    publisher.set_movement_sensor(&movement_sensor);
    publisher.publish_movement_metric(6.5f);

    TEST_ASSERT_FALSE(binary_sensor.has_state());
    TEST_ASSERT_EQUAL(0, binary_sensor.get_publish_count());
    TEST_ASSERT_TRUE(movement_sensor.has_state());
    TEST_ASSERT_EQUAL_FLOAT(6.5f, movement_sensor.get_state());
    TEST_ASSERT_EQUAL(1, movement_sensor.get_publish_count());
}

void test_sensor_publisher_configuration_helpers(void) {
    SensorPublisher publisher;
    esphome::binary_sensor::BinarySensor binary_sensor;
    esphome::sensor::Sensor movement_sensor;

    TEST_ASSERT_FALSE(publisher.has_motion_binary_sensor());
    TEST_ASSERT_FALSE(publisher.has_movement_sensor());

    publisher.set_motion_binary_sensor(&binary_sensor);
    publisher.set_movement_sensor(&movement_sensor);

    TEST_ASSERT_TRUE(publisher.has_motion_binary_sensor());
    TEST_ASSERT_TRUE(publisher.has_movement_sensor());
}

void test_sensor_publisher_log_status_handles_runtime_snapshot(void) {
    SensorPublisher publisher;
    RuntimeSnapshot snapshot{};
    snapshot.motion_state = MotionState::MOTION;
    snapshot.movement_metric = 6.5f;
    snapshot.threshold = 5.0f;

    publisher.log_status("sensor_publisher", snapshot, 25);
    publisher.log_status("sensor_publisher", snapshot, 25);
    publisher.reset_rate_counter();
    publisher.log_status("sensor_publisher", snapshot, 25);

    TEST_ASSERT_TRUE(true);
}

void test_sensor_publisher_log_status_ignores_null_tag(void) {
    SensorPublisher publisher;
    RuntimeSnapshot snapshot{};
    snapshot.motion_state = MotionState::IDLE;

    publisher.log_status(nullptr, snapshot, 10);
    TEST_ASSERT_TRUE(true);
}

int process(void) {
    UNITY_BEGIN();
    RUN_TEST(test_sensor_publisher_publish_motion_binary_only);
    RUN_TEST(test_sensor_publisher_publish_movement_metric_only);
    RUN_TEST(test_sensor_publisher_configuration_helpers);
    RUN_TEST(test_sensor_publisher_log_status_handles_runtime_snapshot);
    RUN_TEST(test_sensor_publisher_log_status_ignores_null_tag);
    return UNITY_END();
}

#if defined(ESP_PLATFORM)
extern "C" void app_main(void) { process(); }
#else
int main(int argc, char **argv) { return process(); }
#endif
