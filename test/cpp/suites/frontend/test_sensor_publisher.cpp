/*
 * ESPectre - SensorPublisher Unit Tests
 *
 * Tests split publishing of the motion binary sensor and movement metric.
 *
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * SPDX-License-Identifier: GPL-3.0-only
 * Commercial licensing available under separate agreement; see LICENSING.md.
 */
#include "test_harness.h"
#include "sensor_publisher.h"
#include "esphome/components/binary_sensor/binary_sensor.h"
#include "esphome/components/sensor/sensor.h"

using namespace esphome::espectre_component;

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

int process(void) {
    UNITY_BEGIN();
    RUN_TEST(test_sensor_publisher_publish_motion_binary_only);
    RUN_TEST(test_sensor_publisher_publish_movement_metric_only);
    RUN_TEST(test_sensor_publisher_configuration_helpers);
    return UNITY_END();
}

#if defined(ESP_PLATFORM)
extern "C" void app_main(void) { process(); }
#else
int main(int argc, char **argv) { return process(); }
#endif
