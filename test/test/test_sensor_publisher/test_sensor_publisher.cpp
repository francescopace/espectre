/*
 * ESPectre - SensorPublisher Unit Tests
 *
 * Tests split publishing of the motion binary sensor and movement metric.
 */

#include <unity.h>
#include "sensor_publisher.h"
#include "esphome/components/binary_sensor/binary_sensor.h"
#include "esphome/components/sensor/sensor.h"
#include "sensor_publisher.cpp"

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

int process(void) {
    UNITY_BEGIN();
    RUN_TEST(test_sensor_publisher_publish_motion_binary_only);
    RUN_TEST(test_sensor_publisher_publish_movement_metric_only);
    return UNITY_END();
}

#if defined(ESP_PLATFORM)
extern "C" void app_main(void) { process(); }
#else
int main(int argc, char **argv) { return process(); }
#endif
