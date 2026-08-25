/*
 * ESPectre - Pending Event Unit Tests
 *
 * Unit tests for Pending Event.
 *
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * SPDX-License-Identifier: GPL-3.0-only
 * Commercial licensing available under separate agreement; see LICENSING.md.
 */
#include "test_harness.h"

#include "base_detector.h"
#include "pending_event.h"
#include "pending_queue.h"

using namespace espectre;

void setUp(void) {}

void tearDown(void) {}

void test_pending_event_take_returns_false_when_nothing_posted(void) {
  PendingEvent<uint32_t> event;
  uint32_t value = 42U;

  TEST_ASSERT_FALSE(event.take(value));
  TEST_ASSERT_EQUAL(42U, value);
}

void test_pending_event_roundtrips_payload_once(void) {
  PendingEvent<uint32_t> event;
  uint32_t value = 0U;

  event.post(7U);
  TEST_ASSERT_TRUE(event.take(value));
  TEST_ASSERT_EQUAL(7U, value);
  TEST_ASSERT_FALSE(event.take(value));
}

void test_pending_event_coalesces_to_most_recent_post(void) {
  PendingEvent<MotionState, uint32_t> event;
  MotionState state = MotionState::IDLE;
  uint32_t count = 0U;

  event.post(MotionState::MOTION, 100U);
  event.post(MotionState::IDLE, 250U);

  TEST_ASSERT_TRUE(event.take(state, count));
  TEST_ASSERT_TRUE(state == MotionState::IDLE);
  TEST_ASSERT_EQUAL(250U, count);
  TEST_ASSERT_FALSE(event.take(state, count));
}

void test_pending_event_clear_drops_unconsumed_event(void) {
  PendingEvent<float, float> event;
  float movement = -1.0f;
  float threshold = -1.0f;

  event.post(0.5f, 1.5f);
  event.clear();
  TEST_ASSERT_FALSE(event.take(movement, threshold));
  TEST_ASSERT_EQUAL_FLOAT(-1.0f, movement);
  TEST_ASSERT_EQUAL_FLOAT(-1.0f, threshold);

  event.post(0.25f, 2.0f);
  TEST_ASSERT_TRUE(event.take(movement, threshold));
  TEST_ASSERT_EQUAL_FLOAT(0.25f, movement);
  TEST_ASSERT_EQUAL_FLOAT(2.0f, threshold);
}

void test_pending_event_supports_empty_payload(void) {
  PendingEvent<> event;

  TEST_ASSERT_FALSE(event.take());
  event.post();
  event.post();
  TEST_ASSERT_TRUE(event.take());
  TEST_ASSERT_FALSE(event.take());
}

void test_pending_queue_overwrite_retains_the_newest_ordered_records(void) {
  PendingQueue<uint32_t, 2U> queue;
  uint32_t value = 0U;

  TEST_ASSERT_TRUE(queue.post_overwrite_oldest(1U));
  TEST_ASSERT_TRUE(queue.post_overwrite_oldest(2U));
  TEST_ASSERT_FALSE(queue.post_overwrite_oldest(3U));
  TEST_ASSERT_EQUAL(2U, queue.size());
  TEST_ASSERT_TRUE(queue.take(value));
  TEST_ASSERT_EQUAL(2U, value);
  TEST_ASSERT_TRUE(queue.take(value));
  TEST_ASSERT_EQUAL(3U, value);
  TEST_ASSERT_FALSE(queue.take(value));
}

int process(void) {
  UNITY_BEGIN();
  RUN_TEST(test_pending_event_take_returns_false_when_nothing_posted);
  RUN_TEST(test_pending_event_roundtrips_payload_once);
  RUN_TEST(test_pending_event_coalesces_to_most_recent_post);
  RUN_TEST(test_pending_event_clear_drops_unconsumed_event);
  RUN_TEST(test_pending_event_supports_empty_payload);
  RUN_TEST(test_pending_queue_overwrite_retains_the_newest_ordered_records);
  return UNITY_END();
}

#if defined(ESP_PLATFORM)
extern "C" void app_main(void) { process(); }
#else
int main(int argc, char **argv) { return process(); }
#endif
