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
#include "runtime_event_mailbox.h"

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

void test_runtime_event_mailbox_retains_motion_order(void) {
  RuntimeEventMailbox mailbox;
  RuntimeSnapshot posted;
  posted.motion_state = MotionState::MOTION;
  posted.calibration_packets = 1U;

  TEST_ASSERT_TRUE(mailbox.post_motion_state(posted));
  posted.motion_state = MotionState::IDLE;
  posted.calibration_packets = 2U;
  TEST_ASSERT_TRUE(mailbox.post_motion_state(posted));

  RuntimeSnapshot received;
  TEST_ASSERT_TRUE(mailbox.take_motion_state(received));
  TEST_ASSERT_TRUE(received.motion_state == MotionState::MOTION);
  TEST_ASSERT_EQUAL(1U, received.calibration_packets);
  TEST_ASSERT_TRUE(mailbox.take_motion_state(received));
  TEST_ASSERT_TRUE(received.motion_state == MotionState::IDLE);
  TEST_ASSERT_EQUAL(2U, received.calibration_packets);
  TEST_ASSERT_FALSE(mailbox.take_motion_state(received));
}

void test_runtime_event_mailbox_overflow_keeps_newest_motion_states(void) {
  RuntimeEventMailbox mailbox;
  RuntimeSnapshot posted;
  for (size_t index = 0U; index < RuntimeEventMailbox::kMotionStateCapacity; ++index) {
    posted.calibration_packets = static_cast<uint32_t>(index);
    TEST_ASSERT_TRUE(mailbox.post_motion_state(posted));
  }
  posted.calibration_packets =
      static_cast<uint32_t>(RuntimeEventMailbox::kMotionStateCapacity);
  TEST_ASSERT_FALSE(mailbox.post_motion_state(posted));
  TEST_ASSERT_EQUAL(1U, mailbox.motion_state_drops_total());

  RuntimeSnapshot received;
  for (size_t index = 1U; index <= RuntimeEventMailbox::kMotionStateCapacity; ++index) {
    TEST_ASSERT_TRUE(mailbox.take_motion_state(received));
    TEST_ASSERT_EQUAL(static_cast<uint32_t>(index), received.calibration_packets);
  }
  TEST_ASSERT_FALSE(mailbox.take_motion_state(received));
  mailbox.clear();
  TEST_ASSERT_EQUAL(1U, mailbox.motion_state_drops_total());
}

void test_runtime_event_mailbox_coalesces_live_telemetry(void) {
  RuntimeEventMailbox mailbox;
  RuntimeSnapshot posted;
  posted.movement_metric = 0.25f;
  posted.threshold = 0.4f;
  mailbox.post_live_telemetry(posted);
  posted.movement_metric = 0.75f;
  posted.threshold = 0.6f;
  mailbox.post_live_telemetry(posted);

  RuntimeSnapshot received;
  TEST_ASSERT_TRUE(mailbox.take_live_telemetry(received));
  TEST_ASSERT_EQUAL_FLOAT(0.75f, received.movement_metric);
  TEST_ASSERT_EQUAL_FLOAT(0.6f, received.threshold);
  TEST_ASSERT_FALSE(mailbox.take_live_telemetry(received));
}

void test_runtime_event_mailbox_coalesces_threshold_updates(void) {
  RuntimeEventMailbox mailbox;
  mailbox.post_threshold(0.4f);
  mailbox.post_threshold(0.6f);

  float threshold = 0.0f;
  TEST_ASSERT_TRUE(mailbox.take_threshold(threshold));
  TEST_ASSERT_EQUAL_FLOAT(0.6f, threshold);
  TEST_ASSERT_FALSE(mailbox.take_threshold(threshold));

  mailbox.post_threshold(0.8f);
  mailbox.clear();
  TEST_ASSERT_FALSE(mailbox.take_threshold(threshold));
}

void test_runtime_event_mailbox_clear_discards_both_event_classes(void) {
  RuntimeEventMailbox mailbox;
  RuntimeSnapshot snapshot;
  TEST_ASSERT_TRUE(mailbox.post_motion_state(snapshot));
  mailbox.post_live_telemetry(snapshot);
  mailbox.clear();

  TEST_ASSERT_FALSE(mailbox.take_motion_state(snapshot));
  TEST_ASSERT_FALSE(mailbox.take_live_telemetry(snapshot));
}

int process(void) {
  UNITY_BEGIN();
  RUN_TEST(test_pending_event_take_returns_false_when_nothing_posted);
  RUN_TEST(test_pending_event_roundtrips_payload_once);
  RUN_TEST(test_pending_event_coalesces_to_most_recent_post);
  RUN_TEST(test_pending_event_clear_drops_unconsumed_event);
  RUN_TEST(test_pending_event_supports_empty_payload);
  RUN_TEST(test_pending_queue_overwrite_retains_the_newest_ordered_records);
  RUN_TEST(test_runtime_event_mailbox_retains_motion_order);
  RUN_TEST(test_runtime_event_mailbox_overflow_keeps_newest_motion_states);
  RUN_TEST(test_runtime_event_mailbox_coalesces_live_telemetry);
  RUN_TEST(test_runtime_event_mailbox_coalesces_threshold_updates);
  RUN_TEST(test_runtime_event_mailbox_clear_discards_both_event_classes);
  return UNITY_END();
}

#if defined(ESP_PLATFORM)
extern "C" void app_main(void) { process(); }
#else
int main(int argc, char **argv) { return process(); }
#endif
