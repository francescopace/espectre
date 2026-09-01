/*
 * ESPectre - Native Recovery Button Service Tests
 * SPDX-License-Identifier: GPL-3.0-only
 * Commercial licensing available under separate agreement; see LICENSING.md.
 */
#include "native_frontend_test_support.h"

void test_native_recovery_button_requires_one_complete_long_press(void) {
  unsigned callbacks = 0U;
  RecoveryButtonService button(3000U, [&callbacks]() { ++callbacks; });

  button.update(true, 100U);
  button.update(true, 3099U);
  TEST_ASSERT_EQUAL(0U, callbacks);
  button.update(true, 3100U);
  button.update(true, 8000U);
  TEST_ASSERT_EQUAL(1U, callbacks);

  button.update(false, 8001U);
  button.update(true, UINT32_MAX - 1000U);
  button.update(true, 1999U);
  TEST_ASSERT_EQUAL(2U, callbacks);
}


int main(int argc, char **argv) {
  (void) argc;
  (void) argv;
  UNITY_BEGIN();
  RUN_TEST(test_native_recovery_button_requires_one_complete_long_press);
  return UNITY_END();
}
