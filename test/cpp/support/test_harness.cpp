/*
 * ESPectre - Test Harness
 *
 * Minimal host-side test harness helpers for native C++ suites.
 *
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * SPDX-License-Identifier: GPL-3.0-only
 * Commercial licensing available under separate agreement; see LICENSING.md.
 */
#include "test_harness.h"

#include <cstdio>
#include <string>

void setUp(void) __attribute__((weak));
void tearDown(void) __attribute__((weak));

void setUp(void) {}
void tearDown(void) {}

namespace espectre::test {
namespace {

struct TestSuiteState {
  int total{0};
  int failed{0};
  int skipped{0};
};

TestSuiteState g_state;

}  // namespace

int begin_suite() {
  g_state = TestSuiteState{};
  return 0;
}

void run_test(const char *name, void (*fn)()) {
  g_state.total++;
  bool setup_done = false;

  try {
    ::setUp();
    setup_done = true;
    fn();
    ::tearDown();
    std::printf("[PASS] %s\n", name);
  } catch (const TestSkipped &ex) {
    if (setup_done) {
      ::tearDown();
    }
    g_state.skipped++;
    std::printf("[SKIP] %s: %s\n", name, ex.what());
  } catch (const AssertionFailure &ex) {
    if (setup_done) {
      ::tearDown();
    }
    g_state.failed++;
    std::printf("[FAIL] %s: %s\n", name, ex.what());
  } catch (const std::exception &ex) {
    if (setup_done) {
      ::tearDown();
    }
    g_state.failed++;
    std::printf("[FAIL] %s: unexpected exception: %s\n", name, ex.what());
  } catch (...) {
    if (setup_done) {
      ::tearDown();
    }
    g_state.failed++;
    std::printf("[FAIL] %s: unknown exception\n", name);
  }
}

int end_suite() {
  std::printf("\nSummary: %d total, %d failed, %d skipped\n",
              g_state.total, g_state.failed, g_state.skipped);
  return g_state.failed == 0 ? 0 : 1;
}

[[noreturn]] void fail(const char *file, int line, const std::string &message) {
  throw AssertionFailure(std::string(file) + ":" + std::to_string(line) + ": " + message);
}

[[noreturn]] void skip(const char *file, int line, const std::string &message) {
  throw TestSkipped(std::string(file) + ":" + std::to_string(line) + ": " + message);
}

}  // namespace espectre::test
