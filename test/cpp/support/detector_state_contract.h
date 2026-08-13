/*
 * ESPectre - Shared Detector State Contract
 *
 * One contract every BaseDetector subclass owes the runtime: anything that
 * invalidates the window must also invalidate the evaluation derived from it.
 * Kept here rather than in one detector's suite so a new detector inherits the
 * check instead of repeating the divergence it was written for.
 *
 * Include after the `#define private public` / `#define protected public`
 * block, because the contract sets `current_metric_` directly. Driving the
 * metric through real traffic is not an option: synthetic packets land far
 * outside the distribution the MLP was fitted on and saturate it to exactly
 * 0.0, which would make every assertion below vacuous for ML.
 *
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * SPDX-License-Identifier: GPL-3.0-only
 * Commercial licensing available under separate agreement; see LICENSING.md.
 */
#pragma once

#include "test_harness.h"

namespace espectre {
namespace test_support {

/**
 * clear_buffer() is the cold restart: the window and the last evaluation both go.
 *
 * Regression guard for the divergence where ML kept its last probability here
 * and the runtime published that stale value until the ring refilled, while
 * Lightweight reported 0 through the very same code path.
 */
template <typename Detector>
inline void assert_clear_buffer_drops_evaluation_state(Detector &detector) {
    detector.current_metric_ = 0.91f;
    detector.state_ = MotionState::MOTION;

    detector.clear_buffer();

    TEST_ASSERT_EQUAL_FLOAT(0.0f, detector.get_motion_metric());
    TEST_ASSERT_TRUE(detector.get_state() == MotionState::IDLE);
    TEST_ASSERT_EQUAL(0, detector.get_buffer_count());
}

/**
 * reset() is the warm restart: the window survives, the last evaluation does not.
 */
template <typename Detector>
inline void assert_reset_drops_evaluation_state(Detector &detector) {
    detector.current_metric_ = 0.91f;
    detector.state_ = MotionState::MOTION;

    detector.reset();

    TEST_ASSERT_EQUAL_FLOAT(0.0f, detector.get_motion_metric());
    TEST_ASSERT_TRUE(detector.get_state() == MotionState::IDLE);
}

/**
 * A not-ready evaluation must not resurrect either half.
 *
 * The detector is left with an empty window on purpose, so update_state() takes
 * its early-return branch.
 */
template <typename Detector>
inline void assert_not_ready_evaluation_stays_idle(Detector &detector) {
    detector.clear_buffer();
    detector.current_metric_ = 0.91f;
    detector.state_ = MotionState::MOTION;

    TEST_ASSERT_FALSE(detector.is_ready());
    detector.update_state();

    TEST_ASSERT_EQUAL_FLOAT(0.0f, detector.get_motion_metric());
    TEST_ASSERT_TRUE(detector.get_state() == MotionState::IDLE);
}

}  // namespace test_support
}  // namespace espectre
