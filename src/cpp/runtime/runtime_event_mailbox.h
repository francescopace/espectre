/*
 * ESPectre - Runtime Event Mailbox
 *
 * Bounded handoff for deferring runtime listener work to a frontend loop.
 *
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * SPDX-License-Identifier: GPL-3.0-only
 * Commercial licensing available under separate agreement; see LICENSING.md.
 */
#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <type_traits>

#if defined(ESP_PLATFORM)
#include "freertos/FreeRTOS.h"
#include "freertos/portmacro.h"
#endif

#include "runtime_snapshot.h"

namespace espectre {

/**
 * Fixed-memory handoff from runtime listener callbacks to a frontend loop.
 *
 * Motion-state changes retain FIFO ordering. If their bounded queue fills,
 * posting keeps the newest state by discarding the oldest unconsumed state and
 * reports the loss to the caller. Live telemetry is replaceable by design, so
 * repeated posts coalesce to the newest snapshot.
 *
 * All operations serialize their short value copies and are safe to call from
 * different tasks. Drain the mailbox from one owning frontend task. This class
 * is not intended for raw CSI callbacks or other ISR/driver contexts.
 *
 * The mailbox never allocates and never invokes frontend code. The frontend
 * remains responsible for publication, transport policy, and error handling.
 */
class RuntimeEventMailbox {
 public:
  /** Maximum number of ordered motion-state changes retained. */
  static constexpr size_t kMotionStateCapacity = 4U;

  /** Construct an empty mailbox. */
  RuntimeEventMailbox() = default;

  /** Mailboxes own synchronization state and cannot be copied. */
  RuntimeEventMailbox(const RuntimeEventMailbox &) = delete;
  /** Mailboxes own synchronization state and cannot be assigned. */
  RuntimeEventMailbox &operator=(const RuntimeEventMailbox &) = delete;

  /**
   * Append a motion-state snapshot for ordered delivery.
   *
   * When the queue is full, the oldest unconsumed snapshot is discarded so
   * the eventual consumer still observes the newest state.
   *
   * @param snapshot Runtime state captured by `on_motion_state_changed()`.
   * @return `true` when every queued state was retained, or `false` when the
   * oldest state had to be discarded.
   */
  bool post_motion_state(const RuntimeSnapshot &snapshot) {
    enter_critical_();
    const bool retained_all = motion_count_ != kMotionStateCapacity;
    if (!retained_all) {
      motion_read_index_ = next_motion_index_(motion_read_index_);
      --motion_count_;
      ++motion_state_drops_total_;
    }
    motion_states_[motion_write_index_] = snapshot;
    motion_write_index_ = next_motion_index_(motion_write_index_);
    ++motion_count_;
    exit_critical_();
    return retained_all;
  }

  /**
   * Consume the oldest pending motion-state snapshot.
   *
   * @param snapshot Receives the snapshot when one is pending and remains
   * unchanged otherwise.
   * @return `true` when a snapshot was consumed.
   */
  bool take_motion_state(RuntimeSnapshot &snapshot) {
    enter_critical_();
    if (motion_count_ == 0U) {
      exit_critical_();
      return false;
    }
    snapshot = motion_states_[motion_read_index_];
    motion_read_index_ = next_motion_index_(motion_read_index_);
    --motion_count_;
    exit_critical_();
    return true;
  }

  /**
   * Store the newest replaceable live-telemetry snapshot.
   *
   * @param snapshot Runtime state with the callback's movement and threshold.
   */
  void post_live_telemetry(const RuntimeSnapshot &snapshot) {
    enter_critical_();
    live_telemetry_ = snapshot;
    live_telemetry_pending_ = true;
    exit_critical_();
  }

  /**
   * Consume the newest pending live-telemetry snapshot.
   *
   * @param snapshot Receives the snapshot when one is pending and remains
   * unchanged otherwise.
   * @return `true` when a snapshot was consumed.
   */
  bool take_live_telemetry(RuntimeSnapshot &snapshot) {
    enter_critical_();
    if (!live_telemetry_pending_) {
      exit_critical_();
      return false;
    }
    snapshot = live_telemetry_;
    live_telemetry_pending_ = false;
    exit_critical_();
    return true;
  }

  /** Store the newest replaceable threshold update. */
  void post_threshold(float threshold) {
    enter_critical_();
    threshold_ = threshold;
    threshold_pending_ = true;
    exit_critical_();
  }

  /** Consume the newest pending threshold update. */
  bool take_threshold(float &threshold) {
    enter_critical_();
    if (!threshold_pending_) {
      exit_critical_();
      return false;
    }
    threshold = threshold_;
    threshold_pending_ = false;
    exit_critical_();
    return true;
  }

  /** Cumulative ordered motion events discarded since construction. */
  uint32_t motion_state_drops_total() const {
    enter_critical_();
    const uint32_t drops = motion_state_drops_total_;
    exit_critical_();
    return drops;
  }

  /** Discard every unconsumed runtime event. */
  void clear() {
    enter_critical_();
    motion_read_index_ = 0U;
    motion_write_index_ = 0U;
    motion_count_ = 0U;
    live_telemetry_pending_ = false;
    threshold_pending_ = false;
    exit_critical_();
  }

 private:
  static_assert(std::is_trivially_copyable<RuntimeSnapshot>::value,
                "RuntimeEventMailbox snapshots must be trivially copyable");
  static_assert(std::is_nothrow_copy_assignable<RuntimeSnapshot>::value,
                "RuntimeEventMailbox snapshots must be nothrow copy assignable");

  void enter_critical_() const {
#if defined(ESP_PLATFORM)
    portENTER_CRITICAL_SAFE(&mux_);
#else
    mutex_.lock();
#endif
  }

  void exit_critical_() const {
#if defined(ESP_PLATFORM)
    portEXIT_CRITICAL_SAFE(&mux_);
#else
    mutex_.unlock();
#endif
  }

#if defined(ESP_PLATFORM)
  mutable portMUX_TYPE mux_ = portMUX_INITIALIZER_UNLOCKED;
#else
  mutable std::mutex mutex_;
#endif

  static constexpr size_t next_motion_index_(size_t index) {
    return index + 1U == kMotionStateCapacity ? 0U : index + 1U;
  }

  std::array<RuntimeSnapshot, kMotionStateCapacity> motion_states_{};
  RuntimeSnapshot live_telemetry_{};
  float threshold_{0.0f};
  size_t motion_read_index_{0U};
  size_t motion_write_index_{0U};
  size_t motion_count_{0U};
  uint32_t motion_state_drops_total_{0U};
  bool live_telemetry_pending_{false};
  bool threshold_pending_{false};
};

}  // namespace espectre
