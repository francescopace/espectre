/*
 * ESPectre - Public sensing readiness gate
 *
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * SPDX-License-Identifier: GPL-3.0-only
 * Commercial licensing available under separate agreement; see LICENSING.md.
 */
#pragma once

#include <cstdint>

namespace espectre {

/** The first condition that keeps sensing from being ready, or READY. */
enum class SensingReadinessReason : uint8_t {
  READY,
  SENSING_STOPPED,  ///< Sensing services, Wi-Fi, or the lifecycle gate are down.
  CALIBRATING,
  NO_DETECTOR,
  INPUT_STALE,      ///< No admitted detector input within one detector window.
  WINDOW_FILLING,   ///< The detector window has not filled since its last reset.
  LOW_COVERAGE,     ///< The window is full but below the valid-slot floor.
};

inline const char *sensing_readiness_reason_name(SensingReadinessReason reason) {
  switch (reason) {
    case SensingReadinessReason::READY:
      return "ready";
    case SensingReadinessReason::SENSING_STOPPED:
      return "sensing_stopped";
    case SensingReadinessReason::CALIBRATING:
      return "calibrating";
    case SensingReadinessReason::NO_DETECTOR:
      return "no_detector";
    case SensingReadinessReason::INPUT_STALE:
      return "input_stale";
    case SensingReadinessReason::WINDOW_FILLING:
      return "window_filling";
    case SensingReadinessReason::LOW_COVERAGE:
      return "low_coverage";
  }
  return "unknown";
}

/** Instantaneous conditions behind public sensing readiness. */
struct SensingReadinessInputs {
  bool sensing_active{false};
  bool calibrating{false};
  bool has_detector{false};
  bool input_current{false};
  bool window_full{false};
  bool detector_ready{false};
};

/**
 * Public sensing readiness with a bounded tolerance for coverage dips.
 *
 * Temporal sampling leaves some slots empty even at the nominal packet rate,
 * so window coverage can dip under the detector's valid-slot floor for a few
 * hundred milliseconds while input keeps arriving. Once sensing is ready, the
 * gate holds readiness through such a dip for up to `coverage_grace_ms`. Every
 * other condition, including stale input and a window refilling after a
 * reset, clears readiness immediately.
 */
class SensingReadinessGate {
 public:
  enum class Edge : uint8_t {
    NONE,
    READY,         ///< Readiness became true.
    UNREADY,       ///< Readiness became false; see reason().
    DIP_ABSORBED,  ///< A coverage dip ended inside the grace period.
  };

  Edge update(const SensingReadinessInputs &inputs, uint32_t now_ms, uint32_t coverage_grace_ms) {
    reason_ = classify_(inputs);
    bool next = false;
    Edge edge = Edge::NONE;
    if (reason_ == SensingReadinessReason::LOW_COVERAGE && ready_) {
      if (!dip_active_) {
        dip_active_ = true;
        dip_started_ms_ = now_ms;
      }
      next = now_ms - dip_started_ms_ < coverage_grace_ms;
    } else {
      if (reason_ == SensingReadinessReason::READY && dip_active_ && ready_) {
        last_dip_ms_ = now_ms - dip_started_ms_;
        edge = Edge::DIP_ABSORBED;
      }
      dip_active_ = false;
      next = reason_ == SensingReadinessReason::READY;
    }
    if (next != ready_) {
      ready_ = next;
      edge = next ? Edge::READY : Edge::UNREADY;
    }
    return edge;
  }

  bool ready() const { return ready_; }
  /** Condition seen by the last update; LOW_COVERAGE while a dip is held. */
  SensingReadinessReason reason() const { return reason_; }
  /** Duration of the last absorbed dip, reported with Edge::DIP_ABSORBED. */
  uint32_t last_dip_ms() const { return last_dip_ms_; }

 private:
  static SensingReadinessReason classify_(const SensingReadinessInputs &inputs) {
    if (!inputs.sensing_active) return SensingReadinessReason::SENSING_STOPPED;
    if (inputs.calibrating) return SensingReadinessReason::CALIBRATING;
    if (!inputs.has_detector) return SensingReadinessReason::NO_DETECTOR;
    if (!inputs.input_current) return SensingReadinessReason::INPUT_STALE;
    if (!inputs.window_full) return SensingReadinessReason::WINDOW_FILLING;
    if (!inputs.detector_ready) return SensingReadinessReason::LOW_COVERAGE;
    return SensingReadinessReason::READY;
  }

  bool ready_{false};
  bool dip_active_{false};
  uint32_t dip_started_ms_{0U};
  uint32_t last_dip_ms_{0U};
  SensingReadinessReason reason_{SensingReadinessReason::SENSING_STOPPED};
};

}  // namespace espectre
