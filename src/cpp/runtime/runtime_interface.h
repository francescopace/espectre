/*
 * ESPectre - Runtime Interface
 *
 * Platform-agnostic runtime interface and configuration contract.
 *
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * License: GPLv3
 */
#pragma once

#include <cstdint>
#include <string>

#include "runtime_capabilities.h"
#include "runtime_events.h"
#include "runtime_snapshot.h"
#include "runtime_sensing_schema.h"
#include "csi_traffic_types.h"
#include "threshold.h"
#include "utils.h"

/**
 * @file runtime_interface.h
 * @brief Runtime configuration and the backend contract behind it.
 *
 * Most integrations do not implement `IEspectreRuntime`; they configure a
 * `RuntimeConfig`, hand it to `RuntimeFrontendController`, and let the
 * controller pick the backend. Implement the interface only when you are
 * replacing the ESP-IDF backend, for example in a simulator or a host harness.
 */

namespace espectre {

/**
 * Everything the runtime needs to know before `setup()`.
 *
 * Every member is default-constructed to a supported production value, so
 * `RuntimeConfig{}` is a working configuration for the Classic detector on
 * internally generated traffic. Override only what your product changes.
 *
 * Ranges are declared in `runtime_sensing_schema.h` as
 * `RUNTIME_<FIELD>_MIN` / `_MAX` / `_DEFAULT`, and the free functions in
 * `runtime_config_utils.h` validate against them. On ESP-IDF you can build
 * this from menuconfig with `make_runtime_sensing_config_from_kconfig()`
 * instead of assigning fields by hand.
 *
 * The config is copied into the runtime at `setup()`. Later edits to your own
 * copy have no effect; use the `set_*_runtime()` control methods instead.
 */
struct RuntimeConfig {
  /** Which backend to build: motion sensing, or raw CSI streaming to a collector. */
  RuntimeProfile runtime_profile{RuntimeProfile::SENSING};
  /** Detector to run. Classic self-calibrates; ML uses the trained weights. */
  DetectionAlgorithm detection_algorithm{DetectionAlgorithm::CLASSIC};
  /**
   * Motion probability threshold, on the same 0..1 scale as
   * `RuntimeSnapshot::movement_metric`.
   *
   * Classic overwrites this during startup calibration, so the configured
   * value only governs the pre-calibration window. ML keeps it as given.
   */
  float segmentation_threshold{RUNTIME_SEGMENTATION_THRESHOLD_DEFAULT};
  /**
   * Detector window in packets (100..200).
   *
   * Below 100 the window features become too noisy to separate motion from the
   * quiet floor; above 200 the window smears short movements into the
   * background. See `detector_limits.h` for the measurements behind both bounds.
   */
  uint16_t segmentation_window_size{RUNTIME_SEGMENTATION_WINDOW_SIZE_DEFAULT};
  /**
   * Advertise runtime detector switching.
   *
   * When true the runtime restores the persisted detector choice at `setup()`
   * and sets `RuntimeCapabilities::supports_runtime_detector_selection`.
   */
  bool runtime_detector_selection_enabled{false};
  /**
   * Target rate of valid CSI callbacks, in packets per second.
   *
   * Zero disables the internal traffic generator and expects the environment
   * to supply the traffic. The detector coefficients are fitted at 100 pps;
   * see `docs/ALGORITHMS.md` before moving far from it.
   */
  uint32_t traffic_generator_rate{RUNTIME_TRAFFIC_GENERATOR_RATE_DEFAULT};
  /** Let the runtime retune its send rate to hold `traffic_generator_rate`. */
  bool traffic_generator_adaptive{RUNTIME_TRAFFIC_GENERATOR_ADAPTIVE_DEFAULT};
  /** Which packet the internal generator sends to solicit CSI. */
  RuntimeTrafficMode traffic_generator_mode{RuntimeTrafficMode::PING};
  /** Where the CSI-bearing traffic comes from. See `csi_traffic_types.h`. */
  CsiTrafficMode csi_traffic_mode{CsiTrafficMode::INTERNAL};
  /** UDP port used by the external and pacing CSI traffic modes. */
  uint16_t csi_traffic_udp_port{RUNTIME_CSI_TRAFFIC_UDP_PORT_DEFAULT};
  /** Multicast group to join for externally sourced CSI traffic, when used. */
  std::string csi_traffic_multicast_group;
  /** Payload marker that identifies accepted external CSI traffic, when used. */
  std::string csi_traffic_expected_payload;
  /**
   * Stable device identity used by the ESPectre Protocol and CSI streaming.
   *
   * Leave at zero to derive it from the Wi-Fi MAC via
   * `derive_runtime_device_id()`.
   */
  uint64_t device_id{0U};
  /** `RuntimeProfile::STREAM` only: UDP port of the host CSI collector. */
  uint16_t collector_port{RUNTIME_STREAM_COLLECTOR_PORT_DEFAULT};
  /** `RuntimeProfile::STREAM` only: interval between stream status logs, in ms. */
  uint32_t stream_log_interval_ms{RUNTIME_STREAM_LOG_INTERVAL_MS_DEFAULT};
  /** `RuntimeProfile::STREAM` only: CSI records coalesced into one datagram. */
  uint8_t stream_tx_batch_records{RUNTIME_STREAM_TX_BATCH_RECORDS_DEFAULT};
  /**
   * Packets between `IRuntimeListener::on_periodic_update()` callbacks.
   *
   * At the default 100 pps target, the default 100 is roughly one heartbeat
   * per second.
   */
  uint32_t publish_interval{RUNTIME_PUBLISH_INTERVAL_DEFAULT};
  /**
   * Packet-count fallback cadence for detector evaluation.
   *
   * The runtime normally evaluates on elapsed arrival time and only counts
   * packets during estimator warmup or on sources with no timestamp, so this
   * mainly governs the first second of a session.
   */
  uint32_t evaluation_interval{RUNTIME_EVALUATION_INTERVAL_DEFAULT};
  /** Consecutive above-threshold evaluations before reporting motion (1..20). */
  uint8_t motion_on_hits{RUNTIME_MOTION_ON_HITS_DEFAULT};
  /** Consecutive below-threshold evaluations before clearing motion (1..20). */
  uint8_t motion_off_hits{RUNTIME_MOTION_OFF_HITS_DEFAULT};
  /** Enable the low-pass filter on the turbulence stream. Off by default. */
  bool lowpass_enabled{RUNTIME_LOWPASS_ENABLED_DEFAULT};
  /** Low-pass cutoff in Hz (5.0..20.0). Ignored unless `lowpass_enabled`. */
  float lowpass_cutoff{RUNTIME_LOWPASS_CUTOFF_DEFAULT};
  /** Enable Hampel outlier rejection on the turbulence stream. On by default. */
  bool hampel_enabled{RUNTIME_HAMPEL_ENABLED_DEFAULT};
  /** Hampel window in samples (3..11). Ignored unless `hampel_enabled`. */
  uint8_t hampel_window{RUNTIME_HAMPEL_WINDOW_DEFAULT};
  /** Hampel MAD multiplier (1.0..10.0). Ignored unless `hampel_enabled`. */
  float hampel_threshold{RUNTIME_HAMPEL_THRESHOLD_DEFAULT};
};

/**
 * The sensing backend behind `RuntimeFrontendController`.
 *
 * Implement this only to replace the shipped ESP-IDF backend. Integrations
 * consume it indirectly: the controller owns the instance, forwards control
 * calls, and gates them on `get_capabilities()`.
 *
 * @par Threading
 * Implementations are not required to be thread-safe and the shipped one is
 * not. Run `setup()`, `loop()`, and `shutdown()` on the task that owns the
 * runtime, and deliver listener callbacks on the caller's task rather than
 * from an interrupt or a driver callback. See `espectre_sdk.h` for the
 * complete contract, including the control-call caveat.
 */
class IEspectreRuntime {
 public:
  virtual ~IEspectreRuntime() = default;

  /**
   * Bring the runtime up: radio hooks, CSI capture, detector, traffic.
   *
   * @return false if the runtime cannot sense. The caller must not call
   *         `loop()` afterwards; the controller drops the instance instead.
   */
  virtual bool setup() = 0;
  /** Stop sensing and release everything `setup()` acquired. Safe to repeat. */
  virtual void shutdown() = 0;
  /**
   * Advance runtime work and drain deferred events.
   *
   * Call it continuously from your loop task. This is where listener
   * callbacks are delivered, so a slow callback delays the next iteration.
   */
  virtual void loop() = 0;
  /**
   * Gate the runtime-owned services without tearing the runtime down.
   *
   * Disarmed, the runtime stays configured but starts no CSI capture or
   * traffic. Matter uses it to stay quiet until commissioning completes.
   */
  virtual void set_services_armed(bool armed) = 0;
  /** Enable or suppress the high-rate `on_live_telemetry()` stream. */
  virtual void set_live_telemetry_enabled(bool enabled) = 0;

  /**
   * Retune the motion threshold while running.
   *
   * @return false when the value is out of range for the active detector, or
   *         when the runtime cannot apply it.
   */
  virtual bool set_threshold_runtime(float threshold) = 0;
  /**
   * Retune the hit filter while running.
   *
   * @return false when either count is outside 1..20, or when the runtime
   *         cannot apply the change.
   */
  virtual bool set_motion_hits_runtime(uint8_t motion_on_hits, uint8_t motion_off_hits) = 0;
  /**
   * Switch detector while running, rebuilding detector state.
   *
   * @return false when the algorithm is unknown or the switch fails.
   */
  virtual bool set_detection_algorithm_runtime(DetectionAlgorithm algorithm) = 0;
  /**
   * Restart startup calibration against the current ambient channel.
   *
   * @return false when calibration cannot start, for example with no Wi-Fi
   *         link yet. Progress arrives through the calibration callbacks.
   */
  virtual bool trigger_recalibration() = 0;
  /** True while startup calibration is running and detection is not yet valid. */
  virtual bool is_calibrating() const = 0;

  /** Current sensing state. Cheap enough to poll from your loop. */
  virtual RuntimeSnapshot get_snapshot() const = 0;
  /** Low-rate capture and traffic counters for diagnostic frontends. */
  virtual RuntimeDiagnosticsSnapshot get_diagnostics() const = 0;
  /** What this backend actually supports. Stable after `setup()`. */
  virtual RuntimeCapabilities get_capabilities() const = 0;

  /**
   * Install the event sink, or `nullptr` to detach.
   *
   * Set it before `setup()` so calibration events are not missed. The runtime
   * does not take ownership; the listener must outlive the runtime.
   */
  virtual void set_listener(IRuntimeListener *listener) = 0;
};

}  // namespace espectre
