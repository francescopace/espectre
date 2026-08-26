/*
 * ESPectre - Runtime Interface
 *
 * Platform-agnostic runtime interface and configuration contract.
 *
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * SPDX-License-Identifier: GPL-3.0-only
 * Commercial licensing available under separate agreement; see LICENSING.md.
 */
#pragma once

#include <cstdint>
#include <string>

#include "runtime_capabilities.h"
#include "runtime_events.h"
#include "runtime_snapshot.h"
#include "runtime_sensing_schema.h"
#include "csi_traffic_types.h"
#include "raw_csi.h"

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

/** Wi-Fi band selection requested by the embedding frontend. */
enum class WifiBandPolicy : uint8_t {
  /** Restrict association to 2.4 GHz. This is the validated production default. */
  BAND_2G = 0,
  /** Restrict association to 5 GHz. Supported only by dual-band targets. */
  BAND_5G = 1,
  /** Let a dual-band radio choose between 2.4 GHz and 5 GHz. */
  AUTO = 2,
};

/**
 * Everything the runtime needs to know before `setup()`.
 *
 * Every member is default-constructed to a supported production value, so
 * `RuntimeConfig{}` is a working configuration for Lightweight Detection on
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
  /**
   * Band available to the station while the runtime keeps the PHY at HT20.
   *
   * `BAND_5G` and `AUTO` require dual-band silicon. Keeping `BAND_2G` as the
   * default preserves the band covered by the production detector corpus.
   */
  WifiBandPolicy wifi_band_policy{WifiBandPolicy::BAND_2G};
  /** Detection profile to run. Lightweight self-calibrates; High Accuracy uses trained weights. */
  DetectionAlgorithm detection_algorithm{DetectionAlgorithm::LIGHTWEIGHT};
  /**
   * Motion probability threshold, on the same 0..1 scale as
   * `RuntimeSnapshot::movement_metric`.
   *
   * Lightweight Detection overwrites this during startup calibration, so the
   * configured value only governs the pre-calibration window. High-Accuracy Detection keeps it as given.
   */
  float segmentation_threshold{RUNTIME_SEGMENTATION_THRESHOLD_DEFAULT};
  /**
   * Detector window duration in milliseconds (1000..2000).
   *
   * Runtimes resolve the duration to a fixed temporal grid from
   * `csi_target_pps`; live arrival jitter never resizes the detector.
   */
  uint32_t segmentation_window_size_ms{RUNTIME_SEGMENTATION_WINDOW_SIZE_MS_DEFAULT};
  /**
   * Advertise runtime detector switching.
   *
   * When true the runtime restores the persisted detector choice at `setup()`
   * and sets `RuntimeCapabilities::supports_runtime_detector_selection`.
   */
  bool runtime_detector_selection_enabled{false};
  /**
   * Target CSI sensing cadence, in packets per second.
   *
   * This value is always positive and defines detector temporal slots as well
   * as the target for managed traffic. `csi_traffic_mode` alone selects who
   * supplies traffic. The detector coefficients are fitted at 100 pps; see
   * `docs/ALGORITHMS.md` before moving far from it.
   */
  uint32_t csi_target_pps{RUNTIME_CSI_TARGET_PPS_DEFAULT};
  /** Which packet the internal generator sends to solicit CSI. */
  RuntimeTrafficMode traffic_generator_mode{RuntimeTrafficMode::PING};
  /** Where the CSI-bearing traffic comes from. See `csi_traffic_types.h`. */
  CsiTrafficMode csi_traffic_mode{CsiTrafficMode::INTERNAL};
  /** UDP port used by the external CSI traffic mode. */
  uint16_t csi_traffic_udp_port{RUNTIME_CSI_TRAFFIC_UDP_PORT_DEFAULT};
  /**
   * IPv4 multicast group joined by the UDP listener in `external`.
   *
   * Empty disables the IGMP join. Unicast to the device IP still works.
   */
  std::string csi_traffic_multicast_group{RUNTIME_CSI_TRAFFIC_MULTICAST_GROUP_DEFAULT};
  /**
   * Stable device identity used by the ESPectre Protocol and CSI streaming.
   *
   * Leave at zero to derive it from the Wi-Fi MAC via
   * `derive_runtime_device_id()`.
   */
  uint64_t device_id{0U};
  /** Detector evaluation cadence in milliseconds. */
  uint32_t evaluation_interval_ms{RUNTIME_EVALUATION_INTERVAL_MS_DEFAULT};
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
   * traffic. The current Wi-Fi association is preserved so arming again can
   * restart capture without waiting for another IP event. Matter uses this to
   * stay quiet until commissioning completes; Native uses it to pause sensing
   * while a frontend intentionally suspends sensing. During raw collection,
   * the requested state is staged without interrupting the capture callback
   * and takes effect when collection stops.
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
   * Switch who owns the CSI-bearing traffic while running.
   *
   * Defaulted rather than pure so existing out-of-tree backends keep
   * compiling. A backend that does not implement live traffic retuning should
   * return false and let the frontend reject the command.
   */
  virtual bool set_csi_traffic_mode_runtime(CsiTrafficMode mode) { return false; }
  /**
   * Change the internal traffic generator packet type while running.
   *
   * Backends that do not own traffic retuning keep the default false.
   */
  virtual bool set_traffic_generator_mode_runtime(RuntimeTrafficMode mode) { return false; }
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

  /**
   * Enter transient raw collection while preserving persisted sensing config.
   *
   * Defaulted so existing external runtime implementations remain source
   * compatible. The callback runs in the CSI capture context and must remain
   * bounded and allocation-free.
   */
  virtual bool start_raw_collection(raw_csi_packet_callback_t callback, void *context) {
    (void) callback;
    (void) context;
    return false;
  }
  /** Leave raw collection and restore the previous sensing lifecycle. */
  virtual bool stop_raw_collection(RawCsiStopReason reason) {
    (void) reason;
    return false;
  }
  /** Current transient operation state. */
  virtual RuntimeOperationState operation_state() const {
    return RuntimeOperationState::SENSING;
  }

  /** Current sensing state. Cheap enough to poll from your loop. */
  virtual RuntimeSnapshot get_snapshot() const = 0;
  /**
   * Capture, traffic, and link counters for diagnostic frontends.
   *
   * The counters are cumulative and monotonic within a session. Feed them to
   * `RuntimeDiagnosticsSampler` from an existing periodic sensing callback to
   * get rates without adding a diagnostic timer.
   *
   * Defaulted rather than pure so that adding it does not break out-of-tree
   * backends. A runtime that collects nothing keeps the zeroed snapshot, which
   * is what a frontend reads as "no counters from this backend".
   */
  virtual RuntimeDiagnosticsSnapshot get_diagnostics() const { return {}; }
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
