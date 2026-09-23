/*
 * ESPectre - Runtime Config
 *
 * Platform-agnostic runtime configuration contract.
 *
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * SPDX-License-Identifier: GPL-3.0-only
 * Commercial licensing available under separate agreement; see LICENSING.md.
 */
#pragma once

#include <cstdint>
#include <string>

#include "csi_capture_profile.h"
#include "runtime_sensing_schema.h"

/**
 * @file runtime_config.h
 * @brief Runtime configuration handed to `RuntimeFrontendController`.
 */

namespace espectre {

enum class WifiBandPolicy : uint8_t {
  /** Restrict association to 2.4 GHz. */
  BAND_2G = 0,
  /** Restrict association to 5 GHz. Supported only by dual-band targets. */
  BAND_5G = 1,
  /**
   * Use every band the radio has. A dual-band radio chooses between 2.4 GHz
   * and 5 GHz; a 2.4 GHz-only radio behaves as `BAND_2G`. The default.
   */
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
 * copy have no effect; use the `RuntimeFrontendController` setters instead.
 */
struct RuntimeConfig {
  /**
   * Band available to the station while the runtime keeps the PHY at HT20.
   *
   * `BAND_5G` and `AUTO` require dual-band silicon. Keeping `BAND_2G` as the
   * default preserves the band covered by the production detector corpus.
   */
  WifiBandPolicy wifi_band_policy{WifiBandPolicy::AUTO};
  /** Build-time CSI profile; AUTO resolves from chip, band, and the active traffic source. No runtime setter. */
  CsiCapturePolicy csi_capture_policy{CsiCapturePolicy::AUTO};
  /** Detection profile to run. Lightweight self-calibrates; High Accuracy uses trained weights. */
  DetectionAlgorithm detection_algorithm{DetectionAlgorithm::LIGHTWEIGHT};
  /**
   * Motion probability threshold, on the same 0..1 scale as
   * `RuntimeSnapshot::movement_metric`.
   *
   * Lightweight Detection overwrites this during startup calibration, so the
   * configured value only governs the pre-calibration window. High-Accuracy Detection keeps it as given.
   */
  float threshold{RUNTIME_THRESHOLD_DEFAULT};
  /**
   * Detector window duration in milliseconds (1000..2000).
   *
   * Runtimes resolve the duration to a fixed temporal grid from
   * `csi_target_pps`; live arrival jitter never resizes the detector.
   */
  uint32_t window_size_ms{RUNTIME_WINDOW_SIZE_MS_DEFAULT};
  /**
   * Advertise runtime detector switching.
   *
   * When true the runtime restores the persisted detector choice at `setup()`
   * and sets `RuntimeCapabilities::supports_runtime_detector_selection`. A
   * persisted detector that differs from `detection_algorithm` also replaces
   * `threshold` with that detector's default.
   */
  bool runtime_detector_selection_enabled{false};
  /**
   * Target CSI sensing cadence, in packets per second.
   *
   * This value is always positive and defines detector temporal slots as well
   * as the target for managed traffic. `traffic_generator_mode` alone selects who
   * supplies traffic. The detector coefficients are fitted at 100 pps; see
   * `docs/ALGORITHMS.md` before moving far from it.
   */
  uint32_t csi_target_pps{RUNTIME_CSI_TARGET_PPS_DEFAULT};
  /**
   * How the device gets CSI-bearing traffic: an internal generator packet, or
   * `EXTERNAL` to listen for another host.
   */
  TrafficGeneratorMode traffic_generator_mode{TrafficGeneratorMode::PING};
  /** Unicast IPv4 destination for internal IP traffic; empty uses the Wi-Fi gateway. Ignored by `wifi_raw`. */
  std::string traffic_generator_target_ip;
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
   * Assign `derive_runtime_device_id()` to use the SDK's stable pseudonym from
   * the Wi-Fi MAC. Zero is an unresolved sentinel; the controller does not
   * replace it automatically.
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
  /**
   * The Wi-Fi stack consumes scan results, including CSI recovery scans.
   *
   * Enable for stacks with autonomous scans, such as ESPHome. The runtime
   * must not clear their driver result list, even after its own scan completes.
   * Otherwise, independent scanners must wait until the SDK releases its
   * scanner reservation after cleanup.
   */
  bool wifi_scan_results_managed_externally{false};
  /**
   * Remember runtime control changes across reboots.
   *
   * When true, `setup()` restores the CSI traffic source, generator packet,
   * and motion hits saved by earlier control calls, plus the detector when
   * `runtime_detector_selection_enabled` is set, and those calls save their
   * new values. Set it to false when your firmware owns configuration, for
   * example from YAML or a cloud service: this config is then the only source
   * of truth, and the runtime neither reads nor writes saved controls.
   */
  bool persist_runtime_overrides{true};
};

}  // namespace espectre
