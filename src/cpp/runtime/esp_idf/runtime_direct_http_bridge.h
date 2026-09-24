/*
 * ESPectre - Runtime Direct HTTP Bridge
 *
 * Shared Direct HTTP control surface for firmware frontends.
 *
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * SPDX-License-Identifier: GPL-3.0-only
 * Commercial licensing available under separate agreement; see LICENSING.md.
 */
#pragma once

#include <cstdint>
#include <functional>
#include <string>

#include "runtime/direct_http_service.h"
#include "direct_wifi_snapshot_esp_idf.h"
#include "runtime/frontend_command_engine.h"
#include "runtime/peer_discovery.h"
#include "raw_csi_session_controller.h"
#include "runtime/runtime_diagnostics.h"
#include "runtime/runtime_event_mailbox.h"
#include "runtime_frontend_controller.h"

#include <atomic>

namespace espectre {

/**
 * What RuntimeDirectHttpBridge reports about the device, and the optional
 * hooks that extend its surface. Every callback runs on the owner task.
 */
struct RuntimeDirectHttpBridgeConfig {
  /** Frontend name, for example `native`. */
  std::string frontend;
  /** Display name used when no label is set. */
  std::string device_name;
  /** mDNS host name advertised to peers. */
  std::string hostname;
  /** Application version. */
  std::string firmware_version;
  /** Chip target, normally `CONFIG_IDF_TARGET`. */
  std::string chip;
  /** Device identity; must be nonzero for raw collection. */
  uint64_t device_id{0U};
  uint16_t port{ESPECTRE_DIRECT_HTTP_PORT};
  /** Offer raw collection when the runtime supports it. */
  bool raw_csi{false};
  /** See `DirectHttpServiceConfig::allow_missing_origin`. */
  bool allow_missing_origin{false};
  /** Current user-facing label; omitted, `device_name` is used. */
  std::function<std::string()> device_label_getter;
  /** Apply `update_device`; omitted, the command is not advertised. */
  FrontendDeviceLabelCallback device_label_setter;
  /** Station snapshot for the `wifi` resource; omitted, read_direct_wifi_snapshot() is used. */
  std::function<DirectWifiSnapshot()> wifi_snapshot_getter;
  /** Peer discovery for the `devices` resource; not owned. Omitted, it is not advertised. */
  IPeerDiscoveryService *peer_discovery{nullptr};
  /** Rate diagnostics, normally `RuntimeFrontendController::diagnostics_sample()`. */
  std::function<const RuntimeDiagnosticsSample *()> diagnostics_sample_getter;
  /** Mailbox whose drop counter is reported in diagnostics; not owned. */
  const RuntimeEventMailbox *runtime_events{nullptr};
  /**
   * Apply or clear, for an empty BSSID, the station pin.
   *
   * A frontend that provides it owns the complete live-apply and persistence
   * transaction, for example through WifiBssidPinService. Omitted, the pin is
   * applied to the current station session only.
   */
  std::function<bool(const std::string &bssid, bool force, std::string *message)> wifi_bssid_pin_setter;
  /** Refuse a pin change before it starts; return false with a reason. */
  std::function<bool(std::string *message)> wifi_bssid_pin_preflight;
  /** Latest complete frontend loop duration in milliseconds; omitted callbacks yield null. */
  std::function<float()> loop_time_ms_getter{};
};

/** Apply or clear the ESP-IDF station BSSID pin through a Wi-Fi state-machine restart. */
bool apply_wifi_bssid_pin(const std::string &bssid,
                          std::string *message,
                          bool *station_transition_started = nullptr);

/**
 * Exposes the common runtime controls over the versioned Direct HTTP API.
 *
 * Frontends retain ownership of their runtime and transport. The optional
 * callback lets an adapter republish frontend-native entities after a Direct
 * mutation, for example ESPHome number and select entities.
 */
class RuntimeDirectHttpBridge {
 public:
  /** Called after a Direct request changed configuration. */
  using ConfigChangedCallback = std::function<void()>;

  /**
   * Start `service` for the first-party portals and route its requests to
   * `runtime`. Neither is owned; both must outlive the bridge.
   */
  bool setup(IDirectHttpService *service,
             RuntimeFrontendController *runtime,
             const RuntimeDirectHttpBridgeConfig &config,
             ConfigChangedCallback config_changed = {});
  /** Pump the service, raw collection, and peer discovery. Call it from the owner loop. */
  void loop();
  /** Stop raw collection, peer discovery, and the service. */
  void shutdown();
  /** Whether the service is running. */
  bool running() const;
  /** Connected event stream clients. */
  size_t event_client_count() const;
  /** Publish an event to event stream clients; see IDirectHttpService::publish_event(). */
  bool publish_event(const char *event_name, const std::string &data_json, bool replaceable_telemetry = false);
  /** Publish replaceable `motion` telemetry; false without clients. */
  bool publish_motion(const RuntimeSnapshot &snapshot);
  /** Republish the resources named in `changes` to event stream clients. */
  bool publish_changes(FrontendCommandChange changes);

 private:
  IDirectHttpService::DeferredRequestResult handle_deferred_request_(uint64_t request_token,
                                                                     const DirectRequest &request);
  std::string handle_request_(const DirectRequest &request);
  EspectreCapabilityProfile capability_profile_() const;
  std::string device_label_() const;
  DirectWifiSnapshot wifi_snapshot_() const;
  std::string capabilities_payload_() const;
  std::string device_payload_() const;
  std::string health_payload_() const;
  std::string sensing_payload_() const;
  std::string wifi_payload_() const;
  std::string diagnostics_payload_(const std::vector<std::string> &fields) const;
  std::string wifi_access_points_payload_() const;
  bool handle_wifi_control_(const EspectreCommand &command, std::string *message);
  void refresh_peer_candidate_();
  void notify_config_changed_();

  IDirectHttpService *service_{nullptr};
  RuntimeFrontendController *runtime_{nullptr};
  FrontendCommandEngine command_engine_{};
  RuntimeDirectHttpBridgeConfig config_{};
  ConfigChangedCallback config_changed_{};
  RawCsiSessionController raw_session_controller_{};
  std::atomic<size_t> event_client_count_{0U};
  bool deferred_requests_enabled_{false};
  bool wifi_response_pending_{false};
};

}  // namespace espectre
