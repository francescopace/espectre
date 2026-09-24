/*
 * ESPectre - Shared mDNS Discovery Service
 *
 * Owns the common ESP-IDF mDNS lifecycle used by firmware frontends.
 *
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * SPDX-License-Identifier: GPL-3.0-only
 * Commercial licensing available under separate agreement; see LICENSING.md.
 */
#pragma once

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace espectre {

/**
 * DNS-SD TXT records as key and value pairs.
 *
 * A `txtvers` entry is always transmitted first.
 */
using MdnsTxtRecords = std::vector<std::pair<std::string, std::string>>;

/** Who owns the ESP-IDF mDNS responder. */
enum class MdnsResponderMode : uint8_t {
  /**
   * This service initializes the responder, sets the hostname and instance
   * name, and frees the responder at shutdown when it started it.
   */
  OWN_RESPONDER = 0,
  /**
   * Another component, such as the Matter stack, owns the responder; this
   * service only adds its DNS-SD service and announces it.
   */
  USE_EXISTING_RESPONDER,
};

/**
 * DNS-SD advertisement for MdnsDiscoveryService.
 *
 * See [DISCOVERY.md](https://github.com/francescopace/espectre/blob/main/docs/DISCOVERY.md)
 * for the service type and TXT keys that ESPectre clients expect.
 */
struct MdnsDiscoveryServiceConfig {
  /** Host label, up to 63 bytes. Required with `OWN_RESPONDER`, optional otherwise. */
  std::string hostname;
  /** Human-readable service instance name. Required. */
  std::string instance_name;
  /** Service type such as `_espectre`. Required. */
  std::string service_type;
  /** Transport label such as `_tcp`. Required. */
  std::string service_protocol;
  /** Advertised port. Required. */
  uint16_t port{0U};
  MdnsTxtRecords txt_records;
  MdnsResponderMode responder_mode{MdnsResponderMode::OWN_RESPONDER};
};

/**
 * Advertises one DNS-SD service on the Wi-Fi station interface.
 *
 * Call on_wifi_connected() and on_wifi_disconnected() from the firmware's
 * connection callbacks so the service is announced after every reconnect.
 *
 * @par Threading
 * Call every method from one owner task.
 */
class MdnsDiscoveryService {
 public:
  /**
   * Replace any previous advertisement and add the configured service.
   *
   * @return false for an incomplete configuration or when the responder
   *         rejects it. Resources acquired before the failure are released.
   */
  bool setup(const MdnsDiscoveryServiceConfig &config);
  /** Replace the TXT records; false before setup() or when the responder rejects them. */
  bool update_txt(const MdnsTxtRecords &txt_records);
  /** Enable or announce the service on the station interface. */
  void on_wifi_connected();
  /** Withdraw the station interface from an owned responder. */
  void on_wifi_disconnected();
  /** Remove the service and free an owned responder. Safe to repeat. */
  void shutdown();

  /** True between a successful setup() and shutdown(). */
  bool initialized() const { return mdns_initialized_; }
  /** True while the service is announced on the station interface. */
  bool service_enabled() const { return service_enabled_; }

 private:
  bool set_service_txt_();
  void apply_netif_action_(int action);

  MdnsDiscoveryServiceConfig config_{};
  bool mdns_initialized_{false};
  bool owns_mdns_{false};
  bool service_added_{false};
  bool service_enabled_{false};
};

}  // namespace espectre
