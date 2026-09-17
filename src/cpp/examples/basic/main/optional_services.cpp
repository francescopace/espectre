// SPDX-License-Identifier: GPL-3.0-only
// Commercial licensing available under separate agreement; see LICENSING.md.
#include "espectre_services_sdk.h"

#if CONFIG_ESPECTRE_SDK_ENABLE_MQTT
#include "espectre_mqtt_sdk.h"
#endif

// Exercise real symbols so CI checks linking as well as optional source compilation.
// No transport is started, and no persisted configuration is changed.
void check_optional_services() {
#if CONFIG_ESPECTRE_SDK_ENABLE_FRONTEND_SUPPORT
  espectre::EspectreDeviceConfig config;
  (void) espectre::publish_frontend_mqtt_status(nullptr, config, false, 0);
  espectre::FrontendWifiStationOptions options;
  (void) espectre::setup_frontend_wifi_station(nullptr, nullptr, options, "espectre.example", nullptr);
#endif
#if CONFIG_ESPECTRE_SDK_ENABLE_MQTT
  espectre::EspIdfMqttTransport transport;
#endif
#if CONFIG_ESPECTRE_SDK_ENABLE_PROVISIONING
  espectre::StoredWifiConfig stored;
  (void) espectre::load_stored_wifi_config(&stored);
  espectre::WifiProvisioningService provisioning(nullptr);
  (void) provisioning.setup_station({});
#endif
#if CONFIG_ESPECTRE_SDK_ENABLE_DIRECT
  espectre::EspIdfDirectHttpService direct;
  espectre::MdnsDiscoveryService discovery;
  discovery.shutdown();
  espectre::MdnsBootstrapResponder bootstrap;
  espectre::EspIdfPeerDiscoveryService peers;
  espectre::RuntimeDirectHttpBridge bridge;
  (void) bridge.setup(nullptr, nullptr, {});
  espectre::WifiBssidPinService pin;
  (void) pin.setup({});
#endif
}
