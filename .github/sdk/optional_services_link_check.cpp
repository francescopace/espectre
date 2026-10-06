// SPDX-License-Identifier: GPL-3.0-only
// Commercial licensing available under separate agreement; see LICENSING.md.
//
// CI-only link check for the SDK's optional service groups. verify_sdk_component.py
// adds this file to the prepared example and forces the linker to keep the
// function below with -u, so every referenced service must link. Nothing calls it.
#include "sdkconfig.h"

#include "espectre_services_sdk.h"

#if CONFIG_ESPECTRE_SDK_ENABLE_MQTT
#include "espectre_mqtt_sdk.h"
#endif

extern "C" void espectre_check_optional_services() {
#if CONFIG_ESPECTRE_SDK_ENABLE_FRONTEND_SUPPORT
  espectre::EspectreDeviceConfig config;
  (void) espectre::publish_frontend_mqtt_status(nullptr, config, false, 0);
  espectre::FrontendWifiStationOptions options;
  (void) espectre::setup_frontend_wifi_station(nullptr, nullptr, options, "espectre.link_check", nullptr);
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
