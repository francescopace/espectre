/*
 * ESPectre - Optional ESP-IDF Services SDK
 *
 * SPDX-License-Identifier: GPL-3.0-only
 * Commercial licensing available under separate agreement; see LICENSING.md.
 */
#pragma once

/**
 * @file espectre_services_sdk.h
 * @brief Optional services for firmware integrating the ESPectre runtime.
 *
 * Includes the sensing SDK and the supported command, transport, discovery,
 * provisioning, and bootstrap services. Compile only the capability source
 * groups that your firmware uses; including this facade does not enable them.
 * Requires the ESP-IDF platform headers and declared component dependencies.
 * Firmware retains ownership of networking, service lifetime, and task policy.
 * Concrete service objects retain their existing caller-owned allocation model.
 */
#include "espectre_sdk.h"

#include "runtime/esp_idf/device_config_store.h"
#include "runtime/esp_idf/direct_http_service_esp_idf.h"
#include "runtime/esp_idf/direct_wifi_snapshot_esp_idf.h"
#include "runtime/esp_idf/frontend_bootstrap_helpers.h"
#include "runtime/esp_idf/frontend_ha_mqtt_helpers.h"
#include "runtime/esp_idf/frontend_mqtt_helpers.h"
#include "runtime/esp_idf/mdns_bootstrap_responder.h"
#include "runtime/esp_idf/mdns_discovery_service.h"
#include "runtime/esp_idf/nvs_helpers.h"
#include "runtime/esp_idf/peer_discovery_service_esp_idf.h"
#include "runtime/esp_idf/raw_csi_session_controller.h"
#include "runtime/esp_idf/runtime_direct_http_bridge.h"
#include "runtime/esp_idf/standalone_wifi_service.h"
#include "runtime/esp_idf/task_scheduling_config.h"
#include "runtime/esp_idf/traffic_generator_manager.h"
#include "runtime/esp_idf/wifi_band_helpers.h"
#include "runtime/esp_idf/wifi_bssid_pin_service.h"
#include "runtime/esp_idf/wifi_provisioning_service.h"
#include "runtime/esp_idf/wifi_tx_rate.h"
#include "runtime/espectre_banner.h"
#include "runtime/frontend_command_engine.h"
#include "runtime/peer_discovery.h"
#include "runtime/pending_event.h"
#include "runtime/pending_queue.h"
#include "runtime/runtime_event_mailbox.h"
#include "runtime/runtime_time.h"
