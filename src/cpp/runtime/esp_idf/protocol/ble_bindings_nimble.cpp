/*
 * ESPectre - NimBLE Bindings
 *
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * License: GPLv3
 */

#include "ble_bindings_nimble.h"

#include <cstring>

#include "ble_protocol.h"
#include "esp_log.h"
#include "host/ble_hs.h"
#include "host/ble_hs_mbuf.h"
#include "nimble/ble.h"
#include "nimble/nimble_port.h"
#include "nimble/nimble_port_freertos.h"
#include "os/os_mbuf.h"
#include "services/gap/ble_svc_gap.h"
#include "services/gatt/ble_svc_gatt.h"

namespace esphome {
namespace espectre {

namespace {

static const char *const TAG = "espectre.ble.bind";
static const char *const kAdvertisingName = "ESPectre";

static ble_uuid128_t g_service_uuid =
    BLE_UUID128_INIT(0xf0, 0xf8, 0x6a, 0xc3, 0xa2, 0xb3, 0x6f, 0xbc, 0x75, 0x47, 0x03, 0x22, 0x6b, 0xf4, 0x3f, 0xd3);
static ble_uuid128_t g_telemetry_uuid =
    BLE_UUID128_INIT(0x58, 0x82, 0x86, 0x05, 0x98, 0x16, 0xc3, 0xbf, 0xd9, 0x4b, 0xda, 0x48, 0xac, 0x5c, 0x9d, 0x11);
static ble_uuid128_t g_sysinfo_uuid =
    BLE_UUID128_INIT(0xe3, 0xdf, 0x4a, 0xa0, 0x2f, 0x94, 0xfc, 0x9f, 0x1f, 0x46, 0x01, 0xc4, 0xfa, 0x9f, 0xc8, 0xc8);
static ble_uuid128_t g_control_uuid =
    BLE_UUID128_INIT(0x71, 0xdc, 0xdc, 0x47, 0x27, 0xc8, 0xd1, 0x82, 0xe8, 0x40, 0xd7, 0xa8, 0x14, 0x92, 0xed, 0x33);

static uint16_t g_telemetry_val_handle = 0;
static uint16_t g_sysinfo_val_handle = 0;
static uint16_t g_control_val_handle = 0;
static ble_gatt_chr_def g_characteristics[4];
static ble_gatt_svc_def g_services[2];

}  // namespace

NimbleBleBindings *NimbleBleBindings::instance_ = nullptr;

bool NimbleBleBindings::setup() {
  if (setup_complete_) {
    return true;
  }

  // Hide verbose NimBLE procedure logs and keep only warnings/errors.
  esp_log_level_set("NimBLE", ESP_LOG_WARN);

  instance_ = this;
  if (device_name_.empty()) {
    device_name_ = ESPECTRE_BLE_DEVICE_NAME;
  }
  telemetry_value_.clear();
  sysinfo_value_.clear();
  conn_handle_ = BLE_HS_CONN_HANDLE_NONE;
  telemetry_subscribed_ = false;

  const int init_rc = nimble_port_init();
  if (init_rc != 0) {
    ESP_LOGE(TAG, "nimble_port_init failed: %d", init_rc);
    return false;
  }

  std::memset(g_characteristics, 0, sizeof(g_characteristics));
  g_characteristics[0].uuid = &g_telemetry_uuid.u;
  g_characteristics[0].access_cb = &NimbleBleBindings::gatt_access_static_;
  g_characteristics[0].flags = BLE_GATT_CHR_F_NOTIFY;
  g_characteristics[0].val_handle = &g_telemetry_val_handle;

  g_characteristics[1].uuid = &g_sysinfo_uuid.u;
  g_characteristics[1].access_cb = &NimbleBleBindings::gatt_access_static_;
  g_characteristics[1].flags = BLE_GATT_CHR_F_READ | BLE_GATT_CHR_F_NOTIFY;
  g_characteristics[1].val_handle = &g_sysinfo_val_handle;

  g_characteristics[2].uuid = &g_control_uuid.u;
  g_characteristics[2].access_cb = &NimbleBleBindings::gatt_access_static_;
  g_characteristics[2].flags = BLE_GATT_CHR_F_WRITE | BLE_GATT_CHR_F_WRITE_NO_RSP;
  g_characteristics[2].val_handle = &g_control_val_handle;

  std::memset(g_services, 0, sizeof(g_services));
  g_services[0].type = BLE_GATT_SVC_TYPE_PRIMARY;
  g_services[0].uuid = &g_service_uuid.u;
  g_services[0].characteristics = g_characteristics;

  ble_svc_gap_init();
  ble_svc_gatt_init();
  ble_svc_gap_device_name_set(device_name_.c_str());

  ble_hs_cfg.sync_cb = &NimbleBleBindings::on_sync_static_;
  ble_hs_cfg.reset_cb = &NimbleBleBindings::on_reset_static_;

  int rc = ble_gatts_count_cfg(g_services);
  if (rc != 0) {
    ESP_LOGE(TAG, "ble_gatts_count_cfg failed: %d", rc);
    nimble_port_deinit();
    return false;
  }
  rc = ble_gatts_add_svcs(g_services);
  if (rc != 0) {
    ESP_LOGE(TAG, "ble_gatts_add_svcs failed: %d", rc);
    nimble_port_deinit();
    return false;
  }

  nimble_port_freertos_init(&NimbleBleBindings::host_task_);
  setup_complete_ = true;
  ESP_LOGI(TAG, "NimBLE bindings ready");
  return true;
}

void NimbleBleBindings::shutdown() {
  if (!setup_complete_) {
    return;
  }

  nimble_port_stop();
  nimble_port_deinit();
  setup_complete_ = false;
  conn_handle_ = BLE_HS_CONN_HANDLE_NONE;
  instance_ = nullptr;
}

void NimbleBleBindings::set_connection_state_callback(ConnectionStateCallback callback) {
  connection_state_callback_ = std::move(callback);
}

void NimbleBleBindings::set_control_write_callback(ControlWriteCallback callback) {
  control_write_callback_ = std::move(callback);
}

void NimbleBleBindings::set_telemetry_subscription_callback(TelemetrySubscriptionCallback callback) {
  telemetry_subscription_callback_ = std::move(callback);
}

void NimbleBleBindings::set_device_name(const char *name) {
  device_name_ = (name != nullptr && name[0] != '\0') ? name : ESPECTRE_BLE_DEVICE_NAME;
  if (setup_complete_) {
    const int rc = ble_svc_gap_device_name_set(device_name_.c_str());
    if (rc != 0) {
      ESP_LOGW(TAG, "ble_svc_gap_device_name_set failed: %d", rc);
    }
  }
}

void NimbleBleBindings::publish_telemetry(const uint8_t *payload, size_t payload_len) {
  if (!setup_complete_ || conn_handle_ == BLE_HS_CONN_HANDLE_NONE || g_telemetry_val_handle == 0 || payload == nullptr) {
    return;
  }

  telemetry_value_.assign(payload, payload + payload_len);
  os_mbuf *om = ble_hs_mbuf_from_flat(telemetry_value_.data(), telemetry_value_.size());
  if (om != nullptr) {
    ble_gatts_notify_custom(conn_handle_, g_telemetry_val_handle, om);
  }
}

void NimbleBleBindings::publish_sysinfo_line(const char *line) {
  if (!setup_complete_ || g_sysinfo_val_handle == 0 || line == nullptr) {
    return;
  }

  sysinfo_value_ = line;
  if (conn_handle_ != BLE_HS_CONN_HANDLE_NONE) {
    os_mbuf *om = ble_hs_mbuf_from_flat(sysinfo_value_.data(), sysinfo_value_.size());
    if (om != nullptr) {
      ble_gatts_notify_custom(conn_handle_, g_sysinfo_val_handle, om);
    }
  }
}

void NimbleBleBindings::report_fault(const char *message) {
  if (message != nullptr) {
    ESP_LOGW(TAG, "Runtime fault reported through BLE bindings: %s", message);
  }
}

bool NimbleBleBindings::start_advertising_() {
  ble_hs_adv_fields fields{};
  fields.flags = BLE_HS_ADV_F_DISC_GEN | BLE_HS_ADV_F_BREDR_UNSUP;
  fields.name = reinterpret_cast<const uint8_t *>(kAdvertisingName);
  fields.name_len = std::strlen(kAdvertisingName);
  fields.name_is_complete = 0;
  fields.uuids128 = &g_service_uuid;
  fields.num_uuids128 = 1;
  fields.uuids128_is_complete = 1;

  int rc = ble_gap_adv_set_fields(&fields);
  if (rc != 0) {
    ESP_LOGE(TAG, "ble_gap_adv_set_fields failed: %d", rc);
    return false;
  }

  ble_gap_adv_params adv_params{};
  adv_params.conn_mode = BLE_GAP_CONN_MODE_UND;
  adv_params.disc_mode = BLE_GAP_DISC_MODE_GEN;
  rc = ble_gap_adv_start(addr_type_, nullptr, BLE_HS_FOREVER, &adv_params, &NimbleBleBindings::gap_event_static_, this);
  if (rc != 0) {
    ESP_LOGE(TAG, "ble_gap_adv_start failed: %d", rc);
    return false;
  }
  return true;
}

void NimbleBleBindings::on_sync_() {
  ble_hs_id_infer_auto(0, &addr_type_);
  start_advertising_();
}

void NimbleBleBindings::on_reset_(int reason) { ESP_LOGW(TAG, "NimBLE reset: reason=%d", reason); }

int NimbleBleBindings::on_gap_event_(ble_gap_event *event) {
  if (event == nullptr) {
    return 0;
  }

  switch (event->type) {
    case BLE_GAP_EVENT_CONNECT:
      if (event->connect.status == 0) {
        conn_handle_ = event->connect.conn_handle;
        if (connection_state_callback_) {
          connection_state_callback_(true);
        }
      } else {
        start_advertising_();
      }
      return 0;
    case BLE_GAP_EVENT_DISCONNECT:
      conn_handle_ = BLE_HS_CONN_HANDLE_NONE;
      telemetry_subscribed_ = false;
      if (telemetry_subscription_callback_) {
        telemetry_subscription_callback_(false);
      }
      if (connection_state_callback_) {
        connection_state_callback_(false);
      }
      start_advertising_();
      return 0;
    case BLE_GAP_EVENT_SUBSCRIBE:
      if (event->subscribe.attr_handle == g_telemetry_val_handle) {
        telemetry_subscribed_ = event->subscribe.cur_notify != 0;
        if (telemetry_subscription_callback_) {
          telemetry_subscription_callback_(telemetry_subscribed_);
        }
      }
      return 0;
    case BLE_GAP_EVENT_ADV_COMPLETE:
      start_advertising_();
      return 0;
    default:
      return 0;
  }
}

int NimbleBleBindings::on_gatt_access_(uint16_t conn_handle, uint16_t attr_handle, ble_gatt_access_ctxt *ctxt) {
  (void) conn_handle;
  if (ctxt == nullptr) {
    return BLE_ATT_ERR_UNLIKELY;
  }

  if (attr_handle == g_sysinfo_val_handle && ctxt->op == BLE_GATT_ACCESS_OP_READ_CHR) {
    if (sysinfo_value_.empty()) {
      return 0;
    }
    return os_mbuf_append(ctxt->om, sysinfo_value_.data(), sysinfo_value_.size()) == 0 ? 0
                                                                                        : BLE_ATT_ERR_INSUFFICIENT_RES;
  }

  if (attr_handle == g_control_val_handle && ctxt->op == BLE_GATT_ACCESS_OP_WRITE_CHR) {
    char buffer[192] = {0};
    uint16_t command_len = 0;
    const int rc = ble_hs_mbuf_to_flat(ctxt->om, buffer, sizeof(buffer) - 1, &command_len);
    if (rc != 0) {
      return BLE_ATT_ERR_UNLIKELY;
    }
    buffer[command_len] = '\0';
    if (control_write_callback_) {
      control_write_callback_(std::string(buffer, command_len));
    }
    return 0;
  }

  return 0;
}

void NimbleBleBindings::host_task_(void *param) {
  (void) param;
  nimble_port_run();
  nimble_port_freertos_deinit();
}

void NimbleBleBindings::on_sync_static_() {
  if (instance_ != nullptr) {
    instance_->on_sync_();
  }
}

void NimbleBleBindings::on_reset_static_(int reason) {
  if (instance_ != nullptr) {
    instance_->on_reset_(reason);
  }
}

int NimbleBleBindings::gap_event_static_(ble_gap_event *event, void *arg) {
  NimbleBleBindings *bindings = static_cast<NimbleBleBindings *>(arg);
  return bindings != nullptr ? bindings->on_gap_event_(event) : 0;
}

int NimbleBleBindings::gatt_access_static_(uint16_t conn_handle,
                                           uint16_t attr_handle,
                                           ble_gatt_access_ctxt *ctxt,
                                           void *arg) {
  (void) arg;
  return instance_ != nullptr ? instance_->on_gatt_access_(conn_handle, attr_handle, ctxt) : BLE_ATT_ERR_UNLIKELY;
}

}  // namespace espectre
}  // namespace esphome
