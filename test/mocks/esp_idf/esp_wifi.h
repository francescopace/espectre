#ifndef ESP_WIFI_H
#define ESP_WIFI_H

#include "esp_err.h"
#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// WiFi mode
typedef enum {
  WIFI_MODE_NULL = 0,
  WIFI_MODE_STA,
  WIFI_MODE_AP,
  WIFI_MODE_APSTA,
  WIFI_MODE_MAX
} wifi_mode_t;

// WiFi interface
typedef enum { WIFI_IF_STA = 0, WIFI_IF_AP, WIFI_IF_MAX } wifi_interface_t;

// CSI configuration
// Note: field order must match designated initializer order in csi_manager.cpp
typedef struct {
  // ESP32-S3 fields (used in non-C6 builds)
  bool lltf_en;
  bool htltf_en;
  bool stbc_htltf2_en;
  bool ltf_merge_en;
  bool channel_filter_en;
  bool manu_scale;
  uint8_t shift;
  // ESP32-C6 fields
  bool enable;
  bool acquire_csi_legacy;
  bool acquire_csi_ht20;
  bool acquire_csi_ht40;
  bool acquire_csi_su;
  uint8_t acquire_csi_mu;
  uint8_t acquire_csi_dcm;
  uint8_t acquire_csi_beamformed;
  uint8_t acquire_csi_he_stbc;
  uint8_t val_scale_cfg;
  uint8_t dump_ack_en;
} wifi_csi_config_t;

// RX control structure (matches ESP-IDF wifi_pkt_rx_ctrl_t)
typedef struct {
  int8_t rssi;
  uint8_t rate;
  uint8_t sig_mode;
  uint8_t mcs;
  uint8_t cwb;
  uint8_t smoothing;
  uint8_t not_sounding;
  uint8_t aggregation;
  uint8_t stbc;
  uint8_t fec_coding;
  uint8_t sgi;
  int8_t noise_floor;
  uint8_t ampdu_cnt;
  uint8_t channel;
  uint8_t secondary_channel;
  uint32_t timestamp;
  uint8_t ant;
  uint16_t sig_len;
  uint8_t rx_state;
} wifi_pkt_rx_ctrl_t;

// CSI info structure (matches ESP-IDF wifi_csi_info_t)
typedef struct {
  wifi_pkt_rx_ctrl_t rx_ctrl;
  uint8_t mac[6];
  uint8_t dmac[6];
  bool first_word_invalid;
  int8_t *buf;
  uint16_t len;
  uint8_t *hdr;
  uint8_t *payload;
  uint16_t payload_len;
  uint16_t rx_seq;
} wifi_csi_info_t;

// WiFi Bandwidth
typedef enum {
  WIFI_BW_HT20 = 1,
  WIFI_BW_HT40,
} wifi_bandwidth_t;

// WiFi Protocols
#define WIFI_PROTOCOL_11B 1
#define WIFI_PROTOCOL_11G 2
#define WIFI_PROTOCOL_11N 4
#define WIFI_PROTOCOL_LR 8
#define WIFI_PROTOCOL_11AX 16

// WiFi Power Save
typedef enum {
  WIFI_PS_NONE,
  WIFI_PS_MIN_MODEM,
  WIFI_PS_MAX_MODEM,
} wifi_ps_type_t;

// CSI callback type
typedef void (*wifi_csi_cb_t)(void *ctx, wifi_csi_info_t *data);

// Mock WiFi functions
static inline esp_err_t esp_wifi_set_mode(wifi_mode_t mode) {
  (void)mode;
  return ESP_OK;
}

static inline esp_err_t esp_wifi_get_mode(wifi_mode_t *mode) {
  if (mode)
    *mode = WIFI_MODE_STA;
  return ESP_OK;
}

static inline esp_err_t esp_wifi_start(void) { return ESP_OK; }

static inline esp_err_t esp_wifi_stop(void) { return ESP_OK; }

static inline esp_err_t
esp_wifi_set_csi_config(const wifi_csi_config_t *config) {
  (void)config;
  return ESP_OK;
}

static inline esp_err_t esp_wifi_set_csi_rx_cb(wifi_csi_cb_t cb, void *ctx) {
  (void)cb;
  (void)ctx;
  return ESP_OK;
}

static inline esp_err_t esp_wifi_set_csi(bool en) {
  (void)en;
  return ESP_OK;
}

static inline esp_err_t esp_wifi_set_promiscuous(bool en) {
  (void)en;
  return ESP_OK;
}

static inline esp_err_t esp_wifi_get_promiscuous(bool *en) {
  if (en)
    *en = false;
  return ESP_OK;
}

static inline esp_err_t esp_wifi_set_bandwidth(wifi_interface_t ifx,
                                               wifi_bandwidth_t bw) {
  (void)ifx;
  (void)bw;
  return ESP_OK;
}

static inline esp_err_t esp_wifi_get_protocol(wifi_interface_t ifx,
                                              uint8_t *protocol_bitmap) {
  (void)ifx;
  if (protocol_bitmap)
    *protocol_bitmap =
        WIFI_PROTOCOL_11B | WIFI_PROTOCOL_11G | WIFI_PROTOCOL_11N;
  return ESP_OK;
}

static inline esp_err_t esp_wifi_get_ps(wifi_ps_type_t *ps_type) {
  if (ps_type)
    *ps_type = WIFI_PS_NONE;
  return ESP_OK;
}

static inline esp_err_t esp_wifi_get_bandwidth(wifi_interface_t ifx,
                                                wifi_bandwidth_t *bw) {
  (void)ifx;
  if (bw)
    *bw = WIFI_BW_HT20;
  return ESP_OK;
}

#ifdef __cplusplus
}
#endif

#endif // ESP_WIFI_H
