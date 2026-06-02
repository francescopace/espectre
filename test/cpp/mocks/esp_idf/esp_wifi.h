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

typedef enum {
  WIFI_SECOND_CHAN_NONE = 0,
  WIFI_SECOND_CHAN_ABOVE,
  WIFI_SECOND_CHAN_BELOW,
} wifi_second_chan_t;

// WiFi Band mode (dual-band capable targets)
typedef enum {
  WIFI_BAND_MODE_2G_ONLY = 1,
  WIFI_BAND_MODE_5G_ONLY = 2,
  WIFI_BAND_MODE_AUTO = 3,
} wifi_band_mode_t;

// WiFi Protocols
#define WIFI_PROTOCOL_11B 1
#define WIFI_PROTOCOL_11G 2
#define WIFI_PROTOCOL_11N 4
#define WIFI_PROTOCOL_LR 8
#define WIFI_PROTOCOL_11A 16
#define WIFI_PROTOCOL_11AC 32
#define WIFI_PROTOCOL_11AX 64

typedef struct {
  uint16_t ghz_2g;
  uint16_t ghz_5g;
} wifi_protocols_t;

typedef struct {
  wifi_bandwidth_t ghz_2g;
  wifi_bandwidth_t ghz_5g;
} wifi_bandwidths_t;

// WiFi Power Save
typedef enum {
  WIFI_PS_NONE,
  WIFI_PS_MIN_MODEM,
  WIFI_PS_MAX_MODEM,
} wifi_ps_type_t;

// CSI callback type
typedef void (*wifi_csi_cb_t)(void *ctx, wifi_csi_info_t *data);

typedef struct {
  int8_t rssi;
  uint8_t primary;
} wifi_ap_record_t;

typedef struct {
  esp_err_t set_protocol_results[4];
  int set_protocol_result_count;
  int set_protocol_call_count;
  uint8_t last_protocol_bitmap;

  esp_err_t set_protocols_result;
  int set_protocols_call_count;
  wifi_protocols_t last_protocols;

  esp_err_t get_protocol_result;
  uint8_t protocol_bitmap;

  esp_err_t get_protocols_result;
  wifi_protocols_t protocols;

  esp_err_t set_bandwidth_result;
  int set_bandwidth_call_count;
  wifi_bandwidth_t last_bandwidth;

  esp_err_t set_bandwidths_result;
  int set_bandwidths_call_count;
  wifi_bandwidths_t last_bandwidths;

  esp_err_t get_bandwidth_result;
  wifi_bandwidth_t bandwidth;

  esp_err_t get_bandwidths_result;
  wifi_bandwidths_t bandwidths;

  esp_err_t set_promiscuous_result;
  int set_promiscuous_call_count;
  bool last_promiscuous;

  esp_err_t get_promiscuous_result;
  bool promiscuous;

  esp_err_t get_ps_result;
  wifi_ps_type_t ps_type;

  esp_err_t get_channel_result;
  uint8_t primary_channel;
  wifi_second_chan_t second_channel;

  esp_err_t set_band_mode_result;
  int set_band_mode_call_count;
  wifi_band_mode_t last_band_mode;
} esp_wifi_mock_state_t;

extern esp_wifi_mock_state_t g_esp_wifi_mock;

void esp_wifi_mock_reset(void);

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
  g_esp_wifi_mock.set_promiscuous_call_count++;
  g_esp_wifi_mock.last_promiscuous = en;
  if (g_esp_wifi_mock.set_promiscuous_result == ESP_OK) {
    g_esp_wifi_mock.promiscuous = en;
  }
  return g_esp_wifi_mock.set_promiscuous_result;
}

static inline esp_err_t esp_wifi_get_promiscuous(bool *en) {
  if (en) {
    *en = g_esp_wifi_mock.promiscuous;
  }
  return g_esp_wifi_mock.get_promiscuous_result;
}

static inline esp_err_t esp_wifi_set_bandwidth(wifi_interface_t ifx,
                                               wifi_bandwidth_t bw) {
  (void)ifx;
  g_esp_wifi_mock.set_bandwidth_call_count++;
  g_esp_wifi_mock.last_bandwidth = bw;
  if (g_esp_wifi_mock.set_bandwidth_result == ESP_OK) {
    g_esp_wifi_mock.bandwidth = bw;
  }
  return g_esp_wifi_mock.set_bandwidth_result;
}

static inline esp_err_t esp_wifi_set_band_mode(wifi_band_mode_t band_mode) {
  g_esp_wifi_mock.set_band_mode_call_count++;
  g_esp_wifi_mock.last_band_mode = band_mode;
  return g_esp_wifi_mock.set_band_mode_result;
}

static inline esp_err_t esp_wifi_set_protocol(wifi_interface_t ifx,
                                              uint8_t protocol_bitmap) {
  (void)ifx;
  esp_err_t result = ESP_OK;
  if (g_esp_wifi_mock.set_protocol_call_count < g_esp_wifi_mock.set_protocol_result_count) {
    result =
        g_esp_wifi_mock.set_protocol_results[g_esp_wifi_mock.set_protocol_call_count];
  }
  g_esp_wifi_mock.set_protocol_call_count++;
  g_esp_wifi_mock.last_protocol_bitmap = protocol_bitmap;
  if (result == ESP_OK) {
    g_esp_wifi_mock.protocol_bitmap = protocol_bitmap;
  }
  return result;
}

static inline esp_err_t esp_wifi_set_protocols(wifi_interface_t ifx,
                                               wifi_protocols_t *protocols) {
  (void)ifx;
  g_esp_wifi_mock.set_protocols_call_count++;
  if (protocols) {
    g_esp_wifi_mock.last_protocols = *protocols;
    if (g_esp_wifi_mock.set_protocols_result == ESP_OK) {
      g_esp_wifi_mock.protocols = *protocols;
    }
  }
  return g_esp_wifi_mock.set_protocols_result;
}

static inline esp_err_t esp_wifi_get_protocol(wifi_interface_t ifx,
                                              uint8_t *protocol_bitmap) {
  (void)ifx;
  if (protocol_bitmap) {
    *protocol_bitmap = g_esp_wifi_mock.protocol_bitmap;
  }
  return g_esp_wifi_mock.get_protocol_result;
}

static inline esp_err_t esp_wifi_get_protocols(wifi_interface_t ifx,
                                               wifi_protocols_t *protocols) {
  (void)ifx;
  if (protocols) {
    *protocols = g_esp_wifi_mock.protocols;
  }
  return g_esp_wifi_mock.get_protocols_result;
}

static inline esp_err_t esp_wifi_get_ps(wifi_ps_type_t *ps_type) {
  if (ps_type) {
    *ps_type = g_esp_wifi_mock.ps_type;
  }
  return g_esp_wifi_mock.get_ps_result;
}

static inline esp_err_t esp_wifi_set_bandwidths(wifi_interface_t ifx,
                                                wifi_bandwidths_t *bw) {
  (void)ifx;
  g_esp_wifi_mock.set_bandwidths_call_count++;
  if (bw) {
    g_esp_wifi_mock.last_bandwidths = *bw;
    if (g_esp_wifi_mock.set_bandwidths_result == ESP_OK) {
      g_esp_wifi_mock.bandwidths = *bw;
    }
  }
  return g_esp_wifi_mock.set_bandwidths_result;
}

static inline esp_err_t esp_wifi_get_bandwidth(wifi_interface_t ifx,
                                                wifi_bandwidth_t *bw) {
  (void)ifx;
  if (bw) {
    *bw = g_esp_wifi_mock.bandwidth;
  }
  return g_esp_wifi_mock.get_bandwidth_result;
}

static inline esp_err_t esp_wifi_get_bandwidths(wifi_interface_t ifx,
                                                wifi_bandwidths_t *bw) {
  (void)ifx;
  if (bw) {
    *bw = g_esp_wifi_mock.bandwidths;
  }
  return g_esp_wifi_mock.get_bandwidths_result;
}

static inline esp_err_t esp_wifi_get_channel(uint8_t *primary,
                                             wifi_second_chan_t *second) {
  if (primary) {
    *primary = g_esp_wifi_mock.primary_channel;
  }
  if (second) {
    *second = g_esp_wifi_mock.second_channel;
  }
  return g_esp_wifi_mock.get_channel_result;
}

static inline esp_err_t esp_wifi_sta_get_ap_info(wifi_ap_record_t *ap_info) {
  if (ap_info) {
    ap_info->rssi = -55;
    ap_info->primary = 6;
  }
  return ESP_OK;
}

#ifdef __cplusplus
}
#endif

#endif // ESP_WIFI_H
