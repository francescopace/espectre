/*
 * ESPectre - Wi-Fi CSI Movement Detection
 * Supports: ESP32-S3, ESP32-C6
 *
 * Uses Channel State Information (CSI) from Wi-Fi packets to detect movement.
 * Extracts 10 mathematical features and combines them with configurable weights
 * to distinguish between static environment and human movement.
 * 
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * License: GPLv3
 */

#include <stdio.h>
#include <string.h>
#include <math.h>
#include <stdatomic.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "freertos/semphr.h"
#include "esp_wifi.h"
#include "esp_event.h"
#include "esp_log.h"
#include "esp_system.h"
#include "nvs_flash.h"
#include "esp_timer.h"

// Module headers
#include "nvs_storage.h"
#include "filters.h"
#include "csi_processor.h"
#include "config_manager.h"
#include "mqtt_handler.h"
#include "mqtt_commands.h"
#include "traffic_generator.h"
#include "segmentation.h"
#include "esp_netif.h"

// Configuration - can be overridden via menuconfig
#define WIFI_SSID           CONFIG_WIFI_SSID
#define WIFI_PASSWORD       CONFIG_WIFI_PASSWORD
#define MQTT_BROKER_URI     CONFIG_MQTT_BROKER_URI
#define MQTT_TOPIC          CONFIG_MQTT_TOPIC
#define MQTT_USERNAME       CONFIG_MQTT_USERNAME
#define MQTT_PASSWORD       CONFIG_MQTT_PASSWORD

// Logging intervals
#define LOG_CSI_VALUES_INTERVAL 1
#define STATS_LOG_INTERVAL  100

// Publishing configuration
#define PUBLISH_INTERVAL    1.0f

// WiFi promiscuous mode (false = receive CSI only from connected AP, true = all WiFi packets)
#define PROMISCUOUS_MODE    false

// Array of all CSI feature indices (0-9) for feature extraction
static const uint8_t ALL_CSI_FEATURES[10] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};

static const char *TAG = "ESPectre";

static const char *g_response_topic = NULL;

// WiFi event group
static EventGroupHandle_t s_wifi_event_group = NULL;
#define WIFI_CONNECTED_BIT BIT0

static struct {
    uint32_t packets_received;
    uint32_t packets_processed;
    _Atomic uint32_t packets_dropped;
    
    csi_features_t current_features;
    
    // Module instances
    mqtt_handler_state_t mqtt_state;
    runtime_config_t config;
    
    // WiFi state
    bool wifi_connected;
    
    // Segmentation module
    segmentation_context_t segmentation;
    segmentation_state_t segmentation_state;
    
} g_state = {0};

// Mutex to protect g_state from concurrent access
static SemaphoreHandle_t g_state_mutex = NULL;

// Filter module instances
static butterworth_filter_t g_butterworth = {0};
static filter_buffer_t g_filter_buffer = {0};
static wavelet_state_t g_wavelet = {0};

// MQTT command context
static mqtt_cmd_context_t g_mqtt_cmd_context = {0};

// System start time for uptime calculation
static int64_t g_system_start_time = 0;

// WiFi event handler
static void wifi_event_handler(void* arg, esp_event_base_t event_base,
                               int32_t event_id, void* event_data) {
    if (event_base == WIFI_EVENT && event_id == WIFI_EVENT_STA_START) {
        ESP_LOGI(TAG, "WiFi STA started, attempting connection...");
        esp_wifi_connect();
    } else if (event_base == WIFI_EVENT && event_id == WIFI_EVENT_STA_DISCONNECTED) {
        wifi_event_sta_disconnected_t* disconnected = (wifi_event_sta_disconnected_t*) event_data;
        ESP_LOGW(TAG, "WiFi disconnected, reason: %d, reconnecting...", disconnected->reason);
        g_state.wifi_connected = false;
        esp_wifi_connect();
    } else if (event_base == IP_EVENT && event_id == IP_EVENT_STA_GOT_IP) {
        ip_event_got_ip_t* event = (ip_event_got_ip_t*) event_data;
        ESP_LOGI(TAG, "WiFi connected, got IP: " IPSTR, IP2STR(&event->ip_info.ip));
        g_state.wifi_connected = true;
        if (s_wifi_event_group) {
            xEventGroupSetBits(s_wifi_event_group, WIFI_CONNECTED_BIT);
        }
    }
}

// MQTT command callback
static void mqtt_command_callback(const char *data, int data_len) {
    mqtt_commands_process(data, data_len, &g_mqtt_cmd_context, 
                         &g_state.mqtt_state, MQTT_TOPIC "/response");
}

static inline int64_t get_timestamp_ms(void) {
    return esp_timer_get_time() / 1000;
}

static inline int64_t get_timestamp_sec(void) {
    return esp_timer_get_time() / 1000000;
}

// Build UTF-8 progress bar with threshold marker at 3/4 position
static void format_progress_bar(char *buffer, size_t size, float score, float threshold) {
    const int bar_width = 20;
    
    // Threshold marker at 3/4 position (15 out of 20)
    const int threshold_pos = 15;
    
    // Calculate percentage: score represents (moving_variance / adaptive_threshold)
    // At threshold_pos (75% of bar), score should be 1.0 (100% of threshold)
    // So we scale: filled = score * threshold_pos
    int filled = (int)(score * threshold_pos);
    
    // Clamp filled to bar width
    if (filled < 0) filled = 0;
    if (filled > bar_width) filled = bar_width;
    
    // Calculate percentage for display (score * 100)
    int percent = (int)(score * 100);
    if (percent > 200) percent = 200;  // Cap at 200% for display
    
    // Build bar directly in output buffer
    int pos = 0;
    pos += snprintf(buffer + pos, size - pos, "[");
    
    for (int i = 0; i < bar_width; i++) {
        if (i == threshold_pos) {
            pos += snprintf(buffer + pos, size - pos, "|");  // Threshold marker
        } else if (i < filled) {
            pos += snprintf(buffer + pos, size - pos, "█");  // Filled block (UTF-8)
        } else {
            pos += snprintf(buffer + pos, size - pos, "░");  // Empty block (UTF-8)
        }
    }
    
    snprintf(buffer + pos, size - pos, "] %d%%", percent);
}

static void csi_callback(void *ctx __attribute__((unused)), wifi_csi_info_t *data) {
    
    int8_t *csi_data = data->buf;
    size_t csi_len = data->len;
    
    if (csi_len < 10) {
        ESP_LOGW(TAG, "CSI data too short: %d bytes (minimum 10 required)", csi_len);
        return;
    }
    
    // Capture raw CSI packet if collection is active
    mqtt_commands_capture_csi_packet(csi_data, csi_len);
    
    // Protect g_state modifications with mutex (50ms timeout)
    if (xSemaphoreTake(g_state_mutex, pdMS_TO_TICKS(50)) == pdTRUE) {
        g_state.packets_received++;
        
        // Extract features if enabled (always, regardless of state)
        if (g_state.config.features_enabled) {
            csi_extract_features(csi_data, csi_len, &g_state.current_features,
                               ALL_CSI_FEATURES, 10);
        }
        
        // MVS Segmentation: Calculate spatial turbulence and update segmentation
        float turbulence = csi_calculate_spatial_turbulence(csi_data, csi_len,
                                                            g_state.config.selected_subcarriers,
                                                            g_state.config.num_selected_subcarriers);
        bool segment_completed = segmentation_add_turbulence(&g_state.segmentation, turbulence);
        
        // Update segmentation state
        g_state.segmentation_state = segmentation_get_state(&g_state.segmentation);
        
        if (segment_completed) {
            // A motion segment was just completed
            ESP_LOGD(TAG, "📍 Motion segment completed");
        }
        
        g_state.packets_processed++;
        
        xSemaphoreGive(g_state_mutex);
    } else {
        // Safely increment dropped packet counter
        uint32_t dropped = atomic_fetch_add(&g_state.packets_dropped, 1) + 1;
        
        if ((dropped % 100 == 0) || (dropped == 1)) {
            ESP_LOGW(TAG, "CSI callback: %lu packets dropped due to mutex timeout", 
                     (unsigned long)dropped);
        }
    }
}

static void csi_init(void) {
    // IMPORTANT: For ESP32-C6, promiscuous mode MUST be enabled BEFORE configuring CSI
    // This is different from ESP32-S3 where the order doesn't matter
    ESP_ERROR_CHECK(esp_wifi_set_promiscuous(PROMISCUOUS_MODE));
    ESP_LOGI(TAG, "Promiscuous mode: %s", PROMISCUOUS_MODE ? "enabled" : "disabled");
    
#if CONFIG_IDF_TARGET_ESP32C6
    // ESP32-C6 uses wifi_csi_acquire_config_t (different from ESP32-S3!)
    // Reference: https://github.com/espressif/esp-idf/issues/14271
    // CRITICAL: Must specify which CSI types to acquire, otherwise callback is never invoked!
    wifi_csi_config_t csi_config = {
        .enable = 1,                    // Master enable for CSI acquisition (REQUIRED)
        
        .acquire_csi_legacy = 1,        // Acquire L-LTF CSI from legacy 802.11a/g packets
                                        // CRITICAL: Required for CSI callback to be invoked!
                                        // Captures channel state from non-HT packets
        
        .acquire_csi_ht20 = 1,          // Acquire HT-LTF CSI from 802.11n HT20 packets
                                        // CRITICAL: Required for CSI from HT packets!
                                        // Provides improved channel estimation for MIMO
        
        .acquire_csi_ht40 = 0,          // Acquire HT-LTF CSI from 802.11n HT40 packets (40MHz bandwidth)
                                        // Enabled to capture CSI from HT40 packets (128 subcarriers)
                                        // Router will use HT40 if available and interference is low
        
        .acquire_csi_su = 1,            // Acquire HE-LTF CSI from 802.11ax HE20 SU (Single-User) packets
                                        // Enabled for WiFi 6 support (if router supports 802.11ax)
        
        .acquire_csi_mu = 0,            // Acquire HE-LTF CSI from 802.11ax HE20 MU (Multi-User) packets
                                        // Disabled (not using WiFi 6 MU-MIMO)
        
        .acquire_csi_dcm = 0,           // Acquire CSI from DCM (Dual Carrier Modulation) packets
                                        // DCM is a WiFi 6 feature for long-range transmission
                                        // Disabled (not used)
        
        .acquire_csi_beamformed = 0,    // Acquire CSI from beamformed packets
                                        // Beamforming directs signal toward receiver
                                        // Disabled (not needed for motion detection)
        
        .acquire_csi_he_stbc = 0,       // Acquire CSI from 802.11ax HE STBC packets
                                        // STBC improves reliability using multiple antennas
                                        // Disabled (not used)
        
        .val_scale_cfg = 0,             // CSI value scaling configuration (0-8)
                                        // 0 = automatic scaling (recommended)
                                        // 1-8 = manual scaling with shift bits
                                        // Controls normalization of CSI amplitude values
        
        .dump_ack_en = 0,               // Enable capture of 802.11 ACK frames
                                        // Disabled to reduce overhead (ACK frames not needed)
    };
    
    ESP_ERROR_CHECK(esp_wifi_set_csi_config(&csi_config));
    ESP_LOGI(TAG, "CSI initialized and enabled (ESP32-C6 mode)");
#else
    // ESP32 and ESP32-S3 use wifi_csi_config_t with legacy LTF fields
    wifi_csi_config_t csi_config = {
        .lltf_en = true,                // Enable Legacy Long Training Field (L-LTF) CSI capture
                                        // L-LTF is present in all 802.11a/g packets
                                        // Provides base channel estimation (64 subcarriers)
        
        .htltf_en = true,               // Enable HT Long Training Field (HT-LTF) CSI capture
                                        // HT-LTF is present in 802.11n (HT) packets
                                        // Provides improved channel estimation for MIMO
        
        .stbc_htltf2_en = true,         // Enable Space-Time Block Code HT-LTF2 capture
                                        // STBC uses 2 antennas to improve reliability
                                        // Captures second HT-LTF when STBC is active
        
        .ltf_merge_en = true,           // Merge L-LTF and HT-LTF data by averaging
                                        // true: Average L-LTF and HT-LTF for HT packets (more stable)
                                        // false: Use only HT-LTF for HT packets (more precise)
        
        .channel_filter_en = false,     // Channel filter to smooth adjacent subcarriers
                                        // true: Filter/smooth adjacent subcarriers (52 useful subcarriers)
                                        // false: Keep subcarrier independence (64 total subcarriers)
                                        // DISABLED to get all raw data without smoothing
        
        .manu_scale = false,            // Manual vs automatic CSI data scaling
                                        // false: Auto-scaling (recommended, adapts dynamically)
                                        // true: Manual scaling (requires .shift parameter)
    };
    
    ESP_ERROR_CHECK(esp_wifi_set_csi_config(&csi_config));
    ESP_LOGI(TAG, "CSI initialized and enabled (ESP32/ESP32-S3 mode)");
#endif
    
    // Common CSI setup for all targets
    ESP_ERROR_CHECK(esp_wifi_set_csi_rx_cb(csi_callback, NULL));
    ESP_ERROR_CHECK(esp_wifi_set_csi(true));
}

static void mqtt_publish_task(void *pvParameters) {
    TickType_t last_wake_time = xTaskGetTickCount();
    const TickType_t publish_period = pdMS_TO_TICKS((uint32_t)(PUBLISH_INTERVAL * 1000));
    
    int64_t last_csi_log_time = 0;
    
    // Smart publishing configuration
    mqtt_publish_config_t pub_config = {
        .enabled = g_state.config.smart_publishing_enabled,
        .delta_threshold = 0.05f,
        .max_interval_sec = 5.0f
    };
    
    while (1) {
        vTaskDelayUntil(&last_wake_time, publish_period);
        
        // Read g_state values with mutex protection
        segmentation_state_t seg_state;
        uint32_t packets_received, packets_processed;
        csi_features_t features;
        bool has_features = false;
        
        if (xSemaphoreTake(g_state_mutex, pdMS_TO_TICKS(100)) == pdTRUE) {
            seg_state = g_state.segmentation_state;
            packets_received = g_state.packets_received;
            packets_processed = g_state.packets_processed;
            
            // Copy features if they were extracted
            if (g_state.config.features_enabled) {
                features = g_state.current_features;
                has_features = true;
            }
            
            xSemaphoreGive(g_state_mutex);
        } else {
            ESP_LOGW(TAG, "MQTT publish task: Failed to acquire mutex, skipping publish cycle");
            continue;
        }
        
        // Calculate packet delta (packets processed since last cycle)
        static uint32_t last_packets_processed = 0;
        uint32_t packet_delta = packets_processed - last_packets_processed;
        last_packets_processed = packets_processed;
        
        // CSI logging with progress bar (always enabled)
        int64_t now = get_timestamp_sec();
        if (now - last_csi_log_time >= LOG_CSI_VALUES_INTERVAL) {
            
            // Get segmentation data
            float moving_variance = segmentation_get_moving_variance(&g_state.segmentation);
            float adaptive_threshold = segmentation_get_threshold(&g_state.segmentation);
            
            const char *seg_state_names[] = {"IDLE", "MOTION"};
            const char *seg_state_str = (seg_state < 2) ? seg_state_names[seg_state] : "UNKNOWN";
            
            // Calculate progress based on segmentation (moving_variance / threshold)
            float seg_progress = (adaptive_threshold > 0.0f) ? (moving_variance / adaptive_threshold) : 0.0f;
            
            // Format progress bar with threshold marker at 100%
            char progress_bar[256];
            format_progress_bar(progress_bar, sizeof(progress_bar), seg_progress, 1.0f);
            
            ESP_LOGI(TAG, "📊 %s | pkts:%lu mvmt:%.4f thr:%.4f | %s",
                     progress_bar, (unsigned long)packet_delta, 
                     moving_variance, adaptive_threshold, seg_state_str);
            last_csi_log_time = now;
        }
        
        // Publish segmentation data
        int64_t current_time = get_timestamp_ms();
        float moving_variance = segmentation_get_moving_variance(&g_state.segmentation);
        
        if (mqtt_should_publish(&g_state.mqtt_state, moving_variance, seg_state, 
                                &pub_config, current_time)) {
            // Prepare segmentation result
            segmentation_result_t result = {
                .moving_variance = moving_variance,
                .adaptive_threshold = segmentation_get_threshold(&g_state.segmentation),
                .state = seg_state,
                .timestamp = get_timestamp_sec(),
                .packets_processed = packet_delta,
                .has_features = has_features
            };
            
            if (has_features) {
                result.features = features;
            }
            
            mqtt_publish_segmentation(&g_state.mqtt_state, &result, MQTT_TOPIC);
            mqtt_update_publish_state(&g_state.mqtt_state, moving_variance, seg_state, current_time);
        }
        
        // Statistics logging
        if (packets_received > 0 && packets_received % STATS_LOG_INTERVAL == 0) {
            float success_rate = ((float)packets_processed / packets_received) * 100.0f;
            uint32_t packets_dropped_stat = atomic_load(&g_state.packets_dropped);
            ESP_LOGD(TAG, "Stats: %lu packets received, %lu processed (%.1f%% success), %lu dropped",
                     (unsigned long)packets_received, (unsigned long)packets_processed, 
                     success_rate, (unsigned long)packets_dropped_stat);
            
            // Only log smart publishing stats if the feature is enabled
            if (g_state.config.smart_publishing_enabled) {
                uint32_t published, skipped;
                mqtt_get_publish_stats(&g_state.mqtt_state, &published, &skipped);
                if (published + skipped > 0) {
                    float reduction = (skipped * 100.0f) / (published + skipped);
                    ESP_LOGD(TAG, "Smart Publishing: %lu published, %lu skipped (%.1f%% reduction)",
                             (unsigned long)published, (unsigned long)skipped, reduction);
                }
            }
        }
    }
}

void app_main(void) {
    ESP_LOGI(TAG, "System starting...");
    
    // Record system start time
    g_system_start_time = esp_timer_get_time() / 1000000;  // Convert to seconds
    
    // Initialize NVS
    esp_err_t ret = nvs_flash_init();
    if (ret == ESP_ERR_NVS_NO_FREE_PAGES || ret == ESP_ERR_NVS_NEW_VERSION_FOUND) {
        ESP_ERROR_CHECK(nvs_flash_erase());
        ret = nvs_flash_init();
    }
    ESP_ERROR_CHECK(ret);
    
    // Create WiFi event group
    s_wifi_event_group = xEventGroupCreate();
    if (!s_wifi_event_group) {
        ESP_LOGE(TAG, "Failed to create WiFi event group");
        return;
    }
    
    // Create mutex for g_state protection
    g_state_mutex = xSemaphoreCreateMutex();
    if (g_state_mutex == NULL) {
        ESP_LOGE(TAG, "Failed to create g_state mutex");
        return;
    }
    
    memset(&g_state, 0, sizeof(g_state));
    
    // Initialize configuration with defaults
    config_init_defaults(&g_state.config);
    
    // Initialize filter modules
    filter_buffer_init(&g_filter_buffer);
    butterworth_init(&g_butterworth);
    wavelet_init(&g_wavelet, g_state.config.wavelet_level, 
                 g_state.config.wavelet_threshold, WAVELET_THRESH_SOFT);
    
    // Initialize segmentation system
    segmentation_init(&g_state.segmentation);
    ESP_LOGI(TAG, "📍 Segmentation module initialized");
    
    // Initialize CSI processor with default subcarrier selection
    csi_set_subcarrier_selection(g_state.config.selected_subcarriers,
                                 g_state.config.num_selected_subcarriers);
    ESP_LOGI(TAG, "📡 CSI processor initialized with %d subcarriers", 
             g_state.config.num_selected_subcarriers);
    
    // Initialize NVS storage
    nvs_storage_init();
    
    // Load saved configuration if exists
    if (config_exists_in_nvs()) {
        nvs_config_data_t nvs_cfg;
        if (nvs_load_control_params(&nvs_cfg) == ESP_OK) {
            // Load config parameters (pass nvs_cfg to avoid duplicate NVS read)
            config_load_from_nvs(&g_state.config, &nvs_cfg);
            
            // Apply segmentation parameters from config
            segmentation_set_k_factor(&g_state.segmentation, g_state.config.segmentation_k_factor);
            segmentation_set_window_size(&g_state.segmentation, g_state.config.segmentation_window_size);
            segmentation_set_min_length(&g_state.segmentation, g_state.config.segmentation_min_length);
            segmentation_set_max_length(&g_state.segmentation, g_state.config.segmentation_max_length);
            
            // Load segmentation threshold from NVS
            segmentation_set_threshold(&g_state.segmentation, nvs_cfg.segmentation_threshold);
            
            // Load subcarrier selection from NVS
            if (nvs_cfg.num_selected_subcarriers > 0) {
                csi_set_subcarrier_selection(nvs_cfg.selected_subcarriers,
                                            nvs_cfg.num_selected_subcarriers);
                ESP_LOGI(TAG, "📡 Loaded subcarrier selection: %d subcarriers", 
                         nvs_cfg.num_selected_subcarriers);
            }
            
            ESP_LOGI(TAG, "💾 Loaded saved configuration from NVS");
            ESP_LOGI(TAG, "📍 Segmentation: threshold=%.2f, K=%.2f, window=%d, min=%d, max=%d",
                     nvs_cfg.segmentation_threshold,
                     g_state.config.segmentation_k_factor,
                     g_state.config.segmentation_window_size,
                     g_state.config.segmentation_min_length,
                     g_state.config.segmentation_max_length);
        }
    }
    
    // Initialize WiFi
    ESP_LOGI(TAG, "Initializing WiFi...");
    g_state.wifi_connected = false;
    
    ESP_ERROR_CHECK(esp_netif_init());
    ESP_ERROR_CHECK(esp_event_loop_create_default());
    esp_netif_create_default_wifi_sta();
    
    wifi_init_config_t cfg = WIFI_INIT_CONFIG_DEFAULT();
    ESP_ERROR_CHECK(esp_wifi_init(&cfg));
    
    ESP_ERROR_CHECK(esp_event_handler_register(WIFI_EVENT, ESP_EVENT_ANY_ID, &wifi_event_handler, NULL));
    ESP_ERROR_CHECK(esp_event_handler_register(IP_EVENT, IP_EVENT_STA_GOT_IP, &wifi_event_handler, NULL));
    
    wifi_config_t wifi_config = {0};
    strncpy((char *)wifi_config.sta.ssid, WIFI_SSID, sizeof(wifi_config.sta.ssid) - 1);
    wifi_config.sta.ssid[sizeof(wifi_config.sta.ssid) - 1] = '\0';
    strncpy((char *)wifi_config.sta.password, WIFI_PASSWORD, sizeof(wifi_config.sta.password) - 1);
    wifi_config.sta.password[sizeof(wifi_config.sta.password) - 1] = '\0';
    
    ESP_LOGI(TAG, "WiFi SSID: %s", wifi_config.sta.ssid);
    
    ESP_ERROR_CHECK(esp_wifi_set_mode(WIFI_MODE_STA));
    ESP_ERROR_CHECK(esp_wifi_set_config(WIFI_IF_STA, &wifi_config));
    
    // Configure Wi-Fi country code (regulatory compliance)
    // Note: schan=1 and nchan=13 are standard initial values.
    // With WIFI_COUNTRY_POLICY_AUTO, the ESP32 driver automatically adapts
    // the channel range based on the country code (e.g., 1-11 for US, 1-13 for EU, 1-14 for JP).
    wifi_country_t country = {
        .cc = CONFIG_WIFI_COUNTRY_CODE,
        .schan = 1,        // Standard start channel (driver adapts based on country code)
        .nchan = 13,       // Standard channel count (driver adapts based on country code)
        .policy = WIFI_COUNTRY_POLICY_AUTO,
    };
    ESP_ERROR_CHECK(esp_wifi_set_country(&country));
    ESP_LOGI(TAG, "Wi-Fi country code set to %s (channels auto-configured by driver)", 
             CONFIG_WIFI_COUNTRY_CODE);
    
    ESP_ERROR_CHECK(esp_wifi_start());
    
    // Configure Wi-Fi power management (disable for real-time CSI with minimal latency)
    ESP_ERROR_CHECK(esp_wifi_set_ps(WIFI_PS_NONE));
    ESP_LOGI(TAG, "Wi-Fi power save disabled for real-time CSI");
    
    // Configure Wi-Fi protocol mode
#if CONFIG_IDF_TARGET_ESP32C6
    // ESP32-C6: Enable WiFi 6 (802.11ax) for improved performance and CSI capture
    ESP_ERROR_CHECK(esp_wifi_set_protocol(WIFI_IF_STA, 
        WIFI_PROTOCOL_11B | WIFI_PROTOCOL_11G | WIFI_PROTOCOL_11N | WIFI_PROTOCOL_11AX));
    ESP_LOGI(TAG, "Wi-Fi protocol set to 802.11b/g/n/ax (WiFi 6 enabled)");
#else
    // ESP32-S3: WiFi 4 only (802.11b/g/n)
    ESP_ERROR_CHECK(esp_wifi_set_protocol(WIFI_IF_STA, 
        WIFI_PROTOCOL_11B | WIFI_PROTOCOL_11G | WIFI_PROTOCOL_11N));
    ESP_LOGI(TAG, "Wi-Fi protocol set to 802.11b/g/n");
#endif
    
    // Configure Wi-Fi bandwidth (HT20 for stability, HT40 for more subcarriers)
    ESP_ERROR_CHECK(esp_wifi_set_bandwidth(WIFI_IF_STA, WIFI_BW_HT20));
    ESP_LOGI(TAG, "Wi-Fi bandwidth set to HT20 (20MHz)");
    
    // Wait for WiFi connection
    ESP_LOGI(TAG, "Waiting for WiFi connection...");
    xEventGroupWaitBits(s_wifi_event_group, WIFI_CONNECTED_BIT, pdFALSE, pdTRUE, portMAX_DELAY);
    ESP_LOGI(TAG, "WiFi connected successfully");
    
    // Initialize MQTT
    mqtt_config_t mqtt_cfg = {
        .broker_uri = MQTT_BROKER_URI,
        .username = MQTT_USERNAME,
        .password = MQTT_PASSWORD,
        .base_topic = MQTT_TOPIC,
        .cmd_topic = MQTT_TOPIC "/cmd",
        .response_topic = MQTT_TOPIC "/response"
    };
    
    // Store response topic globally for use in mqtt_publish_task
    g_response_topic = mqtt_cfg.response_topic;
    
    if (mqtt_handler_init(&g_state.mqtt_state, &mqtt_cfg) != 0) {
        ESP_LOGE(TAG, "Failed to initialize MQTT handler");
        return;
    }
    
    if (mqtt_handler_start(&g_state.mqtt_state) != 0) {
        ESP_LOGE(TAG, "Failed to start MQTT client");
        return;
    }
    
    // Setup MQTT command context
    g_mqtt_cmd_context.config = &g_state.config;
    g_mqtt_cmd_context.current_features = &g_state.current_features;
    g_mqtt_cmd_context.current_state = &g_state.segmentation_state;
    g_mqtt_cmd_context.butterworth = &g_butterworth;
    g_mqtt_cmd_context.filter_buffer = &g_filter_buffer;
    g_mqtt_cmd_context.segmentation = &g_state.segmentation;
    g_mqtt_cmd_context.mqtt_base_topic = mqtt_cfg.base_topic;
    g_mqtt_cmd_context.mqtt_cmd_topic = mqtt_cfg.cmd_topic;
    g_mqtt_cmd_context.mqtt_response_topic = mqtt_cfg.response_topic;
    g_mqtt_cmd_context.system_start_time = &g_system_start_time;
    
    // Initialize MQTT commands
    if (mqtt_commands_init(&g_state.mqtt_state, &g_mqtt_cmd_context) != 0) {
        ESP_LOGE(TAG, "Failed to initialize MQTT commands");
        return;
    }
    
    // Set command callback
    mqtt_handler_set_command_callback(mqtt_command_callback);
    
    ESP_LOGI(TAG, "MQTT client started with command support");
    
    // Initialize traffic generator
    traffic_generator_init();
    
    // Start traffic generator with configured rate (if > 0)
    // Can be changed later via MQTT command
    if (g_state.config.traffic_generator_rate > 0) {
        if (traffic_generator_start(g_state.config.traffic_generator_rate)) {
            ESP_LOGI(TAG, "✅ Traffic generator started (%u pps)", 
                     (unsigned int)g_state.config.traffic_generator_rate);
        } else {
            ESP_LOGW(TAG, "⚠️  Failed to start traffic generator");
        }
    }
    
    // Initialize CSI
    csi_init();
    
    // Start MQTT publish task
    xTaskCreate(mqtt_publish_task, "mqtt_pub", 4096, NULL, 5, NULL);
    
    ESP_LOGI(TAG, "");
    ESP_LOGI(TAG, "╔═══════════════════════════════════════════════════════════╗");
    ESP_LOGI(TAG, "║                   🛜  E S P e c t r e 👻                   ║");
    ESP_LOGI(TAG, "║                                                           ║");
    ESP_LOGI(TAG, "║                Wi-Fi motion detection system              ║");
    ESP_LOGI(TAG, "║          based on Channel State Information (CSI)         ║");
    ESP_LOGI(TAG, "║                                                           ║");
    ESP_LOGI(TAG, "╠═══════════════════════════════════════════════════════════╣");
    ESP_LOGI(TAG, "║                                                           ║");
    ESP_LOGI(TAG, "║             System Ready - Monitoring Active              ║");
    ESP_LOGI(TAG, "║               Detecting the invisible... 👁️                ║");
    ESP_LOGI(TAG, "║                                                           ║");
    ESP_LOGI(TAG, "╚═══════════════════════════════════════════════════════════╝");
    ESP_LOGI(TAG, "");
}
