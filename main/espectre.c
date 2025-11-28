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
#include "nbvi_calibrator.h"
#include "esp_netif.h"
#include "espectre.h"
#include "cJSON.h"

// Configuration - can be overridden via menuconfig
#define WIFI_SSID           CONFIG_WIFI_SSID
#define WIFI_PASSWORD       CONFIG_WIFI_PASSWORD
#define MQTT_BROKER_URI     CONFIG_MQTT_BROKER_URI
#define MQTT_TOPIC          CONFIG_MQTT_TOPIC
#define MQTT_USERNAME       CONFIG_MQTT_USERNAME
#define MQTT_PASSWORD       CONFIG_MQTT_PASSWORD

// Array of all CSI feature indices (0-9) for feature extraction
static const uint8_t ALL_CSI_FEATURES[10] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};

static const char *TAG = "ESPectre";

static const char *g_response_topic = NULL;

// WiFi event group
static EventGroupHandle_t s_wifi_event_group = NULL;
#define WIFI_CONNECTED_BIT BIT0

static struct {
    uint32_t packets_processed;
    _Atomic uint32_t packets_dropped;
    
    csi_features_t current_features;
    
    // Module instances
    mqtt_handler_state_t mqtt_state;
    runtime_config_t config;
    
    // WiFi state
    bool wifi_connected;
    
    // CSI processor (unified: feature extraction + motion detection)
    csi_processor_context_t csi_processor;
    csi_motion_state_t motion_state;
    
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

// NBVI calibrator pointer (used during calibration to collect packets)
static nbvi_calibrator_t *g_nbvi_calibrator = NULL;

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

static void csi_callback(void *ctx __attribute__((unused)), wifi_csi_info_t *data) {
    
    int8_t *csi_data = data->buf;
    size_t csi_len = data->len;
    
    if (csi_len < 10) {
        ESP_LOGW(TAG, "CSI data too short: %d bytes (minimum 10 required)", csi_len);
        return;
    }
    
    // If NBVI calibration is in progress, add packet to calibrator buffer
    if (g_nbvi_calibrator != NULL) {
        nbvi_calibrator_add_packet(g_nbvi_calibrator, csi_data, csi_len);
        return;  // Skip normal processing during calibration
    }
    
    // Protect g_state modifications with mutex (50ms timeout)
    if (xSemaphoreTake(g_state_mutex, pdMS_TO_TICKS(50)) == pdTRUE) {
        
        // Process CSI packet: turbulence calculation, motion detection, and optional feature extraction
        // Uses unified csi_process_packet() which handles everything in one call
        csi_process_packet(&g_state.csi_processor,
                          csi_data, csi_len,
                          g_state.config.selected_subcarriers,
                          g_state.config.num_selected_subcarriers,
                          g_state.config.features_enabled ? &g_state.current_features : NULL,
                          ALL_CSI_FEATURES, 10);
        
        // Update motion state
        g_state.motion_state = csi_processor_get_state(&g_state.csi_processor);
        
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

/**
 * Run NBVI auto-calibration
 * 
 * Collects CSI packets and automatically selects optimal subcarriers
 * using NBVI Weighted α=0.3 algorithm with percentile-based detection.
 * 
 * @return true if calibration successful, false otherwise
 */
bool run_nbvi_calibration(void) {
    ESP_LOGI(TAG, "");
    ESP_LOGI(TAG, "═══════════════════════════════════════════════════════════");
    ESP_LOGI(TAG, "🧬 NBVI Auto-Calibration Starting");
    ESP_LOGI(TAG, "═══════════════════════════════════════════════════════════");
    ESP_LOGI(TAG, "Please remain still for 5 seconds...");
    ESP_LOGI(TAG, "Collecting CSI data for automatic subcarrier selection");
    ESP_LOGI(TAG, "");
    
    // Initialize NBVI calibrator
    nbvi_calibrator_t calibrator;
    esp_err_t err = nbvi_calibrator_init(&calibrator);
    if (err != ESP_OK) {
        ESP_LOGE(TAG, "Failed to initialize NBVI calibrator: %s", esp_err_to_name(err));
        return false;
    }
    
    // Set global pointer for CSI callback to populate calibrator buffer
    g_nbvi_calibrator = &calibrator;
    
    // Wait for CSI callback to collect packets
    ESP_LOGI(TAG, "NBVI: Collecting packets via CSI callback...");
    uint16_t timeout_counter = 0;
    const uint16_t max_timeout_ms = 15000;  // 15 seconds timeout
    uint16_t last_count = 0;
    
    while (calibrator.buffer_count < NBVI_BUFFER_SIZE) {
        vTaskDelay(pdMS_TO_TICKS(100));  // Check every 100ms
        timeout_counter += 100;
        
        // Print progress when count changes
        if (calibrator.buffer_count != last_count) {
            if (calibrator.buffer_count % 100 == 0) {
                ESP_LOGI(TAG, "NBVI: Collected %d/%d packets...",
                         calibrator.buffer_count, NBVI_BUFFER_SIZE);
            }
            last_count = calibrator.buffer_count;
            timeout_counter = 0;  // Reset timeout on progress
        }
        
        // Check for timeout
        if (timeout_counter >= max_timeout_ms) {
            ESP_LOGE(TAG, "NBVI: Timeout waiting for CSI packets (collected %d/%d)",
                     calibrator.buffer_count, NBVI_BUFFER_SIZE);
            ESP_LOGE(TAG, "NBVI: Calibration aborted - using default band");
            g_nbvi_calibrator = NULL;
            nbvi_calibrator_free(&calibrator);
            return false;
        }
    }
    
    // Clear global pointer
    g_nbvi_calibrator = NULL;
    
    ESP_LOGI(TAG, "NBVI: Collection complete (%d packets)", NBVI_BUFFER_SIZE);
    
    // Run calibration
    uint8_t selected_band[12];
    uint8_t band_size = 0;
    
    err = nbvi_calibrator_calibrate(&calibrator,
                                    g_state.config.selected_subcarriers,
                                    g_state.config.num_selected_subcarriers,
                                    selected_band,
                                    &band_size);
    
    bool success = (err == ESP_OK && band_size == 12);
    
    if (success) {
        // Update configuration
        memcpy(g_state.config.selected_subcarriers, selected_band, 12);
        g_state.config.num_selected_subcarriers = 12;
        
        // Update CSI processor
        csi_set_subcarrier_selection(selected_band, 12);
        
        // Save to NVS
        config_save_to_nvs(&g_state.config, 
                          csi_processor_get_threshold(&g_state.csi_processor));
        
        ESP_LOGI(TAG, "");
        ESP_LOGI(TAG, "═══════════════════════════════════════════════════════════");
        ESP_LOGI(TAG, "✅ NBVI Calibration Successful!");
        ESP_LOGI(TAG, "   Selected band: [%d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d]",
                 selected_band[0], selected_band[1], selected_band[2], selected_band[3],
                 selected_band[4], selected_band[5], selected_band[6], selected_band[7],
                 selected_band[8], selected_band[9], selected_band[10], selected_band[11]);
        ESP_LOGI(TAG, "   Configuration saved to NVS");
        ESP_LOGI(TAG, "═══════════════════════════════════════════════════════════");
        ESP_LOGI(TAG, "");
    } else {
        ESP_LOGW(TAG, "");
        ESP_LOGW(TAG, "═══════════════════════════════════════════════════════════");
        ESP_LOGW(TAG, "⚠️  NBVI Calibration Failed");
        ESP_LOGW(TAG, "   Using default band: [%d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d]",
                 g_state.config.selected_subcarriers[0], g_state.config.selected_subcarriers[1],
                 g_state.config.selected_subcarriers[2], g_state.config.selected_subcarriers[3],
                 g_state.config.selected_subcarriers[4], g_state.config.selected_subcarriers[5],
                 g_state.config.selected_subcarriers[6], g_state.config.selected_subcarriers[7],
                 g_state.config.selected_subcarriers[8], g_state.config.selected_subcarriers[9],
                 g_state.config.selected_subcarriers[10], g_state.config.selected_subcarriers[11]);
        ESP_LOGW(TAG, "═══════════════════════════════════════════════════════════");
        ESP_LOGW(TAG, "");
    }
    
    // Cleanup
    nbvi_calibrator_free(&calibrator);
    
    return success;
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
    
    // Initialize CSI processor (unified: feature extraction + motion detection)
    csi_processor_init(&g_state.csi_processor);
    ESP_LOGI(TAG, "📍 CSI processor initialized");
    
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
            
            // Apply motion detection parameters from config
            csi_processor_set_window_size(&g_state.csi_processor, g_state.config.segmentation_window_size);
            
            // Load motion detection threshold from NVS
            csi_processor_set_threshold(&g_state.csi_processor, nvs_cfg.segmentation_threshold);
            
            // Load subcarrier selection from NVS
            if (nvs_cfg.num_selected_subcarriers > 0) {
                csi_set_subcarrier_selection(nvs_cfg.selected_subcarriers,
                                            nvs_cfg.num_selected_subcarriers);
                ESP_LOGI(TAG, "📡 Loaded subcarrier selection: %d subcarriers", 
                         nvs_cfg.num_selected_subcarriers);
            }
            
            ESP_LOGI(TAG, "💾 Loaded saved configuration from NVS");
            ESP_LOGI(TAG, "📍 Motion detection: threshold=%.2f, window=%d",
                     nvs_cfg.segmentation_threshold,
                     g_state.config.segmentation_window_size);
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
    g_mqtt_cmd_context.current_state = &g_state.motion_state;
    g_mqtt_cmd_context.butterworth = &g_butterworth;
    g_mqtt_cmd_context.filter_buffer = &g_filter_buffer;
    g_mqtt_cmd_context.csi_processor = &g_state.csi_processor;
    g_mqtt_cmd_context.mqtt_base_topic = mqtt_cfg.base_topic;
    g_mqtt_cmd_context.mqtt_cmd_topic = mqtt_cfg.cmd_topic;
    g_mqtt_cmd_context.mqtt_response_topic = mqtt_cfg.response_topic;
    g_mqtt_cmd_context.system_start_time = &g_system_start_time;
    g_mqtt_cmd_context.packets_dropped = &g_state.packets_dropped;
    
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
    
    // NBVI Auto-Calibration (if enabled and no saved configuration)
    if (NBVI_ENABLED && !config_exists_in_nvs()) {
        ESP_LOGI(TAG, "🧬 NBVI auto-calibration enabled (no saved configuration)");
        run_nbvi_calibration();
    } else if (NBVI_ENABLED) {
        ESP_LOGI(TAG, "🧬 NBVI: Using saved subcarrier configuration from NVS");
    } else {
        ESP_LOGI(TAG, "🧬 NBVI: Auto-calibration disabled");
    }
    
    // Setup MQTT publisher context (static to persist after app_main returns)
    static mqtt_publisher_context_t pub_context = {0};
    pub_context.mqtt_state = &g_state.mqtt_state;
    pub_context.csi_processor = &g_state.csi_processor;
    pub_context.config = &g_state.config;
    pub_context.current_features = &g_state.current_features;
    pub_context.motion_state = &g_state.motion_state;
    pub_context.packets_processed = &g_state.packets_processed;
    pub_context.packets_dropped = &g_state.packets_dropped;
    pub_context.state_mutex = g_state_mutex;
    pub_context.mqtt_topic = MQTT_TOPIC;
    
    // Start MQTT publisher task
    mqtt_start_publisher(&pub_context);
    
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
    
    // Publish system info after boot (to show current configuration)
    ESP_LOGI(TAG, "📡 Publishing system info...");
    cJSON *empty_root = cJSON_CreateObject();
    if (empty_root) {
        // Call cmd_info through mqtt_commands_process with empty command
        // This will publish the info to MQTT
        const char *info_cmd = "{\"cmd\":\"info\"}";
        mqtt_commands_process(info_cmd, strlen(info_cmd), &g_mqtt_cmd_context,
                            &g_state.mqtt_state, MQTT_TOPIC "/response");
        cJSON_Delete(empty_root);
    }
}
