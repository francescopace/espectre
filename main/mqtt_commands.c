/*
 * ESPectre - MQTT Commands Module Implementation
 * 
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * License: GPLv3
 */

#include "mqtt_commands.h"
#include "nvs_storage.h"
#include "traffic_generator.h"
#include "filters.h"
#include "validation.h"
#include "csi_processor.h"
#include "nbvi_calibrator.h"
#include "esp_log.h"
#include "esp_netif.h"
#include "esp_timer.h"
#include "esp_heap_caps.h"
#include "esp_wifi.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "cJSON.h"
#include <string.h>
#include <stdatomic.h>

static const char *TAG = "MQTT_Commands";

// Macro for converting numbers to strings
#define STRINGIFY(x) #x
#define TOSTRING(x) STRINGIFY(x)

// Forward declaration of NBVI calibration function (defined in espectre.c)
extern bool run_nbvi_calibration(void);

// Global context for command handlers
static mqtt_cmd_context_t *g_cmd_context = NULL;
static mqtt_handler_state_t *g_mqtt_state = NULL;
static const char *g_response_topic = NULL;

// Send MQTT response message to configured topic
static void send_response(const char *message) {
    if (g_mqtt_state && g_response_topic) {
        mqtt_send_response(g_mqtt_state, message, g_response_topic);
    }
}

// Extract and validate boolean parameter from JSON command
static bool get_bool_param(cJSON *root, const char *key, bool *out_value) {
    cJSON *item = cJSON_GetObjectItem(root, key);
    if (item && cJSON_IsBool(item)) {
        *out_value = cJSON_IsTrue(item);
        return true;
    }
    send_response("ERROR: Missing or invalid 'enabled' field");
    return false;
}

// Extract float parameter with range validation
static bool get_float_param(cJSON *root, const char *key, float *out_value, 
                           float min_val, float max_val, const char *error_msg) {
    cJSON *item = cJSON_GetObjectItem(root, key);
    if (!item || !cJSON_IsNumber(item)) {
        send_response("ERROR: Missing or invalid 'value' field");
        return false;
    }
    
    float value = (float)item->valuedouble;
    if (!validate_float(value)) {
        send_response("ERROR: Invalid value (NaN or Infinity)");
        return false;
    }
    
    if (value < min_val || value > max_val) {
        send_response(error_msg);
        return false;
    }
    
    *out_value = value;
    return true;
}

// Extract integer parameter with range validation
static bool get_int_param(cJSON *root, const char *key, int *out_value,
                         int min_val, int max_val, const char *error_msg) {
    cJSON *item = cJSON_GetObjectItem(root, key);
    if (!item || !cJSON_IsNumber(item)) {
        send_response("ERROR: Missing or invalid 'value' field");
        return false;
    }
    
    int value = (int)item->valueint;
    if (value < min_val || value > max_val) {
        send_response(error_msg);
        return false;
    }
    
    *out_value = value;
    return true;
}

static void cmd_segmentation_threshold(cJSON *root) {
    float new_threshold;
    if (get_float_param(root, "value", &new_threshold, 0.5f, 10.0f,
                       "ERROR: Segmentation threshold must be between 0.5 and 10.0")) {
        if (csi_processor_set_threshold(g_cmd_context->csi_processor, new_threshold)) {
            char response[256];
            snprintf(response, sizeof(response), 
                     "Motion detection threshold updated: %.2f", new_threshold);
            send_response(response);
            ESP_LOGI(TAG, "%s", response);
            
            // Save to NVS
            esp_err_t err = config_save_to_nvs(g_cmd_context->config, new_threshold);
            if (err == ESP_OK) {
                ESP_LOGI(TAG, "💾 Configuration saved to NVS");
            } else {
                ESP_LOGE(TAG, "❌ Failed to save configuration to NVS: %s", esp_err_to_name(err));
            }
        } else {
            send_response("ERROR: Failed to set threshold");
        }
    }
}

static void cmd_segmentation_window_size(cJSON *root) {
    int new_window_size;
    if (get_int_param(root, "value", &new_window_size, 
                     SEGMENTATION_WINDOW_SIZE_MIN, SEGMENTATION_WINDOW_SIZE_MAX,
                     "ERROR: Window size must be between " 
                     TOSTRING(SEGMENTATION_WINDOW_SIZE_MIN) " and " 
                     TOSTRING(SEGMENTATION_WINDOW_SIZE_MAX) " packets")) {
        if (csi_processor_set_window_size(g_cmd_context->csi_processor, (uint16_t)new_window_size)) {
            g_cmd_context->config->segmentation_window_size = (uint16_t)new_window_size;
            
            char response[256];
            uint32_t rate = g_cmd_context->config->traffic_generator_rate;
            float duration = (rate > 0) ? (float)new_window_size / (float)rate : 0.0f;
            snprintf(response, sizeof(response), 
                     "Window size updated: %d packets (%.2fs @ %uHz, %s)",
                     new_window_size, duration, (unsigned int)rate,
                     new_window_size < 50 ? "more reactive" : "more stable");
            send_response(response);
            ESP_LOGI(TAG, "%s", response);
            
            config_save_to_nvs(g_cmd_context->config, 
                             csi_processor_get_threshold(g_cmd_context->csi_processor));
        } else {
            send_response("ERROR: Failed to set window size");
        }
    }
}

static void cmd_features_enable(cJSON *root) {
    bool enabled;
    if (get_bool_param(root, "enabled", &enabled)) {
        g_cmd_context->config->features_enabled = enabled;
        char response[128];
        snprintf(response, sizeof(response), "Feature extraction %s", 
                 enabled ? "enabled" : "disabled");
        send_response(response);
        ESP_LOGI(TAG, "%s", response);
        
        esp_err_t err = config_save_to_nvs(g_cmd_context->config, 
                                           csi_processor_get_threshold(g_cmd_context->csi_processor));
        if (err == ESP_OK) {
            ESP_LOGI(TAG, "💾 Configuration saved to NVS");
        }
    }
}

static void cmd_info(cJSON *root) {
    cJSON *response = cJSON_CreateObject();
    
    // Network information (extended with WiFi details)
    cJSON *network = cJSON_CreateObject();
    
    // IP address
    esp_netif_t *netif = esp_netif_get_handle_from_ifkey("WIFI_STA_DEF");
    if (netif) {
        esp_netif_ip_info_t ip_info;
        if (esp_netif_get_ip_info(netif, &ip_info) == ESP_OK) {
            char ip_str[16];
            snprintf(ip_str, sizeof(ip_str), IPSTR, IP2STR(&ip_info.ip));
            cJSON_AddStringToObject(network, "ip_address", ip_str);
        } else {
            cJSON_AddStringToObject(network, "ip_address", "not connected");
        }
    } else {
        cJSON_AddStringToObject(network, "ip_address", "not available");
    }
    
    // MAC address
    uint8_t mac[6];
    if (esp_wifi_get_mac(WIFI_IF_STA, mac) == ESP_OK) {
        char mac_str[18];
        snprintf(mac_str, sizeof(mac_str), "%02X:%02X:%02X:%02X:%02X:%02X",
                 mac[0], mac[1], mac[2], mac[3], mac[4], mac[5]);
        cJSON_AddStringToObject(network, "mac_address", mac_str);
    }
    
    // Traffic generator rate
    cJSON_AddNumberToObject(network, "traffic_generator_rate", g_cmd_context->config->traffic_generator_rate);
    
    // WiFi channel
    uint8_t primary = 0;
    wifi_second_chan_t secondary = WIFI_SECOND_CHAN_NONE;
    if (esp_wifi_get_channel(&primary, &secondary) == ESP_OK) {
        cJSON *channel = cJSON_CreateObject();
        cJSON_AddNumberToObject(channel, "primary", primary);
        cJSON_AddNumberToObject(channel, "secondary", secondary);
        cJSON_AddItemToObject(network, "channel", channel);
    }
    
    // WiFi bandwidth
    wifi_bandwidth_t bw;
    if (esp_wifi_get_bandwidth(WIFI_IF_STA, &bw) == ESP_OK) {
        const char *bw_str = (bw == WIFI_BW_HT20) ? "HT20" : "HT40";
        cJSON_AddStringToObject(network, "bandwidth", bw_str);
    }
    
    // WiFi protocol
    uint8_t protocol;
    if (esp_wifi_get_protocol(WIFI_IF_STA, &protocol) == ESP_OK) {
        char protocol_str[32] = "";
        bool first = true;
        
        if (protocol & WIFI_PROTOCOL_11B) {
            strcat(protocol_str, "802.11b");
            first = false;
        }
        if (protocol & WIFI_PROTOCOL_11G) {
            if (!first) strcat(protocol_str, "/");
            strcat(protocol_str, "g");
            first = false;
        }
        if (protocol & WIFI_PROTOCOL_11N) {
            if (!first) strcat(protocol_str, "/");
            strcat(protocol_str, "n");
            first = false;
        }
#if CONFIG_IDF_TARGET_ESP32C6
        if (protocol & WIFI_PROTOCOL_11AX) {
            if (!first) strcat(protocol_str, "/");
            strcat(protocol_str, "ax");
        }
#endif
        cJSON_AddStringToObject(network, "protocol", protocol_str);
    }
    
    // Promiscuous mode
    bool promiscuous = false;
    esp_wifi_get_promiscuous(&promiscuous);
    cJSON_AddBoolToObject(network, "promiscuous_mode", promiscuous);
    
    cJSON_AddItemToObject(response, "network", network);
    
    // Device information
    cJSON *device = cJSON_CreateObject();
#if CONFIG_IDF_TARGET_ESP32C6
    cJSON_AddStringToObject(device, "type", "ESP32-C6");
#elif CONFIG_IDF_TARGET_ESP32S3
    cJSON_AddStringToObject(device, "type", "ESP32-S3");
#elif CONFIG_IDF_TARGET_ESP32
    cJSON_AddStringToObject(device, "type", "ESP32");
#else
    cJSON_AddStringToObject(device, "type", "Unknown");
#endif
    cJSON_AddItemToObject(response, "device", device);
    
    // MQTT topics
    cJSON *mqtt = cJSON_CreateObject();
    if (g_cmd_context->mqtt_base_topic) {
        cJSON_AddStringToObject(mqtt, "base_topic", g_cmd_context->mqtt_base_topic);
    }
    if (g_cmd_context->mqtt_cmd_topic) {
        cJSON_AddStringToObject(mqtt, "cmd_topic", g_cmd_context->mqtt_cmd_topic);
    }
    if (g_cmd_context->mqtt_response_topic) {
        cJSON_AddStringToObject(mqtt, "response_topic", g_cmd_context->mqtt_response_topic);
    }
    cJSON_AddItemToObject(response, "mqtt", mqtt);
    
    // Segmentation parameters (renamed from motion_detection for compatibility with micro-espectre)
    cJSON *segmentation = cJSON_CreateObject();
    cJSON_AddNumberToObject(segmentation, "threshold", (double)csi_processor_get_threshold(g_cmd_context->csi_processor));
    cJSON_AddNumberToObject(segmentation, "window_size", csi_processor_get_window_size(g_cmd_context->csi_processor));
    cJSON_AddItemToObject(response, "segmentation", segmentation);
    
    // Filters configuration
    cJSON *filters = cJSON_CreateObject();
    cJSON_AddBoolToObject(filters, "butterworth_enabled", g_cmd_context->config->butterworth_enabled);
    cJSON_AddBoolToObject(filters, "wavelet_enabled", g_cmd_context->config->wavelet_enabled);
    cJSON_AddNumberToObject(filters, "wavelet_level", g_cmd_context->config->wavelet_level);
    cJSON_AddNumberToObject(filters, "wavelet_threshold", (double)g_cmd_context->config->wavelet_threshold);
    cJSON_AddBoolToObject(filters, "hampel_enabled", g_cmd_context->config->hampel_filter_enabled);
    cJSON_AddNumberToObject(filters, "hampel_threshold", (double)g_cmd_context->config->hampel_threshold);
    cJSON_AddBoolToObject(filters, "savgol_enabled", g_cmd_context->config->savgol_filter_enabled);
    cJSON_AddNumberToObject(filters, "savgol_window_size", g_cmd_context->config->savgol_window_size);
    cJSON_AddItemToObject(response, "filters", filters);
    
    // Options
    cJSON *options = cJSON_CreateObject();
    cJSON_AddBoolToObject(options, "smart_publishing_enabled", g_cmd_context->config->smart_publishing_enabled);
    cJSON_AddBoolToObject(options, "features_enabled", g_cmd_context->config->features_enabled);
    cJSON_AddItemToObject(response, "options", options);
    
    // Subcarrier selection
    cJSON *subcarriers = cJSON_CreateObject();
    cJSON *indices_array = cJSON_CreateArray();
    for (uint8_t i = 0; i < g_cmd_context->config->num_selected_subcarriers; i++) {
        cJSON_AddItemToArray(indices_array, cJSON_CreateNumber(g_cmd_context->config->selected_subcarriers[i]));
    }
    cJSON_AddItemToObject(subcarriers, "indices", indices_array);
    cJSON_AddNumberToObject(subcarriers, "count", g_cmd_context->config->num_selected_subcarriers);
    cJSON_AddItemToObject(response, "subcarriers", subcarriers);
    
    char *json_str = cJSON_PrintUnformatted(response);
    if (json_str) {
        mqtt_send_response(g_mqtt_state, json_str, g_response_topic);
        free(json_str);
    }
    cJSON_Delete(response);
}

// Format uptime as human-readable string (e.g., "3h 24m 15s")
static void format_uptime(char *buffer, size_t size, int64_t uptime_sec) {
    int hours = uptime_sec / 3600;
    int minutes = (uptime_sec % 3600) / 60;
    int seconds = uptime_sec % 60;
    
    if (hours > 0) {
        snprintf(buffer, size, "%dh %dm %ds", hours, minutes, seconds);
    } else if (minutes > 0) {
        snprintf(buffer, size, "%dm %ds", minutes, seconds);
    } else {
        snprintf(buffer, size, "%ds", seconds);
    }
}

static void cmd_stats(cJSON *root) {
    cJSON *response = cJSON_CreateObject();
    
    // Timestamp
    int64_t current_time = esp_timer_get_time() / 1000000;  // Convert to seconds
    cJSON_AddNumberToObject(response, "timestamp", (double)current_time);
    
    // Uptime (human-readable)
    if (g_cmd_context->system_start_time) {
        int64_t uptime_sec = current_time - (*g_cmd_context->system_start_time);
        char uptime_str[32];
        format_uptime(uptime_str, sizeof(uptime_str), uptime_sec);
        cJSON_AddStringToObject(response, "uptime", uptime_str);
    }
    
    // CPU usage percentage
    TaskStatus_t *task_status_array;
    uint32_t total_run_time;
    UBaseType_t task_count = uxTaskGetNumberOfTasks();
    
    task_status_array = pvPortMalloc(task_count * sizeof(TaskStatus_t));
    if (task_status_array != NULL) {
        task_count = uxTaskGetSystemState(task_status_array, task_count, &total_run_time);
        
        // Find IDLE task runtime
        uint32_t idle_run_time = 0;
        for (UBaseType_t i = 0; i < task_count; i++) {
            if (strstr(task_status_array[i].pcTaskName, "IDLE") != NULL) {
                idle_run_time += task_status_array[i].ulRunTimeCounter;
            }
        }
        
        // Calculate CPU usage: 100 - (idle_time / total_time * 100)
        float cpu_usage = 0.0f;
        if (total_run_time > 0) {
            cpu_usage = 100.0f - ((float)idle_run_time / (float)total_run_time * 100.0f);
            if (cpu_usage < 0.0f) cpu_usage = 0.0f;
            if (cpu_usage > 100.0f) cpu_usage = 100.0f;
        }
        
        cJSON_AddNumberToObject(response, "cpu_usage_percent", (double)cpu_usage);
        vPortFree(task_status_array);
    } else {
        // Fallback if memory allocation fails
        cJSON_AddNumberToObject(response, "cpu_usage_percent", 0.0);
    }
    
    // Heap usage percentage
    size_t free_heap = esp_get_free_heap_size();
    size_t total_heap = heap_caps_get_total_size(MALLOC_CAP_DEFAULT);
    float heap_usage = 0.0f;
    if (total_heap > 0) {
        heap_usage = ((float)(total_heap - free_heap) / (float)total_heap) * 100.0f;
    }
    cJSON_AddNumberToObject(response, "heap_usage_percent", (double)heap_usage);
    
    // Current state
    const char *state_names[] = {"idle", "motion"};
    csi_motion_state_t state = csi_processor_get_state(g_cmd_context->csi_processor);
    cJSON_AddStringToObject(response, "state", state_names[state]);
    
    // Current turbulence (last packet)
    float turbulence = csi_processor_get_last_turbulence(g_cmd_context->csi_processor);
    cJSON_AddNumberToObject(response, "turbulence", (double)turbulence);
    
    // Moving variance
    float moving_variance = csi_processor_get_moving_variance(g_cmd_context->csi_processor);
    cJSON_AddNumberToObject(response, "movement", (double)moving_variance);
    
    // Threshold
    float threshold = csi_processor_get_threshold(g_cmd_context->csi_processor);
    cJSON_AddNumberToObject(response, "threshold", (double)threshold);
    
    // Packets processed
    uint32_t packets_processed = csi_processor_get_total_packets(g_cmd_context->csi_processor);
    cJSON_AddNumberToObject(response, "packets_processed", packets_processed);

    // Packets dropped
    uint32_t packets_dropped = atomic_load(g_cmd_context->packets_dropped);
    cJSON_AddNumberToObject(response, "packets_dropped", packets_dropped);
    
    
    char *json_str = cJSON_PrintUnformatted(response);
    if (json_str) {
        mqtt_send_response(g_mqtt_state, json_str, g_response_topic);
        free(json_str);
    }
    cJSON_Delete(response);
}

static void cmd_hampel_filter(cJSON *root) {
    bool enabled;
    if (get_bool_param(root, "enabled", &enabled)) {
        g_cmd_context->config->hampel_filter_enabled = enabled;
        char response[64];
        snprintf(response, sizeof(response), "Hampel filter %s", 
                 enabled ? "enabled" : "disabled");
        send_response(response);
        ESP_LOGI(TAG, "%s", response);
        
        config_save_to_nvs(g_cmd_context->config, csi_processor_get_threshold(g_cmd_context->csi_processor));
    }
}

static void cmd_hampel_threshold(cJSON *root) {
    float new_threshold;
    if (get_float_param(root, "value", &new_threshold, HAMPEL_THRESHOLD_MIN, HAMPEL_THRESHOLD_MAX,
                       "ERROR: Hampel threshold must be between 1.0 and 10.0")) {
        float old_threshold = g_cmd_context->config->hampel_threshold;
        g_cmd_context->config->hampel_threshold = new_threshold;
        
        char response[256];
        snprintf(response, sizeof(response), 
                 "Hampel threshold updated: %.1f -> %.1f", old_threshold, new_threshold);
        send_response(response);
        ESP_LOGI(TAG, "%s", response);
        
        config_save_to_nvs(g_cmd_context->config, csi_processor_get_threshold(g_cmd_context->csi_processor));
    }
}

static void cmd_savgol_filter(cJSON *root) {
    bool enabled;
    if (get_bool_param(root, "enabled", &enabled)) {
        g_cmd_context->config->savgol_filter_enabled = enabled;
        char response[64];
        snprintf(response, sizeof(response), "Savitzky-Golay filter %s", 
                 enabled ? "enabled" : "disabled");
        send_response(response);
        ESP_LOGI(TAG, "%s", response);
        
        config_save_to_nvs(g_cmd_context->config, csi_processor_get_threshold(g_cmd_context->csi_processor));
    }
}

static void cmd_butterworth_filter(cJSON *root) {
    bool enabled;
    if (get_bool_param(root, "enabled", &enabled)) {
        g_cmd_context->config->butterworth_enabled = enabled;
        char response[64];
        snprintf(response, sizeof(response), "Butterworth filter %s", 
                 enabled ? "enabled" : "disabled");
        send_response(response);
        ESP_LOGI(TAG, "%s", response);
        
        config_save_to_nvs(g_cmd_context->config, csi_processor_get_threshold(g_cmd_context->csi_processor));
    }
}

static void cmd_smart_publishing(cJSON *root) {
    bool enabled;
    if (get_bool_param(root, "enabled", &enabled)) {
        g_cmd_context->config->smart_publishing_enabled = enabled;
        char response[128];
        snprintf(response, sizeof(response), "Smart publishing %s", 
                 enabled ? "enabled" : "disabled");
        send_response(response);
        ESP_LOGI(TAG, "%s", response);
        
        config_save_to_nvs(g_cmd_context->config, csi_processor_get_threshold(g_cmd_context->csi_processor));
    }
}


static void cmd_traffic_generator_rate(cJSON *root) {
    cJSON *value = cJSON_GetObjectItem(root, "value");
    if (value && cJSON_IsNumber(value)) {
        uint32_t new_rate = (uint32_t)value->valueint;
        if (new_rate <= TRAFFIC_RATE_MAX) {
            uint32_t old_rate = g_cmd_context->config->traffic_generator_rate;
            g_cmd_context->config->traffic_generator_rate = new_rate;
            
            // Apply immediately
            if (new_rate == 0) {
                // Disable
                traffic_generator_stop();
                ESP_LOGI(TAG, "📡 Traffic generator disabled");
            } else if (old_rate == 0) {
                // Was disabled, start now
                if (traffic_generator_start(new_rate)) {
                    ESP_LOGI(TAG, "📡 Traffic generator started (%u pps)", (unsigned int)new_rate);
                } else {
                    ESP_LOGE(TAG, "Failed to start traffic generator");
                }
            } else {
                // Change rate
                traffic_generator_set_rate(new_rate);
            }
            
            char response[256];
            if (new_rate == 0) {
                snprintf(response, sizeof(response), 
                         "Traffic generator disabled (was %u pps)", (unsigned int)old_rate);
            } else if (old_rate == 0) {
                snprintf(response, sizeof(response), 
                         "Traffic generator enabled (%u pps)", (unsigned int)new_rate);
            } else {
                snprintf(response, sizeof(response), 
                         "Traffic rate updated: %u -> %u pps", (unsigned int)old_rate, (unsigned int)new_rate);
            }
            send_response(response);
            
            config_save_to_nvs(g_cmd_context->config, csi_processor_get_threshold(g_cmd_context->csi_processor));
        } else {
            char error_msg[128];
            snprintf(error_msg, sizeof(error_msg), 
                     "ERROR: Rate must be 0-%u packets/sec (0=disabled, 100 recommended)", TRAFFIC_RATE_MAX);
            send_response(error_msg);
        }
    } else {
        send_response("ERROR: Missing or invalid 'value' field");
    }
}

static void cmd_wavelet_filter(cJSON *root) {
    bool enabled;
    if (get_bool_param(root, "enabled", &enabled)) {
        g_cmd_context->config->wavelet_enabled = enabled;
        char response[64];
        snprintf(response, sizeof(response), "Wavelet filter %s", 
                 enabled ? "enabled" : "disabled");
        send_response(response);
        ESP_LOGI(TAG, "%s", response);
        
        config_save_to_nvs(g_cmd_context->config, csi_processor_get_threshold(g_cmd_context->csi_processor));
    }
}

static void cmd_wavelet_level(cJSON *root) {
    int new_level;
    if (get_int_param(root, "value", &new_level, WAVELET_LEVEL_MIN, WAVELET_LEVEL_MAX,
                     "ERROR: Wavelet level must be between 1 and 3 (recommended: 3)")) {
        int old_level = g_cmd_context->config->wavelet_level;
        g_cmd_context->config->wavelet_level = new_level;
        
        char response[256];
        snprintf(response, sizeof(response), 
                 "Wavelet decomposition level updated: %d -> %d (denoising %s)", 
                 old_level, new_level,
                 new_level > old_level ? "more aggressive" : "less aggressive");
        send_response(response);
        ESP_LOGI(TAG, "%s", response);
        
        config_save_to_nvs(g_cmd_context->config, csi_processor_get_threshold(g_cmd_context->csi_processor));
    }
}

static void cmd_wavelet_threshold(cJSON *root) {
    float new_threshold;
    if (get_float_param(root, "value", &new_threshold, WAVELET_THRESHOLD_MIN, WAVELET_THRESHOLD_MAX,
                       "ERROR: Wavelet threshold must be between 0.5 and 2.0 (recommended: 1.0)")) {
        float old_threshold = g_cmd_context->config->wavelet_threshold;
        g_cmd_context->config->wavelet_threshold = new_threshold;
        
        char response[256];
        snprintf(response, sizeof(response), 
                 "Wavelet threshold updated: %.2f -> %.2f (noise removal %s)", 
                 old_threshold, new_threshold,
                 new_threshold > old_threshold ? "more aggressive" : "less aggressive");
        send_response(response);
        ESP_LOGI(TAG, "%s", response);
        
        config_save_to_nvs(g_cmd_context->config, csi_processor_get_threshold(g_cmd_context->csi_processor));
    }
}

static void cmd_subcarrier_selection(cJSON *root) {
    cJSON *indices = cJSON_GetObjectItem(root, "indices");
    if (!indices || !cJSON_IsArray(indices)) {
        send_response("ERROR: Missing or invalid 'indices' array");
        return;
    }
    
    int array_size = cJSON_GetArraySize(indices);
    if (array_size < 1 || array_size > 64) {
        send_response("ERROR: Number of subcarriers must be between 1 and 64");
        return;
    }
    
    // Temporary array to validate indices
    uint8_t temp_subcarriers[64];
    
    // Extract and validate indices
    for (int i = 0; i < array_size; i++) {
        cJSON *item = cJSON_GetArrayItem(indices, i);
        if (!cJSON_IsNumber(item)) {
            send_response("ERROR: All indices must be numbers");
            return;
        }
        
        int value = item->valueint;
        if (value < 0 || value > 63) {
            char error_msg[128];
            snprintf(error_msg, sizeof(error_msg), 
                     "ERROR: Subcarrier index %d out of range (must be 0-63)", value);
            send_response(error_msg);
            return;
        }
        
        temp_subcarriers[i] = (uint8_t)value;
    }
    
    // Update configuration
    memcpy(g_cmd_context->config->selected_subcarriers, temp_subcarriers, 
           array_size * sizeof(uint8_t));
    g_cmd_context->config->num_selected_subcarriers = (uint8_t)array_size;
    
    // Update CSI processor
    csi_set_subcarrier_selection(temp_subcarriers, (uint8_t)array_size);
    
    // Save to NVS
    esp_err_t err = config_save_to_nvs(g_cmd_context->config, 
                                       csi_processor_get_threshold(g_cmd_context->csi_processor));
    if (err == ESP_OK) {
        ESP_LOGI(TAG, "💾 Subcarrier selection saved to NVS");
    } else {
        ESP_LOGE(TAG, "❌ Failed to save subcarrier selection to NVS: %s", esp_err_to_name(err));
    }
    
    char response[256];
    snprintf(response, sizeof(response), 
             "Subcarrier selection updated: %d subcarriers", array_size);
    send_response(response);
    ESP_LOGI(TAG, "%s", response);
}

static void cmd_factory_reset(cJSON *root) {
    ESP_LOGW(TAG, "⚠️  Factory reset requested");
    
    nvs_factory_reset();
    
    // Reset config to defaults
    config_init_defaults(g_cmd_context->config);
    
    // Reset CSI processor to defaults
    csi_processor_init(g_cmd_context->csi_processor);
    ESP_LOGI(TAG, "📍 CSI processor reset to defaults (threshold: %.2f)", SEGMENTATION_DEFAULT_THRESHOLD);
    
    // Reset subcarrier selection to defaults
    csi_set_subcarrier_selection(g_cmd_context->config->selected_subcarriers,
                                 g_cmd_context->config->num_selected_subcarriers);
    ESP_LOGI(TAG, "📡 Subcarrier selection reset to defaults (%d subcarriers)", 
             g_cmd_context->config->num_selected_subcarriers);
    
    // Reset traffic generator to default rate
    if (DEFAULT_TRAFFIC_GENERATOR_RATE > 0) {
        if (traffic_generator_is_running()) {
            // Just change the rate
            traffic_generator_set_rate(DEFAULT_TRAFFIC_GENERATOR_RATE);
            ESP_LOGI(TAG, "📡 Traffic generator rate reset to default (%d pps)", 
                     DEFAULT_TRAFFIC_GENERATOR_RATE);
        } else {
            // Start with default rate
            if (traffic_generator_start(DEFAULT_TRAFFIC_GENERATOR_RATE)) {
                ESP_LOGI(TAG, "📡 Traffic generator started at default rate (%d pps)", 
                         DEFAULT_TRAFFIC_GENERATOR_RATE);
            } else {
                ESP_LOGW(TAG, "⚠️  Failed to start traffic generator");
            }
        }
    } else if (traffic_generator_is_running()) {
        // Default rate is 0, stop traffic generator
        traffic_generator_stop();
        ESP_LOGI(TAG, "📡 Traffic generator stopped (default rate is 0)");
    }
    
    send_response("Factory reset complete");
    ESP_LOGI(TAG, "✅ Factory reset complete");
    
    // Run NBVI calibration if enabled
    if (NBVI_ENABLED) {
        send_response("Starting NBVI re-calibration...");
        ESP_LOGI(TAG, "🧬 Starting NBVI re-calibration...");
        
        bool success = run_nbvi_calibration();
        
        if (success) {
            send_response("NBVI re-calibration successful!");
            ESP_LOGI(TAG, "✅ NBVI re-calibration successful");
        } else {
            send_response("NBVI re-calibration failed. Using default band.");
            ESP_LOGW(TAG, "⚠️  NBVI re-calibration failed");
        }
    }
    
    // Publish info to show reset configuration
    ESP_LOGI(TAG, "📡 Publishing system info after factory reset...");
    cmd_info(root);
}

// Command dispatch table
typedef void (*cmd_handler_t)(cJSON *root);

typedef struct {
    const char *name;
    cmd_handler_t handler;
} command_entry_t;

static const command_entry_t command_table[] = {
    {"segmentation_threshold", cmd_segmentation_threshold},
    {"segmentation_window_size", cmd_segmentation_window_size},
    {"subcarrier_selection", cmd_subcarrier_selection},
    {"features_enable", cmd_features_enable},
    {"info", cmd_info},
    {"stats", cmd_stats},
    {"hampel_filter", cmd_hampel_filter},
    {"hampel_threshold", cmd_hampel_threshold},
    {"savgol_filter", cmd_savgol_filter},
    {"butterworth_filter", cmd_butterworth_filter},
    {"wavelet_filter", cmd_wavelet_filter},
    {"wavelet_level", cmd_wavelet_level},
    {"wavelet_threshold", cmd_wavelet_threshold},
    {"smart_publishing", cmd_smart_publishing},
    {"traffic_generator_rate", cmd_traffic_generator_rate},
    {"factory_reset", cmd_factory_reset},
    {NULL, NULL}
};

// Public API implementation

int mqtt_commands_init(mqtt_handler_state_t *mqtt_state, mqtt_cmd_context_t *context) {
    if (!mqtt_state || !context) {
        ESP_LOGE(TAG, "mqtt_commands_init: NULL pointer");
        return -1;
    }
    
    g_mqtt_state = mqtt_state;
    g_cmd_context = context;
    
    ESP_LOGI(TAG, "MQTT commands initialized");
    return 0;
}

int mqtt_commands_process(const char *data, int data_len, 
                         mqtt_cmd_context_t *context,
                         mqtt_handler_state_t *mqtt_state,
                         const char *response_topic) {
    if (!data || !context || !mqtt_state || !response_topic) {
        ESP_LOGE(TAG, "mqtt_commands_process: NULL pointer");
        return -1;
    }
    
    // Set global context for command handlers
    g_cmd_context = context;
    g_mqtt_state = mqtt_state;
    g_response_topic = response_topic;
    
    cJSON *root = cJSON_ParseWithLength(data, data_len);
    if (!root) {
        ESP_LOGW(TAG, "Failed to parse MQTT command JSON");
        send_response("ERROR: Invalid JSON");
        return -1;
    }
    
    cJSON *cmd = cJSON_GetObjectItem(root, "cmd");
    if (!cmd || !cJSON_IsString(cmd)) {
        cJSON_Delete(root);
        send_response("ERROR: Missing 'cmd' field");
        return -1;
    }
    
    const char *command = cmd->valuestring;
    ESP_LOGD(TAG, "Processing MQTT command: %s", command);
    
    // Look up command in dispatch table
    bool command_found = false;
    for (int i = 0; command_table[i].name != NULL; i++) {
        if (strcmp(command, command_table[i].name) == 0) {
            command_table[i].handler(root);
            command_found = true;
            break;
        }
    }
    
    if (!command_found) {
        char response[128];
        snprintf(response, sizeof(response), "ERROR: Unknown command '%s'", command);
        send_response(response);
    }
    
    cJSON_Delete(root);
    return command_found ? 0 : -1;
}
