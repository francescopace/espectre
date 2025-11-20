/*
 * ESPectre - MQTT Commands Module Implementation
 * 
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * License: GPLv3
 */

#include "mqtt_commands.h"
#include "nvs_storage.h"
#include "traffic_generator.h"
#include "esp_log.h"
#include "esp_netif.h"
#include "esp_timer.h"
#include "cJSON.h"
#include <string.h>
#include <math.h>

static const char *TAG = "MQTT_Commands";

// Parameter validation constants
#define HAMPEL_THRESHOLD_MIN    1.0f
#define HAMPEL_THRESHOLD_MAX    10.0f
#define ALPHA_MIN               0.001f
#define ALPHA_MAX               0.1f
#define WAVELET_THRESHOLD_MIN   0.5f
#define WAVELET_THRESHOLD_MAX   2.0f
#define WAVELET_LEVEL_MIN       1
#define WAVELET_LEVEL_MAX       3
#define RESET_TIMEOUT_MAX       300
#define TRAFFIC_RATE_MAX        50

// Global context for command handlers
static mqtt_cmd_context_t *g_cmd_context = NULL;
static mqtt_handler_state_t *g_mqtt_state = NULL;
static const char *g_response_topic = NULL;

// Check for NaN and Infinity to prevent calculation errors
static bool is_valid_float(float value) {
    return !isnan(value) && !isinf(value);
}

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
    if (!is_valid_float(value)) {
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
        float old_threshold = g_cmd_context->segmentation->adaptive_threshold;
        g_cmd_context->segmentation->adaptive_threshold = new_threshold;
        
        char response[256];
        snprintf(response, sizeof(response), 
                 "Segmentation threshold updated: %.2f -> %.2f", old_threshold, new_threshold);
        send_response(response);
        ESP_LOGI(TAG, "%s", response);
        
        // Save to NVS
        esp_err_t err = config_save_to_nvs(g_cmd_context->config, new_threshold);
        if (err == ESP_OK) {
            ESP_LOGI(TAG, "💾 Configuration saved to NVS");
        } else {
            ESP_LOGE(TAG, "❌ Failed to save configuration to NVS: %s", esp_err_to_name(err));
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
                                           g_cmd_context->segmentation->adaptive_threshold);
        if (err == ESP_OK) {
            ESP_LOGI(TAG, "💾 Configuration saved to NVS");
        }
    }
}

static void cmd_info(cJSON *root) {
    cJSON *response = cJSON_CreateObject();
    
    // Network information
    cJSON *network = cJSON_CreateObject();
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
    cJSON_AddNumberToObject(network, "traffic_generator_rate", g_cmd_context->config->traffic_generator_rate);
    cJSON_AddItemToObject(response, "network", network);
    
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
    
    // Segmentation parameters (configuration only - runtime metrics moved to stats command)
    cJSON *segmentation = cJSON_CreateObject();
    cJSON_AddNumberToObject(segmentation, "threshold", (double)g_cmd_context->segmentation->adaptive_threshold);
    cJSON_AddNumberToObject(segmentation, "window_size", SEGMENTATION_WINDOW_SIZE);
    cJSON_AddNumberToObject(segmentation, "k_factor", (double)SEGMENTATION_K_FACTOR);
    cJSON_AddNumberToObject(segmentation, "min_length", SEGMENTATION_MIN_LENGTH);
    cJSON_AddNumberToObject(segmentation, "max_length", SEGMENTATION_MAX_LENGTH);
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
    
    // Current state
    const char *state_names[] = {"idle", "motion"};
    segmentation_state_t state = segmentation_get_state(g_cmd_context->segmentation);
    cJSON_AddStringToObject(response, "state", state_names[state]);
    
    // Current turbulence (last packet)
    float turbulence = segmentation_get_last_turbulence(g_cmd_context->segmentation);
    cJSON_AddNumberToObject(response, "turbulence", (double)turbulence);
    
    // Moving variance
    float moving_variance = segmentation_get_moving_variance(g_cmd_context->segmentation);
    cJSON_AddNumberToObject(response, "movement", (double)moving_variance);
    
    // Adaptive threshold
    float adaptive_threshold = segmentation_get_threshold(g_cmd_context->segmentation);
    cJSON_AddNumberToObject(response, "threshold", (double)adaptive_threshold);
    
    // Packets processed
    uint32_t packets_processed = segmentation_get_total_packets(g_cmd_context->segmentation);
    cJSON_AddNumberToObject(response, "packets_processed", packets_processed);
    
    
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
        
        config_save_to_nvs(g_cmd_context->config, g_cmd_context->segmentation->adaptive_threshold);
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
        
        config_save_to_nvs(g_cmd_context->config, g_cmd_context->segmentation->adaptive_threshold);
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
        
        config_save_to_nvs(g_cmd_context->config, g_cmd_context->segmentation->adaptive_threshold);
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
        
        config_save_to_nvs(g_cmd_context->config, g_cmd_context->segmentation->adaptive_threshold);
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
        
        config_save_to_nvs(g_cmd_context->config, g_cmd_context->segmentation->adaptive_threshold);
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
            
            config_save_to_nvs(g_cmd_context->config, g_cmd_context->segmentation->adaptive_threshold);
        } else {
            send_response("ERROR: Rate must be 0-50 packets/sec (0=disabled, recommended: 15)");
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
        
        config_save_to_nvs(g_cmd_context->config, g_cmd_context->segmentation->adaptive_threshold);
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
        
        config_save_to_nvs(g_cmd_context->config, g_cmd_context->segmentation->adaptive_threshold);
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
        
        config_save_to_nvs(g_cmd_context->config, g_cmd_context->segmentation->adaptive_threshold);
    }
}

static void cmd_factory_reset(cJSON *root) {
    ESP_LOGW(TAG, "⚠️  Factory reset requested");
    
    nvs_factory_reset();
    
    // Reset config to defaults
    config_init_defaults(g_cmd_context->config);
    
    // Reset segmentation to defaults
    segmentation_init(g_cmd_context->segmentation);
    ESP_LOGI(TAG, "📍 Segmentation reset to defaults (threshold: %.2f)", SEGMENTATION_DEFAULT_THRESHOLD);
    
    send_response("✅ Factory reset complete - all settings restored to defaults");
    ESP_LOGI(TAG, "✅ Factory reset complete");
}

// Command dispatch table
typedef void (*cmd_handler_t)(cJSON *root);

typedef struct {
    const char *name;
    cmd_handler_t handler;
} command_entry_t;

static const command_entry_t command_table[] = {
    {"segmentation_threshold", cmd_segmentation_threshold},
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
