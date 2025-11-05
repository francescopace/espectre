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
#include "cJSON.h"
#include <string.h>
#include <math.h>

static const char *TAG = "MQTT_Commands";

// Global context for command handlers
static mqtt_cmd_context_t *g_cmd_context = NULL;
static mqtt_handler_state_t *g_mqtt_state = NULL;
static const char *g_response_topic = NULL;

// Helper function to validate float values
static bool is_valid_float(float value) {
    return !isnan(value) && !isinf(value);
}

// Helper to send response
static void send_response(const char *message) {
    if (g_mqtt_state && g_response_topic) {
        mqtt_send_response(g_mqtt_state, message, g_response_topic);
    }
}

// Command: threshold
static void cmd_threshold(cJSON *root) {
    cJSON *value = cJSON_GetObjectItem(root, "value");
    if (value && cJSON_IsNumber(value)) {
        float new_threshold = (float)value->valuedouble;
        if (!is_valid_float(new_threshold)) {
            send_response("ERROR: Invalid value (NaN or Infinity)");
            return;
        }
        if (new_threshold > 0.0f && new_threshold < 1.0f) {
            float old_threshold = *g_cmd_context->threshold_high;
            *g_cmd_context->threshold_high = new_threshold;
            *g_cmd_context->threshold_low = new_threshold * g_cmd_context->config->hysteresis_ratio;
            
            char response[256];
            snprintf(response, sizeof(response), 
                     "Threshold updated: %.4f -> %.4f", old_threshold, new_threshold);
            send_response(response);
            ESP_LOGI(TAG, "%s", response);
            
            esp_err_t err = config_save_to_nvs(g_cmd_context->config, *g_cmd_context->threshold_high, 
                             *g_cmd_context->threshold_low);
            if (err == ESP_OK) {
                ESP_LOGI(TAG, "💾 Configuration saved to NVS");
            } else {
                ESP_LOGE(TAG, "❌ Failed to save configuration to NVS: %s", esp_err_to_name(err));
            }
        } else {
            send_response("ERROR: Threshold must be between 0.0 and 1.0");
        }
    } else {
        send_response("ERROR: Missing or invalid 'value' field");
    }
}

// Command: stats
static void cmd_stats(cJSON *root) {
    stats_result_t result;
    stats_buffer_analyze(g_cmd_context->stats_buffer, &result);
    
    cJSON *response = cJSON_CreateObject();
    cJSON_AddNumberToObject(response, "min", (double)result.min);
    cJSON_AddNumberToObject(response, "max", (double)result.max);
    cJSON_AddNumberToObject(response, "avg", (double)result.mean);
    cJSON_AddNumberToObject(response, "stddev", (double)result.stddev);
    cJSON_AddNumberToObject(response, "threshold", (double)*g_cmd_context->threshold_high);
    cJSON_AddNumberToObject(response, "samples", (double)result.count);
    
    char *json_str = cJSON_PrintUnformatted(response);
    if (json_str) {
        mqtt_send_response(g_mqtt_state, json_str, g_response_topic);
        free(json_str);
    }
    cJSON_Delete(response);
}

// Command: info
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
    
    // Detection parameters
    cJSON *detection = cJSON_CreateObject();
    cJSON_AddNumberToObject(detection, "threshold", (double)*g_cmd_context->threshold_high);
    cJSON_AddNumberToObject(detection, "debounce", g_cmd_context->config->debounce_count);
    cJSON_AddNumberToObject(detection, "persistence_timeout", g_cmd_context->config->persistence_timeout);
    cJSON_AddNumberToObject(detection, "hysteresis_ratio", (double)g_cmd_context->config->hysteresis_ratio);
    cJSON_AddNumberToObject(detection, "variance_scale", (double)g_cmd_context->config->variance_scale);
    cJSON_AddItemToObject(response, "detection", detection);
    
    // Traffic generator
    cJSON_AddNumberToObject(response, "traffic_generator_rate", g_cmd_context->config->traffic_generator_rate);
    
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
    cJSON_AddBoolToObject(filters, "adaptive_normalizer_enabled", g_cmd_context->config->adaptive_normalizer_enabled);
    cJSON_AddNumberToObject(filters, "adaptive_normalizer_alpha", (double)g_cmd_context->config->adaptive_normalizer_alpha);
    cJSON_AddNumberToObject(filters, "adaptive_normalizer_reset_timeout_sec", g_cmd_context->config->adaptive_normalizer_reset_timeout_sec);
    cJSON_AddItemToObject(response, "filters", filters);
    
    // Features/capabilities
    cJSON *features = cJSON_CreateObject();
    cJSON_AddBoolToObject(features, "csi_logs_enabled", g_cmd_context->config->csi_logs_enabled);
    cJSON_AddBoolToObject(features, "smart_publishing_enabled", g_cmd_context->config->smart_publishing_enabled);
    cJSON_AddItemToObject(response, "features", features);
    
    char *json_str = cJSON_PrintUnformatted(response);
    if (json_str) {
        mqtt_send_response(g_mqtt_state, json_str, g_response_topic);
        free(json_str);
    }
    cJSON_Delete(response);
}

// Command: logs
static void cmd_logs(cJSON *root) {
    cJSON *enabled = cJSON_GetObjectItem(root, "enabled");
    if (enabled && cJSON_IsBool(enabled)) {
        g_cmd_context->config->csi_logs_enabled = cJSON_IsTrue(enabled);
        char response[64];
        snprintf(response, sizeof(response), "CSI logs %s", 
                 g_cmd_context->config->csi_logs_enabled ? "enabled" : "disabled");
        send_response(response);
        ESP_LOGI(TAG, "%s", response);
    } else {
        send_response("ERROR: Missing or invalid 'enabled' field");
    }
}

// Command: analyze
static void cmd_analyze(cJSON *root) {
    if (g_cmd_context->stats_buffer->count < 50) {
        char response[128];
        snprintf(response, sizeof(response), 
                 "ERROR: Need at least 50 samples (have %lu)", 
                 (unsigned long)g_cmd_context->stats_buffer->count);
        send_response(response);
        return;
    }
    
    stats_result_t result;
    stats_buffer_analyze(g_cmd_context->stats_buffer, &result);
    
    float p25 = stats_buffer_percentile(g_cmd_context->stats_buffer, 25.0f);
    float p50 = stats_buffer_percentile(g_cmd_context->stats_buffer, 50.0f);
    float p75 = stats_buffer_percentile(g_cmd_context->stats_buffer, 75.0f);
    float p95 = stats_buffer_percentile(g_cmd_context->stats_buffer, 95.0f);
    
    float recommended = (p50 + p75) / 2.0f;
    
    cJSON *response = cJSON_CreateObject();
    cJSON_AddNumberToObject(response, "min", (double)result.min);
    cJSON_AddNumberToObject(response, "max", (double)result.max);
    cJSON_AddNumberToObject(response, "avg", (double)result.mean);
    cJSON_AddNumberToObject(response, "stddev", (double)result.stddev);
    cJSON_AddNumberToObject(response, "p25", (double)p25);
    cJSON_AddNumberToObject(response, "p50_median", (double)p50);
    cJSON_AddNumberToObject(response, "p75", (double)p75);
    cJSON_AddNumberToObject(response, "p95", (double)p95);
    cJSON_AddNumberToObject(response, "recommended_threshold", (double)recommended);
    cJSON_AddNumberToObject(response, "current_threshold", (double)*g_cmd_context->threshold_high);
    
    char *json_str = cJSON_PrintUnformatted(response);
    if (json_str) {
        mqtt_send_response(g_mqtt_state, json_str, g_response_topic);
        free(json_str);
    }
    cJSON_Delete(response);
}

// Command: persistence
static void cmd_persistence(cJSON *root) {
    cJSON *value = cJSON_GetObjectItem(root, "value");
    if (value && cJSON_IsNumber(value)) {
        int new_timeout = (int)value->valueint;
        if (new_timeout >= 1 && new_timeout <= 30) {
            int old_timeout = g_cmd_context->config->persistence_timeout;
            g_cmd_context->config->persistence_timeout = new_timeout;
            
            char response[128];
            snprintf(response, sizeof(response), 
                     "Persistence timeout updated: %d -> %d seconds", old_timeout, new_timeout);
            send_response(response);
            ESP_LOGI(TAG, "%s", response);
            
            config_save_to_nvs(g_cmd_context->config, *g_cmd_context->threshold_high, 
                             *g_cmd_context->threshold_low);
        } else {
            send_response("ERROR: Persistence timeout must be between 1 and 30 seconds");
        }
    } else {
        send_response("ERROR: Missing or invalid 'value' field");
    }
}

// Command: debounce
static void cmd_debounce(cJSON *root) {
    cJSON *value = cJSON_GetObjectItem(root, "value");
    if (value && cJSON_IsNumber(value)) {
        int new_debounce = (int)value->valueint;
        if (new_debounce >= 1 && new_debounce <= 10) {
            uint8_t old_debounce = g_cmd_context->config->debounce_count;
            g_cmd_context->config->debounce_count = (uint8_t)new_debounce;
            
            char response[128];
            snprintf(response, sizeof(response), 
                     "Debounce count updated: %d -> %d", old_debounce, new_debounce);
            send_response(response);
            ESP_LOGI(TAG, "%s", response);
            
            config_save_to_nvs(g_cmd_context->config, *g_cmd_context->threshold_high, 
                             *g_cmd_context->threshold_low);
        } else {
            send_response("ERROR: Debounce count must be between 1 and 10");
        }
    } else {
        send_response("ERROR: Missing or invalid 'value' field");
    }
}

// Command: hysteresis
static void cmd_hysteresis(cJSON *root) {
    cJSON *value = cJSON_GetObjectItem(root, "value");
    if (value && cJSON_IsNumber(value)) {
        float new_ratio = (float)value->valuedouble;
        if (!is_valid_float(new_ratio)) {
            send_response("ERROR: Invalid value (NaN or Infinity)");
            return;
        }
        if (new_ratio >= 0.1f && new_ratio <= 1.0f) {
            float old_ratio = g_cmd_context->config->hysteresis_ratio;
            g_cmd_context->config->hysteresis_ratio = new_ratio;
            *g_cmd_context->threshold_low = *g_cmd_context->threshold_high * new_ratio;
            
            char response[256];
            snprintf(response, sizeof(response), 
                     "Hysteresis ratio updated: %.2f -> %.2f", old_ratio, new_ratio);
            send_response(response);
            ESP_LOGI(TAG, "%s", response);
            
            config_save_to_nvs(g_cmd_context->config, *g_cmd_context->threshold_high, 
                             *g_cmd_context->threshold_low);
        } else {
            send_response("ERROR: Hysteresis ratio must be between 0.1 and 1.0");
        }
    } else {
        send_response("ERROR: Missing or invalid 'value' field");
    }
}

// Command: variance_scale
static void cmd_variance_scale(cJSON *root) {
    cJSON *value = cJSON_GetObjectItem(root, "value");
    if (value && cJSON_IsNumber(value)) {
        float new_scale = (float)value->valuedouble;
        if (!is_valid_float(new_scale)) {
            send_response("ERROR: Invalid value (NaN or Infinity)");
            return;
        }
        if (new_scale >= 100.0f && new_scale <= 2000.0f) {
            float old_scale = g_cmd_context->config->variance_scale;
            g_cmd_context->config->variance_scale = new_scale;
            
            char response[256];
            snprintf(response, sizeof(response), 
                     "Variance scale updated: %.0f -> %.0f (sensitivity %s)", 
                     old_scale, new_scale,
                     new_scale < old_scale ? "increased" : "decreased");
            send_response(response);
            ESP_LOGI(TAG, "%s", response);
            
            config_save_to_nvs(g_cmd_context->config, *g_cmd_context->threshold_high, 
                             *g_cmd_context->threshold_low);
        } else {
            send_response("ERROR: Variance scale must be between 100 and 2000");
        }
    } else {
        send_response("ERROR: Missing or invalid 'value' field");
    }
}

// Helper function to check if a feature is selected and get its weight
static bool is_feature_selected(uint8_t feat_idx, uint8_t num_selected, 
                                const uint8_t *selected_features, 
                                const float *weights, float *out_weight) {
    if (num_selected == 0) return false;
    for (uint8_t i = 0; i < num_selected; i++) {
        if (selected_features[i] == feat_idx) {
            if (out_weight) *out_weight = weights[i];
            return true;
        }
    }
    return false;
}

// Helper to add a feature to the response with optional calibration info
static void add_feature_to_response(cJSON *response, const char *name, uint8_t feat_idx, 
                                    double value, uint8_t num_selected,
                                    const uint8_t *selected_features, const float *weights) {
    cJSON *feat_obj = cJSON_CreateObject();
    cJSON_AddNumberToObject(feat_obj, "value", value);
    
    float weight;
    if (is_feature_selected(feat_idx, num_selected, selected_features, weights, &weight)) {
        cJSON_AddBoolToObject(feat_obj, "selected", true);
        cJSON_AddNumberToObject(feat_obj, "weight", (double)weight);
    }
    
    cJSON_AddItemToObject(response, name, feat_obj);
}

// Command: features
static void cmd_features(cJSON *root) {
    cJSON *response = cJSON_CreateObject();
    
    // Get calibration info
    uint8_t num_selected = calibration_get_num_selected();
    const uint8_t *selected_features = NULL;
    const float *weights = NULL;
    
    if (num_selected > 0) {
        selected_features = calibration_get_selected_features();
        weights = calibration_get_weights();
    }
    
    // Add all features with flat naming
    add_feature_to_response(response, "time_domain_variance", 0, 
                           (double)g_cmd_context->current_features->variance,
                           num_selected, selected_features, weights);
    add_feature_to_response(response, "time_domain_skewness", 1, 
                           (double)g_cmd_context->current_features->skewness,
                           num_selected, selected_features, weights);
    add_feature_to_response(response, "time_domain_kurtosis", 2, 
                           (double)g_cmd_context->current_features->kurtosis,
                           num_selected, selected_features, weights);
    add_feature_to_response(response, "time_domain_entropy", 3, 
                           (double)g_cmd_context->current_features->entropy,
                           num_selected, selected_features, weights);
    add_feature_to_response(response, "time_domain_iqr", 4, 
                           (double)g_cmd_context->current_features->iqr,
                           num_selected, selected_features, weights);
    
    add_feature_to_response(response, "spatial_variance", 5, 
                           (double)g_cmd_context->current_features->spatial_variance,
                           num_selected, selected_features, weights);
    add_feature_to_response(response, "spatial_correlation", 6, 
                           (double)g_cmd_context->current_features->spatial_correlation,
                           num_selected, selected_features, weights);
    add_feature_to_response(response, "spatial_gradient", 7, 
                           (double)g_cmd_context->current_features->spatial_gradient,
                           num_selected, selected_features, weights);
    
    char *json_str = cJSON_PrintUnformatted(response);
    if (json_str) {
        mqtt_send_response(g_mqtt_state, json_str, g_response_topic);
        free(json_str);
    }
    cJSON_Delete(response);
}

// Command: hampel_filter
static void cmd_hampel_filter(cJSON *root) {
    cJSON *enabled = cJSON_GetObjectItem(root, "enabled");
    if (enabled && cJSON_IsBool(enabled)) {
        g_cmd_context->config->hampel_filter_enabled = cJSON_IsTrue(enabled);
        char response[64];
        snprintf(response, sizeof(response), "Hampel filter %s", 
                 g_cmd_context->config->hampel_filter_enabled ? "enabled" : "disabled");
        send_response(response);
        ESP_LOGI(TAG, "%s", response);
        
        config_save_to_nvs(g_cmd_context->config, *g_cmd_context->threshold_high, 
                         *g_cmd_context->threshold_low);
    } else {
        send_response("ERROR: Missing or invalid 'enabled' field");
    }
}

// Command: hampel_threshold
static void cmd_hampel_threshold(cJSON *root) {
    cJSON *value = cJSON_GetObjectItem(root, "value");
    if (value && cJSON_IsNumber(value)) {
        float new_threshold = (float)value->valuedouble;
        if (!is_valid_float(new_threshold)) {
            send_response("ERROR: Invalid value (NaN or Infinity)");
            return;
        }
        if (new_threshold >= 1.0f && new_threshold <= 10.0f) {
            float old_threshold = g_cmd_context->config->hampel_threshold;
            g_cmd_context->config->hampel_threshold = new_threshold;
            
            char response[256];
            snprintf(response, sizeof(response), 
                     "Hampel threshold updated: %.1f -> %.1f", old_threshold, new_threshold);
            send_response(response);
            ESP_LOGI(TAG, "%s", response);
            
            config_save_to_nvs(g_cmd_context->config, *g_cmd_context->threshold_high, 
                             *g_cmd_context->threshold_low);
        } else {
            send_response("ERROR: Hampel threshold must be between 1.0 and 10.0");
        }
    } else {
        send_response("ERROR: Missing or invalid 'value' field");
    }
}

// Command: savgol_filter
static void cmd_savgol_filter(cJSON *root) {
    cJSON *enabled = cJSON_GetObjectItem(root, "enabled");
    if (enabled && cJSON_IsBool(enabled)) {
        g_cmd_context->config->savgol_filter_enabled = cJSON_IsTrue(enabled);
        char response[64];
        snprintf(response, sizeof(response), "Savitzky-Golay filter %s", 
                 g_cmd_context->config->savgol_filter_enabled ? "enabled" : "disabled");
        send_response(response);
        ESP_LOGI(TAG, "%s", response);
        
        config_save_to_nvs(g_cmd_context->config, *g_cmd_context->threshold_high, 
                         *g_cmd_context->threshold_low);
    } else {
        send_response("ERROR: Missing or invalid 'enabled' field");
    }
}

// Command: butterworth_filter
static void cmd_butterworth_filter(cJSON *root) {
    cJSON *enabled = cJSON_GetObjectItem(root, "enabled");
    if (enabled && cJSON_IsBool(enabled)) {
        g_cmd_context->config->butterworth_enabled = cJSON_IsTrue(enabled);
        char response[64];
        snprintf(response, sizeof(response), "Butterworth filter %s", 
                 g_cmd_context->config->butterworth_enabled ? "enabled" : "disabled");
        send_response(response);
        ESP_LOGI(TAG, "%s", response);
        
        config_save_to_nvs(g_cmd_context->config, *g_cmd_context->threshold_high, 
                         *g_cmd_context->threshold_low);
    } else {
        send_response("ERROR: Missing or invalid 'enabled' field");
    }
}

// Command: smart_publishing
static void cmd_smart_publishing(cJSON *root) {
    cJSON *enabled = cJSON_GetObjectItem(root, "enabled");
    if (enabled && cJSON_IsBool(enabled)) {
        g_cmd_context->config->smart_publishing_enabled = cJSON_IsTrue(enabled);
        char response[128];
        snprintf(response, sizeof(response), "Smart publishing %s", 
                 g_cmd_context->config->smart_publishing_enabled ? "enabled" : "disabled");
        send_response(response);
        ESP_LOGI(TAG, "%s", response);
        
        config_save_to_nvs(g_cmd_context->config, *g_cmd_context->threshold_high, 
                         *g_cmd_context->threshold_low);
    } else {
        send_response("ERROR: Missing or invalid 'enabled' field");
    }
}

// Command: calibrate
static void cmd_calibrate(cJSON *root) {
    cJSON *action = cJSON_GetObjectItem(root, "action");
    if (!action || !cJSON_IsString(action)) {
        send_response("ERROR: Missing or invalid 'action' field");
        return;
    }
    
    const char *action_str = action->valuestring;
    
    if (strcmp(action_str, "start") == 0) {
        int samples = 1000; // Default samples
        cJSON *samples_obj = cJSON_GetObjectItem(root, "samples");
        if (samples_obj && cJSON_IsNumber(samples_obj)) {
            samples = (int)samples_obj->valueint;
        }
        
        if (calibration_start(samples, g_cmd_context->config, g_cmd_context->normalizer)) {
            char response[128];
            snprintf(response, sizeof(response), 
                     "Calibration started (%d samples per phase)", samples);
            send_response(response);
        } else {
            send_response("ERROR: Failed to start calibration (already in progress?)");
        }
        
    } else if (strcmp(action_str, "stop") == 0) {
        calibration_stop(g_cmd_context->config);
        send_response("Calibration stopped");
        
    } else if (strcmp(action_str, "status") == 0) {
        cJSON *response = cJSON_CreateObject();
        
        calibration_phase_t phase = calibration_get_phase();
        const char *phase_names[] = {"IDLE", "BASELINE", "MOVEMENT", "ANALYZING"};
        cJSON_AddStringToObject(response, "phase", phase_names[phase]);
        cJSON_AddBoolToObject(response, "active", calibration_is_active());
        
        // Add sample information for all active phases
        if (calibration_is_active()) {
            calibration_state_t calib_state;
            calibration_get_results(&calib_state);
            
            uint32_t samples_collected = calibration_get_samples_collected();
            uint32_t samples_target = calib_state.phase_target_samples;
            uint32_t traffic_rate = calib_state.traffic_rate;
            
            cJSON_AddNumberToObject(response, "samples_collected", (double)samples_collected);
            cJSON_AddNumberToObject(response, "samples_target", (double)samples_target);
            
            // Calculate estimated time remaining
            if (traffic_rate > 0 && samples_target > samples_collected) {
                uint32_t samples_remaining = samples_target - samples_collected;
                uint32_t estimated_sec = samples_remaining / traffic_rate;
                cJSON_AddNumberToObject(response, "estimated_time_remaining_sec", (double)estimated_sec);
            }
        }
        
        if (calibration_get_num_selected() > 0) {
            cJSON_AddNumberToObject(response, "num_selected", calibration_get_num_selected());
            cJSON_AddNumberToObject(response, "optimal_threshold", (double)calibration_get_threshold());
            
            // Add recommended filter configuration
            bool butterworth, wavelet, hampel, savgol, adaptive_norm;
            int wavelet_level;
            float wavelet_threshold, hampel_threshold, norm_alpha;
            calibration_get_filter_config(&butterworth, &wavelet, &wavelet_level, &wavelet_threshold,
                                         &hampel, &hampel_threshold, &savgol, &adaptive_norm, &norm_alpha);
            
            cJSON *filter_config = cJSON_CreateObject();
            cJSON_AddBoolToObject(filter_config, "butterworth_enabled", butterworth);
            cJSON_AddBoolToObject(filter_config, "wavelet_enabled", wavelet);
            cJSON_AddNumberToObject(filter_config, "wavelet_level", wavelet_level);
            cJSON_AddNumberToObject(filter_config, "wavelet_threshold", (double)wavelet_threshold);
            cJSON_AddBoolToObject(filter_config, "hampel_enabled", hampel);
            cJSON_AddNumberToObject(filter_config, "hampel_threshold", (double)hampel_threshold);
            cJSON_AddBoolToObject(filter_config, "savgol_enabled", savgol);
            cJSON_AddBoolToObject(filter_config, "adaptive_normalizer_enabled", adaptive_norm);
            cJSON_AddNumberToObject(filter_config, "adaptive_normalizer_alpha", (double)norm_alpha);
            cJSON_AddItemToObject(response, "filter_config", filter_config);
        }
        
        char *json_str = cJSON_PrintUnformatted(response);
        if (json_str) {
            mqtt_send_response(g_mqtt_state, json_str, g_response_topic);
            free(json_str);
        }
        cJSON_Delete(response);
        
    } else {
        char response[128];
        snprintf(response, sizeof(response), 
                 "ERROR: Unknown calibration action '%s' (use: start, stop, status)", action_str);
        send_response(response);
    }
}

// Command: adaptive_normalizer
static void cmd_adaptive_normalizer(cJSON *root) {
    cJSON *enabled = cJSON_GetObjectItem(root, "enabled");
    if (enabled && cJSON_IsBool(enabled)) {
        g_cmd_context->config->adaptive_normalizer_enabled = cJSON_IsTrue(enabled);
        char response[64];
        snprintf(response, sizeof(response), "Adaptive normalizer %s", 
                 g_cmd_context->config->adaptive_normalizer_enabled ? "enabled" : "disabled");
        send_response(response);
        ESP_LOGI(TAG, "%s", response);
        
        config_save_to_nvs(g_cmd_context->config, *g_cmd_context->threshold_high, 
                         *g_cmd_context->threshold_low);
    } else {
        send_response("ERROR: Missing or invalid 'enabled' field");
    }
}

// Command: adaptive_normalizer_alpha
static void cmd_adaptive_normalizer_alpha(cJSON *root) {
    cJSON *value = cJSON_GetObjectItem(root, "value");
    if (value && cJSON_IsNumber(value)) {
        float new_alpha = (float)value->valuedouble;
        if (!is_valid_float(new_alpha)) {
            send_response("ERROR: Invalid value (NaN or Infinity)");
            return;
        }
        if (new_alpha >= 0.001f && new_alpha <= 0.1f) {
            float old_alpha = g_cmd_context->config->adaptive_normalizer_alpha;
            g_cmd_context->config->adaptive_normalizer_alpha = new_alpha;
            
            // Reinitialize normalizer with new alpha
            adaptive_normalizer_init(g_cmd_context->normalizer, new_alpha);
            
            char response[256];
            snprintf(response, sizeof(response), 
                     "Adaptive normalizer alpha updated: %.4f -> %.4f (learning rate %s)", 
                     old_alpha, new_alpha,
                     new_alpha > old_alpha ? "increased" : "decreased");
            send_response(response);
            ESP_LOGI(TAG, "%s", response);
            
            config_save_to_nvs(g_cmd_context->config, *g_cmd_context->threshold_high, 
                             *g_cmd_context->threshold_low);
        } else {
            send_response("ERROR: Alpha must be between 0.001 and 0.1");
        }
    } else {
        send_response("ERROR: Missing or invalid 'value' field");
    }
}

// Command: adaptive_normalizer_reset_timeout
static void cmd_adaptive_normalizer_reset_timeout(cJSON *root) {
    cJSON *value = cJSON_GetObjectItem(root, "value");
    if (value && cJSON_IsNumber(value)) {
        uint32_t new_timeout = (uint32_t)value->valueint;
        if (new_timeout <= 300) {
            uint32_t old_timeout = g_cmd_context->config->adaptive_normalizer_reset_timeout_sec;
            g_cmd_context->config->adaptive_normalizer_reset_timeout_sec = new_timeout;
            
            char response[256];
            if (new_timeout == 0) {
                snprintf(response, sizeof(response), 
                         "Adaptive normalizer auto-reset disabled (was %lu sec)", 
                         (unsigned long)old_timeout);
            } else {
                snprintf(response, sizeof(response), 
                         "Adaptive normalizer reset timeout updated: %lu -> %lu seconds", 
                         (unsigned long)old_timeout, (unsigned long)new_timeout);
            }
            send_response(response);
            ESP_LOGI(TAG, "%s", response);
            
            config_save_to_nvs(g_cmd_context->config, *g_cmd_context->threshold_high, 
                             *g_cmd_context->threshold_low);
        } else {
            send_response("ERROR: Reset timeout must be between 0 and 300 seconds (0 = disabled)");
        }
    } else {
        send_response("ERROR: Missing or invalid 'value' field");
    }
}

// Command: adaptive_normalizer_stats
static void cmd_adaptive_normalizer_stats(cJSON *root) {
    cJSON *response = cJSON_CreateObject();
    
    cJSON_AddBoolToObject(response, "enabled", g_cmd_context->config->adaptive_normalizer_enabled);
    cJSON_AddNumberToObject(response, "alpha", (double)g_cmd_context->config->adaptive_normalizer_alpha);
    cJSON_AddNumberToObject(response, "reset_timeout_sec", g_cmd_context->config->adaptive_normalizer_reset_timeout_sec);
    
    // Get current normalizer statistics
    float mean, variance;
    adaptive_normalizer_get_stats(g_cmd_context->normalizer, &mean, &variance);
    
    cJSON_AddNumberToObject(response, "current_mean", (double)mean);
    cJSON_AddNumberToObject(response, "current_variance", (double)variance);
    cJSON_AddNumberToObject(response, "current_stddev", (double)sqrtf(variance));
    
    char *json_str = cJSON_PrintUnformatted(response);
    if (json_str) {
        mqtt_send_response(g_mqtt_state, json_str, g_response_topic);
        free(json_str);
    }
    cJSON_Delete(response);
}

// Command: traffic_generator_rate
static void cmd_traffic_generator_rate(cJSON *root) {
    cJSON *value = cJSON_GetObjectItem(root, "value");
    if (value && cJSON_IsNumber(value)) {
        uint32_t new_rate = (uint32_t)value->valueint;
        if (new_rate <= 50) {  // 0-50 pps
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
            
            config_save_to_nvs(g_cmd_context->config, *g_cmd_context->threshold_high, 
                             *g_cmd_context->threshold_low);
        } else {
            send_response("ERROR: Rate must be 0-50 packets/sec (0=disabled, recommended: 15)");
        }
    } else {
        send_response("ERROR: Missing or invalid 'value' field");
    }
}

// Command: wavelet_filter
static void cmd_wavelet_filter(cJSON *root) {
    cJSON *enabled = cJSON_GetObjectItem(root, "enabled");
    if (enabled && cJSON_IsBool(enabled)) {
        g_cmd_context->config->wavelet_enabled = cJSON_IsTrue(enabled);
        char response[64];
        snprintf(response, sizeof(response), "Wavelet filter %s", 
                 g_cmd_context->config->wavelet_enabled ? "enabled" : "disabled");
        send_response(response);
        ESP_LOGI(TAG, "%s", response);
        
        config_save_to_nvs(g_cmd_context->config, *g_cmd_context->threshold_high, 
                         *g_cmd_context->threshold_low);
    } else {
        send_response("ERROR: Missing or invalid 'enabled' field");
    }
}

// Command: wavelet_level
static void cmd_wavelet_level(cJSON *root) {
    cJSON *value = cJSON_GetObjectItem(root, "value");
    if (value && cJSON_IsNumber(value)) {
        int new_level = (int)value->valueint;
        if (new_level >= 1 && new_level <= 3) {
            int old_level = g_cmd_context->config->wavelet_level;
            g_cmd_context->config->wavelet_level = new_level;
            
            char response[256];
            snprintf(response, sizeof(response), 
                     "Wavelet decomposition level updated: %d -> %d (denoising %s)", 
                     old_level, new_level,
                     new_level > old_level ? "more aggressive" : "less aggressive");
            send_response(response);
            ESP_LOGI(TAG, "%s", response);
            
            config_save_to_nvs(g_cmd_context->config, *g_cmd_context->threshold_high, 
                             *g_cmd_context->threshold_low);
        } else {
            send_response("ERROR: Wavelet level must be between 1 and 3 (recommended: 3)");
        }
    } else {
        send_response("ERROR: Missing or invalid 'value' field");
    }
}

// Command: wavelet_threshold
static void cmd_wavelet_threshold(cJSON *root) {
    cJSON *value = cJSON_GetObjectItem(root, "value");
    if (value && cJSON_IsNumber(value)) {
        float new_threshold = (float)value->valuedouble;
        if (!is_valid_float(new_threshold)) {
            send_response("ERROR: Invalid value (NaN or Infinity)");
            return;
        }
        if (new_threshold >= 0.5f && new_threshold <= 2.0f) {
            float old_threshold = g_cmd_context->config->wavelet_threshold;
            g_cmd_context->config->wavelet_threshold = new_threshold;
            
            char response[256];
            snprintf(response, sizeof(response), 
                     "Wavelet threshold updated: %.2f -> %.2f (noise removal %s)", 
                     old_threshold, new_threshold,
                     new_threshold > old_threshold ? "more aggressive" : "less aggressive");
            send_response(response);
            ESP_LOGI(TAG, "%s", response);
            
            config_save_to_nvs(g_cmd_context->config, *g_cmd_context->threshold_high, 
                             *g_cmd_context->threshold_low);
        } else {
            send_response("ERROR: Wavelet threshold must be between 0.5 and 2.0 (recommended: 1.0)");
        }
    } else {
        send_response("ERROR: Missing or invalid 'value' field");
    }
}

// Command: factory_reset
static void cmd_factory_reset(cJSON *root) {
    ESP_LOGW(TAG, "⚠️  Factory reset requested");
    
    nvs_factory_reset();
    
    // Reset config to defaults
    config_init_defaults(g_cmd_context->config);
    *g_cmd_context->threshold_high = 0.43f;
    *g_cmd_context->threshold_low = 0.43f * g_cmd_context->config->hysteresis_ratio;
    
    calibration_init();
    
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
    {"threshold", cmd_threshold},
    {"stats", cmd_stats},
    {"info", cmd_info},
    {"logs", cmd_logs},
    {"analyze", cmd_analyze},
    {"persistence", cmd_persistence},
    {"debounce", cmd_debounce},
    {"hysteresis", cmd_hysteresis},
    {"variance_scale", cmd_variance_scale},
    {"features", cmd_features},
    {"hampel_filter", cmd_hampel_filter},
    {"hampel_threshold", cmd_hampel_threshold},
    {"savgol_filter", cmd_savgol_filter},
    {"butterworth_filter", cmd_butterworth_filter},
    {"wavelet_filter", cmd_wavelet_filter},
    {"wavelet_level", cmd_wavelet_level},
    {"wavelet_threshold", cmd_wavelet_threshold},
    {"smart_publishing", cmd_smart_publishing},
    {"calibrate", cmd_calibrate},
    {"adaptive_normalizer", cmd_adaptive_normalizer},
    {"adaptive_normalizer_alpha", cmd_adaptive_normalizer_alpha},
    {"adaptive_normalizer_reset_timeout", cmd_adaptive_normalizer_reset_timeout},
    {"adaptive_normalizer_stats", cmd_adaptive_normalizer_stats},
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
