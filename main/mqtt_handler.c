/*
 * ESPectre - MQTT Handler Module Implementation
 * 
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * License: GPLv3
 */

#include "mqtt_handler.h"
#include "calibration.h"
#include <string.h>
#include <math.h>
#include "esp_log.h"
#include "esp_event.h"
#include "esp_timer.h"
#include "cJSON.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "freertos/queue.h"

static const char *TAG = "MQTT_Handler";

// Callback for command processing
static void (*g_command_callback)(const char *data, int data_len) = NULL;
static const char *g_cmd_topic = NULL;


// Helper function to publish JSON object
static int mqtt_publish_json(esp_mqtt_client_handle_t client, const char *topic, 
                             cJSON *root, int qos, int retain) {
    if (!client || !topic || !root) {
        return -1;
    }
    
    char *json_str = cJSON_PrintUnformatted(root);
    if (!json_str) {
        return -1;
    }
    
    int msg_id = esp_mqtt_client_publish(client, topic, json_str, 0, qos, retain);
    free(json_str);
    
    return (msg_id >= 0) ? 0 : -1;
}

// MQTT event handler
static void mqtt_event_handler(void *handler_args, esp_event_base_t base,
                               int32_t event_id, void *event_data) {
    esp_mqtt_event_handle_t event = (esp_mqtt_event_handle_t)event_data;
    mqtt_handler_state_t *state = (mqtt_handler_state_t *)handler_args;
    
    switch (event->event_id) {
        case MQTT_EVENT_CONNECTED:
            ESP_LOGD(TAG, "MQTT connected to broker");
            state->connected = true;
            
            // Subscribe to command topic if configured
            if (g_cmd_topic) {
                esp_mqtt_client_subscribe(state->client, g_cmd_topic, 0);
                ESP_LOGD(TAG, "Subscribed to command topic: %s", g_cmd_topic);
            }
            break;
            
        case MQTT_EVENT_DISCONNECTED:
            ESP_LOGI(TAG, "MQTT disconnected from broker");
            state->connected = false;
            break;
            
        case MQTT_EVENT_DATA:
            ESP_LOGD(TAG, "MQTT data received on topic: %.*s", event->topic_len, event->topic);
            
            // Check if this is a command message
            if (g_cmd_topic && g_command_callback && 
                strncmp(event->topic, g_cmd_topic, event->topic_len) == 0) {
                g_command_callback(event->data, event->data_len);
            }
            break;
            
        case MQTT_EVENT_ERROR:
            ESP_LOGE(TAG, "MQTT error");
            break;
            
        default:
            break;
    }
}

int mqtt_handler_init(mqtt_handler_state_t *state, const mqtt_config_t *config) {
    if (!state || !config) {
        ESP_LOGE(TAG, "mqtt_handler_init: NULL pointer");
        return -1;
    }
    
    memset(state, 0, sizeof(mqtt_handler_state_t));
    
    // Store command topic for subscription
    g_cmd_topic = config->cmd_topic;
    
    esp_mqtt_client_config_t mqtt_cfg = {
        .broker.address.uri = config->broker_uri,
    };
    
    if (config->username && strlen(config->username) > 0) {
        mqtt_cfg.credentials.username = config->username;
    }
    
    if (config->password && strlen(config->password) > 0) {
        mqtt_cfg.credentials.authentication.password = config->password;
    }
    
    state->client = esp_mqtt_client_init(&mqtt_cfg);
    if (!state->client) {
        ESP_LOGE(TAG, "Failed to initialize MQTT client");
        return -1;
    }
    
    // Register event handler
    ESP_ERROR_CHECK(esp_mqtt_client_register_event(state->client, ESP_EVENT_ANY_ID, 
                                                   mqtt_event_handler, state));
    
    state->connected = false;
    state->publish_initialized = false;
    state->publish_count = 0;
    state->skip_count = 0;
    
    return 0;
}

int mqtt_handler_start(mqtt_handler_state_t *state) {
    if (!state || !state->client) {
        ESP_LOGE(TAG, "mqtt_handler_start: Invalid state");
        return -1;
    }
    
    esp_err_t err = esp_mqtt_client_start(state->client);
    if (err != ESP_OK) {
        ESP_LOGE(TAG, "Failed to start MQTT client: %d", err);
        return -1;
    }
    
    return 0;
}

void mqtt_handler_stop(mqtt_handler_state_t *state) {
    if (state && state->client) {
        esp_mqtt_client_stop(state->client);
        esp_mqtt_client_destroy(state->client);
        state->client = NULL;
        state->connected = false;
    }
}

bool mqtt_handler_is_connected(const mqtt_handler_state_t *state) {
    return state ? state->connected : false;
}

int mqtt_publish_detection(mqtt_handler_state_t *state,
                           const detection_result_t *result,
                           const char *topic) {
    if (!state || !result || !topic || !state->connected) {
        return -1;
    }
    
    cJSON *root = cJSON_CreateObject();
    if (!root) {
        ESP_LOGE(TAG, "Failed to create JSON object");
        return -1;
    }
    
    cJSON_AddNumberToObject(root, "movement", (double)result->score);
    cJSON_AddNumberToObject(root, "confidence", (double)result->confidence);
    cJSON_AddStringToObject(root, "state", detection_state_to_string(result->state));
    cJSON_AddNumberToObject(root, "timestamp", (double)result->timestamp);
    
    int ret = mqtt_publish_json(state->client, topic, root, 0, 0);
    cJSON_Delete(root);
    
    if (ret == 0) {
        state->publish_count++;
    }
    return ret;
}

int mqtt_publish_calibration_status(mqtt_handler_state_t *state,
                                    uint8_t phase,
                                    uint32_t phase_target_samples,
                                    uint32_t samples_collected,
                                    uint32_t traffic_rate,
                                    const char *topic) {
    if (!state || !topic || !state->connected) {
        return -1;
    }
    
    cJSON *root = cJSON_CreateObject();
    if (!root) {
        ESP_LOGE(TAG, "Failed to create JSON object");
        return -1;
    }
    
    cJSON_AddStringToObject(root, "type", "calibration_status");
    
    // Map phase enum to string
    const char *phase_names[] = {"IDLE", "BASELINE", "MOVEMENT", "ANALYZING"};
    const char *phase_str = (phase < 4) ? phase_names[phase] : "UNKNOWN";
    cJSON_AddStringToObject(root, "phase", phase_str);
    
    // Only add target samples (not collected - shown only in calibration_complete)
    cJSON_AddNumberToObject(root, "phase_target_samples", (double)phase_target_samples);
    
    int ret = mqtt_publish_json(state->client, topic, root, 0, 0);
    cJSON_Delete(root);
    
    if (ret == 0) {
        ESP_LOGD(TAG, "Published calibration status: phase=%s, target=%lu", 
                 phase_str, (unsigned long)phase_target_samples);
    }
    return ret;
}

int mqtt_publish_calibration_complete(mqtt_handler_state_t *state,
                                      const void *calib_results,
                                      const char *topic) {
    if (!state || !calib_results || !topic) {
        ESP_LOGE(TAG, "mqtt_publish_calibration_complete: NULL pointer");
        return -1;
    }
    
    if (!state->connected) {
        return -1;
    }
    
    // Cast to calibration_state_t (defined in calibration.h)
    const calibration_state_t *calib = (const calibration_state_t*)calib_results;
    
    cJSON *root = cJSON_CreateObject();
    if (!root) {
        ESP_LOGE(TAG, "Failed to create JSON object");
        return -1;
    }
    
    cJSON_AddStringToObject(root, "type", "calibration_complete");
    
    // Summary object
    cJSON *summary = cJSON_CreateObject();
    // Add sample counts
    cJSON_AddNumberToObject(summary, "baseline_samples_collected", (double)calib->baseline_stats[0].count);
    cJSON_AddNumberToObject(summary, "movement_samples_collected", (double)calib->movement_stats[0].count);
    
    cJSON_AddNumberToObject(summary, "baseline_score", (double)calib->baseline_mean_score);
    cJSON_AddNumberToObject(summary, "movement_score", (double)calib->movement_mean_score);
    cJSON_AddNumberToObject(summary, "separation_ratio", (double)calib->separation_ratio);
    
    // Separation quality assessment
    const char *quality = "poor";
    if (calib->separation_ratio >= 2.5f) quality = "excellent";
    else if (calib->separation_ratio >= 2.0f) quality = "good";
    else if (calib->separation_ratio >= 1.5f) quality = "fair";
    cJSON_AddStringToObject(summary, "separation_quality", quality);
    
    cJSON_AddNumberToObject(summary, "optimal_threshold", (double)calib->optimal_threshold);
    cJSON_AddNumberToObject(summary, "num_features_selected", calib->num_selected);
    
    // Top features array (up to 3 for brevity)
    const char *feature_names[] = {
        "variance", "skewness", "kurtosis", "entropy", "iqr",
        "spatial_variance", "spatial_correlation", "spatial_gradient",
        "temporal_delta_mean", "temporal_delta_variance"
    };
    
    cJSON *top_features = cJSON_CreateArray();
    uint8_t num_to_show = (calib->num_selected < 3) ? calib->num_selected : 3;
    for (uint8_t i = 0; i < num_to_show; i++) {
        cJSON *feat = cJSON_CreateObject();
        uint8_t feat_idx = calib->selected_features[i];
        const char *feat_name = (feat_idx < 10) ? feature_names[feat_idx] : "unknown";
        cJSON_AddStringToObject(feat, "name", feat_name);
        cJSON_AddNumberToObject(feat, "weight", (double)calib->optimized_weights[i]);
        cJSON_AddItemToArray(top_features, feat);
    }
    cJSON_AddItemToObject(summary, "top_features", top_features);
    
    // Filter config
    cJSON *filters = cJSON_CreateObject();
    cJSON_AddBoolToObject(filters, "butterworth", calib->recommended_butterworth);
    cJSON_AddBoolToObject(filters, "wavelet", calib->recommended_wavelet);
    if (calib->recommended_wavelet) {
        cJSON_AddNumberToObject(filters, "wavelet_level", calib->recommended_wavelet_level);
        cJSON_AddNumberToObject(filters, "wavelet_threshold", (double)calib->recommended_wavelet_threshold);
    }
    cJSON_AddBoolToObject(filters, "hampel", calib->recommended_hampel);
    if (calib->recommended_hampel) {
        cJSON_AddNumberToObject(filters, "hampel_threshold", (double)calib->recommended_hampel_threshold);
    }
    cJSON_AddBoolToObject(filters, "savgol", calib->recommended_savgol);
    cJSON_AddBoolToObject(filters, "adaptive_normalizer", calib->recommended_adaptive_normalizer);
    if (calib->recommended_adaptive_normalizer) {
        cJSON_AddNumberToObject(filters, "adaptive_normalizer_alpha", (double)calib->recommended_normalizer_alpha);
    }
    cJSON_AddItemToObject(summary, "filter_config", filters);
    
    cJSON_AddItemToObject(root, "summary", summary);
    
    // Warnings array
    cJSON *warnings = cJSON_CreateArray();
    if (calib->separation_ratio < 2.0f) {
        char warning[128];
        snprintf(warning, sizeof(warning), 
                "Low separation ratio (%.2f < 2.0) - consider more intense movement", 
                calib->separation_ratio);
        cJSON_AddItemToArray(warnings, cJSON_CreateString(warning));
    }
    cJSON_AddItemToObject(root, "warnings", warnings);
    
    int ret = mqtt_publish_json(state->client, topic, root, 0, 0);
    cJSON_Delete(root);
    
    if (ret == 0) {
        ESP_LOGI(TAG, "📊 Published calibration complete recap");
    }
    return ret;
}

int mqtt_send_response(mqtt_handler_state_t *state,
                      const char *message,
                      const char *response_topic) {
    if (!state || !message || !response_topic) {
        ESP_LOGE(TAG, "mqtt_send_response: NULL pointer");
        return -1;
    }
    
    if (!state->connected) {
        ESP_LOGW(TAG, "Cannot send response: MQTT not connected");
        return -1;
    }
    
    // Check if message is already valid JSON
    cJSON *test_json = cJSON_Parse(message);
    if (test_json) {
        // Message is already valid JSON, send it directly
        cJSON_Delete(test_json);
        int msg_id = esp_mqtt_client_publish(state->client, response_topic, message, 0, 1, 0);
        return (msg_id >= 0) ? 0 : -1;
    }
    
    // Message is plain text, wrap it in {"response": "..."}
    cJSON *root = cJSON_CreateObject();
    if (!root) {
        ESP_LOGE(TAG, "Failed to create JSON response");
        return -1;
    }
    
    cJSON_AddStringToObject(root, "response", message);
    
    int ret = mqtt_publish_json(state->client, response_topic, root, 1, 0);
    cJSON_Delete(root);
    return ret;
}

bool mqtt_should_publish(mqtt_handler_state_t *state,
                        float current_movement,
                        detection_state_t current_state,
                        const mqtt_publish_config_t *config,
                        int64_t current_time) {
    if (!state || !config) {
        return true;
    }
    
    if (!config->enabled) {
        return true;
    }
    
    if (!state->publish_initialized) {
        state->publish_initialized = true;
        return true;
    }
    
    int64_t time_since_last = current_time - state->last_publish_time;
    
    // State changed
    if (current_state != state->last_published_state) {
        return true;
    }
    
    // Significant movement change
    float delta = fabsf(current_movement - state->last_published_movement);
    if (delta > config->delta_threshold) {
        return true;
    }
    
    // Heartbeat interval
    if (time_since_last >= (int64_t)(config->max_interval_sec * 1000)) {
        return true;
    }
    
    return false;
}

void mqtt_update_publish_state(mqtt_handler_state_t *state,
                              float movement,
                              detection_state_t det_state,
                              int64_t current_time) {
    if (!state) return;
    
    state->last_published_movement = movement;
    state->last_published_state = det_state;
    state->last_publish_time = current_time;
}

void mqtt_get_publish_stats(const mqtt_handler_state_t *state,
                           uint32_t *published,
                           uint32_t *skipped) {
    if (!state) return;
    
    if (published) *published = state->publish_count;
    if (skipped) *skipped = state->skip_count;
}

void mqtt_handler_set_command_callback(void (*callback)(const char *data, int data_len)) {
    g_command_callback = callback;
}
