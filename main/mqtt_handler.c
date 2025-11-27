/*
 * ESPectre - MQTT Handler Module Implementation
 * 
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * License: GPLv3
 */

#include "mqtt_handler.h"
#include <string.h>
#include <math.h>
#include <stdatomic.h>
#include "esp_log.h"
#include "esp_event.h"
#include "esp_timer.h"
#include "cJSON.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "freertos/queue.h"
#include "freertos/semphr.h"
#include "sdkconfig.h"
#include "config_manager.h"
#include "espectre.h"

static const char *TAG = "MQTT_Handler";

// State names for motion detection
static const char* motion_state_to_string(csi_motion_state_t state) {
    switch (state) {
        case CSI_STATE_IDLE: return "idle";
        case CSI_STATE_MOTION: return "motion";
        default: return "unknown";
    }
}

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

int mqtt_publish_segmentation(mqtt_handler_state_t *state,
                              const segmentation_result_t *result,
                              const char *topic) {
    if (!state || !result || !topic || !state->connected) {
        return -1;
    }
    
    cJSON *root = cJSON_CreateObject();
    if (!root) {
        ESP_LOGE(TAG, "Failed to create JSON object");
        return -1;
    }
    
    cJSON_AddNumberToObject(root, "movement", (double)result->moving_variance);
    cJSON_AddNumberToObject(root, "threshold", (double)result->threshold);
    cJSON_AddStringToObject(root, "state", motion_state_to_string(result->state));
    cJSON_AddNumberToObject(root, "packets_processed", (double)result->packets_processed);
    cJSON_AddNumberToObject(root, "packets_dropped", (double)result->packets_dropped);
    
    // Add features if available (only during MOTION with features_enabled)
    if (result->has_features) {
        cJSON *features = cJSON_CreateObject();
        if (features) {
            cJSON_AddNumberToObject(features, "variance", (double)result->features.variance);
            cJSON_AddNumberToObject(features, "skewness", (double)result->features.skewness);
            cJSON_AddNumberToObject(features, "kurtosis", (double)result->features.kurtosis);
            cJSON_AddNumberToObject(features, "entropy", (double)result->features.entropy);
            cJSON_AddNumberToObject(features, "iqr", (double)result->features.iqr);
            cJSON_AddNumberToObject(features, "spatial_variance", (double)result->features.spatial_variance);
            cJSON_AddNumberToObject(features, "spatial_correlation", (double)result->features.spatial_correlation);
            cJSON_AddNumberToObject(features, "spatial_gradient", (double)result->features.spatial_gradient);
            cJSON_AddNumberToObject(features, "temporal_delta_mean", (double)result->features.temporal_delta_mean);
            cJSON_AddNumberToObject(features, "temporal_delta_variance", (double)result->features.temporal_delta_variance);
            cJSON_AddItemToObject(root, "features", features);
        }
    }
    
    cJSON_AddNumberToObject(root, "timestamp", (double)result->timestamp);
    
    int ret = mqtt_publish_json(state->client, topic, root, 0, 0);
    cJSON_Delete(root);
    
    if (ret == 0) {
        state->publish_count++;
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
                        csi_motion_state_t current_state,
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
                              csi_motion_state_t motion_state,
                              int64_t current_time) {
    if (!state) return;
    
    state->last_published_movement = movement;
    state->last_published_state = motion_state;
    state->last_publish_time = current_time;
}

void mqtt_handler_set_command_callback(void (*callback)(const char *data, int data_len)) {
    g_command_callback = callback;
}

int mqtt_publish_binary(mqtt_handler_state_t *state,
                       const char *topic,
                       const uint8_t *data,
                       size_t data_len) {
    if (!state || !topic || !data || data_len == 0) {
        ESP_LOGE(TAG, "mqtt_publish_binary: Invalid parameters");
        return -1;
    }
    
    if (!state->connected) {
        ESP_LOGW(TAG, "Cannot publish binary: MQTT not connected");
        return -1;
    }
    
    // Publish binary data directly (QoS 0, no retain)
    int msg_id = esp_mqtt_client_publish(state->client, topic, (const char *)data, data_len, 0, 0);
    
    if (msg_id < 0) {
        ESP_LOGE(TAG, "Failed to publish binary data to %s", topic);
        return -1;
    }
    
    return 0;
}

// Helper function to get timestamp in seconds
static inline int64_t get_timestamp_sec(void) {
    return esp_timer_get_time() / 1000000;
}

// Helper function to get timestamp in milliseconds
static inline int64_t get_timestamp_ms(void) {
    return esp_timer_get_time() / 1000;
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

// MQTT publisher task
static void mqtt_publisher(void *pvParameters) {
    mqtt_publisher_context_t *ctx = (mqtt_publisher_context_t *)pvParameters;
    
    TickType_t last_wake_time = xTaskGetTickCount();
    const TickType_t publish_period = pdMS_TO_TICKS((uint32_t)(PUBLISH_INTERVAL * 1000));
    
    int64_t last_csi_log_time = 0;
    
    // Smart publishing configuration
    runtime_config_t *config = (runtime_config_t *)ctx->config;
    mqtt_publish_config_t pub_config = {
        .enabled = config->smart_publishing_enabled,
        .delta_threshold = 0.05f,
        .max_interval_sec = 5.0f
    };
    
    while (1) {
        vTaskDelayUntil(&last_wake_time, publish_period);
        
        // Read state values with mutex protection
        csi_motion_state_t motion_state;
        uint32_t packets_processed;
        csi_features_t features;
        bool has_features = false;
        
        if (xSemaphoreTake(ctx->state_mutex, pdMS_TO_TICKS(100)) == pdTRUE) {
            motion_state = *(csi_motion_state_t *)ctx->motion_state;
            packets_processed = *(uint32_t *)ctx->packets_processed;
            
            // Copy features if they were extracted
            if (config->features_enabled) {
                features = *(csi_features_t *)ctx->current_features;
                has_features = true;
            }
            
            xSemaphoreGive(ctx->state_mutex);
        } else {
            ESP_LOGW(TAG, "MQTT publisher: Failed to acquire mutex, skipping publish cycle");
            continue;
        }
        
        // Get CSI processor for motion detection data
        csi_processor_context_t *csi_proc = (csi_processor_context_t *)ctx->csi_processor;
        
        // Calculate packet delta (packets processed since last cycle)
        static uint32_t last_packets_processed = 0;
        static uint32_t last_packets_dropped = 0;
        uint32_t packet_delta = packets_processed - last_packets_processed;
        uint32_t packets_dropped_stat = atomic_load((_Atomic uint32_t *)ctx->packets_dropped);
        uint32_t dropped_delta = packets_dropped_stat - last_packets_dropped;
        last_packets_processed = packets_processed;
        last_packets_dropped = packets_dropped_stat;
        
        // CSI logging with progress bar (always enabled)
        int64_t now = get_timestamp_sec();
        if (now - last_csi_log_time >= LOG_CSI_VALUES_INTERVAL) {
            
            // Get motion detection data
            float moving_variance = csi_processor_get_moving_variance(csi_proc);
            float threshold = csi_processor_get_threshold(csi_proc);
            
            const char *state_names[] = {"IDLE", "MOTION"};
            const char *state_str = (motion_state < 2) ? state_names[motion_state] : "UNKNOWN";
            
            // Calculate progress based on motion detection (moving_variance / threshold)
            float progress = (threshold > 0.0f) ? (moving_variance / threshold) : 0.0f;
            
            // Format progress bar with threshold marker at 100%
            char progress_bar[256];
            format_progress_bar(progress_bar, sizeof(progress_bar), progress, 1.0f);
            
            ESP_LOGI(TAG, "📊 %s | pkts:%lu drop:%lu | mvmt:%.4f thr:%.4f | %s",
                     progress_bar, (unsigned long)packet_delta, (unsigned long)dropped_delta,
                     moving_variance, threshold, state_str);
            last_csi_log_time = now;
        }
        
        // Publish motion detection data
        int64_t current_time = get_timestamp_ms();
        float moving_variance = csi_processor_get_moving_variance(csi_proc);
        
        if (mqtt_should_publish(ctx->mqtt_state, moving_variance, motion_state, 
                                &pub_config, current_time)) {
            // Prepare segmentation result
            segmentation_result_t result = {
                .moving_variance = moving_variance,
                .threshold = csi_processor_get_threshold(csi_proc),
                .state = motion_state,
                .timestamp = get_timestamp_sec(),
                .packets_processed = packet_delta,
                .packets_dropped = dropped_delta,
                .has_features = has_features
            };
            
            if (has_features) {
                result.features = features;
            }
            
            mqtt_publish_segmentation(ctx->mqtt_state, &result, ctx->mqtt_topic);
            mqtt_update_publish_state(ctx->mqtt_state, moving_variance, motion_state, current_time);
        }
    }
}

void mqtt_start_publisher(mqtt_publisher_context_t *context) {
    if (!context) {
        ESP_LOGE(TAG, "mqtt_start_publisher: NULL context");
        return;
    }
    
    xTaskCreate(mqtt_publisher, "mqtt_pub", 4096, context, 5, NULL);
    ESP_LOGI(TAG, "MQTT publisher task started");
}
