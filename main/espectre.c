/*
 * ESPectre - Wi-Fi CSI Movement Detection for ESP32-S3
 *
 * Uses Channel State Information (CSI) from Wi-Fi packets to detect movement.
 * Extracts 15 mathematical features and combines them with configurable weights
 * to distinguish between static environment and human movement.
 * 
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * License: GPLv3
 */

#include <stdio.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "freertos/queue.h"
#include "freertos/event_groups.h"
#include "esp_wifi.h"
#include "esp_event.h"
#include "esp_log.h"
#include "esp_system.h"
#include "nvs_flash.h"
#include "mqtt_client.h"
#include "cJSON.h"
#include "esp_timer.h"

// Configuration - can be overridden via menuconfig

#define WIFI_SSID           CONFIG_WIFI_SSID
#define WIFI_PASSWORD       CONFIG_WIFI_PASSWORD
#define MQTT_BROKER_URI     CONFIG_MQTT_BROKER_URI
#define MQTT_TOPIC          CONFIG_MQTT_TOPIC
#define MQTT_CMD_TOPIC      CONFIG_MQTT_TOPIC "/cmd"
#define MQTT_RESPONSE_TOPIC CONFIG_MQTT_TOPIC "/response"
#define MQTT_USERNAME       CONFIG_MQTT_USERNAME
#define MQTT_PASSWORD       CONFIG_MQTT_PASSWORD

// Signal processing - these values reduce false positives
#define DEBOUNCE_COUNT      3         // consecutive detections needed
#define HYSTERESIS_RATIO    0.7f      // prevents rapid state flipping
#define PERSISTENCE_TIMEOUT 3         // seconds to wait before downgrading state
#define DEFAULT_THRESHOLD   0.40f     // tuned from real-world testing

#define PUBLISH_INTERVAL    1.0f
#define CONFIDENCE_THRESHOLD 0.5f

// Smart publishing reduces MQTT traffic by only publishing significant changes
#define SMART_PUBLISHING_ENABLED true
#define PUBLISH_DELTA_THRESHOLD 0.05f  // 5% change triggers publish
#define PUBLISH_MAX_INTERVAL 5.0f      // heartbeat interval

#define VARIANCE_SCALE      400.0f
#define BUFFER_SIZE         100
#define STATS_BUFFER_SIZE   100
#define BASELINE_UPDATE_RATE 0.01f
#define LOG_CSI_VALUES_INTERVAL 1
#define STATS_LOG_INTERVAL  100  // log stats every N packets

// Feature weights - tuned through experimentation (should sum to ~1.0)
#define WEIGHT_VARIANCE     0.30f
#define WEIGHT_ENTROPY      0.20f
#define WEIGHT_SPATIAL_VAR  0.20f
#define WEIGHT_TEMPORAL_CORR 0.15f
#define WEIGHT_PEAK_RATE    0.15f

// Multi-window analysis looks at different time scales
#define WINDOW_SHORT_SIZE   10   // ~1 second
#define WINDOW_MEDIUM_SIZE  50   // ~5 seconds
#define WINDOW_LONG_SIZE    100  // ~10 seconds

// Hampel filter removes outliers using Median Absolute Deviation
#define HAMPEL_THRESHOLD    3.0f
#define SAVGOL_WINDOW_SIZE  5     // must be odd
#define SAVGOL_POLY_ORDER   2

#define CUSUM_THRESHOLD     0.5f
#define CUSUM_DRIFT         0.01f

// State machine thresholds
#define MICRO_MOVEMENT_THRESHOLD    0.10f
#define INTENSE_MOVEMENT_THRESHOLD  0.70f


static const char *TAG = "ESPectre";

// Runtime configuration - can be changed via MQTT commands
static struct {
    bool csi_logs_enabled;
    bool verbose_mode;
    uint8_t debounce_count;
    float hysteresis_ratio;
    int persistence_timeout;
    float variance_scale;
    
    // Feature weights - optimized through testing
    float weight_variance;
    float weight_spatial_gradient;
    float weight_variance_short;
    float weight_iqr;
    
    int window_short_size;
    int window_medium_size;
    int window_long_size;
    
    bool hampel_filter_enabled;
    float hampel_threshold;
    bool savgol_filter_enabled;
    int savgol_window_size;
    
    bool cusum_enabled;
    float cusum_threshold;
    float cusum_drift;
    
    bool granular_states_enabled;
    float micro_movement_threshold;
    float intense_movement_threshold;
    
} control_state = {
    .csi_logs_enabled = true,
    .verbose_mode = false,
    .debounce_count = DEBOUNCE_COUNT,
    .hysteresis_ratio = HYSTERESIS_RATIO,
    .persistence_timeout = PERSISTENCE_TIMEOUT,
    .variance_scale = VARIANCE_SCALE,
    
    .weight_variance = 0.25f,
    .weight_spatial_gradient = 0.25f,
    .weight_variance_short = 0.35f,
    .weight_iqr = 0.15f,
    
    .window_short_size = WINDOW_SHORT_SIZE,
    .window_medium_size = WINDOW_MEDIUM_SIZE,
    .window_long_size = WINDOW_LONG_SIZE,
    
    .hampel_filter_enabled = true,
    .hampel_threshold = 2.0f,
    .savgol_filter_enabled = true,
    .savgol_window_size = SAVGOL_WINDOW_SIZE,
    
    .cusum_enabled = false,
    .cusum_threshold = CUSUM_THRESHOLD,
    .cusum_drift = CUSUM_DRIFT,
    
    .granular_states_enabled = false,  // simple 2-state by default
    .micro_movement_threshold = MICRO_MOVEMENT_THRESHOLD,
    .intense_movement_threshold = INTENSE_MOVEMENT_THRESHOLD,
};


typedef enum {
    STATE_IDLE,
    STATE_MICRO_MOVEMENT,
    STATE_DETECTED,
    STATE_INTENSE_MOVEMENT
} detection_state_t;

typedef struct {
    float movement;
    float confidence;
    detection_state_t state;
    float baseline;
    float threshold;
    int64_t timestamp;
} detection_result_t;

typedef struct {
    float *data;
    size_t size;
    size_t index;
    size_t count;
} history_buffer_t;

// CSI features extracted from raw data
typedef struct {
    float mean;
    float variance;
    float skewness;
    float kurtosis;
    float entropy;
    float iqr;  // Interquartile range
    
    float spatial_variance;
    float spatial_correlation;
    float spatial_gradient;
    
    float autocorr_lag1;
    float zero_crossing_rate;
    float peak_rate;
    
    float variance_short;
    float variance_medium;
    float variance_long;
    
} csi_features_t;

typedef struct {
    float cumsum_pos;
    float cumsum_neg;
    float mean_estimate;
    bool initialized;
} cusum_state_t;

static struct {
    history_buffer_t history_buffer;
    
    float detection_score;
    
    float threshold_high;
    float threshold_low;
    
    detection_state_t state;
    uint8_t consecutive_detections;
    int64_t last_detection_time;
    float confidence;
    
    esp_mqtt_client_handle_t mqtt_client;
    bool mqtt_connected;
    
    uint32_t packets_received;
    uint32_t packets_processed;
    
    bool wifi_connected;
    
    float stats_buffer[STATS_BUFFER_SIZE];
    size_t stats_index;
    size_t stats_count;
    
    csi_features_t current_features;
    
} g_state = {0};

static EventGroupHandle_t s_wifi_event_group;
#define WIFI_CONNECTED_BIT BIT0


static int64_t get_timestamp_ms(void) {
    return esp_timer_get_time() / 1000;
}

static int64_t get_timestamp_sec(void) {
    return esp_timer_get_time() / 1000000;
}

static float calculate_variance(const int8_t *data, size_t len) {
    if (len == 0) return 0.0f;
    
    float mean = 0.0f;
    for (size_t i = 0; i < len; i++) {
        mean += data[i];
    }
    mean /= len;
    
    float variance = 0.0f;
    for (size_t i = 0; i < len; i++) {
        float diff = data[i] - mean;
        variance += diff * diff;
    }
    return variance / len;
}

static float calculate_mean(const float *data, size_t len) {
    if (len == 0) return 0.0f;
    
    float sum = 0.0f;
    for (size_t i = 0; i < len; i++) {
        sum += data[i];
    }
    return sum / len;
}

static int compare_float(const void *a, const void *b) {
    float fa = *(const float*)a;
    float fb = *(const float*)b;
    return (fa > fb) - (fa < fb);
}

static int compare_int8(const void *a, const void *b) {
    int8_t ia = *(const int8_t*)a;
    int8_t ib = *(const int8_t*)b;
    return (ia > ib) - (ia < ib);
}

// Helper to calculate mean for int8_t arrays
static float calculate_mean_int8(const int8_t *data, size_t len) {
    if (len == 0) return 0.0f;
    
    float sum = 0.0f;
    for (size_t i = 0; i < len; i++) {
        sum += data[i];
    }
    return sum / len;
}

// Helper to calculate moments (variance, skewness, kurtosis) in one pass
typedef struct {
    float mean;
    float m2;  // second moment (variance * n)
    float m3;  // third moment
    float m4;  // fourth moment
} moments_t;

static moments_t calculate_moments(const int8_t *data, size_t len) {
    moments_t moments = {0};
    if (len == 0) return moments;
    
    moments.mean = calculate_mean_int8(data, len);
    
    for (size_t i = 0; i < len; i++) {
        float diff = data[i] - moments.mean;
        float diff2 = diff * diff;
        moments.m2 += diff2;
        moments.m3 += diff2 * diff;
        moments.m4 += diff2 * diff2;
    }
    
    return moments;
}


static esp_err_t history_buffer_init(history_buffer_t *hb, size_t size) {
    hb->data = (float*)malloc(size * sizeof(float));
    if (!hb->data) {
        return ESP_ERR_NO_MEM;
    }
    hb->size = size;
    hb->index = 0;
    hb->count = 0;
    return ESP_OK;
}

static void history_buffer_add(history_buffer_t *hb, float value) {
    hb->data[hb->index] = value;
    hb->index = (hb->index + 1) % hb->size;
    if (hb->count < hb->size) {
        hb->count++;
    }
}


static void stats_buffer_add(float value) {
    g_state.stats_buffer[g_state.stats_index] = value;
    g_state.stats_index = (g_state.stats_index + 1) % STATS_BUFFER_SIZE;
    if (g_state.stats_count < STATS_BUFFER_SIZE) {
        g_state.stats_count++;
    }
}

static void stats_buffer_analyze(float *min, float *max, float *avg, float *stddev) {
    if (g_state.stats_count == 0) {
        *min = *max = *avg = *stddev = 0.0f;
        return;
    }
    
    *min = g_state.stats_buffer[0];
    *max = g_state.stats_buffer[0];
    *avg = 0.0f;
    
    for (size_t i = 0; i < g_state.stats_count; i++) {
        float val = g_state.stats_buffer[i];
        if (val < *min) *min = val;
        if (val > *max) *max = val;
        *avg += val;
    }
    *avg /= g_state.stats_count;
    
    float variance = 0.0f;
    for (size_t i = 0; i < g_state.stats_count; i++) {
        float diff = g_state.stats_buffer[i] - *avg;
        variance += diff * diff;
    }
    *stddev = sqrtf(variance / g_state.stats_count);
}

static float stats_buffer_percentile(float percentile) {
    if (g_state.stats_count == 0) return 0.0f;
    
    float sorted[STATS_BUFFER_SIZE];
    memcpy(sorted, g_state.stats_buffer, g_state.stats_count * sizeof(float));
    qsort(sorted, g_state.stats_count, sizeof(float), compare_float);
    
    size_t index = (size_t)((percentile / 100.0f) * (g_state.stats_count - 1));
    return sorted[index];
}

// Hampel filter - detects and replaces outliers with the median
static float hampel_filter(const float *window, size_t window_size, float current_value) {
    if (window_size < 3) return current_value;
    
    float sorted[window_size];
    memcpy(sorted, window, window_size * sizeof(float));
    qsort(sorted, window_size, sizeof(float), compare_float);
    
    float median = (window_size % 2 == 1) ? 
                   sorted[window_size / 2] : 
                   (sorted[window_size / 2 - 1] + sorted[window_size / 2]) / 2.0f;
    
    float abs_deviations[window_size];
    for (size_t i = 0; i < window_size; i++) {
        abs_deviations[i] = fabsf(window[i] - median);
    }
    qsort(abs_deviations, window_size, sizeof(float), compare_float);
    
    float mad = (window_size % 2 == 1) ? 
                abs_deviations[window_size / 2] : 
                (abs_deviations[window_size / 2 - 1] + abs_deviations[window_size / 2]) / 2.0f;
    
    // Scale MAD to approximate standard deviation
    float mad_scaled = 1.4826f * mad;
    
    float deviation = fabsf(current_value - median);
    float threshold = control_state.hampel_threshold * mad_scaled;
    
    if (deviation > threshold) {
        return median;
    }
    
    return current_value;
}

// Pre-computed Savitzky-Golay coefficients (window=5, poly=2)
static const float savgol_coeffs_5_2[] = {-0.0857f, 0.3429f, 0.4857f, 0.3429f, -0.0857f};

static float savitzky_golay_filter(const float *window, size_t window_size) {
    if (window_size != SAVGOL_WINDOW_SIZE) {
        float sum = 0.0f;
        for (size_t i = 0; i < window_size; i++) {
            sum += window[i];
        }
        return sum / window_size;
    }
    
    float result = 0.0f;
    for (size_t i = 0; i < window_size; i++) {
        result += window[i] * savgol_coeffs_5_2[i];
    }
    
    return result;
}

// Adaptive normalizer - tracks running mean and variance
typedef struct {
    float running_mean;
    float running_variance;
    size_t sample_count;
    float alpha;
    bool initialized;
} adaptive_normalizer_t;

static adaptive_normalizer_t g_normalizer = {
    .running_mean = 0.0f,
    .running_variance = 1.0f,
    .sample_count = 0,
    .alpha = 0.01f,
    .initialized = false
};

static void adaptive_normalizer_update(adaptive_normalizer_t *norm, float value) {
    if (!norm->initialized) {
        norm->running_mean = value;
        norm->running_variance = 1.0f;
        norm->initialized = true;
        norm->sample_count = 1;
        return;
    }
    
    float delta = value - norm->running_mean;
    norm->running_mean += norm->alpha * delta;
    
    // Welford's algorithm for online variance
    float delta2 = value - norm->running_mean;
    norm->running_variance = (1.0f - norm->alpha) * norm->running_variance + 
                             norm->alpha * delta * delta2;
    
    norm->sample_count++;
}

typedef struct {
    float data[SAVGOL_WINDOW_SIZE];
    size_t index;
    size_t count;
} filter_buffer_t;

static filter_buffer_t g_filter_buffer = {0};

static void filter_buffer_add(filter_buffer_t *fb, float value) {
    fb->data[fb->index] = value;
    fb->index = (fb->index + 1) % SAVGOL_WINDOW_SIZE;
    if (fb->count < SAVGOL_WINDOW_SIZE) {
        fb->count++;
    }
}

static void filter_buffer_get_window(filter_buffer_t *fb, float *window, size_t *size) {
    *size = fb->count;
    if (fb->count < SAVGOL_WINDOW_SIZE) {
        memcpy(window, fb->data, fb->count * sizeof(float));
    } else {
        size_t start = fb->index;
        for (size_t i = 0; i < SAVGOL_WINDOW_SIZE; i++) {
            window[i] = fb->data[(start + i) % SAVGOL_WINDOW_SIZE];
        }
    }
}

static float apply_filters(float raw_value) {
    float filtered_value = raw_value;
    
    filter_buffer_add(&g_filter_buffer, raw_value);
    
    float window[SAVGOL_WINDOW_SIZE];
    size_t window_size;
    filter_buffer_get_window(&g_filter_buffer, window, &window_size);
    
    if (control_state.hampel_filter_enabled && window_size >= 3) {
        filtered_value = hampel_filter(window, window_size, raw_value);
    }
    
    if (control_state.savgol_filter_enabled && window_size == SAVGOL_WINDOW_SIZE) {
        if (control_state.hampel_filter_enabled) {
            window[window_size - 1] = filtered_value;
        }
        filtered_value = savitzky_golay_filter(window, window_size);
    }
    
    adaptive_normalizer_update(&g_normalizer, filtered_value);
    
    return filtered_value;
}

// Feature extraction functions
// These calculate statistical properties of the CSI data

static float calculate_skewness(const int8_t *data, size_t len) {
    if (len < 3) return 0.0f;
    
    moments_t moments = calculate_moments(data, len);
    float m2 = moments.m2 / len;
    float m3 = moments.m3 / len;
    
    float stddev = sqrtf(m2);
    if (stddev < 1e-6f) return 0.0f;
    
    return m3 / (stddev * stddev * stddev);
}

static float calculate_kurtosis(const int8_t *data, size_t len) {
    if (len < 4) return 0.0f;
    
    moments_t moments = calculate_moments(data, len);
    float m2 = moments.m2 / len;
    float m4 = moments.m4 / len;
    
    if (m2 < 1e-6f) return 0.0f;
    
    return (m4 / (m2 * m2)) - 3.0f;
}

static float calculate_entropy(const int8_t *data, size_t len) {
    if (len == 0) return 0.0f;
    
    int histogram[256] = {0};
    
    for (size_t i = 0; i < len; i++) {
        int bin = (int)data[i] + 128;
        histogram[bin]++;
    }
    
    float entropy = 0.0f;
    for (int i = 0; i < 256; i++) {
        if (histogram[i] > 0) {
            float p = (float)histogram[i] / len;
            entropy -= p * log2f(p);
        }
    }
    
    return entropy;
}

static float calculate_iqr(const int8_t *data, size_t len) {
    if (len < 4) return 0.0f;
    
    int8_t *sorted = (int8_t*)malloc(len * sizeof(int8_t));
    if (!sorted) return 0.0f;
    
    memcpy(sorted, data, len * sizeof(int8_t));
    qsort(sorted, len, sizeof(int8_t), compare_int8);
    
    size_t q1_idx = len / 4;
    size_t q3_idx = (3 * len) / 4;
    
    float q1 = sorted[q1_idx];
    float q3 = sorted[q3_idx];
    
    free(sorted);
    
    return q3 - q1;
}

static float calculate_spatial_variance(const int8_t *data, size_t len) {
    if (len < 2) return 0.0f;
    
    return calculate_variance(data, len);
}

static float calculate_spatial_correlation(const int8_t *data, size_t len) {
    if (len < 2) return 0.0f;
    
    float sum_xy = 0.0f;
    float sum_x = 0.0f;
    float sum_y = 0.0f;
    float sum_x2 = 0.0f;
    float sum_y2 = 0.0f;
    size_t n = len - 1;
    
    for (size_t i = 0; i < n; i++) {
        float x = data[i];
        float y = data[i + 1];
        sum_xy += x * y;
        sum_x += x;
        sum_y += y;
        sum_x2 += x * x;
        sum_y2 += y * y;
    }
    
    float numerator = n * sum_xy - sum_x * sum_y;
    float denominator = sqrtf((n * sum_x2 - sum_x * sum_x) * (n * sum_y2 - sum_y * sum_y));
    
    if (denominator < 1e-6f) return 0.0f;
    
    return numerator / denominator;
}

static float calculate_spatial_gradient(const int8_t *data, size_t len) {
    if (len < 2) return 0.0f;
    
    float sum_diff = 0.0f;
    for (size_t i = 0; i < len - 1; i++) {
        sum_diff += fabsf((float)(data[i + 1] - data[i]));
    }
    
    return sum_diff / (len - 1);
}

static float calculate_autocorrelation_lag1(const float *data, size_t len) {
    if (len < 2) return 0.0f;
    
    float mean = calculate_mean(data, len);
    
    float numerator = 0.0f;
    float denominator = 0.0f;
    
    for (size_t i = 0; i < len - 1; i++) {
        float diff_i = data[i] - mean;
        float diff_i1 = data[i + 1] - mean;
        numerator += diff_i * diff_i1;
        denominator += diff_i * diff_i;
    }
    
    if (denominator < 1e-6f) return 0.0f;
    
    return numerator / denominator;
}

static float calculate_zero_crossing_rate(const float *data, size_t len) {
    if (len < 2) return 0.0f;
    
    float mean = calculate_mean(data, len);
    
    int crossings = 0;
    for (size_t i = 0; i < len - 1; i++) {
        if ((data[i] - mean) * (data[i + 1] - mean) < 0) {
            crossings++;
        }
    }
    
    return (float)crossings / (len - 1);
}

static float calculate_peak_rate(const float *data, size_t len) {
    if (len < 3) return 0.0f;
    
    int peaks = 0;
    for (size_t i = 1; i < len - 1; i++) {
        if (data[i] > data[i - 1] && data[i] > data[i + 1]) {
            peaks++;
        }
    }
    
    return (float)peaks / len;
}

static void extract_csi_features(const int8_t *csi_data, size_t csi_len, 
                                 const history_buffer_t *history, 
                                 csi_features_t *features) {
    features->mean = 0.0f;
    for (size_t i = 0; i < csi_len; i++) {
        features->mean += csi_data[i];
    }
    features->mean /= csi_len;
    
    features->variance = calculate_variance(csi_data, csi_len);
    features->skewness = calculate_skewness(csi_data, csi_len);
    features->kurtosis = calculate_kurtosis(csi_data, csi_len);
    features->entropy = calculate_entropy(csi_data, csi_len);
    features->iqr = calculate_iqr(csi_data, csi_len);
    
    features->spatial_variance = calculate_spatial_variance(csi_data, csi_len);
    features->spatial_correlation = calculate_spatial_correlation(csi_data, csi_len);
    features->spatial_gradient = calculate_spatial_gradient(csi_data, csi_len);
    
    if (history->count >= 2) {
        features->autocorr_lag1 = calculate_autocorrelation_lag1(history->data, history->count);
        features->zero_crossing_rate = calculate_zero_crossing_rate(history->data, history->count);
        features->peak_rate = calculate_peak_rate(history->data, history->count);
        
        // Calculate variance at different time scales
        size_t short_len = (history->count < control_state.window_short_size) ? 
                          history->count : control_state.window_short_size;
        size_t medium_len = (history->count < control_state.window_medium_size) ? 
                           history->count : control_state.window_medium_size;
        size_t long_len = history->count;
        
        size_t start_short = history->count - short_len;
        size_t start_medium = history->count - medium_len;
        
        float sum_short = 0.0f, sum_medium = 0.0f, sum_long = 0.0f;
        for (size_t i = 0; i < history->count; i++) {
            if (i >= start_short) sum_short += history->data[i];
            if (i >= start_medium) sum_medium += history->data[i];
            sum_long += history->data[i];
        }
        
        float mean_short = sum_short / short_len;
        float mean_medium = sum_medium / medium_len;
        float mean_long = sum_long / long_len;
        
        features->variance_short = 0.0f;
        features->variance_medium = 0.0f;
        features->variance_long = 0.0f;
        
        for (size_t i = 0; i < history->count; i++) {
            if (i >= start_short) {
                float diff = history->data[i] - mean_short;
                features->variance_short += diff * diff;
            }
            if (i >= start_medium) {
                float diff = history->data[i] - mean_medium;
                features->variance_medium += diff * diff;
            }
            float diff = history->data[i] - mean_long;
            features->variance_long += diff * diff;
        }
        
        features->variance_short /= short_len;
        features->variance_medium /= medium_len;
        features->variance_long /= long_len;
    } else {
        features->autocorr_lag1 = 0.0f;
        features->zero_crossing_rate = 0.0f;
        features->peak_rate = 0.0f;
        features->variance_short = 0.0f;
        features->variance_medium = 0.0f;
        features->variance_long = 0.0f;
    }
}

// Detection scoring - combines multiple features with weights

static float normalize_feature(float value, float min_val, float max_val) {
    if (max_val - min_val < 1e-6f) return 0.0f;
    float normalized = (value - min_val) / (max_val - min_val);
    return fmaxf(0.0f, fminf(1.0f, normalized));
}

static float calculate_detection_score(const csi_features_t *features) {
    // Normalize features to 0-1 range
    // Ranges determined from real-world testing
    
    float norm_variance = normalize_feature(features->variance, 0.0f, 400.0f);
    
    float norm_spatial_gradient = normalize_feature(features->spatial_gradient, 0.0f, 25.0f);
    
    float norm_variance_short = normalize_feature(features->variance_short, 0.0f, 0.1f);
    
    float norm_iqr = normalize_feature(features->iqr, 0.0f, 25.0f);
    
    float score = control_state.weight_variance * norm_variance +
                  control_state.weight_spatial_gradient * norm_spatial_gradient +
                  control_state.weight_variance_short * norm_variance_short +
                  control_state.weight_iqr * norm_iqr;
    
    return score;
}

static void update_detection_state_granular(float score) {
    int64_t now = get_timestamp_sec();
    detection_state_t previous_state = g_state.state;
    
    stats_buffer_add(score);
    
    if (g_state.threshold_high > 0) {
        g_state.confidence = fminf(1.0f, score / g_state.threshold_high);
    } else {
        g_state.confidence = (score > 0) ? 1.0f : 0.0f;
    }
    
    detection_state_t new_state;
    if (score > control_state.intense_movement_threshold) {
        new_state = STATE_INTENSE_MOVEMENT;
    } else if (score > g_state.threshold_high) {
        new_state = STATE_DETECTED;
    } else if (score > control_state.micro_movement_threshold) {
        new_state = STATE_MICRO_MOVEMENT;
    } else {
        new_state = STATE_IDLE;
    }
    
    // Debouncing prevents rapid state changes
    if (new_state > g_state.state) {
        g_state.consecutive_detections++;
        if (g_state.consecutive_detections >= control_state.debounce_count) {
            g_state.state = new_state;
            g_state.last_detection_time = now;
            
            if (previous_state != new_state) {
                const char *state_names[] = {"IDLE", "MICRO", "DETECTED", "INTENSE"};
                ESP_LOGI(TAG, "🚶 STATE CHANGE: %s -> %s (score: %.4f, confidence: %.2f)",
                         state_names[previous_state], state_names[new_state], score, g_state.confidence);
            }
        }
    } else if (new_state < g_state.state) {
        if (now - g_state.last_detection_time > control_state.persistence_timeout) {
            g_state.consecutive_detections = 0;
            
            if (previous_state != new_state) {
                const char *state_names[] = {"IDLE", "MICRO", "DETECTED", "INTENSE"};
                ESP_LOGI(TAG, "✋ STATE CHANGE: %s -> %s", state_names[previous_state], state_names[new_state]);
            }
            
            g_state.state = new_state;
            g_state.confidence = (new_state == STATE_IDLE) ? 0.0f : g_state.confidence;
        }
    }
    
    g_state.detection_score = score;
}


static float normalize_variance(float variance) {
    return tanhf(variance / control_state.variance_scale);
}

static void update_detection_state(float value) {
    int64_t now = get_timestamp_sec();
    detection_state_t previous_state = g_state.state;
    
    g_state.detection_score = value;
    
    stats_buffer_add(value);
    
    if (g_state.threshold_high > 0) {
        g_state.confidence = fminf(1.0f, fmaxf(0.0f, value / g_state.threshold_high));
    } else {
        g_state.confidence = (value > 0) ? 1.0f : 0.0f;
    }
    
    if (value > g_state.threshold_high) {
        g_state.consecutive_detections++;
        g_state.last_detection_time = now;
        
        if (g_state.consecutive_detections >= control_state.debounce_count) {
            g_state.state = STATE_DETECTED;
            
            if (previous_state != STATE_DETECTED) {
                ESP_LOGI(TAG, "🚶 MOVEMENT DETECTED - score: %.4f, threshold: %.4f, confidence: %.2f",
                         value, g_state.threshold_high, g_state.confidence);
            }
        }
    } else if (value < g_state.threshold_low) {
        if (now - g_state.last_detection_time > control_state.persistence_timeout) {
            g_state.consecutive_detections = 0;
            
            if (previous_state == STATE_DETECTED) {
                ESP_LOGI(TAG, "✋ MOVEMENT STOPPED - returning to IDLE state");
            }
            
            g_state.state = STATE_IDLE;
        }
    }
}


static void csi_callback(void *ctx, wifi_csi_info_t *data) {
    g_state.packets_received++;
    
    int8_t *csi_data = data->buf;
    size_t csi_len = data->len;
    
    if (csi_len < 10) {
        return;
    }
    
    float variance = calculate_variance(csi_data, csi_len);
    float normalized = normalize_variance(variance);
    
    float filtered_value = apply_filters(normalized);
    
    history_buffer_add(&g_state.history_buffer, filtered_value);
    
    extract_csi_features(csi_data, csi_len, &g_state.history_buffer, &g_state.current_features);
    
    float detection_score = calculate_detection_score(&g_state.current_features);
    
    if (control_state.granular_states_enabled) {
        update_detection_state_granular(detection_score);
    } else {
        update_detection_state(detection_score);
    }
    
    g_state.packets_processed++;
}

// MQTT handling

static void handle_mqtt_command(const char *data, int data_len);

static void mqtt_event_handler(void *handler_args, esp_event_base_t base,
                               int32_t event_id, void *event_data) {
    esp_mqtt_event_handle_t event = (esp_mqtt_event_handle_t)event_data;
    
    switch (event->event_id) {
        case MQTT_EVENT_CONNECTED:
            ESP_LOGI(TAG, "MQTT connected to broker");
            g_state.mqtt_connected = true;
            esp_mqtt_client_subscribe(g_state.mqtt_client, MQTT_CMD_TOPIC, 0);
            ESP_LOGI(TAG, "Subscribed to command topic: %s", MQTT_CMD_TOPIC);
            break;
            
        case MQTT_EVENT_DISCONNECTED:
            ESP_LOGI(TAG, "MQTT disconnected from broker");
            g_state.mqtt_connected = false;
            break;
            
        case MQTT_EVENT_DATA:
            ESP_LOGI(TAG, "MQTT data received on topic: %.*s", event->topic_len, event->topic);
            if (strncmp(event->topic, MQTT_CMD_TOPIC, event->topic_len) == 0) {
                handle_mqtt_command(event->data, event->data_len);
            }
            break;
            
        case MQTT_EVENT_ERROR:
            ESP_LOGE(TAG, "MQTT error");
            break;
            
        default:
            break;
    }
}

static void mqtt_publish_detection(void) {
    if (!g_state.mqtt_connected) {
        return;
    }
    
    cJSON *root = cJSON_CreateObject();
    if (!root) {
        ESP_LOGE(TAG, "Failed to create JSON object");
        return;
    }
    
    cJSON_AddNumberToObject(root, "movement", (double)g_state.detection_score);
    cJSON_AddNumberToObject(root, "confidence", (double)g_state.confidence);
    
    const char *state_str;
    switch (g_state.state) {
        case STATE_DETECTED:
            state_str = "detected";
            break;
        default:
            state_str = "idle";
            break;
    }
    cJSON_AddStringToObject(root, "state", state_str);
    
    cJSON_AddNumberToObject(root, "timestamp", (double)get_timestamp_sec());
    
    cJSON_AddNumberToObject(root, "threshold", (double)g_state.threshold_high);
    
    char *json_str = cJSON_PrintUnformatted(root);
    if (json_str) {
        int msg_id = esp_mqtt_client_publish(g_state.mqtt_client, MQTT_TOPIC, json_str, 0, 0, 0);
        if (msg_id >= 0) {
            ESP_LOGD(TAG, "Published: %s", json_str);
        } else {
            ESP_LOGW(TAG, "Failed to publish MQTT message");
        }
        
        free(json_str);
    }
    
    cJSON_Delete(root);
}


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
        xEventGroupSetBits(s_wifi_event_group, WIFI_CONNECTED_BIT);
    }
}

static void wifi_init(void) {
    s_wifi_event_group = xEventGroupCreate();
    
    ESP_ERROR_CHECK(esp_netif_init());
    ESP_ERROR_CHECK(esp_event_loop_create_default());
    esp_netif_create_default_wifi_sta();
    
    wifi_init_config_t cfg = WIFI_INIT_CONFIG_DEFAULT();
    ESP_ERROR_CHECK(esp_wifi_init(&cfg));
    
    ESP_ERROR_CHECK(esp_event_handler_register(WIFI_EVENT, ESP_EVENT_ANY_ID, &wifi_event_handler, NULL));
    ESP_ERROR_CHECK(esp_event_handler_register(IP_EVENT, IP_EVENT_STA_GOT_IP, &wifi_event_handler, NULL));
    
    wifi_config_t wifi_config = {0};
    
    strncpy((char *)wifi_config.sta.ssid, WIFI_SSID, sizeof(wifi_config.sta.ssid) - 1);
    strncpy((char *)wifi_config.sta.password, WIFI_PASSWORD, sizeof(wifi_config.sta.password) - 1);
    
    ESP_LOGI(TAG, "WiFi SSID: %s", wifi_config.sta.ssid);
    ESP_LOGI(TAG, "WiFi password length: %d", strlen((char *)wifi_config.sta.password));
    
    ESP_ERROR_CHECK(esp_wifi_set_mode(WIFI_MODE_STA));
    ESP_ERROR_CHECK(esp_wifi_set_config(WIFI_IF_STA, &wifi_config));
    ESP_ERROR_CHECK(esp_wifi_start());
    
    ESP_LOGI(TAG, "WiFi initialization complete");
}


static void csi_init(void) {
    wifi_csi_config_t csi_config = {
        .lltf_en = true,
        .htltf_en = true,
        .stbc_htltf2_en = true,
        .ltf_merge_en = true,
        .channel_filter_en = true,
        .manu_scale = false,
    };
    
    ESP_ERROR_CHECK(esp_wifi_set_csi_config(&csi_config));
    ESP_ERROR_CHECK(esp_wifi_set_csi_rx_cb(csi_callback, NULL));
    ESP_ERROR_CHECK(esp_wifi_set_csi(true));
    
    ESP_LOGI(TAG, "CSI initialized and enabled");
}

// Smart publishing - reduces MQTT traffic by only sending significant changes

static struct {
    float last_published_movement;
    detection_state_t last_published_state;
    int64_t last_publish_time;
    bool initialized;
} publish_state = {0};

static bool should_publish(float current_movement, detection_state_t current_state) {
    if (!SMART_PUBLISHING_ENABLED) {
        return true;
    }
    
    if (!publish_state.initialized) {
        publish_state.initialized = true;
        return true;
    }
    
    int64_t now = get_timestamp_ms();
    int64_t time_since_last = now - publish_state.last_publish_time;
    
    if (current_state != publish_state.last_published_state) {
        ESP_LOGD(TAG, "Publishing: state changed (%d -> %d)", 
                 publish_state.last_published_state, current_state);
        return true;
    }
    
    float delta = fabsf(current_movement - publish_state.last_published_movement);
    if (delta > PUBLISH_DELTA_THRESHOLD) {
        ESP_LOGD(TAG, "Publishing: movement delta %.3f > %.3f", delta, PUBLISH_DELTA_THRESHOLD);
        return true;
    }
    
    if (time_since_last >= (int64_t)(PUBLISH_MAX_INTERVAL * 1000)) {
        ESP_LOGD(TAG, "Publishing: heartbeat (%.1fs since last)", time_since_last / 1000.0f);
        return true;
    }
    
    return false;
}

static void update_publish_state(float movement, detection_state_t state) {
    publish_state.last_published_movement = movement;
    publish_state.last_published_state = state;
    publish_state.last_publish_time = get_timestamp_ms();
}

// MQTT command handling - allows runtime configuration

// Forward declaration
static void mqtt_send_response(const char *message);

// Command handler function pointer type
typedef void (*mqtt_cmd_handler_t)(cJSON *root);

// Individual command handlers
static void cmd_threshold(cJSON *root) {
    cJSON *value = cJSON_GetObjectItem(root, "value");
    if (value && cJSON_IsNumber(value)) {
        float new_threshold = (float)value->valuedouble;
        if (new_threshold > 0.0f && new_threshold < 1.0f) {
            float old_threshold = g_state.threshold_high;
            g_state.threshold_high = new_threshold;
            g_state.threshold_low = new_threshold * HYSTERESIS_RATIO;
            
            char response[128];
            snprintf(response, sizeof(response), 
                     "Threshold updated: %.4f -> %.4f", old_threshold, new_threshold);
            mqtt_send_response(response);
            ESP_LOGI(TAG, "%s", response);
        } else {
            mqtt_send_response("ERROR: Threshold must be between 0.0 and 1.0");
        }
    } else {
        mqtt_send_response("ERROR: Missing or invalid 'value' field");
    }
}

static void cmd_stats(cJSON *root) {
    float min, max, avg, stddev;
    stats_buffer_analyze(&min, &max, &avg, &stddev);
    
    cJSON *response = cJSON_CreateObject();
    cJSON_AddNumberToObject(response, "min", (double)min);
    cJSON_AddNumberToObject(response, "max", (double)max);
    cJSON_AddNumberToObject(response, "avg", (double)avg);
    cJSON_AddNumberToObject(response, "stddev", (double)stddev);
    cJSON_AddNumberToObject(response, "current", (double)g_state.detection_score);
    cJSON_AddNumberToObject(response, "threshold", (double)g_state.threshold_high);
    cJSON_AddNumberToObject(response, "samples", (double)g_state.stats_count);
    
    char *json_str = cJSON_PrintUnformatted(response);
    if (json_str) {
        esp_mqtt_client_publish(g_state.mqtt_client, MQTT_RESPONSE_TOPIC, json_str, 0, 1, 0);
        free(json_str);
    }
    cJSON_Delete(response);
}

static void cmd_info(cJSON *root) {
    cJSON *response = cJSON_CreateObject();
    cJSON_AddNumberToObject(response, "threshold", (double)g_state.threshold_high);
    cJSON_AddNumberToObject(response, "debounce", control_state.debounce_count);
    cJSON_AddNumberToObject(response, "persistence_timeout", control_state.persistence_timeout);
    cJSON_AddNumberToObject(response, "hysteresis_ratio", (double)control_state.hysteresis_ratio);
    cJSON_AddNumberToObject(response, "variance_scale", (double)control_state.variance_scale);
    cJSON_AddBoolToObject(response, "granular_states", control_state.granular_states_enabled);
    
    const char *state_str = g_state.state == STATE_DETECTED ? "DETECTED" : "IDLE";
    cJSON_AddStringToObject(response, "state", state_str);
    cJSON_AddBoolToObject(response, "csi_logs", control_state.csi_logs_enabled);
    
    char *json_str = cJSON_PrintUnformatted(response);
    if (json_str) {
        esp_mqtt_client_publish(g_state.mqtt_client, MQTT_RESPONSE_TOPIC, json_str, 0, 1, 0);
        free(json_str);
    }
    cJSON_Delete(response);
}

static void cmd_logs(cJSON *root) {
    cJSON *enabled = cJSON_GetObjectItem(root, "enabled");
    if (enabled && cJSON_IsBool(enabled)) {
        control_state.csi_logs_enabled = cJSON_IsTrue(enabled);
        char response[64];
        snprintf(response, sizeof(response), "CSI logs %s", 
                 control_state.csi_logs_enabled ? "enabled" : "disabled");
        mqtt_send_response(response);
        ESP_LOGI(TAG, "%s", response);
    } else {
        mqtt_send_response("ERROR: Missing or invalid 'enabled' field");
    }
}

static void cmd_analyze(cJSON *root) {
    if (g_state.stats_count < 50) {
        char response[128];
        snprintf(response, sizeof(response), 
                 "ERROR: Need at least 50 samples (have %lu)", (unsigned long)g_state.stats_count);
        mqtt_send_response(response);
        return;
    }
    
    float min, max, avg, stddev;
    stats_buffer_analyze(&min, &max, &avg, &stddev);
    
    float p25 = stats_buffer_percentile(25.0f);
    float p50 = stats_buffer_percentile(50.0f);
    float p75 = stats_buffer_percentile(75.0f);
    float p95 = stats_buffer_percentile(95.0f);
    
    float recommended = (p50 + p75) / 2.0f;
    
    cJSON *response = cJSON_CreateObject();
    cJSON_AddNumberToObject(response, "min", (double)min);
    cJSON_AddNumberToObject(response, "max", (double)max);
    cJSON_AddNumberToObject(response, "avg", (double)avg);
    cJSON_AddNumberToObject(response, "stddev", (double)stddev);
    cJSON_AddNumberToObject(response, "p25", (double)p25);
    cJSON_AddNumberToObject(response, "p50_median", (double)p50);
    cJSON_AddNumberToObject(response, "p75", (double)p75);
    cJSON_AddNumberToObject(response, "p95", (double)p95);
    cJSON_AddNumberToObject(response, "recommended_threshold", (double)recommended);
    cJSON_AddNumberToObject(response, "current_threshold", (double)g_state.threshold_high);
    
    char *json_str = cJSON_PrintUnformatted(response);
    if (json_str) {
        esp_mqtt_client_publish(g_state.mqtt_client, MQTT_RESPONSE_TOPIC, json_str, 0, 1, 0);
        free(json_str);
    }
    cJSON_Delete(response);
}

static void cmd_persistence(cJSON *root) {
    cJSON *value = cJSON_GetObjectItem(root, "value");
    if (value && cJSON_IsNumber(value)) {
        int new_timeout = (int)value->valueint;
        if (new_timeout >= 1 && new_timeout <= 30) {
            int old_timeout = control_state.persistence_timeout;
            control_state.persistence_timeout = new_timeout;
            
            char response[128];
            snprintf(response, sizeof(response), 
                     "Persistence timeout updated: %d -> %d seconds", old_timeout, new_timeout);
            mqtt_send_response(response);
            ESP_LOGI(TAG, "%s", response);
        } else {
            mqtt_send_response("ERROR: Persistence timeout must be between 1 and 30 seconds");
        }
    } else {
        mqtt_send_response("ERROR: Missing or invalid 'value' field");
    }
}

static void cmd_debounce(cJSON *root) {
    cJSON *value = cJSON_GetObjectItem(root, "value");
    if (value && cJSON_IsNumber(value)) {
        int new_debounce = (int)value->valueint;
        if (new_debounce >= 1 && new_debounce <= 10) {
            uint8_t old_debounce = control_state.debounce_count;
            control_state.debounce_count = (uint8_t)new_debounce;
            
            char response[128];
            snprintf(response, sizeof(response), 
                     "Debounce count updated: %d -> %d", old_debounce, new_debounce);
            mqtt_send_response(response);
            ESP_LOGI(TAG, "%s", response);
        } else {
            mqtt_send_response("ERROR: Debounce count must be between 1 and 10");
        }
    } else {
        mqtt_send_response("ERROR: Missing or invalid 'value' field");
    }
}

static void cmd_hysteresis(cJSON *root) {
    cJSON *value = cJSON_GetObjectItem(root, "value");
    if (value && cJSON_IsNumber(value)) {
        float new_ratio = (float)value->valuedouble;
        if (new_ratio >= 0.1f && new_ratio <= 1.0f) {
            float old_ratio = control_state.hysteresis_ratio;
            control_state.hysteresis_ratio = new_ratio;
            g_state.threshold_low = g_state.threshold_high * new_ratio;
            
            char response[128];
            snprintf(response, sizeof(response), 
                     "Hysteresis ratio updated: %.2f -> %.2f", old_ratio, new_ratio);
            mqtt_send_response(response);
            ESP_LOGI(TAG, "%s", response);
        } else {
            mqtt_send_response("ERROR: Hysteresis ratio must be between 0.1 and 1.0");
        }
    } else {
        mqtt_send_response("ERROR: Missing or invalid 'value' field");
    }
}

static void cmd_variance_scale(cJSON *root) {
    cJSON *value = cJSON_GetObjectItem(root, "value");
    if (value && cJSON_IsNumber(value)) {
        float new_scale = (float)value->valuedouble;
        if (new_scale >= 100.0f && new_scale <= 2000.0f) {
            float old_scale = control_state.variance_scale;
            control_state.variance_scale = new_scale;
            
            char response[128];
            snprintf(response, sizeof(response), 
                     "Variance scale updated: %.0f -> %.0f (sensitivity %s)", 
                     old_scale, new_scale,
                     new_scale < old_scale ? "increased" : "decreased");
            mqtt_send_response(response);
            ESP_LOGI(TAG, "%s", response);
        } else {
            mqtt_send_response("ERROR: Variance scale must be between 100 and 2000");
        }
    } else {
        mqtt_send_response("ERROR: Missing or invalid 'value' field");
    }
}

static void cmd_features(cJSON *root) {
    cJSON *response = cJSON_CreateObject();
    
    cJSON *time_domain = cJSON_CreateObject();
    cJSON_AddNumberToObject(time_domain, "mean", (double)g_state.current_features.mean);
    cJSON_AddNumberToObject(time_domain, "variance", (double)g_state.current_features.variance);
    cJSON_AddNumberToObject(time_domain, "skewness", (double)g_state.current_features.skewness);
    cJSON_AddNumberToObject(time_domain, "kurtosis", (double)g_state.current_features.kurtosis);
    cJSON_AddNumberToObject(time_domain, "entropy", (double)g_state.current_features.entropy);
    cJSON_AddNumberToObject(time_domain, "iqr", (double)g_state.current_features.iqr);
    cJSON_AddItemToObject(response, "time_domain", time_domain);
    
    cJSON *spatial = cJSON_CreateObject();
    cJSON_AddNumberToObject(spatial, "variance", (double)g_state.current_features.spatial_variance);
    cJSON_AddNumberToObject(spatial, "correlation", (double)g_state.current_features.spatial_correlation);
    cJSON_AddNumberToObject(spatial, "gradient", (double)g_state.current_features.spatial_gradient);
    cJSON_AddItemToObject(response, "spatial", spatial);
    
    cJSON *temporal = cJSON_CreateObject();
    cJSON_AddNumberToObject(temporal, "autocorr_lag1", (double)g_state.current_features.autocorr_lag1);
    cJSON_AddNumberToObject(temporal, "zero_crossing_rate", (double)g_state.current_features.zero_crossing_rate);
    cJSON_AddNumberToObject(temporal, "peak_rate", (double)g_state.current_features.peak_rate);
    cJSON_AddItemToObject(response, "temporal", temporal);
    
    cJSON *multi_window = cJSON_CreateObject();
    cJSON_AddNumberToObject(multi_window, "variance_short", (double)g_state.current_features.variance_short);
    cJSON_AddNumberToObject(multi_window, "variance_medium", (double)g_state.current_features.variance_medium);
    cJSON_AddNumberToObject(multi_window, "variance_long", (double)g_state.current_features.variance_long);
    cJSON_AddItemToObject(response, "multi_window", multi_window);
    
    char *json_str = cJSON_PrintUnformatted(response);
    if (json_str) {
        esp_mqtt_client_publish(g_state.mqtt_client, MQTT_RESPONSE_TOPIC, json_str, 0, 1, 0);
        free(json_str);
    }
    cJSON_Delete(response);
}

static void cmd_granular_states(cJSON *root) {
    cJSON *enabled = cJSON_GetObjectItem(root, "enabled");
    if (enabled && cJSON_IsBool(enabled)) {
        control_state.granular_states_enabled = cJSON_IsTrue(enabled);
        char response[64];
        snprintf(response, sizeof(response), "Granular states %s", 
                 control_state.granular_states_enabled ? "enabled" : "disabled");
        mqtt_send_response(response);
        ESP_LOGI(TAG, "%s", response);
    } else {
        mqtt_send_response("ERROR: Missing or invalid 'enabled' field");
    }
}

static void cmd_weights(cJSON *root) {
    cJSON *response = cJSON_CreateObject();
    cJSON_AddNumberToObject(response, "variance", (double)control_state.weight_variance);
    cJSON_AddNumberToObject(response, "spatial_gradient", (double)control_state.weight_spatial_gradient);
    cJSON_AddNumberToObject(response, "variance_short", (double)control_state.weight_variance_short);
    cJSON_AddNumberToObject(response, "iqr", (double)control_state.weight_iqr);
    
    float sum = control_state.weight_variance + control_state.weight_spatial_gradient + 
                control_state.weight_variance_short + control_state.weight_iqr;
    cJSON_AddNumberToObject(response, "sum", (double)sum);
    
    char *json_str = cJSON_PrintUnformatted(response);
    if (json_str) {
        esp_mqtt_client_publish(g_state.mqtt_client, MQTT_RESPONSE_TOPIC, json_str, 0, 1, 0);
        free(json_str);
    }
    cJSON_Delete(response);
}

static void cmd_weight_variance(cJSON *root) {
    cJSON *value = cJSON_GetObjectItem(root, "value");
    if (value && cJSON_IsNumber(value)) {
        float new_weight = (float)value->valuedouble;
        if (new_weight >= 0.0f && new_weight <= 1.0f) {
            control_state.weight_variance = new_weight;
            mqtt_send_response("Weight variance updated");
        } else {
            mqtt_send_response("ERROR: Weight must be between 0.0 and 1.0");
        }
    } else {
        mqtt_send_response("ERROR: Missing or invalid 'value' field");
    }
}

static void cmd_weight_spatial_gradient(cJSON *root) {
    cJSON *value = cJSON_GetObjectItem(root, "value");
    if (value && cJSON_IsNumber(value)) {
        float new_weight = (float)value->valuedouble;
        if (new_weight >= 0.0f && new_weight <= 1.0f) {
            control_state.weight_spatial_gradient = new_weight;
            mqtt_send_response("Weight spatial_gradient updated");
        } else {
            mqtt_send_response("ERROR: Weight must be between 0.0 and 1.0");
        }
    } else {
        mqtt_send_response("ERROR: Missing or invalid 'value' field");
    }
}

static void cmd_weight_variance_short(cJSON *root) {
    cJSON *value = cJSON_GetObjectItem(root, "value");
    if (value && cJSON_IsNumber(value)) {
        float new_weight = (float)value->valuedouble;
        if (new_weight >= 0.0f && new_weight <= 1.0f) {
            control_state.weight_variance_short = new_weight;
            mqtt_send_response("Weight variance_short updated");
        } else {
            mqtt_send_response("ERROR: Weight must be between 0.0 and 1.0");
        }
    } else {
        mqtt_send_response("ERROR: Missing or invalid 'value' field");
    }
}

static void cmd_weight_iqr(cJSON *root) {
    cJSON *value = cJSON_GetObjectItem(root, "value");
    if (value && cJSON_IsNumber(value)) {
        float new_weight = (float)value->valuedouble;
        if (new_weight >= 0.0f && new_weight <= 1.0f) {
            control_state.weight_iqr = new_weight;
            mqtt_send_response("Weight IQR updated");
        } else {
            mqtt_send_response("ERROR: Weight must be between 0.0 and 1.0");
        }
    } else {
        mqtt_send_response("ERROR: Missing or invalid 'value' field");
    }
}

static void cmd_hampel_filter(cJSON *root) {
    cJSON *enabled = cJSON_GetObjectItem(root, "enabled");
    if (enabled && cJSON_IsBool(enabled)) {
        control_state.hampel_filter_enabled = cJSON_IsTrue(enabled);
        char response[64];
        snprintf(response, sizeof(response), "Hampel filter %s", 
                 control_state.hampel_filter_enabled ? "enabled" : "disabled");
        mqtt_send_response(response);
        ESP_LOGI(TAG, "%s", response);
    } else {
        mqtt_send_response("ERROR: Missing or invalid 'enabled' field");
    }
}

static void cmd_hampel_threshold(cJSON *root) {
    cJSON *value = cJSON_GetObjectItem(root, "value");
    if (value && cJSON_IsNumber(value)) {
        float new_threshold = (float)value->valuedouble;
        if (new_threshold >= 1.0f && new_threshold <= 10.0f) {
            float old_threshold = control_state.hampel_threshold;
            control_state.hampel_threshold = new_threshold;
            
            char response[128];
            snprintf(response, sizeof(response), 
                     "Hampel threshold updated: %.1f -> %.1f", old_threshold, new_threshold);
            mqtt_send_response(response);
            ESP_LOGI(TAG, "%s", response);
        } else {
            mqtt_send_response("ERROR: Hampel threshold must be between 1.0 and 10.0");
        }
    } else {
        mqtt_send_response("ERROR: Missing or invalid 'value' field");
    }
}

static void cmd_savgol_filter(cJSON *root) {
    cJSON *enabled = cJSON_GetObjectItem(root, "enabled");
    if (enabled && cJSON_IsBool(enabled)) {
        control_state.savgol_filter_enabled = cJSON_IsTrue(enabled);
        char response[64];
        snprintf(response, sizeof(response), "Savitzky-Golay filter %s", 
                 control_state.savgol_filter_enabled ? "enabled" : "disabled");
        mqtt_send_response(response);
        ESP_LOGI(TAG, "%s", response);
    } else {
        mqtt_send_response("ERROR: Missing or invalid 'enabled' field");
    }
}

static void cmd_filters(cJSON *root) {
    cJSON *response = cJSON_CreateObject();
    cJSON_AddBoolToObject(response, "hampel_enabled", control_state.hampel_filter_enabled);
    cJSON_AddNumberToObject(response, "hampel_threshold", (double)control_state.hampel_threshold);
    cJSON_AddBoolToObject(response, "savgol_enabled", control_state.savgol_filter_enabled);
    cJSON_AddNumberToObject(response, "savgol_window_size", control_state.savgol_window_size);
    cJSON_AddNumberToObject(response, "normalizer_mean", (double)g_normalizer.running_mean);
    cJSON_AddNumberToObject(response, "normalizer_variance", (double)g_normalizer.running_variance);
    cJSON_AddNumberToObject(response, "normalizer_samples", (double)g_normalizer.sample_count);
    
    char *json_str = cJSON_PrintUnformatted(response);
    if (json_str) {
        esp_mqtt_client_publish(g_state.mqtt_client, MQTT_RESPONSE_TOPIC, json_str, 0, 1, 0);
        free(json_str);
    }
    cJSON_Delete(response);
}

// Command dispatch table
typedef struct {
    const char *name;
    mqtt_cmd_handler_t handler;
} mqtt_command_entry_t;

static const mqtt_command_entry_t command_table[] = {
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
    {"granular_states", cmd_granular_states},
    {"weights", cmd_weights},
    {"weight_variance", cmd_weight_variance},
    {"weight_spatial_gradient", cmd_weight_spatial_gradient},
    {"weight_variance_short", cmd_weight_variance_short},
    {"weight_iqr", cmd_weight_iqr},
    {"hampel_filter", cmd_hampel_filter},
    {"hampel_threshold", cmd_hampel_threshold},
    {"savgol_filter", cmd_savgol_filter},
    {"filters", cmd_filters},
    {NULL, NULL}
};

static void mqtt_send_response(const char *message) {
    if (!g_state.mqtt_connected) {
        ESP_LOGW(TAG, "Cannot send response: MQTT not connected");
        return;
    }
    
    ESP_LOGI(TAG, "Preparing response: %s", message);
    
    cJSON *root = cJSON_CreateObject();
    if (!root) {
        ESP_LOGE(TAG, "Failed to create JSON response");
        return;
    }
    
    cJSON_AddStringToObject(root, "response", message);
    cJSON_AddNumberToObject(root, "timestamp", (double)get_timestamp_sec());
    
    char *json_str = cJSON_PrintUnformatted(root);
    if (json_str) {
        ESP_LOGI(TAG, "Publishing response to %s: %s", MQTT_RESPONSE_TOPIC, json_str);
        int msg_id = esp_mqtt_client_publish(g_state.mqtt_client, MQTT_RESPONSE_TOPIC, json_str, 0, 1, 0);
        if (msg_id >= 0) {
            ESP_LOGI(TAG, "Response published successfully, msg_id=%d", msg_id);
        } else {
            ESP_LOGE(TAG, "Failed to publish response, error=%d", msg_id);
        }
        free(json_str);
    } else {
        ESP_LOGE(TAG, "Failed to create JSON string");
    }
    
    cJSON_Delete(root);
}

static void handle_mqtt_command(const char *data, int data_len) {
    cJSON *root = cJSON_ParseWithLength(data, data_len);
    if (!root) {
        ESP_LOGW(TAG, "Failed to parse MQTT command JSON");
        mqtt_send_response("ERROR: Invalid JSON");
        return;
    }
    
    cJSON *cmd = cJSON_GetObjectItem(root, "cmd");
    if (!cmd || !cJSON_IsString(cmd)) {
        cJSON_Delete(root);
        mqtt_send_response("ERROR: Missing 'cmd' field");
        return;
    }
    
    const char *command = cmd->valuestring;
    ESP_LOGI(TAG, "Received MQTT command: %s", command);
    
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
        mqtt_send_response(response);
    }
    
    cJSON_Delete(root);
}


static void mqtt_publish_task(void *pvParameters) {
    TickType_t last_wake_time = xTaskGetTickCount();
    const TickType_t publish_period = pdMS_TO_TICKS((uint32_t)(PUBLISH_INTERVAL * 1000));
    
    uint32_t publish_count = 0;
    uint32_t skip_count = 0;
    int64_t last_csi_log_time = 0;
    
    while (1) {
        vTaskDelayUntil(&last_wake_time, publish_period);
        
        int64_t now = get_timestamp_sec();
        if (control_state.csi_logs_enabled && (now - last_csi_log_time >= LOG_CSI_VALUES_INTERVAL)) {
            const char *state_names[] = {"IDLE", "MICRO", "DETECTED", "INTENSE"};
            const char *state_str = (g_state.state < 4) ? state_names[g_state.state] : "UNKNOWN";
            
            ESP_LOGI(TAG, "📊 CSI: movement=%.4f, threshold=%.4f, state=%s, confidence=%.2f",
                     g_state.detection_score, g_state.threshold_high,
                     state_str, g_state.confidence);
            last_csi_log_time = now;
        }
        
        bool do_publish = false;
        
        if (g_state.confidence >= CONFIDENCE_THRESHOLD) {
            if (should_publish(g_state.detection_score, g_state.state)) {
                do_publish = true;
            } else {
                skip_count++;
            }
        }
        
        if (do_publish) {
            mqtt_publish_detection();
            update_publish_state(g_state.detection_score, g_state.state);
            publish_count++;
        }
        
        if (g_state.packets_received > 0 && g_state.packets_received % STATS_LOG_INTERVAL == 0) {
            float success_rate = ((float)g_state.packets_processed / g_state.packets_received) * 100.0f;
            ESP_LOGI(TAG, "Stats: %lu packets received, %lu processed (%.1f%% success)",
                     g_state.packets_received, g_state.packets_processed, success_rate);
            
            if (SMART_PUBLISHING_ENABLED && (publish_count + skip_count) > 0) {
                float reduction = (skip_count * 100.0f) / (publish_count + skip_count);
                ESP_LOGI(TAG, "Smart Publishing: %lu published, %lu skipped (%.1f%% reduction)",
                         publish_count, skip_count, reduction);
            }
        }
    }
}


void app_main(void) {
    ESP_LOGI(TAG, "=== ESPectre ESP32-S3 Starting ===");
    ESP_LOGI(TAG, "Version: 1.0.0");
    
    esp_err_t ret = nvs_flash_init();
    if (ret == ESP_ERR_NVS_NO_FREE_PAGES || ret == ESP_ERR_NVS_NEW_VERSION_FOUND) {
        ESP_ERROR_CHECK(nvs_flash_erase());
        ret = nvs_flash_init();
    }
    ESP_ERROR_CHECK(ret);
    
    memset(&g_state, 0, sizeof(g_state));
    
    if (history_buffer_init(&g_state.history_buffer, BUFFER_SIZE) != ESP_OK) {
        ESP_LOGE(TAG, "Failed to initialize history buffer");
        return;
    }
    
    g_state.threshold_high = DEFAULT_THRESHOLD;
    g_state.threshold_low = DEFAULT_THRESHOLD * HYSTERESIS_RATIO;
    
    ESP_LOGI(TAG, "Advanced feature extraction enabled, threshold=%.4f", g_state.threshold_high);
    
    wifi_init();
    
    ESP_LOGI(TAG, "Waiting for WiFi connection...");
    xEventGroupWaitBits(s_wifi_event_group, WIFI_CONNECTED_BIT, pdFALSE, pdTRUE, portMAX_DELAY);
    
    esp_mqtt_client_config_t mqtt_cfg = {
        .broker.address.uri = MQTT_BROKER_URI,
    };
    
    #ifdef CONFIG_MQTT_USERNAME
    if (strlen(MQTT_USERNAME) > 0) {
        mqtt_cfg.credentials.username = MQTT_USERNAME;
    }
    #endif
    
    #ifdef CONFIG_MQTT_PASSWORD
    if (strlen(MQTT_PASSWORD) > 0) {
        mqtt_cfg.credentials.authentication.password = MQTT_PASSWORD;
    }
    #endif
    
    g_state.mqtt_client = esp_mqtt_client_init(&mqtt_cfg);
    if (!g_state.mqtt_client) {
        ESP_LOGE(TAG, "Failed to initialize MQTT client");
        return;
    }
    
    ESP_ERROR_CHECK(esp_mqtt_client_register_event(g_state.mqtt_client, ESP_EVENT_ANY_ID, 
                                                   mqtt_event_handler, NULL));
    ESP_ERROR_CHECK(esp_mqtt_client_start(g_state.mqtt_client));
    
    ESP_LOGI(TAG, "MQTT client started");
    
    csi_init();
    
    xTaskCreate(mqtt_publish_task, "mqtt_pub", 4096, NULL, 5, NULL);
    
    ESP_LOGI(TAG, "=== ESPectre initialization complete ===");
    ESP_LOGI(TAG, "Ready to detect movement!");
}
