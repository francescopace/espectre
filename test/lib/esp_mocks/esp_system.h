#ifndef ESP_SYSTEM_H
#define ESP_SYSTEM_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Mock heap size - simulates ESP32 with ~200KB free heap
static inline size_t esp_get_free_heap_size(void) {
  return 200000; // Return constant value for testing
}

// Mock minimum free heap size
static inline size_t esp_get_minimum_free_heap_size(void) { return 180000; }

// Mock heap info structure
typedef struct {
  size_t total_free_bytes;
  size_t total_allocated_bytes;
  size_t largest_free_block;
  size_t minimum_free_bytes;
  size_t allocated_blocks;
  size_t free_blocks;
  size_t total_blocks;
} multi_heap_info_t;

// Mock heap info function
static inline void esp_get_heap_info(multi_heap_info_t *info) {
  if (info) {
    info->total_free_bytes = 200000;
    info->total_allocated_bytes = 100000;
    info->largest_free_block = 150000;
    info->minimum_free_bytes = 180000;
    info->allocated_blocks = 50;
    info->free_blocks = 10;
    info->total_blocks = 60;
  }
}

#ifdef __cplusplus
}
#endif

#endif // ESP_SYSTEM_H
