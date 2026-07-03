#include "firmware_version.h"

#if __has_include("esp_app_desc.h")
#include "esp_app_desc.h"
#endif

namespace esphome {
namespace espectre {

const char *espectre_firmware_version() {
#ifdef APP_PROJECT_VER
  return APP_PROJECT_VER;
#elif __has_include("esp_app_desc.h")
  const esp_app_desc_t *app_desc = esp_app_get_description();
  return (app_desc != nullptr && app_desc->version[0] != '\0') ? app_desc->version : "unknown";
#else
  return "unknown";
#endif
}

}  // namespace espectre
}  // namespace esphome
