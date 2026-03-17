#ifndef CONFIG_H
#define CONFIG_H

// WiFi Configuration
#define WIFI_SSID "YourNetworkName"
#define WIFI_PASSWORD "YourPassword"

// Detection Parameters
#define WINDOW_SIZE 50
#define DEFAULT_THRESHOLD 1.0f

// Hardware Configuration
#define NUM_SUBCARRIERS 64
#define BAND_SIZE 12

// TFT Pins for Adafruit Feather ESP32-S3 Reverse TFT
// These should be defined by the board support package
// Only define if not already defined
#ifndef TFT_CS
  #define TFT_CS        7
#endif
#ifndef TFT_DC
  #define TFT_DC        39
#endif
#ifndef TFT_RST
  #define TFT_RST       40
#endif
#ifndef TFT_BACKLITE
  #define TFT_BACKLITE  45
#endif
#ifndef NEOPIXEL_PIN
  #define NEOPIXEL_PIN  33
#endif

// Traffic Generator
#define TRAFFIC_RATE_PPS 100  // Packets per second

#endif
