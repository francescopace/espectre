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
#define TFT_CS        7
#define TFT_DC        39
#define TFT_RST       40
#define TFT_BACKLIGHT 45
#define NEOPIXEL_PIN  33

// Traffic Generator
#define TRAFFIC_RATE_PPS 100  // Packets per second

#endif
