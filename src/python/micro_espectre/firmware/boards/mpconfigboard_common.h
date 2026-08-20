// SPDX-License-Identifier: GPL-3.0-only
// Commercial licensing available under separate agreement; see LICENSING.md.

// Micro-ESPectre uses Wi-Fi only. Keep mDNS queries for `.local` MQTT
// brokers, but do not advertise an mDNS responder from the sensing node.
#define MICROPY_PY_BLUETOOTH (0)
#define MICROPY_PY_ESPNOW (0)
#define MICROPY_PY_ASYNCIO (0)
#define MICROPY_HW_ENABLE_MDNS_QUERIES (1)
#define MICROPY_HW_ENABLE_MDNS_RESPONDER (0)

// Keep only the generic modules used by the filesystem-deployed sensing
// runtime. These are public MicroPython build options, not port patches.
#define MICROPY_PY_ARRAY (0)
#define MICROPY_PY_BINASCII (0)
#define MICROPY_PY_BUILTINS_HELP (0)
#define MICROPY_PY_CMATH (0)
#define MICROPY_PY_COLLECTIONS (0)
#define MICROPY_PY_CRYPTOLIB (0)
#define MICROPY_PY_DEFLATE (0)
#define MICROPY_PY_FRAMEBUF (0)
#define MICROPY_PY_HASHLIB (1)
#define MICROPY_PY_HASHLIB_MD5 (0)
#define MICROPY_PY_HASHLIB_SHA1 (0)
#define MICROPY_PY_HASHLIB_SHA256 (1)
#define MICROPY_PY_HEAPQ (0)
#define MICROPY_PY_JSON_SEPARATORS (0)
#define MICROPY_PY_MATH_FACTORIAL (0)
#define MICROPY_PY_MATH_ISCLOSE (0)
#define MICROPY_PY_MATH_SPECIAL_FUNCTIONS (0)
#define MICROPY_PY_MICROPYTHON_HEAP_LOCKED (0)
#define MICROPY_PY_MICROPYTHON_MEM_INFO (0)
#define MICROPY_PY_MICROPYTHON_RINGIO (0)
#define MICROPY_PY_MICROPYTHON_STACK_USE (0)
#define MICROPY_PY_OS_STATVFS (0)
#define MICROPY_PY_PLATFORM (0)
#define MICROPY_PY_RANDOM (0)
#define MICROPY_PY_RE (0)
#define MICROPY_PY_SELECT (0)
#define MICROPY_PY_SYS_ARGV (0)
#define MICROPY_PY_SYS_MAXSIZE (0)
#define MICROPY_PY_SYS_PS1_PS2 (0)
#define MICROPY_PY_UCTYPES (0)

// Disable optional ESP32 peripherals exposed through configurable port gates.
// The application needs machine.reset(), but none of these device classes.
#define MICROPY_HW_ENABLE_SDCARD (0)
#define MICROPY_HW_RTC_USER_MEM_MAX (0)
#define MICROPY_PY_ESP32_PCNT (0)
#define MICROPY_PY_MACHINE_DAC (0)
#define MICROPY_PY_MACHINE_I2S (0)
#define MICROPY_PY_MACHINE_MEM_BACKUP (0)
#define MICROPY_PY_MACHINE_SDCARD (0)
#define MICROPY_PY_NETWORK_LAN (0)
