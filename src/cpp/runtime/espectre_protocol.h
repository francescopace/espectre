/*
 * ESPectre - ESPectre Protocol
 *
 * Shared device, command, and OTA protocol types used by frontend
 * transports.
 *
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * SPDX-License-Identifier: GPL-3.0-only
 * Commercial licensing available under separate agreement; see LICENSING.md.
 */
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

#include "runtime_snapshot.h"

/**
 * @file espectre_protocol.h
 * @brief Wire types and payload builders for the ESPectre Protocol.
 *
 * The protocol is the contract between a device and whatever consumes it:
 * MQTT topics, JSON payloads, BLE control commands, and the OTA status model.
 * It is specified in `docs/ESPECTRE_PROTOCOL.md`; this header is the C++ view
 * of that specification.
 *
 * Use it whenever your integration should stay interoperable with the shipped
 * clients — the CLI, Home Assistant discovery, and the web BLE client all
 * speak it. The builders take a `RuntimeSnapshot` and return a serialized
 * payload, so your transport only moves bytes and never formats them.
 *
 * The parsers never throw: they validate and report failure through an out
 * parameter. They do not all roll back cleanly on rejection, so parse into a
 * copy of your live configuration and commit it only on success, which is what
 * the shipped frontends do.
 */

namespace espectre {

struct RuntimeDiagnosticsSample;

/** Protocol version reported in payloads. Bumped on a wire-format change. */
inline constexpr const char *ESPECTRE_PROTOCOL_VERSION = "1.0";
/** Default MQTT topic root. Override per device with `EspectreDeviceConfig::topic_prefix`. */
inline constexpr const char *ESPECTRE_TOPIC_PREFIX = "espectre/v1/devices";
/** Official tagged GitHub Release OTA channel. */
inline constexpr const char *ESPECTRE_OTA_CHANNEL_RELEASE = "release";
/** Rolling `main` OTA channel. Fetches GitHub Releases tag `snapshot`. */
inline constexpr const char *ESPECTRE_OTA_CHANNEL_PREVIEW = "preview";
/** Rolling `develop` OTA channel. Fetches GitHub Releases tag `snapshot-dev`. */
inline constexpr const char *ESPECTRE_OTA_CHANNEL_DEVELOP = "develop";
/** GitHub Releases tag for the `preview` OTA channel. Distinct from branch `main`. */
inline constexpr const char *ESPECTRE_OTA_RELEASE_TAG_PREVIEW = "snapshot";
/** GitHub Releases tag for the `develop` OTA channel. Distinct from branch `develop`. */
inline constexpr const char *ESPECTRE_OTA_RELEASE_TAG_DEVELOP = "snapshot-dev";
/** Sentinel meaning "derive the device id from the Wi-Fi MAC". */
inline constexpr uint64_t ESPECTRE_DEFAULT_DEVICE_ID = 0U;
/** Empty label, meaning the device id is used as the display name. */
inline constexpr const char *ESPECTRE_DEFAULT_DEVICE_LABEL = "";

/**
 * Device identity and broker settings.
 *
 * Frontends persist this so a device keeps its identity and connection across
 * reboots and BLE reprovisioning.
 */
struct EspectreDeviceConfig {
  /** Stable device identity. Zero means derive it from the Wi-Fi MAC. */
  uint64_t device_id{ESPECTRE_DEFAULT_DEVICE_ID};
  /** Human-readable name. Empty falls back to the formatted device id. */
  std::string device_label{ESPECTRE_DEFAULT_DEVICE_LABEL};
  /** Broker hostname or IP. Empty disables MQTT: `IMqttTransport::setup()` fails. */
  std::string mqtt_host;
  uint16_t mqtt_port{1883};
  /** Broker credentials. Leave empty for anonymous brokers. */
  std::string mqtt_username;
  std::string mqtt_password;
  /** Topic root. Change it only if you also change every consumer. */
  std::string topic_prefix{ESPECTRE_TOPIC_PREFIX};
};

/** Link details reported in the device info payload. */
struct EspectreNetworkInfo {
  std::string ip_address;
  std::string mac_address;
  /** Wi-Fi channel in use. Zero when unknown. */
  uint8_t channel{0U};
};

/**
 * What the device advertises about itself.
 *
 * The `supports_*` flags are how a client learns which controls to offer.
 * Derive them from `RuntimeCapabilities` rather than hardcoding, and let
 * `normalize_protocol_device_info()` fill the gaps from a snapshot.
 * MQTT clients that need command names should request `commands` rather than
 * reconstructing the list from these flags.
 */
struct EspectreDeviceInfo {
  /** Frontend name, for example `"native"`, `"matter"`, or your own. */
  std::string frontend{"ble"};
  /** Application version, normally `espectre_firmware_version()`. */
  std::string firmware_version{"unknown"};
  /** Chip target, normally `CONFIG_IDF_TARGET`. */
  std::string chip{"unknown"};
  /** Active detector. Left empty, it is filled from the snapshot. */
  std::string detector;
  bool supports_info{true};
  bool supports_stats{false};
  bool supports_runtime_threshold{false};
  bool supports_runtime_motion_hits{false};
  bool supports_runtime_detector{false};
  bool supports_manual_recalibration{false};
  bool supports_traffic_control{false};
  bool supports_ota{false};
  /** MQTT `set_ble` is honored. Native setup/recovery uses this; other frontends leave it false. */
  bool supports_ble{false};
  /**
   * CSI traffic ownership mode: `"internal"`, `"external"`, `"pacing"`, or `"disabled"`.
   *
   * Omitted from `info` when empty. Sensing MQTT frontends that own traffic control fill it.
   */
  std::string csi_traffic_mode;
  /**
   * Internal traffic generator mode, such as `"ping"` or `"dns"`.
   *
   * Omitted from `info` when empty.
   */
  std::string traffic_mode;
  /**
   * Internal traffic generator and temporal-grid target rate, in packets per second.
   *
   * Omitted from `info` when zero.
   */
  uint32_t csi_target_pps{0U};
  EspectreNetworkInfo network{};
};

/**
 * A parsed control command.
 *
 * Fields are optional by design: each `has_*` flag says whether the peer
 * actually sent that field, so an unset value is never confused with a zero
 * the caller meant. Apply only the flagged fields.
 */
struct EspectreCommand {
  /** Correlation id echoed in the result payload. May be empty. */
  std::string command_id;
  /** Command verb, for example `"set_threshold"` or `"recalibrate"`. */
  std::string command;
  float threshold{0.0f};
  bool has_threshold{false};
  uint8_t motion_on_hits{0U};
  uint8_t motion_off_hits{0U};
  bool has_motion_hits{false};
  std::string csi_traffic_mode;
  bool has_csi_traffic_mode{false};
  std::string traffic_generator_mode;
  bool has_traffic_generator_mode{false};
  std::string detector;
  bool has_detector{false};
  /** BLE radio request for Native `set_ble`: `"on"` or `"off"`. */
  std::string ble;
  bool has_ble{false};
  /**
   * OTA release channel for `ota_check` and `ota_start`: `"release"`, `"preview"`,
   * or `"develop"`. Empty with `has_ota_channel` false means the firmware default.
   */
  std::string ota_channel;
  bool has_ota_channel{false};
};

/**
 * OTA progress, as reported to clients.
 *
 * A check runs `IDLE` -> `CHECKING` -> `UPDATE_AVAILABLE` or `UP_TO_DATE`.
 * An update continues `DOWNLOADING` -> `APPLYING` -> `REBOOT_SCHEDULED`.
 * `ERROR` is terminal for the attempt and carries the reason in
 * `EspectreOtaStatus::message`.
 */
enum class EspectreOtaState : uint8_t {
  IDLE = 0,
  CHECKING,
  UPDATE_AVAILABLE,
  UP_TO_DATE,
  DOWNLOADING,
  APPLYING,
  REBOOT_SCHEDULED,
  ERROR,
};

/** Full OTA status: state, the versions involved, and the resolved URLs. */
struct EspectreOtaStatus {
  EspectreOtaState state{EspectreOtaState::IDLE};
  std::string current_version{"unknown"};
  std::string target_version;
  std::string manifest_url;
  std::string image_url;
  std::string message;
  /** Resolved OTA channel for the current attempt. Empty when unused. */
  std::string channel;
  bool busy{false};
  bool update_available{false};
};

/**
 * @name Device identity
 * Resolve, format, and parse the identity a device presents on the wire.
 * @{
 */

/** Format a device id in its canonical wire form. */
std::string format_espectre_device_id(uint64_t device_id);
/**
 * Parse a device id from its wire form.
 *
 * @param value Formatted device id, as produced by `format_espectre_device_id()`.
 * @param device_id Written only when parsing succeeds.
 * @return false on a malformed value, leaving the output untouched.
 */
bool parse_espectre_device_id(const std::string &value, uint64_t *device_id);
/** Derive a stable device id from a Wi-Fi MAC. Returns zero for a bad address. */
uint64_t espectre_device_id_from_mac(const uint8_t *mac, size_t mac_len);
/** Conventional advertised name, so devices stay identifiable in a BLE scan. */
std::string espectre_device_name(uint64_t device_id, const char *chip = nullptr);
/** The id actually in use: the configured one, or the MAC-derived fallback. */
uint64_t espectre_effective_device_id_u64(const EspectreDeviceConfig &config);
/** `espectre_effective_device_id_u64()` in wire form. */
std::string espectre_effective_device_id(const EspectreDeviceConfig &config);
/** The configured label, or the effective device id when no label is set. */
std::string espectre_effective_device_label(const EspectreDeviceConfig &config);
/**
 * Fill in the parts of a device info block the frontend did not set.
 *
 * Takes the detector from `snapshot`, and `default_frontend` / `default_chip`
 * where the caller left the field empty, so each frontend only states what is
 * genuinely its own.
 *
 * @param info What the frontend knows about itself.
 * @param snapshot Source of the detector name. May be `nullptr` when no
 *        snapshot exists yet.
 * @param supports_ota Whether this frontend exposes firmware updates.
 * @param default_frontend Frontend name used when `info.frontend` is empty.
 * @param default_chip Chip name used when `info.chip` is empty.
 * @return A copy of `info` with the gaps filled.
 */
EspectreDeviceInfo normalize_protocol_device_info(const EspectreDeviceInfo &info,
                                                  const RuntimeSnapshot *snapshot,
                                                  bool supports_ota,
                                                  const char *default_frontend,
                                                  const char *default_chip = nullptr);
/** Erase broker settings while preserving identity, for a config reset. */
void clear_espectre_mqtt_config(EspectreDeviceConfig *config);

/** @} */

/**
 * @name Topics and payloads
 * Build the wire representation from runtime state. Each returns a complete
 * payload ready to hand to a transport.
 * @{
 */

/** Build a full topic from this device's prefix and a trailing segment. */
std::string espectre_topic(const EspectreDeviceConfig &config, const char *suffix);
/** Availability payload. Publish it retained so late subscribers see it. */
std::string espectre_status_payload(const EspectreDeviceConfig &config, bool online, uint32_t timestamp_ms);
/** Device description, supported controls, and optional CSI traffic settings. Publish retained on connect. */
std::string espectre_info_payload(const EspectreDeviceConfig &config, const EspectreDeviceInfo &info);
/**
 * MQTT command catalog for the current frontend.
 *
 * Published on `commands/catalog` in response to `commands`. The list is
 * derived from the same `supports_*` flags carried by `info`.
 */
std::string espectre_commands_payload(const EspectreDeviceConfig &config, const EspectreDeviceInfo &info);
/** Motion state, metric, and threshold. The payload behind every motion update. */
std::string espectre_telemetry_payload(const EspectreDeviceConfig &config,
                                    const RuntimeSnapshot &snapshot,
                                    uint32_t timestamp_ms,
                                    uint32_t uptime_s,
                                    const char *frontend);
/**
 * Health counters plus optional rate and link diagnostics.
 *
 * `diagnostics` carries CSI and link rates from `RuntimeDiagnosticsSampler`.
 * Pass `nullptr` only for a frontend that does not expose extended diagnostics.
 */
std::string espectre_stats_payload(const EspectreDeviceConfig &config,
                                const RuntimeSnapshot &snapshot,
                                uint32_t timestamp_ms,
                                uint32_t uptime_s,
                                float free_memory_kb,
                                float loop_time_ms,
                                const RuntimeDiagnosticsSample *diagnostics = nullptr);
/**
 * Acknowledge a command, echoing its `command_id`.
 *
 * Publish one for every command you parse, accepted or not; clients correlate
 * on the id and otherwise cannot tell rejection from packet loss.
 */
std::string espectre_command_result_payload(const EspectreDeviceConfig &config,
                                         const EspectreCommand &command,
                                         bool accepted,
                                         const char *message);
/** OTA progress payload, for each `IOtaService` status callback worth publishing. */
std::string espectre_ota_status_payload(const EspectreDeviceConfig &config,
                                    const EspectreOtaStatus &status,
                                    uint32_t timestamp_ms);

/** @} */

/**
 * @name Command parsing
 * Turn received bytes into validated values.
 * @{
 */

/**
 * Parse a JSON command payload from the MQTT command topic.
 *
 * @param payload Raw message body as received.
 * @param command Populated only on success. Check the `has_*` flags to see
 *        which fields the peer actually sent.
 * @param error Receives a human-readable reason on failure. May be `nullptr`.
 * @return false on malformed input or an unknown command.
 */
bool parse_espectre_command(const std::string &payload, EspectreCommand *command, std::string *error);
/**
 * Parse a BLE ASCII OTA control command.
 *
 * Accepts `OTA_STATUS`, `OTA_CHECK`, `OTA_START`, and the optional
 * `OTA_CHECK:channel=preview` / `OTA_START:channel=develop` forms. Other
 * suffixes and unknown channels are rejected.
 *
 * @param command Full BLE control string.
 * @param parsed Populated only on success.
 * @param error Receives a human-readable reason on failure. May be `nullptr`.
 */
bool parse_espectre_ble_ota_command(const std::string &command, EspectreCommand *parsed, std::string *error);
/**
 * Whether `channel` is a published OTA channel name.
 *
 * Accepted values are `release`, `preview`, and `develop`. Empty is not
 * accepted here; omit the field to keep the firmware default.
 */
bool espectre_ota_channel_accepted(const std::string &channel);
/**
 * Built-in GitHub Releases manifest URL for a frontend, chip, and channel.
 *
 * `release` uses `/releases/latest/download/`. `preview` uses tag
 * `ESPECTRE_OTA_RELEASE_TAG_PREVIEW` (`snapshot`). `develop` uses tag
 * `ESPECTRE_OTA_RELEASE_TAG_DEVELOP` (`snapshot-dev`).
 *
 * @return Empty when `frontend`, `chip`, or `channel` is not a published value.
 */
std::string espectre_ota_manifest_url(const char *frontend, const char *chip, const std::string &channel);
/**
 * Parse a `SET_DEVICE_CONFIG:` command from the BLE control characteristic.
 *
 * Carries one `key=value` pair, applied in place. A rejected command writes
 * nothing.
 *
 * @param command Full command string, including the `SET_DEVICE_CONFIG:` prefix.
 * @param config Updated in place on success.
 * @param error Receives a human-readable reason on failure. May be `nullptr`.
 */
bool parse_espectre_config_command(const std::string &command, EspectreDeviceConfig *config, std::string *error);
/**
 * Parse a `SET_MQTT_CONFIG:` command, carrying the broker settings.
 *
 * Fields are applied as they are read, so a command rejected part-way through
 * leaves `config` partially updated. Pass a copy and commit only on success.
 * `host` and `port` are required; the rest keep their previous values.
 */
bool parse_espectre_mqtt_config_command(const std::string &command, EspectreDeviceConfig *config, std::string *error);

/** @} */

}  // namespace espectre
