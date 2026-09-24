/*
 * ESPectre - ESPectre Protocol
 *
 * Shared device and extensible command protocol types used by frontend
 * transports.
 *
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * SPDX-License-Identifier: GPL-3.0-only
 * Commercial licensing available under separate agreement; see LICENSING.md.
 */
#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "runtime_snapshot.h"
#include "protocol_json.h"

/**
 * @file espectre_protocol.h
 * @brief Wire types and payload builders for the ESPectre Protocol.
 *
 * The protocol is the contract between a device and whatever consumes it:
 * MQTT topics, Direct HTTP messages, JSON payloads, and frontend extensions.
 * It is specified in
 * [API.md](https://github.com/francescopace/espectre/blob/main/docs/API.md);
 * this header is the C++ view of that specification.
 *
 * Use it whenever your integration should stay interoperable with the shipped
 * clients — the CLI, Home Assistant discovery, and the web portal all
 * speak it. The builders take a `RuntimeSnapshot` and return a serialized
 * payload, so your transport only moves bytes and never formats them.
 *
 * The parsers never throw: they validate and report failure through an out
 * parameter. On failure the command parsers reset their output, keeping at
 * most the identifiers needed for the result payload. The device
 * configuration parsers leave their output unchanged.
 */

namespace espectre {

/** Matter Basic Information NodeLabel limit, shared by all Direct frontends. */
inline constexpr size_t ESPECTRE_DEVICE_LABEL_MAX_LENGTH = 32U;
/** Maximum serialized command-request size accepted by every transport. */
inline constexpr size_t ESPECTRE_COMMAND_MAX_PAYLOAD_SIZE = 2048U;

/**
 * A capability a frontend advertises in `EspectreCapabilityProfile`.
 *
 * Each value gates the commands and routes named after it; the route
 * registry returned by espectre_api_routes() lists the exact mapping.
 */
enum class EspectreDirectMethod : uint8_t {
  /** `capabilities`. */
  CAPABILITIES = 0,
  /** `device`. */
  INFO,
  /** `health` and the event stream. */
  STATUS,
  /** `sensing`, `wifi`, and `mqtt` reads. */
  CONFIG,
  /** `read_diagnostics`. */
  DIAGNOSTICS,
  /** The `sensing_enabled` field of `update_sensing`. */
  SET_SENSING,
  /** `update_device`. */
  SET_DEVICE_LABEL,
  /** The `threshold` field of `update_sensing`. */
  SET_THRESHOLD,
  /** The motion hit fields of `update_sensing`. */
  SET_MOTION_HITS,
  /** The `detector` field of `update_sensing`. */
  SET_DETECTOR,
  /** `recalibrate`. */
  RECALIBRATE,
  /** Opening raw collection with `GET /csi`; advertise it with `STOP_RAW_STREAM`. */
  START_RAW_STREAM,
  /** Ending raw collection; advertise it with `START_RAW_STREAM`. */
  STOP_RAW_STREAM,
  /** The `traffic_generator_mode` field of `update_sensing`. */
  SET_TRAFFIC_GENERATOR_MODE,
  /** `wifi_access_points`. */
  WIFI_ACCESS_POINTS,
  /** `scan_wifi`. */
  SCAN_WIFI_ACCESS_POINTS,
  /** `set_wifi_bssid`. */
  SET_WIFI_BSSID,
  /** `clear_wifi_bssid`. */
  CLEAR_WIFI_BSSID,
  /** `clear_wifi_credentials`. */
  CLEAR_WIFI_CONFIG,
  /** `update_mqtt`. */
  SET_MQTT_CONFIG,
  /** `clear_mqtt`. */
  CLEAR_MQTT_CONFIG,
  /** `devices`, the peer discovery result. */
  DISCOVER_PEERS,
  /** Number of values; not a capability. */
  COUNT,
};

/** A readable configuration section of the `capabilities` catalog. */
enum class EspectreConfigSection : uint8_t {
  /** Sensing configuration. */
  RUNTIME = 0,
  /** Device label. */
  DEVICE,
  /** Wi-Fi station settings. */
  WIFI,
  /** Broker settings. */
  MQTT,
  /** Number of values; not a section. */
  COUNT,
};

/** A family of events a frontend publishes. */
enum class EspectreEvent : uint8_t {
  /** `motion` telemetry. */
  TELEMETRY = 0,
  /** `health`. */
  STATUS,
  /** `device`. */
  INFO,
  /** `sensing` and `wifi`. */
  CONFIG,
  /** `fault`. */
  FAULT,
  /** Number of values; not an event. */
  COUNT,
};

/** How a route behaves. */
enum class EspectreApiRouteKind : uint8_t {
  /** Reads a resource. */
  RESOURCE = 0,
  /** Changes state or starts an action. */
  OPERATION,
  /** Opens a long-lived stream, such as events or raw CSI. */
  STREAM,
};

struct EspectreCommand;

/** Validate decoded parameters and populate the command before dispatch.
 * The parser supplies valid JSON fields and the command identity. Fields from
 * parse_espectre_command() also include the request's `command_id` and
 * `command`. Validators must not execute commands or change device state.
 * Ignore command output after a rejected validation.
 */
using EspectreCommandValidator = bool (*)(const std::vector<JsonObjectField> &fields,
                                        EspectreCommand *command, std::string *error);

/** One canonical HTTP/resource mapping used by routing and capability output. */
struct EspectreApiRoute {
  /** HTTP method, such as `GET` or `PATCH`. */
  const char *http_method;
  /** Absolute path, such as `/espectre/v1/sensing`. */
  const char *path;
  /** Resource or operation name in the `capabilities` catalog. */
  const char *name;
  /** Canonical command name; empty for streams. */
  const char *command;
  /** Capability that must be advertised for the route to exist. */
  EspectreDirectMethod capability;
  EspectreApiRouteKind kind;
  /** Whether the operation completes after the response. */
  bool asynchronous;
  /** Parameter validator; `nullptr` for streams. */
  EspectreCommandValidator validate{nullptr};
};

/** One canonical event and the capability that gates it. */
struct EspectreApiEventDescriptor {
  /** Event name on the wire, such as `motion`. */
  const char *name;
  /** Family the event belongs to. */
  EspectreEvent event;
  /** Capability that must be advertised for the event to be published. */
  EspectreDirectMethod capability;
};

/** One frontend-owned route, shared by capability output and both transports. */
struct EspectreExtensionRoute {
  /** HTTP method, such as `POST`. */
  const char *http_method;
  /** Absolute path under `/espectre/v1`. */
  const char *path;
  /** Resource or operation name in the `capabilities` catalog. */
  const char *name;
  /** Command name; must not collide with an SDK command. */
  const char *command;
  EspectreApiRouteKind kind;
  /** Whether the operation completes after the response. */
  bool asynchronous{false};
  /** Whether the frontend command binding permits invocation over MQTT. */
  bool mqtt{false};
  /** Whether the command may run while raw CSI collection is active. */
  bool allowed_during_raw_collection{false};
  /** Parameter validator. Required. */
  EspectreCommandValidator validate{nullptr};
};

/** Immutable frontend additions. Keep this object alive while adapters use it. */
struct EspectreProtocolExtension {
  /** Extra routes and their commands. */
  std::vector<EspectreExtensionRoute> routes;
  /** Extra event names the application publishes. */
  std::vector<std::string> events;
};

/** Reject malformed descriptors and collisions with the SDK or another extension entry. */
bool validate_protocol_extension(const EspectreProtocolExtension &extension, std::string *error = nullptr);
/** Find a command in a valid extension; a null or invalid extension has no routes. */
const EspectreExtensionRoute *find_extension_route(const EspectreProtocolExtension *extension,
                                                 const std::string &command);

/** Return the immutable v1 resource registry and its entry count. */
const EspectreApiRoute *espectre_api_routes(size_t *count);
/** Return the immutable canonical event registry and its entry count. */
const EspectreApiEventDescriptor *espectre_api_events(size_t *count);

/** Exact Direct command, event, and readable-configuration surface advertised by a frontend. */
struct EspectreCapabilityProfile {
  /** Advertised capabilities, indexed by `EspectreDirectMethod`. All off by default. */
  std::array<bool, static_cast<size_t>(EspectreDirectMethod::COUNT)> methods{};
  /** Readable sections, indexed by `EspectreConfigSection`. All off by default. */
  std::array<bool, static_cast<size_t>(EspectreConfigSection::COUNT)> config_sections{};
  /** Published event families, indexed by `EspectreEvent`. All on by default. */
  std::array<bool, static_cast<size_t>(EspectreEvent::COUNT)> events{{true, true, true, true, true}};
  /** Optional frontend-owned catalog, shared with its transport parsers. */
  const EspectreProtocolExtension *extension{nullptr};

  /** Whether a capability is advertised. */
  bool supports(EspectreDirectMethod method) const {
    return methods[static_cast<size_t>(method)];
  }
  /** Advertise or withdraw a capability. */
  void set(EspectreDirectMethod method, bool enabled = true) {
    methods[static_cast<size_t>(method)] = enabled;
  }
  /** Whether a configuration section is readable. */
  bool has(EspectreConfigSection section) const {
    return config_sections[static_cast<size_t>(section)];
  }
  /** Expose or hide a configuration section. */
  void set(EspectreConfigSection section, bool enabled = true) {
    config_sections[static_cast<size_t>(section)] = enabled;
  }
  /** Whether an event family is published. */
  bool publishes(EspectreEvent event) const {
    return events[static_cast<size_t>(event)];
  }
  /** Publish or suppress an event family. */
  void set(EspectreEvent event, bool enabled = true) {
    events[static_cast<size_t>(event)] = enabled;
  }
  /** Suppress every event family. */
  void clear_events() {
    events.fill(false);
  }
};

struct RuntimeDiagnosticsSample;

/** Protocol version reported by capabilities and discovery. */
inline constexpr const char *ESPECTRE_PROTOCOL_VERSION = "1.0";
/** DNS-SD TXT record schema advertised as the RFC 6763 `txtvers` value. */
inline constexpr const char *ESPECTRE_DNS_SD_TXT_SCHEMA_VERSION = "1";
/** Maximum canonical command correlation identifier length. */
inline constexpr size_t ESPECTRE_COMMAND_ID_MAX_LENGTH = 64U;
/** Default MQTT topic root. Override per device with `EspectreDeviceConfig::topic_prefix`. */
inline constexpr const char *ESPECTRE_TOPIC_PREFIX = "espectre/v1/devices";
/** Sentinel meaning "use the runtime-generated device id". */
inline constexpr uint64_t ESPECTRE_DEFAULT_DEVICE_ID = 0U;
/** Empty label, meaning the device id is used as the display name. */
inline constexpr const char *ESPECTRE_DEFAULT_DEVICE_LABEL = "";

/**
 * Device identity and broker settings.
 *
 * Frontends persist this so a device keeps its identity and connection across
 * reboots and reprovisioning.
 */
struct EspectreDeviceConfig {
  /** Stable device identity. Zero means use the runtime-generated value. */
  uint64_t device_id{ESPECTRE_DEFAULT_DEVICE_ID};
  /** Human-readable name. Empty falls back to the formatted device id. */
  std::string device_label{ESPECTRE_DEFAULT_DEVICE_LABEL};
  /** Broker transport scheme: `mqtt` or `mqtts`. Empty disables MQTT. */
  std::string mqtt_scheme;
  /** Broker DNS hostname, IPv4 address, or IPv6 address, without URI framing. */
  std::string mqtt_host;
  /** Broker port. Zero means MQTT is not configured. */
  uint16_t mqtt_port{0U};
  /** Broker user name. Leave empty for anonymous brokers. */
  std::string mqtt_username;
  /** Broker password. Never published by the protocol. */
  std::string mqtt_password;
  /** Topic root. Change it only if you also change every consumer. */
  std::string topic_prefix{ESPECTRE_TOPIC_PREFIX};
};

/** Link details available to frontends. Canonical MQTT info publishes only the channel. */
struct EspectreNetworkInfo {
  /** Dotted IPv4 address, or empty. */
  std::string ip_address;
  /** Station MAC address, or empty. */
  std::string mac_address;
  /** Wi-Fi channel in use. Zero when unknown. */
  uint8_t channel{0U};
};

/**
 * What the device advertises about itself.
 *
 * The `supports_*` flags are internal inputs used to build the filtered
 * `capabilities` catalog. They are deliberately omitted from `info` so clients
 * have one authoritative feature-discovery surface.
 */
struct EspectreDeviceInfo {
  /** Frontend name, for example `"native"`, `"matter"`, or your own. */
  std::string frontend{"unknown"};
  /** Application version supplied by the frontend or integrator. */
  std::string firmware_version{"unknown"};
  /** Chip target, normally `CONFIG_IDF_TARGET`. */
  std::string chip{"unknown"};
  /** Active detector. Left empty, it is filled from the snapshot. */
  std::string detector;
  /** Automatically selected CSI capture profile. Left empty, it is filled from the snapshot. */
  std::string csi_profile;
  /**
   * @name Capability flags for the flag-per-section catalog
   * Read only by the flag-per-section espectre_capabilities_payload()
   * overload; each flag enables the `EspectreDirectMethod` of the same name.
   * @{
   */
  bool supports_info{true};
  bool supports_diagnostics{false};
  /** `update_device` is honored and persists the user-facing label. */
  bool supports_device_config{false};
  bool supports_runtime_threshold{false};
  bool supports_runtime_motion_hits{false};
  bool supports_runtime_detector{false};
  bool supports_manual_recalibration{false};
  bool supports_traffic_control{false};
  /** @} */
  /**
   * Traffic generator mode: `"ping"`, `"dns"`, `"dns_tcp"`, `"wifi_raw"`, or `"external"`.
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
  /** UDP destination port used by the external CSI traffic generator. */
  uint16_t csi_traffic_udp_port{0U};
  /** IPv4 multicast group used by external CSI traffic, or empty for unicast-only operation. */
  std::string csi_traffic_multicast_group;
  /**
   * Detector evaluation cadence, in milliseconds.
   *
   * Omitted from `info` when zero. Canonical MQTT telemetry follows this interval.
   */
  uint32_t evaluation_interval_ms{0U};
  /** Current link details. */
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
  /** Command verb, for example `"update_sensing"` or `"recalibrate"`. */
  std::string command;
  /** Diagnostic paths to return; empty requests the catalog, and ["*"] requests all values. */
  std::vector<std::string> diagnostic_fields;
  /** Requested sensing-service state for `update_sensing`. */
  bool sensing_enabled{false};
  bool has_sensing_enabled{false};
  /** User-facing label requested by `update_device`; empty clears it. */
  std::string device_label;
  /** Whether the command carried a valid string-valued `device_label`. */
  bool has_device_label{false};
  /** `update_sensing` threshold, on the 0..1 scale. */
  float threshold{0.0f};
  bool has_threshold{false};
  /** `update_sensing` hit counts, 1..20 each; sent together. */
  uint8_t motion_on_hits{0U};
  uint8_t motion_off_hits{0U};
  bool has_motion_hits{false};
  /** `update_sensing` mode name; see parse_traffic_generator_mode(). */
  std::string traffic_generator_mode;
  bool has_traffic_generator_mode{false};
  /** `update_sensing` detector name; see parse_detection_algorithm(). */
  std::string detector;
  bool has_detector{false};
  /** `set_wifi_bssid` target, as `AA:BB:CC:DD:EE:FF`. */
  std::string wifi_bssid;
  bool has_wifi_bssid{false};
  /** Force reassociation even when `wifi_bssid` is already active. */
  bool wifi_bssid_force{false};
  bool has_wifi_bssid_force{false};
  /**
   * @name update_mqtt fields
   * Same meaning as the matching `EspectreDeviceConfig` fields.
   * @{
   */
  std::string mqtt_scheme;
  std::string mqtt_host;
  std::string mqtt_username;
  std::string mqtt_password;
  std::string mqtt_topic_prefix;
  uint16_t mqtt_port{0U};
  bool has_mqtt_scheme{false};
  bool has_mqtt_host{false};
  bool has_mqtt_username{false};
  bool has_mqtt_password{false};
  bool has_mqtt_topic_prefix{false};
  bool has_mqtt_port{false};
  /** @} */
  /** JSON parameters for a frontend extension command. Initially the whole
   * request from parse_espectre_command(), or the parameter object from
   * parse_espectre_command_request(); its validator may normalize them.
   */
  std::string extension_parameters;
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
/**
 * Pack the first six MAC bytes into the historical numeric representation.
 *
 * @deprecated Runtime firmware uses the cached, domain-separated SHA-256
 * pseudonym from `derive_runtime_device_id()` instead.
 */
[[deprecated("use the runtime-generated device identity")]] uint64_t espectre_device_id_from_mac(
    const uint8_t *mac, size_t mac_len);
/** Conventional device name derived from the immutable device identifier. */
std::string espectre_device_name(uint64_t device_id, const char *chip = nullptr);
/** The id actually in use. Frontend startup replaces the zero sentinel. */
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
 * @param default_frontend Frontend name used when `info.frontend` is empty.
 * @param default_chip Chip name used when `info.chip` is empty.
 * @return A copy of `info` with the gaps filled.
 */
EspectreDeviceInfo normalize_protocol_device_info(const EspectreDeviceInfo &info,
                                                  const RuntimeSnapshot *snapshot,
                                                  const char *default_frontend,
                                                  const char *default_chip = nullptr);
/** Erase broker settings while preserving identity, for a config reset. */
void clear_espectre_mqtt_config(EspectreDeviceConfig *config);
/**
 * Validate the complete MQTT endpoint in a device configuration.
 *
 * The endpoint requires an exact `mqtt` or `mqtts` scheme, a DNS hostname,
 * IPv4 address, or IPv6 address without URI framing, and a non-zero port.
 *
 * @param config Device configuration carrying the MQTT endpoint.
 * @param error Receives a human-readable reason on failure. May be `nullptr`.
 * @return `true` only when the complete endpoint is valid.
 */
bool validate_espectre_mqtt_config(const EspectreDeviceConfig &config, std::string *error = nullptr);
/** Return whether `config` contains a complete, valid MQTT endpoint. */
bool espectre_mqtt_configured(const EspectreDeviceConfig &config);

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
std::string espectre_health_payload(const EspectreDeviceConfig &config, bool online, uint32_t timestamp_ms);
/** Stable device identity and build description. Publish retained on connect. */
std::string espectre_device_payload(const EspectreDeviceConfig &config, const EspectreDeviceInfo &info);
/**
 * Filtered command, event, feature, and configuration catalog.
 */
std::string espectre_capabilities_payload(const EspectreDeviceConfig &config,
                                          const EspectreDeviceInfo &info,
                                          const EspectreCapabilityProfile &capabilities);
/**
 * Flag-per-section form of the capability catalog. Prefer the
 * `EspectreCapabilityProfile` overload, which represents readable sections and
 * individual commands independently.
 */
std::string espectre_capabilities_payload(const EspectreDeviceConfig &config,
                                          const EspectreDeviceInfo &info,
                                          bool supports_status = true,
                                          bool supports_config = false,
                                          bool supports_sensing_control = false,
                                          bool supports_wifi_bssid = false,
                                          bool supports_mqtt_config = false,
                                          bool supports_peer_discovery = false,
                                          bool supports_raw_csi = false,
                                          const EspectreProtocolExtension *extension = nullptr);
/** Current motion state and score. The payload behind every detector evaluation. */
std::string espectre_motion_payload(const EspectreDeviceConfig &config,
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
std::string espectre_diagnostics_payload(const EspectreDeviceConfig &config,
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
                                            const char *code,
                                            const char *message,
                                            const std::string &data_json = {});
/** Build the canonical flat command request carried by MQTT and Direct HTTP. */
std::string espectre_command_request_payload(const std::string &command_id,
                                             const std::string &command,
                                             const std::string &params_json = "{}");
/** Runtime fault event shared by every transport. */
std::string espectre_fault_payload(const EspectreDeviceConfig &config,
                                   const char *message,
                                   uint32_t timestamp_ms);
/** One sample of every canonical message, for protocol inspection and conformance tests. */
std::string espectre_message_catalog_payload(const EspectreProtocolExtension *extension = nullptr);
/** @} */

/**
 * @name Command parsing
 * Turn received bytes into validated values.
 * @{
 */

/**
 * Parse a canonical flat command request, as received on the MQTT command topic.
 *
 * @param payload Raw message body as received.
 * @param command Populated on success. Check the `has_*` flags to see which
 *        fields the peer actually sent. On failure it is reset and may keep
 *        the `command_id` for the result payload.
 * @param error Receives a human-readable reason on failure. May be `nullptr`.
 * @param extension Optional frontend routes and their parameter validators.
 * @return false on malformed input or invalid parameters. An unknown command
 *         name parses successfully; FrontendCommandEngine rejects it with the
 *         `unsupported` result code.
 */
bool parse_espectre_command(const std::string &payload, EspectreCommand *command, std::string *error,
                            const EspectreProtocolExtension *extension = nullptr);
/**
 * Parse an already separated command name plus a JSON parameter object.
 *
 * Frontend adapters use this after separating a canonical flat request into
 * internal fields. Validation follows canonical envelope order: correlation
 * identifier, protocol version, command name, and parameters.
 */
bool parse_espectre_command_request(const std::string &command_id,
                                    const std::string &command_name,
                                    const std::string &params_json,
                                    EspectreCommand *command,
                                    std::string *error,
                                    const std::string &protocol_version = ESPECTRE_PROTOCOL_VERSION,
                                    const EspectreProtocolExtension *extension = nullptr);
/**
 * Parse a legacy ASCII `SET_DEVICE_CONFIG:` command.
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
 * The complete command is parsed and validated before `config` is changed, so
 * rejection leaves the previous value intact. `scheme`, `host`, and `port` are
 * required; the rest keep their previous values.
 */
bool parse_espectre_mqtt_config_command(const std::string &command, EspectreDeviceConfig *config, std::string *error);

/** @} */

}  // namespace espectre
