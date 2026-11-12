# SPDX-License-Identifier: GPL-3.0-only
# Commercial licensing available under separate agreement; see LICENSING.md.
"""ESPectre Direct API for the espectre component built into ESPHome.

The built-in component owns sensing and its entities. This add-on serves the
ESPectre Direct API next to it: device, sensing, and diagnostics resources,
live events, raw CSI collection, and discovery. Sensing changes are handed to
the built-in component, so its entities and saved settings stay right.

Author: Francesco Pace <francesco.pace@gmail.com>
"""

from pathlib import Path

import esphome.codegen as cg
from esphome.components import esp32, espectre
from esphome.components.esp32.const import KEY_COMPONENTS, KEY_ESP32, KEY_REF
import esphome.config_validation as cv
from esphome.const import CONF_ID, CONF_PATH, CONF_PORT, CONF_VERSION
from esphome.core import CORE, EsphomeError
from esphome.coroutine import CoroPriority, coroutine_with_priority
from esphome.helpers import copy_file_if_changed

CODEOWNERS = ["@francescopace"]
DEPENDENCIES = ["espectre", "wifi"]
AUTO_LOAD = ["mdns"]

CONF_RAW_CSI = "raw_csi"
CONF_SDK = "sdk"

# ESPECTRE_DIRECT_HTTP_PORT in the SDK.
DIRECT_HTTP_PORT = 62587
SDK_COMPONENT = "francescopace/espectre"
SDK_DIRECTORIES = ("francescopace__espectre", "espectre")
# The source directories of an SDK checkout; its other directories hold firmware and examples.
SDK_SOURCE_DIRECTORIES = ("core", "runtime")
SDK_STAGING_EXCLUDES = ("build", "managed_components")
# The first SDK with guest access to a runtime that another component owns.
MIN_SDK_VERSION = (3, 2, 0)

espectre_direct_ns = cg.esphome_ns.namespace("espectre_direct")
ESPectreDirect = espectre_direct_ns.class_("ESPectreDirect", cg.Component)


def _sdk_path(value):
    path = Path(cv.directory(value))
    if not (path / "espectre_sdk.h").is_file():
        raise cv.Invalid("This directory does not contain the ESPectre SDK (espectre_sdk.h is missing)")
    return str(path)


def _sdk_source_files(source):
    """List the files of an SDK directory that the component build reads."""
    files = [path for path in source.iterdir() if path.is_file() and not path.name.startswith(".")]
    for directory in SDK_SOURCE_DIRECTORIES:
        for path in (source / directory).rglob("*"):
            parts = path.relative_to(source).parts
            if path.is_file() and not any(
                part.startswith((".", *SDK_STAGING_EXCLUDES)) for part in parts
            ):
                files.append(path)
    return files


def _stage_sdk(source):
    """Copy an SDK directory into the build tree under a name that ESP-IDF resolves.

    ESP-IDF names a local component after its real directory, so a repository
    checkout (`src/cpp`) or a symbolic link to it cannot be used in place.
    """
    staged = CORE.relative_build_path("espectre_sdk", SDK_DIRECTORIES[1])
    wanted = set()
    for path in _sdk_source_files(source):
        target = staged / path.relative_to(source)
        copy_file_if_changed(path, target)
        wanted.add(target)
    # A file removed from the SDK must not survive in the copy.
    for path in staged.rglob("*"):
        if path.is_file() and path not in wanted:
            path.unlink()
    return staged


CONFIG_SCHEMA = cv.All(
    cv.Schema(
        {
            cv.GenerateID(): cv.declare_id(ESPectreDirect),
            cv.GenerateID(espectre.CONF_ESPECTRE_ID): cv.use_id(espectre.ESPectreComponent),
            cv.Optional(CONF_PORT, default=DIRECT_HTTP_PORT): cv.port,
            cv.Optional(CONF_RAW_CSI, default=True): cv.boolean,
            # Replace the SDK that ESPHome pins when it is older than this add-on needs.
            cv.Optional(CONF_SDK): cv.Schema(
                {
                    cv.Exclusive(CONF_VERSION, CONF_SDK): cv.string_strict,
                    cv.Exclusive(CONF_PATH, CONF_SDK): _sdk_path,
                }
            ),
        }
    ).extend(cv.COMPONENT_SCHEMA),
    cv.only_on_esp32,
)


def _sdk_version(ref):
    """Return a release version as a tuple, or None for a branch, a commit, or a pre-release."""
    parts = str(ref).removeprefix("v").split(".")
    if len(parts) != 3 or not all(part.isdigit() for part in parts):
        return None
    return tuple(int(part) for part in parts)


def _require_guest_access_sdk():
    """Fail before the build when the SDK that ESPHome pins is too old for this add-on."""
    pinned = CORE.data[KEY_ESP32][KEY_COMPONENTS].get(SDK_COMPONENT, {})
    version = _sdk_version(pinned.get(KEY_REF))
    if version is not None and version < MIN_SDK_VERSION:
        minimum = ".".join(str(part) for part in MIN_SDK_VERSION)
        raise EsphomeError(
            f"espectre_direct needs ESPectre SDK {minimum} or later, and this ESPHome builds with "
            f"{pinned[KEY_REF]}. Add `sdk: {{version: \"{minimum}\"}}` under espectre_direct."
        )


# Runs after the built-in component, whose SDK version this may replace.
@coroutine_with_priority(CoroPriority.LATE)
async def to_code(config):
    var = cg.new_Pvariable(config[CONF_ID])
    await cg.register_component(var, config)
    parent = await cg.get_variable(config[espectre.CONF_ESPECTRE_ID])
    cg.add(var.set_parent(parent))
    cg.add(var.set_port(config[CONF_PORT]))
    cg.add(var.set_raw_csi(config[CONF_RAW_CSI]))

    sdk = config.get(CONF_SDK, {})
    if CONF_PATH in sdk:
        path = Path(sdk[CONF_PATH]).resolve()
        if path.name not in SDK_DIRECTORIES:
            path = _stage_sdk(path)
        esp32.add_idf_component(name=SDK_COMPONENT, path=str(path))
    elif CONF_VERSION in sdk:
        esp32.add_idf_component(name=SDK_COMPONENT, ref=sdk[CONF_VERSION])
    else:
        _require_guest_access_sdk()
    esp32.add_idf_sdkconfig_option("CONFIG_ESPECTRE_SDK_ENABLE_DIRECT", True)
    esp32.include_builtin_idf_component("esp_http_server")
