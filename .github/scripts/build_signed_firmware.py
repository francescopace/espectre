#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-only
# Commercial licensing available under separate agreement; see LICENSING.md.
"""Build signed CI firmware without changing local Native or ESPHome defaults."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ec, rsa

from firmware_signing import ESP32_SECRET, RSA_SECRET, REPO_ROOT, factory_app, private_key_from_secret, verify_app

NATIVE_APP = REPO_ROOT / "src/cpp/frontend/native/app"
ESPHOME_EXAMPLES = REPO_ROOT / "src/cpp/frontend/esphome/examples"


def validate_sdkconfig(path: Path, *, legacy: bool) -> None:
    options = dict(line.split("=", 1) for line in path.read_text().splitlines()
                   if line.startswith("CONFIG_") and "=" in line)
    required = ["SECURE_SIGNED_APPS_NO_SECURE_BOOT", "SECURE_SIGNED_ON_UPDATE_NO_SECURE_BOOT",
                "SECURE_BOOT_BUILD_SIGNED_BINARIES",
                "SECURE_SIGNED_APPS_ECDSA_SCHEME" if legacy else "SECURE_SIGNED_APPS_RSA_SCHEME"]
    if any(options.get(f"CONFIG_{name}") != "y" for name in required):
        raise ValueError("The compiled firmware does not enforce the requested OTA signature scheme")
    if any(options.get(f"CONFIG_{name}") == "y" for name in
           ("SECURE_BOOT", "SECURE_FLASH_ENC_ENABLED", "BOOTLOADER_APP_ANTI_ROLLBACK")):
        raise ValueError("Published general-purpose firmware must not enable irreversible eFuse policy")


def native_profile(key_path: str, *, legacy: bool) -> str:
    return "\n".join([
        "CONFIG_SECURE_SIGNED_APPS_NO_SECURE_BOOT=y",
        "CONFIG_SECURE_SIGNED_ON_UPDATE_NO_SECURE_BOOT=y",
        "CONFIG_SECURE_BOOT_BUILD_SIGNED_BINARIES=y",
        f"CONFIG_SECURE_BOOT_SIGNING_KEY={json.dumps(key_path)}",
        f"CONFIG_SECURE_SIGNED_APPS_ECDSA_SCHEME={'y' if legacy else 'n'}",
        f"CONFIG_SECURE_SIGNED_APPS_RSA_SCHEME={'n' if legacy else 'y'}",
        "CONFIG_SECURE_SIGNED_APPS_ECDSA_V2_SCHEME=n",
        "CONFIG_SECURE_BOOT=n",
        "CONFIG_SECURE_SIGNED_ON_BOOT_NO_SECURE_BOOT=n",
        "CONFIG_SECURE_FLASH_ENC_ENABLED=n",
        "CONFIG_BOOTLOADER_APP_ANTI_ROLLBACK=n",
        "",
    ])


def validate_images(factory: Path, ota: Path, key, *, legacy: bool) -> None:
    verify_app(ota, key.public_key(), legacy=legacy)
    data = factory.read_bytes()
    address, size = factory_app(data)
    app = ota.read_bytes()
    if len(app) > size or data[address:address + len(app)] != app:
        raise ValueError("The factory image must contain the complete signed OTA application")


def build_native(directory: Path, key_path: Path, args, key) -> None:
    target = os.environ["NATIVE_TARGET"]
    defaults = ["sdkconfig.defaults"]
    if (NATIVE_APP / f"sdkconfig.defaults.{target}").is_file():
        defaults.append(f"sdkconfig.defaults.{target}")
    container_key = "/work/" + key_path.relative_to(REPO_ROOT).as_posix()
    profile = directory / "sdkconfig.signed"
    profile.write_text(native_profile(container_key, legacy=args.legacy))
    defaults.append("/work/" + profile.relative_to(REPO_ROOT).as_posix())
    build_dir = f"build-container-signed-{target}"
    env = {**os.environ, "NATIVE_SDKCONFIG_DEFAULTS": ";".join(defaults), "NATIVE_BUILD_DIR": build_dir}
    # Secrets are materialized only in the temporary key file, outside cached paths.
    env.pop(RSA_SECRET, None)
    env.pop(ESP32_SECRET, None)
    subprocess.run(["bash", str(REPO_ROOT / ".github/scripts/build_native_firmware.sh")], env=env, check=True)
    build = NATIVE_APP / build_dir
    validate_sdkconfig(build / "sdkconfig", legacy=args.legacy)
    validate_images(Path(os.environ["NATIVE_OUTPUT"]), Path(os.environ["NATIVE_OTA_OUTPUT"]), key, legacy=args.legacy)


def build_esphome(key_path: Path, args, key) -> None:
    config = args.config.resolve()
    build_root = config.parent / ".esphome/build" / f"signed-{args.target}"
    # Keep the wrapper beside the example so relative component paths retain their meaning.
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", prefix=".signed-", dir=config.parent) as wrapper:
        wrapper.write(
            f"packages:\n  base: !include {json.dumps(config.name)}\n"
            f"esphome:\n  build_path: {json.dumps(str(build_root))}\n"
            "esp32:\n  framework:\n    advanced:\n      signed_ota_verification:\n"
            f"        signing_scheme: {'ecdsa_v1' if args.legacy else 'rsa3072'}\n"
            f"        signing_key: {json.dumps(str(key_path))}\n")
        wrapper.flush()
        env = dict(os.environ)
        env.pop(RSA_SECRET, None)
        env.pop(ESP32_SECRET, None)
        subprocess.run([sys.executable, "-m", "esphome", "-s", "component_source", "local",
                        "-s", "project_version", args.version, "compile", wrapper.name], env=env, check=True)
    descriptions = [path for path in build_root.glob("**/project_description.json")
                    if path.parent.name != "bootloader"
                    and json.loads(path.read_text()).get("target") == args.target]
    if len(descriptions) != 1:
        raise ValueError("Expected one ESPHome application build for the requested target")
    description = json.loads(descriptions[0].read_text())
    validate_sdkconfig(Path(description["config_file"]), legacy=args.legacy)
    build = descriptions[0].parent
    validate_images(build / "firmware.factory.bin", build / "firmware.ota.bin", key, legacy=args.legacy)
    if output := os.environ.get("GITHUB_OUTPUT"):
        with open(output, "a", encoding="utf-8") as file:
            file.write(f"factory_bin={build / 'firmware.factory.bin'}\n"
                       f"ota_bin={build / 'firmware.ota.bin'}\n"
                       f"project_description={descriptions[0]}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("frontend", choices=("native", "esphome"))
    parser.add_argument("--config", type=Path)
    parser.add_argument("--target", choices=("esp32", "esp32s2", "esp32s3", "esp32c3", "esp32c5", "esp32c6"))
    parser.add_argument("--version")
    parser.add_argument("--test-key", action="store_true", help="Use an ephemeral key for non-publishing CI")
    args = parser.parse_args()
    if args.frontend == "esphome" and not all((args.config, args.target, args.version)):
        parser.error("ESPHome requires --config, --target, and --version")
    args.legacy = (args.target or os.environ.get("NATIVE_TARGET")) == "esp32"
    if args.test_key:
        key = ec.generate_private_key(ec.SECP256R1()) if args.legacy else rsa.generate_private_key(65537, 3072)
        print("Building with a temporary CI test key; this firmware must not be published.")
    else:
        key = private_key_from_secret(ESP32_SECRET if args.legacy else RSA_SECRET,
                                      "ecdsa_v1" if args.legacy else "rsa3072")
    (REPO_ROOT / ".cache").mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="firmware-signing-", dir=REPO_ROOT / ".cache") as temporary:
        directory = Path(temporary)
        key_path = directory / "signing.pem"
        with os.fdopen(os.open(key_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), "wb") as file:
            file.write(key.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8,
                                         serialization.NoEncryption()))
        if args.frontend == "native":
            build_native(directory, key_path, args, key)
        else:
            build_esphome(key_path, args, key)
    print("Verified OTA signature, factory application, and software-only signing configuration.")


if __name__ == "__main__":
    main()
