#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-only
# Commercial licensing available under separate agreement; see LICENSING.md.
"""Release keys, signed firmware catalogs, and ESP-IDF image verification."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec, padding, rsa

REPO_ROOT = Path(__file__).resolve().parents[2]
PUBLIC_KEYS = REPO_ROOT / "docs/web/assets/firmware-signing-keys.json"
RSA_SECRET = "FIRMWARE_SIGNING_KEY_RSA"
ESP32_SECRET = "FIRMWARE_SIGNING_KEY_ESP32"
MANIFEST_FORMAT = "espectre-firmware-v1"
ARTIFACT_FIELDS = ("chip", "chip_family", "build_type", "filename", "size", "sha256")
CATALOG_FIELDS = ("channel", "version", "release_tag", "commit")


def key_record(key) -> dict:
    public = key.public_key() if hasattr(key, "private_bytes") else key
    if isinstance(public, rsa.RSAPublicKey) and public.key_size == 3072:
        algorithm = "rsa3072"
    elif isinstance(public, ec.EllipticCurvePublicKey) and isinstance(public.curve, ec.SECP256R1):
        algorithm = "ecdsa_v1"
    else:
        raise ValueError("Firmware keys must use RSA-3072 or ECDSA P-256")
    der = public.public_bytes(serialization.Encoding.DER, serialization.PublicFormat.SubjectPublicKeyInfo)
    return {"id": hashlib.sha256(der).hexdigest(), "algorithm": algorithm,
            "spki": base64.b64encode(der).decode("ascii")}


def trusted_keys(path: Path | None = None) -> dict:
    path = path or PUBLIC_KEYS
    document = json.loads(path.read_text(encoding="utf-8"))
    if document.get("schema_version") != 1:
        raise ValueError("Unsupported firmware key registry")
    keys = {}
    for record in document["keys"]:
        public = serialization.load_der_public_key(base64.b64decode(record["spki"], validate=True))
        if key_record(public) != record or record["id"] in keys:
            raise ValueError("Invalid or duplicate firmware public key")
        keys[record["id"]] = public
    return keys


def private_key_from_secret(name: str, algorithm: str, *, registry: Path | None = None):
    value = os.environ.get(name)
    if not value:
        raise ValueError(f"Required signing secret {name} is not configured")
    key = serialization.load_pem_private_key(value.encode("ascii"), password=None)
    record = key_record(key)
    if record["algorithm"] != algorithm or record["id"] not in trusted_keys(registry):
        raise ValueError(f"{name} does not match an enrolled {algorithm} public key")
    return key


def artifact_claim(frontend: str, artifact: dict) -> dict:
    claim = {"frontend": frontend, **{field: artifact[field] for field in ARTIFACT_FIELDS}}
    filename = claim["filename"]
    if (not isinstance(filename, str) or not filename or Path(filename).name != filename
            or "\\" in filename or filename in (".", "..")):
        raise ValueError("Invalid firmware filename")
    if type(claim["size"]) is not int or claim["size"] <= 0:
        raise ValueError("Invalid firmware size")
    digest = claim["sha256"]
    if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
        raise ValueError("Invalid firmware SHA-256")
    return claim


def manifest_claims(manifest: dict) -> dict:
    artifacts = [artifact_claim(frontend, artifact)
                 for frontend, metadata in manifest["frontends"].items()
                 for artifact in metadata["artifacts"]]
    if len({item["filename"] for item in artifacts}) != len(artifacts):
        raise ValueError("Duplicate firmware filename")
    return {"format": MANIFEST_FORMAT, **{field: manifest.get(field) for field in CATALOG_FIELDS},
            "artifacts": sorted(artifacts, key=lambda item: item["filename"])}


def sign_manifest(manifest: dict, key) -> None:
    if key_record(key)["algorithm"] != "rsa3072":
        raise ValueError("Catalog signatures require RSA-3072")
    payload = json.dumps(manifest_claims(manifest), sort_keys=True, separators=(",", ":"),
                         ensure_ascii=True).encode("utf-8")
    signature = key.sign(payload, padding.PSS(mgf=padding.MGF1(hashes.SHA256()), salt_length=32), hashes.SHA256())
    manifest["authentication"] = {
        "key_id": key_record(key)["id"],
        "payload": base64.b64encode(payload).decode("ascii"),
        "signature": base64.b64encode(signature).decode("ascii"),
    }


def verify_manifest(manifest: dict, *, registry: Path | None = None,
                    firmware_dir: Path | None = None) -> dict:
    authentication = manifest.get("authentication")
    if not isinstance(authentication, dict):
        raise ValueError("Firmware catalog is not signed")
    key = trusted_keys(registry).get(authentication.get("key_id"))
    if not isinstance(key, rsa.RSAPublicKey):
        raise ValueError("Firmware catalog uses an untrusted signing key")
    payload = base64.b64decode(authentication["payload"], validate=True)
    signature = base64.b64decode(authentication["signature"], validate=True)
    try:
        key.verify(signature, payload, padding.PSS(mgf=padding.MGF1(hashes.SHA256()), salt_length=32), hashes.SHA256())
    except InvalidSignature as error:
        raise ValueError("Invalid firmware catalog signature") from error
    signed = json.loads(payload)
    expected = manifest_claims(manifest)
    if any(signed.get(field) != expected[field] for field in ("format", *CATALOG_FIELDS)):
        raise ValueError("Firmware catalog identity does not match its signature")
    by_name = {item["filename"]: item for item in signed["artifacts"]}
    if len(by_name) != len(signed["artifacts"]):
        raise ValueError("Duplicate signed firmware filename")
    # A website catalog may select only factory images and change their URLs.
    # Every retained artifact must still match the immutable signed inventory.
    for claim in expected["artifacts"]:
        if by_name.get(claim["filename"]) != claim:
            raise ValueError(f"Firmware metadata is not authenticated: {claim['filename']}")
        if firmware_dir is not None:
            data = (firmware_dir / claim["filename"]).read_bytes()
            if len(data) != claim["size"] or hashlib.sha256(data).hexdigest() != claim["sha256"]:
                raise ValueError(f"Firmware does not match its signed hash: {claim['filename']}")
    return signed


def verify_app(image: Path, public_key, *, legacy: bool) -> None:
    with tempfile.TemporaryDirectory(prefix="espectre-verify-") as directory:
        public_path = Path(directory) / "public.pem"
        public_path.write_bytes(public_key.public_bytes(
            serialization.Encoding.PEM, serialization.PublicFormat.SubjectPublicKeyInfo))
        result = subprocess.run(
            [sys.executable, "-m", "espsecure", "verify-signature", "--version", "1" if legacy else "2",
             "--keyfile", str(public_path), str(image)], capture_output=True, text=True)
        if result.returncode:
            raise ValueError(f"Invalid ESP-IDF application signature: {image.name}")


def factory_app(factory: bytes) -> tuple[int, int]:
    for offset in range(0x8000, 0x8C00, 32):
        entry = factory[offset:offset + 32]
        if len(entry) != 32 or entry[:2] != b"\xaa\x50":
            break
        if entry[2] == 0 and entry[3] in (0, 0x10):
            address = int.from_bytes(entry[4:8], "little")
            size = int.from_bytes(entry[8:12], "little")
            if address >= len(factory):
                raise ValueError("Factory application is missing")
            return address, size
    raise ValueError("Factory image has no initial application partition")


def verify_published_apps(manifest: dict, firmware_dir: Path, *, registry: Path | None = None) -> None:
    keys = trusted_keys(registry)
    for frontend in ("native", "esphome"):
        artifacts = manifest["frontends"][frontend]["artifacts"]
        for artifact in artifacts:
            if artifact["build_type"] != "ota":
                continue
            legacy = artifact["chip"] == "esp32"
            candidates = [key for key in keys.values()
                          if isinstance(key, ec.EllipticCurvePublicKey) == legacy]
            image = firmware_dir / artifact["filename"]
            for key in candidates:
                try:
                    verify_app(image, key, legacy=legacy)
                    break
                except ValueError:
                    continue
            else:
                raise ValueError(f"No trusted OTA signature: {image.name}")
            factory = next(item for item in artifacts
                           if item["chip"] == artifact["chip"] and item["build_type"] == "factory")
            data = (firmware_dir / factory["filename"]).read_bytes()
            address, size = factory_app(data)
            app = image.read_bytes()
            if len(app) > size or data[address:address + len(app)] != app:
                raise ValueError(f"Factory image does not contain its signed OTA app: {factory['filename']}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--enroll", type=Path, required=True, help="Enroll a PEM public key (never a private key)")
    parser.add_argument("--registry", type=Path, default=PUBLIC_KEYS)
    args = parser.parse_args()
    public = serialization.load_pem_public_key(args.enroll.read_bytes())
    record = key_record(public)
    keys = trusted_keys(args.registry) if args.registry.exists() else {}
    keys[record["id"]] = public
    document = {"schema_version": 1, "keys": [key_record(key) for key in keys.values()]}
    args.registry.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    print(f"Enrolled {record['algorithm']} public key {record['id']}")


if __name__ == "__main__":
    main()
