#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-only
# Commercial licensing available under separate agreement; see LICENSING.md.
"""Build per-firmware SPDX and third-party license artifacts."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import re
import zipfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
LICENSE_BASENAMES = ("LICENSE", "LICENCE", "COPYING", "NOTICE", "COPYRIGHT")
ESP_MATTER_NOTICE = (
    REPO_ROOT / "src" / "cpp" / "frontend" / "matter" / "third_party" / "esp_matter" / "NOTICE"
)


@dataclass
class Package:
    name: str
    version: str
    license_expression: str
    roots: set[Path] = field(default_factory=set)
    component_names: set[str] = field(default_factory=set)

    @property
    def spdx_id(self) -> str:
        normalized = re.sub(r"[^A-Za-z0-9.-]+", "-", self.name).strip("-")
        return f"SPDXRef-Package-{normalized}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build compliance companions for one firmware image.")
    parser.add_argument("--frontend", choices=("esphome", "matter", "native"), required=True)
    parser.add_argument("--project-description", required=True)
    parser.add_argument("--firmware", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def is_license_file(path: Path) -> bool:
    name = path.name.upper()
    return any(name == base or name.startswith(f"{base}.") or name.startswith(f"{base}-") for base in LICENSE_BASENAMES)


def package_version_from_manifest(root: Path) -> str:
    manifest = root / "idf_component.yml"
    if not manifest.is_file():
        return "NOASSERTION"
    match = re.search(r'^version:\s*["\']?([^"\'\s]+)', manifest.read_text(encoding="utf-8"), re.MULTILINE)
    return match.group(1) if match else "NOASSERTION"


def infer_license(root: Path, fallback: str = "NOASSERTION") -> str:
    candidates = sorted(path for path in root.glob("LICENSE*") if path.is_file())
    if not candidates:
        return fallback
    text = candidates[0].read_text(encoding="utf-8", errors="ignore")[:4000].lower()
    if "apache license" in text and "version 2.0" in text:
        return "Apache-2.0"
    if "mit license" in text or "permission is hereby granted, free of charge" in text:
        return "MIT"
    if "bsd 2-clause" in text:
        return "BSD-2-Clause"
    if "bsd 3-clause" in text or "neither the name" in text:
        return "BSD-3-Clause"
    return fallback


def managed_component_root(path: Path) -> Path | None:
    parts = path.parts
    if "managed_components" not in parts:
        return None
    index = parts.index("managed_components")
    if index + 1 >= len(parts):
        return None
    return Path(*parts[: index + 2])


def collect_packages(description: dict, frontend: str) -> dict[str, Package]:
    project_path = Path(description["project_path"]).resolve()
    idf_path = Path(description["idf_path"]).resolve()
    project_version = description.get("project_version", "NOASSERTION")
    if project_version == "1":
        project_version = "NOASSERTION"
    packages: dict[str, Package] = {
        "ESPectre": Package(
            name="ESPectre",
            version=project_version,
            license_expression="GPL-3.0-only OR LicenseRef-ESPectre-Commercial",
        ),
        "ESP-IDF": Package(
            name="ESP-IDF",
            version=description.get("git_revision", "NOASSERTION"),
            license_expression="Apache-2.0",
            roots={idf_path},
        ),
    }

    for component_name, component in description.get("build_component_info", {}).items():
        component_dir_raw = component.get("dir")
        if not component_dir_raw:
            continue
        component_dir = Path(component_dir_raw).resolve()
        component_sources = component.get("sources", [])
        component_library = component.get("file", "")
        if not component_sources and not component_library:
            continue

        try:
            component_dir.relative_to(idf_path / "components")
        except ValueError:
            pass
        else:
            packages["ESP-IDF"].component_names.add(component_name)
            packages["ESP-IDF"].roots.add(component_dir)
            continue

        managed_root = managed_component_root(component_dir)
        if managed_root is not None:
            package_name = managed_root.name.replace("__", "/", 1)
            package = packages.setdefault(
                package_name,
                Package(
                    name=package_name,
                    version=package_version_from_manifest(managed_root),
                    license_expression=infer_license(managed_root),
                    roots={managed_root},
                ),
            )
            package.component_names.add(component_name)
            continue

        if component_name == "espectre":
            packages["ESPectre"].component_names.add(component_name)
            continue

        if "/src/esphome/" in f"{component_dir.as_posix()}/":
            try:
                esphome_version = importlib.metadata.version("esphome")
            except importlib.metadata.PackageNotFoundError:
                esphome_version = "NOASSERTION"
            package = packages.setdefault(
                "ESPHome C++ runtime",
                Package(
                    name="ESPHome C++ runtime",
                    version=esphome_version,
                    license_expression="GPL-3.0-only",
                    roots={component_dir},
                ),
            )
            package.component_names.add(component_name)
            continue

        try:
            component_dir.relative_to(REPO_ROOT)
        except ValueError:
            package_name = component_dir.name or component_name
            package = packages.setdefault(
                package_name,
                Package(
                    name=package_name,
                    version=package_version_from_manifest(component_dir),
                    license_expression=infer_license(component_dir),
                    roots={component_dir},
                ),
            )
            package.component_names.add(component_name)
        else:
            packages["ESPectre"].component_names.add(component_name)

    if frontend == "esphome" and "ESPHome C++ runtime" not in packages:
        try:
            esphome_version = importlib.metadata.version("esphome")
        except importlib.metadata.PackageNotFoundError:
            esphome_version = "NOASSERTION"
        packages["ESPHome C++ runtime"] = Package(
            name="ESPHome C++ runtime",
            version=esphome_version,
            license_expression="GPL-3.0-only",
        )
    if frontend == "matter" and not any(package.name == "espressif/esp_matter" for package in packages.values()):
        packages["espressif/esp_matter"] = Package(
            name="espressif/esp_matter",
            version="NOASSERTION",
            license_expression="Apache-2.0",
        )
    return packages


def collect_license_files(packages: dict[str, Package], frontend: str) -> list[tuple[str, Path]]:
    collected: dict[str, Path] = {
        "ESPectre/LICENSE": REPO_ROOT / "LICENSE",
        "ESPectre/LICENSING.md": REPO_ROOT / "LICENSING.md",
    }
    for package in packages.values():
        if package.name == "ESPectre":
            continue
        package_base = min(package.roots, key=lambda entry: len(entry.parts)) if package.roots else None
        for root in sorted(package.roots):
            if not root.exists():
                continue
            if package.name == "ESP-IDF" and root == package_base:
                candidates = root.iterdir()
            else:
                candidates = [root] if root.is_file() else root.rglob("*")
            for path in candidates:
                if not path.is_file() or not is_license_file(path):
                    continue
                try:
                    relative = path.relative_to(package_base) if package_base is not None else Path(path.name)
                except ValueError:
                    relative = Path(path.name)
                archive_name = f"{package.name.replace('/', '__')}/{relative.as_posix()}"
                collected.setdefault(archive_name, path)

    if frontend == "esphome":
        try:
            distribution = importlib.metadata.distribution("esphome")
        except importlib.metadata.PackageNotFoundError:
            pass
        else:
            for entry in distribution.files or ():
                path = Path(distribution.locate_file(entry))
                if path.is_file() and is_license_file(path):
                    collected.setdefault(f"ESPHome/{path.name}", path)

    if frontend == "matter" and ESP_MATTER_NOTICE.is_file():
        collected["espressif__esp_matter/NOTICE"] = ESP_MATTER_NOTICE
    return sorted(collected.items())


def build_spdx(description: dict, frontend: str, firmware: Path, packages: dict[str, Package]) -> dict:
    firmware_sha256 = hashlib.sha256(firmware.read_bytes()).hexdigest()
    document_name = f"{firmware.name} SBOM"
    document_namespace = f"https://espectre.dev/sbom/{firmware.name}/{firmware_sha256}"
    firmware_spdx_id = "SPDXRef-Package-Firmware"
    package_entries = [
        {
            "SPDXID": firmware_spdx_id,
            "name": firmware.name,
            "versionInfo": packages["ESPectre"].version,
            "downloadLocation": "NOASSERTION",
            "filesAnalyzed": False,
            "checksums": [{"algorithm": "SHA256", "checksumValue": firmware_sha256}],
            "licenseConcluded": "NOASSERTION",
            "licenseDeclared": "NOASSERTION",
            "copyrightText": "NOASSERTION",
            "comment": f"ESPectre {frontend} firmware image for {description.get('target', 'unknown')}",
        }
    ]
    relationships = [
        {
            "spdxElementId": "SPDXRef-DOCUMENT",
            "relationshipType": "DESCRIBES",
            "relatedSpdxElement": firmware_spdx_id,
        }
    ]
    for package in sorted(packages.values(), key=lambda entry: entry.name.casefold()):
        package_entries.append(
            {
                "SPDXID": package.spdx_id,
                "name": package.name,
                "versionInfo": package.version,
                "downloadLocation": "NOASSERTION",
                "filesAnalyzed": False,
                "licenseConcluded": "NOASSERTION" if package.name == "ESP-IDF" else package.license_expression,
                "licenseDeclared": package.license_expression,
                "copyrightText": "NOASSERTION",
                "comment": "Build components: " + ", ".join(sorted(package.component_names)),
            }
        )
        relationships.append(
            {
                "spdxElementId": firmware_spdx_id,
                "relationshipType": "DEPENDS_ON",
                "relatedSpdxElement": package.spdx_id,
            }
        )
    return {
        "spdxVersion": "SPDX-2.3",
        "dataLicense": "CC0-1.0",
        "SPDXID": "SPDXRef-DOCUMENT",
        "name": document_name,
        "documentNamespace": document_namespace,
        "creationInfo": {
            "created": datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
            "creators": ["Tool: ESPectre build_firmware_compliance.py"],
        },
        "documentDescribes": [firmware_spdx_id],
        "packages": package_entries,
        "relationships": relationships,
        "hasExtractedLicensingInfos": [
            {
                "licenseId": "LicenseRef-ESPectre-Commercial",
                "name": "ESPectre commercial license",
                "extractedText": "Separate written commercial terms are available from the ESPectre maintainer and are not included in this artifact.",
            }
        ],
        "annotations": [
            {
                "annotationDate": datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
                "annotationType": "OTHER",
                "annotator": "Tool: ESPectre build_firmware_compliance.py",
                "comment": f"Firmware SHA-256: {firmware_sha256}; frontend: {frontend}; target: {description.get('target', 'unknown')}",
            }
        ],
    }


def write_notice(path: Path, firmware: Path, frontend: str, description: dict, packages: dict[str, Package]) -> None:
    lines = [
        "ESPectre firmware third-party notices",
        "========================================",
        "",
        f"Firmware: {firmware.name}",
        f"Frontend: {frontend}",
        f"Target: {description.get('target', 'unknown')}",
        "",
        "This file is a summary. The adjacent third-party-licenses ZIP contains the ESPectre license terms and the license, copying, copyright, and NOTICE files collected from the components used by this build. The adjacent SPDX JSON records the firmware checksum and build package inventory.",
        "",
        "ESPectre itself is available under GPL-3.0-only, or under a separate commercial license from the maintainer. A commercial ESPectre license does not replace or modify third-party terms.",
        "",
        "Packages",
        "--------",
    ]
    for package in sorted(packages.values(), key=lambda entry: entry.name.casefold()):
        components = ", ".join(sorted(package.component_names)) or "framework/runtime package"
        lines.append(f"- {package.name} {package.version} — {package.license_expression} — components: {components}")
    if frontend == "esphome":
        lines.extend(("", "The ESPHome firmware frontend combines ESPectre with the GPL-3.0-only ESPHome C++ runtime and is distributed only under GPLv3."))
    if frontend == "matter":
        lines.extend(("", "The Matter SDK NOTICE is included in the license archive. Matter certification and use of Connectivity Standards Alliance trademarks require separate authorization; the software licenses do not grant those rights."))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_compliance(args: argparse.Namespace) -> tuple[Path, Path, Path]:
    project_description = Path(args.project_description).resolve()
    firmware = Path(args.firmware).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    description = json.loads(project_description.read_text(encoding="utf-8"))
    packages = collect_packages(description, args.frontend)
    stem = firmware.stem
    sbom_path = output_dir / f"{stem}-sbom.spdx.json"
    notice_path = output_dir / f"{stem}-THIRD_PARTY_NOTICES.txt"
    licenses_path = output_dir / f"{stem}-third-party-licenses.zip"

    sbom = build_spdx(description, args.frontend, firmware, packages)
    sbom_path.write_text(json.dumps(sbom, indent=2) + "\n", encoding="utf-8")
    write_notice(notice_path, firmware, args.frontend, description, packages)
    with zipfile.ZipFile(licenses_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for archive_name, source in collect_license_files(packages, args.frontend):
            archive.write(source, archive_name)
    return sbom_path, notice_path, licenses_path


def main() -> int:
    artifacts = build_compliance(parse_args())
    for artifact in artifacts:
        print(artifact)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
