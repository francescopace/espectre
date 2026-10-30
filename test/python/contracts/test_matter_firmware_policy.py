# SPDX-License-Identifier: GPL-3.0-only
# Commercial licensing available under separate agreement; see LICENSING.md.
"""Enforce the Matter firmware identity and partition policy."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from tools.lib.repo_paths import repo_root


REPO_ROOT = repo_root()
MATTER_APP = REPO_ROOT / "src" / "cpp" / "frontend" / "matter" / "app"
VERSION_MODULE = (MATTER_APP / "espectre_matter_software_version.cmake").as_posix()

ORDERED_IDENTITIES = [
    "2.8.0",
    "2.8.0-280-gac7af68",
    "3.0.0-rc1",
    "3.0.0-rc2",
    "3.0.0-rc2-5-gabcdef0",
    "3.0.0-rc3",
    "3.0.0",
    "3.0.0-36-g7e7b8a0",
    "3.0.1",
    "3.1.0-rc1",
    "3.1.0",
    "4.0.0",
]


def _run_cmake(tmp_path: Path, body: str) -> str:
    cmake = shutil.which("cmake")
    if cmake is None:
        pytest.skip("cmake is not installed")
    script = tmp_path / "probe.cmake"
    script.write_text(f'include("{VERSION_MODULE}")\n{body}', encoding="utf-8")
    result = subprocess.run([cmake, "-P", str(script)], check=False, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr or result.stdout
    return result.stderr


def _encode(tmp_path: Path, identities: list[str]) -> dict[str, int]:
    body = "".join(
        f'espectre_matter_software_version("{identity}" value)\n'
        f'message("VERSION {identity} ${{value}}")\n'
        for identity in identities
    )
    values = {}
    for line in _run_cmake(tmp_path, body).splitlines():
        if line.startswith("VERSION "):
            _, identity, value = line.split()
            values[identity] = int(value)
    return values


def test_software_version_follows_release_order(tmp_path: Path) -> None:
    values = _encode(tmp_path, ORDERED_IDENTITIES)
    encoded = [values[identity] for identity in ORDERED_IDENTITIES]
    assert encoded == sorted(encoded)
    assert len(set(encoded)) == len(encoded)
    assert all(0 < value < 2**32 for value in encoded)
    assert values["3.0.0"] == (3 << 27) | (15 << 12)


def test_unencodable_identities_report_zero(tmp_path: Path) -> None:
    values = _encode(tmp_path, ["0.0.0", "3.1.0-beta1", "3.0.0-rc10", "64.0.0", "3.0.0-4096-gabcdef0"])
    assert set(values.values()) == {0}


def test_build_sets_the_number_esp_matter_reports() -> None:
    # esp-matter ignores CONFIG_DEVICE_SOFTWARE_VERSION_NUMBER unless the app
    # version comes from Kconfig; it reads PROJECT_VER_NUMBER instead.
    cmake_lists = (MATTER_APP / "CMakeLists.txt").read_text(encoding="utf-8")
    assert 'espectre_matter_software_version("${ESPECTRE_GIT_VERSION}" PROJECT_VER_NUMBER)' in cmake_lists
    assert cmake_lists.index("PROJECT_VER_NUMBER") < cmake_lists.index("project.cmake")


def _partitions() -> dict[str, tuple[str, str, int, int]]:
    rows = {}
    for line in (MATTER_APP / "partitions.csv").read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        name, kind, subtype, offset, size, *_ = (field.strip() for field in line.split(","))
        rows[name] = (kind, subtype, int(offset, 0), int(size, 0))
    return rows


def test_partition_offsets_keep_fabrics_and_onboarding_data() -> None:
    rows = _partitions()
    # Moving these would erase fabrics or onboarding codes on a USB update.
    assert rows["nvs"][2:] == (0x9000, 0x5000)
    assert rows["otadata"][2] == 0xE000
    assert rows["app0"][2] == 0x20000
    assert rows["matter_factory"][2:] == (0x3F0000, 0x1000)


def test_partition_table_reserves_product_factory_data() -> None:
    rows = _partitions()
    assert rows["fctry"][:2] == ("data", "nvs")
    assert rows["esp_secure_cert"][0] == "0x3F"
    spans = sorted((offset, offset + size, name) for name, (_, _, offset, size) in rows.items())
    for (_, end, name), (start, _, next_name) in zip(spans, spans[1:], strict=False):
        assert end <= start, f"{name} overlaps {next_name}"
    assert spans[-1][1] <= 4 * 1024 * 1024
    example = (MATTER_APP / "sdkconfig.factory.example").read_text(encoding="utf-8")
    assert 'CONFIG_CHIP_FACTORY_NAMESPACE_PARTITION_LABEL="fctry"' in example


def test_matter_defaults_leave_identity_to_the_build() -> None:
    for defaults in MATTER_APP.glob("sdkconfig.defaults*"):
        text = defaults.read_text(encoding="utf-8")
        assert "\nCONFIG_DEVICE_SOFTWARE_VERSION_NUMBER=" not in f"\n{text}"
    shared = (MATTER_APP / "sdkconfig.defaults").read_text(encoding="utf-8")
    assert "CONFIG_TEST_EVENT_TRIGGER_ENABLED=n" in shared.splitlines()
