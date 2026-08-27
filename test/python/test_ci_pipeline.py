# SPDX-License-Identifier: GPL-3.0-only
# Commercial licensing available under separate agreement; see LICENSING.md.
"""Regression tests for release, rolling, and GitHub Pages automation."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import re
import shutil
import subprocess
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path
from urllib.parse import urlparse

import pytest

from espectre_cli.idf_container import IDF_DOCKER_IMAGE


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = REPO_ROOT / ".github" / "scripts"
WORKFLOWS_DIR = REPO_ROOT / ".github" / "workflows"
PROTOCOL_HEADER = REPO_ROOT / "src" / "cpp" / "runtime" / "espectre_protocol.h"


def _ota_release_tags() -> tuple[str, str]:
    header = PROTOCOL_HEADER.read_text(encoding="utf-8")
    preview = re.search(r'ESPECTRE_OTA_RELEASE_TAG_PREVIEW\s*=\s*"([^"]+)"', header)
    develop = re.search(r'ESPECTRE_OTA_RELEASE_TAG_DEVELOP\s*=\s*"([^"]+)"', header)
    assert preview is not None and develop is not None
    return preview.group(1), develop.group(1)


def load_script(name: str):
    path = SCRIPTS_DIR / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"test_{name}", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_sdk_archives_and_manifest_are_reproducible(tmp_path: Path) -> None:
    builder = load_script("build_sdk_package")
    component_cmake = (REPO_ROOT / "src" / "cpp" / "CMakeLists.txt").read_text(encoding="utf-8")
    for dependency in (
        "mqtt",
        "app_update",
        "esp_http_client",
        "esp_http_server",
        "esp_https_ota",
        "esp-tls",
        "improv",
        "mdns",
    ):
        assert re.search(rf"(?m)^    {re.escape(dependency)}$", component_cmake)
    outputs = [tmp_path / "first", tmp_path / "second"]
    for output in outputs:
        args = argparse.Namespace(
            channel="release",
            version="3.0.0",
            release_tag="3.0.0",
            output_dir=str(output),
            commit="0123456789abcdef",
            source_date_epoch=1_800_000_000,
            url_prefix=None,
        )
        builder.build_sdk_package(args)

    first_files = sorted(path.name for path in outputs[0].iterdir())
    second_files = sorted(path.name for path in outputs[1].iterdir())
    assert first_files == second_files
    for filename in first_files:
        assert (outputs[0] / filename).read_bytes() == (outputs[1] / filename).read_bytes()

    manifest_path = next(outputs[0].glob("sdk-manifest-*.json"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["generated_at"] == "2027-01-15T08:00:00+00:00"
    assert manifest["install_surfaces"]["cmake"]["optional_source_groups"] == [
        "ESPECTRE_RUNTIME_FRONTEND_SUPPORT_SOURCES",
        "ESPECTRE_RUNTIME_ESP_IDF_MQTT_SOURCES",
        "ESPECTRE_RUNTIME_ESP_IDF_PROVISIONING_SOURCES",
        "ESPECTRE_RUNTIME_ESP_IDF_OTA_SOURCES",
        "ESPECTRE_RUNTIME_ESP_IDF_DIRECT_SOURCES",
    ]
    zip_path = next(outputs[0].glob("*.zip"))
    with zipfile.ZipFile(zip_path) as archive:
        archived = set(archive.namelist())
        component_cmake_name = next(
            name for name in archived if name.endswith("/src/cpp/CMakeLists.txt")
        )
        doxy_name = next(name for name in archived if name.endswith("/src/cpp/Doxyfile"))
        bundled_doxyfile = archive.read(doxy_name).decode("utf-8")
    bundle_root = component_cmake_name.removesuffix("/src/cpp/CMakeLists.txt")
    assert f"{bundle_root}/CMakeLists.txt" not in archived
    assert manifest["install_surfaces"]["esp_idf_component"]["component_root"] == "src/cpp"
    assert re.search(r"(?m)^OUTPUT_DIRECTORY\s*=\s*output\s*$", bundled_doxyfile)
    assert re.search(r"(?m)^PROJECT_NUMBER\s*=\s*3\.0\.0\s*$", bundled_doxyfile)
    assert re.search(r"(?m)^GENERATE_HTML\s*=\s*NO\s*$", bundled_doxyfile)
    assert re.search(r"(?m)^GENERATE_XML\s*=\s*YES\s*$", bundled_doxyfile)
    assert not any("/src/cpp/doxygen/" in path for path in archived)
    assert "docs/web/artifacts/sdk" not in bundled_doxyfile
    repo_doxyfile = (REPO_ROOT / "src" / "cpp" / "Doxyfile").read_text(encoding="utf-8")
    assert re.search(r"(?m)^OUTPUT_DIRECTORY\s*=\s*docs/web/artifacts/sdk\s*$", repo_doxyfile)
    assert re.search(r"(?m)^PROJECT_NUMBER\s*=\s*UNSTAMPED\s*$", repo_doxyfile)
    assert any(path.endswith("/THIRD_PARTY_NOTICES.md") for path in archived)
    for artifact in manifest["artifacts"]:
        assert artifact["sha256"] == file_sha256(outputs[0] / artifact["filename"])


def test_web_sdk_rejects_a_channel_mismatch_before_cleaning(tmp_path: Path) -> None:
    stage = load_script("stage_web_sdk")
    sdk_dir = tmp_path / "sdk"
    output_dir = tmp_path / "output"
    sdk_dir.mkdir()
    output_dir.mkdir()
    (sdk_dir / "sdk-manifest-release.json").write_text(
        json.dumps({"channel": "release"}), encoding="utf-8"
    )
    sentinel = output_dir / "index.html"
    sentinel.write_text("keep", encoding="utf-8")

    with pytest.raises(ValueError, match="channel mismatch"):
        stage.stage_web_sdk(
            argparse.Namespace(
                sdk_dir=str(sdk_dir),
                output_dir=str(output_dir),
                channel="preview",
            )
        )
    assert sentinel.read_text(encoding="utf-8") == "keep"


@pytest.mark.parametrize(
    ("version", "expected_stability", "production_ready"),
    [
        ("3.0.0", "final", True),
        ("3.0.0-rc1", "prerelease", False),
    ],
)
def test_release_sdk_page_exposes_version_stability(
    version: str, expected_stability: str, production_ready: bool
) -> None:
    stage = load_script("stage_web_sdk")
    manifest = {
        "channel": "release",
        "version": version,
        "package_version": version,
        "release_tag": version,
        "protocol_version": 1,
        "supported_esp_idf": ">=5.5.0",
        "commit": "0123456789abcdef",
        "artifacts": [
            {
                "url": f"https://example.invalid/espectre-sdk-{version}.zip",
                "format": "zip",
                "filename": f"espectre-sdk-{version}.zip",
            }
        ],
        "install_surfaces": {
            "cmake": {
                "entrypoint": "src/cpp/espectre_sources.cmake",
                "optional_source_groups": [],
            },
            "esp_idf_component": {
                "component_root": "src/cpp",
                "cmake": "src/cpp/CMakeLists.txt",
                "kconfig": "src/cpp/Kconfig.projbuild",
            },
        },
    }

    page = stage.render_page(manifest, "release")

    assert f'data-sdk-stability="{expected_stability}"' in page
    assert ('data-sdk-production-ready="false"' in page) is not production_ready


@pytest.mark.parametrize("tag", ["v3.0.0", "03.0.0", "3.0.0-01", "3.0", "release"])
def test_release_validator_rejects_non_semver_tags(tag: str) -> None:
    validator = load_script("validate_release")
    with pytest.raises(ValueError, match="semantic versioning"):
        validator.validate(tag)


def test_release_validator_requires_a_finalized_matching_changelog(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    validator = load_script("validate_release")
    changelog = tmp_path / "CHANGELOG.md"
    monkeypatch.setattr(validator, "CHANGELOG", changelog)
    monkeypatch.setattr(validator, "detect_git_version", lambda **_kwargs: "3.0.0-rc1")

    changelog.write_text("## [3.0.0-rc1] - Unreleased\n", encoding="utf-8")
    with pytest.raises(ValueError, match="not finalized"):
        validator.validate("3.0.0-rc1")

    changelog.write_text("## [3.0.0-rc1] - 2026-08-12\n", encoding="utf-8")
    validator.validate("3.0.0-rc1")


def test_unstamped_sdk_header_has_no_numeric_fallback() -> None:
    builder = load_script("build_sdk_package")
    header = (REPO_ROOT / "src" / "cpp" / "runtime" / "espectre_sdk_version.h").read_text(
        encoding="utf-8"
    )
    assert "#define ESPECTRE_SDK_VERSION_STRING" not in header
    assert "ESPectre SDK version is unresolved" in header
    with pytest.raises(ValueError, match="Unable to detect ESPECTRE_SDK_VERSION_STRING"):
        builder.detect_sdk_version()


def test_git_version_cmake_reads_environment_before_git_describe() -> None:
    cmake = (REPO_ROOT / "src" / "cpp" / "espectre_git_version.cmake").read_text(encoding="utf-8")
    env_index = cmake.index("ENV{ESPECTRE_GIT_VERSION}")
    describe_index = cmake.index('git describe --tags --match "[0-9]*" --abbrev=7')
    workspace_index = cmake.index("ENV{GITHUB_WORKSPACE}")
    assert env_index < describe_index
    assert describe_index < cmake.index("header is not stamped")
    assert workspace_index < cmake.index("header is not stamped")


def test_native_loop_processes_wifi_events_before_frontend_updates() -> None:
    source = (
        REPO_ROOT / "src" / "cpp" / "frontend" / "native" / "app" / "main" / "app_main.cpp"
    ).read_text(encoding="utf-8")
    loop = source[source.index("void espectre_loop_task") : source.index("bool init_wifi_station")]

    assert loop.index("g_wifi_manager.loop();") < loop.index("g_frontend->loop();")


def test_esphome_forwards_numeric_project_version_to_sdk_cmake() -> None:
    pytest.importorskip("esphome")
    path = (
        REPO_ROOT
        / "src"
        / "cpp"
        / "frontend"
        / "esphome"
        / "components"
        / "espectre"
        / "__init__.py"
    )
    spec = importlib.util.spec_from_file_location("espectre_esphome_component", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    module._git_describe_version = lambda _root: "2.8.0-1-gabcdef0"
    assert module.resolve_espectre_git_version("9.9.9-ci-gdeadbee") == "9.9.9-ci-gdeadbee"
    assert module.resolve_espectre_git_version("main") == "2.8.0-1-gabcdef0"

    module._git_describe_version = lambda _root: None
    assert module.resolve_espectre_git_version("main") is None


def test_git_version_cmake_honors_environment(tmp_path: Path) -> None:
    cmake = shutil.which("cmake")
    if cmake is None:
        pytest.skip("cmake is not installed")
    script = tmp_path / "probe.cmake"
    cmake_file = (REPO_ROOT / "src" / "cpp" / "espectre_git_version.cmake").as_posix()
    script.write_text(
        f'include("{cmake_file}")\n'
        "if(NOT ESPECTRE_GIT_VERSION STREQUAL \"2.8.0-99-gdeadbee\")\n"
        "  message(FATAL_ERROR \"got ${ESPECTRE_GIT_VERSION}\")\n"
        "endif()\n",
        encoding="utf-8",
    )
    env = os.environ.copy()
    env["ESPECTRE_GIT_VERSION"] = "2.8.0-99-gdeadbee"
    result = subprocess.run(
        [cmake, "-P", str(script)],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )
    assert result.returncode == 0, result.stderr or result.stdout


def test_detect_git_version_ignores_rolling_tags(monkeypatch: pytest.MonkeyPatch) -> None:
    detector = load_script("detect_git_version")

    def fake_run(command, **_kwargs):
        assert command == list(detector.GIT_DESCRIBE_CMD)
        class Result:
            returncode = 0
            stdout = "2.8.0-237-g7439944\n"
            stderr = ""

        return Result()

    monkeypatch.setattr(detector.subprocess, "run", fake_run)
    assert detector.detect_git_version() == "2.8.0-237-g7439944"
    assert detector.parse_version_core("2.8.0-237-g7439944") == (2, 8, 0)
    assert detector.parse_version_core("3.0.0-rc1") == (3, 0, 0)
    with pytest.raises(ValueError, match="numeric MAJOR.MINOR.PATCH"):
        detector.parse_version_core("preview")


def test_sdk_snapshot_stamps_git_describe_identity(tmp_path: Path) -> None:
    builder = load_script("build_sdk_package")
    _, develop_tag = _ota_release_tags()
    args = argparse.Namespace(
        channel="develop",
        version="2.8.0-237-g7439944",
        release_tag=develop_tag,
        output_dir=str(tmp_path),
        commit="7439944d441e9a8e485a1d610d99265d743e93f8",
        source_date_epoch=1_800_000_000,
        url_prefix=None,
    )
    manifest = builder.build_sdk_package(args)
    assert manifest["version"] == "2.8.0-237-g7439944"
    assert manifest["package_version"] == "2.8.0-237-g7439944"
    assert manifest["sdk_version"] == "2.8.0-237-g7439944"
    assert manifest["release_tag"] == develop_tag
    assert manifest["supported_esp_idf"] == ">=5.5.0"
    assert manifest["artifacts"][0]["filename"] == "espectre-sdk-develop.tar.gz"
    assert f"/releases/download/{develop_tag}/" in manifest["artifacts"][0]["url"]

    zip_path = tmp_path / "espectre-sdk-develop.zip"
    with zipfile.ZipFile(zip_path) as archive:
        header_name = next(name for name in archive.namelist() if name.endswith("/src/cpp/runtime/espectre_sdk_version.h"))
        header = archive.read(header_name).decode("utf-8")
        yml_name = next(name for name in archive.namelist() if name.endswith("/src/cpp/idf_component.yml"))
        yml = archive.read(yml_name).decode("utf-8")
        doxy_name = next(name for name in archive.namelist() if name.endswith("/src/cpp/Doxyfile"))
        bundled_doxyfile = archive.read(doxy_name).decode("utf-8")
    assert '#define ESPECTRE_SDK_VERSION_STRING "2.8.0-237-g7439944"' in header
    assert "#define ESPECTRE_SDK_VERSION_MAJOR 2" in header
    assert "ESPectre SDK version is unresolved" not in header
    assert 'version: "2.8.0-237-g7439944"' in yml
    assert 'version: ">=5.5.0"' in yml
    assert "https://github.com/improv-wifi/sdk-cpp.git" in yml
    assert "version: 17898613a1c17062ca5af295ceb639b16b4930bf" in yml
    assert 'espressif/mdns:\n    version: "^1.9.0"' in yml
    assert re.search(r"(?m)^PROJECT_NUMBER\s*=\s*2\.8\.0-237-g7439944\s*$", bundled_doxyfile)
    for relative_path in (
        "src/cpp/frontend/native/espectre/idf_component.yml",
        "src/cpp/frontend/matter/espectre/idf_component.yml",
        "src/cpp/frontend/matter/app/main/idf_component.yml",
    ):
        frontend_manifest = (REPO_ROOT / relative_path).read_text(encoding="utf-8")
        assert 'version: ">=5.5.0"' in frontend_manifest

    native_manifest = (
        REPO_ROOT / "src" / "cpp" / "frontend" / "native" / "espectre" / "idf_component.yml"
    ).read_text(encoding="utf-8")
    assert "https://github.com/improv-wifi/sdk-cpp.git" in native_manifest
    assert "version: 17898613a1c17062ca5af295ceb639b16b4930bf" in native_manifest
    assert 'espressif/mdns:\n    version: "^1.9.0"' in native_manifest


def test_generate_sdk_api_stamps_a_working_copy_without_mutating_the_repo(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    generator = load_script("generate_sdk_api")
    repo_doxyfile = (REPO_ROOT / "src" / "cpp" / "Doxyfile").read_text(encoding="utf-8")
    stamped_versions: list[str] = []
    api_output = tmp_path / "sdk" / "api"
    api_output.mkdir(parents=True)
    stale_page = api_output / "stale-internal-type.html"
    stale_page.write_text("stale", encoding="utf-8")

    def stamped_output(path: Path) -> Path:
        source = path.read_text(encoding="utf-8")
        match = re.search(r"(?m)^OUTPUT_DIRECTORY\s*=\s*(\S+)\s*$", source)
        assert match is not None
        return Path(match.group(1))

    def fake_doxygen(path: Path) -> None:
        stamped = path.read_text(encoding="utf-8")
        match = re.search(r"(?m)^PROJECT_NUMBER\s*=\s*(\S+)\s*$", stamped)
        assert match is not None
        stamped_versions.append(match.group(1))
        xml = stamped_output(path) / "xml"
        xml.mkdir(parents=True)
        (xml / "index.xml").write_text(
            '<doxygenindex version="1.17">'
            '<compound refid="classespectre_1_1_runtime_frontend_controller" kind="class"><name>espectre::RuntimeFrontendController</name></compound>'
            '<compound refid="espectre__sdk__version_8h" kind="file"><name>espectre_sdk_version.h</name></compound>'
            "</doxygenindex>",
            encoding="utf-8",
        )
        (xml / "classespectre_1_1_runtime_frontend_controller.xml").write_text(
            '<doxygen><compounddef><sectiondef kind="private-func"><memberdef prot="private" kind="function"/></sectiondef></compounddef></doxygen>',
            encoding="utf-8",
        )

    def fake_mcss(path: Path, _root: Path | None) -> None:
        output = stamped_output(path)
        assert 'prot="private"' not in (
            output / "xml" / "classespectre_1_1_runtime_frontend_controller.xml"
        ).read_text(encoding="utf-8")
        rendered = output / "rendered"
        rendered.mkdir()
        (rendered / "index.html").write_text(
            '<article data-api-reference-fragment="index"><a href="classespectre_1_1_runtime_frontend_controller.html">Controller</a></article>',
            encoding="utf-8",
        )
        (rendered / "classespectre_1_1_runtime_frontend_controller.html").write_text(
            '<article data-api-reference-fragment="classespectre_1_1_runtime_frontend_controller"><h1>Controller</h1><nav class="m-block m-default"><h3>Local navigation</h3></nav><section id="members"><h2>Members</h2></section></article>',
            encoding="utf-8",
        )
        (rendered / "files.html").write_text(
            '<article data-api-reference-fragment="files"><h1>Files</h1></article>',
            encoding="utf-8",
        )
        (rendered / "espectre__sdk__version_8h.html").write_text(
            '<article data-api-reference-fragment="espectre__sdk__version_8h"><section class="m-doc-details">Version defines</section></article>',
            encoding="utf-8",
        )

    monkeypatch.setattr(generator, "run_doxygen", fake_doxygen)
    monkeypatch.setattr(generator, "run_mcss", fake_mcss)
    monkeypatch.setattr(generator, "API_OUTPUT_DIR", api_output)
    version = generator.generate_sdk_api("3.0.0-12-gabcdef1")
    assert version == "3.0.0-12-gabcdef1"
    assert stamped_versions == ["3.0.0-12-gabcdef1"]
    assert not stale_page.exists()
    manifest = json.loads((api_output / "api-index.json").read_text(encoding="utf-8"))
    assert manifest["sdk_version"] == "3.0.0-12-gabcdef1"
    assert manifest["renderer"] == "m.css"
    entries = {entry["refid"]: entry for entry in manifest["entries"]}
    assert entries["index"]["discoverable"] is True
    assert entries["classespectre_1_1_runtime_frontend_controller"]["discoverable"] is True
    assert entries["files"]["discoverable"] is False
    assert entries["espectre__sdk__version_8h"]["discoverable"] is True
    controller_fragment = (
        api_output / "fragments" / "classespectre_1_1_runtime_frontend_controller.html"
    ).read_text(encoding="utf-8")
    assert '<nav class="m-block' not in controller_fragment
    assert '<section id="members">' in controller_fragment
    assert (api_output / "fragments" / "index.html").is_file()
    assert (REPO_ROOT / "src" / "cpp" / "Doxyfile").read_text(encoding="utf-8") == repo_doxyfile
    assert re.search(r"(?m)^PROJECT_NUMBER\s*=\s*UNSTAMPED\s*$", repo_doxyfile)


def test_sdk_bundle_rewrites_the_repo_doxyfile_preamble(tmp_path: Path) -> None:
    packager = load_script("build_sdk_package")
    bundled = tmp_path / "Doxyfile"
    bundled.write_text((REPO_ROOT / "src" / "cpp" / "Doxyfile").read_text(encoding="utf-8"))
    packager.rewrite_bundle_doxyfile(bundled, "3.0.0")
    rewritten = bundled.read_text(encoding="utf-8")
    assert "# Usage, from the unpacked SDK bundle root:" in rewritten
    assert re.search(r"(?m)^OUTPUT_DIRECTORY\s*=\s*output\s*$", rewritten)
    assert re.search(r"(?m)^PROJECT_NUMBER\s*=\s*3\.0\.0\s*$", rewritten)


def test_generate_sdk_api_requires_the_pinned_doxygen_version(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    generator = load_script("generate_sdk_api")
    monkeypatch.setattr(generator, "detect_doxygen_version", lambda: "1.9.8")
    with pytest.raises(ValueError, match="1.17.0"):
        generator.require_doxygen_version()
    monkeypatch.setattr(generator, "detect_doxygen_version", lambda: "1.17.0")
    generator.require_doxygen_version()


def test_indexnow_retries_transient_failures_and_sends_the_sitemap(tmp_path: Path) -> None:
    indexnow = load_script("notify_indexnow")
    sitemap = tmp_path / "sitemap.xml"
    sitemap.write_text(
        '<?xml version="1.0"?><urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">'
        "<url><loc>https://espectre.dev/</loc></url>"
        "<url><loc>https://espectre.dev/sdk/</loc></url></urlset>",
        encoding="utf-8",
    )
    urls = indexnow.sitemap_urls(sitemap)
    calls: list[tuple[object, float]] = []
    sleeps: list[float] = []

    class Response:
        status = 202

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

    def request(request, *, timeout):
        calls.append((request, timeout))
        if len(calls) < 3:
            raise OSError("temporary failure")
        return Response()

    indexnow.notify(urls, request_fn=request, sleep_fn=sleeps.append, timeout=7.5)

    assert len(calls) == 3
    assert sleeps == [1.0, 2.0]
    assert all(timeout == 7.5 for _, timeout in calls)
    payload = json.loads(calls[-1][0].data)
    assert payload["host"] == "espectre.dev"
    assert payload["urlList"] == urls


def test_sitemap_lastmod_dates_use_utc() -> None:
    sitemap_builder = load_script("build_sitemap")
    assert sitemap_builder.normalized_date("2026-08-19T00:41:27+02:00") == "2026-08-18"
    assert sitemap_builder.normalized_date("2026-08-18T23:30:00Z") == "2026-08-18"
    assert sitemap_builder.normalized_date("2026-08-19T00:00:00+00:00") == "2026-08-19"
    assert sitemap_builder.normalized_date("2026-08-19") == "2026-08-19"


def test_sitemap_builder_uses_git_and_sdk_manifest_dates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sitemap_builder = load_script("build_sitemap")
    web_root = tmp_path / "web"
    release_dir = web_root / "artifacts" / "sdk" / "release"
    preview_dir = web_root / "artifacts" / "sdk" / "preview"
    develop_dir = web_root / "artifacts" / "sdk" / "develop"
    release_dir.mkdir(parents=True)
    preview_dir.mkdir(parents=True)
    develop_dir.mkdir(parents=True)
    (release_dir / "sdk-manifest-release.json").write_text(
        json.dumps({"channel": "release", "generated_at": "2026-08-01T09:30:00Z"}),
        encoding="utf-8",
    )
    (preview_dir / "sdk-manifest-preview.json").write_text(
        json.dumps({"channel": "preview", "generated_at": "2026-08-12T10:45:00+00:00"}),
        encoding="utf-8",
    )
    (develop_dir / "sdk-manifest-develop.json").write_text(
        json.dumps({"channel": "develop", "generated_at": "2026-08-14T11:15:00+00:00"}),
        encoding="utf-8",
    )
    monkeypatch.setattr(sitemap_builder, "WEB_ROOT", web_root)

    def fake_git_date(paths):
        if paths == (sitemap_builder.SDK_PAGE_BUILDER,):
            return "2026-08-10"
        if sitemap_builder.DOXYFILE in paths:
            return "2026-08-08"
        return "2026-08-09"

    monkeypatch.setattr(sitemap_builder, "latest_git_date", fake_git_date)
    sitemap = tmp_path / "sitemap.xml"
    sitemap.write_text(
        '<?xml version="1.0"?><urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">'
        "<url><loc>https://espectre.dev/</loc><changefreq>daily</changefreq></url>"
        "<url><loc>https://espectre.dev/sdk/api/</loc></url>"
        "<url><loc>https://espectre.dev/artifacts/sdk/release/</loc></url>"
        "<url><loc>https://espectre.dev/artifacts/sdk/preview/</loc></url>"
        "<url><loc>https://espectre.dev/artifacts/sdk/develop/</loc></url>"
        "</urlset>",
        encoding="utf-8",
    )
    output = tmp_path / "generated.xml"
    sitemap_builder.build_sitemap(sitemap, output)

    root = ET.parse(output).getroot()
    namespace = {"s": sitemap_builder.SITEMAP_NAMESPACE}
    entries = {
        entry.findtext("s:loc", namespaces=namespace): entry.findtext("s:lastmod", namespaces=namespace)
        for entry in root.findall("s:url", namespace)
    }
    assert entries == {
        "https://espectre.dev/": "2026-08-09",
        "https://espectre.dev/sdk/api/": "2026-08-08",
        "https://espectre.dev/artifacts/sdk/release/": "2026-08-10",
        "https://espectre.dev/artifacts/sdk/preview/": "2026-08-12",
        "https://espectre.dev/artifacts/sdk/develop/": "2026-08-14",
    }
    assert root.findall("s:url/s:changefreq", namespace) == []

    with pytest.raises(ValueError, match="must use separate paths"):
        sitemap_builder.build_sitemap(sitemap, sitemap)


def test_pages_build_outputs_do_not_overlap_committed_sources() -> None:
    indexnow = load_script("notify_indexnow")
    static_pages = load_script("build_static_pages")
    sitemap_builder = load_script("build_sitemap")
    source_paths = set(
        subprocess.run(
            [
                "git",
                "ls-files",
                "--cached",
                "--others",
                "--exclude-standard",
                ".github/scripts",
                "docs/web",
            ],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.splitlines()
    )
    source_paths.difference_update(
        subprocess.run(
            ["git", "ls-files", "--deleted", "docs/web"],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.splitlines()
    )
    generated_paths = {
        "docs/web/artifacts",
        "docs/web/node_modules",
        "docs/web/sitemap.xml",
        "docs/web/vendor",
        *(
            f"docs/web/{page['output'].strip('/')}"
            for page in static_pages.PAGES
        ),
    }

    assert sitemap_builder.DEFAULT_SITEMAP_TEMPLATE == (
        REPO_ROOT / ".github" / "scripts" / "sitemap.template.xml"
    )
    assert sitemap_builder.DEFAULT_SITEMAP_OUTPUT == (
        REPO_ROOT / "docs" / "web" / "sitemap.xml"
    )
    assert indexnow.DEFAULT_SITEMAP == sitemap_builder.DEFAULT_SITEMAP_TEMPLATE
    assert ".github/scripts/sitemap.template.xml" in source_paths
    for generated_path in generated_paths:
        assert not any(
            path == generated_path or path.startswith(f"{generated_path}/")
            for path in source_paths
        ), f"Pages build output overlaps committed source: {generated_path}"


def test_pages_verifier_spa_routes_match_the_route_registry() -> None:
    verifier = load_script("verify_web_build")
    verifier.verify_spa_routes()


def test_pages_verifier_requires_api_reference_to_show_sdk_version(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    verifier = load_script("verify_web_build")
    monkeypatch.setattr(verifier, "WEB_ROOT", tmp_path)
    version = "2.8.0-237-g7439944"
    monkeypatch.setattr(verifier, "detect_git_version", lambda: version)
    api = tmp_path / "artifacts" / "sdk" / "api"
    fragments = api / "fragments"
    fragments.mkdir(parents=True)
    refids = (
        "classespectre_1_1_runtime_frontend_controller",
        "structespectre_1_1_runtime_config",
        "classespectre_1_1_i_runtime_listener",
    )
    entries = []
    for refid in refids:
        fragment = f"fragments/{refid}.html"
        (api / fragment).write_text("<article>API reference</article>", encoding="utf-8")
        entries.append({"refid": refid, "fragment": fragment, "discoverable": True})
    manifest = {
        "sdk_version": version,
        "renderer": "m.css",
        "renderer_revision": "0123456789abcdef",
        "entries": entries,
    }
    manifest_path = api / "api-index.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    verifier.verify_sdk_api_version()
    manifest["sdk_version"] = "UNSTAMPED"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="does not show version"):
        verifier.verify_sdk_api_version()


def test_pages_verifier_rejects_missing_spa_routes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    verifier = load_script("verify_web_build")
    monkeypatch.setattr(verifier, "WEB_ROOT", tmp_path)
    (tmp_path / "index.html").write_text('<main data-page="home"></main>', encoding="utf-8")
    registry_dir = tmp_path / "assets" / "js"
    registry_dir.mkdir(parents=True)
    (registry_dir / "route-registry.js").write_text(
        "{ name: 'home' }\n{ name: 'device' }\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match=r"missing=\['device'\]"):
        verifier.verify_spa_routes()


def test_pages_verifier_requires_every_registered_static_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    verifier = load_script("verify_web_build")
    monkeypatch.setattr(verifier, "WEB_ROOT", tmp_path)
    registry_dir = tmp_path / "assets" / "js"
    registry_dir.mkdir(parents=True)
    (registry_dir / "route-registry.js").write_text(
        "{ name: 'guides', staticPath: '/guides/' }\n"
        "{ name: 'guide-home-assistant', staticPath: '/guides/home-assistant/' }\n",
        encoding="utf-8",
    )
    guides_dir = tmp_path / "guides"
    guides_dir.mkdir()
    (guides_dir / "index.html").write_text("<main></main>", encoding="utf-8")

    with pytest.raises(
        FileNotFoundError,
        match="guides/home-assistant/index.html",
    ):
        verifier.verify_generated_pages()


def test_generated_pages_have_sitemap_lastmod_ownership() -> None:
    static_pages = load_script("build_static_pages")
    sitemap_builder = load_script("build_sitemap")
    verifier = load_script("verify_web_build")

    namespace = {"s": sitemap_builder.SITEMAP_NAMESPACE}
    root = ET.parse(REPO_ROOT / ".github" / "scripts" / "sitemap.template.xml").getroot()
    sitemap_paths = {
        urlparse(location).path
        for location in (
            entry.findtext("s:loc", namespaces=namespace)
            for entry in root.findall("s:url", namespace)
        )
        if location
    }
    generated_pages = {
        f"/{page['output'].strip('/')}/": Path("docs/web") / page["source"]
        for page in static_pages.PAGES
    }

    assert len(generated_pages) == len(static_pages.PAGES), "Generated page routes must be unique"
    assert not generated_pages.keys() - sitemap_paths, (
        "Generated pages missing from the sitemap: "
        f"{sorted(generated_pages.keys() - sitemap_paths)}"
    )
    assert sitemap_paths == verifier.EXPECTED_SITEMAP_PATHS

    for route, source in generated_pages.items():
        assert route in sitemap_builder.ROUTE_SOURCES, (
            f"Generated page {route} has no sitemap lastmod ownership mapping"
        )
        ownership = sitemap_builder.ROUTE_SOURCES[route]
        assert source in ownership, f"Sitemap lastmod for {route} does not track {source}"
        assert sitemap_builder.STATIC_PAGE_BUILDER in ownership, (
            f"Sitemap lastmod for {route} does not track the static page builder"
        )


def test_sitemap_verifier_requires_accurate_dates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    verifier = load_script("verify_web_build")
    monkeypatch.setattr(verifier, "WEB_ROOT", tmp_path)
    namespace = verifier.SITEMAP_NAMESPACE
    entries = "".join(
        f"<url><loc>https://espectre.dev{path}</loc>"
        + ("" if path in {"/artifacts/sdk/preview/", "/artifacts/sdk/release/", "/artifacts/sdk/develop/"} else "<lastmod>2026-08-12</lastmod>")
        + "</url>"
        for path in sorted(verifier.EXPECTED_SITEMAP_PATHS)
    )
    sitemap = tmp_path / "sitemap.xml"
    sitemap.write_text(
        f'<?xml version="1.0"?><urlset xmlns="{namespace}">{entries}</urlset>',
        encoding="utf-8",
    )
    verifier.verify_sitemap(require_preview=False, require_release=False, require_develop=False)

    future = sitemap.read_text(encoding="utf-8").replace(
        "<lastmod>2026-08-12</lastmod>",
        "<lastmod>2099-01-01</lastmod>",
        1,
    )
    sitemap.write_text(future, encoding="utf-8")
    with pytest.raises(ValueError, match="lastmod is in the future"):
        verifier.verify_sitemap(require_preview=False, require_release=False, require_develop=False)
    sitemap.write_text(
        future.replace("<lastmod>2099-01-01</lastmod>", "<lastmod>2026-08-12</lastmod>", 1),
        encoding="utf-8",
    )

    source = sitemap.read_text(encoding="utf-8").replace(
        "<lastmod>2026-08-12</lastmod>",
        "<lastmod>2026-08-12</lastmod><changefreq>daily</changefreq>",
        1,
    )
    sitemap.write_text(source, encoding="utf-8")
    with pytest.raises(ValueError, match="must not contain changefreq"):
        verifier.verify_sitemap(require_preview=False, require_release=False, require_develop=False)


def test_pages_verifier_enforces_exact_artifact_contracts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    verifier = load_script("verify_web_build")
    monkeypatch.setattr(verifier, "WEB_ROOT", tmp_path)
    firmware_dir = tmp_path / "artifacts" / "firmware" / "preview"
    sdk_dir = tmp_path / "artifacts" / "sdk" / "preview"
    firmware_dir.mkdir(parents=True)
    sdk_dir.mkdir(parents=True)

    frontends = {}
    for frontend in sorted(verifier.EXPECTED_FRONTENDS):
        artifacts = []
        for chip in sorted(verifier.EXPECTED_CHIPS_BY_FRONTEND[frontend]):
            filename = f"espectre-{frontend}-{chip}.bin"
            (firmware_dir / filename).write_bytes(b"firmware")
            artifacts.append({"build_type": "factory", "chip": chip, "filename": filename})
        frontends[frontend] = {"artifacts": artifacts}
    firmware_manifest = {"channel": "preview", "frontends": frontends}
    firmware_manifest_path = firmware_dir / "firmware-manifest-preview.json"
    firmware_manifest_path.write_text(json.dumps(firmware_manifest), encoding="utf-8")
    verifier.verify_firmware_channel("preview")

    frontends["native"]["artifacts"].append(frontends["native"]["artifacts"][0])
    firmware_manifest_path.write_text(json.dumps(firmware_manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate firmware"):
        verifier.verify_firmware_channel("preview")

    (sdk_dir / "index.html").write_text("SDK", encoding="utf-8")
    sdk_manifest_path = sdk_dir / "sdk-manifest-preview.json"
    sdk_manifest = {
        "channel": "preview",
        "artifacts": [
            {"format": "tar.gz", "sha256": "a" * 64},
            {"format": "zip", "sha256": "b" * 64},
        ],
    }
    sdk_manifest_path.write_text(json.dumps(sdk_manifest), encoding="utf-8")
    verifier.verify_sdk_channel("preview")
    sdk_manifest["artifacts"][0]["sha256"] = "invalid"
    sdk_manifest_path.write_text(json.dumps(sdk_manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="SHA-256"):
        verifier.verify_sdk_channel("preview")

    with pytest.raises(ValueError, match="escapes"):
        verifier.require_file("../outside")


def test_workflows_keep_publication_and_supply_chain_guardrails() -> None:
    workflow_sources = {
        path.name: path.read_text(encoding="utf-8")
        for path in sorted(WORKFLOWS_DIR.glob("*.yml"))
    }
    combined = "\n".join(workflow_sources.values())
    assert "ubuntu-latest" not in combined
    assert combined.count("runs-on:") == combined.count("timeout-minutes:")

    external_action = re.compile(r"^\s*uses:\s*([^./\s][^@\s]*)@([^\s#]+)", re.MULTILINE)
    refs = external_action.findall(combined)
    assert refs
    assert all(re.fullmatch(r"[0-9a-f]{40}", ref) for _, ref in refs)

    ci = workflow_sources["ci.yml"]
    snapshot = workflow_sources["snapshot.yml"]
    release = workflow_sources["release.yml"]
    assert "HEAD~1" not in snapshot
    assert "gh release delete" not in snapshot
    assert "git.getRef" in snapshot
    assert "git.updateRef" in snapshot and "git.createRef" in snapshot
    assert "workflow_dispatch:" in snapshot
    assert "ci_run_id:" in snapshot
    assert "needs.validate-run.outputs.run_id" in snapshot
    assert "github.event.workflow_run.id" not in snapshot
    assert "validate-release:" in release
    assert "No successful main CI push run" in release
    assert "git merge-base --is-ancestor" in release
    preview_tag, develop_tag = _ota_release_tags()
    assert re.search(rf'(?m)^              echo "tag={re.escape(develop_tag)}"$', snapshot)
    assert re.search(rf'(?m)^              echo "tag={re.escape(preview_tag)}"$', snapshot)
    assert f"gh release view {preview_tag}" in ci
    assert f"gh release view {develop_tag}" in ci
    assert f"gh release view {preview_tag}" in release
    assert f"gh release view {develop_tag}" in release
    assert f"gh release view {develop_tag}" in snapshot
    assert re.search(rf'(?m)^              echo "release_tag={re.escape(develop_tag)}"$', ci)
    assert re.search(rf'(?m)^              echo "release_tag={re.escape(preview_tag)}"$', ci)
    assert "detect_git_version.py" in ci
    assert "detect_git_version.py" in snapshot
    assert "detect_git_version.py" in release
    assert "ESPECTRE_GIT_VERSION: ${{ steps.git-version.outputs.version }}" in ci
    assert "ESPECTRE_GIT_VERSION: ${{ github.ref_name }}" in release
    for source in (ci, snapshot, release):
        assert "uses: ./.github/actions/build-pages" in source
        assert "fetch-depth: 0" in source
        assert "--output-dir docs/web/artifacts/firmware/release" in source
        assert "--channel release" in source
        assert "--url-prefix /artifacts/firmware/release" in source
    for source in (snapshot, release):
        assert 'require-release: "true"' in source
    pages_action = (REPO_ROOT / ".github" / "actions" / "build-pages" / "action.yml").read_text(
        encoding="utf-8"
    )
    assert ".github/scripts/build_sitemap.py" in pages_action
    assert ".github/scripts/generate_sdk_api.py" in pages_action
    assert "doxygen src/cpp/Doxyfile" not in pages_action
    generator = load_script("generate_sdk_api")
    assert generator.REQUIRED_DOXYGEN_VERSION == "1.17.0"
    assert f'version="{generator.REQUIRED_DOXYGEN_VERSION}"' in pages_action
    assert "doxygen-${version}.linux.bin.tar.gz" in pages_action
    assert "apt-get install -y --no-install-recommends doxygen" not in pages_action

    for script_name in (
        "build_matter_firmware.sh",
        "build_native_firmware.sh",
    ):
        source = (SCRIPTS_DIR / script_name).read_text(encoding="utf-8")
        assert IDF_DOCKER_IMAGE in source
        assert 'BUILD_DIR="build-container-${' in source
        assert 'detect_git_version.py' in source
        assert '-e ESPECTRE_GIT_VERSION="${ESPECTRE_GIT_VERSION}"' in source
        assert ".espectre-requirements-\\${REQUIREMENTS_HASH}" in source
        assert "--backend local" in source


def test_website_sources_integrate_sdk_api_fragments_in_portal_page() -> None:
    sdk_landing = (REPO_ROOT / "docs" / "web" / "content" / "sdk.html").read_text(
        encoding="utf-8"
    )
    api_orientation = (
        REPO_ROOT / "docs" / "web" / "content" / "sdk" / "api.html"
    ).read_text(encoding="utf-8")

    assert 'href="/sdk/api/" class="doc-link"' in sdk_landing
    assert 'href="/sdk/api/" class="btn-secondary"' in sdk_landing
    assert 'data-api-reference-browser' in api_orientation
    assert 'data-api-index="/artifacts/sdk/api/api-index.json"' in api_orientation
    assert 'data-api-reference-content' in api_orientation
    assert 'data-api-reference-picker' in api_orientation
    assert 'data-api-reference-filter' in api_orientation
    assert 'data-api-reference-results' in api_orientation
    assert 'data-api-reference-toggle' not in api_orientation
    assert 'data-page-toc' in api_orientation
    assert 'data-page-path="sdk"' in api_orientation
    assert 'api-reference-index' not in api_orientation
    assert '<iframe' not in api_orientation
