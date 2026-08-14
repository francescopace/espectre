# SPDX-License-Identifier: GPL-3.0-only
# Commercial licensing available under separate agreement; see LICENSING.md.
"""Regression tests for release, snapshot, and GitHub Pages automation."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import re
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path
from urllib.parse import urlparse

import pytest

from espectre_cli.idf_container import IDF_DOCKER_IMAGE


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = REPO_ROOT / ".github" / "scripts"
WORKFLOWS_DIR = REPO_ROOT / ".github" / "workflows"


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
    outputs = [tmp_path / "first", tmp_path / "second"]
    for output in outputs:
        args = argparse.Namespace(
            channel="stable",
            version=builder.detect_sdk_version(),
            release_tag=builder.detect_sdk_version(),
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
    assert "LICENSES/Apache-2.0.txt" not in manifest["bundle"]["top_level_files"]
    zip_path = next(outputs[0].glob("*.zip"))
    with zipfile.ZipFile(zip_path) as archive:
        archived = set(archive.namelist())
        doxy_name = next(name for name in archived if name.endswith("/src/cpp/Doxyfile"))
        bundled_doxyfile = archive.read(doxy_name).decode("utf-8")
    assert re.search(r"(?m)^OUTPUT_DIRECTORY\s*=\s*output\s*$", bundled_doxyfile)
    assert "docs/web/artifacts/sdk" not in bundled_doxyfile
    repo_doxyfile = (REPO_ROOT / "src" / "cpp" / "Doxyfile").read_text(encoding="utf-8")
    assert re.search(r"(?m)^OUTPUT_DIRECTORY\s*=\s*docs/web/artifacts/sdk\s*$", repo_doxyfile)
    assert not any(path.endswith("/LICENSES/Apache-2.0.txt") for path in archived)
    assert any(path.endswith("/THIRD_PARTY_NOTICES.md") for path in archived)
    for artifact in manifest["artifacts"]:
        assert artifact["sha256"] == file_sha256(outputs[0] / artifact["filename"])


def test_web_sdk_rejects_a_channel_mismatch_before_cleaning(tmp_path: Path) -> None:
    stage = load_script("stage_web_sdk")
    sdk_dir = tmp_path / "sdk"
    output_dir = tmp_path / "output"
    sdk_dir.mkdir()
    output_dir.mkdir()
    (sdk_dir / "sdk-manifest-stable.json").write_text(
        json.dumps({"channel": "stable"}), encoding="utf-8"
    )
    sentinel = output_dir / "index.html"
    sentinel.write_text("keep", encoding="utf-8")

    with pytest.raises(ValueError, match="channel mismatch"):
        stage.stage_web_sdk(
            argparse.Namespace(
                sdk_dir=str(sdk_dir),
                output_dir=str(output_dir),
                channel="main",
            )
        )
    assert sentinel.read_text(encoding="utf-8") == "keep"


@pytest.mark.parametrize("tag", ["v3.0.0", "03.0.0", "3.0.0-01", "3.0", "release"])
def test_release_validator_rejects_non_semver_tags(tag: str) -> None:
    validator = load_script("validate_release")
    with pytest.raises(ValueError, match="semantic versioning"):
        validator.validate(tag)


def test_release_validator_requires_a_finalized_matching_changelog(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    validator = load_script("validate_release")
    version_header = tmp_path / "version.h"
    changelog = tmp_path / "CHANGELOG.md"
    version_header.write_text(
        '#define ESPECTRE_SDK_VERSION_STRING "3.0.0"\n', encoding="utf-8"
    )
    monkeypatch.setattr(validator, "SDK_VERSION_HEADER", version_header)
    monkeypatch.setattr(validator, "CHANGELOG", changelog)

    changelog.write_text("## [3.0.0-rc1] - Unreleased\n", encoding="utf-8")
    with pytest.raises(ValueError, match="not finalized"):
        validator.validate("3.0.0-rc1")

    changelog.write_text("## [3.0.0-rc1] - 2026-08-12\n", encoding="utf-8")
    validator.validate("3.0.0-rc1")


def test_indexnow_retries_transient_failures_and_sends_the_sitemap(tmp_path: Path) -> None:
    indexnow = load_script("notify_indexnow")
    sitemap = tmp_path / "sitemap.xml"
    sitemap.write_text(
        '<?xml version="1.0"?><urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">'
        "<url><loc>https://espectre.dev/</loc></url>"
        "<url><loc>https://espectre.dev/docs/</loc></url></urlset>",
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


def test_sitemap_builder_uses_git_and_sdk_manifest_dates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sitemap_builder = load_script("build_sitemap")
    web_root = tmp_path / "web"
    stable_dir = web_root / "artifacts" / "sdk" / "stable"
    main_dir = web_root / "artifacts" / "sdk" / "main"
    stable_dir.mkdir(parents=True)
    main_dir.mkdir(parents=True)
    (stable_dir / "sdk-manifest-stable.json").write_text(
        json.dumps({"channel": "stable", "generated_at": "2026-08-01T09:30:00Z"}),
        encoding="utf-8",
    )
    (main_dir / "sdk-manifest-main.json").write_text(
        json.dumps({"channel": "main", "generated_at": "2026-08-12T10:45:00+00:00"}),
        encoding="utf-8",
    )
    monkeypatch.setattr(sitemap_builder, "WEB_ROOT", web_root)

    def fake_git_date(paths):
        if paths == (sitemap_builder.SDK_PAGE_BUILDER,):
            return "2026-08-10"
        if paths == sitemap_builder.doxygen_sources():
            return "2026-08-08"
        return "2026-08-09"

    monkeypatch.setattr(sitemap_builder, "latest_git_date", fake_git_date)
    sitemap = tmp_path / "sitemap.xml"
    sitemap.write_text(
        '<?xml version="1.0"?><urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">'
        "<url><loc>https://espectre.dev/</loc><changefreq>daily</changefreq></url>"
        "<url><loc>https://espectre.dev/artifacts/sdk/api/</loc></url>"
        "<url><loc>https://espectre.dev/artifacts/sdk/stable/</loc></url>"
        "<url><loc>https://espectre.dev/artifacts/sdk/main/</loc></url>"
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
        "https://espectre.dev/artifacts/sdk/api/": "2026-08-08",
        "https://espectre.dev/artifacts/sdk/stable/": "2026-08-10",
        "https://espectre.dev/artifacts/sdk/main/": "2026-08-12",
    }
    assert root.findall("s:url/s:changefreq", namespace) == []


def test_generated_pages_have_sitemap_lastmod_ownership() -> None:
    static_pages = load_script("build_static_pages")
    sitemap_builder = load_script("build_sitemap")
    verifier = load_script("verify_web_build")

    namespace = {"s": sitemap_builder.SITEMAP_NAMESPACE}
    root = ET.parse(REPO_ROOT / "docs" / "web" / "sitemap.xml").getroot()
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
        + ("" if path in {"/artifacts/sdk/main/", "/artifacts/sdk/stable/"} else "<lastmod>2026-08-12</lastmod>")
        + "</url>"
        for path in sorted(verifier.EXPECTED_SITEMAP_PATHS)
    )
    sitemap = tmp_path / "sitemap.xml"
    sitemap.write_text(
        f'<?xml version="1.0"?><urlset xmlns="{namespace}">{entries}</urlset>',
        encoding="utf-8",
    )
    verifier.verify_sitemap(require_main=False, require_stable=False)

    source = sitemap.read_text(encoding="utf-8").replace(
        "<lastmod>2026-08-12</lastmod>",
        "<lastmod>2026-08-12</lastmod><changefreq>daily</changefreq>",
        1,
    )
    sitemap.write_text(source, encoding="utf-8")
    with pytest.raises(ValueError, match="must not contain changefreq"):
        verifier.verify_sitemap(require_main=False, require_stable=False)


def test_pages_verifier_enforces_exact_artifact_contracts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    verifier = load_script("verify_web_build")
    monkeypatch.setattr(verifier, "WEB_ROOT", tmp_path)
    firmware_dir = tmp_path / "artifacts" / "firmware" / "main"
    sdk_dir = tmp_path / "artifacts" / "sdk" / "main"
    firmware_dir.mkdir(parents=True)
    sdk_dir.mkdir(parents=True)

    frontends = {}
    for frontend in sorted(verifier.EXPECTED_FRONTENDS):
        artifacts = []
        for chip in sorted(verifier.EXPECTED_CHIPS):
            filename = f"espectre-{frontend}-{chip}.bin"
            (firmware_dir / filename).write_bytes(b"firmware")
            artifacts.append({"build_type": "factory", "chip": chip, "filename": filename})
        frontends[frontend] = {"artifacts": artifacts}
    firmware_manifest = {"channel": "main", "frontends": frontends}
    firmware_manifest_path = firmware_dir / "firmware-manifest-main.json"
    firmware_manifest_path.write_text(json.dumps(firmware_manifest), encoding="utf-8")
    verifier.verify_firmware_channel("main")

    frontends["native"]["artifacts"].append(frontends["native"]["artifacts"][0])
    firmware_manifest_path.write_text(json.dumps(firmware_manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate firmware"):
        verifier.verify_firmware_channel("main")

    (sdk_dir / "index.html").write_text("SDK", encoding="utf-8")
    sdk_manifest_path = sdk_dir / "sdk-manifest-main.json"
    sdk_manifest = {
        "channel": "main",
        "artifacts": [
            {"format": "tar.gz", "sha256": "a" * 64},
            {"format": "zip", "sha256": "b" * 64},
        ],
    }
    sdk_manifest_path.write_text(json.dumps(sdk_manifest), encoding="utf-8")
    verifier.verify_sdk_channel("main")
    sdk_manifest["artifacts"][0]["sha256"] = "invalid"
    sdk_manifest_path.write_text(json.dumps(sdk_manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="SHA-256"):
        verifier.verify_sdk_channel("main")

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
    assert "git.updateRef" in snapshot and "git.createRef" in snapshot
    assert "github.event.workflow_run.id" in snapshot
    assert "validate-release:" in release
    assert "No successful main CI push run" in release
    assert "git merge-base --is-ancestor" in release
    for source in (ci, snapshot, release):
        assert "uses: ./.github/actions/build-pages" in source
        assert "fetch-depth: 0" in source
    pages_action = (REPO_ROOT / ".github" / "actions" / "build-pages" / "action.yml").read_text(
        encoding="utf-8"
    )
    assert ".github/scripts/build_sitemap.py" in pages_action

    for script_name in (
        "build_matter_firmware.sh",
        "build_native_firmware.sh",
        "build_streamer_firmware.sh",
    ):
        source = (SCRIPTS_DIR / script_name).read_text(encoding="utf-8")
        assert IDF_DOCKER_IMAGE in source
        assert ".espectre-requirements-\\${REQUIREMENTS_HASH}" in source
        assert "--backend local" in source


def test_website_sources_use_the_generated_sdk_api_path() -> None:
    content_root = REPO_ROOT / "docs" / "web" / "content"
    invalid = [
        path.relative_to(REPO_ROOT)
        for path in content_root.rglob("*.html")
        if re.search(r'href="/sdk/api(?:/|\")', path.read_text(encoding="utf-8"))
    ]
    assert invalid == []
