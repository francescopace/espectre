#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-only
# Commercial licensing available under separate agreement; see LICENSING.md.
"""Archive and restore SDK API references identified by version and source commit."""

from __future__ import annotations

import argparse
import io
import json
import re
import subprocess
import sys
import tarfile
import tempfile
import zipfile
from pathlib import Path
from urllib.parse import quote, urlencode

_SCRIPTS_DIR = str(Path(__file__).resolve().parent)
if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)

from web_html_security import validate_passive_api_fragment

API_URL = "https://espectre.dev/sdk/api/"
VERSION_RE = re.compile(r"\d+\.\d+\.\d+(?:-[A-Za-z0-9.-]+)?(?:\+[A-Za-z0-9.-]+)?")
COMMIT_RE = re.compile(r"[0-9a-f]{40}")
FRAGMENT_RE = re.compile(r"fragments/[A-Za-z0-9_.-]+\.html")


def reference_path(version: str, commit: str) -> Path:
    if not VERSION_RE.fullmatch(version) or not COMMIT_RE.fullmatch(commit):
        raise ValueError("SDK API references require a numeric version and a full source commit")
    return Path("revisions") / commit / version


def reference_url(version: str, commit: str) -> str:
    reference_path(version, commit)
    return API_URL + "?" + urlencode({"sdk": version, "commit": commit})


def archive_name(version: str, commit: str) -> str:
    reference_path(version, commit)
    return f"sdk-api-{version}-{commit}.zip"


def validate_reference(files: dict[str, bytes]) -> dict:
    manifest = json.loads(files["api-index.json"])
    reference_path(manifest["sdk_version"], manifest["source_commit"])
    entries = manifest["entries"]
    if manifest.get("schema_version") != 1:
        raise ValueError("Unsupported SDK API reference schema")
    if len({entry["refid"] for entry in entries}) != len(entries):
        raise ValueError("SDK API reference has duplicate page identifiers")
    if not entries or manifest["default"] not in {entry["refid"] for entry in entries}:
        raise ValueError("SDK API reference has no default page")
    fragments = [entry["fragment"] for entry in entries]
    if len(set(fragments)) != len(fragments) or any(not FRAGMENT_RE.fullmatch(name) for name in fragments):
        raise ValueError("SDK API reference has invalid or duplicate fragment paths")
    if set(files) != {"api-index.json", *fragments}:
        raise ValueError("SDK API archive does not match its page inventory")
    for name in fragments:
        validate_passive_api_fragment(files[name].decode("utf-8"))
    return manifest


def read_archive(data: bytes) -> tuple[dict, dict[str, bytes]]:
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)):
            raise ValueError("SDK API archive contains duplicate files")
        files = {name: archive.read(name) for name in names}
    return validate_reference(files), files


def write_archive(api_directory: Path, destination: Path) -> Path:
    manifest = json.loads((api_directory / "api-index.json").read_text())
    names = ["api-index.json", *(entry["fragment"] for entry in manifest["entries"])]
    if any(name != "api-index.json" and not FRAGMENT_RE.fullmatch(name) for name in names):
        raise ValueError("SDK API reference has invalid fragment paths")
    files = {name: (api_directory / name).read_bytes() for name in names}
    validate_reference(files)
    destination.mkdir(parents=True, exist_ok=True)
    path = destination / archive_name(manifest["sdk_version"], manifest["source_commit"])
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, data in sorted(files.items()):
            info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            archive.writestr(info, data)
    return path


def stage_archive(data: bytes, api_directory: Path) -> None:
    manifest, files = read_archive(data)
    destination = api_directory / reference_path(manifest["sdk_version"], manifest["source_commit"])
    if destination.exists():
        existing = {path.relative_to(destination).as_posix(): path.read_bytes()
                    for path in destination.rglob("*") if path.is_file()}
        if existing != files:
            raise ValueError(f"Published SDK API reference changed: {destination}")
        return
    for name, content in files.items():
        path = destination / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)


def github_api(repository: str, resource: str, *, pages: bool = False, binary: bool = False):
    if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", repository):
        raise ValueError("Expected a GitHub owner/repository")
    command = ["gh", "api", f"repos/{repository}/{resource}"]
    if pages:
        command += ["--paginate", "--slurp"]
    if binary:
        command += ["-H", "Accept: application/octet-stream"]
    result = subprocess.run(command, check=True, capture_output=True)
    if binary:
        return result.stdout
    value = json.loads(result.stdout)
    return [item for page in value for item in page] if pages else value


def restore_references(repository: str, api_directory: Path) -> int:
    count = 0
    for release in github_api(repository, "releases?per_page=100", pages=True):
        if release["draft"]:
            continue
        for asset in github_api(repository, f"releases/{release['id']}/assets?per_page=100", pages=True):
            if not asset["name"].startswith("sdk-api-") or not asset["name"].endswith(".zip"):
                continue
            data = github_api(repository, f"releases/assets/{asset['id']}", binary=True)
            manifest, _ = read_archive(data)
            if asset["name"] != archive_name(manifest["sdk_version"], manifest["source_commit"]):
                raise ValueError(f"SDK API asset identity mismatch: {asset['name']}")
            stage_archive(data, api_directory)
            count += 1
    return count


def check_uploads(repository: str, tag: str, directory: Path) -> None:
    """Leave existing, identical API assets out of the release action's upload set."""
    uploads = {}
    for path in directory.glob("sdk-api-*.zip"):
        manifest, files = read_archive(path.read_bytes())
        if path.name != archive_name(manifest["sdk_version"], manifest["source_commit"]):
            raise ValueError(f"SDK API upload identity mismatch: {path}")
        uploads[path] = files
    try:
        release = github_api(repository, f"releases/tags/{quote(tag, safe='')}")
    except subprocess.CalledProcessError as error:
        if b"HTTP 404" in error.stderr:
            return
        raise
    assets = {asset["name"]: asset for asset in
              github_api(repository, f"releases/{release['id']}/assets?per_page=100", pages=True)}
    for path, files in uploads.items():
        if path.name not in assets:
            continue
        existing = github_api(repository, f"releases/assets/{assets[path.name]['id']}", binary=True)
        if read_archive(existing)[1] != files:
            raise ValueError(f"Refusing to replace published SDK API asset: {path.name}")
        path.unlink()


def restore_pages_archive(repository: str, archive_path: Path) -> int:
    """Merge history in the serialized deployment job, after all publication builds."""
    with tempfile.TemporaryDirectory(prefix="sdk-api-pages-") as temporary:
        root = Path(temporary) / "pages"
        with tarfile.open(archive_path) as archive:
            archive.extractall(root, filter="data")
        api_directory = root / "artifacts/sdk/api"
        if not (api_directory / "api-index.json").is_file():
            raise ValueError("Pages archive has no SDK API reference")
        count = restore_references(repository, api_directory)
        with tempfile.TemporaryDirectory(dir=archive_path.parent, prefix="sdk-api-merge-") as staged:
            staged_archive = Path(staged) / "artifact.tar"
            with tarfile.open(staged_archive, "w") as archive:
                for path in sorted(root.iterdir()):
                    archive.add(path, arcname=path.name)
            staged_archive.replace(archive_path)
    return count


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    restore = subparsers.add_parser("restore")
    restore.add_argument("--repository", required=True)
    restore.add_argument("--output-dir", type=Path, default=Path("docs/web/artifacts/sdk/api"))
    check = subparsers.add_parser("check-uploads")
    check.add_argument("--repository", required=True)
    check.add_argument("--tag", required=True)
    check.add_argument("--directory", type=Path, required=True)
    pages = subparsers.add_parser("restore-pages")
    pages.add_argument("--repository", required=True)
    pages.add_argument("--archive", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "restore":
        print(f"Restored {restore_references(args.repository, args.output_dir)} SDK API archives.")
    elif args.command == "check-uploads":
        check_uploads(args.repository, args.tag, args.directory)
    else:
        print(f"Restored {restore_pages_archive(args.repository, args.archive)} SDK API archives into Pages.")


if __name__ == "__main__":
    main()
