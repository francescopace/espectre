#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-only
# Commercial licensing available under separate agreement; see LICENSING.md.
"""Generate SPA-ready SDK API fragments from Doxygen XML through m.css."""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator
from urllib.parse import quote

_SCRIPTS_DIR = str(Path(__file__).resolve().parent)
if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)

from build_sdk_package import stamp_doxyfile_project_number
from detect_git_version import detect_git_version, parse_version_core

REPO_ROOT = Path(__file__).resolve().parents[2]
DOXYFILE = REPO_ROOT / "src" / "cpp" / "Doxyfile"
MCSS_TEMPLATES = REPO_ROOT / ".github" / "mcss" / "templates"
API_OUTPUT_DIR = REPO_ROOT / "docs" / "web" / "artifacts" / "sdk" / "api"
MCSS_REPOSITORY = "https://github.com/mosra/m.css.git"
MCSS_COMMIT = "0a460a7a9973a41db48f735e7b49e4da9a876325"
HTML_LINK_RE = re.compile(r'href="(?P<href>[^"]+)"')
MCSS_CONTENTS_NAV_RE = re.compile(
    r'\s*<nav class="m-block m-default">.*?</nav>',
    re.DOTALL,
)
OUTPUT_DIRECTORY_RE = re.compile(r"(?m)^OUTPUT_DIRECTORY\s*=\s*.*$")
INDEX_PAGES = {
    "index": ("Overview", "page"),
    "annotated": ("Classes", "index"),
    "namespaces": ("Namespaces", "index"),
    "files": ("Files", "index"),
    "pages": ("Pages", "index"),
    "modules": ("Modules", "index"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate the ESPectre SDK API fragments with the current SDK identity."
    )
    parser.add_argument(
        "--version",
        help="Override git describe. Must start with numeric MAJOR.MINOR.PATCH.",
    )
    parser.add_argument(
        "--mcss-root",
        type=Path,
        help="Use an existing m.css checkout instead of fetching the pinned revision.",
    )
    return parser.parse_args()


def stamp_doxyfile_output_directory(path: Path, output_directory: Path) -> None:
    source, count = OUTPUT_DIRECTORY_RE.subn(
        f"OUTPUT_DIRECTORY       = {output_directory}",
        path.read_text(encoding="utf-8"),
        count=1,
    )
    if count != 1:
        raise ValueError(f"Unable to stamp OUTPUT_DIRECTORY in {path}")
    path.write_text(source, encoding="utf-8")


def run_doxygen(doxyfile: Path) -> None:
    try:
        subprocess.run(["doxygen", str(doxyfile)], cwd=REPO_ROOT, check=True)
    except FileNotFoundError as error:
        raise FileNotFoundError("doxygen is not installed or not on PATH") from error


def prune_private_members(xml_directory: Path) -> None:
    """Remove implementation-only members Doxygen leaves in the public XML."""
    for xml_path in xml_directory.glob("*.xml"):
        tree = ET.parse(xml_path)
        root = tree.getroot()
        changed = False
        for compound in root.findall(".//compounddef"):
            for section in list(compound.findall("sectiondef")):
                for member in list(section.findall("memberdef")):
                    if member.get("prot") == "private":
                        section.remove(member)
                        changed = True
                if not section.findall("memberdef") and section.get("kind", "").startswith("private-"):
                    compound.remove(section)
                    changed = True
        if changed:
            tree.write(xml_path, encoding="utf-8", xml_declaration=True)


def validate_mcss_root(root: Path) -> Path:
    script = root / "documentation" / "doxygen.py"
    if not script.is_file():
        raise FileNotFoundError(f"m.css Doxygen renderer is missing: {script}")
    return script


@contextmanager
def mcss_checkout(explicit_root: Path | None) -> Iterator[Path]:
    if explicit_root is not None:
        root = explicit_root.resolve()
        validate_mcss_root(root)
        yield root
        return

    with tempfile.TemporaryDirectory(prefix="espectre-mcss-") as tmp_dir:
        root = Path(tmp_dir) / "m.css"
        try:
            subprocess.run(["git", "init", "--quiet", str(root)], cwd=REPO_ROOT, check=True)
            subprocess.run(
                ["git", "-C", str(root), "fetch", "--quiet", "--depth", "1", MCSS_REPOSITORY, MCSS_COMMIT],
                cwd=REPO_ROOT,
                check=True,
            )
            subprocess.run(
                ["git", "-C", str(root), "checkout", "--quiet", "FETCH_HEAD"],
                cwd=REPO_ROOT,
                check=True,
            )
        except FileNotFoundError as error:
            raise FileNotFoundError("git is required to fetch the pinned m.css renderer") from error
        validate_mcss_root(root)
        yield root


def run_mcss(doxyfile: Path, mcss_root: Path | None) -> None:
    config = doxyfile.with_name("mcss_config.py")
    config.write_text(
        f'DOXYFILE = r"{doxyfile}"\n'
        "SEARCH_DISABLED = True\n"
        "SHOW_UNDOCUMENTED = True\n",
        encoding="utf-8",
    )
    with mcss_checkout(mcss_root) as root:
        subprocess.run(
            [
                sys.executable,
                str(validate_mcss_root(root)),
                str(config),
                "--no-doxygen",
                "--templates",
                str(MCSS_TEMPLATES),
                "--sort-globbed-files",
            ],
            cwd=REPO_ROOT,
            check=True,
        )


def compound_metadata(index_path: Path) -> dict[str, dict[str, str]]:
    root = ET.parse(index_path).getroot()
    metadata: dict[str, dict[str, str]] = {}
    for compound in root.findall("compound"):
        refid = compound.get("refid", "")
        name = (compound.findtext("name") or refid).strip()
        if refid:
            metadata[refid] = {
                "refid": refid,
                "name": name,
                "kind": compound.get("kind", "reference"),
            }
    return metadata


def api_url(refid: str, anchor: str = "") -> str:
    url = f"/sdk/api/?api={quote(refid)}"
    if anchor:
        url += f"&member={quote(anchor)}"
    return url


def rewrite_fragment_links(source: str, current_refid: str, known_refids: set[str]) -> str:
    def replace(match: re.Match[str]) -> str:
        href = match.group("href")
        if href.startswith(("http://", "https://", "mailto:", "tel:")):
            return match.group(0)
        if href.startswith("#"):
            anchor = href[1:]
            if not anchor:
                return match.group(0)
            return (
                f'href="{api_url(current_refid, anchor)}" '
                f'data-api-reference-ref="{current_refid}" data-api-reference-member="{anchor}"'
            )
        page, separator, anchor = href.partition("#")
        if not page.endswith(".html"):
            return match.group(0)
        target_refid = Path(page).stem
        if target_refid not in known_refids:
            return match.group(0)
        return (
            f'href="{api_url(target_refid, anchor if separator else "")}" '
            f'data-api-reference-ref="{target_refid}"'
            + (f' data-api-reference-member="{anchor}"' if separator else "")
        )

    return HTML_LINK_RE.sub(replace, source)


def strip_contents_navigation(fragment: str) -> str:
    return MCSS_CONTENTS_NAV_RE.sub("", fragment)


def picker_discoverable(refid: str, kind: str, fragment: str) -> bool:
    if refid == "index" or kind in {"class", "struct", "union", "namespace", "page"}:
        return True
    return kind == "file" and 'class="m-doc-details"' in fragment


def publish_fragments(rendered_directory: Path, xml_directory: Path, sdk_version: str) -> None:
    sources = sorted(rendered_directory.glob("*.html"))
    if not sources:
        raise ValueError("m.css generated no API fragments")
    known_refids = {path.stem for path in sources}
    metadata = compound_metadata(xml_directory / "index.xml")

    if API_OUTPUT_DIR.exists():
        shutil.rmtree(API_OUTPUT_DIR)
    fragments = API_OUTPUT_DIR / "fragments"
    fragments.mkdir(parents=True)

    entries: list[dict[str, object]] = []
    for source in sources:
        refid = source.stem
        fragment = strip_contents_navigation(
            rewrite_fragment_links(source.read_text(encoding="utf-8"), refid, known_refids)
        )
        if "<html" in fragment.lower() or "<body" in fragment.lower():
            raise ValueError(f"m.css template emitted a standalone document: {source.name}")
        unresolved_links = [
            match.group("href")
            for match in HTML_LINK_RE.finditer(fragment)
            if match.group("href").partition("#")[0].endswith(".html")
        ]
        if unresolved_links:
            raise ValueError(f"m.css fragment contains unresolved page links: {source.name}: {unresolved_links[:3]}")
        (fragments / source.name).write_text(fragment, encoding="utf-8")

        item = metadata.get(refid)
        if item is None:
            label, kind = INDEX_PAGES.get(refid, (refid, "index"))
            item = {"refid": refid, "name": label, "kind": kind}
        entries.append(
            {
                **item,
                "fragment": f"fragments/{source.name}",
                "discoverable": picker_discoverable(refid, item["kind"], fragment),
            }
        )

    order = {refid: index for index, refid in enumerate(INDEX_PAGES)}
    entries.sort(
        key=lambda item: (
            order.get(item["refid"], len(order)),
            item["kind"] not in {"class", "struct", "union"},
            item["name"].casefold(),
        )
    )
    default_refid = "index" if "index" in known_refids else entries[0]["refid"]
    manifest = {
        "schema_version": 1,
        "sdk_version": sdk_version,
        "renderer": "m.css",
        "renderer_revision": MCSS_COMMIT,
        "default": default_refid,
        "entries": entries,
    }
    (API_OUTPUT_DIR / "api-index.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def generate_sdk_api(version: str | None = None, mcss_root: Path | None = None) -> str:
    sdk_version = version or detect_git_version()
    parse_version_core(sdk_version)

    with tempfile.TemporaryDirectory(prefix="espectre-doxy-") as tmp_dir:
        work_directory = Path(tmp_dir) / "api"
        work_directory.mkdir(parents=True)
        stamped = Path(tmp_dir) / "Doxyfile"
        shutil.copy2(DOXYFILE, stamped)
        stamp_doxyfile_project_number(stamped, sdk_version)
        stamp_doxyfile_output_directory(stamped, work_directory)
        run_doxygen(stamped)

        xml_directory = work_directory / "xml"
        prune_private_members(xml_directory)
        run_mcss(stamped, mcss_root)
        publish_fragments(work_directory / "rendered", xml_directory, sdk_version)

    return sdk_version


def main() -> int:
    args = parse_args()
    sdk_version = generate_sdk_api(args.version, args.mcss_root)
    print(f"Generated SDK API fragments for {sdk_version}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
