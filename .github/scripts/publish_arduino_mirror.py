#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-only
# Commercial licensing available under separate agreement; see LICENSING.md.
"""
ESPectre - Arduino Library Mirror Publisher

Publish the Arduino library archive to the repository that the Arduino Library
Manager indexes. The Library Manager reads `library.properties` from the
repository root at each tag, whatever the tag is called, so the mirror holds
the archive contents at its root. Stable releases get one commit on `main` and
one tag each. Development snapshots get a commit on `develop` and no tag, so
they never reach the Library Manager.

Author: Francesco Pace <francesco.pace@gmail.com>
"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import tempfile
import zipfile
from pathlib import Path, PurePosixPath

LIBRARY_NAME = "ESPectre"
RELEASE_BRANCH = "main"
SNAPSHOT_BRANCH = "develop"
STABLE_VERSION = re.compile(r"\d+\.\d+\.\d+")
BOT_NAME = "github-actions[bot]"
BOT_EMAIL = "41898282+github-actions[bot]@users.noreply.github.com"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Publish the Arduino library to its mirror repository.")
    parser.add_argument("--archive", required=True, help="Library ZIP built by build_arduino_package.py.")
    parser.add_argument("--version", required=True, help="Library version, for example 3.2.0.")
    parser.add_argument("--snapshot", action="store_true", help="Publish a development snapshot without a tag.")
    parser.add_argument("--remote", required=True, help="Git URL of the mirror repository.")
    parser.add_argument("--source-repository", required=True, help="Source repository, as owner/name.")
    parser.add_argument("--source-sha", required=True, help="Source commit the archive was built from.")
    return parser.parse_args()


def git(worktree: Path, *args: str) -> str:
    settings = [f"user.name={BOT_NAME}", f"user.email={BOT_EMAIL}", "commit.gpgSign=false", "tag.gpgSign=false"]
    command = ["git", "-C", str(worktree), *(arg for item in settings for arg in ("-c", item)), *args]
    return subprocess.run(command, check=True, capture_output=True, text=True).stdout.strip()


def has_remote_branch(worktree: Path, branch: str) -> bool:
    """Whether the mirror has the branch yet; a new, empty mirror has none."""
    probe = ["git", "-C", str(worktree), "rev-parse", "--verify", "--quiet", f"refs/remotes/origin/{branch}"]
    return subprocess.run(probe, capture_output=True).returncode == 0


def extract_library(archive: Path, destination: Path) -> Path:
    """Extract the archive, which must hold exactly one library folder."""
    with zipfile.ZipFile(archive) as bundle:
        for name in bundle.namelist():
            parts = PurePosixPath(name).parts
            if PurePosixPath(name).is_absolute() or ".." in parts or parts[0] != LIBRARY_NAME:
                raise ValueError(f"Unexpected path in the library archive: {name}")
        bundle.extractall(destination)
    return destination / LIBRARY_NAME


def library_version(library: Path) -> str:
    match = re.search(r"(?m)^version=(.*)$", (library / "library.properties").read_text(encoding="utf-8"))
    if match is None:
        raise ValueError("The library archive has no version in library.properties")
    return match.group(1).strip()


def mirror_readme(source: str, source_repository: str) -> str:
    """Point readers of the mirror back to the source repository."""
    heading, separator, body = source.partition("\n")
    if not heading.startswith("# ") or not separator:
        raise ValueError("The library README must start with a level-one heading")
    url = f"https://github.com/{source_repository}"
    notice = (
        f"This repository is generated from [{source_repository}]({url}). "
        f"Report issues and send pull requests to [{source_repository}]({url})."
    )
    return f"{heading}\n\n> {notice}\n{body}"


def sync_worktree(library: Path, worktree: Path, source_repository: str) -> None:
    for entry in worktree.iterdir():
        if entry.name == ".git":
            continue
        if entry.is_dir() and not entry.is_symlink():
            shutil.rmtree(entry)
        else:
            entry.unlink()
    shutil.copytree(library, worktree, dirs_exist_ok=True)
    readme = worktree / "README.md"
    readme.write_text(mirror_readme(readme.read_text(encoding="utf-8"), source_repository), encoding="utf-8")


def publish(args: argparse.Namespace) -> str:
    """Publish the library and return the mirror commit; unchanged contents publish nothing."""
    if not args.snapshot and not STABLE_VERSION.fullmatch(args.version):
        raise ValueError(f"Only stable versions get a release tag, not {args.version}")
    branch = SNAPSHOT_BRANCH if args.snapshot else RELEASE_BRANCH
    with tempfile.TemporaryDirectory() as staging:
        library = extract_library(Path(args.archive), Path(staging) / "archive")
        if (version := library_version(library)) != args.version:
            raise ValueError(f"The archive holds version {version}, not {args.version}")

        worktree = Path(staging) / "mirror"
        subprocess.run(["git", "clone", "--quiet", args.remote, str(worktree)], check=True)
        existing = has_remote_branch(worktree, branch)
        if existing:
            git(worktree, "checkout", "--quiet", "-B", branch, f"origin/{branch}")
        else:
            git(worktree, "symbolic-ref", "HEAD", f"refs/heads/{branch}")
        sync_worktree(library, worktree, args.source_repository)
        git(worktree, "add", "--all")
        tree = git(worktree, "write-tree")

        tag = f"refs/tags/{args.version}"
        if args.snapshot:
            if existing and git(worktree, "rev-parse", "HEAD^{tree}") == tree:
                return git(worktree, "rev-parse", "HEAD")
        elif git(worktree, "tag", "--list", args.version):
            if git(worktree, "rev-parse", f"{tag}^{{tree}}") != tree:
                raise ValueError(f"The mirror already has {args.version} with different contents")
            return git(worktree, "rev-parse", f"{tag}^{{commit}}")

        kind = "snapshot" if args.snapshot else "release"
        message = f"{kind} {args.version}\n\nGenerated from {args.source_repository}@{args.source_sha}."
        git(worktree, "commit", "--quiet", "--allow-empty", "--no-verify", "-m", message)
        refs = [f"HEAD:refs/heads/{branch}"]
        if not args.snapshot:
            git(worktree, "tag", "--annotate", args.version, "-m", f"ESPectre {args.version}")
            refs.append(tag)
        git(worktree, "push", "--quiet", "--atomic", "origin", *refs)
        return git(worktree, "rev-parse", "HEAD")


def main() -> int:
    print(publish(parse_args()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
