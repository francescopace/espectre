#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-only
# Commercial licensing available under separate agreement; see LICENSING.md.
"""Keep recent SDK snapshots and delete old ones from staging only."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import re

from idf_component_tools.registry.api_client import APIClient
from idf_component_tools.semver import Version


REGISTRY_URL = "https://components-staging.espressif.com"
COMPONENT_NAME = "francescopace/espectre"


def snapshot_branch(version: str) -> str | None:
    """Recognize current and legacy snapshot versions, excluding release tags."""
    if not Version(version).prerelease:
        return None
    legacy = re.fullmatch(r".+-snapshot\.(preview|develop)\.g[0-9a-f]{40}", version)
    if legacy:
        return {"preview": "main", "develop": "develop"}[legacy[1]]
    current = re.fullmatch(r".+\.(main|develop)", version)
    return current[1] if current else None


def cleanup_plan(versions: list, keep: int) -> list[dict]:
    """Order by upload time, retaining each branch independently."""
    if keep < 1:
        raise ValueError("Keep count must be positive")
    branches: dict[str, list] = {"main": [], "develop": []}
    plan = []
    seen = set()
    for version in versions:
        if version.version in seen:
            raise ValueError(f"Duplicate registry version: {version.version}")
        seen.add(version.version)
        branch = snapshot_branch(version.version)
        entry = {"version": version.version, "created_at": version.created_at,
                 "branch": branch, "action": "keep", "reason": "not a snapshot"}
        if branch is None:
            plan.append(entry)
            continue
        # Missing or ambiguous upload times must fail before any deletion.
        if not version.created_at:
            raise ValueError(f"Missing upload time: {version.version}")
        created = datetime.fromisoformat(version.created_at.replace("Z", "+00:00"))
        if created.tzinfo is None:
            raise ValueError(f"Upload time has no timezone: {version.version}")
        branches[branch].append((created, entry))
    for entries in branches.values():
        entries.sort(key=lambda item: (item[0], item[1]["version"]), reverse=True)
        for index, (_, entry) in enumerate(entries):
            if index < keep:
                entry["reason"] = "within branch retention"
            else:
                entry.update(action="delete", reason="outside branch retention")
            plan.append(entry)
    return plan


def write_report(report: dict, destination: Path | None) -> None:
    if destination:
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary:
        candidates = [entry["version"] for entry in report["versions"] if entry["action"] == "delete"]
        with Path(summary).open("a", encoding="utf-8") as output:
            output.write("## SDK staging cleanup\n\n")
            output.write(f"Registry: {REGISTRY_URL}\n\nComponent: `{COMPONENT_NAME}`\n\n")
            output.write(f"Mode: {'apply' if report['apply'] else 'dry run'}. "
                         f"Candidates: {len(candidates)}. Deleted: {len(report['deleted'])}.\n\n")
            for version in candidates:
                status = "deleted" if version in report["deleted"] else "not deleted"
                output.write(f"- `{version}`: {status}\n")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--keep-per-branch", type=int, default=10)
    parser.add_argument("--apply", action="store_true", help="Delete candidates; default is a dry run")
    parser.add_argument("--report", type=Path, help="Write the plan and completed deletions as JSON")
    args = parser.parse_args(argv)
    if args.keep_per_branch < 1:
        parser.error("Keep count must be positive")

    # Explicit URLs prevent local profiles or environment variables selecting production.
    os.environ["IDF_COMPONENT_CACHE_HTTP_REQUESTS"] = "0"
    reader = APIClient(registry_url=REGISTRY_URL)
    component = reader.get_component_response(component_name=COMPONENT_NAME)
    if f"{component.namespace}/{component.name}" != COMPONENT_NAME:
        raise ValueError("Registry returned an unexpected component")
    report = {
        "registry": REGISTRY_URL,
        "component": COMPONENT_NAME,
        "apply": args.apply,
        "keep_per_branch": args.keep_per_branch,
        "versions": cleanup_plan(component.versions, args.keep_per_branch),
        "deleted": [],
    }
    candidates = [entry["version"] for entry in report["versions"] if entry["action"] == "delete"]
    for entry in report["versions"]:
        print(f"{entry['action'].upper()}: {entry['version']} ({entry['reason']})", flush=True)
    try:
        if args.apply and candidates:
            token = os.environ.get("IDF_COMPONENT_API_TOKEN")
            if not token:
                raise ValueError("Set SDK_REGISTRY_STAGING_CLEANUP_TOKEN in the sdk-registry-staging environment")
            writer = APIClient(registry_url=REGISTRY_URL, api_token=token)
            for version in candidates:
                writer.delete_version(component_name=COMPONENT_NAME, component_version=version)
                report["deleted"].append(version)
                print(f"DELETED: {version}", flush=True)
    finally:
        write_report(report, args.report)


if __name__ == "__main__":
    main()
