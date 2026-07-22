#!/usr/bin/env python3
"""
ESPectre - Batch synthetic low-RSSI dataset generation

Generate deterministic low-RSSI derivatives for every compatible registered
real dataset. Quiet datasets are processed before motion so shared-session
generation can reuse the paired quiet calibration.

Author: Francesco Pace <francesco.pace@gmail.com>
License: GPLv3
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Any, Dict, Iterable, Optional


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools import generate_low_rssi_dataset as generator  # noqa: E402
from tools.lib import dataset_metadata  # noqa: E402


PAIRED_LABELS = ("static_presence", "motion")
LABEL_ORDER = PAIRED_LABELS
PROFILE_BY_CHIP = {
    profile.reference_chip.upper(): profile_name
    for profile_name, profile in generator.LOW_RSSI_PROFILES.items()
}


@dataclass(frozen=True)
class GenerationJob:
    """One real source and its matching synthetic link profile."""

    label: str
    chip: str
    source_path: Path
    profile_name: str


def _real_low_rssi_pair_groups(info: Dict[str, Any]) -> set[tuple[str, str]]:
    """Return chip/environment groups covered by a real low-RSSI pair."""
    files = info.get("files", {})
    motion_by_filename = {
        str(entry.get("filename", "")): entry
        for entry in files.get("motion", [])
        if entry.get("filename")
    }
    groups = set()
    for static_entry in files.get("static_presence", []):
        if static_entry.get("synthetic") or not static_entry.get("low_rssi"):
            continue
        motion_entry = motion_by_filename.get(
            str(static_entry.get("optimal_pair_motion_file", ""))
        )
        if (
            motion_entry is None
            or motion_entry.get("synthetic")
            or not motion_entry.get("low_rssi")
        ):
            continue
        chip = str(static_entry.get("chip", "")).upper()
        if chip:
            environment = str(static_entry.get("environment", ""))
            groups.add((chip, environment))
    return groups


def collect_jobs(
    info: Dict[str, Any],
    *,
    chips: Iterable[str],
    labels: Iterable[str],
    environment: Optional[str] = None,
) -> list[GenerationJob]:
    """Collect compatible real sources in quiet-before-motion order."""
    selected_chips = {str(chip).upper() for chip in chips}
    selected_labels = set(labels)
    real_low_rssi_pair_groups = _real_low_rssi_pair_groups(info)
    jobs = []
    for label in LABEL_ORDER:
        if label not in selected_labels:
            continue
        entries = sorted(
            info.get("files", {}).get(label, []),
            key=lambda entry: (
                str(entry.get("collected_at", "")),
                str(entry.get("filename", "")),
            ),
        )
        for entry in entries:
            if (
                entry.get("synthetic")
                or entry.get("auto_generated")
                or entry.get("low_rssi")
            ):
                continue
            chip = str(entry.get("chip", "")).upper()
            if chip not in selected_chips or chip not in PROFILE_BY_CHIP:
                continue
            entry_environment = str(entry.get("environment", ""))
            if (
                label in PAIRED_LABELS
                and (chip, entry_environment) in real_low_rssi_pair_groups
            ):
                continue
            if (
                environment is not None
                and entry_environment != environment
            ):
                continue
            jobs.append(
                GenerationJob(
                    label=label,
                    chip=chip,
                    source_path=dataset_metadata.resolve_entry_path(label, entry),
                    profile_name=PROFILE_BY_CHIP[chip],
                )
            )
    return jobs


def _output_is_registered(path: Path) -> bool:
    target = path.resolve()
    info = dataset_metadata.load_dataset_info()
    for label, entries in info.get("files", {}).items():
        for entry in entries:
            if not entry.get("synthetic"):
                continue
            if dataset_metadata.resolve_entry_path(label, entry).resolve() == target:
                return True
    return False


def run_jobs(
    jobs: Iterable[GenerationJob],
    *,
    mode: str,
    seed: int,
    dry_run: bool,
    force: bool,
) -> tuple[int, int, int]:
    """Run a generation plan and return generated, skipped, and failed counts."""
    planned = list(jobs)
    generated = 0
    skipped = 0
    failed = 0
    for index, job in enumerate(planned, start=1):
        output_path = generator.build_output_path(
            job.source_path,
            job.label,
            job.profile_name,
            mode,
            seed,
        )
        print(
            f"\n[{index}/{len(planned)}] {job.label} {job.chip}: "
            f"{job.source_path.name}"
        )
        print(f"  profile: {job.profile_name}")
        print(f"  output:  {output_path}")
        if dry_run:
            continue
        if output_path.exists() and not force:
            if _output_is_registered(output_path):
                print("  skipped: output already exists and is registered")
                skipped += 1
            else:
                print("  failed: output exists but is not registered; use --force")
                failed += 1
            continue
        try:
            generator.generate_dataset(
                job.source_path,
                profile_name=job.profile_name,
                seed=seed,
                generation_mode=mode,
                output_path=output_path,
                register=True,
                force=force,
            )
            generated += 1
        except Exception as exc:
            print(f"  failed: {type(exc).__name__}: {exc}", file=sys.stderr)
            failed += 1
    return generated, skipped, failed


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate synthetic low-RSSI derivatives for every compatible "
            "registered real dataset."
        )
    )
    parser.add_argument(
        "--mode",
        choices=("shared_session", "reference_match"),
        default="shared_session",
        help="Generation mode (default: shared_session for ML augmentation)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=20260722,
        help="Reproducible generation seed",
    )
    parser.add_argument(
        "--chips",
        nargs="+",
        choices=tuple(sorted(PROFILE_BY_CHIP)),
        default=tuple(sorted(PROFILE_BY_CHIP)),
        help="Source chips to process (default: every chip with a profile)",
    )
    parser.add_argument(
        "--labels",
        nargs="+",
        choices=LABEL_ORDER,
        default=LABEL_ORDER,
        help="Semantic labels to process (default: static_presence and motion)",
    )
    parser.add_argument(
        "--environment",
        help="Only process datasets with this exact environment metadata",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the generation plan without writing files or metadata",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Regenerate and replace deterministic synthetic outputs",
    )
    return parser


def main(argv: Optional[Iterable[str]] = None) -> int:
    args = build_argument_parser().parse_args(argv)
    info = dataset_metadata.load_dataset_info()
    jobs = collect_jobs(
        info,
        chips=args.chips,
        labels=args.labels,
        environment=args.environment,
    )
    if not jobs:
        print("No compatible real datasets matched the requested filters.")
        return 0

    print(
        f"Planned {len(jobs)} synthetic datasets in {args.mode} mode "
        f"with seed {args.seed}."
    )
    generated, skipped, failed = run_jobs(
        jobs,
        mode=args.mode,
        seed=args.seed,
        dry_run=args.dry_run,
        force=args.force,
    )
    if args.dry_run:
        print(f"\nDry run complete: {len(jobs)} datasets planned.")
        return 0

    print(
        f"\nBatch complete: {generated} generated, {skipped} skipped, "
        f"and {failed} failed."
    )
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
