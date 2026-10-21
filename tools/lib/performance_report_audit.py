# SPDX-License-Identifier: GPL-3.0-only
# Commercial licensing available under separate agreement; see LICENSING.md.
"""Operational replay measurements, kept separate from conditional parity metrics."""

from __future__ import annotations

import hashlib
import json
import re
from collections import defaultdict

import numpy as np

from . import performance_report as replay
from .high_accuracy_detector import predict
from .repo_paths import repo_root
from .temporal_replay import packet_timestamp_us


def packet_times(packets, interval_us):
    """Unwrap the recorded device clock without replacing receive-time gaps."""
    raw = np.asarray([
        packet_timestamp_us(p, i, interval_us) for i, p in enumerate(packets)
    ], dtype=np.int64)
    if not len(raw):
        return raw
    delta = np.diff(raw)
    # Device timestamps are uint32; a reversed timestamp is not a 71-minute gap.
    delta = np.where(delta < -(1 << 31), delta + (1 << 32), delta)
    if np.any(delta < 0):
        raise ValueError("operational replay requires monotonic device timestamps")
    return np.r_[0, np.cumsum(delta)]


# A ready evaluation covers the grid until it is older than this many expected
# evaluation intervals. On the report corpus, 99.9% of consecutive evaluations
# are at most 1.17 intervals apart, so cadence jitter stays available while a
# skipped evaluation still becomes unavailable.
AVAILABILITY_TOLERANCE_INTERVALS = 1.5


def score_timeline(times, indices, states, ready, *, cutoff_us=0):
    """Score emitted evaluations on a fixed recorded-time grid.

    The production hit policy advances once per emitted evaluation; an unready
    evaluation holds its state. The grid only weights time: a tick is available
    when the latest evaluation at or before its end is ready and at most
    AVAILABILITY_TOLERANCE_INTERVALS expected intervals old. Unavailable raw
    ticks are IDLE. A prediction is never used before its timestamp, and no CSI
    or feature value is imputed.
    """
    step = int(replay.EVALUATION_INTERVAL_MS * 1000)
    tolerance = int(round(step * AVAILABILITY_TOLERANCE_INTERVALS))
    first = int(cutoff_us) // step + 1
    last = int(times[-1]) // step if len(times) else 0
    n = max(0, last - first + 1)
    span_end = last * step
    policy = replay.RuntimeMotionPolicy(
        evaluation_interval_ms=replay.EVALUATION_INTERVAL_MS,
        motion_on_hits=replay.MOTION_ON_HITS,
        motion_off_hits=replay.MOTION_OFF_HITS,
    )
    event_times, event_ready, event_raw, event_effective = [], [], [], []
    alarms = 0
    first_detection = None
    off_at = None
    interruptions = []
    for index, state, valid in zip(indices, states, ready, strict=True):
        elapsed = int(times[int(index)])
        if elapsed <= cutoff_us or elapsed > span_end:
            continue
        raw = bool(state) and bool(valid)
        effective, changed = policy.apply_state(
            replay.MotionState.MOTION if raw else replay.MotionState.IDLE,
            ready=bool(valid),
        )
        motion = effective == replay.MotionState.MOTION
        if changed and motion:
            alarms += 1
            if first_detection is None:
                first_detection = elapsed
            if off_at is not None:
                interruptions.append((elapsed - off_at) / 1e6)
                off_at = None
        elif changed and first_detection is not None:
            off_at = elapsed
        event_times.append(elapsed)
        event_ready.append(bool(valid))
        event_raw.append(raw)
        event_effective.append(motion)
    tick_ends = (first + np.arange(n, dtype=np.int64)) * step
    latest = np.searchsorted(np.asarray(event_times, dtype=np.int64), tick_ends, side="right") - 1
    seen = latest >= 0
    position = np.where(seen, latest, 0)
    if event_times:
        fresh = tick_ends - np.asarray(event_times, dtype=np.int64)[position] <= tolerance
        available = seen & fresh & np.asarray(event_ready, dtype=bool)[position]
        raw = available & np.asarray(event_raw, dtype=bool)[position]
        effective = seen & np.asarray(event_effective, dtype=bool)[position]
    else:
        available = raw = effective = np.zeros(n, dtype=bool)
    return {
        "ticks": n,
        "available": int(available.sum()),
        "raw_positive": int(raw.sum()),
        "effective_positive": int(effective.sum()),
        "alarms": alarms,
        "seconds": n * step / 1e6,
        # Only IDLE runs that end in renewed MOTION are interruptions.
        "interruptions": len(interruptions),
        "max_interruption_s": max(interruptions, default=0.0),
        "first_detection_s": None if first_detection is None else first_detection / 1e6,
    }


def score_pair(static_path, motion_path):
    """Replay a pair without scoring the Lightweight calibration prefix."""
    static, motion = replay.load_real_data_cached(static_path, motion_path)
    rows = replay.load_or_compute_classic_replay_rows(
        static_path, motion_path,
        selected_subcarriers=replay.DEFAULT_SUBCARRIERS,
        replay_kind="classic_dataset",
    )
    timing = rows["timing"]
    times = {
        "static": packet_times(static, timing["interval_us"]),
        "motion": packet_times(motion, timing["interval_us"]),
    }
    calibrated = replay._calibrate_classic_replay_rows_with_boundary(rows["calibration"], timing)
    result = {}
    if calibrated is not None:
        threshold, row = calibrated
        boundary = int(rows["calibration"]["packet_index"][row])
        detector = replay.build_lightweight_detector(threshold=threshold, timing=timing)
        detector._adapted_threshold_ready = True
        phases = {}
        for phase in ("static", "motion"):
            phase_rows = rows[phase]
            cutoff = int(times[phase][boundary]) if phase == "static" else 0
            # The adaptation state must not observe the calibration prefix twice.
            if phase == "static":
                mask = np.asarray(phase_rows["packet_index"]) > boundary
                phase_rows = {k: np.asarray(v)[mask] for k, v in phase_rows.items()}
            states = replay._score_classic_replay_phase_rows(
                phase_rows, detector, include_unready=True,
            )
            phases[phase] = score_timeline(
                times[phase], phase_rows["packet_index"], states, phase_rows["ready"],
                cutoff_us=cutoff,
            )
        phases["calibration_seconds"] = int(times["static"][boundary]) / 1e6
        result["classic"] = phases
    else:
        result["classic"] = {
            phase: score_timeline(times[phase], [], [], []) for phase in ("static", "motion")
        }
        result["classic"]["calibration_failed"] = True
    ml_phases = {}
    for phase, path in (("static", static_path), ("motion", motion_path)):
        ml = replay.load_or_compute_ml_replay_rows(
            path, selected_subcarriers=replay.DEFAULT_SUBCARRIERS,
            window_size=timing["window_packets"], sample_contract="replay_tick",
        )
        states = [predict(x) > 0.5 for x in ml["X"]]
        ml_phases[phase] = score_timeline(
            times[phase], ml["packet_index"], states, np.ones(len(states), dtype=bool),
        )
    result["ml"] = ml_phases
    return result


def score_quiet(path):
    """Score an independent quiet capture, with disjoint Lightweight calibration."""
    packets = replay._load_report_packet_view(path)
    rows = replay.load_or_compute_classic_replay_rows(
        path, selected_subcarriers=replay.DEFAULT_SUBCARRIERS,
        replay_kind="classic_long",
    )
    timing = rows["timing"]
    times = packet_times(packets, timing["interval_us"])
    result = {}
    calibrated = replay._calibrate_classic_replay_rows_with_boundary(rows["calibration"], timing)
    if calibrated is not None:
        threshold, row = calibrated
        boundary = int(rows["calibration"]["packet_index"][row])
        detector = replay.build_lightweight_detector(threshold=threshold, timing=timing)
        detector._adapted_threshold_ready = True
        mask = np.asarray(rows["static"]["packet_index"]) > boundary
        phase = {k: np.asarray(v)[mask] for k, v in rows["static"].items()}
        states = replay._score_classic_replay_phase_rows(phase, detector, include_unready=True)
        result["classic"] = score_timeline(
            times, phase["packet_index"], states, phase["ready"],
            cutoff_us=int(times[boundary]),
        )
    else:
        result["classic"] = score_timeline(times, [], [], [])
        result["classic"]["calibration_failed"] = True
    ml = replay.load_or_compute_ml_replay_rows(
        path, selected_subcarriers=replay.DEFAULT_SUBCARRIERS,
        window_size=timing["window_packets"], sample_contract="replay_tick",
    )
    states = [predict(x) > 0.5 for x in ml["X"]]
    result["ml"] = score_timeline(times, ml["packet_index"], states, np.ones(len(states), dtype=bool))
    return result


def summarize(entries):
    """Pool wall-clock ticks and retain the weakest per-recording recall."""
    totals = {
        phase: {key: sum(e[phase][key] for e in entries)
                for key in ("ticks", "available", "raw_positive", "effective_positive", "alarms", "seconds")}
        for phase in ("static", "motion")
    }
    def percent(n, d):
        return 100.0 * n / d if d else None
    idle, motion = totals["static"], totals["motion"]
    first = [e["motion"]["first_detection_s"] for e in entries
             if e["motion"]["first_detection_s"] is not None]
    per_pair = [percent(e["motion"]["raw_positive"], e["motion"]["ticks"])
                for e in entries if e["motion"]["ticks"]]
    return {
        "pairs": len(entries),
        "motion_coverage": percent(motion["available"], motion["ticks"]),
        "idle_coverage": percent(idle["available"], idle["ticks"]),
        "conditional_recall": percent(motion["raw_positive"], motion["available"]),
        "operational_recall": percent(motion["raw_positive"], motion["ticks"]),
        "min_operational_recall": min(per_pair) if per_pair else None,
        "effective_recall": percent(motion["effective_positive"], motion["ticks"]),
        "effective_fp": percent(idle["effective_positive"], idle["ticks"]),
        "alarms": idle["alarms"],
        "alarms_per_hour": idle["alarms"] * 3600 / idle["seconds"] if idle["seconds"] else None,
        "max_acquisition_s": max(first) if first else None,
        "never_detected": len(entries) - len(first),
        "calibration_failures": sum(bool(e.get("calibration_failed")) for e in entries),
        "interruptions": sum(e["motion"]["interruptions"] for e in entries),
        "max_interruption_s": max((e["motion"]["max_interruption_s"] for e in entries), default=0.0),
        "idle_seconds": idle["seconds"],
        "motion_seconds": motion["seconds"],
    }


def collect_audit(progress=None):
    """Score selection and holdout separately, plus each quiet recording."""
    groups = defaultdict(list)
    pairs = replay.get_available_paired_datasets(
        synthetic=False, roles=replay.REPORT_DATASET_ROLES,
    )
    for static, motion, _sc, chip, _id in pairs:
        role = replay.get_paired_dataset_role(static)
        link = "weak" if replay.is_low_rssi_paired_dataset(static) else "normal"
        if progress:
            progress(f"operational audit: {role} {chip} {static.stem}")
        for algorithm, metrics in score_pair(static, motion).items():
            groups[(role, link, algorithm)].append(metrics)
    quiet = []
    for path, motion_start, packet_count, chip, entry in replay.get_available_long_test_dataset_specs():
        if motion_start != packet_count:
            continue
        if progress:
            progress(f"quiet operational audit: {path.stem}")
        for algorithm, metrics in score_quiet(path).items():
            quiet.append({"file": path.name, "role": entry["dataset_role"], "chip": chip,
                          "detector": algorithm, **metrics})
    catalog = replay._load_dataset_info()
    environments = defaultdict(set)
    for entries in catalog["files"].values():
        for entry in entries:
            environments[entry.get("dataset_role", "exclude")].add(entry.get("environment", "unknown"))
    return {
        "exposed_holdout": replay.DATA_DIR.resolve() == (repo_root() / "data").resolve(),
        "splits": [{"role": role, "link": link, "detector": algorithm, **summarize(entries)}
                   for (role, link, algorithm), entries in sorted(groups.items())],
        "quiet": quiet,
        "unseen_reserved_environments": sorted(
            (environments["selection"] | environments["holdout"]) - environments["train"]
        ),
    }


def model_provenance():
    """Read actual artifacts; never infer the original fit catalog from today's catalog."""
    from . import ml_weights
    root = repo_root()
    paths = [root / "tools/lib/ml_weights.py", root / "src/cpp/core/ml_weights.h"]
    header = paths[0].read_text()
    metadata = re.search(r"^Training provenance: (.+)$", header, re.M)
    training = None
    if metadata and metadata.group(1) != "unavailable":
        training = json.loads(metadata.group(1))
    return {
        "seed": re.search(r"^Seed: (\d+)", header, re.M).group(1),
        "trained": re.search(r"^Trained: (.+)$", header, re.M).group(1),
        "normalization": ml_weights.NORMALIZATION_MODE,
        "features": ml_weights.FEATURE_NAMES,
        "training": training,
        "artifacts": {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
    }


def render_audit(audit):
    """Render the operational replay tables with the definitions they need."""
    split_note = (
        "The holdout was inspected during development, so it is a regression set, not a sealed test."
        if audit.get("exposed_holdout", True) else
        "A holdout role alone does not make this catalog a sealed test."
    )
    def fmt(value, suffix="%"):
        return "n/a" if value is None else f"{value:.2f}{suffix}"

    def name(detector):
        return "Lightweight" if detector == "classic" else "High Accuracy"

    lines = [
        "## Operational replay", "",
        f"Selection is development data. {split_note}", "",
        f"- **Grid**: {replay.EVALUATION_INTERVAL_MS} ms of recorded time. A tick is covered when the latest ready evaluation is at most {AVAILABILITY_TOLERANCE_INTERVALS:g} evaluation intervals old.",
        "- **Raw**: detector output per tick; uncovered ticks count as IDLE.",
        f"- **Filtered**: after the {replay.MOTION_ON_HITS}/{replay.MOTION_OFF_HITS} hit filter, applied per evaluation as in firmware.",
        "- **Idle**: Lightweight scoring starts after its calibration.",
        "- **Acquisition**: first filtered MOTION from the start of the motion file. Lightweight keeps feature history from the static file; High Accuracy starts with an empty window.",
        "- **Interruption**: a filtered IDLE run that ends in MOTION again.",
        "",
        "Counts are pooled per split and link.", "",
        "| Split | Link | Detector | Pairs | Idle coverage | Motion coverage | Recall when covered | Raw recall | Min raw recall | Filtered recall | Filtered idle FP | Idle alarms/h | Max acquisition | Never detected | Calibration failures | Interruptions | Max interruption |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in audit["splits"]:
        lines.append(f"| {row['role']} | {row['link']} | {name(row['detector'])} | {row['pairs']} | " + " | ".join(
            fmt(row[k]) for k in ("idle_coverage", "motion_coverage", "conditional_recall", "operational_recall", "min_operational_recall", "effective_recall", "effective_fp")
        ) + f" | {fmt(row['alarms_per_hour'], '')} | {fmt(row['max_acquisition_s'], ' s')} | {row['never_detected']} | {row['calibration_failures']} | {row['interruptions']} | {fmt(row['max_interruption_s'], ' s')} |")
    unseen = ", ".join(audit["unseen_reserved_environments"]) or "none"
    lines.extend(["", f"Reserved environments not in `train`: {unseen}.", "",
                  "### Quiet recordings", "",
                  "| Recording | Split | Detector | Minutes | Coverage | Filtered FP | Alarms | Alarms/h |",
                  "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |"])
    for row in audit.get("quiet", []):
        n = row["ticks"]
        coverage = 100 * row["available"] / n if n else None
        fp = 100 * row["effective_positive"] / n if n else None
        rate = row["alarms"] * 3600 / row["seconds"] if row["seconds"] else None
        lines.append(f"| `{row['file']}` | {row['role']} | {name(row['detector'])} | {row['seconds']/60:.2f} | {fmt(coverage)} | {fmt(fp)} | {row['alarms']} | {fmt(rate, '')} |")
    return lines + ["", "---", ""]
