# SPDX-License-Identifier: GPL-3.0-only
# Commercial licensing available under separate agreement; see LICENSING.md.
"""Console and Markdown cell and table rendering."""

from . import core
import datetime
import os
import re
from pathlib import Path

import numpy as np

from tools.lib import dataset_metadata
from .catalog import (
    _is_missing_metadata_value,
    load_dataset_info,
)
from .core import (
    MINIMUM_TEMPORAL_OCCUPANCY_RATIO,
    TEMPORAL_OCCUPANCY_WARN_RATIO,
)
from .severity import (
    FAIL_MOTION_ACTIVE_RATIO,
    FAIL_STATIC_ACTIVE_RATIO,
    FEATURE_EXCURSION_FAIL_RATIO,
    FEATURE_EXCURSION_WARN_RATIO,
    MAX_STATIC_ACTIVE_RATIO,
    MIN_MOTION_ACTIVE_RATIO,
    REFERENCE_EXCURSION_FAIL_RATIO,
    REFERENCE_EXCURSION_WARN_RATIO,
    REFERENCE_LONGEST_BURST_FAIL_SECONDS,
    REFERENCE_LONGEST_BURST_WARN_SECONDS,
    _metric_thresholds,
    _pair_separation_severity,
    _reference_cleanliness_severity,
    _row_severity_profile,
    _score_value_severity,
    _threshold_severity,
)

def _mark_cell(text, severity, *, markdown=False):
    """Append soft WARN/FAIL icons to a cell value."""
    if severity == 'fail':
        marked = f"{text} ❌"
    elif severity == 'warn':
        marked = f"{text} ⚠️"
    else:
        return text
    if markdown:
        return f"**{marked}**"
    return marked


def _format_unusable_na_cell(*, markdown=False):
    """Mark a missing cleanliness score when temporal admission produced no rows."""
    return _mark_cell("n/a", "warn", markdown=markdown)


def _format_percent_ratio_cell(
    value,
    *,
    warn_above=None,
    fail_above=None,
    warn_below=None,
    fail_below=None,
    markdown=False,
):
    """Format a percentage-ratio cell and mark soft WARN/FAIL breaches."""
    text = f"{float(value):.1%}"
    severity = _threshold_severity(
        value,
        warn_above=warn_above,
        fail_above=fail_above,
        warn_below=warn_below,
        fail_below=fail_below,
    )
    return _mark_cell(text, severity, markdown=markdown)


def _format_static_above_cell(value, *, markdown=False):
    return _format_percent_ratio_cell(
        value,
        warn_above=MAX_STATIC_ACTIVE_RATIO,
        fail_above=FAIL_STATIC_ACTIVE_RATIO,
        markdown=markdown,
    )


def _format_motion_above_cell(value, *, markdown=False):
    return _format_percent_ratio_cell(
        value,
        warn_below=MIN_MOTION_ACTIVE_RATIO,
        fail_below=FAIL_MOTION_ACTIVE_RATIO,
        markdown=markdown,
    )


def _format_quiet_fp_cell(value, *, markdown=False):
    return _format_percent_ratio_cell(
        value,
        warn_above=FEATURE_EXCURSION_WARN_RATIO,
        fail_above=FEATURE_EXCURSION_FAIL_RATIO,
        markdown=markdown,
    )


def _format_margin_mad_cell(value, *, markdown=False, severity_profile=None):
    """Format a logit-margin MAD cell and mark soft WARN/FAIL breaches."""
    severity = _threshold_severity(value, **_metric_thresholds("mad", severity_profile))
    return _mark_cell(f"{float(value):.2f}", severity, markdown=markdown)


def _format_packet_rate_cell(value, *, markdown=False):
    """Format one observed packet-rate cell."""
    del markdown
    return f"{float(value):.1f}"


def _format_occupancy_cell(value, *, markdown=False):
    """Format mean temporal occupancy with the production admission floor."""
    if value is None:
        return "n/a"
    return _format_percent_ratio_cell(
        value,
        warn_below=TEMPORAL_OCCUPANCY_WARN_RATIO,
        fail_below=MINIMUM_TEMPORAL_OCCUPANCY_RATIO,
        markdown=markdown,
    )


def _format_pair_occupancy_cell(
    static_occupancy,
    motion_occupancy,
    *,
    markdown=False,
):
    """Format static/motion mean temporal occupancy in one shared cell."""
    return " / ".join(
        (
            _format_occupancy_cell(static_occupancy, markdown=markdown),
            _format_occupancy_cell(motion_occupancy, markdown=markdown),
        )
    )


def _format_burst_cell(value, *, markdown=False, severity_profile=None):
    """Format a longest-activation-burst cell and mark soft WARN/FAIL breaches."""
    severity = _threshold_severity(value, **_metric_thresholds("burst", severity_profile))
    return _mark_cell(f"{float(value):.1f}s", severity, markdown=markdown)


def _format_margin_q95_cell(value, *, markdown=False, severity_profile=None):
    """Format one idle q95 margin cell with exploratory soft marks."""
    severity = _threshold_severity(value, **_metric_thresholds("q95", severity_profile))
    return _mark_cell(f"{float(value):.2f}", severity, markdown=markdown)


def _format_margin_drift_cell(value, *, markdown=False, severity_profile=None):
    """Format one absolute half-to-half margin drift cell."""
    severity = _threshold_severity(value, **_metric_thresholds("drift", severity_profile))
    return _mark_cell(f"{float(value):.2f}", severity, markdown=markdown)


def _format_pair_separation_cell(pair_separation, *, markdown=False, severity_profile=None):
    """Format Sep as an idle/motion AUC with soft marks."""
    value = float(pair_separation)
    text = "n/a" if not np.isfinite(value) else f"{value:.4f}"
    return _mark_cell(
        text,
        _pair_separation_severity(pair_separation, severity_profile),
        markdown=markdown,
    )


def _format_score_cell(score, severity=None, *, markdown=False):
    """Format a 0-100 score cell, optionally with soft WARN/FAIL icons."""
    return _mark_cell(f"{float(score):.1f}", severity, markdown=markdown)


def _median_rssi_dbm(data):
    """Return the median per-packet RSSI in dBm, or None when unavailable."""
    if not hasattr(data, "files") or "rssi_dbm" not in data.files:
        return None
    rssi = np.asarray(data["rssi_dbm"], dtype=np.float64)
    if rssi.size == 0:
        return None
    return float(np.median(rssi))


def _format_rssi_value(rssi_dbm):
    """Format one RSSI value for table display."""
    if rssi_dbm is None:
        return "n/a"
    return f"{int(round(float(rssi_dbm)))}"


def _format_rssi_cell(rssi_dbm):
    """Format the RSSI cell for one single-capture row."""
    return _format_rssi_value(rssi_dbm)


def _format_pair_rssi_cell(static_rssi_dbm, motion_rssi_dbm):
    """Format the shared RSSI cell for one static/motion pair."""
    if static_rssi_dbm is None and motion_rssi_dbm is None:
        return "n/a"
    if static_rssi_dbm is None:
        return f"n/a / {_format_rssi_value(motion_rssi_dbm)}"
    if motion_rssi_dbm is None:
        return f"{_format_rssi_value(static_rssi_dbm)} / n/a"
    return (
        f"{int(round(float(static_rssi_dbm)))} / "
        f"{int(round(float(motion_rssi_dbm)))}"
    )


def _format_pair_packet_rate_cell(static_packet_rate_pps, motion_packet_rate_pps):
    """Format the shared PPS cell for one static/motion pair."""
    if static_packet_rate_pps is None and motion_packet_rate_pps is None:
        return "n/a"
    if static_packet_rate_pps is None:
        return f"n/a / {_format_packet_rate_cell(motion_packet_rate_pps)}"
    if motion_packet_rate_pps is None:
        return f"{_format_packet_rate_cell(static_packet_rate_pps)} / n/a"
    return (
        f"{_format_packet_rate_cell(static_packet_rate_pps)} / "
        f"{_format_packet_rate_cell(motion_packet_rate_pps)}"
    )


def _format_reference_basis_cell(reference_stats):
    """Format reference scope and capture count."""
    if reference_stats is None:
        return "n/a"
    basis = "env" if "env" in reference_stats["basis"] else "chip"
    return f"{basis}/{reference_stats['reference_count']}"


def _format_reference_excursion_cell(reference_stats, *, markdown=False):
    """Format the share of blocks above the reference p95."""
    if reference_stats is None:
        return "n/a"
    severity = _threshold_severity(
        reference_stats["excursion_ratio"],
        warn_above=REFERENCE_EXCURSION_WARN_RATIO,
        fail_above=REFERENCE_EXCURSION_FAIL_RATIO,
    )
    return _mark_cell(
        f"{reference_stats['excursion_ratio']:.1%}",
        severity,
        markdown=markdown,
    )


def _format_reference_burst_cell(reference_stats, *, markdown=False):
    """Format the longest run above the reference p99."""
    if reference_stats is None:
        return "n/a"
    severity = _threshold_severity(
        reference_stats["longest_burst_seconds"],
        warn_above=REFERENCE_LONGEST_BURST_WARN_SECONDS,
        fail_above=REFERENCE_LONGEST_BURST_FAIL_SECONDS,
    )
    return _mark_cell(
        f"{reference_stats['longest_burst_seconds']:.1f}s",
        severity,
        markdown=markdown,
    )


_IDLE_EVIDENCE_SCORE_HEADER = (
    "| Chip | Env | File | RSSI | PPS | Occ | Exc | Burst | Tail | Drift | Score |"
)


_IDLE_EVIDENCE_SCORE_SEPARATOR = (
    "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"
)


_IDLE_EVIDENCE_SCORE_CONSOLE_SEPARATOR = (
    "  |------|-----|------|---------:|----:|------:|-----:|------:|-----:|------:|------:|"
)


def _idle_evidence_file_cell(row, label, *, markdown=False):
    """Return the File cell for one idle-evidence score row."""
    if markdown:
        return _md_file_link(row["display_date"], label, row["filename"])
    return row["display_date"]


def _format_idle_evidence_score_row(
    row,
    *,
    label,
    markdown=False,
    review_profiles=None,
):
    """Format one idle-evidence score row with the shared column schema.

    Every baseline-score component is shown next to the final Score, plus
    observed packet rate, temporal occupancy, and exploratory tail/drift signals.
    """
    file_cell = _idle_evidence_file_cell(row, label, markdown=markdown)
    baseline = row["baseline"]
    severity_profile = _row_severity_profile(review_profiles, label, row["chip"])
    score_value = baseline["score"]
    baseline_cell = _format_score_cell(
        score_value,
        _score_value_severity(score_value, severity_profile),
        markdown=markdown,
    )
    if markdown:
        return (
            f"| {row['chip']} | {row.get('environment', '?')} | {file_cell} | "
            f"{_format_rssi_cell(row.get('rssi_dbm'))} | "
            f"{_format_packet_rate_cell(baseline['packet_rate_pps'])} | "
            f"{_format_occupancy_cell(row.get('mean_occupancy'), markdown=True)} | "
            f"{_format_quiet_fp_cell(baseline['fp_rate'], markdown=True)} | "
            f"{_format_burst_cell(baseline['longest_burst_seconds'], markdown=True, severity_profile=severity_profile)} | "
            f"{_format_margin_q95_cell(baseline['margin_q95'], markdown=True, severity_profile=severity_profile)} | "
            f"{_format_margin_drift_cell(baseline['margin_drift_abs'], markdown=True, severity_profile=severity_profile)} | "
            f"{baseline_cell} |"
        )
    return (
        f"  | {row['chip']:<4} | {row.get('environment', '?'):<11} | "
        f"{file_cell:<16} | "
        f"{_format_rssi_cell(row.get('rssi_dbm')):>9} | "
        f"{_format_packet_rate_cell(baseline['packet_rate_pps']):>4} | "
        f"{_format_occupancy_cell(row.get('mean_occupancy')):>6} | "
        f"{_format_quiet_fp_cell(baseline['fp_rate']):>5} | "
        f"{_format_burst_cell(baseline['longest_burst_seconds'], severity_profile=severity_profile):>6} | "
        f"{_format_margin_q95_cell(baseline['margin_q95'], severity_profile=severity_profile):>5} | "
        f"{_format_margin_drift_cell(baseline['margin_drift_abs'], severity_profile=severity_profile):>6} | "
        f"{baseline_cell:>8} |"
    )


def _render_score_table(rows, table_spec, *, markdown=False, review_profiles=None):
    """Return lines for one indicative score table, or [] when empty."""
    if not rows:
        return []

    lines = []
    title = table_spec["title"]
    if markdown:
        lines.append(f"\n## {title}\n")
        intro = table_spec.get("intro")
        if intro:
            lines.append(f"{intro}\n")
        lines.append(table_spec["header"])
        lines.append(table_spec["separator"])
    else:
        if table_spec.get("console_heading", True):
            lines.append(f"  {title}:")
        console_header = table_spec.get("console_header", table_spec["header"])
        lines.append(f"  {console_header}")
        lines.append(table_spec["console_separator"])

    format_row = table_spec["format_row"]
    for row in sorted(rows, key=table_spec["sort_key"]):
        lines.append(
            format_row(row, markdown=markdown, review_profiles=review_profiles)
        )
    return lines


def _idle_evidence_table_spec(title, table_key, *, file_label=None):
    """Build one idle-evidence score-table spec for the shared renderer."""
    return {
        "title": title,
        "table_key": table_key,
        "header": _IDLE_EVIDENCE_SCORE_HEADER,
        "separator": _IDLE_EVIDENCE_SCORE_SEPARATOR,
        "console_separator": _IDLE_EVIDENCE_SCORE_CONSOLE_SEPARATOR,
        "sort_key": lambda item: -item["baseline"]["score"],
        "format_row": lambda row, *, markdown=False, review_profiles=None: _format_idle_evidence_score_row(
            row,
            label=file_label or table_key,
            markdown=markdown,
            review_profiles=review_profiles,
        ),
    }


_PRESENCE_SCORE_TABLE = _idle_evidence_table_spec("Presence Scores", "static_presence")


_EMPTY_SCORE_TABLE = _idle_evidence_table_spec("Empty Scores", "empty")


_LONG_TEST_SCORE_TABLE = _idle_evidence_table_spec(
    "Long-recording scores",
    "long_test",
    file_label="empty",
)


def _format_pair_score_row(row, *, markdown=False, review_profiles=None):
    """Format one static_presence/motion pair score row."""
    score_value = row.get("feature_score", 0.0)
    reference_stats = row.get("reference_cleanliness")
    reference_severity = _reference_cleanliness_severity(reference_stats)
    severity_profile = _row_severity_profile(review_profiles, "pair", row["chip"])
    severity = reference_severity or _score_value_severity(score_value, severity_profile)
    files_cell = _pair_files_cell(
        row["static_presence"],
        row["motion"],
        row.get("static_date", "?"),
        row.get("motion_date", "?"),
        markdown=markdown,
    )
    if markdown:
        return (
            f"| {row['chip']} | {row.get('environment', '?')} | {files_cell} | "
            f"{_format_pair_rssi_cell(row.get('static_rssi_dbm'), row.get('motion_rssi_dbm'))} | "
            f"{_format_pair_packet_rate_cell(row.get('static_packet_rate_pps'), row.get('motion_packet_rate_pps'))} | "
            f"{_format_pair_occupancy_cell(row.get('static_mean_occupancy'), row.get('motion_mean_occupancy'), markdown=True)} | "
            f"{_format_reference_basis_cell(reference_stats)} | "
            f"{_format_motion_above_cell(row['motion_coverage'], markdown=True)} | "
            f"{_format_pair_separation_cell(row['pair_separation'], markdown=True, severity_profile=severity_profile)} | "
            f"{_format_reference_excursion_cell(reference_stats, markdown=True)} | "
            f"{_format_reference_burst_cell(reference_stats, markdown=True)} | "
            f"{_format_score_cell(score_value, severity, markdown=True)} |"
        )
    return (
        f"  | {row['chip']:<4} | {row.get('environment', '?'):<11} | "
        f"{files_cell:<23} | "
        f"{_format_pair_rssi_cell(row.get('static_rssi_dbm'), row.get('motion_rssi_dbm')):>17} | "
        f"{_format_pair_packet_rate_cell(row.get('static_packet_rate_pps'), row.get('motion_packet_rate_pps')):>13} | "
        f"{_format_pair_occupancy_cell(row.get('static_mean_occupancy'), row.get('motion_mean_occupancy')):>15} | "
        f"{_format_reference_basis_cell(reference_stats):>7} | "
        f"{_format_motion_above_cell(row['motion_coverage']):>5} | "
        f"{_format_pair_separation_cell(row['pair_separation'], severity_profile=severity_profile):>6} | "
        f"{_format_reference_excursion_cell(reference_stats):>7} | "
        f"{_format_reference_burst_cell(reference_stats):>8} | "
        f"{_format_score_cell(score_value, severity):>8} |"
    )


_PAIR_SCORE_TABLE = {
    "title": "Pair Scores",
    "table_key": "pair",
    "header": (
        "| Chip | Env | static_presence / motion | RSSI | PPS | Occ | Ref | Cover | Sep | RefExc | RefBurst | Score |"
    ),
    "separator": "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    "console_header": (
        "| Chip | Env | static_presence / motion | RSSI | PPS | Occ | Ref | Cover | Sep | RefExc | RefBurst | Score |"
    ),
    "console_separator": (
        "  |------|-----|-------------------------|-----------------:|-------------:|---------------:|--------:|------:|------:|-------:|---------:|------:|"
    ),
    "console_heading": False,
    "sort_key": lambda item: -item.get("feature_score", 0.0),
    "format_row": _format_pair_score_row,
}


_EXCLUDED_PAIR_SCORE_TABLE = {
    **_PAIR_SCORE_TABLE,
    "title": "Excluded Pair Diagnostics",
    "intro": (
        "These pairs keep `dataset_role: exclude` and stay outside the validation "
        "summary. `Cover` and `Sep` measure static/motion separation, while "
        "`RefExc` and `RefBurst` measure the static capture against independent "
        "idle references. The final `Score` is capped by both views, so a "
        "contaminated static capture cannot receive 100 merely because motion "
        "separates from it."
    ),
}


def _format_excluded_idle_row(row, *, markdown=False, review_profiles=None):
    """Format one excluded idle capture against independent references."""
    del review_profiles
    reference_stats = row.get("reference_cleanliness")
    severity = _reference_cleanliness_severity(reference_stats)
    file_cell = _idle_evidence_file_cell(row, row["label"], markdown=markdown)
    if row.get("unusable"):
        na_cell = _format_unusable_na_cell(markdown=markdown)
        reference_cell = na_cell
        excursion_cell = na_cell
        burst_cell = na_cell
        score_cell = na_cell
    else:
        reference_cell = _format_reference_basis_cell(reference_stats)
        excursion_cell = _format_reference_excursion_cell(
            reference_stats, markdown=markdown
        )
        burst_cell = _format_reference_burst_cell(
            reference_stats, markdown=markdown
        )
        score_cell = (
            _format_score_cell(reference_stats["score"], severity, markdown=markdown)
            if reference_stats
            else "n/a"
        )
    if markdown:
        return (
            f"| {row['chip']} | {row.get('environment', '?')} | {file_cell} | "
            f"{_format_rssi_cell(row.get('rssi_dbm'))} | "
            f"{_format_packet_rate_cell(row.get('packet_rate_pps'))} | "
            f"{_format_occupancy_cell(row.get('mean_occupancy'), markdown=True)} | "
            f"{reference_cell} | "
            f"{excursion_cell} | "
            f"{burst_cell} | "
            f"{score_cell} |"
        )
    return (
        f"  | {row['chip']:<4} | {row.get('environment', '?'):<11} | "
        f"{file_cell:<16} | {_format_rssi_cell(row.get('rssi_dbm')):>4} | "
        f"{_format_packet_rate_cell(row.get('packet_rate_pps')):>5} | "
        f"{_format_occupancy_cell(row.get('mean_occupancy')):>6} | "
        f"{reference_cell:>7} | "
        f"{excursion_cell:>7} | "
        f"{burst_cell:>8} | "
        f"{score_cell:>8} |"
    )


_EXCLUDED_IDLE_SCORE_TABLE = {
    "title": "Excluded Idle Diagnostics",
    "table_key": "excluded_idle",
    "intro": (
        "These excluded idle recordings are compared with admitted, non-long "
        "idle captures from the same chip, link class, and packet-rate class; "
        "the same environment is preferred when enough references exist. "
        "`RefExc` is the share of five-second blocks above the "
        "reference p95, and `RefBurst` is the longest contiguous run above its "
        "p99. These are contamination signals for review, not automatic labels. "
        "When fixed temporal admission produces no usable feature rows, "
        "`Ref`, `RefExc`, `RefBurst`, and `Score` are marked `n/a ⚠️` and the "
        "row is listed first."
    ),
    "header": "| Chip | Env | File | RSSI | PPS | Occ | Ref | RefExc | RefBurst | Score |",
    "separator": "|---|---|---|---:|---:|---:|---:|---:|---:|---:|",
    "console_separator": (
        "  |------|-----|------------------|-----:|------:|------:|--------:|-------:|---------:|------:|"
    ),
    "console_heading": False,
    "sort_key": lambda item: (
        0 if item.get("unusable") else 1,
        (item.get("reference_cleanliness") or {"score": float("inf")})["score"],
    ),
    "format_row": _format_excluded_idle_row,
}


def _dataset_file_href(label, filename):
    """Return a report-relative href for one catalogued dataset NPZ."""
    target = core.DATA_DIR / label / filename
    for entry in load_dataset_info().get("files", {}).get(label, []):
        if str(entry.get("filename", "")) == str(filename):
            target = dataset_metadata.resolve_entry_path(label, entry)
            break
    return Path(os.path.relpath(target, core.REPORT_OUTPUT.parent)).as_posix()


def _report_source_path():
    """Return a stable repository-relative catalog path when possible."""
    try:
        return core.DATASET_INFO.relative_to(core.REPO_ROOT).as_posix()
    except ValueError:
        return core.DATASET_INFO.as_posix()


def _report_evaluation_view_is_current(chip_filter=None):
    """Return whether the report matches the selected packet view and scope."""
    if not core.REPORT_OUTPUT.exists():
        return False
    expected = f"Evaluation view: `{core._report_evaluation_view()}`"
    lines = core.REPORT_OUTPUT.read_text(encoding="utf-8").splitlines()
    scope = f"Chip filter: `{core._report_chip_filter(chip_filter)}`"
    return expected in lines and scope in lines


def _md_file_link(text, label, filename):
    """Markdown link with a short readable label pointing at one dataset NPZ."""
    return f"[{text}]({_dataset_file_href(label, filename)})"


def _pair_files_cell(
    static_filename,
    motion_filename,
    static_date,
    motion_date,
    *,
    markdown=False,
):
    """Render static_presence/motion links using readable capture dates."""
    if markdown:
        return (
            f"{_md_file_link(static_date, 'static_presence', static_filename)} / "
            f"{_md_file_link(motion_date, 'motion', motion_filename)}"
        )
    return f"{static_date} / {motion_date}"


def _entry_display_date(entry, filename=None):
    """Return a compact capture date for quiet-test table display."""
    collected_at = entry.get("collected_at") if isinstance(entry, dict) else None
    if not _is_missing_metadata_value(collected_at):
        try:
            return datetime.datetime.fromisoformat(str(collected_at)).strftime(
                "%Y-%m-%d %H:%M"
            )
        except ValueError:
            pass

    name = filename or (entry.get("filename") if isinstance(entry, dict) else None)
    if name:
        match = re.search(r"_(\d{8})_(\d{6})(?:_\d+)*\.npz$", str(name))
        if match:
            day = datetime.datetime.strptime(match.group(1), "%Y%m%d")
            clock = datetime.datetime.strptime(match.group(2), "%H%M%S")
            return f"{day.strftime('%Y-%m-%d')} {clock.strftime('%H:%M')}"
    return "?"


def _render_unusable_excluded_idle_section(rows):
    """Return markdown lines listing excluded idle captures with no feature rows."""
    unusable = [
        row for row in rows
        if row.get("unusable")
    ]
    if not unusable:
        return []
    lines = ["\n## Unscorable excluded idle\n"]
    lines.append(
        "These excluded idle captures produced no usable feature rows after "
        "fixed temporal admission, so cleanliness cannot be scored. They remain "
        "in the catalog for provenance and are listed first in Excluded Idle Diagnostics.\n"
    )
    for row in sorted(
        unusable,
        key=lambda item: (
            str(item.get("chip", "")),
            str(item.get("display_date", "")),
            str(item.get("filename", "")),
        ),
    ):
        filename = str(row.get("filename", "?"))
        file_cell = _md_file_link(
            row.get("display_date", filename),
            str(row.get("label", "empty")),
            filename,
        )
        lines.append(
            f"- {row.get('chip', '?')} {row.get('environment', '?')} {file_cell}"
        )
    return lines
