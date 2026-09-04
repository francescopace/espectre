# SPDX-License-Identifier: GPL-3.0-only
# Commercial licensing available under separate agreement; see LICENSING.md.
"""Shared runtime state, dependencies, and validation result types."""

import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parents[2]
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.lib.bootstrap import setup_paths  # noqa: E402,F401
from tools.lib.repo_paths import generated_data_dir
from tools.lib import dataset_metadata, performance_report
from tools.lib.temporal_csi_sampler import (  # noqa: E402
    MINIMUM_COVERAGE_DENOMINATOR,
    MINIMUM_COVERAGE_NUMERATOR,
)

DATA_DIR = SCRIPT_DIR.parent / "data"


DATASET_INFO = DATA_DIR / "dataset_info.json"


REPORT_OUTPUT = generated_data_dir() / "DATASET_QUALITY_CHECK.md"


DIAGNOSTIC_ALL_PHY = False


MINIMUM_TEMPORAL_OCCUPANCY_RATIO = (
    MINIMUM_COVERAGE_NUMERATOR / MINIMUM_COVERAGE_DENOMINATOR
)


TEMPORAL_OCCUPANCY_WARN_RATIO = 0.85


def configure_dataset_paths(data_dir, report_output=None):
    """Point the validator and shared report helpers at one dataset root."""
    global DATA_DIR, DATASET_INFO, REPORT_OUTPUT

    DATA_DIR = Path(data_dir).resolve()
    DATASET_INFO = DATA_DIR / "dataset_info.json"
    REPORT_OUTPUT = (
        DATA_DIR / "auto_generated" / "DATASET_QUALITY_CHECK.md"
        if report_output is None
        else Path(report_output).resolve()
    )
    dataset_metadata.DATA_DIR = DATA_DIR
    dataset_metadata.DATASET_INFO_FILE = DATASET_INFO
    performance_report.DATA_DIR = DATA_DIR


def configure_validation_mode(*, diagnostic_all_phy=False):
    """Select the packet view used after the supported-contract check."""
    global DIAGNOSTIC_ALL_PHY

    DIAGNOSTIC_ALL_PHY = bool(diagnostic_all_phy)


def _report_evaluation_view():
    """Return the stable report label for the selected packet view."""
    return (
        "all explicit PHY rows (diagnostic)"
        if DIAGNOSTIC_ALL_PHY
        else "HT20/HT-LTF"
    )


def _report_chip_filter(chip_filter=None):
    """Return the report scope using the validator's case-insensitive filter."""
    return f"chip={str(chip_filter).lower()}" if chip_filter else "all"


def _report_input_paths():
    """Return implementation and capture inputs to the quality report."""
    roots = (
        REPO_ROOT / "tools" / "lib",
        REPO_ROOT / "src" / "python" / "micro_espectre",
    )
    paths = {
        path
        for root in roots
        for path in root.rglob("*.py")
        if path.is_file()
    }
    paths.add(REPO_ROOT / "tools" / "validate_dataset_quality.py")
    paths.add(REPO_ROOT / "tools" / "train_ml_model.py")
    paths.update(DATA_DIR.glob("*/*.npz"))
    return tuple(sorted(paths))


class ValidationResult:
    """Single validation check result."""

    def __init__(self, name, status, message, value=None, domain='integrity'):
        self.name = name
        self.status = status  # 'PASS', 'WARN', 'FAIL'
        self.message = message
        self.value = value
        self.domain = domain

    def __repr__(self):
        icon = {'PASS': '✅', 'WARN': '⚠️', 'FAIL': '❌'}[self.status]
        val_str = f" ({self.value})" if self.value is not None else ""
        return f"{icon} {self.name}: {self.message}{val_str}"


def _tag_results(results, domain):
    """Assign a validation domain to results produced by one pipeline phase."""
    for result in results:
        result.domain = domain
    return results


def _is_issue_result(result):
    """Return True for console-worthy WARN/FAIL results."""
    return getattr(result, "status", None) in ("WARN", "FAIL")


def _issue_results(results):
    """Return only WARN/FAIL results."""
    return [result for result in results if _is_issue_result(result)]


def _result_counts(results):
    """Return stable PASS/WARN/FAIL counts for a result collection."""
    return {
        status: sum(1 for result in results if result.status == status)
        for status in ('PASS', 'WARN', 'FAIL')
    }
