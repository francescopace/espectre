# SPDX-License-Identifier: GPL-3.0-only
# Commercial licensing available under separate agreement; see LICENSING.md.
"""
ESPectre - Lightweight low-RSSI regression test

Validates the production Lightweight detector on every real low-RSSI pair that the
metadata exposes, across all covered chips and dataset roles.

Author: Francesco Pace <francesco.pace@gmail.com>
"""

import pytest

import config

from tools.lib.performance_report import (
    compute_classic_dataset_result,
    get_available_paired_datasets,
    get_paired_dataset_role,
    is_low_rssi_paired_dataset,
)


def _real_low_rssi_pairs():
    params = []
    for static_path, motion_path, _num_sc, chip, dataset_id in get_available_paired_datasets(
        synthetic=False
    ):
        if not is_low_rssi_paired_dataset(static_path):
            continue
        dataset_role = get_paired_dataset_role(static_path)
        assert dataset_role is not None
        params.append(
            pytest.param(
                static_path,
                motion_path,
                chip,
                dataset_role,
                dataset_id,
                id=f"{chip}:{dataset_role}:{dataset_id}",
            )
        )
    return params


@pytest.mark.parametrize(
    ("static_path", "motion_path", "chip", "dataset_role", "dataset_id"),
    _real_low_rssi_pairs(),
)
def test_production_classic_handles_real_low_rssi_pair(
    static_path,
    motion_path,
    chip,
    dataset_role,
    dataset_id,
):
    result = compute_classic_dataset_result(
        static_path,
        motion_path,
        tuple(config.DEFAULT_SUBCARRIERS),
        None,
    )

    assert result is not None, f"Lightweight startup calibration failed for {chip}:{dataset_role}:{dataset_id}"
    _threshold, metrics = result
    label = f"{chip}:{dataset_role}:{dataset_id}"
    # The physical-time replay exposes the real 10.85 ms cadence instead of
    # normalizing it to 10 ms. The weakest S3 pair now scores 83.6%, so 82 is a
    # coarse collapse guard; aggregate and empty-room tests remain the binding
    # production gates. Raise it only after the Lightweight feature work lands.
    assert metrics["recall"] >= 82.0, f"Lightweight weak-link recall too low for {label}: {metrics['recall']:.1f}%"
    # This is a sanity bound, not a false-positive gate. Static-presence
    # recordings contain a stationary person, whose breathing and small shifts
    # are real channel motion, so a share of these evaluations is the detector
    # working rather than failing. The empty-room gate in
    # test_validation_real_data.py is where zero alarms is asserted, and it is
    # the only stream in the corpus with nobody in the room. Corpus maximum is
    # 10.6%, so this bounds drift without encoding micro-motion as an error.
    assert metrics["fp_rate"] <= 12.0, (
        f"Lightweight static-presence motion share too high for {label}: {metrics['fp_rate']:.1f}%"
    )
