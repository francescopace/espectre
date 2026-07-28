"""
ESPectre - Host-Side Candidate Feature Tests

Tests for the evaluation-only candidates in tools/lib/candidate_features.py.

Author: Francesco Pace <francesco.pace@gmail.com>
License: GPLv3
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
TRAIN_SCRIPT = REPO_ROOT / "tools" / "train_ml_model.py"

from csi_features import ALL_FEATURES, DEFAULT_FEATURES, L1_DELTA_LAG
from tools.lib.candidate_features import (
    CANDIDATE_FEATURES,
    HT20_LIVE_BINS,
    ChannelCoherenceTracker,
    assemble_feature_vector,
    candidate_values,
    complex_profile,
    delay_compensated_coherence,
    needs_channel_coherence,
    split_feature_names,
)


def _load_train_module():
    spec = importlib.util.spec_from_file_location("train_ml_model_candidates", TRAIN_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


# Payloads stay well inside int8 so SCALE_FACTOR is exact and cannot clip:
# a scaled stream must be the same channel at another gain, not a re-quantized
# and saturated one, or the test would measure its own arithmetic.
SAMPLE_LIMIT = 31
SCALE_FACTOR = 4


def _payload(rng, amplitude=8.0):
    """Build one HT20 payload with the guard bins left null."""
    raw = np.zeros(128, dtype=np.int8)
    for sc in HT20_LIVE_BINS:
        imag, real = rng.normal(0.0, amplitude, 2)
        raw[2 * sc] = int(np.clip(round(imag), -SAMPLE_LIMIT, SAMPLE_LIMIT))
        raw[2 * sc + 1] = int(np.clip(round(real), -SAMPLE_LIMIT, SAMPLE_LIMIT))
    return raw


def _still_stream(count, rng, amplitude=8.0):
    """A channel that does not move: one profile plus small per-packet noise."""
    base = _payload(rng, amplitude)
    stream = []
    for _ in range(count):
        jitter = rng.normal(0.0, 0.8, 128).round()
        stream.append(
            np.clip(base.astype(np.int16) + jitter, -SAMPLE_LIMIT, SAMPLE_LIMIT)
            .astype(np.int8)
        )
    return stream


def _scaled(stream, factor=SCALE_FACTOR):
    """Exactly the same stream at another gain."""
    return [(packet.astype(np.int16) * factor).astype(np.int8) for packet in stream]


class TestCandidateSurface:
    def test_candidates_stay_out_of_the_production_set(self):
        """The production surface is the exported set and nothing else."""
        assert tuple(ALL_FEATURES) == tuple(DEFAULT_FEATURES)
        for name in CANDIDATE_FEATURES:
            assert name not in ALL_FEATURES

    def test_split_and_assemble_preserve_the_requested_order(self):
        requested = ['turb_zcr', 'chan_coh_gap', 'turb_autocorr']
        production, candidates = split_feature_names(requested)
        assert production == ['turb_zcr', 'turb_autocorr']
        assert candidates == ['chan_coh_gap']
        vector = assemble_feature_vector(
            requested, production, [0.25, 0.75], {'chan_coh_gap': 0.5}
        )
        assert vector == [0.25, 0.5, 0.75]

    def test_needs_channel_coherence_only_for_its_own_features(self):
        assert needs_channel_coherence(['chan_coh_gap_q20'])
        assert not needs_channel_coherence(list(DEFAULT_FEATURES))

    def test_candidate_values_require_the_preprocessed_tracker(self):
        with pytest.raises(ValueError, match="coherence tracker"):
            candidate_values(['chan_coh_lag_ratio'], None)


class TestDelayCompensatedCoherence:
    def test_identical_profiles_are_fully_coherent(self):
        rng = np.random.default_rng(11)
        profile = complex_profile(_payload(rng))
        assert delay_compensated_coherence(profile, profile) == pytest.approx(1.0)

    def test_a_pure_delay_is_compensated(self):
        """A sampling-time offset is a phase ramp, not decorrelation."""
        rng = np.random.default_rng(12)
        profile = complex_profile(_payload(rng))
        ramp = np.exp(1j * 0.05 * np.asarray(HT20_LIVE_BINS, dtype=np.float64))
        assert delay_compensated_coherence(profile * ramp, profile) == pytest.approx(
            1.0, abs=1e-6
        )

    def test_a_common_phase_offset_is_ignored(self):
        """The carrier offset shifts every bin alike and must not register."""
        rng = np.random.default_rng(13)
        profile = complex_profile(_payload(rng))
        rotated = profile * np.exp(1j * 1.1)
        assert delay_compensated_coherence(rotated, profile) == pytest.approx(1.0)

    def test_empty_payload_is_reported_as_zero(self):
        empty = np.zeros(len(HT20_LIVE_BINS), dtype=np.complex128)
        assert delay_compensated_coherence(empty, empty) == 0.0


class TestScaleInvariance:
    def test_coherence_ignores_a_gain_change_between_packets(self):
        """Per-packet gain must not read as channel change."""
        rng = np.random.default_rng(20)
        current = complex_profile(_payload(rng))
        reference = complex_profile(_payload(rng))
        plain = delay_compensated_coherence(current, reference)
        assert delay_compensated_coherence(current * 7.5, reference * 0.3) == (
            pytest.approx(plain)
        )

    def test_every_candidate_is_scale_invariant(self):
        """The int8 scaling factor varies per packet and is never recorded."""
        stream = _still_stream(60, np.random.default_rng(21))
        plain = ChannelCoherenceTracker(window_size=40, lag=L1_DELTA_LAG)
        scaled = ChannelCoherenceTracker(window_size=40, lag=L1_DELTA_LAG)
        for packet, lifted in zip(stream, _scaled(stream)):
            plain.process_packet(packet)
            scaled.process_packet(lifted)
        for name in CANDIDATE_FEATURES:
            left = candidate_values([name], plain)[name]
            right = candidate_values([name], scaled)[name]
            assert left == pytest.approx(right, rel=1e-9), (
                f"{name} moved when the payload was scaled"
            )


class TestChannelCoherenceTracker:
    def test_a_still_channel_keeps_the_lag_ratio_near_one(self):
        rng = np.random.default_rng(31)
        tracker = ChannelCoherenceTracker(window_size=40, lag=L1_DELTA_LAG)
        for packet in _still_stream(80, rng):
            tracker.process_packet(packet)
        assert tracker.coherence_lag_ratio() == pytest.approx(1.0, abs=0.05)
        assert 0.0 <= tracker.mean_coherence() <= 1.0
        assert tracker.coherence_gap() == pytest.approx(0.0, abs=0.05)
        assert 0.0 <= tracker.coherence_gap_low_frac() <= 1.0
        assert tracker.coherence_gap_q20() == pytest.approx(0.0, abs=0.08)

    def test_a_changing_channel_lowers_the_lag_ratio(self):
        """The long lag must decorrelate before the adjacent one does."""
        rng = np.random.default_rng(32)
        still = ChannelCoherenceTracker(window_size=40, lag=L1_DELTA_LAG)
        moving = ChannelCoherenceTracker(window_size=40, lag=L1_DELTA_LAG)
        for packet in _still_stream(120, np.random.default_rng(32)):
            still.process_packet(packet)
        base = _payload(rng).astype(np.float64)
        drift = rng.normal(0.0, 1.0, 128)
        for step in range(120):
            packet = np.clip(
                base + drift * step * 0.35 + rng.normal(0.0, 0.8, 128),
                -SAMPLE_LIMIT, SAMPLE_LIMIT,
            ).round().astype(np.int8)
            moving.process_packet(packet)
        assert moving.coherence_lag_ratio() < still.coherence_lag_ratio()
        assert moving.coherence_gap() > still.coherence_gap()
        assert moving.coherence_gap_q20() > still.coherence_gap_q20()
        assert 0.0 <= moving.coherence_gap_low_frac() <= 1.0

    def test_reset_clears_the_running_window(self):
        rng = np.random.default_rng(33)
        tracker = ChannelCoherenceTracker(window_size=40, lag=L1_DELTA_LAG)
        for packet in _still_stream(60, rng):
            tracker.process_packet(packet)
        tracker.reset()
        assert tracker.coherence_lag_ratio() == 1.0
        assert tracker.mean_coherence() == 1.0
        assert tracker.coherence_gap() == 0.0
        assert tracker.coherence_gap_low_frac() == 0.0
        assert tracker.coherence_gap_q20() == 0.0


class TestTrainerIntegration:
    def test_candidates_are_selectable_but_not_production(self):
        module = _load_train_module()
        for name in CANDIDATE_FEATURES:
            assert name in module.selectable_features()
            assert name not in module.TRAINING_FEATURES
            assert name not in module.CPP_FEATURE_IDS

    def test_export_refuses_a_candidate_without_a_cpp_id(self):
        module = _load_train_module()
        with pytest.raises(ValueError, match="no C\\+\\+ extractor id"):
            module.resolve_cpp_feature_ids(
                list(module.TRAINING_FEATURES) + [CANDIDATE_FEATURES[0]]
            )

    def test_redundancy_report_separates_explained_from_new(self, capsys):
        module = _load_train_module()
        rng = np.random.default_rng(51)
        baseline = list(module.TRAINING_FEATURES)
        columns = [rng.normal(size=400) for _ in baseline]
        explained = 2.0 * columns[0] - 0.5 * columns[1] + 3.0
        independent = rng.normal(size=400)
        X = np.column_stack(columns + [explained, independent])
        module.print_candidate_redundancy(
            X, baseline + ['cand_explained', 'cand_independent'], baseline
        )
        report = capsys.readouterr().out
        rows = {
            line.split()[0]: line.split()
            for line in report.splitlines()
            if line.startswith('cand_')
        }
        assert float(rows['cand_explained'][-1]) == pytest.approx(1.0, abs=1e-6)
        assert float(rows['cand_independent'][-1]) < 0.05

    def test_redundancy_report_is_silent_without_candidates(self, capsys):
        module = _load_train_module()
        baseline = list(module.TRAINING_FEATURES)
        X = np.random.default_rng(52).normal(size=(50, len(baseline)))
        module.print_candidate_redundancy(X, baseline, baseline)
        assert capsys.readouterr().out == ""

    def test_streaming_extractor_appends_candidates_in_order(self):
        module = _load_train_module()
        rng = np.random.default_rng(41)
        requested = list(module.TRAINING_FEATURES) + ['chan_coh_gap_q20']
        baseline = module.StreamingFeatureExtractor(list(module.TRAINING_FEATURES))
        extended = module.StreamingFeatureExtractor(requested)
        stream = _still_stream(module.SEG_WINDOW_SIZE + 20, rng)
        base_vector = extended_vector = None
        for packet in stream:
            payload = packet.tolist()
            base_vector = baseline.process_packet(payload) or base_vector
            extended_vector = extended.process_packet(payload) or extended_vector
        assert base_vector is not None and extended_vector is not None
        assert len(extended_vector) == len(base_vector) + 1
        # The production columns keep their identity and their position.
        assert extended_vector[:len(base_vector)] == pytest.approx(base_vector)
        assert np.isfinite(extended_vector[-1])
        assert -1.0 <= extended_vector[-1] <= 1.0
