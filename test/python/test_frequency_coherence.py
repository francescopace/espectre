"""Frequency-coherence parity between the runtime path and the plain formula.

The MicroPython and host implementations walk the two contiguous live-band
halves and share one squared-magnitude cache across offsets. The reference here
spells the definition out pair by pair from the bin table, so a regression in
the pair set, the pair order, or the normalization shows up as a value
mismatch rather than a silent feature drift.
"""

import math
import random

import numpy as np
import pytest

from ml_feature_trackers import (
    FREQUENCY_COHERENCE_OFFSETS,
    HT20_LIVE_BINS,
    HT20_LIVE_WIDTH,
    ChannelShapeTracker,
    complex_profile,
    frequency_coherence,
    frequency_coherences,
    new_frequency_coherence_squares,
)
from tools.lib import host_feature_trackers as host

# Bit-level agreement is not expected: the optimized paths read |h|^2 as
# `real * real + imag * imag` while the reference squares `abs()`, and NumPy
# reassociates its sums. Both land far below this bound, which is still many
# orders tighter than any real change to the pair set or the normalization.
RELATIVE_TOLERANCE = 1e-12

SUPPORTED_OFFSETS = (4, 12)
REFERENCE_OFFSETS = (2, 4, 12)
EXPECTED_PAIR_COUNTS = {4: 48, 12: 32}
UNSUPPORTED_OFFSETS = (-4, -1, 0, 1, 3, 5, 11, 13, 28, 56)


def reference_pairs(offset):
    """Pairs the original full bin-table scan produced, in its original order."""
    return tuple(
        (left, right)
        for left, left_bin in enumerate(HT20_LIVE_BINS)
        for right, right_bin in enumerate(HT20_LIVE_BINS)
        if right_bin - left_bin == offset and not (left_bin < 32 < right_bin)
    )


def reference_frequency_coherence(profile, offset):
    """The definition, written straight from the pair table."""
    if int(offset) not in REFERENCE_OFFSETS:
        return 0.0
    numerator = 0j
    left_norm = 0.0
    right_norm = 0.0
    for left_index, right_index in reference_pairs(int(offset)):
        left = complex(profile[left_index])
        right = complex(profile[right_index])
        numerator += left.conjugate() * right
        left_norm += abs(left) * abs(left)
        right_norm += abs(right) * abs(right)
    denominator = math.sqrt(left_norm) * math.sqrt(right_norm)
    if denominator <= 0.0:
        return 0.0
    return abs(numerator) / denominator


def deterministic_profiles():
    """Named profiles that exercise flat, ramped, and sign-flipping channels."""
    flat = [complex(7.0, -3.0)] * HT20_LIVE_WIDTH
    ramp = [complex(i - 28.0, 0.5 * i) for i in range(HT20_LIVE_WIDTH)]
    alternating = [
        complex(9.0, 2.0) if i % 2 == 0 else complex(-9.0, -2.0)
        for i in range(HT20_LIVE_WIDTH)
    ]
    # One live half silent: the halves must stay independently normalized.
    half_silent = [
        complex(0.0, 0.0) if i < HT20_LIVE_WIDTH // 2 else complex(5.0, 4.0)
        for i in range(HT20_LIVE_WIDTH)
    ]
    single_tone = [complex(0.0, 0.0)] * HT20_LIVE_WIDTH
    single_tone[10] = complex(42.0, -17.0)
    return {
        "flat": flat,
        "ramp": ramp,
        "alternating": alternating,
        "half_silent": half_silent,
        "single_tone": single_tone,
    }


def random_profiles(count=32, seed=20260805):
    """Fixed-seed complex profiles in the int8 range CSI actually delivers."""
    rng = random.Random(seed)
    return [
        [
            complex(rng.uniform(-128.0, 127.0), rng.uniform(-128.0, 127.0))
            for _ in range(HT20_LIVE_WIDTH)
        ]
        for _ in range(count)
    ]


def test_half_walk_reproduces_the_reference_pair_table() -> None:
    """The optimization rests on right == left + offset inside a half."""
    for offset in SUPPORTED_OFFSETS:
        walked = tuple(
            (left, left + offset)
            for start, stop in _micro_spans()
            for left in range(start, stop - offset)
        )
        assert walked == reference_pairs(offset)
        assert len(walked) == EXPECTED_PAIR_COUNTS[offset]


def _micro_spans():
    from ml_feature_trackers import _FREQUENCY_COHERENCE_SPANS

    return _FREQUENCY_COHERENCE_SPANS


def test_offsets_match_the_documented_set() -> None:
    assert FREQUENCY_COHERENCE_OFFSETS == SUPPORTED_OFFSETS
    assert host.FREQUENCY_COHERENCE_OFFSETS == (2, 4, 12)


def test_research_offset_two_coherence_is_host_only() -> None:
    profile = random_profiles(count=1)[0]
    expected = reference_frequency_coherence(profile, 2)

    assert frequency_coherence(profile, 2) == 0.0
    assert host.frequency_coherence(
        np.asarray(profile, dtype=np.complex128), 2
    ) == pytest.approx(expected, rel=RELATIVE_TOLERANCE)


@pytest.mark.parametrize("offset", SUPPORTED_OFFSETS)
def test_null_csi_gives_a_zero_profile_and_zero_coherence(offset) -> None:
    """A missing or short payload yields the zero profile, hence no coherence."""
    for csi_data in (None, [], [0] * 16):
        profile = complex_profile(csi_data)
        assert frequency_coherence(profile, offset) == 0.0
        assert host.frequency_coherence(
            np.asarray(profile, dtype=np.complex128), offset
        ) == 0.0


@pytest.mark.parametrize("offset", SUPPORTED_OFFSETS)
def test_zero_profile_is_guarded_rather_than_divided(offset) -> None:
    zeros = [0j] * HT20_LIVE_WIDTH
    assert frequency_coherence(zeros, offset) == 0.0
    assert reference_frequency_coherence(zeros, offset) == 0.0
    assert host.frequency_coherence(np.zeros(HT20_LIVE_WIDTH, dtype=np.complex128), offset) == 0.0


@pytest.mark.parametrize("offset", SUPPORTED_OFFSETS)
@pytest.mark.parametrize("name", sorted(deterministic_profiles()))
def test_deterministic_profiles_match_the_reference(name, offset) -> None:
    profile = deterministic_profiles()[name]
    expected = reference_frequency_coherence(profile, offset)

    assert frequency_coherence(profile, offset) == pytest.approx(
        expected, rel=RELATIVE_TOLERANCE, abs=RELATIVE_TOLERANCE
    )
    assert host.frequency_coherence(
        np.asarray(profile, dtype=np.complex128), offset
    ) == pytest.approx(expected, rel=RELATIVE_TOLERANCE, abs=RELATIVE_TOLERANCE)


@pytest.mark.parametrize("offset", SUPPORTED_OFFSETS)
def test_random_profiles_match_the_reference(offset) -> None:
    for profile in random_profiles():
        expected = reference_frequency_coherence(profile, offset)
        # A real channel keeps this well away from the guard, so a plain
        # relative bound is the strict reading here.
        assert expected > 0.0
        assert frequency_coherence(profile, offset) == pytest.approx(
            expected, rel=RELATIVE_TOLERANCE
        )
        assert host.frequency_coherence(
            np.asarray(profile, dtype=np.complex128), offset
        ) == pytest.approx(expected, rel=RELATIVE_TOLERANCE)


@pytest.mark.parametrize("offset", UNSUPPORTED_OFFSETS)
def test_unsupported_offsets_return_zero(offset) -> None:
    profile = random_profiles(count=1)[0]
    assert frequency_coherence(profile, offset) == 0.0
    assert host.frequency_coherence(np.asarray(profile, dtype=np.complex128), offset) == 0.0


def test_combined_output_equals_the_two_single_calls() -> None:
    """The tracker reads the combined form, so it must not drift from singles."""
    squares = new_frequency_coherence_squares()
    out = [0.0] * len(FREQUENCY_COHERENCE_OFFSETS)
    profiles = list(deterministic_profiles().values()) + random_profiles()

    for profile in profiles:
        combined = frequency_coherences(profile, out, squares)
        singles = [frequency_coherence(profile, o) for o in FREQUENCY_COHERENCE_OFFSETS]
        # Same buffers, same arithmetic: this one is exact, not approximate.
        assert list(combined) == singles


def test_combined_allocates_its_own_buffers_when_omitted() -> None:
    profile = random_profiles(count=1)[0]
    combined = frequency_coherences(profile)

    assert len(combined) == len(FREQUENCY_COHERENCE_OFFSETS)
    assert combined == frequency_coherences(
        profile, [0.0] * len(FREQUENCY_COHERENCE_OFFSETS), new_frequency_coherence_squares()
    )


def test_reused_squares_buffer_does_not_leak_between_packets() -> None:
    """The tracker keeps one buffer for its lifetime; packets must not blend."""
    squares = new_frequency_coherence_squares()
    out = [0.0] * len(FREQUENCY_COHERENCE_OFFSETS)
    first, second = random_profiles(count=2)

    frequency_coherences(first, out, squares)
    reused = list(frequency_coherences(second, out, squares))
    fresh = list(frequency_coherences(second))

    assert reused == fresh


def test_host_rejects_a_profile_that_is_not_the_live_band() -> None:
    assert host.frequency_coherence(np.zeros(8, dtype=np.complex128), 4) == 0.0
    assert host.frequency_coherence(
        np.zeros((2, HT20_LIVE_WIDTH), dtype=np.complex128), 4
    ) == 0.0


def test_tracker_curve_features_follow_the_reference_coherences() -> None:
    """End-to-end: the pushed features are the reference ones, not just parity."""
    rng = random.Random(4242)
    packets = 24
    # A window wider than the run keeps every packet in the ring, so the
    # expected statistics below cover exactly the packets that were fed.
    tracker = ChannelShapeTracker(window_size=packets * 2, lag=3)
    expected_curve = []

    for _ in range(packets):
        csi_data = [rng.randrange(0, 256) for _ in range(128)]
        profile = complex_profile(csi_data)
        short = reference_frequency_coherence(profile, 4)
        long = reference_frequency_coherence(profile, 12)
        total = short + long
        expected_curve.append((short - long) / total if total > 0.0 else 0.0)
        tracker.process_packet(csi_data)

    mean_curve = sum(expected_curve) / len(expected_curve)
    curve_std = math.sqrt(
        max(0.0, sum(v * v for v in expected_curve) / len(expected_curve) - mean_curve**2)
    )
    assert tracker.frequency_coherence_curve_std() == pytest.approx(
        curve_std, rel=1e-9, abs=1e-12
    )


def test_host_three_offset_curve_candidates_follow_their_definitions() -> None:
    rng = random.Random(9876)
    packets = 24
    tracker = host.ChannelShapeTracker(
        window_size=packets * 2,
        lag=3,
        feature_names=(
            "chan_freq_coh_curve_iqr",
            "chan_freq_coh_curve_2_4_std",
            "chan_freq_coh_curve_4_12_std",
            "chan_freq_coh_decay_std",
            "chan_freq_coh_curvature_std",
        ),
    )
    contrasts = []
    decays = []
    curvatures = []
    short_mid_curves = []
    mid_long_curves = []

    for _ in range(packets):
        csi_data = [rng.randrange(-128, 128) for _ in range(128)]
        profile = host.complex_profile(csi_data)
        short = host.frequency_coherence(profile, 2)
        mid = host.frequency_coherence(profile, 4)
        long = host.frequency_coherence(profile, 12)
        endpoint_sum = mid + long
        total = short + mid + long
        contrasts.append(
            (mid - long) / endpoint_sum if endpoint_sum > 0.0 else 0.0
        )
        decays.append(
            (2.0 * short + mid - 3.0 * long) / (3.0 * total)
            if total > 0.0 else 0.0
        )
        curvatures.append(
            (mid - 0.8 * short - 0.2 * long) / total
            if total > 0.0 else 0.0
        )
        short_mid_sum = short + mid
        mid_long_sum = mid + long
        short_mid_curves.append(
            (short - mid) / short_mid_sum if short_mid_sum > 0.0 else 0.0
        )
        mid_long_curves.append(
            (mid - long) / mid_long_sum if mid_long_sum > 0.0 else 0.0
        )
        tracker.process_packet(csi_data)

    q25, q75 = np.quantile(contrasts, [0.25, 0.75])
    assert tracker.frequency_coherence_curve_iqr() == pytest.approx(q75 - q25)
    assert tracker.frequency_coherence_candidate_std(
        "chan_freq_coh_decay_std"
    ) == pytest.approx(float(np.std(decays)))
    assert tracker.frequency_coherence_candidate_std(
        "chan_freq_coh_curvature_std"
    ) == pytest.approx(float(np.std(curvatures)))
    assert tracker.frequency_coherence_candidate_std(
        "chan_freq_coh_curve_2_4_std"
    ) == pytest.approx(float(np.std(short_mid_curves)))
    assert tracker.frequency_coherence_candidate_std(
        "chan_freq_coh_curve_4_12_std"
    ) == pytest.approx(float(np.std(mid_long_curves)))


def test_curve_only_trackers_match_full_trackers_without_shape_history() -> None:
    rng = random.Random(1977)
    runtime_full = ChannelShapeTracker(window_size=32, lag=3)
    runtime_curve = ChannelShapeTracker(
        window_size=32,
        lag=3,
        track_shape=False,
    )
    host_full = host.ChannelShapeTracker(window_size=32, lag=3)
    host_curve = host.ChannelShapeTracker(
        window_size=32,
        lag=3,
        feature_names=("chan_freq_coh_curve_std",),
    )

    for _ in range(40):
        csi_data = [rng.randrange(-128, 128) for _ in range(128)]
        runtime_full.process_packet(csi_data)
        runtime_curve.process_packet(csi_data)
        host_full.process_packet(csi_data)
        host_curve.process_packet(csi_data)

    assert runtime_curve.frequency_coherence_curve_std() == pytest.approx(
        runtime_full.frequency_coherence_curve_std(), abs=1e-12
    )
    assert host_curve.frequency_coherence_curve_std() == pytest.approx(
        host_full.frequency_coherence_curve_std(), abs=1e-12
    )
    assert runtime_curve._ring == []
    assert runtime_curve._motion_energy_ring == []
    assert host_curve._ring == []
    assert host_curve._motion_energy_ring.size == 0
