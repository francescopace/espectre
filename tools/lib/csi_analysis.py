"""
Shared CSI analysis helpers for tool-side workflows.
"""

from __future__ import annotations

from typing import List, Tuple

from .bootstrap import setup_paths

setup_paths()

try:
    import config
except ImportError:
    import src.config as config

try:
    from detector_interface import MotionState
    from mvs_detector import MVSDetector as MVSDetectorNew
    from segmentation import SegmentationContext
except ImportError:  # pragma: no cover
    from src.detector_interface import MotionState
    from src.mvs_detector import MVSDetector as MVSDetectorNew
    from src.segmentation import SegmentationContext


def calculate_spatial_turbulence(csi_data, selected_subcarriers=None) -> float:
    """
    Calculate spatial turbulence from CSI data using the normalized AGC-active path.
    """
    band = config.DEFAULT_SUBCARRIERS if selected_subcarriers is None else selected_subcarriers
    turbulence, _ = SegmentationContext.compute_spatial_turbulence(csi_data, band)
    return turbulence


def calculate_variance_two_pass(values) -> float:
    """Calculate variance using the shared numerically stable implementation."""
    return SegmentationContext.compute_variance_two_pass(values)


class MVSDetector:
    """
    Thin compatibility adapter around the production ``mvs_detector.MVSDetector``.
    """

    def __init__(
        self,
        window_size: int,
        threshold: float,
        selected_subcarriers=None,
        track_data: bool = False,
        enable_hampel: bool = True,
        hampel_window: int = config.HAMPEL_WINDOW,
        hampel_threshold: float = config.HAMPEL_THRESHOLD,
        enable_lowpass: bool = False,
        lowpass_cutoff: float = 11.0,
    ):
        self.window_size = window_size
        self.threshold = threshold
        self.fixed_subcarriers = (
            config.DEFAULT_SUBCARRIERS if selected_subcarriers is None else list(selected_subcarriers)
        )
        self.track_data = track_data

        self._detector = MVSDetectorNew(
            window_size=window_size,
            threshold=threshold,
            enable_hampel=enable_hampel,
            hampel_window=hampel_window,
            hampel_threshold=hampel_threshold,
            enable_lowpass=enable_lowpass,
            lowpass_cutoff=lowpass_cutoff,
        )
        self._context = self._detector._context

        self.state = "IDLE"
        self.motion_packet_count = 0
        self.turbulence_buffer: List[float] = []

        if track_data:
            self.moving_var_history: List[float] = []
            self.state_history: List[str] = []

    def process_packet(self, packet_or_csi) -> None:
        """Process a single CSI packet or raw CSI array."""
        csi_data = packet_or_csi["csi_data"] if isinstance(packet_or_csi, dict) else packet_or_csi
        self._detector.process_packet(csi_data, self.fixed_subcarriers)
        state = self._detector.update_state()
        raw_state = state.get("state", MotionState.IDLE)
        new_state = "MOTION" if raw_state == MotionState.MOTION or str(raw_state).upper() == "MOTION" else "IDLE"

        if self.track_data:
            self.moving_var_history.append(float(state.get("moving_variance", 0.0)))
            self.state_history.append(new_state)

        self.state = new_state
        if self.state == "MOTION":
            self.motion_packet_count += 1

    def reset(self) -> None:
        """Reset detector state."""
        self._detector.reset()
        self._context = self._detector._context
        self.state = "IDLE"
        self.motion_packet_count = 0
        self.turbulence_buffer = []
        if self.track_data:
            self.moving_var_history = []
            self.state_history = []

    def get_motion_count(self) -> int:
        """Get the number of packets detected as motion."""
        return self.motion_packet_count


def test_mvs_configuration(
    static_presence_packets,
    motion_packets,
    threshold,
    window_size,
) -> Tuple[int, int, float]:
    """Test one MVS configuration and return ``(fp, tp, score)``."""
    num_static_presence = len(static_presence_packets)
    num_motion = len(motion_packets)

    detector = MVSDetector(window_size, threshold)
    for pkt in static_presence_packets:
        detector.process_packet(pkt)
    fp = detector.get_motion_count()

    detector.motion_packet_count = 0
    for pkt in motion_packets:
        detector.process_packet(pkt)
    tp = detector.get_motion_count()

    fn = max(0, num_motion - tp)
    recall = (tp / num_motion * 100.0) if num_motion > 0 else 0.0
    precision = (tp / (tp + fp) * 100.0) if (tp + fp) > 0 else 0.0
    fp_rate = (fp / num_static_presence * 100.0) if num_static_presence > 0 else 100.0
    f1_score = 0.0
    if (precision + recall) > 0.0:
        f1_score = 2.0 * precision * recall / (precision + recall)

    recall_target = 95.0
    fp_target = 10.0
    fn_rate = (fn / num_motion * 100.0) if num_motion > 0 else 100.0

    if recall >= recall_target and fp_rate <= fp_target:
        score = 1_000_000.0 + f1_score * 100.0 - fp_rate
    elif recall >= recall_target:
        score = 100_000.0 - (fp_rate - fp_target) * 1_000.0 + f1_score * 10.0
    else:
        score = (
            -1_000_000.0
            - (recall_target - recall) * 2_000.0
            - fn_rate * 200.0
            - fp_rate * 20.0
            + precision
        )

    return fp, tp, score

