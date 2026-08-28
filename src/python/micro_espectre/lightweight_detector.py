# SPDX-License-Identifier: GPL-3.0-only
# Commercial licensing available under separate agreement; see LICENSING.md.
"""MicroPython facade over the ESPectre core Lightweight detector."""

try:
    from array import array
except ImportError:
    array = None

try:
    import espectre_native_features as _native_features
except ImportError:
    _native_features = None

try:
    from src.config import DEFAULT_SUBCARRIERS
    from src.detector_interface import IDetector, MotionState
except ImportError:
    from config import DEFAULT_SUBCARRIERS
    from detector_interface import IDetector, MotionState


class LightweightDetector(IDetector):
    """Expose the native Lightweight detector through the device interface."""

    ALGORITHM = "lightweight"
    STARTUP_GATE = True
    STARTUP_THRESHOLD_FACTOR = 1.0
    BASE_THRESHOLD = 0.6621854538596202

    def __init__(self, window_size=100, threshold=BASE_THRESHOLD,
                 enable_lowpass=False, lowpass_cutoff=11.0,
                 enable_hampel=True, hampel_window=7, hampel_threshold=5.0,
                 autocorr_lag=1):
        if (
            array is None
            or _native_features is None
            or not hasattr(_native_features, "Detector")
        ):
            raise RuntimeError(
                "Micro-ESPectre requires a compatible espectre core module"
            )
        self._window_size = int(window_size)
        self._selected_subcarriers = tuple(DEFAULT_SUBCARRIERS)
        self._native_detector_output = array("f", (0.0,) * 6)
        self._native_detector_state = _native_features.Detector(
            "lightweight",
            window_size=self._window_size,
            threshold=threshold,
            lag=max(1, int(autocorr_lag)),
            enable_hampel=enable_hampel,
            hampel_window=hampel_window,
            hampel_threshold=hampel_threshold,
            enable_lowpass=enable_lowpass,
            lowpass_cutoff=lowpass_cutoff,
            subcarriers=self._selected_subcarriers,
        )
        # Retain these attributes for diagnostics and compatibility with the
        # previous mixed Python/native facade. The hot path is always native.
        self._context = None
        self._aggregated_context = None
        self._state = MotionState.IDLE
        self._threshold = float(threshold)
        self._current_probability = 0.0

    def process_packet(self, csi_data, selected_subcarriers=None, rssi_dbm=None,
                       timestamp_us=None):
        """Forward one normalized CSI packet to the native core detector."""
        del rssi_dbm
        selected = (
            self._selected_subcarriers
            if selected_subcarriers is None
            else tuple(selected_subcarriers)
        )
        if selected != self._selected_subcarriers:
            self._native_detector_state.set_subcarriers(selected)
            self._selected_subcarriers = selected
        self._native_detector_state.process(
            csi_data,
            0 if timestamp_us is None else timestamp_us,
        )

    def advance_missing_slots(self, count):
        self._native_detector_state.advance_missing(max(0, int(count)))

    def set_minimum_valid_samples(self, count):
        resolved = max(1, min(int(count), self._window_size))
        self._native_detector_state.set_minimum_valid(resolved)

    def update_state(self):
        self._native_detector_state.update(self._native_detector_output)
        output = self._native_detector_output
        self._state = MotionState.MOTION if output[0] >= 0.5 else MotionState.IDLE
        self._current_probability = output[1]
        self._threshold = output[2]
        return {
            "state": self._state,
            "motion_metric": self._current_probability,
            "probability": self._current_probability,
            "turb_autocorr": output[3],
            "turb_iqr_over_mean_aggr": output[4],
            "threshold": self._threshold,
        }

    def set_adaptive_threshold(self, _shared_threshold):
        self._native_detector_state.calibration_complete()
        self._native_detector_state.apply_adaptive_threshold()
        self._threshold = self._native_detector_state.get_threshold()

    def on_startup_calibration_begin(self):
        self._native_detector_state.calibration_begin()

    def set_threshold(self, threshold):
        value = float(threshold)
        if value < 0.0 or value > 1.0:
            return False
        try:
            self._native_detector_state.set_threshold(value)
        except ValueError:
            return False
        self._threshold = value
        return True

    def get_threshold(self):
        self._threshold = self._native_detector_state.get_threshold()
        return self._threshold

    def get_motion_metric(self):
        return self._native_detector_state.get_metric()

    def get_state(self):
        return self._state

    def is_ready(self):
        return self._native_detector_state.is_ready()

    def reset(self):
        self._native_detector_state.reset()
        self._state = MotionState.IDLE
        self._current_probability = 0.0

    def get_name(self):
        return "Lightweight"

    def get_backend(self):
        return "espectre_core"

    def get_window_size(self):
        return self._window_size

    @property
    def total_packets(self):
        return self._native_detector_state.get_total_packets()

    def close(self):
        native_detector = getattr(self, "_native_detector_state", None)
        if native_detector is not None:
            native_detector.deinit()
            self._native_detector_state = None

    def __del__(self):
        self.close()
