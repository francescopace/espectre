#!/usr/bin/env python3
"""
ML Gesture Detection - Streaming Test Script

Simulates a mixed CSI stream by concatenating model-aligned chunks sampled
from data/<label>/ and evaluates runtime GestureDetector behavior.

Configuration:
  - mode: fixed to continuous (production-like live inference)
  - runtime subcarriers: fixed to GESTURE_SUBCARRIERS (movement-model aligned)
  - no_gesture: loaded only from data/no_gesture/

Author: Francesco Pace <francesco.pace@gmail.com>
License: GPLv3
"""

import argparse
import random
import signal
from dataclasses import dataclass

from csi_utils import (
    DATA_DIR,
    DEFAULT_PORT,
    CSIReceiver,
    load_npz_as_packets,
    TARGET_NO_GESTURE_RECALL,
    TARGET_GESTURE_RECALL,
)
from gesture_detector import GestureDetector
from gesture_detector import GESTURE_WINDOW_LEN
from gesture_detector import GESTURE_SUBCARRIERS
import gesture_detector as gd_mod

DEFAULT_PACKET_RATE = 100.0
FIXED_GESTURE_LABELS = ("wave", "circle_cw")
NO_GESTURE_LABEL = "no_gesture"


def _discover_available_labels():
    labels = []
    files_by_label = {}

    for item in sorted(DATA_DIR.iterdir()):
        if not item.is_dir():
            continue
        name = item.name
        # no_gesture is loaded separately below.
        if name in ("baseline", "movement", "no_gesture"):
            continue
        # Keep this benchmark fixed to the project gesture subset.
        if name not in FIXED_GESTURE_LABELS:
            continue
        npz_files = sorted(item.glob("*.npz"))
        if npz_files:
            labels.append(name)
            files_by_label[name] = npz_files

    no_gesture_dir = DATA_DIR / NO_GESTURE_LABEL
    no_gesture_files = []
    if no_gesture_dir.exists() and no_gesture_dir.is_dir():
        no_gesture_files = sorted(no_gesture_dir.glob("*.npz"))
    if no_gesture_files:
        labels.append(NO_GESTURE_LABEL)
        files_by_label[NO_GESTURE_LABEL] = no_gesture_files

    # Defensive dedup (should be redundant with the skip above).
    seen = set()
    dedup_labels = []
    for lbl in labels:
        if lbl not in seen:
            seen.add(lbl)
            dedup_labels.append(lbl)

    return dedup_labels, files_by_label


def _load_file_cache(npz_files):
    cache = {}
    for fp in npz_files:
        packets = load_npz_as_packets(fp)
        if packets:
            cache[fp] = packets
    return cache


def _sample_random_chunk(file_packets, chunk_packets, rng):
    if len(file_packets) < chunk_packets:
        return []
    start_max = len(file_packets) - chunk_packets
    start = rng.randint(0, start_max) if start_max > 0 else 0
    return file_packets[start:start + chunk_packets]


def _format_confusion_matrix(gt_classes, pred_classes, matrix):
    col_w = max(12, max(len(c) for c in pred_classes) + 2)
    lines = ["Confusion Matrix (rows=ground truth, cols=predicted):"]
    lines.append(" " * col_w + "".join(h.rjust(col_w) for h in pred_classes))
    for gt in gt_classes:
        row = gt.ljust(col_w)
        for pr in pred_classes:
            row += str(matrix[gt][pr]).rjust(col_w)
        lines.append(row)
    return "\n".join(lines)


def _evaluate_predictions(effective_labels, predictions):
    from sklearn.metrics import f1_score, balanced_accuracy_score

    classes_with_unknown = list(effective_labels)
    if "unknown" not in classes_with_unknown:
        classes_with_unknown.append("unknown")

    confusion = {gt: {pr: 0 for pr in classes_with_unknown} for gt in effective_labels}
    per_class_total = {lbl: 0 for lbl in effective_labels}
    per_class_correct = {lbl: 0 for lbl in effective_labels}

    for gt_label, pred_label in predictions:
        if pred_label not in classes_with_unknown:
            pred_label = "unknown"
        per_class_total[gt_label] += 1
        if pred_label == gt_label:
            per_class_correct[gt_label] += 1
        confusion[gt_label][pred_label] += 1

    total = len(predictions)
    correct = sum(1 for gt, pr in predictions if gt == pr)
    accuracy = (correct / total * 100.0) if total else 0.0

    # Balanced metrics on known classes only.
    label_to_id = {lbl: i for i, lbl in enumerate(effective_labels)}
    y_true = []
    y_pred = []
    for gt_label, pred_label in predictions:
        if pred_label not in label_to_id:
            # Count unknowns as no_gesture when available; this mirrors runtime reject intent.
            pred_label = "no_gesture" if "no_gesture" in label_to_id else gt_label
        y_true.append(label_to_id[gt_label])
        y_pred.append(label_to_id[pred_label])

    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0) * 100.0 if y_true else 0.0
    balanced_acc = balanced_accuracy_score(y_true, y_pred) * 100.0 if y_true else 0.0

    return (
        classes_with_unknown, confusion, per_class_total, per_class_correct,
        total, correct, accuracy, macro_f1, balanced_acc
    )


def _choose_segment_prediction(seg_preds, final_label, allowed_labels):
    """Production-aligned segment decision: use latest live emit."""
    if seg_preds:
        label = seg_preds[-1]
        return label if label in allowed_labels else "unknown"
    if final_label is not None:
        return final_label if final_label in allowed_labels else "unknown"
    return "unknown"


@dataclass
class LiveRuntimeConfig:
    udp_port: int
    print_every_n: int
    log_no_gesture: bool


class LiveGestureRunner:
    """Live gesture inference over UDP CSI stream with runtime pipeline."""

    def __init__(self, cfg: LiveRuntimeConfig):
        self.cfg = cfg
        self.running = True
        self.packet_count = 0
        self.last_stats_packet = 0
        self.detected_count = 0
        self.detector = None
        self.last_label = None
        self.first_meta_printed = False

        self.receiver = CSIReceiver(port=cfg.udp_port, buffer_size=4000)
        self.receiver.add_callback(self._on_packet)
        signal.signal(signal.SIGINT, self._on_sigint)

    def _on_sigint(self, _signum, _frame):
        self.running = False
        self.receiver.stop()

    def _init_detector(self, pkt):
        use_cv_norm = not bool(getattr(pkt, "gain_locked", True))
        self.detector = GestureDetector(use_cv_normalization=use_cv_norm)
        self.detector.start_detection()

        if not self.first_meta_printed:
            print(
                f"[stream] chip={pkt.chip} subcarriers={pkt.num_subcarriers} "
                f"gain_locked={bool(pkt.gain_locked)} use_cv_norm={use_cv_norm}"
            )
            print(
                f"conf>={gd_mod.GESTURE_MIN_CONFIDENCE:.2f} "
                f"margin>={gd_mod.GESTURE_MIN_MARGIN:.2f} "
                f"min_cons={gd_mod.GESTURE_MIN_CONSECUTIVE}"
            )
            self.first_meta_printed = True

    def _on_packet(self, pkt):
        if not self.running:
            return
        self.packet_count += 1

        if self.detector is None:
            self._init_detector(pkt)

        self.detector.process_packet(pkt.iq_raw, GESTURE_SUBCARRIERS)
        candidate = self.detector.consume_live_prediction()

        should_log = False
        if candidate is not None:
            if candidate != self.last_label:
                should_log = True
            elif candidate != "no_gesture":
                should_log = True
            self.last_label = candidate

        if candidate == "no_gesture" and not self.cfg.log_no_gesture:
            should_log = False

        if should_log:
            self.detected_count += 1
            print(f"[live] pkt={self.packet_count:6d} pps~{self.receiver.pps:4d} label={candidate}")

        if (self.packet_count - self.last_stats_packet) >= self.cfg.print_every_n:
            self.last_stats_packet = self.packet_count
            print(
                f"[stats] pkt={self.packet_count} pps={self.receiver.pps} "
                f"drop={self.receiver.dropped_count} detections={self.detected_count}"
            )

    def run(self):
        if not gd_mod._weights_available:
            print("Error: gesture_weights.py not found. Train first with 12_train_gesture_model.py.")
            return 1

        print("\n" + "=" * 72)
        print("  LIVE GESTURE INFERENCE")
        print("=" * 72)
        print(f"UDP port:        {self.cfg.udp_port}")
        print(f"Classes:         {getattr(gd_mod._gw, 'GESTURE_CLASS_LABELS', [])}")
        print(f"Subcarriers:     {GESTURE_SUBCARRIERS}")
        print("")
        print("Start ESP32 stream in another terminal:")
        print("  ./me stream --ip <PC_IP>")
        print("")
        print("Press Ctrl+C to stop.")
        print("-" * 72)

        while self.running:
            self.receiver.run(timeout=1.0, quiet=True)

        print("\nDone.\n")
        return 0


def run_live_test(
    udp_port: int,
    print_every_n: int,
    log_no_gesture: bool,
):
    cfg = LiveRuntimeConfig(
        udp_port=udp_port,
        print_every_n=max(1, int(print_every_n)),
        log_no_gesture=bool(log_no_gesture),
    )
    return LiveGestureRunner(cfg).run()


def run_stream_test(seed: int | None, return_metrics: bool = False):
    if seed is None:
        seed = random.SystemRandom().randrange(0, 2**32)

    packet_rate = DEFAULT_PACKET_RATE
    chunk_packets = max(10, int(GESTURE_WINDOW_LEN))
    chunk_seconds = chunk_packets / packet_rate
    rng = random.Random(seed)

    print("\n" + "=" * 66)
    print("  GESTURE STREAMING TEST")
    print("=" * 66 + "\n")
    print(f"Seed: {seed}")
    print("Mode: continuous")
    print(f"Chunk size: {chunk_seconds:.2f}s ({chunk_packets} packets)")
    print("Target segments: full coverage (1 chunk per readable file)")
    print(f"Data dir: {DATA_DIR}")
    print(f"Runtime subcarriers: {GESTURE_SUBCARRIERS} (fixed)")

    labels, files_by_label = _discover_available_labels()
    if not labels:
        print("\nError: no labels with NPZ files found.")
        if return_metrics:
            return {"error": "no_labels"}
        return 1
    if NO_GESTURE_LABEL not in files_by_label:
        print(
            f"\nError: required dataset '{NO_GESTURE_LABEL}' not found or empty in "
            f"{(DATA_DIR / NO_GESTURE_LABEL)}."
        )
        if return_metrics:
            return {"error": "missing_no_gesture"}
        return 1

    print(f"\nDiscovered classes: {labels}")
    for lbl in labels:
        print(f"  {lbl}: {len(files_by_label[lbl])} files")

    cache_by_label = {}
    for lbl in labels:
        cache_by_label[lbl] = _load_file_cache(files_by_label[lbl])
        if not cache_by_label[lbl]:
            print(f"Warning: label '{lbl}' has no readable packet files, it will be skipped.")

    effective_labels = [lbl for lbl in labels if cache_by_label[lbl]]
    if len(effective_labels) < 2:
        print("\nError: at least two classes with readable data are required.")
        if return_metrics:
            return {"error": "too_few_effective_labels"}
        return 1
    print(f"\nEffective classes: {effective_labels}")

    stream_segments = []
    target_by_label = {lbl: len(cache_by_label[lbl]) for lbl in effective_labels}
    target_total = sum(target_by_label.values())
    print(f"Sampling mode: full-coverage (target segments by class: {target_by_label})")
    for gt_label in effective_labels:
        file_map = cache_by_label[gt_label]
        file_list = list(file_map.keys())
        rng.shuffle(file_list)
        for src_file in file_list:
            chunk_packets_list = _sample_random_chunk(file_map[src_file], chunk_packets, rng)
            if not chunk_packets_list:
                continue
            stream_segments.append((gt_label, src_file, chunk_packets_list))
    rng.shuffle(stream_segments)

    if not stream_segments:
        print("\nError: unable to sample stream chunks with requested configuration.")
        if return_metrics:
            return {"error": "no_stream_segments"}
        return 1
    if len(stream_segments) < target_total:
        print(f"\nWarning: sampled {len(stream_segments)}/{target_total} segments "
              f"(some files were shorter than chunk size).")

    allowed_labels = set(effective_labels)
    fixed_subcarriers = GESTURE_SUBCARRIERS
    detector = GestureDetector()
    predictions = []
    print("\nRunning continuous evaluation (production-like live inference)...")
    for idx, (gt_label, src_file, packets) in enumerate(stream_segments, start=1):
        detector.start_detection()
        seg_preds = []
        for pkt in packets:
            detector.process_packet(pkt["csi_data"], fixed_subcarriers)
            candidate = detector.consume_live_prediction()
            if candidate is not None:
                seg_preds.append(candidate)
        final_label = detector.finalize_detection()
        pred_label = _choose_segment_prediction(seg_preds, final_label, allowed_labels)
        predictions.append((gt_label, pred_label))
        if idx <= 12:
            emits = len(seg_preds)
            print(f"  #{idx:03d} gt={gt_label:<12} pred={pred_label:<12} emits={emits:<2d} file={src_file.name}")

    (classes_with_unknown, confusion, per_class_total, per_class_correct,
     total, correct, accuracy, macro_f1, balanced_acc) = _evaluate_predictions(effective_labels, predictions)

    print("\n" + "-" * 66)
    print(f"Segments evaluated: {total}")
    print(f"Overall accuracy: {accuracy:.1f}% ({correct}/{total})")
    print(f"Macro-F1 (3-class): {macro_f1:.1f}%")
    print(f"Balanced accuracy (3-class): {balanced_acc:.1f}%")
    print("\nPer-class accuracy:")
    for lbl in effective_labels:
        n = per_class_total[lbl]
        acc = (per_class_correct[lbl] / n * 100.0) if n else 0.0
        print(f"  {lbl:<12} {acc:>6.1f}% ({per_class_correct[lbl]}/{n})")
    default_threshold = TARGET_GESTURE_RECALL * 100.0
    no_gesture_threshold = TARGET_NO_GESTURE_RECALL * 100.0
    constraints_ok = True
    constraint_details = []
    for lbl in effective_labels:
        n = per_class_total[lbl]
        val = (per_class_correct[lbl] / n * 100.0) if n else 0.0
        threshold = no_gesture_threshold if lbl == NO_GESTURE_LABEL else default_threshold
        constraint_details.append(f"{lbl}>={threshold:.0f}%")
        if val < threshold:
            constraints_ok = False
    print(
        f"\nConstraint check ({', '.join(constraint_details)}): "
        f"{'PASS' if constraints_ok else 'FAIL'}"
    )

    print("\n" + _format_confusion_matrix(effective_labels, classes_with_unknown, confusion))
    print("\nDone.\n")
    if return_metrics:
        per_class_accuracy = {}
        for lbl in effective_labels:
            n = per_class_total[lbl]
            per_class_accuracy[lbl] = (per_class_correct[lbl] / n * 100.0) if n else 0.0
        return {
            "seed": int(seed),
            "overall_accuracy": float(accuracy),
            "macro_f1": float(macro_f1),
            "balanced_accuracy": float(balanced_acc),
            "constraint_pass": bool(constraints_ok),
            "per_class_accuracy": per_class_accuracy,
            "segments_evaluated": int(total),
        }
    return 0


def main():
    parser = argparse.ArgumentParser(
        description="Test gesture model on full-coverage CSI stream benchmark",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--seed", type=int, default=None,
                        help="Optional random seed (default: random each run)")
    parser.add_argument(
        "--live",
        action="store_true",
        help="Run live inference from UDP stream instead of offline dataset test",
    )
    parser.add_argument("--udp-port", type=int, default=DEFAULT_PORT,
                        help="UDP port to listen for live CSI stream")
    parser.add_argument("--print-every-n", type=int, default=200,
                        help="Print live periodic stats every N packets")
    parser.add_argument("--log-no-gesture", action="store_true",
                        help="In live mode, also print no_gesture predictions")
    args = parser.parse_args()

    if args.live:
        return run_live_test(
            udp_port=args.udp_port,
            print_every_n=args.print_every_n,
            log_no_gesture=args.log_no_gesture,
        )

    return run_stream_test(
        seed=args.seed,
    )


if __name__ == "__main__":
    raise SystemExit(main())

