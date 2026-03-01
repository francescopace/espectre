#!/usr/bin/env python3
"""
ML Gesture Detection - Streaming Test Script

Simulates a mixed CSI stream by concatenating random 2s chunks sampled from
data/<label>/ and evaluates runtime GestureDetector behavior.

Configuration:
  - mode: fixed to continuous (production-like live inference)
  - runtime subcarriers: fixed to GESTURE_SUBCARRIERS (movement-model aligned)
  - synthetic no_gesture: always included from baseline/movement sources

Author: Francesco Pace <francesco.pace@gmail.com>
License: GPLv3
"""

import argparse
import random

from csi_utils import DATA_DIR, load_npz_as_packets
from gesture_detector import GestureDetector
from gesture_detector import GESTURE_SUBCARRIERS

DEFAULT_PACKET_RATE = 100.0
DEFAULT_CHUNK_SECONDS = 2.0
DEFAULT_SEGMENTS = 80
DEFAULT_NO_GESTURE_SOURCES = ("baseline", "movement")


def _discover_available_labels(no_gesture_sources, only_labels):
    requested = set(only_labels or [])
    labels = []
    files_by_label = {}

    for item in sorted(DATA_DIR.iterdir()):
        if not item.is_dir():
            continue
        name = item.name
        if name in ("baseline", "movement"):
            continue
        if requested and name not in requested:
            continue
        npz_files = sorted(item.glob("*.npz"))
        if npz_files:
            labels.append(name)
            files_by_label[name] = npz_files

    no_gesture_files = []
    for src in no_gesture_sources:
        src_dir = DATA_DIR / src
        if src_dir.exists() and src_dir.is_dir():
            no_gesture_files.extend(sorted(src_dir.glob("*.npz")))
    if no_gesture_files:
        labels.append("no_gesture")
        files_by_label["no_gesture"] = no_gesture_files

    return labels, files_by_label


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


def run_stream_test(
    seed: int,
    chunk_seconds: float,
    segments: int,
    labels_csv: str,
):
    if chunk_seconds <= 0:
        raise ValueError("--chunk-seconds must be > 0")
    if segments < 1:
        raise ValueError("--segments must be >= 1")

    packet_rate = DEFAULT_PACKET_RATE
    chunk_packets = max(10, int(round(chunk_seconds * packet_rate)))
    rng = random.Random(seed)
    only_labels = [x.strip() for x in labels_csv.split(",") if x.strip()]

    print("\n" + "=" * 66)
    print("  GESTURE STREAMING TEST")
    print("=" * 66 + "\n")
    print(f"Seed: {seed}")
    print("Mode: continuous")
    print(f"Chunk size: {chunk_seconds:.2f}s ({chunk_packets} packets)")
    print(f"Target segments: {segments}")
    print(f"Data dir: {DATA_DIR}")
    print(f"Runtime subcarriers: {GESTURE_SUBCARRIERS} (fixed)")

    labels, files_by_label = _discover_available_labels(
        no_gesture_sources=list(DEFAULT_NO_GESTURE_SOURCES),
        only_labels=only_labels,
    )
    if not labels:
        print("\nError: no labels with NPZ files found.")
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
        return 1
    print(f"\nEffective classes: {effective_labels}")

    stream_segments = []
    attempts = 0
    max_attempts = segments * 20
    while len(stream_segments) < segments and attempts < max_attempts:
        attempts += 1
        gt_label = rng.choice(effective_labels)
        file_map = cache_by_label[gt_label]
        src_file = rng.choice(list(file_map.keys()))
        chunk_packets_list = _sample_random_chunk(file_map[src_file], chunk_packets, rng)
        if not chunk_packets_list:
            continue
        stream_segments.append((gt_label, src_file, chunk_packets_list))

    if not stream_segments:
        print("\nError: unable to sample stream chunks with requested configuration.")
        return 1
    if len(stream_segments) < segments:
        print(f"\nWarning: sampled {len(stream_segments)}/{segments} segments "
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
    if "no_gesture" in per_class_total:
        ng_recall = (per_class_correct["no_gesture"] / per_class_total["no_gesture"] * 100.0) if per_class_total["no_gesture"] else 0.0
        gesture_labels = [lbl for lbl in effective_labels if lbl != "no_gesture"]
        gesture_ok = True
        for lbl in gesture_labels:
            n = per_class_total[lbl]
            val = (per_class_correct[lbl] / n * 100.0) if n else 0.0
            if val < 65.0:
                gesture_ok = False
                break
        constraints_ok = (ng_recall >= 50.0) and gesture_ok
        print(f"\nConstraint check: no_gesture>=50%, gestures>=65% -> {'PASS' if constraints_ok else 'FAIL'}")

    print("\n" + _format_confusion_matrix(effective_labels, classes_with_unknown, confusion))
    print("\nDone.\n")
    return 0


def main():
    parser = argparse.ArgumentParser(
        description="Test gesture model on synthetic random CSI stream",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducible stream generation")
    parser.add_argument("--chunk-seconds", type=float, default=DEFAULT_CHUNK_SECONDS,
                        help="Duration of each sampled chunk in seconds")
    parser.add_argument("--segments", type=int, default=DEFAULT_SEGMENTS,
                        help="Number of random chunks in the synthetic stream")
    parser.add_argument("--labels", type=str, default="",
                        help="Comma-separated gesture labels to include (empty = auto-discover all)")
    args = parser.parse_args()

    return run_stream_test(
        seed=args.seed,
        chunk_seconds=args.chunk_seconds,
        segments=args.segments,
        labels_csv=args.labels,
    )


if __name__ == "__main__":
    raise SystemExit(main())

