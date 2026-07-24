# Collection Priorities

This list is derived from the current `dataset_info.json` role split and
`DATASET_QUALITY_CHECK.md` diagnostics. It does not mirror an external plan.

## Priority 1: Reserved normal-link coverage outside bedroom

Collect one reserved `static_presence` / `motion` pair for each non-ESP32 chip
in a non-bedroom environment:

- C3: `living_room` or `hobby_room`
- C5: `living_room` or `hobby_room`
- C6: `living_room` or `hobby_room`
- S3: `living_room` or `hobby_room`

Why:

- Current reserved non-low-RSSI validation is bedroom-only.
- Generalization outside bedroom is therefore under-measured.

## Priority 2: Low-RSSI empty-room controls

Collect one low-RSSI `empty` recording in `bedroom` for each non-ESP32 chip:

- C3
- C5
- C6
- S3

Why:

- Weak-link false positives currently mix link noise and possible human
  micro-motion.
- There are no explicit low-RSSI empty-room controls to separate those causes.

## Priority 3: Second low-RSSI reserved checkpoint for S3

Collect one additional low-RSSI reserved pair in `bedroom` for:

- S3

Why:

- S3 remains the least comfortable non-ESP32 weak-link case in the current
  stress report.
- One reserved weak-link checkpoint is still too brittle for strong claims.

## Priority 4: Optional second low-RSSI reserved checkpoint for C3

Collect one additional low-RSSI reserved `bedroom` pair for C3 when a
less-extreme candidate is available.

Why:

- C3 stress behavior looks materially better after the current role cleanup and
  sample pruning.
- A second C3 checkpoint is still useful, but it is no longer as urgent as S3.

## Priority 5: Replacement-quality S3 low-RSSI training sample

Collect one additional S3 low-RSSI `bedroom` pair intended for `train`.

Why:

- The current S3 low-RSSI train candidate is extremely pathological in the
  Classic diagnostics.
- A second sample would let us keep weak-link training coverage without relying
  on a single outlier-like pair.

## Priority 6: Nice-to-have low-RSSI living-room diversity

Collect one low-RSSI pair in `living_room` for any two of:

- C3
- C5
- C6
- S3

Why:

- Current real weak-link coverage is concentrated in `bedroom`.
- A second environment would show whether the weak-link failure mode is
  environment-specific or link-class-specific.
