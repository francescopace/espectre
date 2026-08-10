# Mix Complementary Training Augmentation Seeds

- Status: Accepted
- Date: 2026-08-11

## Context

The promoted `base,drift,burst-loss` packet augmentation previously used only seed `20260807`. Seed `20260807` was comparatively severe on false-positive tails under drift, while seed `20260808` was comparatively severe on worst weak-link recall. Training on both complete views would double the augmented matrix and would change the clean-to-synthetic balance.

A five-model-seed comparison evaluated the single `20260807` view and a constant-size mix that retained alternating rows from `20260807` and `20260808`. Median blocked OOF F1 changed from `99.160%` to `99.214%`, median worst-chip CV FP changed from `3.429%` to `1.714%`, median paired worst recall changed from `93.983%` to `92.264%`, and median paired maximum FP changed from `0.143%` to `0.286%`. The mixed candidate produced zero effective quiet alarms, and its paired worst recall remained at least `91.404%` across the five model seeds, above the `90%` weak-link target. The recall cost was concentrated in the known C5 weak-link selection replay, while some S3 weak-link runs improved.

## Decision

Production packet augmentation uses fixed seeds `20260807` and `20260808`. For each source recording, view index `i` contributes row positions whose index modulo the number of views equals `i`. With two views, seed `20260807` contributes even row positions and seed `20260808` contributes odd row positions. The assignment is local to the source file, deterministic, independent of the model seed, and approximately preserves the size and weight of one augmented matrix.

The trainer persists the selected per-source result as `ml_training_augmentation_rows`. Its cache identity includes the source capture, requested features, feature implementation, ordered view provenance, augmentation configuration, both seeds, and the mixing implementation. A warm cache hit loads the combined rows before either augmented packet stream is materialized.

Single-view packet-stress diagnostics continue to use seed `20260807`; the two-view policy applies to model training, cross-group training diagnostics, and targeted feature ablations.

## Consequences

- Training covers complementary augmentation tails without doubling synthetic sample weight or memory use.
- The accepted trade reduces the worst false-positive tail materially at a small weak-link recall cost that remains above the deployment target in the measured campaign.
- A cold cache must build both augmented views once per source. Subsequent runs load one combined artifact per source.
- Changing seed order, the selection rule, augmentation code, feature code, or source content invalidates the combined cache entry.
