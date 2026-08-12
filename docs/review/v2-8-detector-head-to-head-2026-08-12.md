# v2.8 Detector Head-to-Head

Date: 2026-08-12
Status: Complete
Dataset revision: `sha256:6fd0edc261f4132f563c0ce82a1bf2403ffe5b73637957614be8cc817aae79ee`

## 1. Question

Quantify the detection change since v2.8.0 without comparing metrics from different corpora. The controlled ML comparison retains the v2.8 Raw-9 feature definitions and `9 -> 32 -> 16 -> 1` layout, retrains that model on the current role-isolated corpus with the current training policy, and compares it with the promoted Subband-7 model. The current Classic detector provides the non-ML base-detector reference on the same recorded corpus.

This is not a replay of the original v2.8 weights. Those weights were fitted to the smaller v2.8 corpus, so using them directly would mix model-age and dataset-shift effects. The comparison instead holds the current corpus, split policy, seed, scaler, augmentation, false-positive weight, and runtime replay policy constant while preserving the historical feature family and network layout.

## 2. Controlled ML Protocol

Both ML fits used seed `1584727888`, standard scaling, `fp_weight=1.75`, `stream_dense` training, `base,drift,burst-loss` augmentation with packet seeds `20260807` and `20260808`, three-fold lineage-grouped blocked CV, and the current `train`, `selection`, `holdout`, and `exclude` role isolation. The training corpus contained 49 clean and 49 augmented captures, 27 lineage groups, and 617,883 final training samples after the constant-size augmented row mix. Selection and holdout replays were excluded from fitting.

The historical model used the exact v2.8 Raw-9 inputs: `turb_mean`, `turb_std`, `turb_max`, `turb_min`, `turb_iqr`, `turb_skewness`, `turb_autocorr`, `turb_mad`, and `waveform_length`. These definitions remain host-only and do not alter the production runtime feature surface.

Reproduction command:

```bash
.venv/bin/python tools/train_ml_model.py --seed 1584727888 --augment --fp-weight 1.75 --scaler standard --hidden-layers 32,16 --features turb_mean,turb_std,turb_max,turb_min,turb_iqr,turb_skewness,turb_autocorr,turb_mad,waveform_length --evaluate-gates --no-export
```

## 3. Controlled ML Results

All rates are percentages. Worst-session values come from blocked OOF evaluation; paired and quiet values use the current production replay cadence and hit policy across the reserved selection and holdout gates.

| Metric | Retrained v2.8 Raw-9 | Current Subband-7 | Change |
| --- | ---: | ---: | ---: |
| Inputs | `9` | `7` | `-2` |
| Parameters | `865` | `505` | `-360` (`-41.6%`) |
| Blocked OOF F1 | `94.2` | `99.2` | `+5.0` points |
| Worst-session recall | `96.5` | `88.8` | `-7.7` points |
| Worst-session FP | `99.4` | `1.2` | `-98.2` points |
| Paired replay gates passed | `11/14` | `14/14` | `+3` gates |
| Paired worst recall | `95.98` | `95.69` | `-0.29` points |
| Paired maximum FP | `21.71` | `0.14` | `-21.57` points |
| Paired effective alarms | `34` | `0` | `-34` |
| Quiet maximum FP | `2.13` | `0.73` | `-1.40` points |
| Quiet effective alarms | `3` | `0` | `-3` |

The current model does not improve every recall statistic: Raw-9 retains a higher worst-session OOF recall and is `0.29` points better on the limiting reserved paired replay. Its failure mode is nevertheless unacceptable for production because one C3 static-presence OOF session reaches `99.4%` FP, three reserved paired gates fail, paired maximum FP reaches `21.71%`, and the replay policy produces 37 false alarms across paired and quiet gates. Subband-7 trades the small paired-recall difference and the larger OOF recall tail for a `5.0`-point OOF F1 gain, removal of every effective reserved alarm, a `21.57`-point reduction in paired maximum FP, and a 41.6% smaller MLP.

## 4. Cross-Environment Generalization

Leave-one-environment-out retrains each architecture after excluding every row from the tested environment. It is diagnostic rather than a promotion gate, but directly measures transfer to an unseen room under the same seed, augmentation, scaler, and training policy.

| Held-out environment | Raw-9 recall | Raw-9 FP | Raw-9 F1 | Subband-7 recall | Subband-7 FP | Subband-7 F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Bedroom | `99.8` | `10.1` | `89.5` | `98.9` | `0.9` | `98.4` |
| Hobby room | `97.7` | `0.0` | `98.8` | `97.3` | `0.0` | `98.7` |
| Living room | `96.4` | `0.0` | `98.2` | `99.1` | `0.0` | `99.6` |
| Macro average | `98.0` | `3.4` | `95.5` | `98.5` | `0.3` | `98.9` |

Subband-7 improves macro recall by `0.5` points, macro F1 by `3.4` points, and macro FP by `3.1` points. The decisive tail is the unseen bedroom: Raw-9 reaches `10.1%` FP overall and `28.0%` FP on C3, while Subband-7 holds the environment to `0.9%` FP. Raw-9 is `0.4` points better on hobby-room recall, but that small advantage does not offset the bedroom quiet-tail failure.

## 5. Cross-Chip Generalization

Leave-one-chip-out applies the same diagnostic procedure after excluding every row from the tested chip.

| Held-out chip | Raw-9 recall | Raw-9 FP | Raw-9 F1 | Subband-7 recall | Subband-7 FP | Subband-7 F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| C3 | `98.2` | `17.7` | `82.4` | `95.9` | `0.0` | `97.9` |
| C5 | `98.9` | `0.1` | `99.3` | `99.1` | `0.0` | `99.5` |
| C6 | `96.7` | `0.4` | `97.9` | `99.1` | `1.8` | `97.5` |
| ESP32 | `100.0` | `0.0` | `100.0` | `100.0` | `0.0` | `100.0` |
| S3 | `99.8` | `2.6` | `97.0` | `99.1` | `1.0` | `98.4` |
| Macro average | `98.7` | `4.1` | `95.3` | `98.6` | `0.6` | `98.7` |

Macro recall is effectively tied (`98.7%` versus `98.6%`), while Subband-7 improves macro F1 by `3.4` points and macro FP by `3.5` points. Raw-9's worst held-out recall is `96.7%` on C6, compared with `95.9%` on C3 for Subband-7, but its worst held-out FP is `17.7%` on C3, compared with `1.8%` on C6 for Subband-7. Subband-7 therefore gives up `0.8` points in the worst recall tail while reducing the worst FP tail by `15.9` points. C6 is the one local trade-off where Raw-9 has lower FP (`0.4%` versus `1.8%`) despite lower recall (`96.7%` versus `99.1%`).

The two diagnostic commands append `--cross-environment` or `--cross-chip` to the Raw-9 reproduction command and omit `--evaluate-gates`. The Subband-7 runs use the same options without `--features` and with `--hidden-layers 24,12`. Every run used `--no-export`, and runtime artifacts remained unchanged.

## 6. Current Base-Detector Reference

Classic is not trained, so it cannot participate in blocked OOF evaluation. Its fair reference is the generated current-corpus replay in [README.md](../performance/README.md), using the same runtime cadence and effective-alarm policy.

| Current-corpus replay | Classic | Current Subband-7 ML |
| --- | ---: | ---: |
| Normal-link per-chip recall range | `95.0-100.0` | `100.0` on every reserved chip |
| Normal-link worst per-recording FP | `14.9` | `0.0` on reserved recordings |
| Normal-link effective alarms | `41` | `0` on reserved recordings |
| Weak-link minimum recall | `89.1` | `95.7` on reserved recordings |
| Weak-link worst per-recording FP | `9.0` | `0.1` on reserved recordings |
| Weak-link effective alarms | `28` | `0` on reserved recordings |
| Long-quiet maximum FP | `9.60` | `0.90` |
| Long-quiet effective alarms | `43` | `0` |

Classic remains the lower-complexity, self-calibrating base detector, but the current ML model provides the stronger false-alarm envelope on this corpus. The largest operational gain since v2.8 is therefore not higher already-saturated aggregate recall; it is robust quiet and cross-session discrimination with zero effective alarms on the reserved normal-link, weak-link, and long-quiet report slices.

## 7. Decision

The roadmap item is complete. On the current corpus, the promoted Subband-7 model materially outperforms a Raw-9 model that preserves the v2.8 feature family and layout under the current training contract. Leave-one-environment-out and leave-one-chip-out diagnostics confirm that the gain is a substantially safer unseen-group FP tail rather than uniformly higher recall. Subband-7 also provides a substantially safer false-positive and effective-alarm envelope than the current Classic base detector. The evidence supports retaining Subband-7 as the production ML model while keeping Classic as the non-ML fallback.
