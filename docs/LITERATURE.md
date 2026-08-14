# Wi-Fi CSI Literature Notes

This document is the durable literature index for ESPectre sensing research. It records the parts of each source that matter to this project: signal features, preprocessing and filters, algorithms, evaluation results, hardware assumptions, and the resulting research action.

This index is for sensing researchers and feature contributors. It is intentionally detailed and does not explain the operational detector; start with [ALGORITHMS.md](ALGORITHMS.md) for current behavior or [TUNING.md](TUNING.md) for device settings. In the notes, HT20 means a 20 MHz Wi-Fi channel, CIR means channel impulse response, and a transfer limit explains why a published result may not apply directly to ESP32 hardware or ESPectre data.

The index covers every external publication reviewed from the local `.papers` collection and the additional online sources reviewed through 2026-08-14. Internally authored ESPectre research is excluded; the historical NBVI work is retained in the decision history of the [fixed-band ADR](adr/2026-07-25-select-the-classic-band-from-channel-coherence.md). Primary publisher, DOI, institutional, or arXiv links are preferred so the local PDF collection is not required. Release dates refer to the first public version or online publication date, not the date on which ESPectre reviewed the source.

This is not evidence that an algorithm works on ESPectre data. Published accuracy values are rarely comparable because tasks, labels, radio hardware, packet rates, environments, splits, and leakage controls differ. Use [FEATURES.md](FEATURES.md) for ESPectre measurements and verdicts, and use ADRs for durable production decisions.

## How To Read The Notes

Transfer labels mean:

- **Direct**: the physical idea is compatible with single-link ESP32 HT20 CSI and can be expressed without absolute CSI scale.
- **Adapt**: the idea is useful, but the published pipeline depends on multi-antenna phase, wider bandwidth, heavy models, or a formulation that must be made scale invariant.
- **Validation**: the main value is evaluation design, corpus construction, or operational robustness rather than a new scalar feature.
- **Context**: useful background or a separate sensing task, but not a current motion-feature candidate.
- **Defer**: the required resolution or topology is not available in the current ESP32 HT20 contract.

Unless a note explicitly says otherwise, amplitude-domain pipelines use absolute magnitudes and are not scale invariant as published. Normalization, ratios, correlations, ranks, phase differences, and within-map power fractions can be scale invariant, but every concrete implementation still needs a gain stress test and near-zero handling.

## Conclusions For ESPectre

The literature reinforces six project-level conclusions:

1. **Remove nuisances before extracting richer features.** Packet gain, carrier-frequency offset, sampling-frequency offset, packet detection delay, phase rotation, null tones, and irregular sampling can otherwise dominate the apparent sensing signal.
2. **Frequency structure is the strongest unexploited HT20 axis.** Cross-subcarrier ratios, rank changes, coherence-versus-frequency-offset, and robust distributional aggregation retain channel-shape information without trusting absolute magnitude.
3. **A single selected subcarrier is fragile.** Several works improve results by combining subcarriers, selecting them per task, or learning from their distribution. ESPectre should prefer compact robust summaries over a brittle winner-takes-all choice.
4. **Hardware and environment generalization dominate benchmark accuracy.** Room-random or packet-random splits can substantially overstate deployment performance. Recording lineage, device, room, person, and quiet-replay groups must remain outside one another across validation folds.
5. **Presence is not merely a lower motion threshold.** Respiration and other micro-motion work uses different bands, windows, subcarrier selection, and stationarity assumptions. It belongs in a separate Presence-versus-Empty task.
6. **Bandwidth limits the physical representation.** CIR, delay, range, and range-Doppler papers are useful long-term directions, but results obtained with 80-160 MHz, multiple antennas, or monostatic Wi-Fi cannot justify an HT20 ESP32 feature.

The most actionable scale-invariant experiments remain:

| Priority | Experiment | Literature basis | ESPectre requirement |
| ---: | --- | --- | --- |
| 1 | Cross-subcarrier ratio or rank dynamics | SA-WiSense and statistical subcarrier utilization | Prove near-zero stability and incremental value over `chan_shape_spread` |
| 2 | Multi-offset frequency-coherence curve | Subcarrier-utilization and high-resolution sensing work | Compress to one ratio or normalized slope and test HT20 resolution |
| 3 | Local phase closure | Phase sanitization and amplitude-plus-phase fusion work | Cancel common phase analytically before any classifier experiment |
| 4 | Packet-age or cadence robustness | Resource-efficient sensing and large deployments | Treat as timing/provenance input unless it carries independent physical information |
| 5 | Micro-motion spectral concentration | SA-WiSense, RF-DS, and heart-rate sensing | Evaluate only on a separate Presence-versus-Empty corpus |

## Preprocessing And Physical Feature Sources

### Optimal Preprocessing of WiFi CSI for Sensing Applications

- **Source:** [arXiv](https://arxiv.org/abs/2307.12126), later published in IEEE Transactions on Wireless Communications
- **Released:** 2023-07-22; journal version 2024
- **Setup and method:** derives amplitude-gain and phase-error models, then corrects those hardware and synchronization distortions before downstream sensing.
- **Results:** reports substantial simulated noise reduction and roughly 20% respiration-SNR improvement on real measurements in the latest public manuscript.
- **ESPectre:** **Direct** for the principle, **Adapt** for the exact corrections. It is the strongest support for validating the captured profile `scale` and for sanitizing phase before feature extraction. It does not validate absolute amplitude as a stable feature by itself.

### SA-WiSense: A Blind-Spot-Free Respiration Sensing Framework for Single-Antenna Wi-Fi Devices

- **Source:** [arXiv](https://arxiv.org/abs/2507.17623)
- **Released:** 2025-07-23
- **Setup and method:** single-antenna ESP32 links; cross-subcarrier CSI ratios cancel random phase offsets, and a genetic algorithm selects ratio pairs by sensing-signal-to-noise ratio.
- **Results:** 91.2% respiration detection at distances up to 8 m.
- **ESPectre:** **Direct** physical evidence for cross-subcarrier ratios on ESP32-class hardware. Ratios are scale invariant when numerator and denominator share the same packet scale, but low-magnitude denominators and clipping need explicit guards. The reported genetic search is too heavy and task-specific for the runtime; a compact ratio, rank, or robust summary is the useful lead.

### Statistical CSI Subcarrier Utilization for Wi-Fi Sensing

- **Source:** [Journal of Information and Intelligence](https://www.sciencedirect.com/science/article/pii/S2949715924000374)
- **Released:** 2024-05; issue date 2024-07
- **Method:** derives detection and estimation information for each subcarrier, then consolidates evidence statistically instead of selecting a fixed carrier or using a simple weighted sum.
- **Results:** the proposed utilization improves the paper's detection and estimation tasks over fixed extraction and weighted aggregation.
- **ESPectre:** **Direct** motivation for robust subcarrier distributions, rank turnover, and aggregation. The published estimator is not itself an approved feature; a candidate must be rewritten as a dimensionless statistic and compared with the existing channel-shape and coherence candidates.

### Subcarrier Selection for Efficient CSI-Based Indoor Localization

- **Source:** [DOI 10.1088/1757-899X/383/1/012017](https://doi.org/10.1088/1757-899X/383/1/012017)
- **Released:** 2018-07
- **Setup and method:** Intel 5300 with two transmit and three receive antennas; histogram-equalizes magnitude CSI, ranks individual carriers by information gain, and retains a lower-dimensional localization fingerprint.
- **Results:** outperforms the paper's conventional feature-selection comparisons while using fewer CSI inputs.
- **ESPectre:** **Adapt**. It establishes that carriers differ in task information, but supervised information gain can encode room-specific fingerprints and absolute magnitude. Rank-dynamics or dimensionless distribution summaries are safer motion candidates than copying the location-trained selector.

### Linear-Complexity Subcarrier Selection for Wi-Fi Sensing

- **Source:** [DOI 10.1049/ell2.70237](https://doi.org/10.1049/ell2.70237)
- **Released:** 2025
- **Method:** supervised subcarrier selection with linear complexity in the number of subcarriers, intended to avoid PCA covariance and decomposition cost.
- **Results:** reports classification comparable to or better than the tested PCA-style preprocessing while reducing preprocessing work.
- **ESPectre:** **Adapt**. Runtime cost is attractive, but label-supervised carrier selection can overfit a corpus or environment. Prefer a stable scale-invariant summary unless grouped cross-device validation proves that a selected set transfers.

### Device-Free Wireless Sensing for Gesture Recognition Based on Complementary CSI Amplitude and Phase

- **Source:** [Sensors](https://www.mdpi.com/1424-8220/24/11/3414)
- **Released:** 2024-05-25
- **Method:** transforms phase to reduce linear offsets, then fuses sanitized phase with amplitude because their blind spots are complementary.
- **Results:** the joint representation outperforms either modality alone on the paper's gesture task.
- **ESPectre:** **Adapt**. It supports a local closure-phase experiment, not reuse of raw phase or absolute amplitude. Any phase candidate must first survive chip, packet-rate, and clipping tests and add information beyond the five production features.

### Channel Phase Processing in Wireless Networks for Human Activity Recognition

- **Source:** [Internet of Things](https://www.sciencedirect.com/science/article/pii/S2542660523002834); [arXiv manuscript](https://arxiv.org/abs/2303.16873)
- **Released:** 2023-03-29 preprint; 2023-10-12 online publication
- **Method:** the TSFR pipeline applies linear-regression phase sanitization, Savitzky-Golay smoothing over time, and reconstruction across frequency.
- **Results:** reports more than 90% on most combinations of five datasets and three deep models; in one few-shot setting, preprocessing raises accuracy from about 35% to 85%.
- **ESPectre:** **Adapt**. A simpler ESPectre phase-residual candidate did not validate, but that does not reproduce full TSFR. Revisit only as a distinct, analytically phase-invariant formulation; do not infer portability from Intel-NIC results.

### An Integrated Method for Tunnel Health Monitoring Data Analysis and Early Warning

- **Source:** [Sensors](https://www.mdpi.com/1424-8220/23/17/7460)
- **Released:** 2023-08-28
- **Method:** fills missing values, handles three-sigma outliers, removes slow trends, compares several smoothers, selects Savitzky-Golay, and then uses wavelet denoising and non-uniform coefficient variation for alerts.
- **Results:** improves extraction of slow structural deformation signals in a tunnel-monitoring dataset.
- **ESPectre:** **Context**. It is useful as a disciplined filter-comparison example, not evidence for a Wi-Fi CSI feature. Its time scale and sensor physics differ sharply from motion CSI.

### High-Resolution Indoor Sensing Using Channel State Information of WiFi Networks

- **Source:** [Electronics](https://www.mdpi.com/2079-9292/12/18/3931)
- **Released:** 2023-09-18
- **Method:** phase calibration, error correction, minimum-entropy subcarrier selection, frequency or instantaneous-phase respiration estimation, and MUSIC angle-of-arrival estimation.
- **Results:** identifies a 0.3 Hz respiration example and reports AoA errors of 1-4 degrees after calibration.
- **ESPectre:** **Adapt** for phase-calibration lessons and **Defer** for AoA. MUSIC requires multiple antennas and stable inter-antenna phase, which the current sensing contract does not expose.

### Non-Contact Heart Rate Monitoring Method Based on Wi-Fi CSI Signal

- **Source:** [Sensors](https://www.mdpi.com/1424-8220/24/7/2111)
- **Released:** 2024-03-26
- **Setup and method:** Intel 5300 at 200 Hz; combines antenna CSI ratios, rotational projection, Savitzky-Golay filtering, frequency-domain heartbeat-to-subcomponent ratio selection, and fusion of five spectral peaks.
- **Results:** average accuracy 96.8%, median error about 0.8 bpm, 80% of estimates within 2 bpm, and 90% within 4.1 bpm across nine participants.
- **ESPectre:** **Context** for a future stationary micro-motion task. The multi-antenna ratio and 200 Hz cadence are not current ESPectre assumptions. The transferable idea is a within-spectrum power ratio or concentration, never absolute power.

## Motion, Activity, And Recognition Sources

### Indoor Motion Detection Using Wi-Fi CSI in Flat Floors Versus Staircases

- **Source:** [Sensors](https://www.mdpi.com/1424-8220/18/7/2177)
- **Released:** 2018-07-06
- **Setup and method:** Intel 5300 multi-stream CSI; Hampel and low-pass filtering, correlation-guided subcarrier averaging, PCA across streams, moving variance segmentation, and Doppler-spread features.
- **Results:** demonstrates reliable motion segmentation and finds staircase motion easier than flat-floor motion because the multipath geometry produces stronger changes.
- **ESPectre:** **Context** and historical foundation for the Classic detector. Moving variance and absolute spread are gain-sensitive as published. The useful durable lessons are robust outlier handling, correlation-aware fusion, and evaluation across geometries.

### WiFi Motion Detection: A Study into Efficacy and Classification

- **Source:** [arXiv](https://arxiv.org/abs/1908.08476)
- **Released:** 2019-08-20; the manuscript says the original project completed in 2018-08
- **Method:** moving averages and variance-like anomaly signals, followed by unsupervised analysis and decision-tree, Naive Bayes, and LSTM comparisons.
- **Results:** exploratory evidence that commodity Wi-Fi channel variation can separate motion states; the paper is closer to a study report than a strong cross-environment benchmark.
- **ESPectre:** **Context**. It motivates the original problem, but does not justify a current feature or validation threshold.

### CSI-HC: A WiFi-Based Indoor Complex Human Motion Recognition Method

- **Source:** [Wireless Communications and Mobile Computing](https://onlinelibrary.wiley.com/doi/10.1155/2020/3185416)
- **Released:** 2020-02-26
- **Setup and method:** Atheros AR9380 amplitude CSI; Butterworth low-pass and Symlet-8 wavelet denoising, restricted Boltzmann machine fingerprints, and a SoftMax classifier.
- **Results:** average 85.4% across meeting-room, corridor, and office experiments using 10,000 packets per activity.
- **ESPectre:** **Context**. The environment spread is useful, but the amplitude representation, dataset size, and heavy classifier are not a direct scalar-feature recipe.

### Device-Free Human Activity Recognition Based on GMM-HMM Using CSI

- **Source:** [DOI 10.1109/ACCESS.2021.3082627](https://doi.org/10.1109/ACCESS.2021.3082627)
- **Released:** 2021-05-21
- **Method:** linear phase correction, Savitzky-Golay phase filtering, mean and standard-deviation summaries of an expanded phase-difference matrix, and a GMM-HMM temporal classifier.
- **Results:** reports above 97% on the self-collected set, 97.8% on a comparison set, 99.0% on the ITI set, and above 95.9% in the tested cross-environment cases.
- **ESPectre:** **Adapt**. It supports phase-difference and temporal-state research, but controlled activity classes and Intel-style phase do not transfer directly to noisy binary motion replays.

### A CSI-Based Multi-Environment Human Activity Recognition Framework

- **Source:** [Applied Sciences](https://www.mdpi.com/2076-3417/12/2/930)
- **Released:** 2022-01-17
- **Setup and method:** Intel 5300, one transmit and three receive antennas, 30 reported subcarriers, 20 MHz, and 320 packets/s; PCA, Hampel filtering, Gaussian smoothing, segmentation, handcrafted time/frequency features, and SVM classification.
- **Results:** 91.27% average six-activity accuracy over 20 subjects in office and hallway environments; a harder cross-environment scenario reports 88.82%.
- **ESPectre:** **Adapt**. It supports grouped environment evaluation and robust filtering. The selected handcrafted amplitude features are not automatically scale invariant, and the radio topology is richer than one ESP32 link.

### ROCKET: Exceptionally Fast and Accurate Time Series Classification Using Random Convolutional Kernels

- **Source:** [arXiv](https://arxiv.org/abs/1910.13051), later published in Data Mining and Knowledge Discovery
- **Released:** 2019-10-29; journal version 2020
- **Method:** transforms univariate time series with 10,000 random convolution kernels by default, varying kernel length, weights, bias, dilation, and padding; each response contributes its maximum and proportion of positive values before ridge regression or logistic classification.
- **Results:** ranks competitively with HIVE-COTE, TS-CHIEF, and InceptionTime across 85 UCR datasets without a statistically significant accuracy difference among those leading methods. The 10,000-kernel variant trains on one million synthetic time series in about 1 hour 15 minutes, while a 100-kernel variant takes under one minute at lower accuracy.
- **ESPectre:** **Context** and the algorithmic basis for LiteHAR, not Wi-Fi sensing evidence. Random multiscale filters and the dimensionless proportion-positive summary are useful host-side research leads, but 10,000 kernels and up to 20,000 output features are not a compact firmware design. The maximum response is gain-sensitive, and proportion positive is not scale invariant when the random bias is nonzero unless the input normalization contract is preserved.

### LiteHAR: Lightweight Human Activity Recognition From WiFi Signals With Random Convolution Kernels

- **Source:** [arXiv](https://arxiv.org/abs/2201.09310); IEEE ICASSP 2022
- **Released:** 2022-01-23
- **Setup and method:** public indoor activity CSI with three receive antennas, 30 subcarriers per antenna, 1 kHz capture, and 20-second samples; downsamples amplitude to 500 Hz, subtracts its mean, divides by its L2 norm, applies 10,000 ROCKET-style random kernels independently per subcarrier, fits ridge classifiers, and votes across subcarriers.
- **Results:** ten-fold cross-validation reports 93% average accuracy over six activities and 91% over all seven. Six-class training takes 157.8 seconds, about 82 times less than the compared ABLSTM, while per-sample inference takes 0.013 seconds; restricting voting to the two stronger antennas improves the reported accuracy by about one percentage point.
- **ESPectre:** **Adapt** for lightweight host-side sequence classification and evidence that carrier contributions are unequal. The three-antenna topology, 500 Hz input, 20-second windows, 10,000 kernels per carrier, and ten-fold cross-validation without reported grouped device, subject, or environment holdouts do not match ESPectre's single-link runtime or grouped deployment gates. Mean and L2 normalization removes offset and scale within each signal, but the resulting representation should be tested only as a host candidate with device-, room-, and replay-group isolation before any compact distillation is considered.

### CSI-F: Human Motion Recognition Through CSI Feature Fusion

- **Source:** [Sensors](https://www.mdpi.com/1424-8220/24/3/862)
- **Released:** 2024-01-29
- **Method:** discrete wavelet denoising, antenna-diversity phase-offset elimination, PCA, STFT/Doppler and FFT motion energy, and a multi-task CNN-GRU that fuses periodic action features.
- **Results:** 94.68% versus 92.36% for LSTM, 85.47% for HMM, and 78.51% for decision tree in the reported comparison; coverage is tested to 6 m.
- **ESPectre:** **Adapt**. Periodic spectral structure is useful for gesture or micro-motion work, but the multi-antenna processing and model are too heavy for the current scalar MLP. Power fractions can be scale invariant; raw FFT energy is not.

### Wi-Limb: Recognizing Moving Body Limbs Using a Single WiFi Link

- **Source:** [author manuscript](https://ebulutvcu.github.io/MobiCom4AgeTech2.pdf); [DOI 10.1145/3636534.3698117](https://doi.org/10.1145/3636534.3698117)
- **Released:** 2024-11-18
- **Setup and method:** one ESP32 transmitter-receiver link at 100 Hz; removes 12 constant tones, applies 100-sample window averaging and PCA, then compares flat DNN/GAN classifiers, multi-label limb classification, and a domain-adversarial hierarchical GAN.
- **Results:** exact-match accuracy is 33.39% for the flat DNN, 42.87% for the flat GAN, 51.43% for multi-label classification, and 76.47% for the hierarchical GAN; corresponding Hamming loss improves to 0.0705.
- **ESPectre:** **Context** for a future gesture task and **Validation** for decomposing labels that share physical parts. Window-averaged absolute amplitude and PCA are not scale invariant as published.

### Gesture Recognition Using WiFi

- **Source:** NTU WNFA Team 3 final project report; no stable public project URL located
- **Released:** 2022-04-01 according to PDF metadata
- **Setup and method:** Raspberry Pi CSI extraction from beacon packets at 40 MHz and about 50 Hz; horizontal-shift augmentation and a CNN with four convolutional and three fully connected layers.
- **Results:** seven gestures, reported precision 0.943, and inference within 0.5 s. The report notes reboot/device drift and collection consistency as practical issues.
- **ESPectre:** **Context** and gray literature, not peer-reviewed evidence. Preserve the device-drift warning and separate gesture corpus idea; do not reuse the magnitude image pipeline as a scale-invariant motion feature.

### MORIC: Multi-Order Random Invariant Convolution for Wi-Fi Sensing

- **Source:** [arXiv](https://arxiv.org/abs/2506.12997)
- **Released:** 2025-06-15
- **Method:** delay-profile and Doppler-velocity representations followed by random convolution kernels and max pooling designed to be invariant to input order and repetition.
- **Results:** on a four-class leave-one-subject-out task, 56.3% without calibration; binary tasks average 80.9%; a best-link setting reaches 88.5% with four calibration samples per class and 98.8% with ten. Removing phase compensation drops the four-class result to 27.8%.
- **ESPectre:** **Adapt** for invariant representation and few-shot calibration lessons. Delay/Doppler inputs need more resolution than the present scalar path, and the calibration sensitivity warns against reading high within-domain accuracy as generalization.

### Why Commodity WiFi Sensors Fail at Multi-Person Gait Identification

- **Source:** [arXiv](https://arxiv.org/abs/2601.02177)
- **Released:** 2026-01-05
- **Setup and method:** commodity ESP32; compares FastICA, SOBI, PCA, NMF, wavelet, and tensor decomposition across seven scenarios containing one to ten people.
- **Results:** all separation methods remain at 45-56% accuracy with statistically insignificant differences; the best, NMF, reaches 56%. Intra-person variability is high, inter-person distinguishability is low, and performance degrades sharply as the number of people rises.
- **ESPectre:** **Validation** and important negative evidence. Do not place multi-person identity separation on the current single-link ESP32 roadmap without a material change in topology or signal quality.

## Presence, Localization, And High-Resolution Sources

### Home Presence Detection and Localization Using Wi-Fi CSI

- **Source:** [University of Washington ECE poster](https://www.ece.uw.edu/wp-content/uploads/2024/08/Amazon-Home-Presence-Detection-and-Localization-using-Wi-Fi-CSI.pdf)
- **Released:** 2024; poster published online 2024-08
- **Setup and method:** two ESP32-S3 devices, 20 MHz, 52 subcarriers, and 100 packets/s across 25 rooms. Presence uses a three-level wavelet transform and an RNN; localization compares ResNet50 and LSTM/RNN variants.
- **Results:** presence accuracy is 92.84-98.93% across four rooms and exceeds 90% true-positive rate. Leave-one-room-out near-device versus near-AP localization is weaker at 71-75% accuracy and 61-78% true-positive rate.
- **ESPectre:** **Direct** evidence that ESP32 HT20 contains presence signal, and **Validation** evidence that location transfer is much harder than presence. The result motivates a separate Presence-versus-Empty corpus, not a motion-threshold recalibration.

### Location Intelligence for People Estimation During Emergency Operation

- **Source:** [HICSS repository](https://aisel.aisnet.org/hicss-55/li/research/7/)
- **Released:** 2022-01
- **Setup and method:** ESP32 CSI, fourth-order Daubechies wavelet filtering, PCA dimensionality reduction, and DNN comparison with SVM, k-nearest neighbors, and other classifiers.
- **Results:** individual room and occupancy-class experiments range from high 70s to high 90s depending on model and scenario; the spread is itself the important result because environment and split dominate.
- **ESPectre:** **Adapt**. It confirms ESP32 feasibility and the cost of room transfer. PCA over raw magnitude is not a compact scale-invariant feature, and per-room accuracy must not substitute for held-out-room validation.

### Wi-Fi Sensing for Human Identification Through ESP32 Devices

- **Source:** [University of Bologna thesis repository](https://amslaurea.unibo.it/id/eprint/29166/)
- **Released:** 2023-07-19
- **Method:** adapts the SHARP Wi-Fi activity/identity pipeline to ESP32 CSI, using image-like signal representations and learned classification.
- **Results:** about 95% for two people and 74% for three in the controlled experiments.
- **ESPectre:** **Context**. It demonstrates low-cost identity signal but also rapid degradation as class count grows. Identity inference has distinct privacy implications and is not part of the current detector.

### Person Re-Identification Through Wi-Fi Extracted Radio Biometric

- **Source:** [institutional record](https://iris.uniroma1.it/handle/11573/1619785); [DOI 10.1109/TIFS.2022.3158058](https://doi.org/10.1109/TIFS.2022.3158058)
- **Released:** 2022-03
- **Method:** sanitized amplitude heatmaps and phase vectors feed parallel siamese branches to create radio-biometric signatures.
- **Results:** for 20 known people across three rooms, joint amplitude and phase reach 93.51% Rank-1 and 92.17% mAP with 300 packets. For 15 identities unseen during training, the same setting reaches 88.82% Rank-1 and 87.52% mAP; ten packets fall to 72.12% and 64.52%.
- **ESPectre:** **Context** and privacy evidence. Amplitude and phase are complementary, but the heavy representation and identity objective do not justify a motion feature. The packet-count ablation is a useful reminder that short windows trade latency for stability.

### CIRSense: High-Resolution Wi-Fi Sensing With CIR

- **Source:** [arXiv](https://arxiv.org/abs/2510.11374)
- **Released:** 2025-10-13
- **Method:** 160 MHz CSI, fractional-delay CIR modeling, hardware-distortion compensation, high-resolution distance estimation, and aggregation across subcarriers.
- **Results:** mean respiration error 0.25 bpm, mean distance error 0.09 m, and at 20 m at least threefold accuracy and more than 4.5-fold computational efficiency improvement over tested baselines.
- **ESPectre:** **Defer**. The calibration ideas are informative, but 160 MHz delay resolution is not evidence for a useful HT20 ESP32 CIR scalar.

### Range-Doppler Sensing From Commodity Wi-Fi CSI

- **Source:** [arXiv](https://arxiv.org/abs/2508.02799)
- **Released:** 2025-08-04
- **Setup and method:** monostatic Intel AX211; cancels time offset, corrects phase alignment, mitigates transmitter-receiver coupling, and constructs a range-Doppler representation.
- **Results:** reports centimeter-level ranging together with Doppler estimation on its wideband monostatic setup.
- **ESPectre:** **Defer**. Neither monostatic topology nor its bandwidth and synchronization assumptions match the current ESP32 link.

### RF-DS: Device-Free Presence Through Range-Filtered Doppler Signatures

- **Source:** [arXiv](https://arxiv.org/abs/2603.10845)
- **Released:** 2026-03-11; revised 2026-04-23
- **Method:** filters CIR by range before computing Doppler, uses temporal spectral windows, and switches adaptively between 10 Hz and 100 Hz sensing rates.
- **Results:** reports robust device-free presence without per-room calibration or retraining on the tested platform.
- **ESPectre:** **Context** for separating presence from motion and for multi-rate acquisition. The CIR resolution is not transferable to HT20, but a dimensionless narrowband spectral concentration remains worth testing on a dedicated presence corpus.

## Surveys, Datasets, Deployment, And Security

### WiFi Sensing on the Edge: Signal Processing Techniques and Challenges

- **Source:** [author manuscript](https://ebulutvcu.github.io/COMST22_WiFi_Sensing_Survey.pdf); [DOI 10.1109/COMST.2022.3209144](https://doi.org/10.1109/COMST.2022.3209144)
- **Released:** 2022 online; IEEE Communications Surveys & Tutorials issue 2023
- **Scope:** surveys collection, denoising, PCA, wavelets, subcarrier selection, feature extraction, and edge models, and also measures several sensing workloads on constrained devices.
- **Results:** link placement strongly changes activity accuracy; one multi-link example reports 58.52%, 71.24%, and 49.89% for individual links, while best-link selection adds 17.90 points over one link and 39.25 over another.
- **ESPectre:** **Validation**. Placement, link quality, memory, latency, and energy belong beside F1 in any promotion decision.

### SenseFi: A Library and Benchmark on Deep-Learning-Empowered WiFi Sensing

- **Source:** [arXiv](https://arxiv.org/abs/2207.07859)
- **Released:** 2022-07-16; journal publication 2023
- **Scope:** common datasets and implementations for convolutional, recurrent, attention, and other deep models, evaluated across sensing tasks, platforms, complexity, transfer, and adaptation.
- **Results:** shows that no architecture dominates every dataset and that transfer and unsupervised adaptation can matter as much as within-dataset accuracy.
- **ESPectre:** **Validation**. Reuse the comparison discipline, not the model leaderboard. ESPectre's grouped replays, runtime budget, and export parity remain stricter constraints than a public-dataset top score.

### Wi-Fi CSI Features Collection in Access Points for IoT Forensics

- **Source:** [Politecnico di Milano thesis](https://www.politesi.polimi.it/handle/10589/196727); related [paper preprint](https://arxiv.org/abs/2305.10554)
- **Released:** thesis 2022-12-20; conference paper 2023
- **Scope:** integrates CSI collection into access points and explores storage, compression, and basic presence or passage classification for forensic use.
- **Results:** establishes that useful CSI can be collected at the AP boundary, while exposing retention, traffic, and privacy tradeoffs.
- **ESPectre:** **Context** and architecture evidence. The important lessons are provenance, bounded retention, and derived-data privacy, not a new detector feature.

### A Survey on Device-Free Human Identification Using Wi-Fi CSI

- **Source:** [Sensors](https://www.mdpi.com/1424-8220/24/19/6413)
- **Released:** 2024-10
- **Scope:** organizes identity systems by signal representation, preprocessing, classifier, environment, and evaluation protocol.
- **Results:** highlights persistent sensitivity to person count, environment, position, hardware, and limited public data.
- **ESPectre:** **Context**. Identity is technically possible and therefore a privacy concern even when ESPectre does not implement it.

### A Survey of Device-Free Wi-Fi Sensing

- **Source:** [Computer Communications](https://www.sciencedirect.com/science/article/abs/pii/S0140366424002214); [DOI 10.1016/j.comcom.2024.06.011](https://doi.org/10.1016/j.comcom.2024.06.011)
- **Released:** 2024-08-01
- **Scope:** structures the field into collection, preprocessing, detection algorithm, and application stages.
- **Results:** identifies CSI availability, non-line-of-sight behavior, data density, coexistence, and security as unresolved deployment challenges.
- **ESPectre:** **Validation**. These are acceptance dimensions for the platform, not merely literature caveats.

### BFId: Identity Inference Attacks Using Beamforming Feedback Information

- **Source:** [KIT institutional copy](https://publikationen.bibliothek.kit.edu/1000185756/168100988); [DOI 10.1145/3719027.3765062](https://doi.org/10.1145/3719027.3765062)
- **Released:** 2025; ACM CCS 2025, 2025-10-13 to 2025-10-17
- **Setup and method:** beamforming feedback information with simple machine learning and little preprocessing; 197 participants, with usable BFI for 161 and CSI for 170.
- **Results:** BFI identity accuracy 99.5% +/- 0.38 versus CSI 82.4% +/- 0.62. With only 20% training data, BFI remains above 90% while CSI falls to 28%.
- **ESPectre:** **Context** and security warning. Derived radio telemetry can reveal identity even when a system intends only presence detection. Avoid exporting raw CSI or BFI and retain the local-first, derived-telemetry boundary.

### Experience Paper: Scaling Wi-Fi Sensing to Millions of Devices

- **Source:** [arXiv](https://arxiv.org/abs/2506.04322)
- **Released:** 2025-06-04
- **Scope:** two years of deployment over more than 10 million routers and 100 million bulbs, with controlled evaluation on 280 edge devices, 16 scenarios, and more than 4 million samples.
- **Results:** 92.61% reported accuracy, non-human false alarms reduced from 63.1% to 8.4%, and CSI transmission reduced by 99.72%.
- **ESPectre:** **Validation**. Pets, hardware heterogeneity, multi-user interference, and edge/cloud constraints dominate at scale. The result supports per-device gates and derived telemetry rather than raw-CSI upload.

### CSI-Bench: A Large-Scale In-the-Wild Dataset for Multi-Task WiFi Sensing

- **Source:** [arXiv](https://arxiv.org/abs/2505.21866)
- **Released:** 2025-05-28
- **Scope:** 461 hours of effective data from 35 users in 26 indoor environments, with fall, breathing, localization, motion-source, and co-labeled multi-task data plus standardized splits.
- **Results:** establishes realistic baselines and exposes the generalization gap hidden by short, homogeneous, session-based datasets.
- **ESPectre:** **Validation**. Continuous quiet time, natural transitions, multi-label provenance, and predeclared environment splits are more valuable to the project than copying its model architecture.

### Multi-Station WiFi CSI Sensing Under Missing Stations and Few Labels

- **Source:** [arXiv](https://arxiv.org/abs/2603.11858)
- **Released:** 2026-03-12
- **Method:** cross-modal self-supervised representation learning plus station-wise masking augmentation to model long station outages and scarce labels.
- **Results:** reports that missingness-invariant pretraining or masking alone is insufficient; their combination is needed for robustness to both station loss and limited labeled data.
- **ESPectre:** **Validation** for the future multi-node orchestration layer. It is not a current single-device feature, but it argues for training and testing with whole missing nodes rather than random missing samples.

### Resource-Efficient WiFi CSI Sensing Through Sample Age

- **Source:** [arXiv](https://arxiv.org/abs/2606.31690)
- **Released:** 2026-06-30
- **Method:** encodes the age of each retained CSI sample and fuses it with a learned CSI representation under deterministic and stochastic sensing-rate constraints.
- **Results:** on NTU-Fi activity and identity tasks, improves over CSI-only and time-aware baselines, including up to ten percentage points for identity under strict sensing budgets.
- **ESPectre:** **Direct** operational lesson and **Validation** target. Measured packet timing must accompany temporal features when cadence is irregular. Sample age is scale independent, but should remain timing provenance unless it proves incremental physical value.

### The Universal Language of CSI

- **Source:** [arXiv](https://arxiv.org/abs/2607.09727)
- **Released:** 2026-06-30
- **Method:** dataset-specific adapters map heterogeneous CSI dimensions, sampling rates, and labels into a shared self-supervised Transformer representation.
- **Results:** reports stronger cross-dataset and few-shot generalization than task-specific baselines over the curated heterogeneous corpus.
- **ESPectre:** **Validation** and long-term context. The useful idea is to make hardware and cadence heterogeneity explicit. The foundation model is outside the on-device MLP budget and does not replace scale-invariant physical features.

## Maintenance Rules

When adding a source:

1. link a primary publisher, DOI, institutional record, or arXiv page;
2. record the first public release date and any materially different final publication date;
3. summarize the signal representation, preprocessing, filters, model, and evaluation split;
4. retain only results needed to judge the claim;
5. state the radio hardware, bandwidth, antenna, and cadence assumptions when they affect transfer;
6. classify the ESPectre action and scale-invariance implications; and
7. move actual ESPectre measurements and candidate verdicts to [FEATURES.md](FEATURES.md), rather than duplicating them here.

ADRs may link the relevant source heading in this index for the reviewed evidence, while retaining the decision-time conclusion and a direct primary source link when the publication is essential to the decision.

If the file becomes difficult to scan, move the category sections unchanged under `docs/literature/`, keep `LITERATURE.md` as the index, and preserve its stable source links.
