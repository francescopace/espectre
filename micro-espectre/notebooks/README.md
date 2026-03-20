# Jupyter Notebooks

**Interactive CSI data exploration and ML pipeline walkthrough for ESPectre**

These notebooks are designed for researchers, developers, and contributors who want to understand how Wi-Fi CSI motion detection works — from raw signal data to ML inference.

## Notebooks

| # | Notebook | Description |
|---|----------|-------------|
| 01 | [CSI Data Explorer](01_csi_data_explorer.ipynb) | Load, visualize, and understand raw CSI data. Covers amplitude heatmaps, spatial turbulence, moving variance segmentation, NBVI subcarrier selection, and cross-chip comparison. |
| 02 | [Feature Extraction & ML](02_feature_extraction_and_ml.ipynb) | Walk through the 12-feature extraction pipeline, visualize feature distributions, and run the MLP neural network inference. Includes confusion matrix, ROC analysis, and comparison to baseline classifier. |

## Setup

The notebooks share the same virtual environment as the rest of Micro-ESPectre.

### Prerequisites

- Python 3.12+ (recommended)
- The CSI datasets in `../data/` (included in the repository)

### 1. Create virtual environment (if not already done)

```bash
cd espectre/micro-espectre

# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate  # macOS/Linux
# venv\Scripts\activate   # Windows
```

### 2. Install dependencies

```bash
# Core dependencies (enough for both notebooks)
pip install numpy matplotlib scipy

# Full dependencies (includes ML training, CLI tools, etc.)
pip install -r requirements.txt
```

### 3. Install Jupyter

```bash
pip install jupyter
```

### 4. Run

```bash
# From the micro-espectre directory
cd notebooks
jupyter notebook
```

Or open a specific notebook:

```bash
jupyter notebook 01_csi_data_explorer.ipynb
```

### Using VS Code

If you prefer VS Code, install the **Jupyter** extension and select the `venv` Python interpreter when prompted. VS Code will handle the rest.

## Data

The notebooks load CSI datasets from `../data/`:

```
data/
├── baseline/     # Quiet room recordings (10 .npz files, 5 chips)
├── movement/     # Human movement recordings (10 .npz files, 5 chips)
├── test/         # Mixed baseline+movement recordings
└── dataset_info.json   # Metadata for all files
```

Each `.npz` file contains:
- `csi_data`: Raw CSI I/Q values, shape `(num_packets, 128)` — 64 subcarriers × 2 (I/Q)
- `timestamps`: Packet timestamps (when available)

Chip variants in the dataset: ESP32, ESP32-S3, ESP32-C3, ESP32-C5, ESP32-C6.

## Contributing New Notebooks

When adding a notebook:

1. Use the naming convention `NN_short_description.ipynb` (e.g., `03_gesture_recognition.ipynb`)
2. Include a markdown header cell with title, purpose, and prerequisites
3. Keep dependencies to `numpy` + `matplotlib` where possible (the notebooks should work without tensorflow/sklearn)
4. Use `../data/` relative paths for datasets
5. Add an entry to this README table

## Related Resources

- [ALGORITHMS.md](../ALGORITHMS.md) — Algorithm documentation (MVS, NBVI, ML features)
- [ML_DATA_COLLECTION.md](../ML_DATA_COLLECTION.md) — How to collect labeled CSI datasets
- [tools/README.md](../tools/README.md) — Analysis scripts for advanced workflows
- [tools/10_train_ml_model.py](../tools/10_train_ml_model.py) — Production ML training script
