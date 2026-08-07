#!/usr/bin/env python3
"""
ESPectre - CSI Amplitude Heatmap Plotter

Plot representative CSI amplitude samples as time × subcarrier heatmaps,
matching the common paper-style figure used for activity / motion sources.

Usage:
    python tools/plot_heatmap.py
    python tools/plot_heatmap.py --chip S3 --environment bedroom
    python tools/plot_heatmap.py --labels empty,static_presence,motion
    python tools/plot_heatmap.py --files data/empty/foo.npz data/motion/bar.npz
    python tools/plot_heatmap.py --packets 400 --offset 100 --output /tmp/csi.png

Author: Francesco Pace <francesco.pace@gmail.com>
License: GPLv3
"""

from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.lib.bootstrap import setup_paths  # noqa: E402

setup_paths()

from tools.lib.csi_analysis import extract_amplitudes_matrix  # noqa: E402
from tools.lib.csi_io import load_npz_csi_data  # noqa: E402
from tools.lib.dataset_metadata import (  # noqa: E402
    load_dataset_info,
    resolve_entry_path,
)
from tools.lib.ui import show_plot_window  # noqa: E402

DEFAULT_LABELS = ("empty", "static_presence", "motion")
LABEL_TITLES = {
    "empty": "Empty",
    "static_presence": "Static presence",
    "motion": "Motion",
    "test": "Test",
}


@dataclass(frozen=True)
class HeatmapSample:
    """One CSI window ready for plotting."""

    title: str
    subtitle: str
    amplitudes: np.ndarray  # shape (num_packets, num_subcarriers)
    path: Path


def _parse_csv_list(value: Optional[str]) -> Optional[List[str]]:
    if value is None:
        return None
    items = [part.strip() for part in value.split(",") if part.strip()]
    return items or None


def _human_label(label: str) -> str:
    return LABEL_TITLES.get(label, label.replace("_", " ").title())


def _entry_sort_key(entry: dict) -> Tuple[str, str]:
    return (str(entry.get("collected_at") or ""), str(entry.get("filename") or ""))


def select_metadata_samples(
    *,
    labels: Sequence[str],
    chip: Optional[str],
    environment: Optional[str],
    num_sc: int = 64,
) -> List[Tuple[str, dict, Path]]:
    """Pick the newest matching dataset_info entry for each requested label."""
    info = load_dataset_info()
    files_section = info.get("files", {})
    selected: List[Tuple[str, dict, Path]] = []

    for label in labels:
        entries = list(files_section.get(label, []))
        matches = []
        for entry in entries:
            if chip is not None and str(entry.get("chip", "")).upper() != chip.upper():
                continue
            if environment is not None and str(entry.get("environment", "")) != environment:
                continue
            if int(entry.get("subcarriers", 0) or 0) != int(num_sc):
                continue
            matches.append(entry)
        if not matches:
            filters = []
            if chip:
                filters.append(f"chip={chip}")
            if environment:
                filters.append(f"env={environment}")
            filter_text = f" ({', '.join(filters)})" if filters else ""
            raise FileNotFoundError(f"No dataset_info entry for label '{label}'{filter_text}")
        entry = max(matches, key=_entry_sort_key)
        selected.append((label, entry, resolve_entry_path(label, entry)))
    return selected


def load_amplitude_window(
    path: Path,
    *,
    packets: int,
    offset: int,
    keep_all_phy: bool = False,
) -> np.ndarray:
    """Load HT20 CSI amplitudes for a contiguous packet window."""
    csi = load_npz_csi_data(path, keep_all_phy=keep_all_phy)
    if csi.ndim != 2 or csi.shape[0] == 0:
        raise ValueError(f"Empty or invalid CSI matrix in {path}")
    if offset >= len(csi):
        raise ValueError(f"Offset {offset} exceeds available packets ({len(csi)}) in {path.name}")
    end = min(offset + packets, len(csi))
    window = csi[offset:end]
    if len(window) == 0:
        raise ValueError(f"No packets selected from {path.name}")
    return extract_amplitudes_matrix(window)


def prepare_amplitudes(
    amplitudes: np.ndarray,
    *,
    detrend: bool,
) -> np.ndarray:
    """Optionally remove the per-subcarrier mean to emphasize temporal structure."""
    values = np.asarray(amplitudes, dtype=np.float64)
    if detrend:
        values = values - np.mean(values, axis=0, keepdims=True)
    return values


def build_samples_from_metadata(
    *,
    labels: Sequence[str],
    chip: Optional[str],
    environment: Optional[str],
    packets: int,
    offset: int,
    detrend: bool,
) -> List[HeatmapSample]:
    samples: List[HeatmapSample] = []
    for label, entry, path in select_metadata_samples(
        labels=labels,
        chip=chip,
        environment=environment,
    ):
        amps = prepare_amplitudes(
            load_amplitude_window(path, packets=packets, offset=offset),
            detrend=detrend,
        )
        env = str(entry.get("environment") or "unknown")
        chip_name = str(entry.get("chip") or "?")
        samples.append(
            HeatmapSample(
                title=_human_label(label),
                subtitle=f"{chip_name} · {env}",
                amplitudes=amps,
                path=path,
            )
        )
    return samples


def build_samples_from_files(
    files: Sequence[Path],
    *,
    packets: int,
    offset: int,
    detrend: bool,
) -> List[HeatmapSample]:
    samples: List[HeatmapSample] = []
    for path in files:
        resolved = path if path.is_absolute() else (REPO_ROOT / path)
        if not resolved.exists():
            raise FileNotFoundError(resolved)
        amps = prepare_amplitudes(
            load_amplitude_window(resolved, packets=packets, offset=offset),
            detrend=detrend,
        )
        stem = resolved.stem
        label = stem.split("_", 1)[0] if "_" in stem else stem
        samples.append(
            HeatmapSample(
                title=_human_label(label),
                subtitle=resolved.name,
                amplitudes=amps,
                path=resolved,
            )
        )
    return samples


def _grid_shape(count: int, cols: Optional[int]) -> Tuple[int, int]:
    if count <= 0:
        raise ValueError("Need at least one sample to plot")
    if cols is not None:
        ncols = max(1, cols)
    elif count <= 3:
        ncols = count
    elif count <= 6:
        ncols = 3
    else:
        ncols = 5
    nrows = int(math.ceil(count / ncols))
    return nrows, ncols


def _shared_clim(samples: Sequence[HeatmapSample]) -> Tuple[float, float]:
    values = np.concatenate([sample.amplitudes.ravel() for sample in samples])
    lo, hi = np.percentile(values, [2.0, 98.0])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo = float(np.min(values))
        hi = float(np.max(values))
        if hi <= lo:
            hi = lo + 1.0
    return float(lo), float(hi)


def _export_image(path: Path, fig, *, dpi: int) -> None:
    """Save a matplotlib figure, converting to WebP when requested."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix.lower()
    if suffix == ".webp":
        from io import BytesIO

        from PIL import Image

        buffer = BytesIO()
        fig.savefig(buffer, dpi=dpi, bbox_inches="tight", facecolor="white")
        buffer.seek(0)
        image = Image.open(buffer).convert("RGB")
        image.save(path, "WEBP", quality=90, method=6)
    else:
        fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
    print(f"Saved: {path}")


def plot_csi_heatmaps(
    samples: Sequence[HeatmapSample],
    *,
    cols: Optional[int] = None,
    shared_scale: bool = False,
    cmap: str = "viridis",
    output: Optional[Path] = None,
    show: bool = True,
    dpi: int = 160,
    figure_title: Optional[str] = None,
    show_subtitles: bool = True,
    publication: bool = False,
    colorbar_label: str = "Amplitude",
) -> None:
    """Render a paper-style grid of CSI amplitude heatmaps."""
    import matplotlib.pyplot as plt

    if publication:
        show_subtitles = False
        if figure_title is None:
            figure_title = "CSI amplitude over time"
        if dpi == 160:
            dpi = 220

    nrows, ncols = _grid_shape(len(samples), cols)
    fig_w = max(3.6 * ncols, 9.0) if publication else max(3.2 * ncols, 8.0)
    fig_h = max(3.0 * nrows, 4.2) if publication else max(2.8 * nrows, 4.0)
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(fig_w, fig_h),
        squeeze=False,
        constrained_layout=True,
    )
    fig.patch.set_facecolor("white")
    clim = _shared_clim(samples) if shared_scale else None
    title_size = 12 if publication else 10
    label_size = 10 if publication else 9

    last_image = None
    for idx, sample in enumerate(samples):
        row, col = divmod(idx, ncols)
        ax = axes[row][col]
        panel = sample.amplitudes.T  # y = subcarrier, x = time
        if clim is None:
            lo, hi = np.percentile(panel, [2.0, 98.0])
            if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
                lo = float(np.min(panel))
                hi = float(np.max(panel))
                if hi <= lo:
                    hi = lo + 1.0
            vmin, vmax = float(lo), float(hi)
        else:
            vmin, vmax = clim

        last_image = ax.imshow(
            panel,
            aspect="auto",
            origin="lower",
            interpolation="nearest",
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
        )
        if show_subtitles and sample.subtitle:
            ax.set_title(f"{sample.title}\n{sample.subtitle}", fontsize=title_size)
        else:
            ax.set_title(sample.title, fontsize=title_size, fontweight="bold")
        ax.set_xlabel("Time (packets)", fontsize=label_size)
        ax.set_ylabel("Subcarrier index", fontsize=label_size)
        ax.tick_params(labelsize=label_size - 1)

    for idx in range(len(samples), nrows * ncols):
        row, col = divmod(idx, ncols)
        axes[row][col].axis("off")

    if figure_title:
        fig.suptitle(figure_title, fontsize=14 if publication else 13)
    if last_image is not None:
        colorbar = fig.colorbar(
            last_image,
            ax=axes.ravel().tolist(),
            shrink=0.82,
            label=colorbar_label,
        )
        colorbar.ax.tick_params(labelsize=label_size - 1)
        colorbar.set_label(colorbar_label, fontsize=label_size)

    if output is not None:
        _export_image(Path(output), fig, dpi=dpi)

    if show:
        show_plot_window(plt)
    else:
        plt.close(fig)


def main(argv: Optional[Iterable[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Plot CSI amplitude heatmaps (time × subcarrier) for dataset samples",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Newest empty / static_presence / motion samples for C6
  python tools/plot_heatmap.py

  # Filter by chip and environment
  python tools/plot_heatmap.py --chip S3 --environment bedroom

  # Explicit files
  python tools/plot_heatmap.py --files data/empty/foo.npz data/motion/bar.npz

  # Save without opening a window
  python tools/plot_heatmap.py --output /tmp/csi_heatmaps.png --no-show

  # Publication-ready export for docs / website
  python tools/plot_heatmap.py --chip C6 --environment living_room --detrend \\
      --publication --output docs/web/assets/images/guides/csi-amplitude-heatmap.webp --no-show
        """,
    )
    parser.add_argument(
        "--chip",
        type=str,
        default="C6",
        help="Chip filter when selecting from dataset_info.json (default: C6)",
    )
    parser.add_argument(
        "--environment",
        type=str,
        default=None,
        help="Optional environment filter (bedroom, living_room, hobby_room)",
    )
    parser.add_argument(
        "--labels",
        type=str,
        default=",".join(DEFAULT_LABELS),
        help=f"Comma-separated labels to plot (default: {','.join(DEFAULT_LABELS)})",
    )
    parser.add_argument(
        "--files",
        nargs="+",
        type=Path,
        default=None,
        help="Explicit NPZ paths; when set, chip/label filters are ignored",
    )
    parser.add_argument(
        "--packets",
        type=int,
        default=400,
        help="Number of contiguous packets per panel (default: 400)",
    )
    parser.add_argument(
        "--offset",
        type=int,
        default=100,
        help="Starting packet index (default: 100)",
    )
    parser.add_argument(
        "--detrend",
        action="store_true",
        help="Subtract per-subcarrier mean to emphasize temporal variation",
    )
    parser.add_argument(
        "--shared-scale",
        action="store_true",
        help="Use one color scale across all panels",
    )
    parser.add_argument(
        "--cols",
        type=int,
        default=None,
        help="Number of subplot columns (default: auto)",
    )
    parser.add_argument(
        "--cmap",
        type=str,
        default="viridis",
        help="Matplotlib colormap (default: viridis)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional PNG/PDF/WebP path to save the figure",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=160,
        help="Output resolution in DPI (default: 160; 220 with --publication)",
    )
    parser.add_argument(
        "--figure-title",
        type=str,
        default=None,
        help="Optional figure title override",
    )
    parser.add_argument(
        "--no-subtitles",
        action="store_true",
        help="Hide per-panel chip/environment subtitles",
    )
    parser.add_argument(
        "--publication",
        action="store_true",
        help="Cleaner layout for docs and the website (implies higher DPI)",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Do not open an interactive plot window",
    )

    args = parser.parse_args(list(argv) if argv is not None else None)
    if args.packets <= 0:
        parser.error("--packets must be > 0")
    if args.offset < 0:
        parser.error("--offset must be >= 0")

    raw_args = list(argv) if argv is not None else sys.argv[1:]
    chip_explicit = "--chip" in raw_args
    chip = args.chip.upper() if (chip_explicit or args.files is None) else None
    if args.files is None and not chip_explicit:
        chip = args.chip.upper()

    print("")
    print("╔═══════════════════════════════════════════════════════╗")
    print("║           CSI Amplitude Heatmap Plotter               ║")
    print("╚═══════════════════════════════════════════════════════╝")

    try:
        if args.files:
            samples = build_samples_from_files(
                args.files,
                packets=args.packets,
                offset=args.offset,
                detrend=args.detrend,
            )
        else:
            labels = _parse_csv_list(args.labels) or list(DEFAULT_LABELS)
            samples = build_samples_from_metadata(
                labels=labels,
                chip=chip,
                environment=args.environment,
                packets=args.packets,
                offset=args.offset,
                detrend=args.detrend,
            )
    except (FileNotFoundError, ValueError) as exc:
        print(f"\nError: {exc}")
        return 1

    print("\nSamples:")
    for sample in samples:
        n_packets, n_sc = sample.amplitudes.shape
        print(f"  {sample.title:16s}  {n_packets:4d} pkt × {n_sc} sc  ({sample.path.name})")

    print("\nGenerating heatmaps...")
    colorbar_label = "Relative amplitude" if args.detrend else "Amplitude"
    figure_title = args.figure_title
    if figure_title is None and args.publication:
        figure_title = "CSI amplitude over time (x = packets, y = subcarrier)"
    plot_csi_heatmaps(
        samples,
        cols=args.cols,
        shared_scale=args.shared_scale,
        cmap=args.cmap,
        output=args.output,
        show=not args.no_show,
        dpi=args.dpi,
        figure_title=figure_title,
        show_subtitles=not args.no_subtitles,
        publication=args.publication,
        colorbar_label=colorbar_label,
    )
    print("Done!\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
