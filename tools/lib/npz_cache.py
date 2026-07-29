"""
ESPectre - Shared NPZ Cache

Shared cache helpers for host-side tooling that derives repeated artifacts from
the same dataset NPZ files.

Author: Francesco Pace <francesco.pace@gmail.com>
License: GPLv3
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import threading
import uuid
from collections import OrderedDict
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, MutableMapping, Optional

import numpy as np

from .repo_paths import repo_root

CACHE_LAYOUT_VERSION = 2
FEATURE_MATRIX_ARTIFACT_VERSION = 2
FEATURE_COLUMN_ARTIFACT_VERSION = 1
IDLE_BASELINE_ARTIFACT_VERSION = 2
DETECTOR_REPLAY_ARTIFACT_VERSION = 2

RUNTIME_CACHE_MAX_ENTRIES = 64

_RUNTIME_CACHE: "OrderedDict[tuple[str, str], Any]" = OrderedDict()
_RUNTIME_CACHE_LOCK = threading.RLock()

_SOURCE_DIGEST_CACHE: dict[tuple[str, int, int], str] = {}
_SOURCE_DIGEST_LOCK = threading.RLock()


NPZ_CACHE_DIR_ENV = "ESPECTRE_NPZ_CACHE_DIR"


def npz_cache_dir() -> Path:
    """Return the NPZ cache directory.

    Defaults to the workspace-local `.npz_cache`. `ESPECTRE_NPZ_CACHE_DIR`
    redirects it, which keeps tests off the working cache and allows placing
    artifacts on another volume.
    """
    override = os.environ.get(NPZ_CACHE_DIR_ENV)
    if override:
        return Path(override).expanduser()
    return repo_root() / ".npz_cache"


def _json_safe(value: Any) -> Any:
    """Convert NumPy-heavy nested structures into JSON-safe Python values."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def ensure_npz_cache_dir() -> Path:
    """Create and return the NPZ cache directory."""
    path = npz_cache_dir()
    path.mkdir(parents=True, exist_ok=True)
    return path


def artifact_dir(artifact_name: str) -> Path:
    """Return the cache directory for one artifact type."""
    return ensure_npz_cache_dir() / str(artifact_name)


def _safe_relative_path(path: Path) -> str:
    """Return a stable relative path inside the repo when possible."""
    resolved = Path(path).resolve()
    try:
        return str(resolved.relative_to(repo_root()))
    except ValueError:
        return str(resolved)


def source_content_digest(source_path: str | Path) -> str:
    """Return the SHA-256 of one source NPZ, memoized per stat signature.

    Size and modification time are only a fast path for skipping a rehash of an
    unchanged file; they never reach the manifest. Full hashing of the whole
    capture corpus costs well under a second, which is the price of an identity
    that survives a checkout.
    """
    resolved = Path(source_path).resolve()
    stat = resolved.stat()
    key = (str(resolved), int(stat.st_size), int(stat.st_mtime_ns))
    with _SOURCE_DIGEST_LOCK:
        cached = _SOURCE_DIGEST_CACHE.get(key)
    if cached is not None:
        return cached
    digest = hashlib.sha256()
    with resolved.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    value = digest.hexdigest()
    with _SOURCE_DIGEST_LOCK:
        _SOURCE_DIGEST_CACHE[key] = value
    return value


def source_manifest(source_path: str | Path) -> dict[str, Any]:
    """Return the stable cache identity for one source NPZ.

    The identity is content-based and machine-independent. Modification time is
    excluded because `git checkout` rewrites it, and the absolute path is
    excluded because it differs between a workstation and a CI runner; either
    one would make a transported or restored cache miss on every entry.
    """
    resolved = Path(source_path).resolve()
    return {
        "path": _safe_relative_path(resolved),
        "size": int(resolved.stat().st_size),
        "content_sha256": source_content_digest(resolved),
    }


def resolve_manifest_source(manifest: Mapping[str, Any]) -> Path:
    """Return the current location of one manifest's source capture."""
    stored = Path(str(manifest.get("path", "") or ""))
    return stored if stored.is_absolute() else repo_root() / stored


def artifact_manifest(
    source_path: str | Path,
    *,
    artifact_name: str,
    artifact_version: int,
    parameters: Optional[Mapping[str, Any]] = None,
) -> dict[str, Any]:
    """Build a stable manifest for one derived artifact.

    Parameters are normalized to JSON-safe values so the in-memory manifest and
    its serialized form stay equal. Without this, NumPy scalars fail to digest
    and tuples digest but never match their own round-trip, which turns every
    lookup into a silent miss.
    """
    return {
        "cache_layout_version": CACHE_LAYOUT_VERSION,
        "artifact_name": str(artifact_name),
        "artifact_version": int(artifact_version),
        "source": source_manifest(source_path),
        "parameters": _json_safe(dict(parameters or {})),
    }


def manifest_digest(manifest: Mapping[str, Any]) -> str:
    """Return a stable hex digest for one manifest."""
    payload = json.dumps(
        manifest,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def artifact_cache_path(
    source_path: str | Path,
    *,
    artifact_name: str,
    artifact_version: int,
    parameters: Optional[Mapping[str, Any]] = None,
    extension: str = ".npz",
) -> tuple[dict[str, Any], Path]:
    """Return ``(manifest, cache_path)`` for one derived artifact."""
    manifest = artifact_manifest(
        source_path,
        artifact_name=artifact_name,
        artifact_version=artifact_version,
        parameters=parameters,
    )
    digest = manifest_digest(manifest)
    cache_dir = artifact_dir(str(artifact_name))
    cache_dir.mkdir(parents=True, exist_ok=True)
    return manifest, cache_dir / f"{digest}{extension}"


def runtime_cache_key(
    source_path: str | Path,
    *,
    artifact_name: str,
    artifact_version: int,
    parameters: Optional[Mapping[str, Any]] = None,
) -> tuple[str, str]:
    """Return the stable in-process memo key for one source artifact."""
    manifest = artifact_manifest(
        source_path,
        artifact_name=artifact_name,
        artifact_version=artifact_version,
        parameters=parameters,
    )
    return str(artifact_name), manifest_digest(manifest)


def get_runtime_artifact(
    source_path: str | Path,
    *,
    artifact_name: str,
    artifact_version: int,
    builder: Callable[[], Any],
    parameters: Optional[Mapping[str, Any]] = None,
) -> Any:
    """Return one in-process memoized artifact for one NPZ source."""
    key = runtime_cache_key(
        source_path,
        artifact_name=artifact_name,
        artifact_version=artifact_version,
        parameters=parameters,
    )
    with _RUNTIME_CACHE_LOCK:
        cached = _RUNTIME_CACHE.pop(key, None)
        if cached is not None:
            _RUNTIME_CACHE[key] = cached
            return cached
    built = builder()
    with _RUNTIME_CACHE_LOCK:
        _RUNTIME_CACHE[key] = built
        while len(_RUNTIME_CACHE) > RUNTIME_CACHE_MAX_ENTRIES:
            _RUNTIME_CACHE.popitem(last=False)
    return built


def clear_runtime_artifacts(*artifact_names: str) -> None:
    """Drop in-process memoized artifacts, optionally filtered by type."""
    selected = {str(name) for name in artifact_names if name}
    with _RUNTIME_CACHE_LOCK:
        if not selected:
            _RUNTIME_CACHE.clear()
            return
        for key in list(_RUNTIME_CACHE):
            if key[0] in selected:
                _RUNTIME_CACHE.pop(key, None)


def clear_persisted_artifacts(*artifact_names: str) -> None:
    """Delete persisted cache files for the selected artifact types."""
    cache_root = npz_cache_dir()
    if not cache_root.exists():
        return
    selected = [str(name) for name in artifact_names if name]
    if not selected:
        shutil.rmtree(cache_root)
        return
    for artifact_name in selected:
        shutil.rmtree(cache_root / artifact_name, ignore_errors=True)


def read_artifact_manifest(artifact_path: str | Path) -> Optional[dict[str, Any]]:
    """Return one persisted artifact's manifest, or None when unreadable."""
    try:
        with np.load(Path(artifact_path), allow_pickle=False) as data:
            return json.loads(str(np.asarray(data["manifest_json"]).item()))
    except Exception:
        return None


def prune_persisted_artifacts(*artifact_names: str) -> dict[str, int]:
    """Delete unreachable persisted artifacts and return what was removed.

    An artifact is unreachable when it is unreadable, when its source capture is
    gone, or when the source no longer matches the recorded identity. Nothing
    else can ever hit these entries again, so they are pure accumulation.
    """
    cache_root = npz_cache_dir()
    removed = {"unreadable": 0, "missing_source": 0, "stale_source": 0}
    if not cache_root.exists():
        return removed
    selected = [str(name) for name in artifact_names if name]
    directories = (
        [cache_root / name for name in selected]
        if selected
        else [entry for entry in cache_root.iterdir() if entry.is_dir()]
    )
    for directory in directories:
        if not directory.is_dir():
            continue
        for artifact_path in sorted(directory.glob("*.npz")):
            if artifact_path.name.endswith(".tmp.npz"):
                continue
            manifest = read_artifact_manifest(artifact_path)
            if manifest is None:
                reason = "unreadable"
            else:
                source = resolve_manifest_source(manifest.get("source", {}))
                if not source.exists():
                    reason = "missing_source"
                elif source_manifest(source) != manifest.get("source"):
                    reason = "stale_source"
                else:
                    continue
            artifact_path.unlink(missing_ok=True)
            removed[reason] += 1
    return removed


def load_npz_artifact(
    source_path: str | Path,
    *,
    artifact_name: str,
    artifact_version: int,
    parameters: Optional[Mapping[str, Any]] = None,
) -> Optional[dict[str, np.ndarray]]:
    """Load one persisted NPZ artifact when the manifest still matches."""
    manifest, artifact_path = artifact_cache_path(
        source_path,
        artifact_name=artifact_name,
        artifact_version=artifact_version,
        parameters=parameters,
    )
    if not artifact_path.exists():
        return None
    try:
        with np.load(artifact_path, allow_pickle=False) as data:
            cached_manifest = json.loads(str(np.asarray(data["manifest_json"]).item()))
            if cached_manifest != manifest:
                return None
            return {key: np.asarray(data[key]) for key in data.files if key != "manifest_json"}
    except Exception:
        return None


def save_npz_artifact(
    source_path: str | Path,
    *,
    artifact_name: str,
    artifact_version: int,
    payload: Mapping[str, Any],
    parameters: Optional[Mapping[str, Any]] = None,
) -> Path:
    """Persist one NPZ artifact atomically and return its path."""
    manifest, artifact_path = artifact_cache_path(
        source_path,
        artifact_name=artifact_name,
        artifact_version=artifact_version,
        parameters=parameters,
    )
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    serializable: MutableMapping[str, np.ndarray] = {
        "manifest_json": np.asarray(json.dumps(manifest, sort_keys=True)),
    }
    for key, value in payload.items():
        serializable[str(key)] = np.asarray(value)
    # Stage under a writer-private name: concurrent producers of the same
    # artifact (pytest workers, or training alongside validation) would
    # otherwise interleave writes into one file and publish a truncated archive.
    tmp_path = artifact_path.parent / f"{artifact_path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp.npz"
    try:
        np.savez(tmp_path, **serializable)
        os.replace(tmp_path, artifact_path)
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise
    return artifact_path


def feature_matrix_base_parameters(
    *,
    window_size: int,
    subcarriers: Any,
    enable_lowpass: bool = False,
    lowpass_cutoff: float = 0.0,
    enable_hampel: bool = False,
    hampel_window: int = 0,
    hampel_threshold: float = 0.0,
    packet_augmentation: Optional[Mapping[str, Any]] = None,
    augmentation_seed: Optional[int] = None,
) -> dict[str, Any]:
    """Return the shared extraction identity for one per-file feature cache.

    Every producer of reusable feature artifacts must build its shared
    extraction parameters here so two tools computing the same columns cannot
    disagree on the key. The arguments must describe the extraction that
    actually runs, not the intended or default one.

    Inactive filter settings are normalized away: two callers that both disable
    a filter must not fragment the cache over a cutoff or window value that
    neither of them applied.
    """
    parameters: dict[str, Any] = {
        "window_size": int(window_size),
        "subcarriers": [int(sc) for sc in subcarriers],
        "enable_lowpass": bool(enable_lowpass),
        "enable_hampel": bool(enable_hampel),
        "packet_augmentation": dict(sorted((packet_augmentation or {}).items())),
        "augmentation_seed": None if augmentation_seed is None else int(augmentation_seed),
    }
    if parameters["enable_lowpass"]:
        parameters["lowpass_cutoff"] = float(lowpass_cutoff)
    if parameters["enable_hampel"]:
        parameters["hampel_window"] = int(hampel_window)
        parameters["hampel_threshold"] = float(hampel_threshold)
    return parameters


def feature_matrix_parameters(
    *,
    feature_names: Any,
    window_size: int,
    subcarriers: Any,
    enable_lowpass: bool = False,
    lowpass_cutoff: float = 0.0,
    enable_hampel: bool = False,
    hampel_window: int = 0,
    hampel_threshold: float = 0.0,
    packet_augmentation: Optional[Mapping[str, Any]] = None,
    augmentation_seed: Optional[int] = None,
) -> dict[str, Any]:
    """Return the legacy whole-matrix identity for one feature request."""
    parameters = feature_matrix_base_parameters(
        window_size=window_size,
        subcarriers=subcarriers,
        enable_lowpass=enable_lowpass,
        lowpass_cutoff=lowpass_cutoff,
        enable_hampel=enable_hampel,
        hampel_window=hampel_window,
        hampel_threshold=hampel_threshold,
        packet_augmentation=packet_augmentation,
        augmentation_seed=augmentation_seed,
    )
    parameters["feature_names"] = [str(name) for name in feature_names]
    return parameters


def feature_column_parameters(
    *,
    base_parameters: Mapping[str, Any],
    feature_name: str,
) -> dict[str, Any]:
    """Return the persisted identity for one reusable feature column."""
    return {
        "feature_cache": _json_safe(dict(base_parameters)),
        "feature_name": str(feature_name),
    }


def load_feature_column_artifact(
    source_path: str | Path,
    *,
    base_parameters: Mapping[str, Any],
    feature_name: str,
) -> Optional[np.ndarray]:
    """Load one persisted per-file feature column."""
    payload = load_npz_artifact(
        source_path,
        artifact_name="feature_column",
        artifact_version=FEATURE_COLUMN_ARTIFACT_VERSION,
        parameters=feature_column_parameters(
            base_parameters=base_parameters,
            feature_name=feature_name,
        ),
    )
    if payload is None:
        return None
    column = payload.get("column")
    if column is None:
        return None
    return np.asarray(column, dtype=np.float32)


def save_feature_column_artifact(
    source_path: str | Path,
    *,
    base_parameters: Mapping[str, Any],
    feature_name: str,
    column: np.ndarray,
) -> Path:
    """Persist one reusable per-file feature column."""
    return save_npz_artifact(
        source_path,
        artifact_name="feature_column",
        artifact_version=FEATURE_COLUMN_ARTIFACT_VERSION,
        parameters=feature_column_parameters(
            base_parameters=base_parameters,
            feature_name=feature_name,
        ),
        payload={"column": np.asarray(column, dtype=np.float32)},
    )


def load_feature_columns(
    source_path: str | Path,
    *,
    base_parameters: Mapping[str, Any],
    feature_names: Any,
) -> dict[str, np.ndarray]:
    """Load any persisted columns available for the requested features."""
    columns: dict[str, np.ndarray] = {}
    for feature_name in feature_names:
        key = str(feature_name)
        column = load_feature_column_artifact(
            source_path,
            base_parameters=base_parameters,
            feature_name=key,
        )
        if column is not None:
            columns[key] = np.asarray(column, dtype=np.float32)
    return columns


def save_feature_columns(
    source_path: str | Path,
    *,
    base_parameters: Mapping[str, Any],
    feature_names: Any,
    X: np.ndarray,
) -> None:
    """Persist reusable per-feature columns for one feature matrix."""
    matrix = np.asarray(X, dtype=np.float32)
    feature_name_list = [str(name) for name in feature_names]
    if matrix.ndim != 2 or matrix.shape[1] != len(feature_name_list):
        raise ValueError("Feature matrix shape does not match feature name count")
    for column_index, feature_name in enumerate(feature_name_list):
        save_feature_column_artifact(
            source_path,
            base_parameters=base_parameters,
            feature_name=feature_name,
            column=matrix[:, column_index],
        )


def assemble_feature_matrix(
    columns: Mapping[str, np.ndarray],
    feature_names: Any,
) -> tuple[np.ndarray, list[str]]:
    """Assemble one feature matrix in request order from cached columns."""
    ordered_names = [str(name) for name in feature_names]
    if not ordered_names:
        return np.empty((0, 0), dtype=np.float32), []
    missing = [name for name in ordered_names if name not in columns]
    if missing:
        raise KeyError(f"Missing cached feature columns: {missing}")
    ordered_columns = [np.asarray(columns[name], dtype=np.float32) for name in ordered_names]
    lengths = {int(column.shape[0]) for column in ordered_columns}
    if len(lengths) != 1:
        raise ValueError("Cached feature columns do not share a row count")
    matrix = np.column_stack(ordered_columns).astype(np.float32, copy=False)
    return matrix, ordered_names


def load_feature_matrix_artifact(
    source_path: str | Path,
    *,
    parameters: Mapping[str, Any],
) -> Optional[dict[str, Any]]:
    """Load one persisted per-file feature matrix artifact."""
    payload = load_npz_artifact(
        source_path,
        artifact_name="feature_matrix",
        artifact_version=FEATURE_MATRIX_ARTIFACT_VERSION,
        parameters=parameters,
    )
    if payload is None:
        return None
    return {
        "X": np.asarray(payload["X"], dtype=np.float32),
        "feature_names": np.asarray(payload["feature_names"]).astype(str).tolist(),
    }


def save_feature_matrix_artifact(
    source_path: str | Path,
    *,
    parameters: Mapping[str, Any],
    X: np.ndarray,
    feature_names: list[str] | tuple[str, ...],
) -> Path:
    """Persist one per-file feature matrix artifact."""
    return save_npz_artifact(
        source_path,
        artifact_name="feature_matrix",
        artifact_version=FEATURE_MATRIX_ARTIFACT_VERSION,
        parameters=parameters,
        payload={
            "X": np.asarray(X, dtype=np.float32),
            "feature_names": np.asarray(list(feature_names)),
        },
    )


def load_idle_baseline_artifact(
    source_path: str | Path,
    *,
    parameters: Mapping[str, Any],
) -> Optional[dict[str, Any]]:
    """Load one persisted idle-baseline artifact."""
    payload = load_npz_artifact(
        source_path,
        artifact_name="idle_baseline",
        artifact_version=IDLE_BASELINE_ARTIFACT_VERSION,
        parameters=parameters,
    )
    if payload is None:
        return None
    baseline_json = payload.get("baseline_json")
    if baseline_json is None:
        return None
    baseline = json.loads(str(np.asarray(baseline_json).item()))
    median_rssi = payload.get("median_rssi_dbm")
    median_rssi_value = None
    if median_rssi is not None:
        median_rssi_value = float(np.asarray(median_rssi).item())
    return {
        "baseline": baseline,
        "median_rssi_dbm": median_rssi_value,
    }


def save_idle_baseline_artifact(
    source_path: str | Path,
    *,
    parameters: Mapping[str, Any],
    baseline: Mapping[str, Any],
    median_rssi_dbm: Optional[float],
) -> Path:
    """Persist one idle-baseline artifact."""
    payload: dict[str, Any] = {
        "baseline_json": np.asarray(json.dumps(_json_safe(dict(baseline)), sort_keys=True)),
    }
    if median_rssi_dbm is not None:
        payload["median_rssi_dbm"] = np.asarray(float(median_rssi_dbm))
    return save_npz_artifact(
        source_path,
        artifact_name="idle_baseline",
        artifact_version=IDLE_BASELINE_ARTIFACT_VERSION,
        parameters=parameters,
        payload=payload,
    )


def detector_replay_parameters(
    *,
    replay_kind: str,
    selected_subcarriers: Any,
    window_size: Optional[int] = None,
    threshold: Optional[float] = None,
    feature_names: Any = (),
    secondary_source: Optional[str | Path] = None,
) -> dict[str, Any]:
    """Return the persisted identity for one detector replay result."""
    parameters: dict[str, Any] = {
        "replay_kind": str(replay_kind),
        "selected_subcarriers": [int(sc) for sc in selected_subcarriers],
        "feature_names": [str(name) for name in feature_names],
    }
    if window_size is not None:
        parameters["window_size"] = int(window_size)
    if threshold is not None:
        parameters["threshold"] = float(threshold)
    if secondary_source is not None:
        parameters["secondary_source"] = source_manifest(secondary_source)
    return parameters


def load_detector_replay_artifact(
    source_path: str | Path,
    *,
    parameters: Mapping[str, Any],
) -> Optional[dict[str, Any]]:
    """Load one persisted detector replay payload."""
    payload = load_npz_artifact(
        source_path,
        artifact_name="detector_replay",
        artifact_version=DETECTOR_REPLAY_ARTIFACT_VERSION,
        parameters=parameters,
    )
    if payload is None:
        return None
    result_json = payload.get("result_json")
    if result_json is None:
        return None
    return json.loads(str(np.asarray(result_json).item()))


def save_detector_replay_artifact(
    source_path: str | Path,
    *,
    parameters: Mapping[str, Any],
    result: Mapping[str, Any],
) -> Path:
    """Persist one detector replay payload."""
    return save_npz_artifact(
        source_path,
        artifact_name="detector_replay",
        artifact_version=DETECTOR_REPLAY_ARTIFACT_VERSION,
        parameters=parameters,
        payload={
            "result_json": np.asarray(
                json.dumps(_json_safe(dict(result)), sort_keys=True)
            ),
        },
    )
