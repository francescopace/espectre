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

from .repo_paths import cpp_core_dir, python_src_dir, repo_root

CACHE_LAYOUT_VERSION = 2
CLASSIC_REPLAY_ROW_ARTIFACT_VERSION = 1
ML_REPLAY_ROW_ARTIFACT_VERSION = 3
ML_TRAINING_AUGMENTATION_ROW_ARTIFACT_VERSION = 1

CURRENT_ARTIFACT_VERSIONS = {
    "classic_replay_rows": CLASSIC_REPLAY_ROW_ARTIFACT_VERSION,
    "ml_replay_rows": ML_REPLAY_ROW_ARTIFACT_VERSION,
    "ml_training_augmentation_rows": ML_TRAINING_AUGMENTATION_ROW_ARTIFACT_VERSION,
}
OBSOLETE_ARTIFACT_NAMES = {
    "feature_matrix",
    "feature_column",
    "idle_baseline",
}

RUNTIME_CACHE_MAX_ENTRIES = 64

_RUNTIME_CACHE: "OrderedDict[tuple[str, str], Any]" = OrderedDict()
_RUNTIME_CACHE_LOCK = threading.RLock()

_SOURCE_DIGEST_CACHE: dict[tuple[str, int, int, int], str] = {}
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


def artifact_dir(artifact_name: str) -> Path:
    """Return the cache directory path for one artifact type without creating it."""
    return npz_cache_dir() / str(artifact_name)


def _safe_relative_path(path: Path) -> str:
    """Return a stable relative path inside the repo when possible."""
    resolved = Path(path).resolve()
    try:
        return str(resolved.relative_to(repo_root()))
    except ValueError:
        return str(resolved)


def _source_digest_memo_enabled() -> bool:
    """Return whether stat metadata can safely key the digest memo."""
    return os.name != "nt"


def source_content_digest(source_path: str | Path) -> str:
    """Return the SHA-256 of one source NPZ, memoized per stat signature.

    Size, modification time, and change time are only a fast path for skipping
    a rehash of an unchanged file; they never reach the manifest. POSIX change
    time catches a same-size rewrite even when modification time is deliberately
    restored. Windows exposes creation time through ``st_ctime_ns``, so it
    always rehashes instead of trusting an unsafe memo key. Full hashing of the
    whole capture corpus costs well under a second, which is the price of an
    identity that survives a checkout.
    """
    resolved = Path(source_path).resolve()
    stat = resolved.stat()
    key = (
        str(resolved),
        int(stat.st_size),
        int(stat.st_mtime_ns),
        int(stat.st_ctime_ns),
    )
    memo_enabled = _source_digest_memo_enabled()
    if memo_enabled:
        with _SOURCE_DIGEST_LOCK:
            cached = _SOURCE_DIGEST_CACHE.get(key)
        if cached is not None:
            return cached
    digest = hashlib.sha256()
    with resolved.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    value = digest.hexdigest()
    if memo_enabled:
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


def _ml_feature_source_manifests() -> dict[str, Any]:
    """Return stable identities for the time-aware ML feature extractor."""
    manifests: dict[str, Any] = {}
    sources = {
        "python_config": python_src_dir() / "config.py",
        "python_csi_features": python_src_dir() / "csi_features.py",
        "python_device_utils": python_src_dir() / "device_utils.py",
        "python_filters": python_src_dir() / "filters.py",
        "python_ml_detector": python_src_dir() / "ml_detector.py",
        "python_ml_feature_trackers": python_src_dir() / "ml_feature_trackers.py",
        "python_runtime_policy": python_src_dir() / "runtime_policy.py",
        "python_segmentation": python_src_dir() / "segmentation.py",
        "host_csi_io": repo_root() / "tools" / "lib" / "csi_io.py",
        "host_dataset_metadata": repo_root() / "tools" / "lib" / "dataset_metadata.py",
        "host_ml_replay": repo_root() / "tools" / "lib" / "performance_report.py",
    }
    for name, path in sources.items():
        if path.exists():
            manifests[name] = source_manifest(path)
    return manifests


def _replay_policy_source_manifests() -> dict[str, Any]:
    """Return identities for shared replay timing and calibration policy."""
    manifests: dict[str, Any] = {}
    sources = {
        "python_runtime_policy": python_src_dir() / "runtime_policy.py",
        "host_dataset_metadata": repo_root() / "tools" / "lib" / "dataset_metadata.py",
        "host_classic_replay": repo_root() / "tools" / "lib" / "performance_report.py",
    }
    for name, path in sources.items():
        if path.exists():
            manifests[name] = source_manifest(path)
    return manifests


def _classic_detector_source_manifests() -> dict[str, Any]:
    """Return stable identities for the current Classic detector sources."""
    manifests: dict[str, Any] = {}
    sources = {
        "python_classic_detector": python_src_dir() / "classic_detector.py",
        "cpp_classic_detector_header": cpp_core_dir() / "classic_detector.h",
        "cpp_classic_detector_impl": cpp_core_dir() / "classic_detector.cpp",
    }
    for name, path in sources.items():
        if path.exists():
            manifests[name] = source_manifest(path)
    return manifests


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
    removed = {
        "unreadable": 0,
        "missing_source": 0,
        "stale_source": 0,
        "obsolete_artifact": 0,
        "obsolete_version": 0,
    }
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
            elif str(manifest.get("artifact_name", "")) in OBSOLETE_ARTIFACT_NAMES:
                reason = "obsolete_artifact"
            elif (
                str(manifest.get("artifact_name", "")) in CURRENT_ARTIFACT_VERSIONS
                and (
                    int(manifest.get("cache_layout_version", -1)) != CACHE_LAYOUT_VERSION
                    or int(manifest.get("artifact_version", -1))
                    != CURRENT_ARTIFACT_VERSIONS[str(manifest["artifact_name"])]
                )
            ):
                reason = "obsolete_version"
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
        try:
            directory.rmdir()
        except OSError:
            pass
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


def classic_replay_row_parameters(
    *,
    replay_kind: str,
    selected_subcarriers: Any,
    timing: Mapping[str, Any],
    replay_interval_us: int,
    warmup_packets: int,
    secondary_source: Optional[str | Path] = None,
    replay_provenance: Optional[Mapping[str, Any]] = None,
) -> dict[str, Any]:
    """Return the identity for one time-aware Classic feature-row replay."""
    parameters: dict[str, Any] = {
        "artifact_version": CLASSIC_REPLAY_ROW_ARTIFACT_VERSION,
        "replay_kind": str(replay_kind),
        "selected_subcarriers": [int(sc) for sc in selected_subcarriers],
        "feature_names": ["turb_autocorr", "chan_freq_coh_curve_std"],
        "timing": {str(key): int(value) for key, value in timing.items()},
        "replay_interval_us": int(replay_interval_us),
        "warmup_packets": int(warmup_packets),
        "replay_policy_sources": _replay_policy_source_manifests(),
        "classic_sources": _classic_detector_source_manifests(),
    }
    if secondary_source is not None:
        parameters["secondary_source"] = source_manifest(secondary_source)
    if replay_provenance is not None:
        parameters["replay_provenance"] = _json_safe(dict(replay_provenance))
    return parameters


def ml_replay_row_parameters(
    *,
    selected_subcarriers: Any,
    window_size: int,
    feature_names: Any,
    stream_provenance: Optional[Mapping[str, Any]] = None,
) -> dict[str, Any]:
    """Return the identity for the canonical time-aware ML row stream.

    The persisted artifact is independent of the consumer's sampling contract:
    replay-tick rows are a projection of the ready-packet stream. Numeric model
    weights are intentionally excluded because feature extraction does not use
    them. Deterministic input transforms, such as packet augmentation, must
    provide their own provenance so they cannot alias the raw capture rows.
    """
    parameters = {
        "artifact_version": ML_REPLAY_ROW_ARTIFACT_VERSION,
        "sample_contract": "time_aware_stream_rows_v1",
        "selected_subcarriers": [int(sc) for sc in selected_subcarriers],
        "window_size": int(window_size),
        "feature_names": [str(name) for name in feature_names],
        "feature_sources": _ml_feature_source_manifests(),
    }
    if stream_provenance is not None:
        parameters["stream_provenance"] = _json_safe(dict(stream_provenance))
    return parameters


def ml_training_augmentation_row_parameters(
    *,
    selected_subcarriers: Any,
    feature_names: Any,
    stream_provenance: Mapping[str, Any],
) -> dict[str, Any]:
    """Return the identity for one deterministic mixed augmentation row set.

    Unlike a single replay stream, a cached mix can contain views with distinct
    effective window sizes. Its identity therefore follows the source capture,
    requested feature schema, feature implementation, and complete mixing
    provenance instead of pretending that the result has one window size.
    """
    return {
        "artifact_version": ML_TRAINING_AUGMENTATION_ROW_ARTIFACT_VERSION,
        "sample_contract": "training_augmentation_row_mix_v1",
        "selected_subcarriers": [int(sc) for sc in selected_subcarriers],
        "feature_names": [str(name) for name in feature_names],
        "feature_sources": _ml_feature_source_manifests(),
        "stream_provenance": _json_safe(dict(stream_provenance)),
    }


def load_ml_training_augmentation_row_artifact(
    source_path: str | Path,
    *,
    parameters: Mapping[str, Any],
) -> Optional[dict[str, Any]]:
    """Load one persisted deterministic mixed augmentation row set."""
    payload = load_npz_artifact(
        source_path,
        artifact_name="ml_training_augmentation_rows",
        artifact_version=ML_TRAINING_AUGMENTATION_ROW_ARTIFACT_VERSION,
        parameters=parameters,
    )
    if payload is None:
        return None
    return {
        "X": np.asarray(payload.get("X", np.empty((0, 0))), dtype=np.float32),
        "feature_names": np.asarray(
            payload.get("feature_names", np.empty(0))
        ).astype(str).tolist(),
        "packet_index": np.asarray(
            payload.get("packet_index", np.empty(0)), dtype=np.int32
        ),
        "evaluation_index": np.asarray(
            payload.get("evaluation_index", np.empty(0)), dtype=np.int32
        ),
        "reset_index": np.asarray(
            payload.get("reset_index", np.empty(0)), dtype=np.int32
        ),
        "evaluation_due": np.asarray(
            payload.get("evaluation_due", np.empty(0)), dtype=bool
        ),
    }


def save_ml_training_augmentation_row_artifact(
    source_path: str | Path,
    *,
    parameters: Mapping[str, Any],
    rows: Mapping[str, Any],
) -> Path:
    """Persist one deterministic mixed augmentation row set."""
    return save_npz_artifact(
        source_path,
        artifact_name="ml_training_augmentation_rows",
        artifact_version=ML_TRAINING_AUGMENTATION_ROW_ARTIFACT_VERSION,
        parameters=parameters,
        payload={
            "X": np.asarray(rows.get("X", np.empty((0, 0))), dtype=np.float32),
            "feature_names": np.asarray(
                [str(name) for name in rows.get("feature_names", ())]
            ),
            "packet_index": np.asarray(
                rows.get("packet_index", np.empty(0)), dtype=np.int32
            ),
            "evaluation_index": np.asarray(
                rows.get("evaluation_index", np.empty(0)), dtype=np.int32
            ),
            "reset_index": np.asarray(
                rows.get("reset_index", np.empty(0)), dtype=np.int32
            ),
            "evaluation_due": np.asarray(
                rows.get("evaluation_due", np.empty(0)), dtype=bool
            ),
        },
    )


def load_ml_replay_row_artifact(
    source_path: str | Path,
    *,
    parameters: Mapping[str, Any],
) -> Optional[dict[str, Any]]:
    """Load one persisted ML replay-row artifact."""
    payload = load_npz_artifact(
        source_path,
        artifact_name="ml_replay_rows",
        artifact_version=ML_REPLAY_ROW_ARTIFACT_VERSION,
        parameters=parameters,
    )
    if payload is None:
        return None
    return {
        "X": np.asarray(payload.get("X", np.empty((0, 0))), dtype=np.float32),
        "feature_names": np.asarray(
            payload.get("feature_names", np.empty(0))
        ).astype(str).tolist(),
        "packet_index": np.asarray(
            payload.get("packet_index", np.empty(0)), dtype=np.int32
        ),
        "evaluation_index": np.asarray(
            payload.get("evaluation_index", np.empty(0)), dtype=np.int32
        ),
        "reset_index": np.asarray(
            payload.get("reset_index", np.empty(0)), dtype=np.int32
        ),
        "evaluation_due": np.asarray(
            payload.get("evaluation_due", np.empty(0)), dtype=bool
        ),
    }


def save_ml_replay_row_artifact(
    source_path: str | Path,
    *,
    parameters: Mapping[str, Any],
    X: np.ndarray,
    feature_names: Any,
    packet_index: np.ndarray,
    evaluation_index: np.ndarray,
    reset_index: np.ndarray,
    evaluation_due: np.ndarray,
) -> Path:
    """Persist one canonical ML replay-row artifact."""
    return save_npz_artifact(
        source_path,
        artifact_name="ml_replay_rows",
        artifact_version=ML_REPLAY_ROW_ARTIFACT_VERSION,
        parameters=parameters,
        payload={
            "X": np.asarray(X, dtype=np.float32),
            "feature_names": np.asarray([str(name) for name in feature_names]),
            "packet_index": np.asarray(packet_index, dtype=np.int32),
            "evaluation_index": np.asarray(evaluation_index, dtype=np.int32),
            "reset_index": np.asarray(reset_index, dtype=np.int32),
            "evaluation_due": np.asarray(evaluation_due, dtype=bool),
        },
    )


def load_classic_replay_row_artifact(
    source_path: str | Path,
    *,
    parameters: Mapping[str, Any],
) -> Optional[dict[str, Any]]:
    """Load one persisted time-aware Classic replay-row artifact."""
    payload = load_npz_artifact(
        source_path,
        artifact_name="classic_replay_rows",
        artifact_version=CLASSIC_REPLAY_ROW_ARTIFACT_VERSION,
        parameters=parameters,
    )
    if payload is None:
        return None
    rows: dict[str, Any] = {}
    for phase in ("calibration", "static", "motion"):
        prefix = f"{phase}_"
        rows[phase] = {
            "X": np.asarray(
                payload.get(prefix + "X", np.empty((0, 2))), dtype=np.float64
            ),
            "ready": np.asarray(
                payload.get(prefix + "ready", np.empty(0)), dtype=bool
            ),
            "eligible": np.asarray(
                payload.get(prefix + "eligible", np.empty(0)), dtype=bool
            ),
            "packet_index": np.asarray(
                payload.get(prefix + "packet_index", np.empty(0)), dtype=np.int32
            ),
            "packet_weight": np.asarray(
                payload.get(prefix + "packet_weight", np.empty(0)), dtype=np.int32
            ),
            "reset_index": np.asarray(
                payload.get(prefix + "reset_index", np.empty(0)), dtype=np.int32
            ),
        }
    return rows


def save_classic_replay_row_artifact(
    source_path: str | Path,
    *,
    parameters: Mapping[str, Any],
    rows: Mapping[str, Mapping[str, Any]],
) -> Path:
    """Persist one canonical time-aware Classic replay-row artifact."""
    payload: dict[str, np.ndarray] = {}
    for phase in ("calibration", "static", "motion"):
        phase_rows = rows.get(phase, {})
        prefix = f"{phase}_"
        payload[prefix + "X"] = np.asarray(
            phase_rows.get("X", np.empty((0, 2))), dtype=np.float64
        )
        payload[prefix + "ready"] = np.asarray(
            phase_rows.get("ready", np.empty(0)), dtype=bool
        )
        payload[prefix + "eligible"] = np.asarray(
            phase_rows.get("eligible", np.empty(0)), dtype=bool
        )
        payload[prefix + "packet_index"] = np.asarray(
            phase_rows.get("packet_index", np.empty(0)), dtype=np.int32
        )
        payload[prefix + "packet_weight"] = np.asarray(
            phase_rows.get("packet_weight", np.empty(0)), dtype=np.int32
        )
        payload[prefix + "reset_index"] = np.asarray(
            phase_rows.get("reset_index", np.empty(0)), dtype=np.int32
        )
    return save_npz_artifact(
        source_path,
        artifact_name="classic_replay_rows",
        artifact_version=CLASSIC_REPLAY_ROW_ARTIFACT_VERSION,
        parameters=parameters,
        payload=payload,
    )
