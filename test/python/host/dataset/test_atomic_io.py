# SPDX-License-Identifier: GPL-3.0-only
# Commercial licensing available under separate agreement; see LICENSING.md.
import os

import numpy as np
import pytest

from tools.lib import atomic_io


def test_atomic_write_set_rolls_back_existing_and_new_files(monkeypatch, tmp_path):
    existing = tmp_path / "existing.txt"
    created = tmp_path / "created.txt"
    existing.write_bytes(b"before")
    real_replace = os.replace
    commit_calls = 0

    def fail_second_commit(source, destination):
        nonlocal commit_calls
        if str(source).endswith(".set.tmp"):
            commit_calls += 1
            if commit_calls == 2:
                raise OSError("simulated commit failure")
        return real_replace(source, destination)

    monkeypatch.setattr(atomic_io.os, "replace", fail_second_commit)

    with pytest.raises(OSError, match="simulated commit failure"):
        atomic_io.atomic_write_set({
            existing: b"after",
            created: b"new",
        })

    assert existing.read_bytes() == b"before"
    assert not created.exists()


def test_atomic_savez_compressed_publishes_loadable_archive(tmp_path):
    destination = tmp_path / "sample.npz"

    atomic_io.atomic_savez_compressed(
        destination,
        {"values": np.arange(8, dtype=np.int8)},
    )

    with np.load(destination, allow_pickle=False) as archive:
        np.testing.assert_array_equal(archive["values"], np.arange(8, dtype=np.int8))
    assert not list(tmp_path.glob("*.tmp.npz"))
