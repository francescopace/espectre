# SPDX-License-Identifier: GPL-3.0-only
# Commercial licensing available under separate agreement; see LICENSING.md.
from pathlib import Path
import tempfile
from unittest.mock import Mock

import pytest

from tools import espectre_traffic_generator as traffic_generator


def test_pid_file_uses_the_platform_temporary_directory() -> None:
    assert traffic_generator.PID_FILE.parent == Path(tempfile.gettempdir())


def test_start_terminates_the_child_when_pid_persistence_fails(
    monkeypatch, tmp_path
) -> None:
    process = Mock(pid=1234)
    missing_parent_pid = tmp_path / "missing" / "espectre_traffic.pid"
    monkeypatch.setattr(traffic_generator, "PID_FILE", missing_parent_pid)
    monkeypatch.setattr(traffic_generator.subprocess, "Popen", Mock(return_value=process))

    with pytest.raises(FileNotFoundError):
        traffic_generator.start()

    process.terminate.assert_called_once_with()
    process.wait.assert_called_once_with(timeout=5)
