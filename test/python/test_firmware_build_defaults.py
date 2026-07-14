from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
IDF_FRONTENDS = ("native", "matter", "streamer")


@pytest.mark.parametrize("frontend", IDF_FRONTENDS)
def test_idf_frontend_defaults_optimize_for_size(frontend):
    defaults = (
        REPO_ROOT / "src" / "cpp" / "frontend" / frontend / "app" / "sdkconfig.defaults"
    ).read_text(encoding="utf-8")

    assert "CONFIG_COMPILER_OPTIMIZATION_SIZE=y" in defaults
    assert "CONFIG_COMPILER_OPTIMIZATION_DEBUG=y" not in defaults
    assert "CONFIG_COMPILER_OPTIMIZATION_PERF=y" not in defaults
