# SPDX-License-Identifier: GPL-3.0-only
# Commercial licensing available under separate agreement; see LICENSING.md.
"""Enforce ESPectre-owned FreeRTOS task-priority build policy."""

from __future__ import annotations

import re
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
CPP_ROOT = REPO_ROOT / "src" / "cpp"
SHARED_KCONFIG = CPP_ROOT / "runtime" / "esp_idf" / "espectre_config" / "Kconfig.projbuild"
NATIVE_KCONFIG = CPP_ROOT / "frontend" / "native" / "espectre" / "Kconfig.projbuild"
SCHEDULING_HEADER = CPP_ROOT / "runtime" / "esp_idf" / "task_scheduling_config.h"
MICRO_ROOT = REPO_ROOT / "src" / "python" / "micro_espectre"
MICRO_KCONFIG = (
    MICRO_ROOT / "firmware" / "components" / "espectre_core" / "Kconfig.projbuild"
)


def _config_block(path: Path, symbol: str) -> str:
    content = path.read_text(encoding="utf-8")
    match = re.search(
        rf"(?ms)^\s*config {re.escape(symbol)}\s*$\n"
        rf"(?P<body>.*?)(?=^\s*(?:config|choice|menu|endchoice|endmenu)\b|\Z)",
        content,
    )
    assert match is not None, f"missing Kconfig symbol {symbol} in {path}"
    return match.group("body")


def test_shared_task_priorities_have_safe_advanced_defaults() -> None:
    content = SHARED_KCONFIG.read_text(encoding="utf-8")
    assert 'menu "Advanced task scheduling"' in content
    expected = {
        "ESPECTRE_DIRECT_HTTPD_TASK_PRIORITY": 1,
        "ESPECTRE_DIRECT_WORKER_TASK_PRIORITY": 2,
        "ESPECTRE_RAW_WORKER_TASK_PRIORITY": 3,
        "ESPECTRE_TRAFFIC_TASK_PRIORITY": 1,
    }
    header = SCHEDULING_HEADER.read_text(encoding="utf-8")
    for symbol, default in expected.items():
        block = _config_block(SHARED_KCONFIG, symbol)
        assert re.search(r"(?m)^\s*range 1 10\s*$", block)
        assert re.search(rf"(?m)^\s*default {default}\s*$", block)
        assert f"#define CONFIG_{symbol} {default}" in header


def test_native_loop_priority_is_frontend_owned() -> None:
    block = _config_block(NATIVE_KCONFIG, "ESPECTRE_NATIVE_LOOP_TASK_PRIORITY")
    assert re.search(r"(?m)^\s*range 1 10\s*$", block)
    assert re.search(r"(?m)^\s*default 5\s*$", block)
    assert "#define CONFIG_ESPECTRE_NATIVE_LOOP_TASK_PRIORITY 5" in (
        SCHEDULING_HEADER.read_text(encoding="utf-8")
    )
    assert "ESPECTRE_NATIVE_LOOP_TASK_PRIORITY" not in SHARED_KCONFIG.read_text(
        encoding="utf-8"
    )


def test_httpd_priority_uses_validated_target_overrides() -> None:
    block = _config_block(SHARED_KCONFIG, "ESPECTRE_DIRECT_HTTPD_TASK_PRIORITY")
    assert re.search(
        r"(?m)^\s*default 4 if IDF_TARGET_ESP32 \|\| IDF_TARGET_ESP32S2\s*$",
        block,
    )
    assert re.search(r"(?m)^\s*default 1\s*$", block)
    for frontend in ("native", "matter"):
        app_dir = CPP_ROOT / "frontend" / frontend / "app"
        for defaults in app_dir.glob("sdkconfig.defaults.*"):
            assert "CONFIG_ESPECTRE_DIRECT_HTTPD_TASK_PRIORITY=" not in defaults.read_text(
                encoding="utf-8"
            )

    esphome_example = (
        CPP_ROOT / "frontend" / "esphome" / "examples" / "espectre-esp32.yaml"
    ).read_text(encoding="utf-8")
    assert "CONFIG_ESPECTRE_DIRECT_HTTPD_TASK_PRIORITY" not in esphome_example


def test_micro_native_tasks_use_the_shared_build_policy_names() -> None:
    expected = {
        "ESPECTRE_DIRECT_HTTPD_TASK_PRIORITY": 1,
        "ESPECTRE_TRAFFIC_TASK_PRIORITY": 1,
    }
    defaults = (
        MICRO_ROOT / "firmware" / "boards" / "sdkconfig.micro_espectre"
    ).read_text(encoding="utf-8")
    for symbol, default in expected.items():
        block = _config_block(MICRO_KCONFIG, symbol)
        assert re.search(r"(?m)^\s*range 1 10\s*$", block)
        assert re.search(rf"(?m)^\s*default {default}\s*$", block)
        assert f"CONFIG_{symbol}={default}" in defaults

    direct = (
        MICRO_ROOT / "firmware" / "native_components" / "native_direct.c"
    ).read_text(encoding="utf-8")
    traffic = (
        MICRO_ROOT / "firmware" / "native_components" / "native_traffic.c"
    ).read_text(encoding="utf-8")
    assert "CONFIG_ESPECTRE_DIRECT_HTTPD_TASK_PRIORITY" in direct
    assert "CONFIG_ESPECTRE_TRAFFIC_TASK_PRIORITY" in traffic
    assert "NATIVE_TRAFFIC_TASK_PRIORITY" not in traffic


def test_task_creation_uses_policy_constants_without_chip_conditionals() -> None:
    direct = (
        CPP_ROOT / "runtime" / "esp_idf" / "direct_http_service_esp_idf.cpp"
    ).read_text(encoding="utf-8")
    traffic = (
        CPP_ROOT / "runtime" / "esp_idf" / "traffic_generator_manager.cpp"
    ).read_text(encoding="utf-8")
    native = (
        CPP_ROOT / "frontend" / "native" / "app" / "main" / "app_main.cpp"
    ).read_text(encoding="utf-8")

    assert "task_scheduling::kDirectHttpdPriority" in direct
    assert "task_scheduling::kDirectWorkerPriority" in direct
    assert "task_scheduling::kRawWorkerPriority" in direct
    assert "CONFIG_IDF_TARGET_ESP32" not in direct
    assert "task_scheduling::kTrafficPriority" in traffic
    assert "task_scheduling::kNativeLoopPriority" in native
