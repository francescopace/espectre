"""Tests for `espectre micro collect` CLI options."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from types import ModuleType

from espectre_cli.app import build_parser
from espectre_cli import host


def _make_collect_args(**overrides) -> argparse.Namespace:
    args = {
        "info": False,
        "label": "static_presence",
        "samples": 2,
        "duration": 10.0,
        "start_delay": 5.0,
        "interactive": False,
        "udp_port": 5001,
        "bind_ip": None,
        "streamer_ip": "192.168.1.15",
        "stimulus_port": 9999,
        "stimulus_rate": 100,
        "reference_every": 20,
        "contributor": None,
        "description": None,
    }
    args.update(overrides)
    return argparse.Namespace(**args)


def test_collect_parser_accepts_count_alias() -> None:
    parser = build_parser()

    args = parser.parse_args(
        [
            "micro",
            "collect",
            "--label",
            "static_presence",
            "--duration",
            "10",
            "--count",
            "3",
            "--start-delay",
            "15",
            "--streamer-ip",
            "192.168.1.15",
        ]
    )

    assert args.namespace == "micro"
    assert args.micro_command == "collect"
    assert args.samples == 3
    assert args.start_delay == 15.0


def test_collect_parser_keeps_samples_option() -> None:
    parser = build_parser()

    args = parser.parse_args(
        [
            "micro",
            "collect",
            "--label",
            "motion",
            "--samples",
            "4",
            "--streamer-ip",
            "192.168.1.15",
        ]
    )

    assert args.samples == 4
    assert args.start_delay == 0.0


def test_collect_applies_start_delay_before_starting_stimulus(monkeypatch) -> None:
    events: list[object] = []
    fake_csi_utils = ModuleType("tools.csi_utils")

    class FakeCollector:
        def __init__(self, **kwargs):
            events.append(("collector_init", kwargs["label"], kwargs["bind_host"]))

        def collect_timed(self, duration: float, num_samples: int):
            events.append(("collect_timed", duration, num_samples))
            return [Path("sample_1.npz"), Path("sample_2.npz")]

        def collect_interactive(self, num_samples: int, duration: float):
            events.append(("collect_interactive", duration, num_samples))
            return []

    class FakeStimulusSender:
        def __init__(self, **kwargs):
            events.append(("sender_init", kwargs["target_host"], kwargs["reference_every"]))

        def start(self):
            events.append("start")

        def stop(self):
            events.append("stop")

    fake_csi_utils.CSICollector = FakeCollector
    fake_csi_utils.StimulusSender = FakeStimulusSender
    fake_csi_utils.get_dataset_stats = lambda: {"labels": {}, "total_samples": 0}
    fake_csi_utils.get_default_bind_host = lambda: "127.0.0.1"

    monkeypatch.setitem(sys.modules, "tools.csi_utils", fake_csi_utils)
    monkeypatch.setattr(host, "_wait_before_collection", lambda delay: events.append(("delay", delay)))

    host.collect_csi_data(_make_collect_args())

    assert ("delay", 5.0) in events
    assert ("collect_timed", 10.0, 2) in events
    assert events.index(("delay", 5.0)) < events.index("start")
    assert events[-1] == "stop"
