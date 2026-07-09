"""Host-side ESPectre tools."""

from __future__ import annotations

import ipaddress
import inspect
import sys

from .common import Path, REPO_ROOT, WEB_UI_FILE, Fore, Style, cli_command, print_box_banner, signal, time, webbrowser


_WEB_UI_FILES = {
    "mqtt": WEB_UI_FILE,
    "ble": REPO_ROOT / "tools" / "web" / "espectre-ble.html",
    "theremin": REPO_ROOT / "tools" / "web" / "espectre-theremin.html",
}


def _parse_targets(targets: str) -> tuple[list[str], str]:
    """Validate one or more comma-separated IPv4 destinations."""
    parsed_targets: list[str] = []
    mode_counts = {"unicast": 0, "broadcast": 0, "multicast": 0}
    for raw_target in str(targets).split(","):
        target = raw_target.strip()
        if not target:
            continue
        try:
            target_ip = ipaddress.ip_address(target)
        except ValueError as exc:
            raise ValueError(f"invalid target: {target}") from exc
        if target_ip.version != 4:
            raise ValueError(f"target must be an IPv4 address: {target}")
        normalized_target = str(target_ip)
        parsed_targets.append(normalized_target)
        if target_ip.is_multicast:
            mode_counts["multicast"] += 1
        elif int(target_ip) == 0xFFFFFFFF or normalized_target.endswith(".255"):
            mode_counts["broadcast"] += 1
        else:
            mode_counts["unicast"] += 1

    if not parsed_targets:
        raise ValueError("target required. Use --target <ip[,ip,...]>")

    unique_targets = list(dict.fromkeys(parsed_targets))
    if mode_counts["multicast"]:
        mode = "multicast"
    elif mode_counts["broadcast"]:
        mode = "broadcast"
    elif len(unique_targets) > 1:
        mode = "multi-unicast"
    else:
        mode = "unicast"
    return unique_targets, mode


def open_web_ui(interface: str = "mqtt") -> None:
    """Open the selected web interface in the default browser."""
    html_file = _WEB_UI_FILES.get(interface.lower())
    if html_file is None:
        print(f"{Fore.RED}❌ Error: unknown web UI '{interface}'{Style.RESET_ALL}")
        print(f"{Fore.YELLOW}Available interfaces: {', '.join(sorted(_WEB_UI_FILES))}{Style.RESET_ALL}")
        return
    if not html_file.exists():
        print(f"{Fore.RED}❌ Error: {html_file.name} not found{Style.RESET_ALL}")
        print(f"{Fore.YELLOW}Make sure you're running the command from the repo root{Style.RESET_ALL}")
        return

    file_url = html_file.absolute().as_uri()
    print(f"{Fore.BLUE}🌐 Opening web UI: {html_file.name}...{Style.RESET_ALL}")
    try:
        webbrowser.open(file_url)
        print(f"{Fore.GREEN}✅ Web UI opened in browser{Style.RESET_ALL}")
    except Exception as e:
        print(f"{Fore.RED}❌ Error opening browser: {e}{Style.RESET_ALL}")


def _wait_before_collection(delay_seconds: float) -> None:
    """Wait before starting collection so the operator can leave the room."""
    if delay_seconds <= 0:
        return

    print(f"  {Fore.YELLOW}Starting collection in {delay_seconds:.1f}s...{Style.RESET_ALL}")
    remaining = delay_seconds
    while remaining > 0:
        sleep_for = min(1.0, remaining)
        print(f"  {Fore.YELLOW}  {remaining:.1f}s remaining{Style.RESET_ALL}")
        time.sleep(sleep_for)
        remaining = max(0.0, remaining - sleep_for)
    print(f"  {Fore.GREEN}  Starting now.{Style.RESET_ALL}")
    print()


def _uses_legacy_dataset_collection(args) -> bool:
    """Return True when the unified collect command should use legacy timed dataset mode."""
    if getattr(args, "info", False):
        return True
    if float(getattr(args, "start_delay", 0.0) or 0.0) > 0:
        return True
    return int(getattr(args, "samples", 1) or 1) != 1


def _collect_dataset_csi_data(args) -> None:
    """Run the legacy timed dataset collection workflow."""
    try:
        from tools.lib.csi_io import CSICollector, StimulusSender, get_dataset_stats, get_default_bind_host
    except ImportError as e:
        print(f"{Fore.RED}❌ Failed to import tooling helpers: {e}{Style.RESET_ALL}")
        print(f"{Fore.YELLOW}Make sure the tools library package is available{Style.RESET_ALL}")
        raise SystemExit(1)

    if args.info:
        stats = get_dataset_stats()
        print()
        print_box_banner("Dataset Statistics")
        print()
        if not stats["labels"]:
            print(f"  {Fore.YELLOW}No samples collected yet.{Style.RESET_ALL}")
            print()
            print(f"  {Fore.CYAN}To collect data:{Style.RESET_ALL}")
            print("    1. Run the streamer firmware on the device")
            print(f"    2. Collect samples: {cli_command('collect', '--label', 'wave', '--samples', '10', '--target', '192.168.1.50')}")
        else:
            print(f"  {Fore.CYAN}{'Label':<20} {'Samples':>10}{Style.RESET_ALL}")
            print(f"  {'-' * 32}")
            for label, info in stats["labels"].items():
                print(f"  {label:<20} {info['samples']:>10}")
            print(f"  {'-' * 32}")
            print(f"  {Fore.GREEN}{'Total':<20} {stats['total_samples']:>10}{Style.RESET_ALL}")
        print()
        return

    if not args.label:
        print(f"{Fore.RED}❌ Label required. Use --label <name>{Style.RESET_ALL}")
        print(f"\n{Fore.YELLOW}Examples:{Style.RESET_ALL}")
        print(f"  {cli_command('collect', '--label', 'wave', '--samples', '10', '--target', '192.168.1.50')}")
        print(f"  {cli_command('collect', '--label', 'static_presence', '--duration', '10', '--target', '192.168.1.50')}")
        print(f"  {cli_command('collect', '--info')}")
        raise SystemExit(1)

    if not args.target:
        print(f"{Fore.RED}❌ Target required. Use --target <ip[,ip,...]>{Style.RESET_ALL}")
        raise SystemExit(1)

    if args.start_delay < 0:
        print(f"{Fore.RED}❌ Start delay must be >= 0 seconds{Style.RESET_ALL}")
        raise SystemExit(1)

    sample_duration = 2.0 if getattr(args, "duration", None) is None else float(args.duration)
    if sample_duration <= 0:
        print(f"{Fore.RED}❌ Duration must be > 0 seconds{Style.RESET_ALL}")
        raise SystemExit(1)

    try:
        targets, target_mode = _parse_targets(args.target)
    except ValueError as e:
        print(f"{Fore.RED}❌ {e}{Style.RESET_ALL}")
        raise SystemExit(1)

    resolved_bind_ip = args.bind_ip if args.bind_ip else get_default_bind_host()
    print()
    print_box_banner("Dataset Collection")
    print()
    print(f"  {Fore.CYAN}Label:{Style.RESET_ALL}     {args.label}")
    print(f"  {Fore.CYAN}Samples:{Style.RESET_ALL}   {args.samples}")
    print(f"  {Fore.CYAN}Duration:{Style.RESET_ALL}  {sample_duration}s per sample")
    if args.start_delay > 0:
        print(f"  {Fore.CYAN}Start delay:{Style.RESET_ALL} {args.start_delay}s")
    print(f"  {Fore.CYAN}Bind IP:{Style.RESET_ALL}   {resolved_bind_ip}")
    print(f"  {Fore.CYAN}UDP Port:{Style.RESET_ALL}  {args.udp_port}")
    print(f"  {Fore.CYAN}Target:{Style.RESET_ALL}    {', '.join(targets)} ({target_mode})")
    print(f"  {Fore.CYAN}Traffic:{Style.RESET_ALL}   {args.rate} pps -> {len(targets)} target(s) on UDP {args.target_port}")
    if args.reference_every > 0:
        print(f"  {Fore.CYAN}Reference:{Style.RESET_ALL} every {args.reference_every} packets")
    if args.description:
        print(f"  {Fore.CYAN}Description:{Style.RESET_ALL} {args.description}")
    print()
    print(f"  {Fore.YELLOW}Chip type auto-detected from CSI stream{Style.RESET_ALL}")
    print(f"  {Fore.YELLOW}Make sure the ESPectre streamer firmware is listening on the configured target/port{Style.RESET_ALL}")
    print()

    collector = CSICollector(
        label=args.label,
        port=args.udp_port,
        contributor=args.contributor,
        description=args.description,
        bind_host=resolved_bind_ip,
        expected_device_count=len(targets),
        expected_source_hosts=targets,
    )
    stimulus_sender = StimulusSender(
        target_host=targets,
        target_port=args.target_port,
        rate_pps=args.rate,
        reference_every=args.reference_every,
        source_host=resolved_bind_ip,
    )
    try:
        _wait_before_collection(args.start_delay)
        stimulus_sender.start()
        saved = collector.collect_timed(duration=sample_duration, num_samples=args.samples)
        if saved:
            print(f"{Fore.GREEN}✅ Collected {len(saved)} device file(s) for label '{args.label}'{Style.RESET_ALL}")
        else:
            print(f"{Fore.RED}❌ No samples collected{Style.RESET_ALL}")
    except KeyboardInterrupt:
        print(f"\n{Fore.YELLOW}Collection cancelled{Style.RESET_ALL}")
    except Exception as e:
        print(f"\n{Fore.RED}❌ Error during collection: {e}{Style.RESET_ALL}")
        raise SystemExit(1)
    finally:
        stimulus_sender.stop()


def collect_csi_data(args) -> None:
    """Run the unified host-side collect command."""
    if _uses_legacy_dataset_collection(args):
        _collect_dataset_csi_data(args)
        return
    _run_live_collect(args)


def _run_live_collect(args) -> None:
    """Run the host-side live collect pipeline."""
    try:
        from tools.lib.csi_io import CSICollector, CSIReceiver, StimulusSender, get_default_bind_host
        import config
        from console_output import format_calibration_status_line, format_detection_publish_line
        from detector_interface import (
            detector_needs_startup_calibration,
            load_detector_class,
            normalize_detector_algorithm,
            supported_detector_algorithms,
        )
        from ml_detector import ML_DEFAULT_THRESHOLD
        from runtime_policy import RuntimeMotionPolicy
        from threshold import (
            StartupThresholdCalibrator,
            get_detector_auto_factor,
            get_detector_startup_gate,
        )
    except ImportError:
        try:
            from tools.lib.csi_io import CSICollector, CSIReceiver, StimulusSender, get_default_bind_host
            import src.config as config
            from src.console_output import format_calibration_status_line, format_detection_publish_line
            from src.detector_interface import (
                detector_needs_startup_calibration,
                load_detector_class,
                normalize_detector_algorithm,
                supported_detector_algorithms,
            )
            from src.ml_detector import ML_DEFAULT_THRESHOLD
            from src.runtime_policy import RuntimeMotionPolicy
            from src.threshold import (
                StartupThresholdCalibrator,
                get_detector_auto_factor,
                get_detector_startup_gate,
            )
        except ImportError as e:
            print(f"{Fore.RED}❌ Failed to import live collect modules: {e}{Style.RESET_ALL}")
            raise SystemExit(1)

    supported_detectors = supported_detector_algorithms()
    detector_kinds = list(dict.fromkeys(
        normalize_detector_algorithm(kind.strip().lower())
        for kind in str(getattr(args, "detector", "classic")).split(",")
        if kind.strip()
    ))
    if not detector_kinds:
        detector_kinds = ["classic"]
    unsupported = [kind for kind in detector_kinds if kind not in supported_detectors]
    if unsupported:
        print(f"{Fore.RED}❌ Unsupported detector(s): {', '.join(unsupported)}{Style.RESET_ALL}")
        print(f"{Fore.YELLOW}Supported detectors: {', '.join(supported_detectors)}{Style.RESET_ALL}")
        raise SystemExit(1)

    calibrated_kinds = [kind for kind in detector_kinds if detector_needs_startup_calibration(kind)]
    detector_tag_width = max(len(kind) for kind in detector_kinds)

    label = getattr(args, "label", None)
    live_duration = getattr(args, "duration", None)
    no_save = bool(getattr(args, "no_save", False))
    save_enabled = bool(label) and not no_save
    ready_stable_seconds = 3.0

    if live_duration is not None and live_duration <= 0:
        print(f"{Fore.RED}❌ Duration must be > 0 seconds{Style.RESET_ALL}")
        raise SystemExit(1)
    if not getattr(args, "target", None):
        print(f"{Fore.RED}❌ Target required. Use --target <ip[,ip,...]>{Style.RESET_ALL}")
        raise SystemExit(1)
    if not save_enabled and not no_save and label is None:
        print(f"{Fore.RED}❌ Label required unless you use --no-save{Style.RESET_ALL}")
        raise SystemExit(1)

    raw_threshold_setting = getattr(config, "SEG_THRESHOLD", ML_DEFAULT_THRESHOLD)
    calibration_target_packets = max(
        1,
        int(getattr(config, "CALIBRATION_BUFFER_SIZE", getattr(config, "SEG_WINDOW_SIZE", 100) * 10)),
    )
    summary_evaluation_interval = max(1, int(getattr(config, "EVALUATION_INTERVAL", 25)))

    def get_initial_threshold(kind):
        if isinstance(raw_threshold_setting, (int, float)):
            return float(raw_threshold_setting)
        if kind == "ml":
            return ML_DEFAULT_THRESHOLD
        return 1.0

    def get_detector_threshold(detector, fallback=1.0):
        if hasattr(detector, "get_threshold"):
            return detector.get_threshold()
        return fallback

    def extract_motion_metric(metrics):
        return metrics.get(
            "motion_metric",
            metrics.get("probability", metrics.get("moving_variance", metrics.get("jitter", 0.0))),
        )

    def supports_inline_terminal(stream=None):
        target_stream = sys.stdout if stream is None else stream
        isatty = getattr(target_stream, "isatty", None)
        return bool(callable(isatty) and isatty())

    def emit_status_block(summary_line, detail_lines, *, previous_line_count=0, inline=None):
        target_stream = sys.stdout
        use_inline = supports_inline_terminal(target_stream) if inline is None else inline
        lines = [summary_line, *detail_lines]

        if not use_inline:
            for line in lines:
                target_stream.write(f"{line}\n")
            target_stream.flush()
            return len(lines)

        if previous_line_count > 0:
            target_stream.write(f"\x1b[{previous_line_count}F")

        total_lines = max(previous_line_count, len(lines))
        for idx in range(total_lines):
            target_stream.write("\x1b[2K")
            if idx < len(lines):
                target_stream.write(lines[idx])
            target_stream.write("\n")

        target_stream.flush()
        return len(lines)

    def clear_status_block():
        line_count = state["summary_line_count"]
        if line_count <= 0:
            return

        target_stream = sys.stdout
        if not state["summary_use_inline"]:
            state["summary_line_count"] = 0
            return

        target_stream.write(f"\x1b[{line_count}F")
        for _ in range(line_count):
            target_stream.write("\x1b[2K\n")
        target_stream.write(f"\x1b[{line_count}F")
        target_stream.flush()
        state["summary_line_count"] = 0

    def format_device_label(device_state):
        source_ip = str(device_state.get("source_ip") or "?")
        chip_label = str(device_state.get("chip") or "unknown").upper()
        channel = device_state.get("channel")
        rssi_dbm = device_state.get("rssi_dbm")
        channel_text = "--" if channel is None else f"{int(channel):02d}"
        rssi_text = "---" if rssi_dbm is None else str(int(rssi_dbm))
        return f"ip={source_ip} chip={chip_label} ch={channel_text} rssi={rssi_text}"

    def compute_sequence_gap(previous_seq, current_seq):
        expected = (previous_seq + 1) & 0xFFFFFFFF
        delta = (current_seq - expected) & 0xFFFFFFFF
        if delta == 0 or delta >= 0x80000000:
            return 0
        return delta

    def get_packet_device_id(pkt):
        device_id = getattr(pkt, "device_id", None)
        if device_id is None:
            return None
        return int(device_id)

    def create_detector(kind, threshold):
        return load_detector_class(kind)(
            window_size=config.SEG_WINDOW_SIZE,
            threshold=threshold,
            enable_lowpass=config.ENABLE_LOWPASS_FILTER,
            lowpass_cutoff=config.LOWPASS_CUTOFF,
            enable_hampel=config.ENABLE_HAMPEL_FILTER,
            hampel_window=config.HAMPEL_WINDOW,
            hampel_threshold=config.HAMPEL_THRESHOLD,
        )

    def check_sequence_by_device(pkt):
        device_id = get_packet_device_id(pkt)
        seq_num = getattr(pkt, "seq_num", None)
        if device_id is None or seq_num is None:
            return 0
        dropped = 0
        if device_id in state["last_seq_by_device"]:
            dropped = compute_sequence_gap(state["last_seq_by_device"][device_id], seq_num)
        state["last_seq_by_device"][device_id] = seq_num
        return dropped

    def build_detector_slot(kind):
        needs_calibration = kind in calibrated_kinds
        slot_initial_threshold = get_initial_threshold(kind)
        detector = create_detector(kind, slot_initial_threshold)
        runtime_policy = RuntimeMotionPolicy(
            evaluation_interval=getattr(config, "EVALUATION_INTERVAL", 25),
            motion_on_hits=getattr(config, "MOTION_ON_HITS", 3),
            motion_off_hits=getattr(config, "MOTION_OFF_HITS", 3),
        )
        return {
            "kind": kind,
            "detector": detector,
            "runtime_policy": runtime_policy,
            "motion_metric": 0.0,
            "metric_threshold": get_detector_threshold(detector, slot_initial_threshold),
            "effective_state": 0,
            "status": "WARMUP" if not needs_calibration else "WAITING",
            "calibration_detector": create_detector(kind, 1.0) if needs_calibration else None,
            "calibration_tracker": StartupThresholdCalibrator(
                calibration_target_packets,
                auto_factor=get_detector_auto_factor(detector),
                gate_enabled=get_detector_startup_gate(detector),
            ) if needs_calibration else None,
            "calibration_done": not needs_calibration,
            "calibration_success": not needs_calibration,
            "calibration_threshold_source": None if needs_calibration else "fixed",
            "ready_below_since": None,
            "ready_stable_for": 0.0,
        }

    def build_device_state(device_id):
        device_state = {
            "device_id": device_id,
            "source_ip": "?",
            "chip": "unknown",
            "channel": None,
            "rssi_dbm": None,
            "label": "",
            "packet_count": 0,
            "publish_counter": 0,
            "dropped_count": 0,
            "pps": 0,
            "pps_window_started_at": None,
            "pps_window_packets": 0,
            "last_publish_at": None,
            "slots": [build_detector_slot(kind) for kind in detector_kinds],
        }
        device_state["label"] = format_device_label(device_state)
        return device_state

    def get_device_state(pkt):
        device_id = get_packet_device_id(pkt)
        device_state = state["devices"].get(device_id)
        if device_state is None:
            device_state = build_device_state(device_id)
            state["devices"][device_id] = device_state
        source_ip = getattr(pkt, "source_ip", None)
        if source_ip:
            device_state["source_ip"] = str(source_ip)
        chip = getattr(pkt, "chip", None)
        if chip not in (None, "", "unknown"):
            device_state["chip"] = str(chip).upper()
        channel = getattr(pkt, "channel", None)
        if channel is not None:
            device_state["channel"] = int(channel)
        rssi_dbm = getattr(pkt, "rssi_dbm", None)
        if rssi_dbm is not None:
            device_state["rssi_dbm"] = int(rssi_dbm)
        device_state["label"] = format_device_label(device_state)
        return device_state

    def update_device_pps(device_state, now):
        if device_state["pps_window_started_at"] is None:
            device_state["pps_window_started_at"] = now
        device_state["pps_window_packets"] += 1
        elapsed = now - device_state["pps_window_started_at"]
        if elapsed >= 1.0:
            device_state["pps"] = int(device_state["pps_window_packets"] / elapsed) if elapsed > 0 else 0
            device_state["pps_window_started_at"] = now
            device_state["pps_window_packets"] = 0

    def update_ready_gate_state(device_state, now):
        for slot in device_state["slots"]:
            if not save_enabled or state["calibration_active"]:
                slot["ready_below_since"] = None
                slot["ready_stable_for"] = 0.0
                continue
            detector = slot["detector"]
            threshold = float(slot.get("metric_threshold", 0.0) or 0.0)
            if threshold <= 0 or not detector.is_ready():
                slot["ready_below_since"] = None
                slot["ready_stable_for"] = 0.0
                continue
            if float(slot["motion_metric"]) <= threshold:
                if slot["ready_below_since"] is None:
                    slot["ready_below_since"] = now
                slot["ready_stable_for"] = max(0.0, now - slot["ready_below_since"])
            else:
                slot["ready_below_since"] = None
                slot["ready_stable_for"] = 0.0

    def get_slot_gate_label(slot):
        if not save_enabled:
            return None
        if state["calibration_active"] and slot["calibration_tracker"] is not None:
            if slot["calibration_done"]:
                return "READY"
            if slot["calibration_tracker"].packet_count > 0:
                return "CALIBRATING"
            return "WAITING"
        detector = slot["detector"]
        if not detector.is_ready():
            return "WARMUP"
        if float(slot["motion_metric"]) > float(slot["metric_threshold"]):
            return "UNSTABLE"
        if float(slot["ready_stable_for"]) >= ready_stable_seconds:
            return "READY"
        return "STABLE"

    def summarize_ready_gate():
        observed_count = len(state["devices"])
        required_count = max(1, len(targets))
        if observed_count < required_count:
            return {
                "ready": False,
                "status": f"DEVICES {observed_count}/{required_count}",
                "stable_elapsed": 0.0,
            }
        relevant_states = list(state["devices"].values())
        warm_count = sum(
            1
            for device_state in relevant_states
            if all(slot["detector"].is_ready() for slot in device_state["slots"])
        )
        if warm_count < observed_count:
            return {
                "ready": False,
                "status": f"WARMUP {warm_count}/{required_count}",
                "stable_elapsed": 0.0,
            }
        stable_count = sum(
            1
            for device_state in relevant_states
            if all(
                float(slot["motion_metric"]) <= float(slot["metric_threshold"])
                for slot in device_state["slots"]
            )
        )
        if stable_count < observed_count:
            return {
                "ready": False,
                "status": f"UNSTABLE {stable_count}/{required_count}",
                "stable_elapsed": 0.0,
            }
        stable_elapsed = min(
            min(float(slot["ready_stable_for"]) for slot in device_state["slots"])
            for device_state in relevant_states
        )
        if stable_elapsed >= ready_stable_seconds:
            return {
                "ready": True,
                "status": f"READY {observed_count}/{required_count}",
                "stable_elapsed": ready_stable_seconds,
            }
        return {
            "ready": False,
            "status": f"STABLE {observed_count}/{required_count}",
            "stable_elapsed": stable_elapsed,
        }

    def get_slot_status(slot):
        if state["calibration_active"] and slot["calibration_tracker"] is not None:
            if slot["calibration_done"]:
                return "READY"
            if slot["calibration_tracker"].packet_count > 0:
                return "CALIBRATING"
            return "WAITING"
        detector = slot["detector"]
        if not detector.is_ready():
            return "WARMUP"
        return "MOTION" if int(slot["effective_state"]) == 1 else "IDLE"

    def finalize_slot_calibration(slot):
        detector = slot["detector"]
        runtime_policy = slot["runtime_policy"]
        calibration_tracker = slot["calibration_tracker"]
        slot["calibration_done"] = True
        if hasattr(runtime_policy, "reset"):
            runtime_policy.reset()
        if hasattr(detector, "reset"):
            detector.reset()

        if calibration_tracker is not None and calibration_tracker.is_successful():
            if isinstance(raw_threshold_setting, str):
                startup_threshold, threshold_formula = calibration_tracker.calculate_threshold(raw_threshold_setting)
                if hasattr(calibration_tracker, "get_floor_snapshot") and hasattr(detector, "apply_startup_floor"):
                    floor_value, vote_enabled, sample_count = calibration_tracker.get_floor_snapshot()
                    detector.apply_startup_floor(floor_value, vote_enabled, sample_count)
                if hasattr(detector, "set_adaptive_threshold"):
                    detector.set_adaptive_threshold(startup_threshold)
                elif hasattr(detector, "set_threshold"):
                    detector.set_threshold(startup_threshold)
                slot["calibration_threshold_source"] = f"{raw_threshold_setting} ({threshold_formula})"
            else:
                startup_threshold, _ = calibration_tracker.calculate_threshold("auto")
                if hasattr(calibration_tracker, "get_floor_snapshot") and hasattr(detector, "apply_startup_floor"):
                    floor_value, vote_enabled, sample_count = calibration_tracker.get_floor_snapshot()
                    detector.apply_startup_floor(floor_value, vote_enabled, sample_count)
                if hasattr(detector, "set_adaptive_threshold"):
                    detector.set_adaptive_threshold(startup_threshold)
                detector.set_threshold(float(raw_threshold_setting))
                slot["calibration_threshold_source"] = "manual"
            slot["calibration_success"] = True
        else:
            slot["calibration_success"] = False
            slot["calibration_threshold_source"] = "failed"
        slot["metric_threshold"] = get_detector_threshold(detector, slot["metric_threshold"])

        slot["motion_metric"] = 0.0
        slot["effective_state"] = 0
        slot["status"] = "IDLE"
        slot["ready_below_since"] = None
        slot["ready_stable_for"] = 0.0

    def process_calibration_packet(device_state, pkt):
        finalized_any = False
        for slot in device_state["slots"]:
            calibration_detector = slot["calibration_detector"]
            calibration_tracker = slot["calibration_tracker"]
            if calibration_detector is None or calibration_tracker is None or slot["calibration_done"]:
                continue

            calibration_detector.process_packet(pkt.iq_raw, subcarriers)
            calibration_metrics = calibration_detector.update_state()
            calibration_tracker.observe_detector(calibration_detector)
            slot["motion_metric"] = extract_motion_metric(calibration_metrics)
            slot["metric_threshold"] = calibration_metrics.get("threshold", calibration_detector.get_threshold())
            slot["status"] = getattr(calibration_tracker, "get_phase_label", lambda: "CALIBRATING")()

            if calibration_tracker.is_complete():
                finalize_slot_calibration(slot)
                finalized_any = True
        if finalized_any:
            device_state["publish_counter"] = 0

    def is_calibration_complete():
        required_count = max(1, len(targets))
        if len(state["devices"]) < required_count:
            return False
        return all(
            slot["calibration_done"]
            for device_state in state["devices"].values()
            for slot in device_state["slots"]
        )

    def maybe_stop_live_session(now):
        start_time = state["capture_started_at"] if save_enabled else state["session_started_at"]
        if live_duration is None or start_time is None:
            return False
        if (now - start_time) <= live_duration:
            return False
        if save_enabled:
            state["capture_completed"] = True
        state["running"] = False
        receiver.stop()
        return True

    def format_slot_label(device_state, slot):
        if len(detector_kinds) == 1:
            return device_state["label"]
        return f"{device_state['label']} [{slot['kind']:<{detector_tag_width}s}]"

    def render_multi_device_summary(now):
        observed_count = len(state["devices"])
        required_count = max(1, len(targets))
        detail_lines = []
        for device_id in sorted(state["devices"], key=lambda value: (value is None, value if value is not None else 0)):
            device_state = state["devices"][device_id]
            for slot in device_state["slots"]:
                status = get_slot_status(slot)
                slot_label = format_slot_label(device_state, slot)
                if state["calibration_active"] and slot["calibration_tracker"] is not None:
                    calibration_tracker = slot["calibration_tracker"]
                    calibration_packets = calibration_tracker.packet_count
                    if slot["calibration_done"]:
                        detail_lines.append(
                            "    "
                            + format_calibration_status_line(
                                progress=1.0,
                                pps=device_state["pps"],
                                packet_count=device_state["packet_count"],
                                dropped_count=device_state["dropped_count"],
                                motion_metric=slot["motion_metric"],
                                calibration_packets=calibration_packets,
                                calibration_target_packets=calibration_target_packets,
                                effective_state_label="READY",
                                device_label=slot_label,
                            )
                            + f" | thr:{slot['metric_threshold']:.4f} src:{slot['calibration_threshold_source']}"
                        )
                    else:
                        detail_lines.append(
                            "    "
                            + format_calibration_status_line(
                                progress=(calibration_packets / calibration_target_packets),
                                pps=device_state["pps"],
                                packet_count=device_state["packet_count"],
                                dropped_count=device_state["dropped_count"],
                                motion_metric=slot["motion_metric"],
                                calibration_packets=calibration_packets,
                                calibration_target_packets=calibration_target_packets,
                                effective_state_label=status,
                                device_label=slot_label,
                            )
                        )
                else:
                    progress_score = (
                        slot["motion_metric"] / slot["metric_threshold"]
                        if slot["metric_threshold"] > 0
                        else 0.0
                    )
                    detail_line = (
                        "    "
                        + format_detection_publish_line(
                            packet_count=device_state["packet_count"],
                            dropped_count=device_state["dropped_count"],
                            pps=device_state["pps"],
                            motion_metric=slot["motion_metric"],
                            threshold=slot["metric_threshold"],
                            effective_state=slot["effective_state"],
                            progress=progress_score,
                            device_label=slot_label,
                        )
                    )
                    if save_enabled and not state["capture_ready"]:
                        detail_line += f" | {get_slot_gate_label(slot)}"
                    detail_lines.append(detail_line)

        if state["calibration_active"]:
            summary_line = (
                f"  STATUS: CALIBRATING {observed_count}/{required_count} | "
                f"target {calibration_target_packets} pkts/device | capture {len(state['capture_packets'])}"
            )
        elif save_enabled and not state["capture_ready"]:
            ready_summary = summarize_ready_gate()
            summary_line = (
                f"  STATUS: STABILIZING {observed_count}/{required_count} | "
                f"{ready_summary['status'].lower()} | ready {ready_summary['stable_elapsed']:.1f}/{ready_stable_seconds:.1f}s "
                f"| packets {state['packet_count']} | capture {len(state['capture_packets'])}"
            )
        elif save_enabled:
            elapsed = 0.0 if state["capture_started_at"] is None else max(0.0, now - state["capture_started_at"])
            if live_duration is None:
                duration_text = "recording until Ctrl+C"
            else:
                duration_text = f"{elapsed:.1f}/{live_duration:.1f}s"
            summary_line = (
                f"  STATUS: RECORDING {observed_count}/{required_count} | "
                f"{duration_text if live_duration is None else f'elapsed {duration_text}'} | capture {len(state['capture_packets'])}"
            )
        else:
            elapsed = 0.0 if state["session_started_at"] is None else max(0.0, now - state["session_started_at"])
            if live_duration is None:
                duration_text = "collecting until Ctrl+C"
            else:
                duration_text = f"{elapsed:.1f}/{live_duration:.1f}s"
            summary_line = (
                f"  STATUS: COLLECTING {observed_count}/{required_count} | "
                f"{duration_text if live_duration is None else f'elapsed {duration_text}'} | packets {state['packet_count']}"
            )

        state["summary_line_count"] = emit_status_block(
            summary_line,
            detail_lines,
            previous_line_count=state["summary_line_count"],
            inline=state["summary_use_inline"],
        )

    try:
        targets, target_mode = _parse_targets(args.target)
    except ValueError as e:
        print(f"{Fore.RED}❌ {e}{Style.RESET_ALL}")
        raise SystemExit(1)

    resolved_bind_ip = args.bind_ip if args.bind_ip else get_default_bind_host()
    subcarriers = list(config.DEFAULT_SUBCARRIERS)
    publish_rate = getattr(config, "PUBLISH_INTERVAL", 100) or 100
    receiver = CSIReceiver(port=args.udp_port, buffer_size=4000, bind_host=resolved_bind_ip)
    stimulus_sender = StimulusSender(
        target_host=targets,
        target_port=args.target_port,
        rate_pps=args.rate,
        reference_every=args.reference_every,
        source_host=resolved_bind_ip,
    )
    capture_writer = None
    if save_enabled:
        capture_writer = CSICollector(
            label=label,
            port=args.udp_port,
            contributor=getattr(args, "contributor", None),
            description=getattr(args, "description", None),
            bind_host=resolved_bind_ip,
            expected_device_count=len(targets),
            expected_source_hosts=targets,
        )

    state = {
        "running": True,
        "packet_count": 0,
        "capture_packets": [],
        "session_started_at": None,
        "capture_started_at": None,
        "capture_ready": not save_enabled,
        "capture_completed": False,
        "interrupted": False,
        "devices": {},
        "last_seq_by_device": {},
        "summary_line_count": 0,
        "summary_use_inline": supports_inline_terminal(),
        "calibration_active": bool(calibrated_kinds),
    }

    def handle_sigint(_signum, _frame):
        state["interrupted"] = True
        state["running"] = False
        receiver.stop()

    def on_packet(pkt):
        if not state["running"]:
            return

        now = time.monotonic()
        state["packet_count"] += 1
        if state["session_started_at"] is None:
            state["session_started_at"] = now

        device_state = get_device_state(pkt)
        device_state["packet_count"] += 1
        device_state["publish_counter"] += 1
        device_state["dropped_count"] += check_sequence_by_device(pkt)
        update_device_pps(device_state, now)

        if state["calibration_active"]:
            process_calibration_packet(device_state, pkt)
            calibration_trackers = [
                slot["calibration_tracker"]
                for slot in device_state["slots"]
                if slot["calibration_tracker"] is not None
            ]
            calibration_render_due = any(
                (tracker.packet_count % summary_evaluation_interval) == 0
                for tracker in calibration_trackers
            )
            if is_calibration_complete():
                state["calibration_active"] = False
                render_multi_device_summary(now)
            elif calibration_render_due:
                render_multi_device_summary(now)
            if not save_enabled and maybe_stop_live_session(now):
                return
            return

        should_publish = device_state["publish_counter"] >= publish_rate
        should_render_summary = False
        for slot in device_state["slots"]:
            detector = slot["detector"]
            runtime_policy = slot["runtime_policy"]
            detector.process_packet(pkt.iq_raw, subcarriers)
            runtime_policy.note_packet()
            metrics = detector.update_state()
            slot["motion_metric"] = extract_motion_metric(metrics)
            slot["metric_threshold"] = metrics["threshold"]

            if runtime_policy.should_evaluate(should_publish):
                effective_state, _ = runtime_policy.apply_state(metrics["state"])
                runtime_policy.after_evaluation()
                slot["effective_state"] = effective_state
                slot["status"] = get_slot_status(slot)
                should_render_summary = True

                if should_publish:
                    device_state["last_publish_at"] = now

        update_ready_gate_state(device_state, now)
        if should_publish:
            device_state["publish_counter"] = 0

        if save_enabled and not state["capture_ready"]:
            ready_summary = summarize_ready_gate()
            if ready_summary["ready"]:
                state["capture_ready"] = True
                state["capture_started_at"] = now
                should_render_summary = True

        if save_enabled and state["capture_ready"]:
            if maybe_stop_live_session(now):
                return
            state["capture_packets"].append(pkt)
        elif not save_enabled and maybe_stop_live_session(now):
            return

        if should_render_summary:
            render_multi_device_summary(now)

    receiver.add_callback(on_packet)
    signal.signal(signal.SIGINT, handle_sigint)

    print()
    print_box_banner("Live CSI Collection")
    print()
    print(f"  {Fore.CYAN}Detector:{Style.RESET_ALL}  {', '.join(kind.upper() for kind in detector_kinds)}")
    print(f"  {Fore.CYAN}Bind IP:{Style.RESET_ALL}   {resolved_bind_ip}")
    print(f"  {Fore.CYAN}UDP Port:{Style.RESET_ALL}  {args.udp_port}")
    print(f"  {Fore.CYAN}Target:{Style.RESET_ALL}    {', '.join(targets)} ({target_mode})")
    print(f"  {Fore.CYAN}Traffic:{Style.RESET_ALL}   {args.rate} pps -> {len(targets)} target(s) on UDP {args.target_port}")
    if args.reference_every > 0:
        print(f"  {Fore.CYAN}Reference:{Style.RESET_ALL} every {args.reference_every} packets")
    if "ml" in detector_kinds:
        ml_suffix = " (ml, fixed)" if len(detector_kinds) > 1 else ""
        print(f"  {Fore.CYAN}Threshold:{Style.RESET_ALL} {get_initial_threshold('ml'):.2f}{ml_suffix}")
    if calibrated_kinds:
        threshold_text = raw_threshold_setting if isinstance(raw_threshold_setting, str) else f"{float(raw_threshold_setting):.4f}"
        print(f"  {Fore.CYAN}Threshold:{Style.RESET_ALL} {threshold_text} (after startup calibration)")
        print(f"  {Fore.CYAN}Calibration:{Style.RESET_ALL} {calibration_target_packets} packets/device")
    print(f"  {Fore.CYAN}Window:{Style.RESET_ALL}    {config.SEG_WINDOW_SIZE} pkts")
    print(f"  {Fore.CYAN}Subcarriers:{Style.RESET_ALL} {subcarriers}")
    print(
        f"  {Fore.CYAN}Hits on/off:{Style.RESET_ALL} "
        f"{getattr(config, 'MOTION_ON_HITS', 3)}/{getattr(config, 'MOTION_OFF_HITS', 3)}"
    )
    print(f"  {Fore.CYAN}Low-pass:{Style.RESET_ALL}  {'ON' if config.ENABLE_LOWPASS_FILTER else 'OFF'}")
    print(f"  {Fore.CYAN}Hampel:{Style.RESET_ALL}    {'ON' if config.ENABLE_HAMPEL_FILTER else 'OFF'}")
    if save_enabled:
        duration_text = "until Ctrl+C" if live_duration is None else f"{live_duration:g}s"
        print(f"  {Fore.CYAN}Save:{Style.RESET_ALL}      label={label} duration={duration_text}")
        print(f"  {Fore.CYAN}Ready gate:{Style.RESET_ALL} {ready_stable_seconds:.1f}s below threshold before saving")
        if getattr(args, "description", None):
            print(f"  {Fore.CYAN}Description:{Style.RESET_ALL} {args.description}")
    else:
        print(f"  {Fore.CYAN}Save:{Style.RESET_ALL}      disabled")
    print()
    print(f"  {Fore.YELLOW}Make sure the ESPectre streamer firmware is listening on the configured target/port{Style.RESET_ALL}")
    if calibrated_kinds:
        print(f"  {Fore.YELLOW}Please remain still during the startup calibration phase{Style.RESET_ALL}")
    print(f"  {Fore.YELLOW}Press Ctrl+C to stop{Style.RESET_ALL}")
    print()

    try:
        stimulus_sender.start()
        while state["running"]:
            announce_socket_rcvbuf = state.get("socket_rcvbuf_reported") is not True
            run_kwargs = {"timeout": 1.0, "quiet": True}
            if "announce_socket_rcvbuf" in inspect.signature(receiver.run).parameters:
                run_kwargs["announce_socket_rcvbuf"] = announce_socket_rcvbuf
            receiver.run(**run_kwargs)
            if announce_socket_rcvbuf and receiver.effective_socket_rcvbuf_bytes is not None:
                state["socket_rcvbuf_reported"] = True
    except KeyboardInterrupt:
        state["interrupted"] = True
        render_multi_device_summary(time.monotonic())
    except Exception as e:
        print(f"\n{Fore.RED}❌ Error during live collect: {e}{Style.RESET_ALL}")
        raise SystemExit(1)
    finally:
        stimulus_sender.stop()
        receiver.stop()
        clear_status_block()
        if capture_writer is not None:
            captured_packets = state["capture_packets"]
            if live_duration is not None and state["interrupted"] and not state["capture_completed"]:
                print(f"{Fore.YELLOW}Live capture interrupted before duration elapsed; nothing saved{Style.RESET_ALL}")
            elif captured_packets:
                try:
                    saved_paths = capture_writer.save_samples_by_device(captured_packets)
                except ValueError as e:
                    print(f"{Fore.RED}❌ Failed to save live capture: {e}{Style.RESET_ALL}")
                    raise SystemExit(1)
                if saved_paths:
                    print(
                        f"{Fore.GREEN}✅ Saved {len(saved_paths)} live capture file(s) "
                        f"from {len(captured_packets)} packets{Style.RESET_ALL}"
                    )
                    for saved_path in saved_paths:
                        print(f"  - {saved_path.name}")
                else:
                    print(f"{Fore.RED}❌ Live capture had no packets to save{Style.RESET_ALL}")
            else:
                print(f"{Fore.YELLOW}No live capture packets received; nothing saved{Style.RESET_ALL}")
        print(f"\n{Fore.GREEN}Done.{Style.RESET_ALL}\n")
