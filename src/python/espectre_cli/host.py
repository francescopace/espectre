"""Host-side Micro-ESPectre tools."""

from __future__ import annotations

import ipaddress
import sys

from .common import Path, REPO_ROOT, WEB_UI_FILE, Fore, Style, cli_command, signal, time, webbrowser


_WEB_UI_FILES = {
    "mqtt": WEB_UI_FILE,
    "ble": REPO_ROOT / "tools" / "web" / "espectre-ble.html",
    "theremin": REPO_ROOT / "tools" / "web" / "espectre-theremin.html",
}


def _parse_stimulus_targets(targets: str) -> tuple[list[str], str]:
    """Validate one or more comma-separated IPv4 stimulus destinations."""
    parsed_targets: list[str] = []
    mode_counts = {"unicast": 0, "broadcast": 0, "multicast": 0}
    for raw_target in str(targets).split(","):
        target = raw_target.strip()
        if not target:
            continue
        try:
            target_ip = ipaddress.ip_address(target)
        except ValueError as exc:
            raise ValueError(f"invalid stimulus target: {target}") from exc
        if target_ip.version != 4:
            raise ValueError(f"stimulus target must be an IPv4 address: {target}")
        normalized_target = str(target_ip)
        parsed_targets.append(normalized_target)
        if target_ip.is_multicast:
            mode_counts["multicast"] += 1
        elif int(target_ip) == 0xFFFFFFFF or normalized_target.endswith(".255"):
            mode_counts["broadcast"] += 1
        else:
            mode_counts["unicast"] += 1

    if not parsed_targets:
        raise ValueError("stimulus target required. Use --stimulus-target <ip[,ip,...]>")

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
    """Return True when the unified collect command should use legacy dataset mode."""
    if getattr(args, "info", False):
        return True
    if getattr(args, "interactive", False):
        return True
    if float(getattr(args, "start_delay", 0.0) or 0.0) > 0:
        return True
    return int(getattr(args, "samples", 1) or 1) != 1


def _collect_dataset_csi_data(args) -> None:
    """Run the legacy timed/interactive dataset collection workflow."""
    try:
        from tools.csi_utils import CSICollector, StimulusSender, get_dataset_stats, get_default_bind_host
    except ImportError as e:
        print(f"{Fore.RED}❌ Failed to import csi_utils: {e}{Style.RESET_ALL}")
        print(f"{Fore.YELLOW}Make sure tools/csi_utils.py exists{Style.RESET_ALL}")
        raise SystemExit(1)

    if args.info:
        stats = get_dataset_stats()
        print(f"\n{Fore.MAGENTA}╔═══════════════════════════════════════════════════════════╗{Style.RESET_ALL}")
        print(f"{Fore.MAGENTA}║           μESPectre - Dataset Statistics                  ║{Style.RESET_ALL}")
        print(f"{Fore.MAGENTA}╚═══════════════════════════════════════════════════════════╝{Style.RESET_ALL}")
        print()
        if not stats["labels"]:
            print(f"  {Fore.YELLOW}No samples collected yet.{Style.RESET_ALL}")
            print()
            print(f"  {Fore.CYAN}To collect data:{Style.RESET_ALL}")
            print("    1. Run the streamer firmware on the device")
            print(f"    2. Collect samples: {cli_command('collect', '--label', 'wave', '--samples', '10', '--stimulus-target', '192.168.1.50')}")
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
        print(f"  {cli_command('collect', '--label', 'wave', '--samples', '10', '--stimulus-target', '192.168.1.50')}")
        print(f"  {cli_command('collect', '--label', 'static_presence', '--duration', '10', '--stimulus-target', '192.168.1.50')}")
        print(f"  {cli_command('collect', '--info')}")
        raise SystemExit(1)

    if not args.stimulus_target:
        print(f"{Fore.RED}❌ Stimulus target required. Use --stimulus-target <ip[,ip,...]>{Style.RESET_ALL}")
        raise SystemExit(1)

    if args.start_delay < 0:
        print(f"{Fore.RED}❌ Start delay must be >= 0 seconds{Style.RESET_ALL}")
        raise SystemExit(1)

    sample_duration = 2.0 if getattr(args, "duration", None) is None else float(args.duration)
    if sample_duration <= 0:
        print(f"{Fore.RED}❌ Duration must be > 0 seconds{Style.RESET_ALL}")
        raise SystemExit(1)

    try:
        stimulus_targets, target_mode = _parse_stimulus_targets(args.stimulus_target)
    except ValueError as e:
        print(f"{Fore.RED}❌ {e}{Style.RESET_ALL}")
        raise SystemExit(1)

    resolved_bind_ip = args.bind_ip if args.bind_ip else get_default_bind_host()
    print(f"\n{Fore.MAGENTA}╔═══════════════════════════════════════════════════════════╗{Style.RESET_ALL}")
    print(f"{Fore.MAGENTA}║           μESPectre - CSI Data Collection                 ║{Style.RESET_ALL}")
    print(f"{Fore.MAGENTA}╚═══════════════════════════════════════════════════════════╝{Style.RESET_ALL}")
    print()
    print(f"  {Fore.CYAN}Label:{Style.RESET_ALL}     {args.label}")
    print(f"  {Fore.CYAN}Samples:{Style.RESET_ALL}   {args.samples}")
    print(f"  {Fore.CYAN}Duration:{Style.RESET_ALL}  {sample_duration}s per sample")
    if args.start_delay > 0:
        print(f"  {Fore.CYAN}Start delay:{Style.RESET_ALL} {args.start_delay}s")
    print(f"  {Fore.CYAN}Bind IP:{Style.RESET_ALL}   {resolved_bind_ip}")
    print(f"  {Fore.CYAN}UDP Port:{Style.RESET_ALL}  {args.udp_port}")
    print(f"  {Fore.CYAN}Stimulus target:{Style.RESET_ALL} {', '.join(stimulus_targets)} ({target_mode})")
    print(f"  {Fore.CYAN}Stimulus:{Style.RESET_ALL}  {args.stimulus_rate} pps -> {len(stimulus_targets)} target(s) on UDP {args.stimulus_port}")
    if args.reference_every > 0:
        print(f"  {Fore.CYAN}Reference:{Style.RESET_ALL} every {args.reference_every} packets")
    if args.description:
        print(f"  {Fore.CYAN}Description:{Style.RESET_ALL} {args.description}")
    print()
    print(f"  {Fore.YELLOW}Chip type auto-detected from CSI stream{Style.RESET_ALL}")
    print(f"  {Fore.YELLOW}Make sure the ESPectre streamer firmware is listening for the configured shared stimulus target{Style.RESET_ALL}")
    print()

    collector = CSICollector(
        label=args.label,
        port=args.udp_port,
        contributor=args.contributor,
        description=args.description,
        bind_host=resolved_bind_ip,
        expected_device_count=len(stimulus_targets),
        expected_source_hosts=stimulus_targets,
    )
    stimulus_sender = StimulusSender(
        target_host=stimulus_targets,
        target_port=args.stimulus_port,
        rate_pps=args.stimulus_rate,
        reference_every=args.reference_every,
        source_host=resolved_bind_ip,
    )
    try:
        _wait_before_collection(args.start_delay)
        stimulus_sender.start()
        if args.interactive:
            saved = collector.collect_interactive(num_samples=args.samples, duration=sample_duration)
        else:
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
        from tools.csi_utils import CSICollector, CSIReceiver, StimulusSender, get_default_bind_host
        import config
        from console_output import format_calibration_status_line, format_detection_publish_line
        from ml_detector import FEATURE_NAMES as ML_FEATURE_NAMES, ML_DEFAULT_THRESHOLD, MLDetector
        from mvs_detector import MVSDetector
        from runtime_policy import RuntimeMotionPolicy
        from threshold import StartupThresholdCalibrator
    except ImportError:
        try:
            from tools.csi_utils import CSICollector, CSIReceiver, StimulusSender, get_default_bind_host
            import src.config as config
            from src.console_output import format_calibration_status_line, format_detection_publish_line
            from src.ml_detector import FEATURE_NAMES as ML_FEATURE_NAMES, ML_DEFAULT_THRESHOLD, MLDetector
            from src.mvs_detector import MVSDetector
            from src.runtime_policy import RuntimeMotionPolicy
            from src.threshold import StartupThresholdCalibrator
        except ImportError as e:
            print(f"{Fore.RED}❌ Failed to import live collect modules: {e}{Style.RESET_ALL}")
            raise SystemExit(1)

    detector_kind = str(getattr(args, "detector", "mvs")).lower()
    if detector_kind not in {"ml", "mvs"}:
        print(f"{Fore.RED}❌ Unsupported detector: {detector_kind}{Style.RESET_ALL}")
        raise SystemExit(1)

    label = getattr(args, "label", None)
    live_duration = getattr(args, "duration", None)
    no_save = bool(getattr(args, "no_save", False))
    save_enabled = bool(label) and not no_save
    ready_stable_seconds = 3.0

    if live_duration is not None and live_duration <= 0:
        print(f"{Fore.RED}❌ Duration must be > 0 seconds{Style.RESET_ALL}")
        raise SystemExit(1)
    if not getattr(args, "stimulus_target", None):
        print(f"{Fore.RED}❌ Stimulus target required. Use --stimulus-target <ip[,ip,...]>{Style.RESET_ALL}")
        raise SystemExit(1)
    if not save_enabled and not no_save and label is None:
        print(f"{Fore.RED}❌ Label required unless you use --no-save{Style.RESET_ALL}")
        raise SystemExit(1)

    feature_names = ML_FEATURE_NAMES if detector_kind == "ml" else []
    raw_threshold_setting = getattr(config, "SEG_THRESHOLD", ML_DEFAULT_THRESHOLD)
    calibration_target_packets = max(
        1,
        int(getattr(config, "CALIBRATION_BUFFER_SIZE", getattr(config, "SEG_WINDOW_SIZE", 100) * 10)),
    )

    def format_feature_vector(features):
        return " ".join(f"{name}={value:.4f}" for name, value in zip(feature_names, features))

    def get_ordered_turbulence_tail(ctx, tail_size):
        if ctx.buffer_count <= 0 or tail_size <= 0:
            return []
        if ctx.buffer_count < ctx.window_size:
            ordered = ctx.turbulence_buffer[:ctx.buffer_count]
        else:
            idx = ctx.buffer_index
            ordered = ctx.turbulence_buffer[idx:] + ctx.turbulence_buffer[:idx]
        return ordered[-tail_size:]

    def format_turbulence_tail(values):
        return " ".join(f"{value:.4f}" for value in values)

    def get_initial_threshold():
        if detector_kind == "ml":
            if isinstance(raw_threshold_setting, (int, float)):
                return float(raw_threshold_setting)
            return ML_DEFAULT_THRESHOLD
        if isinstance(raw_threshold_setting, (int, float)):
            return float(raw_threshold_setting)
        return 1.0

    def get_detector_threshold(detector):
        if hasattr(detector, "get_threshold"):
            return detector.get_threshold()
        return initial_threshold

    def extract_motion_metric(metrics):
        return metrics.get("probability", metrics.get("moving_variance", metrics.get("jitter", 0.0)))

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

    def create_runtime_detector(initial_threshold):
        common_kwargs = {
            "window_size": config.SEG_WINDOW_SIZE,
            "threshold": initial_threshold,
            "enable_lowpass": config.ENABLE_LOWPASS_FILTER,
            "lowpass_cutoff": config.LOWPASS_CUTOFF,
            "enable_hampel": config.ENABLE_HAMPEL_FILTER,
            "hampel_window": config.HAMPEL_WINDOW,
            "hampel_threshold": config.HAMPEL_THRESHOLD,
        }
        if detector_kind == "ml":
            return MLDetector(**common_kwargs)
        return MVSDetector(**common_kwargs)

    def create_calibration_detector():
        return MVSDetector(
            window_size=config.SEG_WINDOW_SIZE,
            threshold=1.0,
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

    def build_device_state(device_id):
        detector = create_runtime_detector(initial_threshold)
        runtime_policy = RuntimeMotionPolicy(
            evaluation_interval=getattr(config, "EVALUATION_INTERVAL", 25),
            motion_on_hits=getattr(config, "MOTION_ON_HITS", 3),
            motion_off_hits=getattr(config, "MOTION_OFF_HITS", 3),
        )
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
            "motion_metric": 0.0,
            "metric_threshold": get_detector_threshold(detector),
            "effective_state": 0,
            "status": "WARMUP" if detector_kind == "ml" else "WAITING",
            "last_publish_at": None,
            "detector": detector,
            "runtime_policy": runtime_policy,
            "calibration_detector": create_calibration_detector() if detector_kind == "mvs" else None,
            "calibration_tracker": StartupThresholdCalibrator(calibration_target_packets) if detector_kind == "mvs" else None,
            "calibration_done": detector_kind == "ml",
            "calibration_success": detector_kind == "ml",
            "calibration_threshold_source": "fixed" if detector_kind == "ml" else None,
            "ready_below_since": None,
            "ready_stable_for": 0.0,
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
        if not save_enabled or state["calibration_active"]:
            device_state["ready_below_since"] = None
            device_state["ready_stable_for"] = 0.0
            return
        detector = device_state["detector"]
        threshold = float(device_state.get("metric_threshold", 0.0) or 0.0)
        if threshold <= 0 or not detector.is_ready():
            device_state["ready_below_since"] = None
            device_state["ready_stable_for"] = 0.0
            return
        if float(device_state["motion_metric"]) <= threshold:
            if device_state["ready_below_since"] is None:
                device_state["ready_below_since"] = now
            device_state["ready_stable_for"] = max(0.0, now - device_state["ready_below_since"])
        else:
            device_state["ready_below_since"] = None
            device_state["ready_stable_for"] = 0.0

    def get_device_gate_label(device_state):
        if not save_enabled:
            return None
        if state["calibration_active"] and detector_kind == "mvs":
            calibration_tracker = device_state["calibration_tracker"]
            if device_state["calibration_done"]:
                return "READY"
            if calibration_tracker is not None and calibration_tracker.packet_count > 0:
                return "CALIBRATING"
            return "WAITING"
        detector = device_state["detector"]
        if not detector.is_ready():
            return "WARMUP"
        if float(device_state["motion_metric"]) > float(device_state["metric_threshold"]):
            return "UNSTABLE"
        if float(device_state["ready_stable_for"]) >= ready_stable_seconds:
            return "READY"
        return "STABLE"

    def summarize_ready_gate():
        observed_count = len(state["devices"])
        required_count = max(1, len(stimulus_targets))
        if observed_count < required_count:
            return {
                "ready": False,
                "status": f"DEVICES {observed_count}/{required_count}",
                "stable_elapsed": 0.0,
            }
        relevant_states = list(state["devices"].values())
        warm_count = sum(1 for device_state in relevant_states if device_state["detector"].is_ready())
        if warm_count < observed_count:
            return {
                "ready": False,
                "status": f"WARMUP {warm_count}/{required_count}",
                "stable_elapsed": 0.0,
            }
        stable_count = sum(
            1
            for device_state in relevant_states
            if float(device_state["motion_metric"]) <= float(device_state["metric_threshold"])
        )
        if stable_count < observed_count:
            return {
                "ready": False,
                "status": f"UNSTABLE {stable_count}/{required_count}",
                "stable_elapsed": 0.0,
            }
        stable_elapsed = min(float(device_state["ready_stable_for"]) for device_state in relevant_states)
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

    def get_device_status(device_state):
        if state["calibration_active"] and detector_kind == "mvs":
            calibration_tracker = device_state["calibration_tracker"]
            if device_state["calibration_done"]:
                return "READY"
            if calibration_tracker is not None and calibration_tracker.packet_count > 0:
                return "CALIBRATING"
            return "WAITING"
        detector = device_state["detector"]
        if not detector.is_ready():
            return "WARMUP"
        return "MOTION" if int(device_state["effective_state"]) == 1 else "IDLE"

    def should_print_publish_details(effective_state):
        if not (args.log_turbulence or args.log_features):
            return False
        if args.log_only_motion and effective_state != 1:
            return False
        return True

    def finalize_device_calibration(device_state):
        detector = device_state["detector"]
        runtime_policy = device_state["runtime_policy"]
        calibration_tracker = device_state["calibration_tracker"]
        device_state["calibration_done"] = True
        device_state["publish_counter"] = 0
        if hasattr(runtime_policy, "reset"):
            runtime_policy.reset()
        if hasattr(detector, "reset"):
            detector.reset()

        if calibration_tracker is not None and calibration_tracker.is_successful():
            if isinstance(raw_threshold_setting, str):
                startup_threshold, threshold_formula = calibration_tracker.calculate_threshold(raw_threshold_setting)
                if hasattr(detector, "set_adaptive_threshold"):
                    detector.set_adaptive_threshold(startup_threshold)
                elif hasattr(detector, "set_threshold"):
                    detector.set_threshold(startup_threshold)
                device_state["calibration_threshold_source"] = f"{raw_threshold_setting} ({threshold_formula})"
            else:
                detector.set_threshold(float(raw_threshold_setting))
                device_state["calibration_threshold_source"] = "manual"
            device_state["calibration_success"] = True
            device_state["metric_threshold"] = get_detector_threshold(detector)
        else:
            device_state["calibration_success"] = False
            device_state["calibration_threshold_source"] = "failed"
            device_state["metric_threshold"] = get_detector_threshold(detector)

        device_state["motion_metric"] = 0.0
        device_state["effective_state"] = 0
        device_state["status"] = "IDLE"
        device_state["ready_below_since"] = None
        device_state["ready_stable_for"] = 0.0

    def process_calibration_packet(device_state, pkt):
        calibration_detector = device_state["calibration_detector"]
        calibration_tracker = device_state["calibration_tracker"]
        if calibration_detector is None or calibration_tracker is None or device_state["calibration_done"]:
            return

        calibration_detector.process_packet(pkt.iq_raw, subcarriers)
        calibration_metrics = calibration_detector.update_state()
        calibration_tracker.observe_detector(calibration_detector)
        device_state["motion_metric"] = extract_motion_metric(calibration_metrics)
        device_state["metric_threshold"] = calibration_metrics.get("threshold", calibration_detector.get_threshold())
        device_state["status"] = "CALIBRATING"

        if calibration_tracker.is_complete():
            finalize_device_calibration(device_state)

    def is_calibration_complete():
        required_count = max(1, len(stimulus_targets))
        if len(state["devices"]) < required_count:
            return False
        return all(device_state["calibration_done"] for device_state in state["devices"].values())

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

    def render_multi_device_summary(now, *, force=False):
        refresh_seconds = 1.0
        if not force and (now - state["summary_last_rendered_at"]) < refresh_seconds:
            return

        observed_count = len(state["devices"])
        required_count = max(1, len(stimulus_targets))
        detail_lines = []
        for device_id in sorted(state["devices"], key=lambda value: (value is None, value if value is not None else 0)):
            device_state = state["devices"][device_id]
            status = get_device_status(device_state)
            if state["calibration_active"] and detector_kind == "mvs":
                calibration_tracker = device_state["calibration_tracker"]
                calibration_packets = calibration_tracker.packet_count if calibration_tracker is not None else 0
                if device_state["calibration_done"]:
                    detail_lines.append(
                        "    "
                        + format_calibration_status_line(
                            progress=1.0,
                            pps=device_state["pps"],
                            motion_metric=device_state["motion_metric"],
                            calibration_packets=calibration_packets,
                            calibration_target_packets=calibration_target_packets,
                            effective_state_label="READY",
                            device_label=device_state["label"],
                        )
                        + f" | thr:{device_state['metric_threshold']:.4f} src:{device_state['calibration_threshold_source']}"
                    )
                else:
                    detail_lines.append(
                        "    "
                        + format_calibration_status_line(
                            progress=(calibration_packets / calibration_target_packets),
                            pps=device_state["pps"],
                            motion_metric=device_state["motion_metric"],
                            calibration_packets=calibration_packets,
                            calibration_target_packets=calibration_target_packets,
                            effective_state_label=status,
                            device_label=device_state["label"],
                        )
                    )
            else:
                progress_score = (
                    device_state["motion_metric"] / device_state["metric_threshold"]
                    if device_state["metric_threshold"] > 0
                    else 0.0
                )
                detail_line = (
                    "    "
                    + format_detection_publish_line(
                        pps=device_state["pps"],
                        motion_metric=device_state["motion_metric"],
                        threshold=device_state["metric_threshold"],
                        effective_state=device_state["effective_state"],
                        progress=progress_score,
                        device_label=device_state["label"],
                    )
                )
                if save_enabled and not state["capture_ready"]:
                    detail_line += f" | {get_device_gate_label(device_state)}"
                detail_lines.append(detail_line)

        if state["calibration_active"] and detector_kind == "mvs":
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
        state["summary_last_rendered_at"] = now

    try:
        stimulus_targets, target_mode = _parse_stimulus_targets(args.stimulus_target)
    except ValueError as e:
        print(f"{Fore.RED}❌ {e}{Style.RESET_ALL}")
        raise SystemExit(1)

    resolved_bind_ip = args.bind_ip if args.bind_ip else get_default_bind_host()
    subcarriers = list(config.DEFAULT_SUBCARRIERS)
    initial_threshold = get_initial_threshold()
    publish_rate = getattr(config, "PUBLISH_INTERVAL", 100) or 100
    receiver = CSIReceiver(port=args.udp_port, buffer_size=4000, bind_host=resolved_bind_ip)
    stimulus_sender = StimulusSender(
        target_host=stimulus_targets,
        target_port=args.stimulus_port,
        rate_pps=args.stimulus_rate,
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
            expected_device_count=len(stimulus_targets),
            expected_source_hosts=stimulus_targets,
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
        "summary_last_rendered_at": 0.0,
        "summary_line_count": 0,
        "summary_use_inline": supports_inline_terminal(),
        "calibration_active": detector_kind == "mvs",
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

        if state["calibration_active"] and detector_kind == "mvs":
            process_calibration_packet(device_state, pkt)
            if is_calibration_complete():
                state["calibration_active"] = False
                state["summary_last_rendered_at"] = 0.0
            render_multi_device_summary(now, force=not state["calibration_active"])
            if not save_enabled and maybe_stop_live_session(now):
                return
            return

        detector = device_state["detector"]
        runtime_policy = device_state["runtime_policy"]
        raw_turbulence = None
        if args.log_turbulence:
            raw_turbulence = detector._context._compute_spatial_turbulence_in_buffer(pkt.iq_raw, subcarriers)
        detector.process_packet(pkt.iq_raw, subcarriers)
        filtered_turbulence = detector._context.last_turbulence
        runtime_policy.note_packet()
        metrics = detector.update_state()
        device_state["motion_metric"] = extract_motion_metric(metrics)
        device_state["metric_threshold"] = metrics["threshold"]
        update_ready_gate_state(device_state, now)

        should_publish = device_state["publish_counter"] >= publish_rate
        if runtime_policy.should_evaluate(should_publish):
            effective_state, _ = runtime_policy.apply_state(metrics["state"])
            runtime_policy.after_evaluation()
            device_state["effective_state"] = effective_state
            device_state["status"] = get_device_status(device_state)

            if should_publish:
                motion_metric = device_state["motion_metric"]
                metric_threshold = device_state["metric_threshold"]
                device_state["last_publish_at"] = now
                progress_score = motion_metric / metric_threshold if metric_threshold > 0 else 0.0

                if should_print_publish_details(effective_state):
                    clear_status_block()
                    print(
                        format_detection_publish_line(
                            pps=device_state["pps"],
                            motion_metric=motion_metric,
                            threshold=metric_threshold,
                            effective_state=effective_state,
                            progress=progress_score,
                            device_label=device_state["label"],
                        )
                    )
                    if args.log_turbulence:
                        window_tail = get_ordered_turbulence_tail(detector._context, args.window_tail)
                        print(f"  turbulence: raw={raw_turbulence:.4f} filtered={filtered_turbulence:.4f}")
                        if window_tail:
                            print(f"  tail[{len(window_tail)}]: {format_turbulence_tail(window_tail)}")
                    if args.log_features and detector_kind == "ml" and detector.is_ready():
                        features = detector._extract_features()
                        print(f"  features: {format_feature_vector(features)}")

                device_state["publish_counter"] = 0

        if save_enabled and not state["capture_ready"]:
            ready_summary = summarize_ready_gate()
            if ready_summary["ready"]:
                state["capture_ready"] = True
                state["capture_started_at"] = now
                state["summary_last_rendered_at"] = 0.0

        if save_enabled and state["capture_ready"]:
            if maybe_stop_live_session(now):
                return
            state["capture_packets"].append(pkt)
        elif not save_enabled and maybe_stop_live_session(now):
            return

        render_multi_device_summary(now)

    receiver.add_callback(on_packet)
    signal.signal(signal.SIGINT, handle_sigint)

    print(f"\n{Fore.MAGENTA}╔═══════════════════════════════════════════════════════════╗{Style.RESET_ALL}")
    print(f"{Fore.MAGENTA}║         μESPectre - Live CSI Collect                     ║{Style.RESET_ALL}")
    print(f"{Fore.MAGENTA}╚═══════════════════════════════════════════════════════════╝{Style.RESET_ALL}")
    print()
    print(f"  {Fore.CYAN}Detector:{Style.RESET_ALL}  {detector_kind.upper()}")
    print(f"  {Fore.CYAN}Bind IP:{Style.RESET_ALL}   {resolved_bind_ip}")
    print(f"  {Fore.CYAN}UDP Port:{Style.RESET_ALL}  {args.udp_port}")
    print(f"  {Fore.CYAN}Stimulus target:{Style.RESET_ALL} {', '.join(stimulus_targets)} ({target_mode})")
    print(f"  {Fore.CYAN}Stimulus:{Style.RESET_ALL}  {args.stimulus_rate} pps -> {len(stimulus_targets)} target(s) on UDP {args.stimulus_port}")
    if args.reference_every > 0:
        print(f"  {Fore.CYAN}Reference:{Style.RESET_ALL} every {args.reference_every} packets")
    if detector_kind == "ml":
        print(f"  {Fore.CYAN}Threshold:{Style.RESET_ALL} {initial_threshold:.1f}")
    else:
        threshold_text = raw_threshold_setting if isinstance(raw_threshold_setting, str) else f"{float(raw_threshold_setting):.4f}"
        print(f"  {Fore.CYAN}Threshold:{Style.RESET_ALL} {threshold_text} (after startup calibration)")
        print(f"  {Fore.CYAN}Calibration:{Style.RESET_ALL} {calibration_target_packets} packets/device")
    print(f"  {Fore.CYAN}Window:{Style.RESET_ALL}    {config.SEG_WINDOW_SIZE} pkts")
    print(f"  {Fore.CYAN}Subcarriers:{Style.RESET_ALL} {subcarriers}")
    print(
        f"  {Fore.CYAN}Consecutive hits motion/idle:{Style.RESET_ALL} "
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
    print(f"  {Fore.YELLOW}Make sure the ESPectre streamer firmware is listening for the configured shared stimulus target{Style.RESET_ALL}")
    if detector_kind == "mvs":
        print(f"  {Fore.YELLOW}Please remain still during the startup calibration phase{Style.RESET_ALL}")
    print(f"  {Fore.YELLOW}Press Ctrl+C to stop{Style.RESET_ALL}")
    print()

    try:
        stimulus_sender.start()
        while state["running"]:
            receiver.run(timeout=1.0, quiet=True)
            render_multi_device_summary(time.monotonic(), force=True)
    except KeyboardInterrupt:
        state["interrupted"] = True
        render_multi_device_summary(time.monotonic(), force=True)
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
