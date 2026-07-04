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


def collect_csi_data(args) -> None:
    """Collect labeled CSI data for training on the host."""
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
    print(f"  {Fore.CYAN}Duration:{Style.RESET_ALL}  {args.duration}s per sample")
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
    print(f"  {Fore.YELLOW}Chip type and gain lock status auto-detected from CSI stream{Style.RESET_ALL}")
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
            saved = collector.collect_interactive(num_samples=args.samples, duration=args.duration)
        else:
            saved = collector.collect_timed(duration=args.duration, num_samples=args.samples)
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


def detect_live_motion(args) -> None:
    """Run live ML motion detection on the host from the UDP CSI stream."""
    try:
        from tools.csi_utils import CSICollector, CSIReceiver, StimulusSender, get_default_bind_host
        import config
        from ml_detector import FEATURE_NAMES, ML_DEFAULT_THRESHOLD, ML_METRIC_SCALE, MLDetector
        from runtime_policy import RuntimeMotionPolicy
    except ImportError:
        try:
            from tools.csi_utils import CSICollector, CSIReceiver, StimulusSender, get_default_bind_host
            import src.config as config
            from src.ml_detector import FEATURE_NAMES, ML_DEFAULT_THRESHOLD, ML_METRIC_SCALE, MLDetector
            from src.runtime_policy import RuntimeMotionPolicy
        except ImportError as e:
            print(f"{Fore.RED}❌ Failed to import motion detection modules: {e}{Style.RESET_ALL}")
            raise SystemExit(1)

    def format_progress_bar(score, threshold, width=20):
        threshold_pos = int((threshold / ML_METRIC_SCALE) * width) if threshold > 0 else 0
        filled = int((score / ML_METRIC_SCALE) * width)
        threshold_pos = max(0, min(threshold_pos, width - 1))
        filled = max(0, min(filled, width))
        bar = "["
        for i in range(width):
            if i == threshold_pos:
                bar += "|"
            elif i < filled:
                bar += "#"
            else:
                bar += "-"
        bar += "]"
        percent = int((score / threshold) * 100) if threshold > 0 else 0
        return f"{bar} {percent}%"

    def format_feature_vector(features):
        return " ".join(f"{name}={value:.4f}" for name, value in zip(FEATURE_NAMES, features))

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

    def get_runtime_ml_threshold():
        threshold = getattr(config, "SEG_THRESHOLD", ML_DEFAULT_THRESHOLD)
        if isinstance(threshold, (int, float)):
            return float(threshold)
        return ML_DEFAULT_THRESHOLD

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

    def format_device_label(device_id):
        if device_id is None:
            return "device=unknown"
        return f"device=dev{int(device_id):016x}"

    def compute_sequence_gap(previous_seq, current_seq):
        expected = (previous_seq + 1) & 0xFFFFFFFF
        delta = (current_seq - expected) & 0xFFFFFFFF
        if delta == 0:
            return 0
        if delta >= 0x80000000:
            return 0
        return delta

    def get_packet_device_id(pkt):
        device_id = getattr(pkt, "device_id", None)
        if device_id is None:
            return None
        return int(device_id)

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
        return {
            "device_id": device_id,
            "label": format_device_label(device_id),
            "packet_count": 0,
            "publish_counter": 0,
            "dropped_count": 0,
            "pps": 0,
            "pps_window_started_at": None,
            "pps_window_packets": 0,
            "motion_metric": 0.0,
            "metric_threshold": threshold,
            "effective_state": 0,
            "status": "WARMUP",
            "last_publish_at": None,
            "detector": MLDetector(
                window_size=config.SEG_WINDOW_SIZE,
                threshold=threshold,
                enable_lowpass=config.ENABLE_LOWPASS_FILTER,
                lowpass_cutoff=config.LOWPASS_CUTOFF,
                enable_hampel=config.ENABLE_HAMPEL_FILTER,
                hampel_window=config.HAMPEL_WINDOW,
                hampel_threshold=config.HAMPEL_THRESHOLD,
            ),
            "runtime_policy": RuntimeMotionPolicy(
                evaluation_interval=getattr(config, "EVALUATION_INTERVAL", 25),
                motion_on_hits=getattr(config, "MOTION_ON_HITS", 3),
                motion_off_hits=getattr(config, "MOTION_OFF_HITS", 3),
            ),
        }

    def get_device_state(pkt):
        device_id = get_packet_device_id(pkt)
        device_state = state["devices"].get(device_id)
        if device_state is None:
            device_state = build_device_state(device_id)
            state["devices"][device_id] = device_state
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

    def get_device_status(device_state):
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
            progress_bar = format_progress_bar(device_state["motion_metric"], device_state["metric_threshold"])
            detail_lines.append(
                f"    {device_state['label']} | {progress_bar} | {status:<6} "
                f"mvmt:{device_state['motion_metric']:.4f}/{device_state['metric_threshold']:.4f} "
                f"pkt:{device_state['packet_count']} drop:{device_state['dropped_count']} pps:{device_state['pps']}"
            )

        for waiting_idx in range(observed_count, required_count):
            detail_lines.append(f"    device=waiting-{waiting_idx + 1:02d} | WAITING")

        summary_line = (
            f"  STATUS: DEVICES {observed_count}/{required_count} | "
            f"packets {state['packet_count']} | capture {len(state['capture_packets'])}"
        )
        state["summary_line_count"] = emit_status_block(
            summary_line,
            detail_lines,
            previous_line_count=state["summary_line_count"],
            inline=state["summary_use_inline"],
        )
        state["summary_last_rendered_at"] = now

    capture_enabled = bool(getattr(args, "capture_label", None))
    capture_duration = getattr(args, "capture_duration", None)
    if capture_duration is not None and capture_duration <= 0:
        print(f"{Fore.RED}❌ Capture duration must be > 0 seconds{Style.RESET_ALL}")
        raise SystemExit(1)
    if capture_duration is not None and not capture_enabled:
        print(f"{Fore.RED}❌ --capture-duration requires --capture-label{Style.RESET_ALL}")
        raise SystemExit(1)

    try:
        stimulus_targets, target_mode = _parse_stimulus_targets(args.stimulus_target)
    except ValueError as e:
        print(f"{Fore.RED}❌ {e}{Style.RESET_ALL}")
        raise SystemExit(1)

    resolved_bind_ip = args.bind_ip if args.bind_ip else get_default_bind_host()
    subcarriers = list(config.DEFAULT_SUBCARRIERS)
    threshold = get_runtime_ml_threshold()
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
    if capture_enabled:
        capture_writer = CSICollector(
            label=args.capture_label,
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
        "capture_started_at": None,
        "devices": {},
        "last_seq_by_device": {},
        "summary_last_rendered_at": 0.0,
        "summary_line_count": 0,
        "summary_use_inline": supports_inline_terminal(),
    }

    def handle_sigint(_signum, _frame):
        state["running"] = False
        receiver.stop()

    def on_packet(pkt):
        if not state["running"]:
            return

        state["packet_count"] += 1

        if capture_enabled:
            now = time.monotonic()
            if state["capture_started_at"] is None:
                state["capture_started_at"] = now

            elapsed = now - state["capture_started_at"]
            if capture_duration is None or elapsed <= capture_duration:
                state["capture_packets"].append(pkt)
            else:
                state["running"] = False
                receiver.stop()
                return

        device_state = get_device_state(pkt)
        device_state["packet_count"] += 1
        device_state["publish_counter"] += 1
        device_state["dropped_count"] += check_sequence_by_device(pkt)
        now = time.monotonic()
        update_device_pps(device_state, now)

        detector = device_state["detector"]
        runtime_policy = device_state["runtime_policy"]
        raw_turbulence = None
        if args.log_turbulence:
            raw_turbulence = detector._context._compute_spatial_turbulence_in_buffer(pkt.iq_raw, subcarriers)
        detector.process_packet(pkt.iq_raw, subcarriers)
        filtered_turbulence = detector._context.last_turbulence
        runtime_policy.note_packet()

        should_publish = device_state["publish_counter"] >= publish_rate
        if not runtime_policy.should_evaluate(should_publish):
            return

        metrics = detector.update_state()
        effective_state, _ = runtime_policy.apply_state(metrics["state"])
        runtime_policy.after_evaluation()
        device_state["motion_metric"] = metrics.get("probability", 0.0)
        device_state["metric_threshold"] = metrics["threshold"]
        device_state["effective_state"] = effective_state
        device_state["status"] = get_device_status(device_state)

        if should_publish:
            motion_metric = device_state["motion_metric"]
            metric_threshold = device_state["metric_threshold"]
            state_str = "MOTION" if effective_state == 1 else "IDLE"
            device_state["last_publish_at"] = now

            if should_print_publish_details(effective_state):
                clear_status_block()
                print(
                    f"{device_state['label']} | {format_progress_bar(motion_metric, metric_threshold)} | mvmt:{motion_metric:.4f} "
                    f"thr:{metric_threshold:.4f} | {state_str} | pkt:{device_state['packet_count']} "
                    f"drop:{device_state['dropped_count']} pps:{device_state['pps']}"
                )
                if args.log_turbulence:
                    window_tail = get_ordered_turbulence_tail(detector._context, args.window_tail)
                    print(f"  turbulence: raw={raw_turbulence:.4f} filtered={filtered_turbulence:.4f}")
                    if window_tail:
                        print(f"  tail[{len(window_tail)}]: {format_turbulence_tail(window_tail)}")
                if args.log_features and detector.is_ready():
                    features = detector._extract_features()
                    print(f"  features: {format_feature_vector(features)}")

            device_state["publish_counter"] = 0

        render_multi_device_summary(now)

    receiver.add_callback(on_packet)
    signal.signal(signal.SIGINT, handle_sigint)

    print(f"\n{Fore.MAGENTA}╔═══════════════════════════════════════════════════════════╗{Style.RESET_ALL}")
    print(f"{Fore.MAGENTA}║        μESPectre - Live Motion Detection                  ║{Style.RESET_ALL}")
    print(f"{Fore.MAGENTA}╚═══════════════════════════════════════════════════════════╝{Style.RESET_ALL}")
    print()
    print(f"  {Fore.CYAN}Bind IP:{Style.RESET_ALL}   {resolved_bind_ip}")
    print(f"  {Fore.CYAN}UDP Port:{Style.RESET_ALL}  {args.udp_port}")
    print(f"  {Fore.CYAN}Stimulus target:{Style.RESET_ALL} {', '.join(stimulus_targets)} ({target_mode})")
    print(f"  {Fore.CYAN}Stimulus:{Style.RESET_ALL}  {args.stimulus_rate} pps -> {len(stimulus_targets)} target(s) on UDP {args.stimulus_port}")
    if args.reference_every > 0:
        print(f"  {Fore.CYAN}Reference:{Style.RESET_ALL} every {args.reference_every} packets")
    print(f"  {Fore.CYAN}Threshold:{Style.RESET_ALL} {threshold:.1f}")
    print(f"  {Fore.CYAN}Window:{Style.RESET_ALL}    {config.SEG_WINDOW_SIZE} pkts")
    print(f"  {Fore.CYAN}Subcarriers:{Style.RESET_ALL} {subcarriers}")
    print(f"  {Fore.CYAN}Hits on/off:{Style.RESET_ALL} {getattr(config, 'MOTION_ON_HITS', 3)}/{getattr(config, 'MOTION_OFF_HITS', 3)}")
    print(f"  {Fore.CYAN}Low-pass:{Style.RESET_ALL}  {'ON' if config.ENABLE_LOWPASS_FILTER else 'OFF'}")
    print(f"  {Fore.CYAN}Hampel:{Style.RESET_ALL}    {'ON' if config.ENABLE_HAMPEL_FILTER else 'OFF'}")
    if capture_enabled:
        duration_text = "until Ctrl+C" if capture_duration is None else f"{capture_duration:g}s"
        print(f"  {Fore.CYAN}Capture:{Style.RESET_ALL}   label={args.capture_label} duration={duration_text}")
        if getattr(args, "description", None):
            print(f"  {Fore.CYAN}Description:{Style.RESET_ALL} {args.description}")
    print()
    print(f"  {Fore.YELLOW}Make sure the ESPectre streamer firmware is listening for the configured shared stimulus target{Style.RESET_ALL}")
    print(f"  {Fore.YELLOW}Press Ctrl+C to stop{Style.RESET_ALL}")
    print()

    try:
        stimulus_sender.start()
        while state["running"]:
            receiver.run(timeout=1.0, quiet=True)
            render_multi_device_summary(time.monotonic(), force=True)
    except KeyboardInterrupt:
        render_multi_device_summary(time.monotonic(), force=True)
        pass
    except Exception as e:
        print(f"\n{Fore.RED}❌ Error during live detection: {e}{Style.RESET_ALL}")
        raise SystemExit(1)
    finally:
        stimulus_sender.stop()
        receiver.stop()
        clear_status_block()
        if capture_writer is not None:
            captured_packets = state["capture_packets"]
            if captured_packets:
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
