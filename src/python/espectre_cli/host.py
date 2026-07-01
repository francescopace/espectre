"""Host-side Micro-ESPectre tools."""

from __future__ import annotations

import sys

from .common import Path, REPO_ROOT, WEB_UI_FILE, Fore, Style, signal, time, webbrowser


_WEB_UI_FILES = {
    "mqtt": WEB_UI_FILE,
    "ble": REPO_ROOT / "tools" / "web" / "espectre-ble.html",
    "theremin": REPO_ROOT / "tools" / "web" / "espectre-theremin.html",
}


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
            print("    2. Collect samples: ./espectre micro collect --label wave --samples 10 --streamer-ip 192.168.1.50")
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
        print("  ./espectre micro collect --label wave --samples 10 --streamer-ip 192.168.1.50")
        print("  ./espectre micro collect --label static_presence --duration 10 --streamer-ip 192.168.1.50")
        print("  ./espectre micro collect --info")
        raise SystemExit(1)

    if not args.streamer_ip:
        print(f"{Fore.RED}❌ Streamer IP required. Use --streamer-ip <device_ip>{Style.RESET_ALL}")
        raise SystemExit(1)

    if args.start_delay < 0:
        print(f"{Fore.RED}❌ Start delay must be >= 0 seconds{Style.RESET_ALL}")
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
    print(f"  {Fore.CYAN}Streamer IP:{Style.RESET_ALL} {args.streamer_ip}")
    print(f"  {Fore.CYAN}Stimulus:{Style.RESET_ALL}  {args.stimulus_rate} pps -> {args.streamer_ip}:{args.stimulus_port}")
    if args.reference_every > 0:
        print(f"  {Fore.CYAN}Reference:{Style.RESET_ALL} every {args.reference_every} packets")
    if args.description:
        print(f"  {Fore.CYAN}Description:{Style.RESET_ALL} {args.description}")
    print()
    print(f"  {Fore.YELLOW}Chip type and gain lock status auto-detected from CSI stream{Style.RESET_ALL}")
    print(f"  {Fore.YELLOW}Make sure the ESPectre streamer firmware is running and reachable at the configured IP{Style.RESET_ALL}")
    print()

    collector = CSICollector(
        label=args.label,
        port=args.udp_port,
        contributor=args.contributor,
        description=args.description,
        bind_host=resolved_bind_ip,
    )
    stimulus_sender = StimulusSender(
        target_host=args.streamer_ip,
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
            print(f"{Fore.GREEN}✅ Collected {len(saved)} samples for label '{args.label}'{Style.RESET_ALL}")
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

    capture_enabled = bool(getattr(args, "capture_label", None))
    capture_duration = getattr(args, "capture_duration", None)
    if capture_duration is not None and capture_duration <= 0:
        print(f"{Fore.RED}❌ Capture duration must be > 0 seconds{Style.RESET_ALL}")
        raise SystemExit(1)
    if capture_duration is not None and not capture_enabled:
        print(f"{Fore.RED}❌ --capture-duration requires --capture-label{Style.RESET_ALL}")
        raise SystemExit(1)

    resolved_bind_ip = args.bind_ip if args.bind_ip else get_default_bind_host()
    subcarriers = list(config.DEFAULT_SUBCARRIERS)
    threshold = get_runtime_ml_threshold()
    detector = MLDetector(
        window_size=config.SEG_WINDOW_SIZE,
        threshold=threshold,
        enable_lowpass=config.ENABLE_LOWPASS_FILTER,
        lowpass_cutoff=config.LOWPASS_CUTOFF,
        enable_hampel=config.ENABLE_HAMPEL_FILTER,
        hampel_window=config.HAMPEL_WINDOW,
        hampel_threshold=config.HAMPEL_THRESHOLD,
    )
    runtime_policy = RuntimeMotionPolicy(
        evaluation_interval=getattr(config, "EVALUATION_INTERVAL", 25),
        motion_on_hits=getattr(config, "MOTION_ON_HITS", 3),
        motion_off_hits=getattr(config, "MOTION_OFF_HITS", 3),
    )
    publish_rate = getattr(config, "PUBLISH_INTERVAL", 100) or 100
    receiver = CSIReceiver(port=args.udp_port, buffer_size=4000, bind_host=resolved_bind_ip)
    stimulus_sender = StimulusSender(
        target_host=args.streamer_ip,
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
        )

    state = {
        "running": True,
        "packet_count": 0,
        "publish_counter": 0,
        "capture_packets": [],
        "capture_started_at": None,
    }

    def should_log_publish(effective_state):
        if not args.log_only_motion:
            return True
        return effective_state == 1

    def handle_sigint(_signum, _frame):
        state["running"] = False
        receiver.stop()

    def on_packet(pkt):
        if not state["running"]:
            return

        state["packet_count"] += 1
        state["publish_counter"] += 1

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

        raw_turbulence = None
        if args.log_turbulence:
            raw_turbulence = detector._context._compute_spatial_turbulence_in_buffer(pkt.iq_raw, subcarriers)
        detector.process_packet(pkt.iq_raw, subcarriers)
        filtered_turbulence = detector._context.last_turbulence
        runtime_policy.note_packet()

        should_publish = state["publish_counter"] >= publish_rate
        if not runtime_policy.should_evaluate(should_publish):
            return

        metrics = detector.update_state()
        effective_state, _ = runtime_policy.apply_state(metrics["state"])
        runtime_policy.after_evaluation()

        if should_publish:
            motion_metric = metrics.get("probability", 0.0)
            metric_threshold = metrics["threshold"]
            progress_bar = format_progress_bar(motion_metric, metric_threshold)
            state_str = "MOTION" if effective_state == 1 else "IDLE"

            if should_log_publish(effective_state):
                print(
                    f"{progress_bar} | mvmt:{motion_metric:.4f} thr:{metric_threshold:.4f} | "
                    f"{state_str} | pkt:{state['packet_count']} drop:{receiver.dropped_count} "
                    f"pps:{receiver.pps}"
                )
                if args.log_turbulence:
                    window_tail = get_ordered_turbulence_tail(detector._context, args.window_tail)
                    print(f"  turbulence: raw={raw_turbulence:.4f} filtered={filtered_turbulence:.4f}")
                    if window_tail:
                        print(f"  tail[{len(window_tail)}]: {format_turbulence_tail(window_tail)}")
                if args.log_features and detector.is_ready():
                    features = detector._extract_features()
                    print(f"  features: {format_feature_vector(features)}")

            state["publish_counter"] = 0

    receiver.add_callback(on_packet)
    signal.signal(signal.SIGINT, handle_sigint)

    print(f"\n{Fore.MAGENTA}╔═══════════════════════════════════════════════════════════╗{Style.RESET_ALL}")
    print(f"{Fore.MAGENTA}║        μESPectre - Live Motion Detection                  ║{Style.RESET_ALL}")
    print(f"{Fore.MAGENTA}╚═══════════════════════════════════════════════════════════╝{Style.RESET_ALL}")
    print()
    print(f"  {Fore.CYAN}Bind IP:{Style.RESET_ALL}   {resolved_bind_ip}")
    print(f"  {Fore.CYAN}UDP Port:{Style.RESET_ALL}  {args.udp_port}")
    print(f"  {Fore.CYAN}Streamer IP:{Style.RESET_ALL} {args.streamer_ip}")
    print(f"  {Fore.CYAN}Stimulus:{Style.RESET_ALL}  {args.stimulus_rate} pps -> {args.streamer_ip}:{args.stimulus_port}")
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
    print(f"  {Fore.YELLOW}Make sure the ESPectre streamer firmware is running and reachable at the configured IP{Style.RESET_ALL}")
    print(f"  {Fore.YELLOW}Press Ctrl+C to stop{Style.RESET_ALL}")
    print()

    try:
        stimulus_sender.start()
        while state["running"]:
            receiver.run(timeout=1.0, quiet=True)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"\n{Fore.RED}❌ Error during live detection: {e}{Style.RESET_ALL}")
        raise SystemExit(1)
    finally:
        stimulus_sender.stop()
        receiver.stop()
        if capture_writer is not None:
            captured_packets = state["capture_packets"]
            if captured_packets:
                saved_path = capture_writer.save_sample(captured_packets)
                if saved_path:
                    print(
                        f"{Fore.GREEN}✅ Saved live capture: {saved_path.name} "
                        f"({len(captured_packets)} packets){Style.RESET_ALL}"
                    )
                else:
                    print(f"{Fore.RED}❌ Live capture had no packets to save{Style.RESET_ALL}")
            else:
                print(f"{Fore.YELLOW}No live capture packets received; nothing saved{Style.RESET_ALL}")
        print(f"\n{Fore.GREEN}Done.{Style.RESET_ALL}\n")
