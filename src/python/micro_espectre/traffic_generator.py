# SPDX-License-Identifier: GPL-3.0-only
# Commercial licensing available under separate agreement; see LICENSING.md.
"""
Micro-ESPectre - WiFi Traffic Generator

Generates UDP/DNS or ICMP ping traffic to ensure continuous CSI data flow.
Essential for maintaining stable CSI packet reception on ESP32 chips.

Author: Francesco Pace <francesco.pace@gmail.com>
"""
import socket
import time
import _thread
import network

# A generation token coordinates lifecycle changes without requiring a lock or
# a MicroPython thread join. Each worker owns its socket and exits when a newer
# start or stop invalidates its generation.

TRAFFIC_RATE_MIN = 0          # Minimum rate (0=disabled)
TRAFFIC_RATE_MAX = 1000       # Maximum rate (packets per second)
METRICS_INTERVAL = 500        # Metrics update interval (packets, ~5s at 100pps)
MODE_DNS = "dns"
MODE_PING = "ping"
ICMP_ECHO_REQUEST = 8
ICMP_ECHO_PACKET_SIZE = 8
SEND_ERROR_LOG_INTERVAL_MS = 1000
SEND_ERROR_BACKOFF_MS = 5
SEND_ERROR_BACKOFF_MAX_MS = 100
ENOMEM_ERRNO = 12
IPPROTO_ICMP = getattr(socket, "IPPROTO_ICMP", 1)
SOCK_RAW = getattr(socket, "SOCK_RAW", 3)

# Minimal DNS query for root domain (smallest valid query).
DNS_QUERY = bytes([
    0x00, 0x01,  # Transaction ID
    0x01, 0x00,  # Flags: standard query
    0x00, 0x01,  # Questions: 1
    0x00, 0x00,  # Answer RRs: 0
    0x00, 0x00,  # Authority RRs: 0
    0x00, 0x00,  # Additional RRs: 0
    0x00,        # Root domain (empty label)
    0x00, 0x01,  # Type: A
    0x00, 0x01   # Class: IN
])

class TrafficGenerator:
    """WiFi traffic generator using DNS queries or ICMP ping."""
    
    def __init__(self, mode=MODE_PING):
        """Initialize traffic generator."""
        self.running = False
        self.rate_pps = 0
        self.target_pps = 0
        self.packet_count = 0
        self.error_count = 0
        self.gateway_ip = None
        self.sock = None
        self._worker_generation = 0
        self.mode = self._normalize_mode(mode)
        self.start_time = 0  # Time when generator started (ticks_ms)
        self.avg_loop_time_ms = 0  # Average loop time for diagnostics
        self.actual_pps = 0  # Actual packets per second (moving window)
        self.ping_identifier = time.ticks_ms() & 0xFFFF
        self.ping_sequence = 0
        self._ping_packet = bytearray(ICMP_ECHO_PACKET_SIZE)
        self._reset_ping_packet()

    def _reset_ping_packet(self):
        """Initialize the reusable ICMP echo-request packet buffer."""
        packet = self._ping_packet
        packet[0] = ICMP_ECHO_REQUEST
        packet[1] = 0  # code
        packet[2] = 0  # checksum high byte
        packet[3] = 0  # checksum low byte
        packet[4] = (self.ping_identifier >> 8) & 0xFF
        packet[5] = self.ping_identifier & 0xFF
        packet[6] = 0  # sequence high byte
        packet[7] = 0  # sequence low byte

    def _normalize_mode(self, mode):
        """Validate and normalize the traffic generator mode."""
        mode = (mode or MODE_PING).lower()
        if mode not in (MODE_DNS, MODE_PING):
            raise ValueError(f"Invalid traffic generator mode: {mode}")
        return mode

    def set_mode(self, mode):
        """Set traffic generation mode while stopped."""
        if self.running:
            print("Cannot change traffic generator mode while running")
            return False
        self.mode = self._normalize_mode(mode)
        return True
        
    def _get_gateway_ip(self):
        """Get gateway IP address from network interface"""
        try:
            wlan = network.WLAN(network.STA_IF)
            if not wlan.isconnected():
                return None
            
            # ifconfig returns: (ip, netmask, gateway, dns)
            ip_info = wlan.ifconfig()
            if len(ip_info) >= 3:
                return ip_info[2]  # Gateway IP
            return None
        except Exception as e:
            print(f"Error getting gateway IP: {e}")
            return None

    @staticmethod
    def _extract_errno(exc):
        """Best-effort errno extraction across CPython and MicroPython."""
        err_no = getattr(exc, 'errno', None)
        if err_no is not None:
            return err_no
        if exc.args:
            first_arg = exc.args[0]
            if isinstance(first_arg, int):
                return first_arg
        return None

    @staticmethod
    def _prepare_sender(sock, dest_addr):
        """
        Prefer connect()+send() to avoid passing destination metadata each send.

        Falls back to sendto() when the socket/protocol does not support connect().
        """
        try:
            sock.connect(dest_addr)
            return sock.send, True
        except Exception:
            return sock.sendto, False

    def _build_ping_packet(self):
        """Build a minimal ICMP echo-request packet in a reusable buffer."""
        packet = self._ping_packet
        seq = self.ping_sequence

        # Zero checksum before recomputing and update only the changing fields.
        packet[2] = 0
        packet[3] = 0
        packet[6] = (seq >> 8) & 0xFF
        packet[7] = seq & 0xFF

        checksum = self._checksum(packet)
        packet[2] = (checksum >> 8) & 0xFF
        packet[3] = checksum & 0xFF

        self.ping_sequence = (seq + 1) & 0xFFFF
        return packet

    def _checksum(self, data):
        """Compute the standard ICMP checksum."""
        checksum = 0
        data_len = len(data)
        limit = data_len - 1

        for i in range(0, limit, 2):
            word = (data[i] << 8) + data[i + 1]
            checksum += word
            checksum = (checksum & 0xFFFF) + (checksum >> 16)

        if data_len % 2:
            checksum += data[-1] << 8
            checksum = (checksum & 0xFFFF) + (checksum >> 16)

        return (~checksum) & 0xFFFF
    
    def _run_sender_task(self, mode, generation=None):
        """Run the shared paced send loop for DNS and ICMP traffic."""
        if generation is None:
            generation = self._worker_generation
        if self.rate_pps <= 0 or generation != self._worker_generation:
            if generation == self._worker_generation:
                self.running = False
            return

        is_ping = mode == MODE_PING
        sock = None
        try:
            if is_ping:
                sock = socket.socket(socket.AF_INET, SOCK_RAW, IPPROTO_ICMP)
            else:
                sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sock.setblocking(False)
            if generation != self._worker_generation:
                sock.close()
                return
            self.sock = sock
        except Exception as e:
            label = "ping socket" if is_ping else "socket"
            print(f"Failed to create {label}: {e}")
            if generation == self._worker_generation:
                self.running = False
            return

        dest_addr = (self.gateway_ip, 1 if is_ping else 53)
        send_packet, use_connected_send = self._prepare_sender(sock, dest_addr)
        build_packet = self._build_ping_packet if is_ping else None
        send_error_label = "Ping socket error" if is_ping else "Socket error"
        ticks_us = time.ticks_us
        ticks_ms = time.ticks_ms
        ticks_diff = time.ticks_diff
        sleep_ms_fn = time.sleep_ms
        sleep_us_fn = time.sleep_us
        last_send_error_log = 0
        last_task_error_log = 0
        consecutive_enomem = 0
        loop_time_sum_us = 0
        window_start_time = ticks_us()
        window_packet_count = 0
        paced_rate = max(1, int(self.rate_pps))
        interval_us = 1000000 // paced_rate
        remainder_us = 1000000 % paced_rate
        accumulator = 0
        next_send_time = ticks_us()

        while self.running and generation == self._worker_generation:
            try:
                loop_start = ticks_us()
                current_rate = max(1, int(self.rate_pps))
                if current_rate != paced_rate:
                    paced_rate = current_rate
                    interval_us = 1000000 // paced_rate
                    remainder_us = 1000000 % paced_rate
                    accumulator = 0
                    next_send_time = ticks_us()
                try:
                    packet = build_packet() if is_ping else DNS_QUERY
                    if use_connected_send:
                        send_packet(packet)
                    else:
                        send_packet(packet, dest_addr)
                    self.packet_count += 1
                    window_packet_count += 1
                    consecutive_enomem = 0
                except OSError as e:
                    self.error_count += 1
                    now_ms = ticks_ms()
                    if ticks_diff(now_ms, last_send_error_log) >= SEND_ERROR_LOG_INTERVAL_MS:
                        print(f"{send_error_label}: {e}")
                        last_send_error_log = now_ms
                    if self._extract_errno(e) == ENOMEM_ERRNO:
                        consecutive_enomem += 1
                        backoff_ms = min(
                            SEND_ERROR_BACKOFF_MS * consecutive_enomem,
                            SEND_ERROR_BACKOFF_MAX_MS,
                        )
                        sleep_ms_fn(backoff_ms)
                        next_send_time = ticks_us()
                        window_start_time = next_send_time
                        window_packet_count = 0
                        loop_time_sum_us = 0
                        continue
                    consecutive_enomem = 0

                accumulator += remainder_us
                if accumulator >= paced_rate:
                    extra_us = 1
                    accumulator -= paced_rate
                else:
                    extra_us = 0
                next_send_time += interval_us + extra_us
                loop_time_sum_us += ticks_diff(ticks_us(), loop_start)

                if window_packet_count >= METRICS_INTERVAL:
                    self.avg_loop_time_ms = (loop_time_sum_us / window_packet_count) / 1000
                    window_elapsed = ticks_diff(ticks_us(), window_start_time)
                    if window_elapsed > 0:
                        self.actual_pps = (window_packet_count * 1000000) / window_elapsed
                    loop_time_sum_us = 0
                    window_start_time = ticks_us()
                    window_packet_count = 0

                sleep_for_us = ticks_diff(next_send_time, ticks_us())
                if sleep_for_us > 100:
                    sleep_ms = sleep_for_us // 1000
                    if sleep_ms > 0:
                        sleep_ms_fn(sleep_ms)
                    else:
                        sleep_us_fn(sleep_for_us)
                elif sleep_for_us < -100000:
                    next_send_time = ticks_us()
                else:
                    sleep_us_fn(100)
            except Exception as e:
                self.error_count += 1
                now_ms = ticks_ms()
                if ticks_diff(now_ms, last_task_error_log) >= SEND_ERROR_LOG_INTERVAL_MS:
                    print(f"Traffic generator error: {e}")
                    last_task_error_log = now_ms
                sleep_ms_fn(max(1, interval_us // 1000))

        sock.close()
        if self.sock is sock:
            self.sock = None

    def _dns_task(self, generation=None):
        """Background task that sends DNS queries."""
        self._run_sender_task(MODE_DNS, generation)

    def _ping_task(self, generation=None):
        """Background task that sends ICMP echo requests."""
        self._run_sender_task(MODE_PING, generation)
    
    def start(self, rate_pps, max_retries=3, retry_delay=2, mode=None):
        """
        Start traffic generator
        
        Args:
            rate_pps: Target valid CSI rate (0-1000, recommended: 100)
            max_retries: Number of retries to get gateway IP (default: 3)
            retry_delay: Seconds between retries (default: 2)
            mode: Optional traffic mode override ('dns' or 'ping')
            
        Returns:
            bool: True if started successfully
        """
        if self.running:
            print("Traffic generator already running")
            return False

        if rate_pps == 0:
            self.rate_pps = 0
            self.target_pps = 0
            return False

        if mode is not None:
            self.mode = self._normalize_mode(mode)
        
        if rate_pps < TRAFFIC_RATE_MIN or rate_pps > TRAFFIC_RATE_MAX:
            print(f"Invalid rate: {rate_pps} (must be {TRAFFIC_RATE_MIN}-{TRAFFIC_RATE_MAX} packets/sec)")
            return False
        
        # Get gateway IP with retries
        for attempt in range(1, max_retries + 1):
            self.gateway_ip = self._get_gateway_ip()
            if self.gateway_ip:
                break
            print(f"Failed to get gateway IP (attempt {attempt}/{max_retries})")
            if attempt < max_retries:
                time.sleep(retry_delay)
        
        if not self.gateway_ip:
            print(f"ERROR: Could not get gateway IP after {max_retries} attempts")
            return False
        
        # Reset counters
        self.packet_count = 0
        self.error_count = 0
        self.actual_pps = 0
        self.avg_loop_time_ms = 0
        self.target_pps = rate_pps
        self.rate_pps = rate_pps
        self.start_time = time.ticks_ms()
        self._worker_generation += 1
        generation = self._worker_generation
        self.running = True
        self.ping_sequence = 0
        self._reset_ping_packet()
        
        # Start background task
        try:
            task = self._ping_task if self.mode == MODE_PING else self._dns_task
            _thread.start_new_thread(task, (generation,))
            return True
        except Exception as e:
            print(f"Failed to start traffic generator: {e}")
            self.running = False
            return False
    
    def stop(self):
        """Invalidate the active worker without blocking the sensing loop."""
        if not self.running:
            return

        self.running = False
        self._worker_generation += 1
        self.rate_pps = 0
        self.target_pps = 0

    def is_running(self):
        """Check if traffic generator is running"""
        return self.running
    
    def get_packet_count(self):
        """Get number of successful sends"""
        return self.packet_count
    
    def get_rate(self):
        """Get current send rate in packets per second"""
        return self.rate_pps

    def get_target_rate(self):
        """Get configured valid-CSI target rate."""
        return self.target_pps

    def get_mode(self):
        """Get current traffic generation mode."""
        return self.mode

    def get_actual_pps(self):
        """Get actual packets per second (moving window)"""
        return round(self.actual_pps, 1)
    
    def get_error_count(self):
        """Get number of errors"""
        return self.error_count
    
    def get_avg_loop_time_ms(self):
        """Get average loop time in milliseconds"""
        return round(self.avg_loop_time_ms, 2)
    
