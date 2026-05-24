"""
Micro-ESPectre - WiFi Traffic Generator

Generates UDP/DNS or ICMP ping traffic to ensure continuous CSI data flow.
Essential for maintaining stable CSI packet reception on ESP32 chips.

Author: Francesco Pace <francesco.pace@gmail.com>
License: GPLv3
"""
import socket
import time
import _thread
import network

# Note: No thread lock needed for simple integer operations on MicroPython/ESP32
# Integer reads/writes are atomic on 32-bit systems

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
        self.packet_count = 0
        self.error_count = 0
        self.gateway_ip = None
        self.sock = None
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
    
    def _dns_task(self): 
        """Background task that sends DNS queries (runs with increased stack)"""
        
        # Use DNS queries to generate bidirectional traffic
        # DNS always generates a reply, which triggers CSI
        try:
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            # Set socket to non-blocking mode to avoid delays
            self.sock.setblocking(False)
        except Exception as e:
            print(f"Failed to create socket: {e}")
            self.running = False
            return
        
        # Pre-resolve destination address (avoid repeated lookups)
        dest_addr = (self.gateway_ip, 53)
        send_packet, use_connected_send = self._prepare_sender(self.sock, dest_addr)
        ticks_us = time.ticks_us
        ticks_ms = time.ticks_ms
        ticks_diff = time.ticks_diff
        sleep_ms_fn = time.sleep_ms
        sleep_us_fn = time.sleep_us
        last_send_error_log = 0
        last_task_error_log = 0
        consecutive_enomem = 0

        # Track loop time and pps for diagnostics (updated periodically)
        loop_time_sum_us = 0
        window_start_time = ticks_us()
        window_packet_count = 0
        
        # Microsecond timing with fractional accumulator (aligned with C++ implementation)
        # This compensates for integer division error (e.g., 1000000/100 = 10000µs exact)
        interval_us = 1000000 // self.rate_pps
        remainder_us = 1000000 % self.rate_pps
        accumulator = 0
        
        next_send_time = ticks_us()
        
        while self.running:
            try:
                loop_start = ticks_us()
                
                # Send DNS query to gateway (port 53)
                # Gateway will forward and reply, generating incoming traffic → CSI
                try:
                    if use_connected_send:
                        send_packet(DNS_QUERY)
                    else:
                        send_packet(DNS_QUERY, dest_addr)
                    self.packet_count += 1
                    window_packet_count += 1
                    consecutive_enomem = 0
                        
                except OSError as e:
                    self.error_count += 1
                    now_ms = ticks_ms()
                    if ticks_diff(now_ms, last_send_error_log) >= SEND_ERROR_LOG_INTERVAL_MS:
                        print(f"Socket error: {e}")
                        last_send_error_log = now_ms
                    if self._extract_errno(e) == ENOMEM_ERRNO:
                        consecutive_enomem += 1
                        backoff_ms = SEND_ERROR_BACKOFF_MS * consecutive_enomem
                        if backoff_ms > SEND_ERROR_BACKOFF_MAX_MS:
                            backoff_ms = SEND_ERROR_BACKOFF_MAX_MS
                        sleep_ms_fn(backoff_ms)
                        next_send_time = ticks_us()
                        window_start_time = next_send_time
                        window_packet_count = 0
                        continue
                    consecutive_enomem = 0
                
                # Calculate next send time with fractional accumulator for precise rate
                accumulator += remainder_us
                extra_us = accumulator // self.rate_pps
                accumulator %= self.rate_pps
                
                next_send_time += interval_us + extra_us
                
                # Track loop time for averaging
                loop_time_us = ticks_diff(ticks_us(), loop_start)
                loop_time_sum_us += loop_time_us
                
                # Periodic metrics update (no GC needed - no allocations in loop)
                if window_packet_count >= METRICS_INTERVAL:
                    # Update average loop time (no lock needed - single writer)
                    self.avg_loop_time_ms = (loop_time_sum_us / METRICS_INTERVAL) / 1000
                    loop_time_sum_us = 0
                    
                    # Update actual pps (moving window)
                    window_elapsed = ticks_diff(ticks_us(), window_start_time)
                    if window_elapsed > 0:
                        self.actual_pps = (window_packet_count * 1000000) / window_elapsed
                    
                    window_start_time = ticks_us()
                    window_packet_count = 0
                
                # Sleep until next send time
                now = ticks_us()
                sleep_for_us = ticks_diff(next_send_time, now)
                
                if sleep_for_us > 100:
                    # Convert to ms for sleep (minimum 1ms to yield to other threads)
                    sleep_ms = sleep_for_us // 1000
                    if sleep_ms > 0:
                        sleep_ms_fn(sleep_ms)
                    else:
                        sleep_us_fn(sleep_for_us)
                elif sleep_for_us < -100000:
                    # We're more than 100ms behind, reset timing
                    next_send_time = ticks_us()
                else:
                    # Small sleep to yield
                    sleep_us_fn(100)
                
            except Exception as e:
                self.error_count += 1
                now_ms = ticks_ms()
                if ticks_diff(now_ms, last_task_error_log) >= SEND_ERROR_LOG_INTERVAL_MS:
                    print(f"Traffic generator error: {e}")
                    last_task_error_log = now_ms
                sleep_ms_fn(interval_us // 1000)
        
        # Cleanup
        if self.sock:
            self.sock.close()
            self.sock = None
        
        #print(f"📡 Traffic generator task stopped ({self.packet_count} packets sent, {self.error_count} errors)")

    def _ping_task(self):
        """Background task that sends ICMP echo requests."""
        try:
            self.sock = socket.socket(socket.AF_INET, SOCK_RAW, IPPROTO_ICMP)
            self.sock.setblocking(False)
        except Exception as e:
            print(f"Failed to create ping socket: {e}")
            self.running = False
            return

        dest_addr = (self.gateway_ip, 1)
        send_packet, use_connected_send = self._prepare_sender(self.sock, dest_addr)
        build_ping_packet = self._build_ping_packet
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

        interval_us = 1000000 // self.rate_pps
        remainder_us = 1000000 % self.rate_pps
        accumulator = 0

        next_send_time = ticks_us()

        while self.running:
            try:
                loop_start = ticks_us()

                try:
                    packet = build_ping_packet()
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
                        print(f"Ping socket error: {e}")
                        last_send_error_log = now_ms
                    if self._extract_errno(e) == ENOMEM_ERRNO:
                        consecutive_enomem += 1
                        backoff_ms = SEND_ERROR_BACKOFF_MS * consecutive_enomem
                        if backoff_ms > SEND_ERROR_BACKOFF_MAX_MS:
                            backoff_ms = SEND_ERROR_BACKOFF_MAX_MS
                        sleep_ms_fn(backoff_ms)
                        next_send_time = ticks_us()
                        window_start_time = next_send_time
                        window_packet_count = 0
                        continue
                    consecutive_enomem = 0

                accumulator += remainder_us
                extra_us = accumulator // self.rate_pps
                accumulator %= self.rate_pps

                next_send_time += interval_us + extra_us

                loop_time_us = ticks_diff(ticks_us(), loop_start)
                loop_time_sum_us += loop_time_us

                if window_packet_count >= METRICS_INTERVAL:
                    self.avg_loop_time_ms = (loop_time_sum_us / METRICS_INTERVAL) / 1000
                    loop_time_sum_us = 0

                    window_elapsed = ticks_diff(ticks_us(), window_start_time)
                    if window_elapsed > 0:
                        self.actual_pps = (window_packet_count * 1000000) / window_elapsed

                    window_start_time = ticks_us()
                    window_packet_count = 0

                now = ticks_us()
                sleep_for_us = ticks_diff(next_send_time, now)

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
                sleep_ms_fn(interval_us // 1000)

        if self.sock:
            self.sock.close()
            self.sock = None
    
    def start(self, rate_pps, max_retries=3, retry_delay=2, mode=None):
        """
        Start traffic generator
        
        Args:
            rate_pps: Packets per second (0-1000, recommended: 100)
            max_retries: Number of retries to get gateway IP (default: 3)
            retry_delay: Seconds between retries (default: 2)
            mode: Optional traffic mode override ('dns' or 'ping')
            
        Returns:
            bool: True if started successfully
        """
        if self.running:
            print("Traffic generator already running")
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
        self.rate_pps = rate_pps
        self.start_time = time.ticks_ms()
        self.running = True
        self.ping_sequence = 0
        self._reset_ping_packet()
        
        # Start background task
        try:
            task = self._ping_task if self.mode == MODE_PING else self._dns_task
            _thread.start_new_thread(task, ())
            return True
        except Exception as e:
            print(f"Failed to start traffic generator: {e}")
            self.running = False
            return False
    
    def stop(self):
        """Stop traffic generator"""
        if not self.running:
            return
        
        self.running = False
        time.sleep(0.5)  # Give thread time to stop
        
        #print(f"📡 Traffic generator stopped ({self.packet_count} packets sent, {self.error_count} errors)")
        
        self.rate_pps = 0
    
    def is_running(self):
        """Check if traffic generator is running"""
        return self.running
    
    def get_packet_count(self):
        """Get number of packets sent"""
        return self.packet_count
    
    def get_rate(self):
        """Get current rate in packets per second"""
        return self.rate_pps

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
    
