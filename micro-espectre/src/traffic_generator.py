"""
Micro-ESPectre - WiFi Traffic Generator

Generates UDP traffic to ensure continuous CSI data flow.
Essential for maintaining stable CSI packet reception on ESP32-C6.

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

class TrafficGenerator:
    """WiFi traffic generator using DNS queries"""
    
    def __init__(self):
        """Initialize traffic generator"""
        self.running = False
        self.rate_pps = 0
        self.packet_count = 0
        self.error_count = 0
        self.gateway_ip = None
        self.sock = None
        self.start_time = 0  # Time when generator started (ticks_ms)
        self.avg_loop_time_ms = 0  # Average loop time for diagnostics
        self.actual_pps = 0  # Actual packets per second (moving window)
        
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
        
        # Minimal DNS query for root domain (smallest possible valid query)
        # 17 bytes instead of 29 for google.com
        dns_query = bytes([
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
        
        # Track loop time and pps for diagnostics (updated periodically)
        loop_time_sum_us = 0
        window_start_time = time.ticks_us()
        window_packet_count = 0
        
        # Microsecond timing with fractional accumulator (aligned with C++ implementation)
        # This compensates for integer division error (e.g., 1000000/100 = 10000µs exact)
        interval_us = 1000000 // self.rate_pps
        remainder_us = 1000000 % self.rate_pps
        accumulator = 0
        
        next_send_time = time.ticks_us()
        
        while self.running:
            try:
                loop_start = time.ticks_us()
                
                # Send DNS query to gateway (port 53)
                # Gateway will forward and reply, generating incoming traffic → CSI
                try:
                    self.sock.sendto(dns_query, dest_addr)
                    self.packet_count += 1
                    window_packet_count += 1
                        
                except OSError as e:
                    # Socket error (e.g., network unavailable, ENOMEM)
                    self.error_count += 1
                    if self.error_count % 100 == 1:
                        print(f"Socket error: {e}")
                
                # Calculate next send time with fractional accumulator for precise rate
                accumulator += remainder_us
                extra_us = accumulator // self.rate_pps
                accumulator %= self.rate_pps
                
                next_send_time += interval_us + extra_us
                
                # Track loop time for averaging
                loop_time_us = time.ticks_diff(time.ticks_us(), loop_start)
                loop_time_sum_us += loop_time_us
                
                # Periodic metrics update (no GC needed - no allocations in loop)
                if window_packet_count >= METRICS_INTERVAL:
                    # Update average loop time (no lock needed - single writer)
                    self.avg_loop_time_ms = (loop_time_sum_us / METRICS_INTERVAL) / 1000
                    loop_time_sum_us = 0
                    
                    # Update actual pps (moving window)
                    window_elapsed = time.ticks_diff(time.ticks_us(), window_start_time)
                    if window_elapsed > 0:
                        self.actual_pps = (window_packet_count * 1000000) / window_elapsed
                    
                    window_start_time = time.ticks_us()
                    window_packet_count = 0
                
                # Sleep until next send time
                now = time.ticks_us()
                sleep_us = time.ticks_diff(next_send_time, now)
                
                if sleep_us > 100:
                    # Convert to ms for sleep (minimum 1ms to yield to other threads)
                    sleep_ms = sleep_us // 1000
                    if sleep_ms > 0:
                        time.sleep_ms(sleep_ms)
                    else:
                        time.sleep_us(sleep_us)
                elif sleep_us < -100000:
                    # We're more than 100ms behind, reset timing
                    next_send_time = time.ticks_us()
                else:
                    # Small sleep to yield
                    time.sleep_us(100)
                
            except Exception as e:
                self.error_count += 1
                
                # Log occasional errors
                if self.error_count % 10 == 1:
                    print(f"Traffic generator error: {e}")
                
                time.sleep_ms(interval_us // 1000)
        
        # Cleanup
        if self.sock:
            self.sock.close()
            self.sock = None
        
        #print(f"📡 Traffic generator task stopped ({self.packet_count} packets sent, {self.error_count} errors)")
    
    def start(self, rate_pps, max_retries=3, retry_delay=2):
        """
        Start traffic generator
        
        Args:
            rate_pps: Packets per second (0-1000, recommended: 100)
            max_retries: Number of retries to get gateway IP (default: 3)
            retry_delay: Seconds between retries (default: 2)
            
        Returns:
            bool: True if started successfully
        """
        if self.running:
            print("Traffic generator already running")
            return False
        
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
        
        # Start background task
        try:
            _thread.start_new_thread(self._dns_task, ())
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
    
    def get_actual_pps(self):
        """Get actual packets per second (moving window)"""
        return round(self.actual_pps, 1)
    
    def get_error_count(self):
        """Get number of errors"""
        return self.error_count
    
    def get_avg_loop_time_ms(self):
        """Get average loop time in milliseconds"""
        return round(self.avg_loop_time_ms, 2)
    
