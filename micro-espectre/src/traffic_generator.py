"""
Micro-ESPectre - WiFi Traffic Generator
Generates continuous DNS queries to ensure CSI data availability

Author: Francesco Pace <francesco.pace@gmail.com>
License: GPLv3
"""
import socket
import time
import _thread
import network
from src.config import TRAFFIC_RATE_MIN, TRAFFIC_RATE_MAX

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
        self.thread_lock = _thread.allocate_lock()
        
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
        import gc
        
        interval_ms = 1000 // self.rate_pps if self.rate_pps > 0 else 1000
        
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
        
        # print(f"📡 Traffic generator task started (DNS queries to {self.gateway_ip}, rate: {self.rate_pps} pps, interval: {interval_ms} ms)")
        
        # Periodic garbage collection counter
        gc_counter = 0
        
        # Simple DNS query for google.com (type A)
        # This is a minimal DNS query that will get a response
        dns_query = bytes([
            0x00, 0x01,  # Transaction ID
            0x01, 0x00,  # Flags: standard query
            0x00, 0x01,  # Questions: 1
            0x00, 0x00,  # Answer RRs: 0
            0x00, 0x00,  # Authority RRs: 0
            0x00, 0x00,  # Additional RRs: 0
            # Query: google.com
            0x06, 0x67, 0x6f, 0x6f, 0x67, 0x6c, 0x65,  # "google"
            0x03, 0x63, 0x6f, 0x6d,  # "com"
            0x00,  # End of name
            0x00, 0x01,  # Type: A
            0x00, 0x01   # Class: IN
        ])
        
        while self.running:
            try:
                # Periodic garbage collection (every 50 packets to reduce overhead)
                gc_counter += 1
                if gc_counter >= 50:
                    gc.collect()
                    gc_counter = 0
                
                loop_start = time.ticks_ms()
                
                # Send DNS query to gateway (port 53)
                # Gateway will forward and reply, generating incoming traffic → CSI
                try:
                    # Send DNS query to gateway
                    # Responses accumulate in socket buffer until full, then OS drops them
                    # No need to read or manage buffer - OS handles overflow automatically
                    self.sock.sendto(dns_query, (self.gateway_ip, 53))
                    
                    with self.thread_lock:
                        self.packet_count += 1
                        
                except OSError as e:
                    # Socket error (e.g., network unavailable)
                    with self.thread_lock:
                        self.error_count += 1
                    if self.error_count % 10 == 1:
                        print(f"Socket error: {e}")
                
                # Calculate how long the loop took and sleep for the remainder
                loop_time = time.ticks_diff(time.ticks_ms(), loop_start)
                sleep_time = interval_ms - loop_time
                
                # Sleep for the target interval (minimum 1ms to yield to other threads)
                if sleep_time > 1:
                    time.sleep_ms(sleep_time)
                else:
                    time.sleep_ms(1)
                
            except Exception as e:
                with self.thread_lock:
                    self.error_count += 1
                
                # Log occasional errors
                if self.error_count % 10 == 1:
                    print(f"Traffic generator error: {e}")
                
                time.sleep_ms(interval_ms)
        
        # Cleanup
        if self.sock:
            self.sock.close()
            self.sock = None
        
        #print(f"📡 Traffic generator task stopped ({self.packet_count} packets sent, {self.error_count} errors)")
    
    def start(self, rate_pps):
        """
        Start traffic generator
        
        Args:
            rate_pps: Packets per second (0-1000, recommended: 100)
            
        Returns:
            bool: True if started successfully
        """
        if self.running:
            print("Traffic generator already running")
            return False
        
        if rate_pps < TRAFFIC_RATE_MIN or rate_pps > TRAFFIC_RATE_MAX:
            print(f"Invalid rate: {rate_pps} (must be {TRAFFIC_RATE_MIN}-{TRAFFIC_RATE_MAX} packets/sec)")
            return False
        
        # Get gateway IP
        self.gateway_ip = self._get_gateway_ip()
        if not self.gateway_ip:
            print("Failed to get gateway IP")
            return False
        
        # Reset counters
        self.packet_count = 0
        self.error_count = 0
        self.rate_pps = rate_pps
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
        with self.thread_lock:
            return self.packet_count
    
    def get_rate(self):
        """Get current rate in packets per second"""
        return self.rate_pps
    
    def set_rate(self, rate_pps):
        """
        Change traffic rate (restarts generator)
        
        Args:
            rate_pps: New rate in packets per second (0-1000)
        """
        if not self.running:
            print("Cannot set rate: traffic generator not running")
            return False
        
        if rate_pps < TRAFFIC_RATE_MIN or rate_pps > TRAFFIC_RATE_MAX:
            print(f"Invalid rate: {rate_pps} (must be {TRAFFIC_RATE_MIN}-{TRAFFIC_RATE_MAX} packets/sec)")
            return False
        
        if rate_pps == self.rate_pps:
            return True  # No change needed
        
        # Restart with new rate
        self.stop()
        time.sleep(0.5)
        return self.start(rate_pps)
