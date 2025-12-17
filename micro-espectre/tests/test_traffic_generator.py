"""
Micro-ESPectre - Traffic Generator Unit Tests

Tests for TrafficGenerator class in src/traffic_generator.py.
Uses mocks for MicroPython-specific modules.

Author: Francesco Pace <francesco.pace@gmail.com>
License: GPLv3
"""

import pytest
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

# Mock MicroPython modules before importing
mock_network = MagicMock()
mock_network.WLAN = MagicMock()
mock_network.STA_IF = 0
sys.modules['network'] = mock_network

mock_thread = MagicMock()
sys.modules['_thread'] = mock_thread

# Add MicroPython-specific time functions to time module for testing
import time
if not hasattr(time, 'ticks_ms'):
    time.ticks_ms = lambda: int(time.time() * 1000)
if not hasattr(time, 'ticks_us'):
    time.ticks_us = lambda: int(time.time() * 1000000)
if not hasattr(time, 'ticks_diff'):
    time.ticks_diff = lambda t1, t2: t1 - t2
if not hasattr(time, 'sleep_ms'):
    time.sleep_ms = lambda ms: time.sleep(ms / 1000)
if not hasattr(time, 'sleep_us'):
    time.sleep_us = lambda us: time.sleep(us / 1000000)

from traffic_generator import TrafficGenerator, TRAFFIC_RATE_MIN, TRAFFIC_RATE_MAX


@pytest.fixture
def traffic_gen():
    """Create a TrafficGenerator instance"""
    return TrafficGenerator()


@pytest.fixture
def mock_wlan():
    """Create mock WLAN interface"""
    wlan = MagicMock()
    wlan.isconnected.return_value = True
    wlan.ifconfig.return_value = ('192.168.1.100', '255.255.255.0', '192.168.1.1', '8.8.8.8')
    return wlan


class TestTrafficGeneratorInit:
    """Test TrafficGenerator initialization"""
    
    def test_init(self, traffic_gen):
        """Test traffic generator initialization"""
        assert traffic_gen.running is False
        assert traffic_gen.rate_pps == 0
        assert traffic_gen.packet_count == 0
        assert traffic_gen.error_count == 0
        assert traffic_gen.gateway_ip is None
        assert traffic_gen.sock is None
    
    def test_initial_metrics(self, traffic_gen):
        """Test initial metrics values"""
        assert traffic_gen.start_time == 0
        assert traffic_gen.avg_loop_time_ms == 0
        assert traffic_gen.actual_pps == 0


class TestTrafficGeneratorGetters:
    """Test getter methods"""
    
    def test_is_running_false(self, traffic_gen):
        """Test is_running when not running"""
        assert traffic_gen.is_running() is False
    
    def test_is_running_true(self, traffic_gen):
        """Test is_running when running"""
        traffic_gen.running = True
        assert traffic_gen.is_running() is True
    
    def test_get_packet_count(self, traffic_gen):
        """Test get_packet_count"""
        traffic_gen.packet_count = 1234
        assert traffic_gen.get_packet_count() == 1234
    
    def test_get_rate(self, traffic_gen):
        """Test get_rate"""
        traffic_gen.rate_pps = 100
        assert traffic_gen.get_rate() == 100
    
    def test_get_actual_pps(self, traffic_gen):
        """Test get_actual_pps"""
        traffic_gen.actual_pps = 99.5678
        assert traffic_gen.get_actual_pps() == 99.6
    
    def test_get_error_count(self, traffic_gen):
        """Test get_error_count"""
        traffic_gen.error_count = 5
        assert traffic_gen.get_error_count() == 5
    
    def test_get_avg_loop_time_ms(self, traffic_gen):
        """Test get_avg_loop_time_ms"""
        traffic_gen.avg_loop_time_ms = 9.5678
        assert traffic_gen.get_avg_loop_time_ms() == 9.57


class TestTrafficGeneratorGetGatewayIP:
    """Test _get_gateway_ip method"""
    
    def test_get_gateway_ip_success(self, traffic_gen, mock_wlan):
        """Test getting gateway IP successfully"""
        mock_network.WLAN.return_value = mock_wlan
        
        result = traffic_gen._get_gateway_ip()
        
        assert result == '192.168.1.1'
    
    def test_get_gateway_ip_not_connected(self, traffic_gen):
        """Test getting gateway IP when not connected"""
        mock_wlan = MagicMock()
        mock_wlan.isconnected.return_value = False
        mock_network.WLAN.return_value = mock_wlan
        
        result = traffic_gen._get_gateway_ip()
        
        assert result is None
    
    def test_get_gateway_ip_short_ifconfig(self, traffic_gen):
        """Test getting gateway IP with short ifconfig response"""
        mock_wlan = MagicMock()
        mock_wlan.isconnected.return_value = True
        mock_wlan.ifconfig.return_value = ('192.168.1.100',)  # Too short
        mock_network.WLAN.return_value = mock_wlan
        
        result = traffic_gen._get_gateway_ip()
        
        assert result is None
    
    def test_get_gateway_ip_exception(self, traffic_gen):
        """Test getting gateway IP when exception occurs"""
        mock_network.WLAN.side_effect = Exception("Network error")
        
        result = traffic_gen._get_gateway_ip()
        
        assert result is None
        
        # Reset mock
        mock_network.WLAN.side_effect = None


class TestTrafficGeneratorStart:
    """Test start method"""
    
    def test_start_already_running(self, traffic_gen):
        """Test start when already running"""
        traffic_gen.running = True
        
        result = traffic_gen.start(100)
        
        assert result is False
    
    def test_start_invalid_rate_too_low(self, traffic_gen):
        """Test start with rate below minimum"""
        result = traffic_gen.start(-1)
        
        assert result is False
    
    def test_start_invalid_rate_too_high(self, traffic_gen):
        """Test start with rate above maximum"""
        result = traffic_gen.start(TRAFFIC_RATE_MAX + 1)
        
        assert result is False
    
    def test_start_no_gateway_ip(self, traffic_gen):
        """Test start when gateway IP cannot be obtained"""
        mock_wlan = MagicMock()
        mock_wlan.isconnected.return_value = False
        mock_network.WLAN.return_value = mock_wlan
        
        result = traffic_gen.start(100, max_retries=1, retry_delay=0)
        
        assert result is False
        assert traffic_gen.running is False
    
    def test_start_success(self, traffic_gen, mock_wlan):
        """Test successful start"""
        mock_network.WLAN.return_value = mock_wlan
        mock_thread.start_new_thread = MagicMock()
        
        result = traffic_gen.start(100)
        
        assert result is True
        assert traffic_gen.running is True
        assert traffic_gen.rate_pps == 100
        assert traffic_gen.gateway_ip == '192.168.1.1'
        mock_thread.start_new_thread.assert_called_once()
        
        # Cleanup
        traffic_gen.running = False
    
    def test_start_thread_exception(self, traffic_gen, mock_wlan):
        """Test start when thread creation fails"""
        mock_network.WLAN.return_value = mock_wlan
        mock_thread.start_new_thread = MagicMock(side_effect=Exception("Thread error"))
        
        result = traffic_gen.start(100)
        
        assert result is False
        assert traffic_gen.running is False


class TestTrafficGeneratorStop:
    """Test stop method"""
    
    def test_stop_not_running(self, traffic_gen):
        """Test stop when not running"""
        traffic_gen.running = False
        
        # Should not raise exception
        traffic_gen.stop()
        
        assert traffic_gen.running is False
    
    def test_stop_running(self, traffic_gen):
        """Test stop when running"""
        traffic_gen.running = True
        traffic_gen.rate_pps = 100
        
        traffic_gen.stop()
        
        assert traffic_gen.running is False
        assert traffic_gen.rate_pps == 0


class TestTrafficGeneratorConstants:
    """Test module constants"""
    
    def test_rate_min(self):
        """Test minimum rate constant"""
        assert TRAFFIC_RATE_MIN == 0
    
    def test_rate_max(self):
        """Test maximum rate constant"""
        assert TRAFFIC_RATE_MAX == 1000


class TestTrafficGeneratorDnsTask:
    """Test _dns_task method (partial coverage due to MicroPython dependencies)"""
    
    def test_dns_task_socket_creation_failure(self, traffic_gen):
        """Test DNS task handles socket creation failure"""
        traffic_gen.running = True
        traffic_gen.rate_pps = 100
        traffic_gen.gateway_ip = '192.168.1.1'
        
        # Mock socket to fail on creation
        with patch('traffic_generator.socket.socket') as mock_socket:
            mock_socket.side_effect = Exception("Socket error")
            
            traffic_gen._dns_task()
            
            assert traffic_gen.running is False
    
    def test_dns_task_runs_and_stops(self, traffic_gen, mock_wlan):
        """Test DNS task runs and stops correctly"""
        mock_network.WLAN.return_value = mock_wlan
        traffic_gen.rate_pps = 100
        traffic_gen.gateway_ip = '192.168.1.1'
        traffic_gen.running = True
        
        # Mock socket
        mock_sock = MagicMock()
        mock_sock.setblocking = MagicMock()
        mock_sock.sendto = MagicMock()
        mock_sock.close = MagicMock()
        
        packets_sent = [0]
        
        def send_and_stop(*args):
            packets_sent[0] += 1
            if packets_sent[0] >= 3:
                traffic_gen.running = False
        
        mock_sock.sendto.side_effect = send_and_stop
        
        with patch('traffic_generator.socket.socket', return_value=mock_sock):
            traffic_gen._dns_task()
        
        assert mock_sock.sendto.call_count >= 1
        mock_sock.close.assert_called_once()
    
    def test_dns_task_socket_error(self, traffic_gen):
        """Test DNS task handles socket send errors"""
        traffic_gen.rate_pps = 100
        traffic_gen.gateway_ip = '192.168.1.1'
        traffic_gen.running = True
        
        mock_sock = MagicMock()
        mock_sock.setblocking = MagicMock()
        mock_sock.close = MagicMock()
        
        error_count = [0]
        
        def send_with_error(*args):
            error_count[0] += 1
            if error_count[0] >= 3:
                traffic_gen.running = False
            raise OSError("Network unavailable")
        
        mock_sock.sendto.side_effect = send_with_error
        
        with patch('traffic_generator.socket.socket', return_value=mock_sock):
            traffic_gen._dns_task()
        
        assert traffic_gen.error_count >= 1
    
    def test_dns_task_general_exception(self, traffic_gen):
        """Test DNS task handles general exceptions"""
        traffic_gen.rate_pps = 100
        traffic_gen.gateway_ip = '192.168.1.1'
        traffic_gen.running = True
        
        mock_sock = MagicMock()
        mock_sock.setblocking = MagicMock()
        mock_sock.close = MagicMock()
        
        exception_count = [0]
        
        def send_with_exception(*args):
            exception_count[0] += 1
            if exception_count[0] >= 3:
                traffic_gen.running = False
            raise Exception("General error")
        
        mock_sock.sendto.side_effect = send_with_exception
        
        with patch('traffic_generator.socket.socket', return_value=mock_sock):
            traffic_gen._dns_task()
        
        assert traffic_gen.error_count >= 1

