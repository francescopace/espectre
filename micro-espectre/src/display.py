"""
Micro-ESPectre - Display Driver

ST7789 display driver for Waveshare ESP32-S3 1.47" Touch Display.
Resolution: 172x320, 262K colors.

Provides simple status display for ESPectre motion detection.

Author: Francesco Pace <francesco.pace@gmail.com>
License: GPLv3
"""

from machine import Pin, SPI
import time
import framebuf

# Waveshare ESP32-S3 1.47" Touch Display pinout (from official documentation)
# https://www.waveshare.com/wiki/ESP32-S3-Touch-LCD-1.47
PIN_SCLK = 38   # LCD_CLK
PIN_MOSI = 39   # LCD_DIN
PIN_CS = 21     # LCD_CS
PIN_DC = 45     # LCD_DC
PIN_RST = 40    # LCD_RST
PIN_BL = 46     # LCD_BL (Backlight)

# Display dimensions (portrait mode)
DISPLAY_WIDTH = 172
DISPLAY_HEIGHT = 320

# ST7789 Commands
ST7789_SWRESET = 0x01
ST7789_SLPOUT = 0x11
ST7789_NORON = 0x13
ST7789_INVON = 0x21
ST7789_DISPON = 0x29
ST7789_CASET = 0x2A
ST7789_RASET = 0x2B
ST7789_RAMWR = 0x2C
ST7789_MADCTL = 0x36
ST7789_COLMOD = 0x3A

# Colors (RGB565)
COLOR_BLACK = 0x0000
COLOR_WHITE = 0xFFFF
COLOR_RED = 0xF800
COLOR_GREEN = 0x07E0
COLOR_BLUE = 0x001F
COLOR_YELLOW = 0xFFE0
COLOR_CYAN = 0x07FF
COLOR_MAGENTA = 0xF81F
COLOR_ORANGE = 0xFD20
COLOR_DARK_GREEN = 0x0320
COLOR_DARK_GRAY = 0x2104
COLOR_LIGHT_GRAY = 0x8410


class ST7789Display:
    """
    Minimal ST7789 display driver for ESPectre status display.
    
    Uses direct SPI communication without external dependencies.
    Optimized for minimal memory usage on ESP32.
    """
    
    def __init__(self, rotation=90):
        """
        Initialize display.
        
        Args:
            rotation: Display rotation (0, 90, 180, 270). Default 90 for landscape.
        """
        self.width = DISPLAY_WIDTH
        self.height = DISPLAY_HEIGHT
        self.rotation = rotation
        self.x_offset = 0
        self.y_offset = 0
        
        # Initialize pins
        self.cs = Pin(PIN_CS, Pin.OUT, value=1)
        self.dc = Pin(PIN_DC, Pin.OUT, value=0)
        self.rst = Pin(PIN_RST, Pin.OUT, value=1)
        self.bl = Pin(PIN_BL, Pin.OUT, value=1)  # Backlight ON immediately
        
        print(f"  Display pins: MOSI={PIN_MOSI}, CLK={PIN_SCLK}, CS={PIN_CS}, DC={PIN_DC}, RST={PIN_RST}, BL={PIN_BL}")
        
        # Initialize SPI - use software SPI first for debugging (more reliable)
        # ESP32-S3 hardware SPI can conflict with USB on certain pins
        try:
            # Try hardware SPI bus 1 (HSPI) with lower speed
            self.spi = SPI(1, baudrate=20_000_000, polarity=0, phase=0,
                           sck=Pin(PIN_SCLK), mosi=Pin(PIN_MOSI))
            print("  SPI initialized (hardware bus 1)")
        except Exception as e:
            print(f"  SPI init error: {e}")
            raise
        
        # Small buffer for commands
        self._buf1 = bytearray(1)
        self._buf4 = bytearray(4)
        
        # Initialize display (this also sets rotation and swaps width/height if needed)
        print("  Sending init commands...")
        self._init_display()
        
        # Clear screen to remove artifacts (after rotation, dimensions are correct)
        print(f"  Rotation: {self.rotation}°, Size: {self.width}x{self.height}, Offset: ({self.x_offset},{self.y_offset})")
        print(f"  Clearing screen...")
        self.fill(COLOR_BLACK)
        print("  Display ST7789 initialized")
    
    def _write_cmd(self, cmd):
        """Write command to display"""
        self.cs.value(0)
        self.dc.value(0)
        self._buf1[0] = cmd
        self.spi.write(self._buf1)
        self.cs.value(1)
    
    def _write_data(self, data):
        """Write data to display"""
        self.cs.value(0)
        self.dc.value(1)
        if isinstance(data, int):
            self._buf1[0] = data
            self.spi.write(self._buf1)
        else:
            self.spi.write(data)
        self.cs.value(1)
    
    def _init_display(self):
        """Initialize ST7789 display"""
        # Ensure backlight is on first
        self.bl.value(1)
        time.sleep_ms(100)
        
        # Hardware reset with longer delays
        self.rst.value(1)
        time.sleep_ms(50)
        self.rst.value(0)
        time.sleep_ms(100)
        self.rst.value(1)
        time.sleep_ms(200)
        
        # Software reset
        self._write_cmd(ST7789_SWRESET)
        time.sleep_ms(150)
        
        # Sleep out
        self._write_cmd(ST7789_SLPOUT)
        time.sleep_ms(150)
        
        # Color mode: 16-bit RGB565
        self._write_cmd(ST7789_COLMOD)
        self._write_data(0x55)  # 16-bit
        time.sleep_ms(10)
        
        # Memory access control (rotation)
        # For Waveshare 1.47" display (172x320)
        self._write_cmd(ST7789_MADCTL)
        if self.rotation == 0:
            self._write_data(0x00)  # Portrait
            self.x_offset = 34  # Center 172 in 240
            self.y_offset = 0
        elif self.rotation == 90:
            self._write_data(0x60)  # Landscape (swap X/Y, mirror X)
            self.width, self.height = self.height, self.width
            self.x_offset = 0
            self.y_offset = 34
        elif self.rotation == 180:
            self._write_data(0xC0)  # Portrait inverted
            self.x_offset = 34
            self.y_offset = 0
        elif self.rotation == 270:
            self._write_data(0xA0)  # Landscape inverted
            self.width, self.height = self.height, self.width
            self.x_offset = 0
            self.y_offset = 34
        
        # Inversion on (required for correct colors on most ST7789)
        self._write_cmd(ST7789_INVON)
        time.sleep_ms(10)
        
        # Normal display mode
        self._write_cmd(ST7789_NORON)
        time.sleep_ms(10)
        
        # Display on
        self._write_cmd(ST7789_DISPON)
        time.sleep_ms(100)
    
    def _set_window(self, x0, y0, x1, y1):
        """Set drawing window with offset correction"""
        # Apply display offset
        x0 += self.x_offset
        x1 += self.x_offset
        y0 += self.y_offset
        y1 += self.y_offset
        
        # Column address set
        self._write_cmd(ST7789_CASET)
        self._buf4[0] = (x0 >> 8) & 0xFF
        self._buf4[1] = x0 & 0xFF
        self._buf4[2] = (x1 >> 8) & 0xFF
        self._buf4[3] = x1 & 0xFF
        self._write_data(self._buf4)
        
        # Row address set
        self._write_cmd(ST7789_RASET)
        self._buf4[0] = (y0 >> 8) & 0xFF
        self._buf4[1] = y0 & 0xFF
        self._buf4[2] = (y1 >> 8) & 0xFF
        self._buf4[3] = y1 & 0xFF
        self._write_data(self._buf4)
        
        # Write to RAM
        self._write_cmd(ST7789_RAMWR)
    
    def fill_rect(self, x, y, w, h, color):
        """Fill rectangle with color"""
        self._set_window(x, y, x + w - 1, y + h - 1)
        
        # Prepare color bytes (RGB565, big endian)
        hi = (color >> 8) & 0xFF
        lo = color & 0xFF
        
        # Fill in chunks to avoid large memory allocation
        chunk_size = 512  # 256 pixels per chunk
        chunk = bytearray([hi, lo] * 256)
        
        total_pixels = w * h
        self.cs.value(0)
        self.dc.value(1)
        
        while total_pixels > 0:
            pixels_to_write = min(256, total_pixels)
            self.spi.write(memoryview(chunk)[:pixels_to_write * 2])
            total_pixels -= pixels_to_write
        
        self.cs.value(1)
    
    def fill(self, color):
        """Fill entire screen with color"""
        self.fill_rect(0, 0, self.width, self.height, color)
    
    def pixel(self, x, y, color):
        """Draw single pixel"""
        if 0 <= x < self.width and 0 <= y < self.height:
            self._set_window(x, y, x, y)
            hi = (color >> 8) & 0xFF
            lo = color & 0xFF
            self._write_data(bytearray([hi, lo]))
    
    def hline(self, x, y, w, color):
        """Draw horizontal line"""
        self.fill_rect(x, y, w, 1, color)
    
    def vline(self, x, y, h, color):
        """Draw vertical line"""
        self.fill_rect(x, y, 1, h, color)
    
    def rect(self, x, y, w, h, color):
        """Draw rectangle outline"""
        self.hline(x, y, w, color)
        self.hline(x, y + h - 1, w, color)
        self.vline(x, y, h, color)
        self.vline(x + w - 1, y, h, color)
    
    def backlight(self, on):
        """Control backlight"""
        self.bl.value(1 if on else 0)


class ESPectreDisplay:
    """
    ESPectre status display manager.
    
    Provides high-level interface for displaying ESPectre status
    on the ST7789 display with minimal memory usage.
    """
    
    def __init__(self, rotation=90):
        """
        Initialize ESPectre display.
        
        Args:
            rotation: Display rotation (0, 90, 180, 270). Default 90 for landscape.
        """
        self.display = ST7789Display(rotation=rotation)
        self.last_state = None
        self.last_progress = -1
        
        # Draw initial screen
        self._draw_initial_screen()
    
    def _draw_initial_screen(self):
        """Draw initial ESPectre screen layout"""
        d = self.display
        
        # Black background
        d.fill(COLOR_BLACK)
        
        # Title bar
        d.fill_rect(0, 0, d.width, 40, COLOR_DARK_GRAY)
        
        # Status area (large circle indicator will go here)
        # Progress bar area
        # Stats area
        
        # Draw static labels (we'll update values only)
        self._draw_static_labels()
    
    def _draw_static_labels(self):
        """Draw static text labels (using simple rectangles as placeholders)"""
        # We'll use colored rectangles to indicate areas
        # since we don't have font rendering without additional libraries
        pass
    
    def update(self, state, moving_variance, threshold, pps=0, ip_addr=None, 
               calibrating=False, calibration_progress=0):
        """
        Update display with current ESPectre status.
        
        Args:
            state: Current state (0=IDLE, 1=MOTION)
            moving_variance: Current moving variance value
            threshold: Detection threshold
            pps: Packets per second
            ip_addr: IP address string (optional)
            calibrating: True if calibration in progress
            calibration_progress: Calibration progress 0-100
        """
        d = self.display
        
        # Calculate progress percentage
        progress = int((moving_variance / threshold) * 100) if threshold > 0 else 0
        progress = min(progress, 200)  # Cap at 200%
        
        # Only update if state or progress changed significantly
        state_changed = state != self.last_state
        progress_changed = abs(progress - self.last_progress) >= 2
        
        if not state_changed and not progress_changed and not calibrating:
            return
        
        self.last_state = state
        self.last_progress = progress
        
        # Update status indicator (large colored area)
        if calibrating:
            # Yellow for calibration
            status_color = COLOR_YELLOW
            self._draw_status_indicator(status_color, "CAL")
            self._draw_calibration_progress(calibration_progress)
        else:
            # Green for IDLE, Red for MOTION
            status_color = COLOR_RED if state == 1 else COLOR_GREEN
            self._draw_status_indicator(status_color, "MOTION" if state == 1 else "IDLE")
            
            # Update progress bar
            self._draw_progress_bar(progress, threshold)
        
        # Update stats
        self._draw_stats(moving_variance, threshold, pps)
    
    def _draw_status_indicator(self, color, label):
        """Draw large status indicator"""
        d = self.display
        
        # Large status circle/rectangle in center-top area
        center_x = d.width // 2
        indicator_y = 60
        indicator_size = 80
        
        # Draw filled rectangle as status indicator
        x = center_x - indicator_size // 2
        d.fill_rect(x, indicator_y, indicator_size, indicator_size, color)
        
        # Draw border
        d.rect(x - 2, indicator_y - 2, indicator_size + 4, indicator_size + 4, COLOR_WHITE)
    
    def _draw_progress_bar(self, progress, threshold):
        """Draw motion level progress bar"""
        d = self.display
        
        bar_x = 20
        bar_y = 160
        bar_width = d.width - 40
        bar_height = 30
        
        # Background
        d.fill_rect(bar_x, bar_y, bar_width, bar_height, COLOR_DARK_GRAY)
        
        # Progress fill
        fill_width = min(int((progress / 100) * bar_width), bar_width)
        
        # Color based on level (green -> yellow -> red)
        if progress < 50:
            fill_color = COLOR_GREEN
        elif progress < 100:
            fill_color = COLOR_YELLOW
        else:
            fill_color = COLOR_RED
        
        if fill_width > 0:
            d.fill_rect(bar_x, bar_y, fill_width, bar_height, fill_color)
        
        # Threshold marker
        threshold_x = bar_x + int(bar_width * 0.75)  # 75% = threshold
        d.vline(threshold_x, bar_y - 5, bar_height + 10, COLOR_WHITE)
        
        # Border
        d.rect(bar_x - 1, bar_y - 1, bar_width + 2, bar_height + 2, COLOR_WHITE)
    
    def _draw_calibration_progress(self, progress):
        """Draw calibration progress bar"""
        d = self.display
        
        bar_x = 20
        bar_y = 160
        bar_width = d.width - 40
        bar_height = 30
        
        # Background
        d.fill_rect(bar_x, bar_y, bar_width, bar_height, COLOR_DARK_GRAY)
        
        # Progress fill
        fill_width = int((progress / 100) * bar_width)
        if fill_width > 0:
            d.fill_rect(bar_x, bar_y, fill_width, bar_height, COLOR_CYAN)
        
        # Border
        d.rect(bar_x - 1, bar_y - 1, bar_width + 2, bar_height + 2, COLOR_WHITE)
    
    def _draw_stats(self, moving_variance, threshold, pps):
        """Draw stats area"""
        d = self.display
        
        stats_y = 210
        line_height = 25
        
        # Clear stats area
        d.fill_rect(0, stats_y, d.width, 100, COLOR_BLACK)
        
        # Draw colored indicators for each stat
        # Movement level indicator
        mv_percent = int((moving_variance / threshold) * 100) if threshold > 0 else 0
        self._draw_stat_bar(10, stats_y, f"MVT", mv_percent, 100)
        
        # PPS indicator (100 = 100%)
        self._draw_stat_bar(10, stats_y + line_height, f"PPS", pps, 150)
        
        # Threshold indicator
        self._draw_stat_bar(10, stats_y + line_height * 2, f"THR", int(threshold * 100), 200)
    
    def _draw_stat_bar(self, x, y, label, value, max_val):
        """Draw a single stat with mini bar"""
        d = self.display
        
        bar_width = d.width - 40
        bar_height = 16
        
        # Mini progress bar
        fill_width = min(int((value / max_val) * bar_width), bar_width)
        
        # Background
        d.fill_rect(x + 25, y, bar_width, bar_height, COLOR_DARK_GRAY)
        
        # Fill
        if fill_width > 0:
            d.fill_rect(x + 25, y, fill_width, bar_height, COLOR_CYAN)
        
        # Border
        d.rect(x + 24, y - 1, bar_width + 2, bar_height + 2, COLOR_LIGHT_GRAY)
    
    def show_boot_screen(self, message="Starting..."):
        """Show boot/startup screen"""
        d = self.display
        
        # Black background
        d.fill(COLOR_BLACK)
        
        # ESPectre logo area (simple colored rectangle)
        logo_y = 80
        d.fill_rect(20, logo_y, d.width - 40, 60, COLOR_MAGENTA)
        d.rect(18, logo_y - 2, d.width - 36, 64, COLOR_WHITE)
        
        # Status message area
        d.fill_rect(20, 200, d.width - 40, 30, COLOR_DARK_GRAY)
    
    def show_calibrating(self, phase="", progress=0):
        """Show calibration screen"""
        d = self.display
        
        # Update status to yellow
        self._draw_status_indicator(COLOR_YELLOW, "CAL")
        
        # Show progress
        self._draw_calibration_progress(progress)
    
    def show_error(self, message="Error"):
        """Show error screen"""
        d = self.display
        
        # Red background for status
        self._draw_status_indicator(COLOR_RED, "ERR")
    
    def off(self):
        """Turn off display"""
        self.display.backlight(False)
    
    def on(self):
        """Turn on display"""
        self.display.backlight(True)


# Singleton instance for easy access
_display_instance = None


def get_display(rotation=0):
    """
    Get or create display instance.
    
    Args:
        rotation: Display rotation (only used on first call)
        
    Returns:
        ESPectreDisplay instance
    """
    global _display_instance
    if _display_instance is None:
        _display_instance = ESPectreDisplay(rotation=rotation)
    return _display_instance


def is_display_available():
    """
    Check if display hardware is available.
    
    Returns:
        bool: True if display pins are accessible
    """
    try:
        # Try to access the backlight pin
        bl = Pin(PIN_BL, Pin.IN)
        return True
    except Exception:
        return False

