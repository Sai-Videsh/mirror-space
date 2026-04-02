"""
Mirror-Space Screen Broadcaster (Python)
Captures screen and broadcasts via UDP using diff-frame encoding
"""

import sys
import time
import socket
import struct
import secrets
from typing import Tuple, List, Optional

import numpy as np
import cv2
from zeroconf import ServiceInfo, Zeroconf

# Windows-specific libraries for HWND-based window capture
try:
    import win32gui
    import win32ui
    import win32con
    HAS_WIN32 = True
except ImportError:
    HAS_WIN32 = False

from diff_encoder import DiffFrameEncoder
from region_selector import get_region_config


DEFAULT_PORT = 9999
DEFAULT_BROADCAST_TARGET = "255.255.255.255"
SERVICE_TYPE = "_mirror-space._udp.local."
DISCOVERY_BEACON_PORT = 10001
DISCOVERY_BEACON_INTERVAL = 1.0
MAX_PACKET_SIZE = 65507  # Max UDP packet size
FRAGMENT_HEADER_SIZE = 12  # total_packets (4) + packet_index (4) + frame_number (4)
MAX_UDP_PAYLOAD_SIZE = 1400  # MTU-safe payload to avoid IP fragmentation on LAN/WiFi
TARGET_FPS = 15
FRAME_INTERVAL = 1.0 / TARGET_FPS
SHOW_HEATMAP = True  # Set to False to disable heatmap overlay
MAX_CHANGED_BLOCK_RATIO = 0.35  # Fallback to key frame when too many blocks change
MAX_DIFF_PAYLOAD_RATIO = 0.25  # Fallback to key frame when diff payload gets too large
JPEG_QUALITY = 60  # Lower quality reduces packet count and loss on busy scenes
MAX_STREAM_WIDTH = 1280  # Resize captured frame if screen width is larger than this
ENABLE_MOTION_DETECTION = True  # Enable optical flow-based motion encoding


def get_primary_ipv4() -> str:
    """Resolve the primary outbound IPv4 address for LAN service registration."""
    probe = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        probe.connect(("8.8.8.8", 80))
        return probe.getsockname()[0]
    except Exception:
        return socket.gethostbyname(socket.gethostname())
    finally:
        probe.close()


class StreamAdvertiser:
    """Publishes this broadcaster on LAN using mDNS/Zeroconf."""

    def __init__(self, stream_port: int, feedback_port: int):
        self.stream_port = stream_port
        self.feedback_port = feedback_port
        self.zeroconf = Zeroconf()
        self.info: ServiceInfo | None = None

    def start(self):
        host_name = socket.gethostname()
        instance = f"{host_name}-{self.stream_port}"
        service_name = f"{instance}.{SERVICE_TYPE}"
        ip_addr = get_primary_ipv4()

        properties = {
            b"stream_name": host_name.encode("utf-8"),
            b"stream_port": str(self.stream_port).encode("utf-8"),
            b"feedback_port": str(self.feedback_port).encode("utf-8"),
            b"version": b"1",
        }

        self.info = ServiceInfo(
            type_=SERVICE_TYPE,
            name=service_name,
            addresses=[socket.inet_aton(ip_addr)],
            port=self.stream_port,
            properties=properties,
            server=f"{host_name}.local.",
        )

        self.zeroconf.register_service(self.info, allow_name_change=True)
        print(f"mDNS stream advertisement started: {host_name} ({ip_addr}:{self.stream_port})")

    def close(self):
        if self.info is not None:
            try:
                self.zeroconf.unregister_service(self.info)
            except Exception:
                pass
        self.zeroconf.close()


class UdpDiscoveryBeacon:
    """Broadcasts stream availability periodically on LAN."""

    def __init__(self, stream_name: str, stream_port: int, feedback_port: int):
        self.stream_name = stream_name
        self.stream_port = stream_port
        self.feedback_port = feedback_port
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        self.last_sent = 0.0

    def tick(self):
        now = time.time()
        if now - self.last_sent < DISCOVERY_BEACON_INTERVAL:
            return

        message = (
            "STREAM_ANNOUNCE "
            f"stream_name={self.stream_name} "
            f"stream_port={self.stream_port} "
            f"feedback_port={self.feedback_port}"
        )
        try:
            self.sock.sendto(message.encode("utf-8"), ("255.255.255.255", DISCOVERY_BEACON_PORT))
            self.last_sent = now
        except Exception:
            # Discovery is best-effort; stream transport should continue even if beacon fails.
            pass

    def close(self):
        self.sock.close()


class UDPBroadcaster:
    """Handles UDP packet transmission with fragmentation"""
    
    def __init__(self, target_ip: str, port: int):
        self.target_ip = target_ip
        self.port = port
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        
        # Increase send buffer size
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 1024*1024)
        
        print(f"UDP broadcaster initialized to {target_ip}:{port}")
    
    def send_data(self, data: bytes, frame_number: int) -> bool:
        """Send data with automatic fragmentation"""
        offset = 0
        packet_index = 0
        max_chunk_size = min(MAX_UDP_PAYLOAD_SIZE, MAX_PACKET_SIZE) - FRAGMENT_HEADER_SIZE
        if max_chunk_size <= 0:
            print("Invalid packet sizing configuration")
            return False

        total_packets = (len(data) + max_chunk_size - 1) // max_chunk_size
        
        while offset < len(data):
            chunk_size = min(max_chunk_size, len(data) - offset)
            
            # Create packet with metadata
            # Format: I=total_packets, I=packet_index, I=frame_number, then data
            packet = struct.pack('<III', total_packets, packet_index, frame_number)
            packet += data[offset:offset + chunk_size]
            
            try:
                self.sock.sendto(packet, (self.target_ip, self.port))
            except Exception as e:
                print(f"Send failed: {e}")
                return False
            
            offset += chunk_size
            packet_index += 1
            
            # Small delay between packets to avoid overwhelming receiver
            if packet_index < total_packets:
                time.sleep(0.0001)  # 100 microseconds
        
        return True
    
    def close(self):
        """Close the socket"""
        self.sock.close()


class FeedbackReceiver:
    """Receives receiver-side health feedback to trigger key frames"""

    def __init__(self, port: int):
        self.port = port
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind(('0.0.0.0', port))
        self.sock.setblocking(False)
        print(f"Feedback channel listening on port {port}")

    def poll_messages(self, max_messages: int = 8) -> List[Tuple[str, Tuple[str, int]]]:
        """Read available feedback messages without blocking"""
        messages: List[Tuple[str, Tuple[str, int]]] = []

        for _ in range(max_messages):
            try:
                data, addr = self.sock.recvfrom(4096)
            except BlockingIOError:
                break
            except Exception as e:
                print(f"Feedback receive failed: {e}")
                break

            message = data.decode('utf-8', errors='ignore').strip()
            if message:
                print(f"Feedback from {addr[0]}:{addr[1]} -> {message}")
                messages.append((message, addr))

        return messages

    def close(self):
        """Close feedback socket"""
        self.sock.close()

    def send_message(self, target_ip: str, target_port: int, message: str):
        """Send a short UDP control message."""
        try:
            self.sock.sendto(message.encode('utf-8'), (target_ip, target_port))
        except Exception as e:
            print(f"Feedback send failed: {e}")


class HWNDWindowCapture:
    """Captures screen from a specific window using its HWND (never changes capture source)"""
    
    def __init__(self, hwnd: Optional[int] = None):
        """
        Initialize window capture
        
        Args:
            hwnd: Window handle. If None, falls back to full screen via region mode
        """
        self.hwnd = hwnd
        self.presentation_mode = False
        self.last_error = None
        
        if self.hwnd is None:
            print("WARNING: No HWND provided. Will use region-based capture.")
            print("This may switch content if tabs/windows change.")
        else:
            print(f"HWNDWindowCapture initialized with HWND: {self.hwnd}")
            print("Capturing ONLY this specific window - tab switches won't affect stream")
    
    def set_presentation_mode(self, enabled: bool):
        """Enable/disable presentation mode (black background)"""
        self.presentation_mode = enabled
        if enabled:
            print("\n*** PRESENTATION MODE ENABLED ***")
            print("    Capturing window on BLACK background")
            print("    Other windows/tabs won't affect broadcast\n")
    
    def _is_window_valid(self) -> bool:
        """Check if window still exists and is not minimized"""
        if self.hwnd is None:
            return False
        
        try:
            # Check if window exists
            if not win32gui.IsWindow(self.hwnd):
                self.last_error = "Window no longer exists"
                return False
            
            # Check if window is minimized
            if win32gui.IsIconic(self.hwnd):
                self.last_error = "Window is minimized"
                return False
            
            return True
        except Exception as e:
            self.last_error = str(e)
            return False
    
    def get_window_dimensions(self) -> Optional[Tuple[int, int, int, int]]:
        """Get current window dimensions (x, y, width, height)"""
        if self.hwnd is None:
            return None
        
        try:
            rect = win32gui.GetWindowRect(self.hwnd)
            x, y, x2, y2 = rect
            width = x2 - x
            height = y2 - y
            return (x, y, width, height)
        except Exception as e:
            print(f"Error getting window dimensions: {e}")
            return None
    
    def capture_frame(self) -> np.ndarray:
        """Capture frame from the selected window using HWND"""
        
        if not HAS_WIN32:
            print("ERROR: win32gui/win32ui not available. Install with: pip install pywin32")
            return np.zeros((480, 640, 3), dtype=np.uint8)
        
        if self.hwnd is None:
            print("ERROR: No HWND set")
            return np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Validate window still exists
        if not self._is_window_valid():
            print(f"ERROR: Cannot capture - {self.last_error}")
            return np.zeros((480, 640, 3), dtype=np.uint8)
        
        try:
            # Get window dimensions
            dims = self.get_window_dimensions()
            if dims is None:
                return np.zeros((480, 640, 3), dtype=np.uint8)
            
            x, y, width, height = dims
            
            # Validate dimensions
            if width <= 0 or height <= 0:
                print(f"WARNING: Invalid window dimensions: {width}x{height}")
                return np.zeros((100, 100, 3), dtype=np.uint8)
            
            # Capture the window
            frame = self._capture_window_direct(x, y, width, height)
            
            if self.presentation_mode:
                # Add presentation mode: black background
                frame = self._add_presentation_background(frame, x, y, width, height)
            
            return frame
            
        except Exception as e:
            print(f"Capture error: {e}")
            return np.zeros((480, 640, 3), dtype=np.uint8)
    
    def _capture_window_direct(self, x: int, y: int, width: int, height: int) -> np.ndarray:
        """Capture window content using pure ctypes - no win32ui dependency"""
        import ctypes
        from ctypes import windll, c_void_p, c_int, c_uint32, byref, create_string_buffer
        try:
            # Get client area dimensions (matches what PrintWindow captures)
            client_rect = win32gui.GetClientRect(self.hwnd)
            client_width = client_rect[2] - client_rect[0]
            client_height = client_rect[3] - client_rect[1]
            if client_width <= 0 or client_height <= 0:
                client_width, client_height = max(1, width), max(1, height)
            gdi32 = windll.gdi32
            user32 = windll.user32
            hwnd_dc = user32.GetDC(self.hwnd)
            if not hwnd_dc:
                raise RuntimeError("GetDC failed")
            mem_dc = gdi32.CreateCompatibleDC(hwnd_dc)
            bitmap = gdi32.CreateCompatibleBitmap(hwnd_dc, client_width, client_height)
            gdi32.SelectObject(mem_dc, bitmap)
            PW_RENDERFULLCONTENT = 0x2
            PW_CLIENTONLY = 0x1

            result = user32.PrintWindow(self.hwnd, mem_dc, PW_RENDERFULLCONTENT)
            if result == 0:
                result = user32.PrintWindow(self.hwnd, mem_dc, PW_CLIENTONLY)
            if result == 0:
                screen_dc = user32.GetDC(0)
                gdi32.BitBlt(mem_dc, 0, 0, client_width, client_height, screen_dc, x, y, 0x00CC0020)
                user32.ReleaseDC(0, screen_dc)
            class BITMAPINFOHEADER(ctypes.Structure):
                _fields_ = [
                    ("biSize", c_uint32), ("biWidth", c_int), ("biHeight", c_int),
                    ("biPlanes", ctypes.c_uint16), ("biBitCount", ctypes.c_uint16),
                    ("biCompression", c_uint32), ("biSizeImage", c_uint32),
                    ("biXPelsPerMeter", c_int), ("biYPelsPerMeter", c_int),
                    ("biClrUsed", c_uint32), ("biClrImportant", c_uint32),
                ]
            bmi = BITMAPINFOHEADER()
            bmi.biSize = ctypes.sizeof(BITMAPINFOHEADER)
            bmi.biWidth = client_width
            bmi.biHeight = -client_height
            bmi.biPlanes = 1
            bmi.biBitCount = 32
            bmi.biCompression = 0
            buf_size = client_width * client_height * 4
            buf = create_string_buffer(buf_size)
            lines = gdi32.GetDIBits(mem_dc, bitmap, 0, client_height, buf, byref(bmi), 0)
            gdi32.DeleteObject(bitmap)
            gdi32.DeleteDC(mem_dc)
            user32.ReleaseDC(self.hwnd, hwnd_dc)
            if lines == 0:
                raise RuntimeError("GetDIBits returned 0 lines")
            frame = np.frombuffer(buf.raw, dtype=np.uint8).reshape((client_height, client_width, 4))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
            return frame
        except Exception as e:
            print(f"Window capture failed: {e}")
            return np.zeros((max(1, height), max(1, width), 3), dtype=np.uint8)
    
    def _add_presentation_background(self, frame: np.ndarray, x: int, y: int, width: int, height: int) -> np.ndarray:
        """Add presentation mode: place window on black background"""
        try:
            # Get primary screen dimensions
            screen_width = win32gui.GetSystemMetrics(0)
            screen_height = win32gui.GetSystemMetrics(1)
            
            # Create black background
            bg_frame = np.zeros((screen_height, screen_width, 3), dtype=np.uint8)
            
            # Place window content at correct position
            y_end = min(y + height, screen_height)
            x_end = min(x + width, screen_width)
            
            bg_frame[y:y_end, x:x_end] = frame[:y_end-y, :x_end-x]
            
            return bg_frame
            
        except Exception as e:
            print(f"Presentation background error: {e}")
            return frame
    
    def get_dimensions(self) -> Tuple[int, int]:
        """Get capture dimensions"""
        dims = self.get_window_dimensions()
        if dims:
            return (dims[2], dims[3])  # width, height
        return (640, 480)  # default fallback
    
    def get_full_dimensions(self) -> Tuple[int, int]:
        """Get full screen dimensions (for presentation mode)"""
        try:
            width = win32gui.GetSystemMetrics(0)
            height = win32gui.GetSystemMetrics(1)
            return (width, height)
        except:
            return (1920, 1080)  # default fallback
    
    def close(self):
        """Clean up resources"""
        pass  # win32 resources are handled automatically


class ScreenCapture:
    """Captures screen using mss library with region support"""
    
    def __init__(self):
        # Dynamic capture: will use either HWND or region-based
        self.use_hwnd = False
        self.hwnd_capture: Optional[HWNDWindowCapture] = None
        
        # Fallback: region-based capture
        try:
            import mss
            self.sct = mss.mss()
            self.monitor = self.sct.monitors[1]  # Primary monitor
            self.full_width = self.monitor['width']
            self.full_height = self.monitor['height']
        except:
            self.sct = None
            self.full_width = 1920
            self.full_height = 1080
        
        # Region configuration (default to full screen)
        self.region_x = 0
        self.region_y = 0
        self.region_width = self.full_width
        self.region_height = self.full_height
        
        # Presentation mode flag
        self.presentation_mode = False
        
        print(f"Screen capture initialized: {self.full_width}x{self.full_height}")
    
    def set_region(self, x: int, y: int, width: int, height: int, hwnd: Optional[int] = None, presentation_mode: bool = False):
        """Set the capture region or HWND"""
        self.presentation_mode = presentation_mode

        # If HWND is provided and win32 is available, use HWND-based capture
        if hwnd is not None and HAS_WIN32:
            print(f"\n*** USING HWND-BASED CAPTURE (Window Handle: {hwnd}) ***")
            print(f"    This window ONLY will be captured")
            print(f"    Tab switches and other windows WILL NOT affect the broadcast")
            print(f"    If window resizes, capture will adjust automatically\n")

            # Ensure we always set HWND mode when valid
            self.use_hwnd = True
            self.hwnd_capture = HWNDWindowCapture(hwnd)
            self.hwnd_capture.set_presentation_mode(presentation_mode)

            # Get initial dimensions
            dims = self.hwnd_capture.get_window_dimensions()
            if dims:
                self.region_width = dims[2]
                self.region_height = dims[3]

            return
        
        # Fall back to region-based capture
        print(f"\n*** USING REGION-BASED CAPTURE ***")
        print(f"    Window coordinates: x={x}, y={y}, width={width}, height={height}")
        print(f"    Note: Tab switches may affect capture\n")
        
        self.use_hwnd = False
        self.hwnd_capture = None
        
        # Window bounds
        window_left = x
        window_top = y
        window_right = x + width
        window_bottom = y + height
        
        # Screen bounds
        screen_left = 0
        screen_top = 0
        screen_right = self.full_width
        screen_bottom = self.full_height
        
        # Calculate intersection (visible portion)
        visible_left = max(window_left, screen_left)
        visible_top = max(window_top, screen_top)
        visible_right = min(window_right, screen_right)
        visible_bottom = min(window_bottom, screen_bottom)
        
        # Convert to coordinates
        self.region_x = visible_left
        self.region_y = visible_top
        self.region_width = max(100, visible_right - visible_left)
        self.region_height = max(100, visible_bottom - visible_top)
        
        # Validate
        if self.region_x < 0 or self.region_y < 0:
            print(f"WARNING: Invalid capture coordinates - using full screen")
            self.region_x = 0
            self.region_y = 0
            self.region_width = self.full_width
            self.region_height = self.full_height
    
    def capture_frame(self) -> np.ndarray:
        """Capture current screen frame"""
        
        # Use HWND-based capture if available
        if self.use_hwnd and self.hwnd_capture:
            return self.hwnd_capture.capture_frame()
        
        # Fall back to region-based capture using mss
        if self.sct is None:
            return np.zeros((self.region_height, self.region_width, 3), dtype=np.uint8)
        
        try:
            region_monitor = {
                'left': self.region_x,
                'top': self.region_y,
                'width': self.region_width,
                'height': self.region_height
            }
            
            screenshot = self.sct.grab(region_monitor)
            frame = np.array(screenshot)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
            
            if self.presentation_mode:
                # Add presentation background
                bg_frame = np.zeros((self.full_height, self.full_width, 3), dtype=np.uint8)
                y_end = min(self.region_y + self.region_height, self.full_height)
                x_end = min(self.region_x + self.region_width, self.full_width)
                bg_frame[self.region_y:y_end, self.region_x:x_end] = frame[:y_end - self.region_y, :x_end - self.region_x]
                return bg_frame
            
            return frame
        except Exception as e:
            print(f"Capture error: {e}")
            return np.zeros((self.region_height, self.region_width, 3), dtype=np.uint8)
    
    def get_dimensions(self) -> Tuple[int, int]:
        """Get capture dimensions (region dimensions, not full screen)"""
        if self.use_hwnd and self.hwnd_capture:
            return self.hwnd_capture.get_dimensions()
        return self.region_width, self.region_height
    
    def get_full_dimensions(self) -> Tuple[int, int]:
        """Get full screen dimensions"""
        if self.use_hwnd and self.hwnd_capture:
            return self.hwnd_capture.get_full_dimensions()
        return self.full_width, self.full_height
    
    def close(self):
        """Clean up resources"""
        if self.hwnd_capture:
            self.hwnd_capture.close()
        if self.sct:
            try:
                self.sct.close()
            except:
                pass


def create_heatmap_overlay(frame: np.ndarray, changed_blocks, motion_blocks, block_size: int) -> np.ndarray:
    """Create a heatmap overlay showing changed blocks and motion regions"""
    overlay = frame.copy()
    heatmap = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)
    
    # Mark changed blocks
    for x, y, w, h in changed_blocks:
        heatmap[y:y+h, x:x+w] = 128
    
    # Mark motion blocks (higher intensity)
    for x, y, w, h in motion_blocks:
        heatmap[y:y+h, x:x+w] = 255
    
    # Apply color map (blue-green-red for intensity)
    heatmap_colored = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    
    # Blend with original frame
    overlay = cv2.addWeighted(frame, 0.7, heatmap_colored, 0.3, 0)
    
    # Draw grid for all blocks
    height, width = frame.shape[:2]
    for y in range(0, height, block_size):
        cv2.line(overlay, (0, y), (width, y), (100, 100, 100), 1)
    for x in range(0, width, block_size):
        cv2.line(overlay, (x, 0), (x, height), (100, 100, 100), 1)
    
    # Draw rectangles around changed blocks
    for x, y, w, h in changed_blocks:
        cv2.rectangle(overlay, (x, y), (x+w, y+h), (0, 255, 0), 2)
    
    # Draw rectangles around motion blocks in blue
    for x, y, w, h in motion_blocks:
        cv2.rectangle(overlay, (x, y), (x+w, y+h), (255, 0, 0), 2)
    
    # Add statistics
    text = f"Changed: {len(changed_blocks)}, Motion: {len(motion_blocks)}"
    cv2.putText(overlay, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                1, (0, 255, 0), 2, cv2.LINE_AA)
    
    return overlay


def main():
    target_ip = DEFAULT_BROADCAST_TARGET  # Default: LAN broadcast (no manual receiver IP)
    port = DEFAULT_PORT
    
    if len(sys.argv) > 1:
        target_ip = sys.argv[1]
    if len(sys.argv) > 2:
        port = int(sys.argv[2])
    
    print("=== Mirror-Space Screen Broadcaster (Python) ===")
    print(f"Target: {target_ip}:{port}")
    print(f"Target FPS: {TARGET_FPS}")
    print(f"Max Changed Block Ratio: {MAX_CHANGED_BLOCK_RATIO:.0%}")
    print(f"Max Diff Payload Ratio: {MAX_DIFF_PAYLOAD_RATIO:.0%}")
    print(f"JPEG Quality: {JPEG_QUALITY}")
    print(f"Max Stream Width: {MAX_STREAM_WIDTH}")
    print(f"Motion Detection: {'Enabled' if ENABLE_MOTION_DETECTION else 'Disabled'}")
    print(f"Heatmap Overlay: {'Enabled' if SHOW_HEATMAP else 'Disabled'}")
    print("Press Ctrl+C to stop...")
    if SHOW_HEATMAP:
        print("Press 'h' to toggle heatmap, 'q' to quit\n")
    else:
        print()
    
    # Initialize components
    capture = None
    broadcaster = None
    feedback_receiver = None
    advertiser = None
    beacon = None
    try:
        
        capture = ScreenCapture()
        
        # Region selection
        full_width, full_height = capture.get_full_dimensions()
        region_config = get_region_config(full_width, full_height)
        
        if region_config is None:
            print("Region selection cancelled. Exiting.")
            return
        
        # Apply selected region to capture
        capture.set_region(region_config.x, region_config.y, region_config.width, region_config.height, region_config.hwnd, region_config.presentation_mode)
        
        encoder = DiffFrameEncoder(
            block_size=16,
            threshold=4,
            max_changed_block_ratio=MAX_CHANGED_BLOCK_RATIO,
            max_diff_payload_ratio=MAX_DIFF_PAYLOAD_RATIO,
            jpeg_quality=JPEG_QUALITY,
            enable_motion_detection=ENABLE_MOTION_DETECTION,
        )
        broadcaster = UDPBroadcaster(target_ip, port)
        feedback_receiver = FeedbackReceiver(port + 1)
        advertiser = StreamAdvertiser(stream_port=port, feedback_port=port + 1)
        advertiser.start()
        stream_name = socket.gethostname()
        access_id = secrets.token_hex(3).upper()
        local_host_name = socket.gethostname().lower()
        local_primary_ip = get_primary_ipv4()
        beacon = UdpDiscoveryBeacon(stream_name=stream_name, stream_port=port, feedback_port=port + 1)

        auto_connect_mode = target_ip == DEFAULT_BROADCAST_TARGET
        active_receiver_ip: Optional[str] = None
        last_wait_log_time = 0.0
        if auto_connect_mode:
            print("\n" + "="*60)
            print("Broadcaster is ready. Waiting for receiver connection...")
            print(f"\n>>> GIVE THIS ACCESS ID TO RECEIVER <<<")
            print(f">>> SESSION ACCESS ID: {access_id} <<<")
            print(f"\nStreaming will start only after RECEIVER_HELLO is received.")
            print("="*60 + "\n")
        
        # Create heatmap window if enabled
        heatmap_enabled = SHOW_HEATMAP
        if SHOW_HEATMAP:
            cv2.namedWindow("Broadcaster Heatmap", cv2.WINDOW_NORMAL)
        
        # Broadcasting loop
        frame_number = 0
        if SHOW_HEATMAP:
            cv2.destroyAllWindows()
        fps_counter = 0
        fps_start_time = time.time()
        
        print("Starting broadcast...\n")
        
        while True:
            frame_start = time.time()
            beacon.tick()

            # Process receiver feedback before encoding the next frame.
            for message, addr in feedback_receiver.poll_messages():
                sender_ip = addr[0]

                if message.startswith("DISCOVERY_QUERY"):
                    feedback_receiver.send_message(
                        sender_ip,
                        addr[1],
                        (
                            "DISCOVERY_RESPONSE "
                            f"stream_name={stream_name} "
                            f"stream_port={port} "
                            f"feedback_port={port + 1}"
                        ),
                    )
                    continue

                if message.startswith("RECEIVER_HELLO") and auto_connect_mode:
                    receiver_access_id = ""
                    receiver_name = ""
                    for token in message.split():
                        if token.startswith("receiver="):
                            receiver_name = token.split("=", 1)[1].strip().lower()
                        elif token.startswith("access_id="):
                            receiver_access_id = token.split("=", 1)[1].strip().upper()

                    if receiver_access_id != access_id:
                        print(
                            "Connection Debug: rejected receiver hello due to invalid access ID "
                            f"from {sender_ip}"
                        )
                        continue

                    same_host = (
                        sender_ip == "127.0.0.1"
                        or sender_ip == local_primary_ip
                        or receiver_name == local_host_name
                    )

                    if active_receiver_ip != sender_ip:
                        active_receiver_ip = sender_ip
                        broadcaster.target_ip = "127.0.0.1" if same_host else active_receiver_ip
                        print(
                            f"Receiver connected: {active_receiver_ip}. "
                            "Starting stream transmission."
                        )
                        if same_host:
                            print("Connection Debug: same-host receiver detected, using loopback target 127.0.0.1")
                    continue

                # Ignore health events from non-selected receivers in connect-gated mode.
                if auto_connect_mode and active_receiver_ip is not None and sender_ip != active_receiver_ip:
                    continue

                

                if message.startswith("KEYFRAME_REQUEST"):
                    encoder.force_key_frame(reason=f"receiver_mismatch {message}")
                elif message.startswith("NETWORK_UNSTABLE"):
                    encoder.force_key_frame(reason=f"network_instability {message}")

            if auto_connect_mode and active_receiver_ip is None:
                now = time.time()
                if now - last_wait_log_time >= 2.0:
                    print("Connection Debug: waiting for RECEIVER_HELLO on feedback port")
                    last_wait_log_time = now
                time.sleep(0.05)
                continue
            
            # Capture screen
            frame = capture.capture_frame()

            # Limit stream resolution for high-motion scenes to reduce packet loss.
            if frame.shape[1] > MAX_STREAM_WIDTH:
                scale = MAX_STREAM_WIDTH / frame.shape[1]
                resized_height = max(1, int(frame.shape[0] * scale))
                frame = cv2.resize(frame, (MAX_STREAM_WIDTH, resized_height), interpolation=cv2.INTER_AREA)
            
            # Encode frame
            encoded_data = encoder.encode(frame, frame_number)
            
            # Show heatmap if enabled
            if heatmap_enabled:
                changed_blocks = encoder.get_changed_block_positions()
                motion_blocks = encoder.motion_blocks
                heatmap = create_heatmap_overlay(frame, changed_blocks, motion_blocks, encoder.block_size)
                
                # Resize for display if too large
                display_height = 720
                if heatmap.shape[0] > display_height:
                    scale = display_height / heatmap.shape[0]
                    display_width = int(heatmap.shape[1] * scale)
                    heatmap = cv2.resize(heatmap, (display_width, display_height))
                
                cv2.imshow("Broadcaster Heatmap", heatmap)
                
                # Check for key press
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('h'):
                    heatmap_enabled = not heatmap_enabled
                    if not heatmap_enabled:
                        cv2.destroyWindow("Broadcaster Heatmap")
                    else:
                        cv2.namedWindow("Broadcaster Heatmap", cv2.WINDOW_NORMAL)
            
            # Send via UDP
            if not broadcaster.send_data(encoded_data, frame_number=frame_number):
                print("Failed to send frame")
            
            frame_number += 1
            fps_counter += 1
            
            # Calculate actual FPS every second
            elapsed = time.time() - fps_start_time
            if elapsed >= 1.0:
                actual_fps = fps_counter / elapsed
                print(f"Actual FPS: {actual_fps:.1f}")
                fps_counter = 0
                fps_start_time = time.time()
            
            # Frame rate limiting
            frame_time = time.time() - frame_start
            sleep_time = FRAME_INTERVAL - frame_time
            if sleep_time > 0:
                time.sleep(sleep_time)
    
    except KeyboardInterrupt:
        print("\nStopping broadcaster...")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if capture is not None:
            capture.close()
        if broadcaster is not None:
            broadcaster.close()
        if feedback_receiver is not None:
            feedback_receiver.close()
        if advertiser is not None:
            advertiser.close()
        if beacon is not None:
            beacon.close()
        print("Broadcaster stopped.")


if __name__ == "__main__":
    main()
