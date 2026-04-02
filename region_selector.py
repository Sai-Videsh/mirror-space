"""
Region selector for Mirror-Space
Allows users to select full screen, specific window, or custom region
"""

import sys
import cv2
import numpy as np
from typing import Tuple, Optional, List, Dict
import threading
import time

try:
    import pygetwindow as gw
    HAS_PYGETWINDOW = True
except ImportError:
    HAS_PYGETWINDOW = False

try:
    import win32gui
    import win32con
    HAS_WIN32 = True
except ImportError:
    HAS_WIN32 = False


class RegionConfig:
    """Configuration for screen capture region"""
    
    def __init__(self, x: int = 0, y: int = 0, width: int = 1920, height: int = 1080, hwnd: Optional[int] = None, presentation_mode: bool = False):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.hwnd = hwnd  # Store window handle for live tracking
        self.presentation_mode = presentation_mode  # Black out everything else, show only this window
    
    def to_dict(self) -> Dict:
        return {
            'x': self.x,
            'y': self.y,
            'width': self.width,
            'height': self.height,
            'presentation_mode': self.presentation_mode
        }
    
    def __str__(self) -> str:
        mode_str = " [PRESENTATION MODE]" if self.presentation_mode else ""
        return f"Region(x={self.x}, y={self.y}, width={self.width}, height={self.height}){mode_str}"


class WindowInfo:
    """Information about a window"""
    
    def __init__(self, title: str, x: int, y: int, width: int, height: int, hwnd: Optional[int] = None):
        self.title = title
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.hwnd = hwnd
    
    def to_region_config(self) -> RegionConfig:
        """Convert window info to region config"""
        return RegionConfig(self.x, self.y, self.width, self.height, self.hwnd)
    
    def __str__(self) -> str:
        return f"{self.title} ({self.width}x{self.height} at {self.x},{self.y})"


class WindowEnumerator:
    """Enumerates available windows on the system"""
    
    @staticmethod
    def get_windows() -> List[WindowInfo]:
        """Get list of available windows"""
        windows = []
        
        if HAS_WIN32:
            windows = WindowEnumerator._get_windows_win32()
        elif HAS_PYGETWINDOW:
            windows = WindowEnumerator._get_windows_pygetwindow()
        else:
            print("Warning: Cannot enumerate windows (requires pygetwindow or pywin32)")
        
        return windows
    
    @staticmethod
    def _get_windows_win32() -> List[WindowInfo]:
        """Get windows using win32gui"""
        windows = []
        
        def callback(hwnd, extra):
            try:
                if not win32gui.IsWindowVisible(hwnd):
                    return
                
                title = win32gui.GetWindowText(hwnd)
                if not title or len(title.strip()) == 0:
                    return
                
                # Use simple GetWindowRect - works reliably
                rect = win32gui.GetWindowRect(hwnd)
                x, y, x2, y2 = rect
                width = x2 - x
                height = y2 - y
                
                # CRITICAL: Validate window is on-screen and has valid dimensions
                if width <= 0 or height <= 0:
                    return
                
                # Reject windows that are completely off-screen or have negative positions 
                # that would cause capture errors
                if x2 <= 0 or y2 <= 0:  # Window entirely off-screen (left/top)
                    return
                
                # Accept windows even if partially off-screen (mss handles this in set_region)
                windows.append(WindowInfo(title, x, y, width, height, hwnd))
            except Exception as e:
                # Silently skip windows that cause errors
                pass
        
        try:
            win32gui.EnumWindows(callback, None)
        except Exception as e:
            print(f"Error enumerating windows: {e}")
        
        # Sort by window size (largest first) and remove duplicates
        windows = sorted(windows, key=lambda w: w.width * w.height, reverse=True)
        
        # Filter out very small windows and duplicates
        seen_titles = set()
        filtered = []
        for w in windows:
            if w.width > 100 and w.height > 100 and w.title not in seen_titles:
                filtered.append(w)
                seen_titles.add(w.title)
        
        return filtered[:30]  # Limit to top 30 windows
    
    @staticmethod
    def _get_windows_pygetwindow() -> List[WindowInfo]:
        """Get windows using pygetwindow.
        
        NOTE: pygetwindow does NOT provide HWNDs, so capture falls back to
        region-based mode — the region will NOT follow the window if moved
        or resized. Install pywin32 for reliable tracking: pip install pywin32
        """
        print(
            "\nWARNING: pywin32 not available. Window capture will use region-based "
            "fallback (coordinates only).\n"
            "  The region will NOT follow the window if moved/resized.\n"
            "  For reliable specific-window capture: pip install pywin32\n"
        )
        windows = []


class RegionSelector:
    """Interactive region selection tool"""
    
    def __init__(self, screen_width: int, screen_height: int):
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.selecting = False
        self.start_point: Optional[Tuple[int, int]] = None
        self.end_point: Optional[Tuple[int, int]] = None
        self.region: Optional[RegionConfig] = None
    
    def _mouse_callback(self, event, x, y, flags, param):
        """Mouse callback for region selection"""
        if event == cv2.EVENT_LBUTTONDOWN:
            self.start_point = (x, y)
            self.selecting = True
        
        elif event == cv2.EVENT_MOUSEMOVE and self.selecting:
            self.end_point = (x, y)
        
        elif event == cv2.EVENT_LBUTTONUP:
            self.end_point = (x, y)
            self.selecting = False
            if self.start_point and self.end_point:
                self._finalize_selection()
    
    def _finalize_selection(self):
        """Finalize the selected region"""
        if not self.start_point or not self.end_point:
            return
        
        x1, y1 = self.start_point
        x2, y2 = self.end_point
        
        # Ensure coordinates are in correct order
        x_min = min(x1, x2)
        y_min = min(y1, y2)
        x_max = max(x1, x2)
        y_max = max(y1, y2)
        
        # Ensure minimum size
        width = max(100, x_max - x_min)
        height = max(100, y_max - y_min)
        
        self.region = RegionConfig(x_min, y_min, width, height)
    
    def select_region_interactive(self) -> Optional[RegionConfig]:
        """Allow user to select region interactively with visual feedback"""
        print("\n=== Custom Region Selection ===")
        print("Instructions:")
        print("  - Click and drag to select a region")
        print("  - Release to confirm selection")
        print("  - Press 'r' to reset")
        print("  - Press 'Enter' or 'Space' to confirm")
        print("  - Press 'Esc' or 'q' to cancel")
        print()
        
        # Create a temporary window showing the screen for selection
        window_name = "Region Selector - Click and drag to select region (Press Enter to confirm, Esc to cancel)"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(window_name, self._mouse_callback)
        
        # Create a sample overlay canvas (gray rectangle showing capture area)
        canvas = np.zeros((self.screen_height, self.screen_width, 3), dtype=np.uint8)
        canvas[:] = (50, 50, 50)  # Dark gray background
        
        try:
            while True:
                display = canvas.copy()
                
                # Draw crosshair at mouse position
                mouse_x, mouse_y = 0, 0
                try:
                    mouse_x, mouse_y = cv2.getTrackbarPos("X", window_name), cv2.getTrackbarPos("Y", window_name)
                except:
                    pass
                
                # Draw selection rectangle if selecting
                if self.start_point and self.end_point:
                    x1, y1 = self.start_point
                    x2, y2 = self.end_point
                    x_min = min(x1, x2)
                    y_min = min(y1, y2)
                    x_max = max(x1, x2)
                    y_max = max(y1, y2)
                    
                    # Fill selection area with semi-transparent highlight
                    overlay = display.copy()
                    cv2.rectangle(overlay, (x_min, y_min), (x_max, y_max), (0, 255, 0), -1)
                    cv2.addWeighted(overlay, 0.3, display, 0.7, 0, display)
                    
                    # Draw border
                    cv2.rectangle(display, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                    
                    # Draw size information
                    width = x_max - x_min
                    height = y_max - y_min
                    text = f"{width}x{height}"
                    cv2.putText(display, text, (x_min + 10, y_min + 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                # Draw guides (center crosshairs)
                cv2.line(display, (self.screen_width // 2, 0), 
                        (self.screen_width // 2, self.screen_height), (100, 100, 100), 1)
                cv2.line(display, (0, self.screen_height // 2), 
                        (self.screen_width, self.screen_height // 2), (100, 100, 100), 1)
                
                # Draw instructions
                cv2.putText(display, "Click and drag to select region", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
                cv2.putText(display, "Enter=Confirm, Esc=Cancel, R=Reset", (10, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
                
                cv2.imshow(window_name, display)
                
                key = cv2.waitKey(30) & 0xFF
                if key == ord('\r') or key == ord(' '):  # Enter or Space
                    if self.region:
                        break
                elif key == 27 or key == ord('q'):  # Esc or 'q'
                    self.region = None
                    break
                elif key == ord('r'):  # Reset
                    self.start_point = None
                    self.end_point = None
                    self.region = None
        
        finally:
            cv2.destroyWindow(window_name)
        
        return self.region


def show_region_menu(screen_width: int, screen_height: int) -> Tuple[int, Optional[RegionConfig]]:
    """Show menu for region selection and return user choice"""
    print("\n" + "="*60)
    print("REGION-BASED SCREEN CAPTURE - SELECT CAPTURE MODE")
    print("="*60 + "\n")
    
    # Option 1: Full screen
    print("1. Full Screen")
    print(f"   Captures entire screen ({screen_width}x{screen_height})")
    
    # Option 2: Specific window (if available)
    if HAS_WIN32 or HAS_PYGETWINDOW:
        print("\n2. Specific Window")
        print("   Select from available application windows")
        print("   Captures only that window - tab switches won't affect the stream")
    else:
        print("\n2. Specific Window (Not available - requires pygetwindow or pywin32)")
    
    # Option 3: Custom region
    print("\n3. Custom Region")
    print("   Click and drag to select a rectangular region")
    
    print("\n" + "-"*60)
    
    while True:
        try:
            choice = input("\nEnter your choice (1-3): ").strip()
            
            if choice in ['1', '2', '3']:
                choice_num = int(choice)
                
                # Validate window selection is available
                if choice_num == 2 and not (HAS_WIN32 or HAS_PYGETWINDOW):
                    print("Error: Window selection not available. Install with: pip install pygetwindow pywin32")
                    continue
                
                return choice_num, None
            else:
                print(f"Invalid choice. Please enter 1-3.")
        except KeyboardInterrupt:
            print("\nSelection cancelled.")
            return 0, None
    
    return 0, None


def select_window(enable_presentation_mode: bool = False) -> Optional[RegionConfig]:
    """Show window selection menu and return selected window region"""
    windows = WindowEnumerator.get_windows()
    
    if not windows:
        print("No windows found.")
        return None
    
    print("\n" + "="*60)
    print("AVAILABLE WINDOWS")
    print("="*60 + "\n")
    
    for i, window in enumerate(windows, 1):
        # Display window info with validation status
        on_screen_status = "✓ On-screen" if window.x >= 0 and window.y >= 0 else "⚠ Partially off-screen"
        print(f"{i:2d}. {window.title}")
        print(f"    Size: {window.width}x{window.height} at ({window.x}, {window.y}) [{on_screen_status}]")
    
    print("\n" + "-"*60)
    print("Note: Windows partially off-screen will capture the visible portion")
    
    while True:
        try:
            choice = input("\nSelect window number (0 to cancel): ").strip()
            choice_num = int(choice)
            
            if choice_num == 0:
                return None
            
            if 1 <= choice_num <= len(windows):
                selected = windows[choice_num - 1]
                print(f"\nSelected: {selected}")
                print(f"Capturing window region: x={selected.x}, y={selected.y}, width={selected.width}, height={selected.height}")
                
                # Create region config with presentation mode setting
                region_config = selected.to_region_config()
                region_config.presentation_mode = enable_presentation_mode
                if enable_presentation_mode:
                    print("✓ Presentation Mode ENABLED - All other areas will be BLACK (like Google Meet)")
                return region_config
            else:
                print(f"Invalid choice. Please enter 1-{len(windows)} or 0 to cancel.")
        except ValueError:
            print("Invalid input. Please enter a number.")
        except KeyboardInterrupt:
            print("\nSelection cancelled.")
            return None


def get_region_config(screen_width: int, screen_height: int, auto_mode: Optional[int] = None) -> Optional[RegionConfig]:
    """
    Get region configuration from user
    
    Args:
        screen_width: Screen width in pixels
        screen_height: Screen height in pixels
        auto_mode: If provided, skip menu and use this mode (1=full, 2=window, 3=custom)
    
    Returns:
        RegionConfig object or None if cancelled
    """
    mode = auto_mode
    
    if mode is None:
        mode, _ = show_region_menu(screen_width, screen_height)
    
    if mode == 0:
        return None
    
    if mode == 1:
        # Full screen
        config = RegionConfig(0, 0, screen_width, screen_height)
        print(f"\nFull screen selected: {config}")
        return config
    
    elif mode == 2:
        # Specific window - go straight to window selection, no presentation mode
        return select_window(enable_presentation_mode=False)
    
    elif mode == 3:
        # Custom region
        selector = RegionSelector(screen_width, screen_height)
        region = selector.select_region_interactive()
        if region:
            print(f"\nCustom region selected: {region}")
        return region
    
    return None


if __name__ == "__main__":
    # Test the region selector
    print("Region Selector Test")
    config = get_region_config(1920, 1080)
    if config:
        print(f"Selected: {config}")
