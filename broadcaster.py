"""
Mirror-Space Screen Broadcaster (Python)
Captures screen and broadcasts via UDP using diff-frame encoding
"""

import sys
import time
import socket
import struct
from typing import Tuple, List

import mss
import numpy as np
import cv2

from diff_encoder import DiffFrameEncoder


DEFAULT_PORT = 9999
MAX_PACKET_SIZE = 65507  # Max UDP packet size
TARGET_FPS = 30
FRAME_INTERVAL = 1.0 / TARGET_FPS
SHOW_HEATMAP = True  # Set to False to disable heatmap overlay


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
    
    def send_data(self, data: bytes) -> bool:
        """Send data with automatic fragmentation"""
        offset = 0
        packet_index = 0
        total_packets = (len(data) + MAX_PACKET_SIZE - 9) // (MAX_PACKET_SIZE - 8)
        
        while offset < len(data):
            chunk_size = min(MAX_PACKET_SIZE - 8, len(data) - offset)
            
            # Create packet with metadata
            # Format: I=total_packets, I=packet_index, then data
            packet = struct.pack('<II', total_packets, packet_index)
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

    def poll_messages(self, max_messages: int = 8) -> List[str]:
        """Read available feedback messages without blocking"""
        messages: List[str] = []

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
                messages.append(message)

        return messages

    def close(self):
        """Close feedback socket"""
        self.sock.close()


class ScreenCapture:
    """Captures screen using mss library"""
    
    def __init__(self):
        self.sct = mss.mss()
        self.monitor = self.sct.monitors[1]  # Primary monitor
        print(f"Screen capture initialized: {self.monitor['width']}x{self.monitor['height']}")
    
    def capture_frame(self) -> np.ndarray:
        """Capture current screen frame"""
        # Capture screen
        screenshot = self.sct.grab(self.monitor)
        
        # Convert to numpy array (BGR format for OpenCV)
        frame = np.array(screenshot)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
        
        return frame
    
    def get_dimensions(self) -> Tuple[int, int]:
        """Get screen dimensions"""
        return self.monitor['width'], self.monitor['height']
    
    def close(self):
        """Clean up resources"""
        self.sct.close()


def create_heatmap_overlay(frame: np.ndarray, changed_blocks, block_size: int) -> np.ndarray:
    """Create a heatmap overlay showing changed blocks"""
    overlay = frame.copy()
    heatmap = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)
    
    # Mark changed blocks
    for x, y, w, h in changed_blocks:
        heatmap[y:y+h, x:x+w] = 255
    
    # Apply color map (red for changed areas)
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
    
    # Add statistics
    text = f"Changed Blocks: {len(changed_blocks)}"
    cv2.putText(overlay, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                1, (0, 255, 0), 2, cv2.LINE_AA)
    
    return overlay


def main():
    target_ip = "127.0.0.1"  # Default: localhost
    port = DEFAULT_PORT
    
    if len(sys.argv) > 1:
        target_ip = sys.argv[1]
    if len(sys.argv) > 2:
        port = int(sys.argv[2])
    
    print("=== Mirror-Space Screen Broadcaster (Python) ===")
    print(f"Target: {target_ip}:{port}")
    print(f"Target FPS: {TARGET_FPS}")
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
    try:
        capture = ScreenCapture()
        encoder = DiffFrameEncoder(block_size=32, threshold=10, max_changed_block_ratio=0.30)
        broadcaster = UDPBroadcaster(target_ip, port)
        feedback_receiver = FeedbackReceiver(port + 1)
        
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

            # Process receiver feedback before encoding the next frame.
            for message in feedback_receiver.poll_messages():
                if message.startswith("KEYFRAME_REQUEST"):
                    encoder.force_key_frame(reason=f"receiver_mismatch {message}")
                elif message.startswith("NETWORK_UNSTABLE"):
                    encoder.force_key_frame(reason=f"network_instability {message}")
            
            # Capture screen
            frame = capture.capture_frame()
            
            # Encode frame
            encoded_data = encoder.encode(frame, frame_number)
            
            # Show heatmap if enabled
            if heatmap_enabled:
                changed_blocks = encoder.get_changed_block_positions()
                heatmap = create_heatmap_overlay(frame, changed_blocks, encoder.block_size)
                
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
            if not broadcaster.send_data(encoded_data):
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
        print("Broadcaster stopped.")


if __name__ == "__main__":
    main()
