"""
Mirror-Space Screen Broadcaster (Python)
Captures screen and broadcasts via UDP using diff-frame encoding
"""

import sys
import time
import socket
import struct
import secrets
from typing import Tuple, List, Optional, Dict

import mss
import numpy as np
import cv2
from zeroconf import ServiceInfo, Zeroconf

from diff_encoder import DiffFrameEncoder


DEFAULT_PORT = 9999
DEFAULT_BROADCAST_TARGET = "255.255.255.255"
SERVICE_TYPE = "_mirror-space._udp.local."
DISCOVERY_BEACON_PORT = 10001
DISCOVERY_BEACON_INTERVAL = 1.0
MAX_PACKET_SIZE = 65507  # Max UDP packet size
FRAGMENT_HEADER_SIZE = 12  # total_packets (4) + packet_index (4) + frame_number (4)
MAX_UDP_PAYLOAD_SIZE = 1400  # MTU-safe payload to avoid IP fragmentation on LAN/WiFi
INITIAL_TARGET_FPS = 20
MIN_STREAM_FPS = 8
MAX_STREAM_FPS = 45
SHOW_HEATMAP = False  # Disable by default to prioritize low-latency pipeline timing.
MAX_CHANGED_BLOCK_RATIO = 0.35  # Fallback to key frame when too many blocks change
MAX_DIFF_PAYLOAD_RATIO = 0.25  # Fallback to key frame when diff payload gets too large
INITIAL_JPEG_QUALITY = 60
MIN_JPEG_QUALITY = 35
MAX_JPEG_QUALITY = 85
INITIAL_DIFF_THRESHOLD = 10
MIN_DIFF_THRESHOLD = 6
MAX_DIFF_THRESHOLD = 24
ADAPTIVE_WIDTH_STEPS = [640, 854, 960, 1280, 1600, 1920, 2560]
INITIAL_STREAM_WIDTH = 1280
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
        try:
            self.sock.setsockopt(socket.IPPROTO_IP, socket.IP_TOS, 0x10)  # IPTOS_LOWDELAY
        except OSError:
            pass
        
        # Keep buffer moderate to reduce queue-induced latency under congestion.
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 256 * 1024)
        
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


def _parse_message_tokens(message: str) -> Dict[str, str]:
    values: Dict[str, str] = {}
    for token in message.split()[1:]:
        if "=" not in token:
            continue
        key, value = token.split("=", 1)
        values[key] = value
    return values


def _clamp_int(value: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, value))


class AdaptiveStreamingController:
    """Adjusts stream quality knobs to keep latency low under changing network conditions."""

    def __init__(self):
        self.min_fps = MIN_STREAM_FPS
        self.max_fps = MAX_STREAM_FPS
        self.width_steps = ADAPTIVE_WIDTH_STEPS

        self.current_fps = _clamp_int(INITIAL_TARGET_FPS, self.min_fps, self.max_fps)
        self.current_width = self._pick_width(INITIAL_STREAM_WIDTH)
        self.current_jpeg_quality = _clamp_int(INITIAL_JPEG_QUALITY, MIN_JPEG_QUALITY, MAX_JPEG_QUALITY)
        self.current_diff_threshold = _clamp_int(
            INITIAL_DIFF_THRESHOLD,
            MIN_DIFF_THRESHOLD,
            MAX_DIFF_THRESHOLD,
        )

        self._bad_windows = 0
        self._good_windows = 0
        self._last_adjust_time = 0.0

    def _pick_width(self, requested_width: int) -> int:
        for width in reversed(self.width_steps):
            if width <= requested_width:
                return width
        return self.width_steps[0]

    def _width_index(self) -> int:
        return self.width_steps.index(self.current_width)

    def get_frame_interval(self) -> float:
        return 1.0 / float(self.current_fps)

    def _degrade(self, severe: bool) -> bool:
        changed = False

        fps_scale = 0.70 if severe else 0.85
        next_fps = _clamp_int(int(round(self.current_fps * fps_scale)), self.min_fps, self.max_fps)
        if next_fps == self.current_fps and self.current_fps > self.min_fps:
            next_fps = self.current_fps - 1
        if next_fps != self.current_fps:
            self.current_fps = next_fps
            changed = True

        width_drop = 2 if severe else 1
        next_width_index = max(0, self._width_index() - width_drop)
        if self.width_steps[next_width_index] != self.current_width:
            self.current_width = self.width_steps[next_width_index]
            changed = True

        quality_drop = 8 if severe else 4
        next_quality = _clamp_int(self.current_jpeg_quality - quality_drop, MIN_JPEG_QUALITY, MAX_JPEG_QUALITY)
        if next_quality != self.current_jpeg_quality:
            self.current_jpeg_quality = next_quality
            changed = True

        threshold_rise = 3 if severe else 2
        next_threshold = _clamp_int(
            self.current_diff_threshold + threshold_rise,
            MIN_DIFF_THRESHOLD,
            MAX_DIFF_THRESHOLD,
        )
        if next_threshold != self.current_diff_threshold:
            self.current_diff_threshold = next_threshold
            changed = True

        return changed

    def _upgrade(self) -> bool:
        changed = False

        next_fps = _clamp_int(self.current_fps + 2, self.min_fps, self.max_fps)
        if next_fps != self.current_fps:
            self.current_fps = next_fps
            changed = True

        next_width_index = min(len(self.width_steps) - 1, self._width_index() + 1)
        if self.width_steps[next_width_index] != self.current_width:
            self.current_width = self.width_steps[next_width_index]
            changed = True

        next_quality = _clamp_int(self.current_jpeg_quality + 2, MIN_JPEG_QUALITY, MAX_JPEG_QUALITY)
        if next_quality != self.current_jpeg_quality:
            self.current_jpeg_quality = next_quality
            changed = True

        next_threshold = _clamp_int(
            self.current_diff_threshold - 1,
            MIN_DIFF_THRESHOLD,
            MAX_DIFF_THRESHOLD,
        )
        if next_threshold != self.current_diff_threshold:
            self.current_diff_threshold = next_threshold
            changed = True

        return changed

    def apply_feedback(self, stats: Dict[str, float], now: float) -> Tuple[bool, str]:
        """Adapt quality levels from receiver packet loss, partial frames, and receiver FPS."""
        if now - self._last_adjust_time < 1.0:
            return False, ""

        packet_loss = stats.get("packet_loss", 0.0)
        partial_ratio = stats.get("partial_ratio", 0.0)
        recv_fps = stats.get("recv_fps", 0.0)

        severe = packet_loss >= 0.12 or partial_ratio >= 0.25
        bad = severe or packet_loss >= 0.05 or partial_ratio >= 0.10
        if recv_fps > 0.0 and recv_fps < (self.current_fps * 0.70):
            bad = True

        good = packet_loss <= 0.01 and partial_ratio <= 0.02
        if recv_fps > 0.0:
            good = good and recv_fps >= (self.current_fps * 0.90)

        changed = False
        decision = ""

        if bad:
            self._bad_windows += 1
            self._good_windows = 0
            required = 1 if severe else 2
            if self._bad_windows >= required:
                changed = self._degrade(severe=severe)
                self._bad_windows = 0
                decision = "severe_congestion" if severe else "congestion"
        elif good:
            self._good_windows += 1
            self._bad_windows = 0
            if self._good_windows >= 4:
                changed = self._upgrade()
                self._good_windows = 0
                decision = "network_recovered"
        else:
            self._bad_windows = 0
            self._good_windows = 0

        if changed:
            self._last_adjust_time = now
            summary = (
                f"{decision} recv_fps={recv_fps:.1f} "
                f"loss={packet_loss:.1%} partial={partial_ratio:.1%}"
            )
            return True, summary

        return False, ""


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
    print(f"Initial FPS: {INITIAL_TARGET_FPS}")
    print(f"Max Changed Block Ratio: {MAX_CHANGED_BLOCK_RATIO:.0%}")
    print(f"Max Diff Payload Ratio: {MAX_DIFF_PAYLOAD_RATIO:.0%}")
    print(f"Initial JPEG Quality: {INITIAL_JPEG_QUALITY}")
    print(f"Initial Max Stream Width: {INITIAL_STREAM_WIDTH}")
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
    adaptive = AdaptiveStreamingController()
    try:
        capture = ScreenCapture()
        encoder = DiffFrameEncoder(
            block_size=32,
            threshold=adaptive.current_diff_threshold,
            max_changed_block_ratio=MAX_CHANGED_BLOCK_RATIO,
            max_diff_payload_ratio=MAX_DIFF_PAYLOAD_RATIO,
            jpeg_quality=adaptive.current_jpeg_quality,
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
            print("Broadcaster is ready. Waiting for receiver connection...")
            print(f"Session Access ID: {access_id}")
            print("Streaming will start only after RECEIVER_HELLO is received.\n")
        
        # Create heatmap window if enabled
        heatmap_enabled = SHOW_HEATMAP
        if SHOW_HEATMAP:
            cv2.namedWindow("Broadcaster Heatmap", cv2.WINDOW_NORMAL)
        
        # Broadcasting loop
        frame_number = 0
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

                if active_receiver_ip is not None and sender_ip != active_receiver_ip:
                    # Ignore health events from non-selected receivers in connect-gated mode.
                    continue

                if message.startswith("KEYFRAME_REQUEST"):
                    encoder.force_key_frame(reason=f"receiver_mismatch {message}")
                elif message.startswith("NETWORK_UNSTABLE"):
                    encoder.force_key_frame(reason=f"network_instability {message}")
                elif message.startswith("STREAM_STATS"):
                    fields = _parse_message_tokens(message)

                    def _as_float(name: str) -> float:
                        raw = fields.get(name, "0")
                        try:
                            return float(raw)
                        except ValueError:
                            return 0.0

                    stats = {
                        "recv_fps": _as_float("recv_fps"),
                        "packet_loss": _as_float("packet_loss"),
                        "partial_ratio": _as_float("partial_ratio"),
                    }

                    changed, reason = adaptive.apply_feedback(stats, now=time.time())
                    if changed:
                        encoder.set_jpeg_quality(adaptive.current_jpeg_quality)
                        encoder.set_threshold(adaptive.current_diff_threshold)
                        encoder.force_key_frame(reason=f"adaptive_reconfigure {reason}")
                        print(
                            "Adaptive update: "
                            f"fps={adaptive.current_fps} "
                            f"width={adaptive.current_width} "
                            f"jpeg={adaptive.current_jpeg_quality} "
                            f"threshold={adaptive.current_diff_threshold}"
                        )

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
            if frame.shape[1] > adaptive.current_width:
                scale = adaptive.current_width / frame.shape[1]
                resized_height = max(1, int(frame.shape[0] * scale))
                frame = cv2.resize(
                    frame,
                    (adaptive.current_width, resized_height),
                    interpolation=cv2.INTER_AREA,
                )
            
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
                print(
                    f"Actual FPS: {actual_fps:.1f} | target={adaptive.current_fps} "
                    f"width={adaptive.current_width} jpeg={adaptive.current_jpeg_quality}"
                )
                fps_counter = 0
                fps_start_time = time.time()
            
            # Frame rate limiting
            frame_time = time.time() - frame_start
            sleep_time = adaptive.get_frame_interval() - frame_time
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
