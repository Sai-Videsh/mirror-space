"""
Mirror-Space Screen Receiver (Python)
Receives and displays screen broadcasts via UDP
"""

import sys
import time
import socket
import struct
from typing import Dict, Optional, Tuple

import cv2
import numpy as np

from diff_encoder import DiffFrameDecoder


DEFAULT_PORT = 9999
MAX_PACKET_SIZE = 65507
RECEIVE_TIMEOUT = 5.0  # seconds


class UDPReceiver:
    """Handles UDP packet reception and reassembly"""
    
    def __init__(self, port: int):
        self.port = port
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        
        # Bind to all interfaces
        self.sock.bind(('0.0.0.0', port))
        
        # Set receive timeout
        self.sock.settimeout(RECEIVE_TIMEOUT)
        
        # Increase receive buffer size
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1024*1024)

        # Packet health statistics
        self.complete_frames = 0
        self.partial_frames = 0
        self.missing_packets = 0
        
        print(f"UDP receiver initialized on port {port}")
    
    def receive_data(self) -> Tuple[Optional[bytes], Optional[Dict], Optional[Tuple[str, int]]]:
        """Receive and reassemble fragmented data"""
        packets: Dict[int, bytes] = {}
        total_packets = 0
        start_time = time.time()
        source_addr: Optional[Tuple[str, int]] = None
        partial_due_to_timeout = False
        
        while True:
            # Check timeout
            if time.time() - start_time > 1.0 and packets:
                partial_due_to_timeout = True
                break  # Partial frame, return what we have
            
            try:
                data, addr = self.sock.recvfrom(MAX_PACKET_SIZE)
            except socket.timeout:
                continue
            except Exception as e:
                print(f"Receive error: {e}")
                return None, None, None
            
            if len(data) < 8:
                continue  # Too small for header

            source_addr = addr
            
            # Parse packet header
            total, index = struct.unpack('<II', data[:8])
            payload = data[8:]
            
            if total_packets == 0:
                total_packets = total
            
            # Store packet
            packets[index] = payload
            
            # Check if we have all packets
            if len(packets) == total_packets:
                break
        
        if not packets:
            return None, None, source_addr

        missing_count = 0
        
        # Reassemble data in order
        result = bytearray()
        for i in range(total_packets):
            if i in packets:
                result.extend(packets[i])
            else:
                print(f"Missing packet {i} of {total_packets}")
                missing_count += 1

        if missing_count == 0 and not partial_due_to_timeout:
            self.complete_frames += 1
        else:
            self.partial_frames += 1
            self.missing_packets += missing_count

        meta = {
            "complete": missing_count == 0 and not partial_due_to_timeout,
            "missing_packets": missing_count,
            "total_packets": total_packets,
            "timed_out": partial_due_to_timeout,
        }

        return (bytes(result) if result else None), meta, source_addr
    
    def close(self):
        """Close the socket"""
        self.sock.close()


class FeedbackSender:
    """Sends health and mismatch feedback to broadcaster"""

    def __init__(self):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.last_send_time: Dict[str, float] = {}

    def send(
        self,
        host: str,
        port: int,
        message: str,
        throttle_seconds: float = 1.0,
        throttle_key: Optional[str] = None,
    ):
        now = time.time()
        key = throttle_key if throttle_key is not None else message
        last = self.last_send_time.get(key, 0.0)
        if now - last < throttle_seconds:
            return

        try:
            self.sock.sendto(message.encode('utf-8'), (host, port))
            self.last_send_time[key] = now
            print(f"Feedback sent: {message} -> {host}:{port}")
        except Exception as e:
            print(f"Feedback send failed: {e}")

    def close(self):
        self.sock.close()


def main():
    port = DEFAULT_PORT
    
    if len(sys.argv) > 1:
        port = int(sys.argv[1])
    
    print("=== Mirror-Space Screen Receiver (Python) ===")
    print(f"Listening on port: {port}")
    print("Press ESC or 'q' to quit...\n")
    
    # Initialize receiver
    receiver = None
    feedback = None
    try:
        receiver = UDPReceiver(port)
        decoder = DiffFrameDecoder()
        feedback = FeedbackSender()
        
        # Create display window
        cv2.namedWindow("Mirror-Space Receiver", cv2.WINDOW_NORMAL)
        
        # Receiving loop
        frames_received = 0
        fps_start_time = time.time()
        health_window_start = time.time()
        window_partial_frames = 0
        window_missing_packets = 0
        window_total_packets = 0
        sender_ip: Optional[str] = None
        
        print("Waiting for frames...\n")
        
        while True:
            # Receive data
            received_data, meta, source_addr = receiver.receive_data()

            if source_addr is not None:
                sender_ip = source_addr[0]
            
            if received_data:
                # Decode frame
                frame = decoder.decode(received_data)

                decode_error = decoder.consume_decoder_error()
                if decode_error and sender_ip:
                    feedback.send(
                        sender_ip,
                        port + 1,
                        f"KEYFRAME_REQUEST reason={decode_error}",
                        throttle_seconds=0.2,
                        throttle_key="KEYFRAME_REQUEST",
                    )
                
                if frame is not None:
                    # Display frame
                    cv2.imshow("Mirror-Space Receiver", frame)
                    frames_received += 1

            if meta:
                window_total_packets += meta.get("total_packets", 0)
                window_missing_packets += meta.get("missing_packets", 0)
                if not meta.get("complete", True):
                    window_partial_frames += 1

            # Periodically report network instability to force sender key frames.
            health_elapsed = time.time() - health_window_start
            if health_elapsed >= 2.0:
                packet_loss_ratio = (
                    window_missing_packets / window_total_packets if window_total_packets > 0 else 0.0
                )
                unstable = window_partial_frames >= 2 or packet_loss_ratio >= 0.05

                if unstable and sender_ip:
                    feedback.send(
                        sender_ip,
                        port + 1,
                        (
                            "NETWORK_UNSTABLE "
                            f"partial_frames={window_partial_frames} "
                            f"packet_loss={packet_loss_ratio:.1%}"
                        ),
                        throttle_seconds=0.5,
                        throttle_key="NETWORK_UNSTABLE",
                    )

                window_partial_frames = 0
                window_missing_packets = 0
                window_total_packets = 0
                health_window_start = time.time()
            
            # Calculate FPS
            elapsed = time.time() - fps_start_time
            if elapsed >= 1.0:
                actual_fps = frames_received / elapsed
                print(f"Receive FPS: {actual_fps:.1f}")
                frames_received = 0
                fps_start_time = time.time()
            
            # Check for exit (ESC or 'q')
            key = cv2.waitKey(1) & 0xFF
            if key == 27 or key == ord('q'):  # ESC or 'q'
                break
    
    except KeyboardInterrupt:
        print("\nStopping receiver...")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if receiver is not None:
            receiver.close()
        if feedback is not None:
            feedback.close()
        cv2.destroyAllWindows()
        print("Receiver stopped.")


if __name__ == "__main__":
    main()
