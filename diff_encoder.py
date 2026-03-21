"""
Diff-Frame Encoder/Decoder for Mirror-Space
Implements block-based differential encoding for efficient screen streaming
"""

import struct
import cv2
import numpy as np
from typing import Tuple, List, Optional
from enum import IntEnum


class PacketType(IntEnum):
    FULL_FRAME = 0
    DIFF_FRAME = 1
    KEY_FRAME = 2


class DiffFrameEncoder:
    """Encodes frames using block-based differential compression"""
    
    def __init__(
        self,
        block_size: int = 32,
        threshold: int = 10,
        max_changed_block_ratio: float = 0.30,
    ):
        self.block_size = block_size
        self.threshold = threshold
        self.max_changed_block_ratio = max_changed_block_ratio
        self.previous_frame: Optional[np.ndarray] = None
        self.last_frame_number = 0
        self.key_frame_needed = True
        self.key_frame_reason = "initial_sync"
        
        # Statistics
        self.compression_ratio = 1.0
        self.changed_blocks_count = 0
        self.changed_blocks = []  # List of (x, y, w, h) tuples
        self.last_changed_ratio = 0.0
    
    def _has_block_changed(self, frame1: np.ndarray, frame2: np.ndarray,
                          x: int, y: int, width: int, height: int) -> bool:
        """Check if a block has changed between two frames"""
        h, w = frame1.shape[:2]
        
        # Ensure we don't go out of bounds
        y_end = min(y + height, h)
        x_end = min(x + width, w)
        
        block1 = frame1[y:y_end, x:x_end]
        block2 = frame2[y:y_end, x:x_end]
        
        # Calculate mean absolute difference
        diff = np.abs(block1.astype(np.int16) - block2.astype(np.int16))
        avg_diff = np.mean(diff)
        
        return avg_diff > self.threshold
    
    def _encode_full_frame(
        self,
        frame: np.ndarray,
        frame_number: int,
        packet_type: PacketType = PacketType.FULL_FRAME,
    ) -> bytes:
        """Encode a complete frame with JPEG compression"""
        height, width = frame.shape[:2]
        
        # JPEG compress the frame
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 85]
        _, compressed = cv2.imencode('.jpg', frame, encode_param)
        compressed_bytes = compressed.tobytes()
        
        # Build packet header
        # Format: B=type, I=frame_number, I=width, I=height, I=data_size, H=block_size
        header = struct.pack('<BIIIIH',
                           packet_type,
                           frame_number,
                           width,
                           height,
                           len(compressed_bytes),
                           self.block_size)
        
        return header + compressed_bytes
    
    def _encode_diff_frame(self, frame: np.ndarray, frame_number: int) -> bytes:
        """Encode only the changed blocks"""
        height, width = frame.shape[:2]
        diff_data = bytearray()
        self.changed_blocks_count = 0
        self.changed_blocks = []  # Store block positions for heatmap
        
        # Find changed blocks
        for y in range(0, height, self.block_size):
            for x in range(0, width, self.block_size):
                bw = min(self.block_size, width - x)
                bh = min(self.block_size, height - y)
                
                if self._has_block_changed(self.previous_frame, frame, x, y, bw, bh):
                    # Block changed - add to diff data
                    # Format: H=x, H=y, H=width, H=height
                    block_header = struct.pack('<HHHH', x, y, bw, bh)
                    diff_data.extend(block_header)
                    
                    # Add pixel data
                    block = frame[y:y+bh, x:x+bw]
                    diff_data.extend(block.tobytes())
                    
                    # Store for heatmap
                    self.changed_blocks.append((x, y, bw, bh))
                    self.changed_blocks_count += 1
        
        # Calculate compression ratio
        original_size = height * width * 3
        self.compression_ratio = len(diff_data) / original_size if original_size > 0 else 0
        total_blocks = ((width + self.block_size - 1) // self.block_size) * ((height + self.block_size - 1) // self.block_size)
        self.last_changed_ratio = self.changed_blocks_count / total_blocks if total_blocks > 0 else 0.0
        
        # Build packet header
        header = struct.pack('<BIIIIH',
                           PacketType.DIFF_FRAME,
                           frame_number,
                           width,
                           height,
                           len(diff_data),
                           self.block_size)
        
        return header + bytes(diff_data)
    
    def encode(self, frame: np.ndarray, frame_number: int) -> bytes:
        """Encode a frame (returns full or diff frame data)"""
        send_key_frame = self.key_frame_needed or self.previous_frame is None
        diff_data = None

        if not send_key_frame:
            diff_data = self._encode_diff_frame(frame, frame_number)

            # Escalate to key frame during high-motion scenes.
            if self.last_changed_ratio >= self.max_changed_block_ratio:
                self.key_frame_needed = True
                self.key_frame_reason = (
                    f"high_motion changed={self.last_changed_ratio:.1%} "
                    f"threshold={self.max_changed_block_ratio:.1%}"
                )
                send_key_frame = True

        if send_key_frame:
            packet_type = PacketType.FULL_FRAME if self.previous_frame is None else PacketType.KEY_FRAME
            data = self._encode_full_frame(frame, frame_number, packet_type=packet_type)
            reason = self.key_frame_reason
            self.key_frame_needed = False
            self.key_frame_reason = ""
            frame_type = "FULL" if packet_type == PacketType.FULL_FRAME else "KEY"
            print(f"Sending {frame_type} frame #{frame_number} ({len(data)} bytes) reason={reason}")
        else:
            data = diff_data if diff_data is not None else self._encode_diff_frame(frame, frame_number)
            print(f"Sending DIFF frame #{frame_number} ({len(data)} bytes, "
                  f"{self.changed_blocks_count} blocks, "
                  f"{self.compression_ratio*100:.1f}% of original, "
                  f"changed_ratio={self.last_changed_ratio:.1%})")
        
        self.previous_frame = frame.copy()
        self.last_frame_number = frame_number
        
        return data
    
    def force_key_frame(self, reason: str = "external_trigger"):
        """Force next frame to be a key frame"""
        self.key_frame_needed = True
        self.key_frame_reason = reason
    
    def get_compression_ratio(self) -> float:
        """Get current compression ratio"""
        return self.compression_ratio
    
    def get_changed_blocks(self) -> int:
        """Get number of changed blocks in last frame"""
        return self.changed_blocks_count
    
    def get_changed_block_positions(self) -> List[Tuple[int, int, int, int]]:
        """Get positions of changed blocks (x, y, w, h) for heatmap"""
        return self.changed_blocks


class DiffFrameDecoder:
    """Decodes diff-frame encoded video stream"""
    
    def __init__(self):
        self.current_frame: Optional[np.ndarray] = None
        self.last_frame_number: Optional[int] = None
        self._last_error: Optional[str] = None

    def _set_error(self, message: str):
        self._last_error = message
        print(f"Decoder mismatch: {message}")

    def consume_decoder_error(self) -> Optional[str]:
        """Read and clear decoder mismatch reason"""
        error = self._last_error
        self._last_error = None
        return error
    
    def decode(self, data: bytes) -> Optional[np.ndarray]:
        """Decode received packet data"""
        if len(data) < 19:  # Minimum header size (1+4+4+4+4+2)
            self._set_error(f"packet_too_small size={len(data)}")
            return None
        
        # Parse header  
        header_size = 19
        packet_type, frame_number, width, height, data_size, block_size = \
            struct.unpack('<BIIIIH', data[:header_size])
        
        payload = data[header_size:]

        if data_size != len(payload):
            self._set_error(f"payload_size_mismatch expected={data_size} actual={len(payload)}")
            return None
        
        if packet_type == PacketType.FULL_FRAME or packet_type == PacketType.KEY_FRAME:
            # Decode JPEG compressed frame
            nparr = np.frombuffer(payload, np.uint8)
            self.current_frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if self.current_frame is None:
                self._set_error("jpeg_decode_failed")
                return None
            
            label = "FULL" if packet_type == PacketType.FULL_FRAME else "KEY"
            print(f"Received {label} frame #{frame_number}")
            self.last_frame_number = frame_number
            
        elif packet_type == PacketType.DIFF_FRAME:
            if self.current_frame is None:
                self._set_error("missing_reference_frame")
                return None

            if self.last_frame_number is not None and frame_number != self.last_frame_number + 1:
                self._set_error(
                    f"frame_gap expected={self.last_frame_number + 1} got={frame_number}"
                )
                return None
            
            # Apply diff blocks
            offset = 0
            blocks_applied = 0
            
            while offset < len(payload):
                if offset + 8 > len(payload):
                    break
                
                # Parse block header
                x, y, bw, bh = struct.unpack('<HHHH', payload[offset:offset+8])
                offset += 8
                
                # Read pixel data
                block_data_size = bw * bh * 3
                if offset + block_data_size > len(payload):
                    self._set_error("incomplete_block_data")
                    break
                
                # Apply block to current frame
                block_bytes = payload[offset:offset+block_data_size]
                block = np.frombuffer(block_bytes, dtype=np.uint8).reshape(bh, bw, 3)
                
                # Update frame
                h, w = self.current_frame.shape[:2]
                if y + bh <= h and x + bw <= w:
                    self.current_frame[y:y+bh, x:x+bw] = block
                
                offset += block_data_size
                blocks_applied += 1
            
            print(f"Received DIFF frame #{frame_number} ({blocks_applied} blocks)")
            self.last_frame_number = frame_number
        else:
            self._set_error(f"unknown_packet_type={packet_type}")
            return None
        
        return self.current_frame.copy() if self.current_frame is not None else None
