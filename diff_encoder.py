"""
Diff-Frame Encoder/Decoder for Mirror-Space
Implements block-based differential encoding with motion-based delta compression
"""

import struct
import cv2
import numpy as np
from typing import Tuple, List, Optional, Dict
from enum import IntEnum


class PacketType(IntEnum):
    FULL_FRAME = 0
    DIFF_FRAME = 1
    KEY_FRAME = 2
    MOTION_FRAME = 3  # Motion vector + residual blocks


class DiffFrameEncoder:
    """Encodes frames using block-based differential compression"""
    
    def __init__(
        self,
        block_size: int = 32,
        threshold: int = 10,
        max_changed_block_ratio: float = 0.30,
        max_diff_payload_ratio: float = 0.12,
        jpeg_quality: int = 75,
        enable_motion_detection: bool = True,
    ):
        self.block_size = block_size
        self.threshold = threshold
        self.max_changed_block_ratio = max_changed_block_ratio
        self.max_diff_payload_ratio = max_diff_payload_ratio
        self.jpeg_quality = max(30, min(95, jpeg_quality))
        self.enable_motion_detection = enable_motion_detection
        self.previous_frame: Optional[np.ndarray] = None
        self.previous_gray: Optional[np.ndarray] = None  # For optical flow
        self.last_frame_number = 0
        self.key_frame_needed = True
        self.key_frame_reason = "initial_sync"
        
        # Statistics
        self.compression_ratio = 1.0
        self.changed_blocks_count = 0
        self.changed_blocks = []  # List of (x, y, w, h) tuples
        self.last_changed_ratio = 0.0
        self.motion_vectors: Optional[np.ndarray] = None  # Motion vector grid
        self.motion_blocks = []  # Blocks with significant motion

    def set_jpeg_quality(self, jpeg_quality: int):
        """Update JPEG quality used for full/key frames."""
        self.jpeg_quality = max(30, min(95, int(jpeg_quality)))

    def set_threshold(self, threshold: int):
        """Update diff sensitivity threshold (higher means fewer changed blocks)."""
        self.threshold = max(1, int(threshold))

    def set_motion_detection(self, enabled: bool):
        """Enable or disable motion-based encoding at runtime."""
        self.enable_motion_detection = bool(enabled)

    
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
    
    def _calculate_optical_flow(self, frame_gray: np.ndarray) -> Optional[np.ndarray]:
        """Calculate optical flow motion vectors using Lucas-Kanade"""
        if self.previous_gray is None:
            return None
        
        # Use goodFeaturesToTrack + calcOpticalFlowPyrLK for robust motion detection
        corners = cv2.goodFeaturesToTrack(
            self.previous_gray,
            maxCorners=200,
            qualityLevel=0.01,
            minDistance=30,
            blockSize=self.block_size
        )
        
        if corners is None or len(corners) == 0:
            return None
        
        # Calculate motion vectors
        next_pts, status, err = cv2.calcOpticalFlowPyrLK(
            self.previous_gray,
            frame_gray,
            corners,
            None,
            winSize=(self.block_size, self.block_size),
            maxLevel=2,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
        )
        
        if next_pts is None or status is None:
            return None
        
        # Create motion vector grid (one vector per block)
        h, w = frame_gray.shape[:2]
        grid_h = (h + self.block_size - 1) // self.block_size
        grid_w = (w + self.block_size - 1) // self.block_size
        motion_grid = np.zeros((grid_h, grid_w, 2), dtype=np.float32)  # dx, dy per block
        
        # Map feature point motions to block grid
        for i, (corner, next_pt, st) in enumerate(zip(corners, next_pts, status)):
            if st[0]:  # Valid motion
                x, y = corner[0]
                nx, ny = next_pt[0]
                bx = int(x // self.block_size)
                by = int(y // self.block_size)
                
                if 0 <= bx < grid_w and 0 <= by < grid_h:
                    motion_grid[by, bx, 0] += (nx - x)
                    motion_grid[by, bx, 1] += (ny - y)
        
        # Smooth and average motion vectors
        if np.any(motion_grid):
            motion_grid = cv2.GaussianBlur(motion_grid, (3, 3), 1.0)
        
        return motion_grid
    
    def _apply_motion_compensation(self, frame: np.ndarray, motion_grid: np.ndarray) -> np.ndarray:
        """Predict current frame using motion vectors"""
        h, w = frame.shape[:2]
        predicted = self.previous_frame.copy()
        
        # Warp previous frame using motion vectors
        for by in range(motion_grid.shape[0]):
            for bx in range(motion_grid.shape[1]):
                dx, dy = motion_grid[by, bx]
                
                if abs(dx) < 0.5 and abs(dy) < 0.5:
                    continue  # No significant motion
                
                # Block boundaries
                y_start = by * self.block_size
                y_end = min(y_start + self.block_size, h)
                x_start = bx * self.block_size
                x_end = min(x_start + self.block_size, w)
                
                # Source region in previous frame (shifted by motion)
                shift_x = int(round(dx))
                shift_y = int(round(dy))

                src_x0 = x_start + shift_x
                src_y0 = y_start + shift_y
                src_x1 = x_end + shift_x
                src_y1 = y_end + shift_y

                dst_x0 = x_start
                dst_y0 = y_start
                dst_x1 = x_end
                dst_y1 = y_end

                # Clip left/top by moving both source and destination starts.
                if src_x0 < 0:
                    dst_x0 -= src_x0
                    src_x0 = 0
                if src_y0 < 0:
                    dst_y0 -= src_y0
                    src_y0 = 0

                # Clip right/bottom by shrinking both source and destination ends.
                if src_x1 > w:
                    overflow = src_x1 - w
                    dst_x1 -= overflow
                    src_x1 = w
                if src_y1 > h:
                    overflow = src_y1 - h
                    dst_y1 -= overflow
                    src_y1 = h

                # Apply only valid overlapping region.
                if dst_x0 < dst_x1 and dst_y0 < dst_y1:
                    predicted[dst_y0:dst_y1, dst_x0:dst_x1] = self.previous_frame[src_y0:src_y1, src_x0:src_x1]
        
        return predicted
    
    def _encode_motion_frame(self, frame: np.ndarray, frame_number: int, motion_grid: np.ndarray) -> bytes:
        """Encode frame using motion compensation + residual blocks"""
        h, w = frame.shape[:2]
        
        # Predict frame using motion compensation
        predicted = self._apply_motion_compensation(frame, motion_grid)
        
        # Find residual blocks (differences between predicted and actual)
        residual_data = bytearray()
        motion_blocks = []
        
        grid_h = motion_grid.shape[0]
        grid_w = motion_grid.shape[1]
        
        for by in range(grid_h):
            for bx in range(grid_w):
                y_start = by * self.block_size
                y_end = min(y_start + self.block_size, h)
                x_start = bx * self.block_size
                x_end = min(x_start + self.block_size, w)
                
                actual_block = frame[y_start:y_end, x_start:x_end]
                predicted_block = predicted[y_start:y_end, x_start:x_end]
                
                # Check if residual is significant
                residual = np.abs(actual_block.astype(np.int16) - predicted_block.astype(np.int16))
                avg_residual = np.mean(residual)
                
                if avg_residual > self.threshold:
                    # Store motion vector + residual
                    bw = x_end - x_start
                    bh = y_end - y_start
                    
                    motion_vec = struct.pack('<ff', motion_grid[by, bx, 0], motion_grid[by, bx, 1])
                    residual_data.extend(motion_vec)
                    
                    block_header = struct.pack('<HHHH', x_start, y_start, bw, bh)
                    residual_data.extend(block_header)
                    
                    residual_data.extend(actual_block.tobytes())
                    motion_blocks.append((x_start, y_start, bw, bh))
        
        # Build packet header
        header = struct.pack('<BIIIIH',
                           PacketType.MOTION_FRAME,
                           frame_number,
                           w,
                           h,
                           len(residual_data),
                           self.block_size)
        
        self.motion_blocks = motion_blocks
        self.compression_ratio = len(residual_data) / (h * w * 3) if (h * w > 0) else 0
        
        return header + bytes(residual_data)

    def _encode_full_frame(
        self,
        frame: np.ndarray,
        frame_number: int,
        packet_type: PacketType = PacketType.FULL_FRAME,
    ) -> bytes:
        """Encode a complete frame with JPEG compression"""
        height, width = frame.shape[:2]
        
        # JPEG compress the frame
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality]
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
        """Encode a frame using motion-based, diff, or key frame method"""
        if self.previous_frame is not None and self.previous_frame.shape != frame.shape:
            # Resolution changed; reset reference chain and force resync.
            self.previous_gray = None
            self.force_key_frame(reason="resolution_changed")

        send_key_frame = self.key_frame_needed or self.previous_frame is None
        diff_data = None
        
        # Convert to grayscale for optical flow
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if not send_key_frame:
            # Try motion-based encoding first if enabled
            if self.enable_motion_detection and self.previous_gray is not None:
                motion_grid = self._calculate_optical_flow(frame_gray)
                
                if motion_grid is not None and np.any(motion_grid):
                    motion_data = self._encode_motion_frame(frame, frame_number, motion_grid)
                    
                    # Use motion frame if it's efficient enough
                    if self.compression_ratio < self.max_diff_payload_ratio:
                        data = motion_data
                        print(f"Sending MOTION frame #{frame_number} ({len(data)} bytes, "
                              f"{len(self.motion_blocks)} motion blocks, "
                              f"{self.compression_ratio*100:.1f}% of original)")
                        
                        self.previous_frame = frame.copy()
                        self.previous_gray = frame_gray.copy()
                        self.last_frame_number = frame_number
                        return data
            
            # Fall back to diff frame
            diff_data = self._encode_diff_frame(frame, frame_number)

            # If raw diff payload is too large, send a full/key frame instead.
            if self.compression_ratio >= self.max_diff_payload_ratio:
                self.key_frame_needed = True
                self.key_frame_reason = (
                    f"diff_payload_high ratio={self.compression_ratio:.1%} "
                    f"threshold={self.max_diff_payload_ratio:.1%}"
                )
                send_key_frame = True

            # Escalate to key frame during high-motion scenes.
            elif self.last_changed_ratio >= self.max_changed_block_ratio:
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
        self.previous_gray = frame_gray.copy()
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
    
    def get_motion_block_positions(self) -> List[Tuple[int, int, int, int]]:
        """Get positions of motion blocks (x, y, w, h) for heatmap"""
        return self.motion_blocks if self.motion_blocks else []


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
        
        elif packet_type == PacketType.MOTION_FRAME:
            if self.current_frame is None:
                self._set_error("missing_reference_frame_for_motion")
                return None

            if self.last_frame_number is not None and frame_number != self.last_frame_number + 1:
                self._set_error(
                    f"frame_gap expected={self.last_frame_number + 1} got={frame_number}"
                )
                return None
            
            # Apply motion-based residual blocks
            offset = 0
            blocks_applied = 0
            
            while offset < len(payload):
                if offset + 8 > len(payload):
                    break
                
                # Parse motion vector (8 bytes: 2 floats)
                try:
                    dx, dy = struct.unpack('<ff', payload[offset:offset+8])
                    offset += 8
                except:
                    break
                
                if offset + 8 > len(payload):
                    break
                
                # Parse block header
                x, y, bw, bh = struct.unpack('<HHHH', payload[offset:offset+8])
                offset += 8
                
                # Read residual pixel data (actual frame pixels, not delta)
                block_data_size = bw * bh * 3
                if offset + block_data_size > len(payload):
                    self._set_error("incomplete_motion_block_data")
                    break
                
                # Apply block to current frame (residual is already the final pixel data)
                block_bytes = payload[offset:offset+block_data_size]
                block = np.frombuffer(block_bytes, dtype=np.uint8).reshape(bh, bw, 3)
                
                h, w = self.current_frame.shape[:2]
                if y + bh <= h and x + bw <= w:
                    self.current_frame[y:y+bh, x:x+bw] = block
                
                offset += block_data_size
                blocks_applied += 1
            
            print(f"Received MOTION frame #{frame_number} ({blocks_applied} blocks, motion detected)")
            self.last_frame_number = frame_number

        else:
            self._set_error(f"unknown_packet_type={packet_type}")
            return None
        
        return self.current_frame.copy() if self.current_frame is not None else None
