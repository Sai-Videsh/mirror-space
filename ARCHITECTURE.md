# Architecture Overview

## System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    BROADCASTER (Sender)                      │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  ┌──────────────┐    ┌──────────────┐   ┌──────────────┐   │
│  │Screen Capture│───▶│Diff Encoder  │──▶│UDP Broadcaster│   │
│  │  (GDI API)   │    │ (Block-based)│   │  (Fragments)  │──┐│
│  └──────────────┘    └──────────────┘   └──────────────┘  ││
│         │                    │                              ││
│         │                    │                              ││
│    Captures at          Compares with                       ││
│    30 FPS              previous frame                       ││
│                                                              ││
└──────────────────────────────────────────────────────────────┘│
                                                                │
                         UDP Network (Port 9999)               │
                                                                │
┌──────────────────────────────────────────────────────────────┘
│
│  ┌─────────────────────────────────────────────────────────┐
│  │                   RECEIVER (Viewer)                      │
│  ├─────────────────────────────────────────────────────────┤
│  │                                                           │
│  │  ┌──────────────┐   ┌──────────────┐   ┌──────────────┐│
└─▶│UDP Receiver   │──▶│Diff Decoder  │──▶│Display       ││
   │ (Reassemble)  │   │(Apply patches│   │(OpenCV)      ││
   └──────────────┘   └──────────────┘   └──────────────┘│
                                                           │
   └─────────────────────────────────────────────────────────┘
```

## Data Flow

### 1. Screen Capture (ScreenCapture.cpp)
- Uses Windows GDI `BitBlt()` to capture screen
- Converts to OpenCV Mat (BGR format)
- Runs at 30 FPS (configurable)

### 2. Diff Encoding (DiffFrameEncoder.cpp)

#### Full/Key Frame Mode (Adaptive)
```
Input Frame (1920x1080x3) → JPEG Compress → PacketHeader + JPEG Data
```

Key frame triggers:
- Initial synchronization (startup)
- High-motion frame (too many changed blocks)
- Decoder mismatch request from receiver
- Network instability report from receiver

#### Diff Frame Mode
```
1. Divide frame into 32x32 blocks
2. For each block:
   - Calculate MAD vs previous frame
   - If MAD > threshold:
     ├─ Store block position (x, y)
     ├─ Store block size (w, h)
     └─ Store raw pixel data (w × h × 3 bytes)
3. Assemble: PacketHeader + DiffBlocks[]
```

### 3. UDP Transmission (broadcaster.cpp)

```
Encoded Data → Fragment into <65KB chunks → Add packet metadata → Send via UDP
```

Packet structure:
```
[Total Packets: 4B][Packet Index: 4B][Payload: variable]
```

### 4. UDP Reception (receiver.cpp)

```
Receive packets → Reassemble by index → Validate completeness → Decode
```

### 5. Frame Reconstruction (DiffFrameEncoder.cpp)

#### Full Frame
```
JPEG Data → cv::imdecode() → Display Frame
```

#### Diff Frame
```
For each DiffBlock:
    CurrentFrame[y:y+h, x:x+w] = BlockPixelData
Display updated CurrentFrame
```

## Key Algorithms

### Block Change Detection
```cpp
bool hasBlockChanged(frame1, frame2, x, y, w, h) {
    totalDiff = 0
    for each pixel in block:
        totalDiff += |R1-R2| + |G1-G2| + |B1-B2|
    
    avgDiff = totalDiff / (w × h × 3)
    return avgDiff > threshold
}
```

### Compression Ratio
```
CompressionRatio = DiffDataSize / OriginalFrameSize
```

Typical ratios:
- Static screen: 0.5-2%
- Active screen: 10-30%
- Full motion: 80-100% (switches to keyframe)

## Performance Optimizations

### 1. Block-Based Processing
- 32x32 blocks reduce comparison overhead
- Only 3,600 blocks for 1920x1080 (vs 2M pixels)

### 2. UDP vs TCP
- No handshake overhead
- No retransmission delays
- Acceptable packet loss for video

### 3. Adaptive Key Frames
- Triggered by high changed-block ratio
- Triggered by receiver decoder mismatch feedback
- Triggered by receiver network instability feedback
- Prevents drift and speeds up recovery after packet loss

### 4. Multi-Packet Fragmentation
- Splits large frames
- Stays within UDP MTU
- Receiver reassembles

## Code Structure

```
mirror-space/
├── include/
│   ├── ScreenCapture.h       # Windows screen capture API
│   └── DiffFrameEncoder.h    # Diff encoding protocol
├── src/
│   ├── ScreenCapture.cpp     # GDI implementation
│   ├── DiffFrameEncoder.cpp  # Diff algorithm + codec
│   ├── broadcaster.cpp       # Sender main + UDP broadcaster
│   └── receiver.cpp          # Receiver main + UDP receiver
├── CMakeLists.txt            # Build configuration
└── build.bat                 # Windows build script
```

## Network Protocol Specification

### Packet Types
```cpp
enum PacketType {
    FULL_FRAME = 0,  // Complete JPEG frame
    DIFF_FRAME = 1,  // Only changed blocks
    KEY_FRAME = 2    // Reserved for future use
}
```

### PacketHeader (Fixed 18 bytes)
```
Byte 0:     PacketType (1 byte)
Byte 1-4:   Frame Number (4 bytes, uint32)
Byte 5-8:   Width (4 bytes, uint32)
Byte 9-12:  Height (4 bytes, uint32)
Byte 13-16: Data Size (4 bytes, uint32)
Byte 17-18: Block Size (2 bytes, uint16)
```

### DiffBlock (Variable size)
```
Byte 0-1:   X Position (2 bytes, uint16)
Byte 2-3:   Y Position (2 bytes, uint16)
Byte 4-5:   Block Width (2 bytes, uint16)
Byte 6-7:   Block Height (2 bytes, uint16)
Byte 8+:    Pixel Data (width × height × 3 bytes, RGB)
```

## Bandwidth Analysis

### Example: 1920×1080 @ 30 FPS

**Full Frame:**
- Uncompressed: 1920 × 1080 × 3 = 6.2 MB
- JPEG (85% quality): ~500 KB
- Frequency: Every 60 frames (2 seconds)
- Bandwidth: ~2 Mbps

**Diff Frame (10% changed):**
- Changed blocks: 360 of 3,600
- Block data: 360 × 32 × 32 × 3 = 11 MB
- With overhead: ~12 MB
- Per frame: 400 KB
- Bandwidth: ~100 Mbps

**Average (typical browsing):**
- 5% change rate
- ~5-8 Mbps sustained
- Peaks at 15 Mbps during motion

## Future Optimizations

1. **Hardware Acceleration**
   - NVENC for H.264 encoding
   - Reduces CPU usage by 70%

2. **Prediction Encoding**
   - Motion vectors for moving blocks
   - Further 30-50% bandwidth reduction

3. **Adaptive Block Size**
   - Larger blocks for static areas
   - Smaller blocks for details

4. **Multi-threaded Encoding**
   - Parallel block processing
   - Increase FPS to 60+
