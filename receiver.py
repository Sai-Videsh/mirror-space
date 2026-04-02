"""
Mirror-Space Screen Receiver (Python)
Receives and displays screen broadcasts via UDP
"""

import sys
import time
import socket
import struct
import threading
import platform
import ipaddress
import queue
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, List

import cv2
import numpy as np
from zeroconf import ServiceBrowser, ServiceInfo, ServiceStateChange, Zeroconf

from diff_encoder import DiffFrameDecoder


DEFAULT_PORT = 9999
SERVICE_TYPE = "_mirror-space._udp.local."
DISCOVERY_BEACON_PORT = 10001
MAX_PACKET_SIZE = 65507
RECEIVE_TIMEOUT = 0.01  # seconds
FRAGMENT_HEADER_SIZE = 12  # total_packets (4) + packet_index (4) + frame_number (4)
DISCOVERY_INTERVAL_SECONDS = 1.0
REASSEMBLY_WINDOW_SECONDS = 0.18
SUBNET_SCAN_INTERVAL_SECONDS = 8.0
SUBNET_SCAN_BATCH_SIZE = 24


@dataclass
class StreamInfoRecord:
    """Tracks one discovered LAN stream."""

    service_name: str
    stream_name: str
    ip: str
    stream_port: int
    feedback_port: int
    last_seen: float


def _decode_property(raw: Dict[bytes, bytes], key: bytes, default: str = "") -> str:
    value = raw.get(key, default.encode("utf-8"))
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="ignore")
    return str(value)


def get_local_ipv4_addresses() -> set[str]:
    """Collect local IPv4 addresses used to detect same-machine sessions."""
    ips = {"127.0.0.1"}
    try:
        for result in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET, socket.SOCK_DGRAM):
            ips.add(result[4][0])
    except Exception:
        pass
    try:
        probe = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        probe.connect(("8.8.8.8", 80))
        ips.add(probe.getsockname()[0])
        probe.close()
    except Exception:
        pass
    return ips


class StreamDiscovery:
    """Discovers active Mirror-Space broadcasters over mDNS."""

    def __init__(self):
        self.zeroconf = Zeroconf()
        self.lock = threading.Lock()
        self.streams: Dict[str, StreamInfoRecord] = {}
        self.browser = ServiceBrowser(
            self.zeroconf,
            SERVICE_TYPE,
            handlers=[self._on_service_state_change],
        )

    def _on_service_state_change(
        self,
        zeroconf: Zeroconf,
        service_type: str,
        name: str,
        state_change: ServiceStateChange,
    ):
        if state_change == ServiceStateChange.Removed:
            with self.lock:
                self.streams.pop(name, None)
            return

        info = zeroconf.get_service_info(service_type, name, timeout=250)
        if info is None:
            return

        parsed = self._parse_service_info(name, info)
        if parsed is None:
            return

        with self.lock:
            self.streams[name] = parsed

    def _parse_service_info(self, name: str, info: ServiceInfo) -> Optional[StreamInfoRecord]:
        addresses = info.parsed_addresses(version=socket.AF_INET)
        if not addresses:
            return None

        props = info.properties or {}
        stream_name = _decode_property(props, b"stream_name", default=name.split(".")[0])
        stream_port_str = _decode_property(props, b"stream_port", default=str(info.port))
        feedback_port_str = _decode_property(props, b"feedback_port", default=str(info.port + 1))

        try:
            stream_port = int(stream_port_str)
        except ValueError:
            stream_port = info.port

        try:
            feedback_port = int(feedback_port_str)
        except ValueError:
            feedback_port = stream_port + 1

        return StreamInfoRecord(
            service_name=name,
            stream_name=stream_name,
            ip=addresses[0],
            stream_port=stream_port,
            feedback_port=feedback_port,
            last_seen=time.time(),
        )

    def get_streams(self, target_port: int) -> List[StreamInfoRecord]:
        with self.lock:
            items = [
                stream
                for stream in self.streams.values()
                if stream.stream_port == target_port
            ]
        return sorted(items, key=lambda item: item.stream_name.lower())

    def close(self):
        self.zeroconf.close()


class UdpStreamDiscovery:
    """Fallback LAN discovery using UDP broadcast query/response."""

    def __init__(self, feedback_port: int):
        self.feedback_port = feedback_port
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind(("0.0.0.0", 0))
        self.sock.setblocking(False)
        self.last_query_time = 0.0
        self.streams: Dict[str, StreamInfoRecord] = {}

    def _send_query_if_needed(self):
        now = time.time()
        if now - self.last_query_time < DISCOVERY_INTERVAL_SECONDS:
            return

        try:
            self.sock.sendto(b"DISCOVERY_QUERY", ("255.255.255.255", self.feedback_port))
            self.last_query_time = now
        except Exception:
            pass

    def poll(self, stream_port: int):
        self._send_query_if_needed()

        for _ in range(32):
            try:
                data, addr = self.sock.recvfrom(1024)
            except BlockingIOError:
                break
            except Exception:
                break

            text = data.decode("utf-8", errors="ignore").strip()
            if not text.startswith("DISCOVERY_RESPONSE"):
                continue

            tokens = text.split()
            kv: Dict[str, str] = {}
            for token in tokens[1:]:
                if "=" not in token:
                    continue
                key, value = token.split("=", 1)
                kv[key] = value

            try:
                parsed_stream_port = int(kv.get("stream_port", str(stream_port)))
            except ValueError:
                parsed_stream_port = stream_port

            if parsed_stream_port != stream_port:
                continue

            try:
                feedback_port = int(kv.get("feedback_port", str(stream_port + 1)))
            except ValueError:
                feedback_port = stream_port + 1

            stream_name = kv.get("stream_name", addr[0])
            self.streams[addr[0]] = StreamInfoRecord(
                service_name=f"udp-{addr[0]}",
                stream_name=stream_name,
                ip=addr[0],
                stream_port=parsed_stream_port,
                feedback_port=feedback_port,
                last_seen=time.time(),
            )

    def get_streams(self, stream_port: int) -> List[StreamInfoRecord]:
        now = time.time()
        items = [
            stream
            for stream in self.streams.values()
            if stream.stream_port == stream_port and (now - stream.last_seen) <= 5.0
        ]
        return sorted(items, key=lambda item: item.stream_name.lower())

    def close(self):
        self.sock.close()


class UdpBeaconDiscovery:
    """Passive LAN discovery by listening for broadcaster beacons."""

    def __init__(self):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind(("0.0.0.0", DISCOVERY_BEACON_PORT))
        self.sock.setblocking(False)
        self.streams: Dict[str, StreamInfoRecord] = {}

    def poll(self, stream_port: int):
        for _ in range(32):
            try:
                data, addr = self.sock.recvfrom(1024)
            except BlockingIOError:
                break
            except Exception:
                break

            text = data.decode("utf-8", errors="ignore").strip()
            if not text.startswith("STREAM_ANNOUNCE"):
                continue

            kv: Dict[str, str] = {}
            for token in text.split()[1:]:
                if "=" not in token:
                    continue
                key, value = token.split("=", 1)
                kv[key] = value

            try:
                parsed_stream_port = int(kv.get("stream_port", str(stream_port)))
            except ValueError:
                parsed_stream_port = stream_port

            if parsed_stream_port != stream_port:
                continue

            try:
                feedback_port = int(kv.get("feedback_port", str(stream_port + 1)))
            except ValueError:
                feedback_port = stream_port + 1

            stream_name = kv.get("stream_name", addr[0])
            self.streams[addr[0]] = StreamInfoRecord(
                service_name=f"beacon-{addr[0]}",
                stream_name=stream_name,
                ip=addr[0],
                stream_port=parsed_stream_port,
                feedback_port=feedback_port,
                last_seen=time.time(),
            )

    def get_streams(self, stream_port: int) -> List[StreamInfoRecord]:
        now = time.time()
        items = [
            stream
            for stream in self.streams.values()
            if stream.stream_port == stream_port and (now - stream.last_seen) <= 5.0
        ]
        return sorted(items, key=lambda item: item.stream_name.lower())

    def close(self):
        self.sock.close()


class UdpSubnetDiscovery:
    """Active /24 subnet scan for broadcasters that reply to DISCOVERY_QUERY."""

    def __init__(self, feedback_port: int):
        self.feedback_port = feedback_port
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind(("0.0.0.0", 0))
        self.sock.setblocking(False)
        self.streams: Dict[str, StreamInfoRecord] = {}
        self._scan_hosts: List[str] = []
        self._scan_index = 0
        self._last_scan_time = 0.0
        self.subnet_label = "unknown"
        self._init_scan_targets()

    def _init_scan_targets(self):
        local_ip = "127.0.0.1"
        try:
            probe = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            probe.connect(("8.8.8.8", 80))
            local_ip = probe.getsockname()[0]
            probe.close()
        except Exception:
            pass

        network = ipaddress.ip_network(f"{local_ip}/24", strict=False)
        self.subnet_label = str(network)
        self._scan_hosts = [str(host) for host in network.hosts()]

    def _start_scan_if_needed(self):
        now = time.time()
        if self._scan_index < len(self._scan_hosts):
            return
        if now - self._last_scan_time < SUBNET_SCAN_INTERVAL_SECONDS:
            return
        self._scan_index = 0
        self._last_scan_time = now

    def _probe_hosts(self):
        if self._scan_index >= len(self._scan_hosts):
            return

        end = min(self._scan_index + SUBNET_SCAN_BATCH_SIZE, len(self._scan_hosts))
        for i in range(self._scan_index, end):
            host = self._scan_hosts[i]
            try:
                self.sock.sendto(b"DISCOVERY_QUERY", (host, self.feedback_port))
            except Exception:
                pass
        self._scan_index = end

    def _consume_responses(self, stream_port: int):
        for _ in range(64):
            try:
                data, addr = self.sock.recvfrom(1024)
            except BlockingIOError:
                break
            except Exception:
                break

            text = data.decode("utf-8", errors="ignore").strip()
            if not text.startswith("DISCOVERY_RESPONSE"):
                continue

            kv: Dict[str, str] = {}
            for token in text.split()[1:]:
                if "=" not in token:
                    continue
                key, value = token.split("=", 1)
                kv[key] = value

            try:
                parsed_stream_port = int(kv.get("stream_port", str(stream_port)))
            except ValueError:
                parsed_stream_port = stream_port

            if parsed_stream_port != stream_port:
                continue

            try:
                feedback_port = int(kv.get("feedback_port", str(stream_port + 1)))
            except ValueError:
                feedback_port = stream_port + 1

            stream_name = kv.get("stream_name", addr[0])
            self.streams[addr[0]] = StreamInfoRecord(
                service_name=f"subnet-{addr[0]}",
                stream_name=stream_name,
                ip=addr[0],
                stream_port=parsed_stream_port,
                feedback_port=feedback_port,
                last_seen=time.time(),
            )

    def poll(self, stream_port: int):
        self._start_scan_if_needed()
        self._probe_hosts()
        self._consume_responses(stream_port)

    def get_streams(self, stream_port: int) -> List[StreamInfoRecord]:
        now = time.time()
        items = [
            stream
            for stream in self.streams.values()
            if stream.stream_port == stream_port and (now - stream.last_seen) <= 12.0
        ]
        return sorted(items, key=lambda item: item.stream_name.lower())

    def close(self):
        self.sock.close()


class UDPReceiver:
    """Handles UDP packet reception and reassembly"""
    
    def __init__(self, port: int):
        self.port = port
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            self.sock.setsockopt(socket.IPPROTO_IP, socket.IP_TOS, 0x10)  # IPTOS_LOWDELAY
        except OSError:
            pass
        
        # Bind to all interfaces
        self.sock.bind(('0.0.0.0', port))
        
        # Set receive timeout
        self.sock.settimeout(RECEIVE_TIMEOUT)
        
        # Keep receive buffer moderate to avoid deep queue latency.
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 256 * 1024)

        # Packet health statistics
        self.complete_frames = 0
        self.partial_frames = 0
        self.missing_packets = 0
        
        print(f"UDP receiver initialized on port {port}")
    
    def receive_data(
        self,
        expected_source_ip: Optional[str] = None,
    ) -> Tuple[Optional[bytes], Optional[Dict], Optional[Tuple[str, int]]]:
        """Receive and reassemble fragmented data"""
        packets: Dict[int, bytes] = {}
        total_packets = 0
        active_frame_number: Optional[int] = None
        start_time = time.time()
        source_addr: Optional[Tuple[str, int]] = None
        partial_due_to_timeout = False
        
        while True:
            elapsed = time.time() - start_time

            # Keep loop responsive even when no packets are arriving.
            if elapsed > REASSEMBLY_WINDOW_SECONDS and not packets:
                return None, None, source_addr

            # Check timeout
            if elapsed > REASSEMBLY_WINDOW_SECONDS and packets:
                partial_due_to_timeout = True
                break  # Partial frame, return what we have
            
            try:
                data, addr = self.sock.recvfrom(MAX_PACKET_SIZE)
            except socket.timeout:
                if (time.time() - start_time) > REASSEMBLY_WINDOW_SECONDS:
                    if not packets:
                        return None, None, source_addr
                    partial_due_to_timeout = True
                    break
                continue
            except Exception as e:
                print(f"Receive error: {e}")
                return None, None, None
            
            if len(data) < FRAGMENT_HEADER_SIZE:
                continue  # Too small for header

            if expected_source_ip is not None and addr[0] != expected_source_ip:
                continue

            if source_addr is None:
                source_addr = addr
            elif addr[0] != source_addr[0] or addr[1] != source_addr[1]:
                # Ignore packets from other broadcasters while assembling this frame.
                continue
            
            # Parse packet header
            total, index, frame_number = struct.unpack('<III', data[:FRAGMENT_HEADER_SIZE])
            payload = data[FRAGMENT_HEADER_SIZE:]

            if total <= 0 or index >= total:
                continue

            # Start new reassembly only when packet 0 arrives.
            if active_frame_number is None:
                if index != 0:
                    continue
                active_frame_number = frame_number
                total_packets = total
            elif frame_number != active_frame_number:
                continue
            elif total != total_packets:
                continue
            
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
                missing_count += 1

        is_complete = missing_count == 0 and not partial_due_to_timeout

        if is_complete:
            self.complete_frames += 1
        else:
            self.partial_frames += 1
            self.missing_packets += missing_count

        meta = {
            "complete": is_complete,
            "missing_packets": missing_count,
            "total_packets": total_packets,
            "timed_out": partial_due_to_timeout,
            "frame_number": active_frame_number,
        }

        # Never decode partial payloads; wait for a complete frame to keep decoder state valid.
        if not is_complete:
            return None, meta, source_addr

        return (bytes(result) if result else None), meta, source_addr
    
    def close(self):
        """Close the socket"""
        self.sock.close()


class FeedbackSender:
    """Sends health and mismatch feedback to broadcaster"""

    def __init__(self):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
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


class TerminalSelectionReader:
    """Reads stream selection from terminal input (e.g., type 1 + Enter)."""

    def __init__(self):
        self.queue: "queue.Queue[str]" = queue.Queue()
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)

    def start(self):
        self.thread.start()

    def _run(self):
        while not self.stop_event.is_set():
            try:
                text = input().strip()
            except EOFError:
                return
            except Exception:
                continue

            if not text:
                continue

            self.queue.put(text)

    def poll_input(self) -> Optional[str]:
        try:
            return self.queue.get_nowait()
        except queue.Empty:
            return None

    def stop(self):
        self.stop_event.set()


def main():
    port = DEFAULT_PORT
    bootstrap_ip: Optional[str] = None
    
    if len(sys.argv) > 1:
        try:
            port = int(sys.argv[1])
        except ValueError:
            bootstrap_ip = sys.argv[1]
    if len(sys.argv) > 2:
        bootstrap_ip = sys.argv[2]
    
    print("=== Mirror-Space Screen Receiver (Python) ===")
    print(f"Listening on port: {port}")
    if bootstrap_ip:
        print(f"Bootstrap broadcaster IP: {bootstrap_ip}")
    print("Press ESC or 'q' to quit...\n")
    
    # Initialize receiver
    receiver = None
    feedback = None
    discovery = None
    udp_discovery = None
    beacon_discovery = None
    subnet_discovery = None
    terminal_selector = None
    try:
        receiver = UDPReceiver(port)
        decoder = DiffFrameDecoder()
        feedback = FeedbackSender()
        terminal_selector = TerminalSelectionReader()
        terminal_selector.start()
        discovery = StreamDiscovery()
        udp_discovery = UdpStreamDiscovery(feedback_port=port + 1)
        beacon_discovery = UdpBeaconDiscovery()
        subnet_discovery = UdpSubnetDiscovery(feedback_port=port + 1)
        
        # Create display window
        cv2.namedWindow("Mirror-Space Receiver", cv2.WINDOW_NORMAL)
        
        # Receiving loop
        frames_received = 0
        last_measured_receive_fps = 0.0
        fps_start_time = time.time()
        health_window_start = time.time()
        window_partial_frames = 0
        window_complete_frames = 0
        window_meta_frames = 0
        window_timed_out_frames = 0
        window_missing_packets = 0
        window_total_packets = 0
        sender_ip: Optional[str] = None
        selected_stream_ip: Optional[str] = None
        selected_stream_name: Optional[str] = None
        selected_stream_feedback_port: Optional[int] = None
        selected_access_id: Optional[str] = None
        connected = False
        local_name = platform.node() or socket.gethostname()
        local_ips = get_local_ipv4_addresses()
        local_mode_logged = False
        discovery_print_time = 0.0
        last_discovery_signature = ""
        last_hello_log_time = 0.0
        latest_streams: List[StreamInfoRecord] = []
        waiting_for_manual_selection_logged = False
        waiting_for_access_id_logged = False
        
        print("Waiting for frames...\n")
        
        while True:
            # Refresh stream list UI periodically.
            now = time.time()
            udp_discovery.poll(stream_port=port)
            beacon_discovery.poll(stream_port=port)
            subnet_discovery.poll(stream_port=port)
            mdns_streams = discovery.get_streams(port)
            udp_streams = udp_discovery.get_streams(stream_port=port)
            beacon_streams = beacon_discovery.get_streams(stream_port=port)
            subnet_streams = subnet_discovery.get_streams(stream_port=port)

            streams_by_ip: Dict[str, StreamInfoRecord] = {item.ip: item for item in mdns_streams}
            for item in udp_streams:
                if item.ip not in streams_by_ip:
                    streams_by_ip[item.ip] = item
            for item in beacon_streams:
                if item.ip not in streams_by_ip:
                    streams_by_ip[item.ip] = item
            for item in subnet_streams:
                if item.ip not in streams_by_ip:
                    streams_by_ip[item.ip] = item
            streams = sorted(streams_by_ip.values(), key=lambda item: item.stream_name.lower())
            latest_streams = streams
            if selected_stream_ip is not None and all(item.ip != selected_stream_ip for item in streams) and not bootstrap_ip:
                selected_stream_ip = None
                selected_stream_name = None
                selected_stream_feedback_port = None
                selected_access_id = None
                connected = False
            if selected_stream_ip is None and bootstrap_ip:
                selected_stream_ip = bootstrap_ip
                selected_stream_name = f"Manual-{bootstrap_ip}"
                selected_stream_feedback_port = port + 1
                selected_access_id = None
                print(
                    "Connection Debug: using bootstrap target "
                    f"{selected_stream_name} ({selected_stream_ip}:{selected_stream_feedback_port})"
                )
            if selected_stream_ip is None and streams and not waiting_for_manual_selection_logged and not bootstrap_ip:
                print("Connection Debug: waiting for manual stream selection (press 1-9)")
                waiting_for_manual_selection_logged = True

            if now - discovery_print_time >= 2.0:
                signature_parts = [f"{item.stream_name}:{item.ip}" for item in streams]
                signature = "|".join(signature_parts) + f"|selected={selected_stream_ip}"
                if signature != last_discovery_signature:
                    print(
                        "Discovery Debug: "
                        f"mdns={len(mdns_streams)} "
                        f"udp_query={len(udp_streams)} "
                        f"udp_beacon={len(beacon_streams)} "
                        f"subnet_scan={len(subnet_streams)}"
                    )
                    print(f"Available Streams in subnet {subnet_discovery.subnet_label}:")
                    if streams:
                        print("Press number 1-9 in receiver window OR type number + Enter in terminal")
                        print("After selecting, type broadcaster Session Access ID and press Enter")
                        for idx, item in enumerate(streams, start=1):
                            marker = "*" if item.ip == selected_stream_ip else "-"
                            if idx <= 9:
                                print(f"{marker} [{idx}] {item.stream_name} ({item.ip})")
                            else:
                                print(f"{marker} [ ] {item.stream_name} ({item.ip})")
                    else:
                        print("- none")
                    last_discovery_signature = signature
                discovery_print_time = now

            if selected_stream_ip and selected_stream_feedback_port and selected_access_id:
                feedback.send(
                    selected_stream_ip,
                    selected_stream_feedback_port,
                    f"RECEIVER_HELLO receiver={local_name} access_id={selected_access_id}",
                    throttle_seconds=1.0,
                    throttle_key="RECEIVER_HELLO",
                )
                # Fallback path: some LAN setups pass broadcast but drop peer-to-peer unicast.
                feedback.send(
                    "255.255.255.255",
                    selected_stream_feedback_port,
                    f"RECEIVER_HELLO receiver={local_name} access_id={selected_access_id}",
                    throttle_seconds=1.0,
                    throttle_key="RECEIVER_HELLO_BROADCAST",
                )
                if (not connected) and (time.time() - last_hello_log_time >= 2.0):
                    print(
                        "Connection Debug: sent RECEIVER_HELLO to "
                        f"{selected_stream_name} ({selected_stream_ip}:{selected_stream_feedback_port})"
                    )
                    last_hello_log_time = time.time()
            elif selected_stream_ip and selected_stream_feedback_port and not selected_access_id:
                if not waiting_for_access_id_logged:
                    print("Connection Debug: waiting for Session Access ID in terminal")
                    waiting_for_access_id_logged = True

            # Receive data
            expected_source_ip = selected_stream_ip
            if selected_stream_ip in local_ips:
                expected_source_ip = None
                if not local_mode_logged:
                    print("Connection Debug: local receiver mode enabled (accepting loopback stream source)")
                    local_mode_logged = True
            received_data, meta, source_addr = receiver.receive_data(expected_source_ip=expected_source_ip)

            if source_addr is not None:
                sender_ip = source_addr[0]
                if not connected and sender_ip == selected_stream_ip:
                    connected = True
                    print(f"Connected to stream: {selected_stream_name} ({sender_ip})")
            
            if received_data:
                # Decode frame
                frame = decoder.decode(received_data)

                decode_error = decoder.consume_decoder_error()
                if decode_error and sender_ip:
                    feedback.send(
                        sender_ip,
                        selected_stream_feedback_port or (port + 1),
                        f"KEYFRAME_REQUEST reason={decode_error}",
                        throttle_seconds=0.2,
                        throttle_key="KEYFRAME_REQUEST",
                    )
                
                if frame is not None:
                    # Display frame
                    cv2.imshow("Mirror-Space Receiver", frame)
                    frames_received += 1

            if meta:
                window_meta_frames += 1
                window_total_packets += meta.get("total_packets", 0)
                window_missing_packets += meta.get("missing_packets", 0)
                if meta.get("complete", True):
                    window_complete_frames += 1
                else:
                    window_partial_frames += 1
                    if meta.get("timed_out", False):
                        window_timed_out_frames += 1

            # Periodically report network instability to force sender key frames.
            health_elapsed = time.time() - health_window_start
            if health_elapsed >= 1.0:
                packet_loss_ratio = (
                    window_missing_packets / window_total_packets if window_total_packets > 0 else 0.0
                )
                partial_ratio = (
                    window_partial_frames / window_meta_frames if window_meta_frames > 0 else 0.0
                )
                timeout_ratio = (
                    window_timed_out_frames / window_meta_frames if window_meta_frames > 0 else 0.0
                )
                unstable = window_partial_frames >= 2 or packet_loss_ratio >= 0.05

                if sender_ip:
                    feedback.send(
                        sender_ip,
                        selected_stream_feedback_port or (port + 1),
                        (
                            "STREAM_STATS "
                            f"recv_fps={last_measured_receive_fps:.2f} "
                            f"packet_loss={packet_loss_ratio:.6f} "
                            f"partial_ratio={partial_ratio:.6f} "
                            f"timeout_ratio={timeout_ratio:.6f} "
                            f"complete_frames={window_complete_frames} "
                            f"partial_frames={window_partial_frames}"
                        ),
                        throttle_seconds=0.2,
                        throttle_key="STREAM_STATS",
                    )

                if unstable and sender_ip:
                    feedback.send(
                        sender_ip,
                        selected_stream_feedback_port or (port + 1),
                        (
                            "NETWORK_UNSTABLE "
                            f"partial_frames={window_partial_frames} "
                            f"packet_loss={packet_loss_ratio:.1%}"
                        ),
                        throttle_seconds=0.5,
                        throttle_key="NETWORK_UNSTABLE",
                    )

                window_partial_frames = 0
                window_complete_frames = 0
                window_meta_frames = 0
                window_timed_out_frames = 0
                window_missing_packets = 0
                window_total_packets = 0
                health_window_start = time.time()
            
            # Calculate FPS
            elapsed = time.time() - fps_start_time
            if elapsed >= 1.0:
                actual_fps = frames_received / elapsed
                last_measured_receive_fps = actual_fps
                if connected or frames_received > 0:
                    print(f"Receive FPS: {actual_fps:.1f}")
                frames_received = 0
                fps_start_time = time.time()
            
            # Check for exit (ESC or 'q')
            key = cv2.waitKey(1) & 0xFF
            if key == 27 or key == ord('q'):  # ESC or 'q'
                break
            if ord('1') <= key <= ord('9'):
                index = key - ord('1')
                if 0 <= index < len(latest_streams):
                    chosen = latest_streams[index]
                    selected_stream_ip = chosen.ip
                    selected_stream_name = chosen.stream_name
                    selected_stream_feedback_port = chosen.feedback_port
                    selected_access_id = None
                    connected = False
                    local_mode_logged = False
                    waiting_for_manual_selection_logged = False
                    waiting_for_access_id_logged = False
                    print(
                        "Connection Debug: manually selected "
                        f"{selected_stream_name} ({selected_stream_ip}:{selected_stream_feedback_port})"
                    )
                    print("Connection Debug: enter Session Access ID in terminal and press Enter")

            # Terminal fallback selection (type 1..9 + Enter)
            user_text = terminal_selector.poll_input() if terminal_selector is not None else None
            if user_text is not None:
                normalized = user_text.strip()
                if normalized.lower() in {"q", "quit", "exit"}:
                    break
                elif normalized.isdigit() and 1 <= int(normalized) <= 9:
                    selected_index = int(normalized) - 1
                    if 0 <= selected_index < len(latest_streams):
                        chosen = latest_streams[selected_index]
                        selected_stream_ip = chosen.ip
                        selected_stream_name = chosen.stream_name
                        selected_stream_feedback_port = chosen.feedback_port
                        selected_access_id = None
                        connected = False
                        local_mode_logged = False
                        waiting_for_manual_selection_logged = False
                        waiting_for_access_id_logged = False
                        print(
                            "Connection Debug: manually selected (terminal) "
                            f"{selected_stream_name} ({selected_stream_ip}:{selected_stream_feedback_port})"
                        )
                        print("Connection Debug: enter Session Access ID in terminal and press Enter")
                elif selected_stream_ip and selected_stream_feedback_port:
                    selected_access_id = normalized.upper()
                    connected = False
                    waiting_for_access_id_logged = False
                    print("Connection Debug: Session Access ID captured, attempting handshake")
    
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
        if discovery is not None:
            discovery.close()
        if udp_discovery is not None:
            udp_discovery.close()
        if beacon_discovery is not None:
            beacon_discovery.close()
        if subnet_discovery is not None:
            subnet_discovery.close()
        if terminal_selector is not None:
            terminal_selector.stop()
        cv2.destroyAllWindows()
        print("Receiver stopped.")


if __name__ == "__main__":
    main()
