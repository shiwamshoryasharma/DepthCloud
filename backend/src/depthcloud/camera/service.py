import logging
import multiprocessing as mp
import threading
import time
import uuid
from collections import deque
from queue import Empty

import cv2
import psutil

from depthcloud.config import Settings
from depthcloud.schemas import BufferSettings, CameraStart
from depthcloud.vision.transforms import transform_preview
from depthcloud.workflow.schemas import PreviewOptions

from .buffer import FrameBuffer, FramePacket, recommend_capacity
from .discovery import discover_cameras
from .worker import capture_main

logger = logging.getLogger("depthcloud.camera")


class CameraBusy(RuntimeError):
    pass


class CameraService:
    def __init__(self, settings: Settings, worker_target=capture_main):
        self.settings = settings
        self.worker_target = worker_target
        self._lock = threading.RLock()
        self._operation = threading.RLock()
        self._stop = threading.Event()
        self._process = None
        self._output = None
        self._process_stop = None
        self._monitor = None
        self._preview_thread = None
        budget = min(settings.buffer_memory_mb * 1024**2, int(psutil.virtual_memory().available * 0.1))
        self.buffer = FrameBuffer(min(settings.buffer_capacity, settings.maximum_buffer_size), max(1, budget))
        self.automatic = False
        self.preview_options = PreviewOptions()
        self.preview_revision = 0
        self.state = "idle"
        self.error = None
        self.generation = uuid.uuid4().hex
        self.request = None
        self.metadata = {}
        self._preview = None
        self._capture_times = deque(maxlen=120)
        self._preview_times = deque(maxlen=120)
        self._received = 0
        self._ipc_dropped = 0
        self._preview_skipped = 0
        self._last_seq = 0
        self._latency_ms = 0.0
        self._preview_latency_ms = 0.0
        self._adaptive_at = 0.0

    def start(self, request: CameraStart):
        with self._operation:
            with self._lock:
                if self.state in ("starting", "running", "discovering", "stopping"):
                    raise CameraBusy("Stop the active camera or finish discovery before starting another.")
            self.stop()
            if request.width * request.height * 3 > self.buffer.max_bytes:
                raise ValueError("Requested resolution exceeds the camera buffer memory budget.")
            ctx = mp.get_context("spawn")
            self._output = ctx.Queue(maxsize=2)
            self._process_stop = ctx.Event()
            self._stop = threading.Event()
            with self._lock:
                self.generation = uuid.uuid4().hex
                self.state = "starting"
                self.error = None
                self.request = request
                self.metadata = {}
            self._process = ctx.Process(
                target=self.worker_target,
                args=(request.model_dump(), self._output, self._process_stop),
                daemon=True,
                name="depthcloud-camera",
            )
            try:
                self._process.start()
            except Exception:
                self.stop()
                raise
            self._monitor = threading.Thread(target=self._monitor_frames, daemon=True, name="camera-supervisor")
            self._preview_thread = threading.Thread(target=self._encode_preview, daemon=True, name="camera-preview")
            self._monitor.start()
            self._preview_thread.start()
            logger.info("Camera starting: index=%s requested=%sx%s@%s", request.index, request.width, request.height, request.fps)
            return self.status()

    def _clear_transient(self):
        self.buffer.clear()
        self._preview = None
        self._capture_times.clear()
        self._preview_times.clear()
        self._received = self._ipc_dropped = self._preview_skipped = self._last_seq = 0
        self._latency_ms = self._preview_latency_ms = 0.0
        self.metadata = {}

    def stop(self):
        with self._operation:
            with self._lock:
                if self._process is not None:
                    self.state = "stopping"
                self._stop.set()
                if self._process_stop is not None:
                    self._process_stop.set()
            process = self._process
            if process is not None and process.pid is not None:
                process.join(self.settings.worker_join_timeout)
                if process.is_alive():
                    logger.warning("Camera driver did not stop; terminating isolated capture process")
                    process.terminate()
                    process.join(2)
                if process.is_alive():
                    raise RuntimeError("Camera process could not be terminated.")
            for thread in (self._monitor, self._preview_thread):
                if thread is not None and thread is not threading.current_thread():
                    thread.join(2)
            if self._output is not None:
                self._output.close()
            if process is not None and process.pid is not None:
                process.close()
            self._process = self._output = self._process_stop = None
            self._monitor = self._preview_thread = None
            with self._lock:
                self._clear_transient()
                self.generation = uuid.uuid4().hex
                self.state = "idle"
                self.error = None
            logger.info("Camera stopped; transient buffer and preview flushed")
            return self.status()

    def _fail(self, message):
        with self._lock:
            if self._stop.is_set():
                return
            self._stop.set()
            self._process_stop.set()
            self.state = "error"
            self.error = message
            self._clear_transient()
        logger.error("Camera failure: %s", message)
        if self._process.is_alive():
            self._process.terminate()

    def _monitor_frames(self):
        last_frame_at = time.monotonic()
        try:
            while not self._stop.is_set():
                try:
                    message = self._output.get(timeout=0.1)
                except Empty:
                    if not self._process.is_alive():
                        self._fail("Camera worker exited unexpectedly.")
                        return
                    limit = self.settings.startup_timeout if self.state == "starting" else self.settings.camera_timeout
                    if time.monotonic() - last_frame_at > limit:
                        self._fail("Camera timed out while opening or reading. Try another index/backend, or close other camera apps.")
                        return
                    continue
                if message["type"] == "error":
                    self._fail(message["message"])
                    return
                if message["type"] != "frame":
                    continue
                packet = FramePacket(
                    message["sequence"], message["timestamp"], message["monotonic"], message["bgr"]
                )
                last_frame_at = time.monotonic()
                with self._lock:
                    if self._stop.is_set():
                        return
                    packet.bgr.setflags(write=False)
                    self.buffer.push(packet)
                    self._received += 1
                    self._ipc_dropped += max(0, packet.sequence_id - self._last_seq - 1)
                    self._last_seq = packet.sequence_id
                    self._capture_times.append((packet.captured_monotonic, packet.sequence_id))
                    self.metadata = {
                        **message.get("metadata", {}),
                        "width": int(packet.bgr.shape[1]), "height": int(packet.bgr.shape[0]),
                    }
                    self.state = "running"
        except Exception as exc:
            logger.exception("Capture supervisor failed")
            self._fail(f"Capture supervisor failed: {exc}")

    def _encode_preview(self):
        seen = 0
        next_encode = time.monotonic()
        try:
            while not self._stop.is_set():
                if self._stop.wait(max(0, next_encode - time.monotonic())):
                    return
                next_encode = time.monotonic() + 1 / self.settings.preview_fps
                with self._lock:
                    packet = self.buffer.latest()
                    count = self._received
                    generation = self.generation
                    preview_options = self.preview_options
                    preview_revision = self.preview_revision
                if packet is None or packet.sequence_id == seen:
                    continue
                began = time.perf_counter()
                image = packet.bgr
                if image.shape[1] > self.settings.preview_width:
                    scale = self.settings.preview_width / image.shape[1]
                    image = cv2.resize(image, (self.settings.preview_width, max(1, round(image.shape[0] * scale))))
                image = transform_preview(image, preview_options)
                ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, self.settings.jpeg_quality])
                if not ok:
                    raise RuntimeError("JPEG encoding failed.")
                now = time.monotonic()
                with self._lock:
                    if self._stop.is_set() or generation != self.generation:
                        return
                    if preview_revision != self.preview_revision:
                        continue
                    previous_count = self._preview[3] if self._preview is not None else 0
                    self._preview_skipped += max(0, count - previous_count - 1)
                    self._preview = (f"{generation}:{preview_revision}:{packet.sequence_id}", encoded.tobytes(), {
                        "sequence_id": packet.sequence_id, "timestamp": packet.timestamp,
                        "width": int(image.shape[1]), "height": int(image.shape[0]),
                    }, count)
                    self._preview_times.append(now)
                    elapsed = (time.perf_counter() - began) * 1000
                    self._latency_ms = elapsed if not self._latency_ms else 0.8 * self._latency_ms + 0.2 * elapsed
                    self._preview_latency_ms = max(0, (now - packet.captured_monotonic) * 1000)
                    if self.automatic and now - self._adaptive_at >= 2:
                        fps = self._capture_fps()
                        available = min(self.buffer.max_bytes, int(psutil.virtual_memory().available * 0.1))
                        capacity = recommend_capacity(
                            fps, self._preview_latency_ms, packet.nbytes, available,
                            self.settings.maximum_buffer_size,
                        )
                        self.buffer.resize(capacity)
                        self._adaptive_at = now
                seen = packet.sequence_id
        except Exception as exc:
            logger.exception("Preview worker failed")
            self._fail(f"Preview worker failed: {exc}")

    def latest_preview(self):
        with self._lock:
            return self._preview

    def _capture_fps(self):
        times = self._capture_times
        if len(times) < 2 or times[-1][0] <= times[0][0]:
            return 0.0
        return (times[-1][1] - times[0][1]) / (times[-1][0] - times[0][0])

    def configure_buffer(self, request: BufferSettings):
        with self._lock:
            self.automatic = request.automatic
            self.buffer.resize(min(request.capacity, self.settings.maximum_buffer_size))
            self._adaptive_at = 0.0
        return self.status()

    def configure_preview(self, options):
        with self._lock:
            self.preview_options = options
            self.preview_revision += 1
        return self.status()

    def discover(self, backend="auto"):
        with self._operation:
            with self._lock:
                if self.state in ("starting", "running", "stopping"):
                    raise CameraBusy("Stop the camera before discovering devices.")
                self.state = "discovering"
                self.error = None
            try:
                return discover_cameras(self.settings.discovery_max_index, self.settings.discovery_timeout, backend)
            finally:
                with self._lock:
                    self.state = "idle"

    def latest_for_scan(self):
        with self._lock:
            packet = self.buffer.latest()
            if self.state != "running" or packet is None:
                raise ValueError("Camera stopped or disconnected. Generation stopped; surface retained.")
            return packet, self.generation

    def status(self):
        with self._lock:
            stats = self.buffer.stats()
            times = self._preview_times
            processing_fps = (len(times) - 1) / (times[-1] - times[0]) if len(times) > 1 and times[-1] > times[0] else 0.0
            return {
                "state": self.state, "error": self.error, "generation": self.generation,
                "camera_index": self.request.index if self.request else None,
                "requested": self.request.model_dump() if self.request else None,
                "actual": self.metadata,
                "current_buffer_size": stats["capacity"],
                "maximum_buffer_size": self.settings.maximum_buffer_size,
                "buffer_fill": stats["fill"], "buffer_bytes": stats["bytes"],
                "buffer_memory_limit": self.buffer.max_bytes,
                "buffer_evictions": stats["evicted"],
                "buffer_sequence_ids": stats["sequence_ids"],
                "automatic_buffer": self.automatic,
                "preview_options": self.preview_options.model_dump(),
                "frames_received": self._received,
                "frames_dropped": self._ipc_dropped + self._preview_skipped,
                "capture_queue_drops": self._ipc_dropped,
                "preview_skips": self._preview_skipped,
                "capture_fps": round(self._capture_fps(), 2),
                "processing_fps": round(processing_fps, 2),
                "processing_latency_ms": round(self._latency_ms, 2),
                "preview_latency_ms": round(self._preview_latency_ms, 2),
                "latest_sequence": self._last_seq if stats["fill"] else None,
            }




