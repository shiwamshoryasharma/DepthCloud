"""Only these child processes own OpenCV camera handles.

A blocked USB driver can be terminated without wedging the API/event loop.
No model imports or disk frame recording occur here.
"""
import os
import time
from queue import Empty, Full

import cv2


def backend_id(name):
    if name == "dshow" or (name == "auto" and os.name == "nt"):
        return cv2.CAP_DSHOW
    if name == "msmf":
        return cv2.CAP_MSMF
    return cv2.CAP_ANY


def put_latest(output, message):
    # Queue has maxsize=2. A congested consumer never blocks capture.
    for _ in range(3):
        try:
            output.put_nowait(message)
            return
        except Full:
            try:
                output.get_nowait()
            except Empty:
                time.sleep(0.001)


def capture_main(config, output, stop):
    cv2.setNumThreads(1)
    cap = None
    try:
        cap = cv2.VideoCapture(config["index"], backend_id(config["backend"]))
        if not cap.isOpened():
            raise RuntimeError(f"Camera {config['index']} cannot be opened. Check its index, USB connection, and other camera apps.")
        requested = [
            (cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"), "MJPG"),
            (cv2.CAP_PROP_FRAME_WIDTH, config["width"], "width"),
            (cv2.CAP_PROP_FRAME_HEIGHT, config["height"], "height"),
            (cv2.CAP_PROP_FPS, config["fps"], "fps"),
        ]
        unsupported = [name for prop, value, name in requested if not cap.set(prop, value)]
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        metadata = {
            "backend": cap.getBackendName(),
            "driver_fps": cap.get(cv2.CAP_PROP_FPS),
            "unsupported_requests": unsupported,
        }
        seq = 0
        failures = 0
        while not stop.is_set():
            ok, bgr = cap.read()
            if not ok or bgr is None:
                failures += 1
                if failures >= 5:
                    raise RuntimeError("Camera stopped returning frames. It may have disconnected.")
                stop.wait(0.05)
                continue
            failures = 0
            seq += 1
            put_latest(output, {
                "type": "frame", "sequence": seq, "timestamp": time.time(),
                "monotonic": time.monotonic(), "bgr": bgr, "metadata": metadata,
            })
    except Exception as exc:
        # Process boundary: report failure to the supervisor instead of hiding it.
        put_latest(output, {"type": "error", "message": str(exc)})
    finally:
        if cap is not None:
            cap.release()
        output.cancel_join_thread()


def probe_main(index, backend, output):
    cv2.setNumThreads(1)
    cap = cv2.VideoCapture(index, backend_id(backend))
    try:
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            # Some USB cameras return an empty frame while the sensor warms up.
            for _ in range(10):
                ok, frame = cap.read()
                if ok and frame is not None:
                    output.put({
                        "index": index, "label": f"Camera {index}",
                        "backend": cap.getBackendName(),
                        "width": int(frame.shape[1]), "height": int(frame.shape[0]),
                    })
                    break
                time.sleep(0.05)
    finally:
        cap.release()
