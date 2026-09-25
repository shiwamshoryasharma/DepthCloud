import threading

import cv2
import numpy as np

from depthcloud.workflow.schemas import Intrinsics


class CalibrationService:
    def __init__(self):
        self.lock = threading.RLock()
        self.intrinsics = None
        self.samples = []
        self.board = None
        self.image_size = None

    def status(self):
        with self.lock:
            return {"intrinsics": self.intrinsics.model_dump() if self.intrinsics else None,
                    "samples": len(self.samples), "board": self.board, "maximum_samples": 30}

    def set(self, intrinsics):
        with self.lock:
            self.intrinsics = intrinsics
            return self.status()

    def clear(self):
        with self.lock:
            self.intrinsics = None
            self.samples.clear()
            self.board = self.image_size = None
            return self.status()

    def capture(self, bgr, board):
        with self.lock:
            if len(self.samples) >= 30:
                raise ValueError("Calibration sample limit reached. Solve or clear calibration.")
            size = (bgr.shape[1], bgr.shape[0])
            if self.board is not None and (self.board != board.model_dump() or self.image_size != size):
                raise ValueError("Checkerboard dimensions or capture resolution changed. Clear calibration first.")
            gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
            found, corners = cv2.findChessboardCornersSB(gray, (board.columns, board.rows))
            if not found:
                raise ValueError("Checkerboard not found. Show all inner corners clearly at a different angle.")
            if self.samples and min(float(np.linalg.norm(corners - sample, axis=2).mean()) for sample in self.samples) < 8:
                raise ValueError("This checkerboard view is too similar. Move/tilt the board before capturing again.")
            self.board, self.image_size = board.model_dump(), size
            self.samples.append(corners.astype(np.float32))
            return self.status()

    def solve(self):
        with self.lock:
            if len(self.samples) < 8:
                raise ValueError("Capture at least 8 varied checkerboard views before solving calibration.")
            board = self.board
            obj = np.zeros((board["columns"] * board["rows"], 3), np.float32)
            obj[:, :2] = np.mgrid[0:board["columns"], 0:board["rows"]].T.reshape(-1, 2) * board["square_size"]
            rms, matrix, distortion, _, _ = cv2.calibrateCamera([obj] * len(self.samples), self.samples, self.image_size, None, None)
            if not np.isfinite(rms) or rms > 1.5:
                raise ValueError(f"Calibration RMS {rms:.2f}px is too high. Capture more varied, sharp board views.")
            self.intrinsics = Intrinsics(fx=matrix[0, 0], fy=matrix[1, 1], cx=matrix[0, 2], cy=matrix[1, 2],
                                         width=self.image_size[0], height=self.image_size[1],
                                         distortion=distortion.ravel().tolist(), source="checkerboard", rms=rms)
            return self.status()

