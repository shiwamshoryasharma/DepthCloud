"""Capture stabilization: reject blur and fast motion without warping originals."""
import cv2
import numpy as np


def sharpness(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    return float(cv2.Laplacian(gray, cv2.CV_32F).var())


class StabilityGate:
    def __init__(self, minimum_sharpness=25., maximum_motion=80.):
        self.minimum_sharpness, self.maximum_motion = minimum_sharpness, maximum_motion
        self.previous = self.previous_time = self.steady_since = None

    def update(self, bgr, timestamp):
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        ratio = min(1., 640 / gray.shape[1])
        gray = cv2.resize(gray, (round(gray.shape[1] * ratio), round(gray.shape[0] * ratio)))
        focus = sharpness(gray)
        motion, reliable = 0., False
        if self.previous is not None and self.previous.shape == gray.shape and 0 < timestamp - self.previous_time < .75:
            points = cv2.goodFeaturesToTrack(self.previous, 250, .01, 10)
            if points is not None and len(points) >= 12:
                tracked, status, _ = cv2.calcOpticalFlowPyrLK(self.previous, gray, points, None)
                good = status.ravel() > 0
                if good.sum() >= 12:
                    matrix, inliers = cv2.estimateAffinePartial2D(points[good], tracked[good], method=cv2.RANSAC, ransacReprojThreshold=2)
                    if matrix is not None and inliers is not None and inliers.mean() > .6:
                        displacement = np.linalg.norm(tracked[good] - points[good], axis=2).ravel()
                        motion = float(np.median(displacement[inliers.ravel() > 0]) / (timestamp - self.previous_time))
                        reliable = True
        self.previous, self.previous_time = gray, timestamp
        good = reliable and focus >= self.minimum_sharpness and motion <= self.maximum_motion
        if not good:
            self.steady_since = None
        elif self.steady_since is None:
            self.steady_since = timestamp
        held = timestamp - self.steady_since if self.steady_since is not None else 0.
        stable = good and held >= .25
        reason = 'Ready' if stable else ('Blurred image' if focus < self.minimum_sharpness else 'Hold steady' if good else 'Shake or insufficient tracking texture')
        return {'stable': stable, 'reason': reason, 'sharpness': round(focus, 1),
                'motion_px_s': round(motion, 1), 'hold_seconds': round(held, 2)}
