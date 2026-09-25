"""Camera motion from static scene features; the object mask is never fused as scene.

Transforms map the current camera into a keyframe camera, then the first-camera
world. Keyframes contain sparse descriptors/XYZ only, not full RGB/depth images.
"""
from collections import deque

import cv2
import numpy as np

from depthcloud.reconstruction.geometry import camera_matrix
from depthcloud.workflow.schemas import Intrinsics


def solve_scene_pose(points, pixels, current_depths, intrinsics):
    points, pixels = np.asarray(points, np.float64), np.asarray(pixels, np.float64)
    current_depths = np.asarray(current_depths)
    valid = np.isfinite(points).all(axis=1) & np.isfinite(pixels).all(axis=1)
    valid &= np.isfinite(current_depths) & (current_depths > .05) & (points[:, 2] > .05)
    points, pixels, current_depths = points[valid], pixels[valid], current_depths[valid]
    if len(points) < 24:
        raise ValueError('Camera pose uncertain: too few static scene matches. Return to a tracked view.')
    matrix, distortion = camera_matrix(intrinsics), np.asarray(intrinsics.distortion)
    ok, rvec, tvec, inliers = cv2.solvePnPRansac(
        points, pixels, matrix, distortion, iterationsCount=300,
        reprojectionError=3., confidence=.999, flags=cv2.SOLVEPNP_EPNP,
    )
    if not ok or inliers is None or len(inliers) < 18 or len(inliers) / len(points) < .35:
        raise ValueError('Camera pose uncertain: inconsistent static scene correspondences.')
    chosen = inliers.ravel()
    p, uv = points[chosen], pixels[chosen]
    cells = np.floor(uv / [intrinsics.width / 4, intrinsics.height / 3]).astype(int)
    spread = np.ptp(uv, axis=0) / [intrinsics.width, intrinsics.height]
    if len(np.unique(cells, axis=0)) < 4 or np.any(spread < .15):
        raise ValueError('Camera pose uncertain: scene features are too concentrated.')
    rvec, tvec = cv2.solvePnPRefineLM(p, uv, matrix, distortion, rvec, tvec)
    reprojection, _ = cv2.projectPoints(p, rvec, tvec, matrix, distortion)
    error = np.linalg.norm(reprojection.reshape(-1, 2) - uv, axis=1)
    rotation = cv2.Rodrigues(rvec)[0]
    predicted = p @ rotation.T + tvec.ravel()
    angle = float(np.degrees(np.arccos(np.clip((np.trace(rotation) - 1) / 2, -1, 1))))
    if np.percentile(error, 90) > 3 or np.any(predicted[:, 2] <= .05):
        raise ValueError('Camera pose uncertain: reprojection or visibility check failed.')
    if angle > 50 or np.linalg.norm(tvec) > np.median(p[:, 2]) * .6:
        raise ValueError('Camera motion is too large for reliable scene tracking. Return to an overlapping view.')
    # Compare CURRENT inferred depths with the same scene points projected by
    # PnP. A single robust scale correction avoids treating depth scale drift as
    # camera translation; spatially inconsistent predictions are rejected.
    ratios = predicted[:, 2] / current_depths[chosen]
    scale = float(np.median(ratios))
    deviation = float(np.median(np.abs(ratios / scale - 1)))
    if not .67 <= scale <= 1.5 or deviation > .08:
        raise ValueError('Scene depth is inconsistent across frames; camera pose cannot safely fuse this depth.')
    forward = np.eye(4)
    forward[:3, :3], forward[:3, 3] = rotation, tvec.ravel()
    metrics = {'method': 'static scene RGB/depth PnP', 'scene_inliers': len(chosen),
               'scene_matches': len(points), 'reprojection_px': float(np.median(error)),
               'depth_scale': scale, 'depth_scale_mad': deviation, 'rotation_deg': angle,
               'accepted': True}
    return np.linalg.inv(forward), scale, metrics


def scene_features(view, method="orb"):
    intrinsics = Intrinsics(**view.result['intrinsics'])
    image, depth = view.bgr, view.result['depth']
    height, width = image.shape[:2]
    scale = min(1., 960 / width)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Leave a margin around foreground to avoid mixed-depth boundary features.
    mask = cv2.bitwise_not(cv2.dilate(view.result['mask'], np.ones((15, 15), np.uint8)))
    if scale < 1:
        gray = cv2.resize(gray, (round(width * scale), round(height * scale)))
        mask = cv2.resize(mask, (gray.shape[1], gray.shape[0]), interpolation=cv2.INTER_NEAREST)
    detector = cv2.SIFT_create(nfeatures=3000, contrastThreshold=.02) if method == "sift" else cv2.ORB_create(nfeatures=4000, fastThreshold=12)
    keypoints, descriptors = detector.detectAndCompute(gray, mask)
    if descriptors is None:
        raise ValueError('Camera pose uncertain: no static background features. Use a textured stationary background.')
    pixels = np.array([point.pt for point in keypoints], np.float64) / scale
    xy = np.rint(pixels).astype(int)
    xy[:, 0] = np.clip(xy[:, 0], 0, width - 1)
    xy[:, 1] = np.clip(xy[:, 1], 0, height - 1)
    z = depth[xy[:, 1], xy[:, 0]]
    valid = np.isfinite(z) & (z > .05) & (z < 20)
    pixels, descriptors, z = pixels[valid], descriptors[valid], z[valid]
    if len(pixels) < 24:
        raise ValueError('Camera pose uncertain: too few scene features with valid depth.')
    rays = cv2.undistortPoints(pixels.reshape(-1, 1, 2), camera_matrix(intrinsics),
                              np.asarray(intrinsics.distortion)).reshape(-1, 2)
    points = np.column_stack([rays * z[:, None], z])
    if len(points) < 24:
        raise ValueError('Camera pose uncertain: too few scene features with valid depth.')
    return {'pixels': pixels, 'descriptors': descriptors, 'points': points, 'intrinsics': intrinsics}


class SceneTracker:
    def __init__(self, max_keyframes=12, feature_method="orb"):
        self.feature_method = feature_method
        self.keyframes = deque(maxlen=max_keyframes)
        self.current = None

    def estimate(self, view):
        self.current = scene_features(view, self.feature_method)
        source = self.current
        if not self.keyframes:
            return np.eye(4), 1., {'method': 'first scene keyframe', 'scene_inliers': len(source['points']), 'depth_scale': 1., 'reprojection_px': 0., 'accepted': True}
        failures = []
        for target in reversed(self.keyframes):
            pairs = cv2.BFMatcher(cv2.NORM_L2 if self.feature_method == 'sift' else cv2.NORM_HAMMING).knnMatch(source['descriptors'], target['descriptors'], k=2)
            matches = [pair[0] for pair in pairs if len(pair) == 2 and pair[0].distance < .72 * pair[1].distance]
            # One-to-one correspondences prevent repeated patterns dominating RANSAC.
            unique = {}
            for match in sorted(matches, key=lambda match: match.distance):
                unique.setdefault(match.trainIdx, match)
            matches = list(unique.values())
            a, b = [m.queryIdx for m in matches], [m.trainIdx for m in matches]
            try:
                relative, scale, metrics = solve_scene_pose(
                    target['points'][b], source['pixels'][a], source['points'][a, 2], source['intrinsics'])
                metrics['keyframe_id'] = target['id']
                metrics['relocalized'] = target is not self.keyframes[-1]
                return target['pose'] @ relative, scale, metrics
            except ValueError as exc:
                failures.append(str(exc))
        raise ValueError(failures[0] if failures else 'No scene keyframe is available.')

    def commit(self, view, pose, scale):
        features = self.current if self.current is not None else scene_features(view, self.feature_method)
        if self.keyframes:
            relative = np.linalg.inv(self.keyframes[-1]['pose']) @ pose
            angle = np.degrees(np.arccos(np.clip((np.trace(relative[:3, :3]) - 1) / 2, -1, 1)))
            distance = np.linalg.norm(relative[:3, 3])
            if angle < 3 and distance < max(.008, float(np.median(features['points'][:, 2])) * .02):
                self.current = None
                return
        self.keyframes.append({'id': view.id, 'pose': pose.copy(),
                               'points': features['points'] * scale,
                               'descriptors': features['descriptors'].copy()})
        self.current = None


def object_search_box(center, diameter, camera_pose, intrinsics):
    """Project a bounded world-space object neighborhood into the new camera."""
    corners = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)], float)
    corners = np.asarray(center) + corners * max(.025, diameter * .65)
    world_to_camera = np.linalg.inv(camera_pose)
    camera = corners @ world_to_camera[:3, :3].T + world_to_camera[:3, 3]
    if np.any(camera[:, 2] <= .05):
        raise ValueError('Camera is too close to the object volume. Move back to frame the whole object.')
    pixels, _ = cv2.projectPoints(camera, np.zeros(3), np.zeros(3), camera_matrix(intrinsics), np.asarray(intrinsics.distortion))
    pixels = pixels.reshape(-1, 2)
    low = np.maximum([0, 0], pixels.min(axis=0))
    high = np.minimum([intrinsics.width, intrinsics.height], pixels.max(axis=0))
    if np.any(high - low < 4):
        raise ValueError('Tracked object is outside the current camera image.')
    return tuple(np.rint(np.r_[low, high]).astype(int))
