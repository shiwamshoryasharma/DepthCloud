import cv2
import numpy as np

# Right-handed coordinate conversions, applied once at display/integration boundaries.
CAMERA_TO_THREE = np.diag([1.0, -1.0, -1.0, 1.0])
CAMERA_TO_ISAAC = np.array([[0, 0, 1, 0], [-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 0, 1]], dtype=float)


def camera_matrix(intrinsics):
    return np.array([[intrinsics.fx, 0, intrinsics.cx], [0, intrinsics.fy, intrinsics.cy], [0, 0, 1]], dtype=float)


def project_depth(depth, bgr, mask, intrinsics, options):
    h, w = depth.shape
    if bgr.shape[:2] != (h, w) or mask.shape != (h, w):
        raise ValueError("Depth, color and foreground mask dimensions must match.")
    intrinsics = intrinsics.scaled(w, h)
    y, x = np.mgrid[0:h:options.stride, 0:w:options.stride]
    z = depth[y, x]
    valid = np.isfinite(z) & (z >= options.depth_min) & (z <= options.depth_max) & (mask[y, x] > 0)
    y, x, z = y[valid], x[valid], z[valid]
    if len(z) == 0:
        raise ValueError("No valid object depth remains inside the selected range.")
    # Lens distortion is handled in ray directions, preserving source pixel/depth alignment.
    pixels = np.stack([x, y], axis=-1).astype(np.float64).reshape(-1, 1, 2)
    rays = cv2.undistortPoints(pixels, camera_matrix(intrinsics), np.asarray(intrinsics.distortion)).reshape(-1, 2)
    points = np.column_stack([rays[:, 0] * z, rays[:, 1] * z, z]).astype(np.float32)
    colors = bgr[y, x, ::-1].astype(np.float32) / 255
    if len(points) > 200_000:
        chosen = np.linspace(0, len(points) - 1, 200_000, dtype=int)
        points, colors = points[chosen], colors[chosen]
    return points, colors


def rigid_transform(source, target):
    source, target = np.asarray(source, float), np.asarray(target, float)
    if source.shape != target.shape or source.ndim != 2 or len(source) < 3:
        raise ValueError("At least three paired 3D correspondences are required.")
    a, b = source.mean(axis=0), target.mean(axis=0)
    if np.linalg.matrix_rank(source - a) < 2:
        raise ValueError("Correspondences are geometrically degenerate.")
    u, _, vt = np.linalg.svd((source - a).T @ (target - b))
    rotation = vt.T @ u.T
    if np.linalg.det(rotation) < 0:
        vt[-1] *= -1
        rotation = vt.T @ u.T
    transform = np.eye(4)
    transform[:3, :3] = rotation
    transform[:3, 3] = b - rotation @ a
    return transform

