"""Bounded object-space TSDF surface and conservative sequential pose checks."""

import cv2
import numpy as np

from depthcloud.reconstruction.geometry import camera_matrix


def validate_incremental_pose(transform, source_points, target_points):
    transform = np.asarray(transform)
    rotation = transform[:3, :3]
    if (
        not np.isfinite(transform).all()
        or not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-3)
        or not np.isclose(np.linalg.det(rotation), 1, atol=1e-3)
    ):
        raise ValueError("Tracking transform is not rigid.")
    angle = float(np.degrees(np.arccos(np.clip((np.trace(rotation) - 1) / 2, -1, 1))))
    source_center = np.median(source_points, axis=0)
    target_center = np.median(target_points, axis=0)
    diameter = max(0.02, float(np.linalg.norm(np.ptp(target_points, axis=0))))
    residual_center = float(np.linalg.norm(rotation @ source_center + transform[:3, 3] - target_center))
    motion = float(np.linalg.norm(source_center - target_center))
    scale_ratio = np.linalg.norm(np.ptp(source_points, axis=0)) / diameter
    if angle > 35 or motion > max(0.04, diameter * 0.35) or residual_center > diameter * 0.2:
        raise ValueError("Tracking motion is too large. Rotate slowly and return to the last accepted view.")
    if not 0.65 < scale_ratio < 1.5:
        raise ValueError("Depth/shape changed too much. Check the mask, lighting and occlusion.")
    return {"rotation_deg": angle, "center_motion_m": motion, "center_residual_m": residual_center}


def view_direction(transform):
    direction = transform[:3, :3] @ np.array([0.0, 0.0, -1.0])
    axis = int(np.argmax(np.abs(direction)))
    return (("left", "right"), ("top", "bottom"), ("front", "back"))[axis][int(direction[axis] > 0)]


class SurfaceVolume:
    def __init__(self, points, options):
        import open3d as o3d

        # Fixed allocation; a rotating object cannot grow an unbounded sparse map.
        self.resolution = options.volume_resolution
        extent = np.ptp(points, axis=0)
        self.length = max(0.15, float(np.linalg.norm(extent)) * 1.6)
        self.origin = np.median(points, axis=0) - self.length / 2
        self.voxel_size = self.length / self.resolution
        self.volume = o3d.pipelines.integration.UniformTSDFVolume(
            length=self.length,
            resolution=self.resolution,
            sdf_trunc=self.voxel_size * 4,
            color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8,
            origin=self.origin,
        )
        self.integrated = 0

    def integrate(self, bgr, depth, mask, intrinsics, transform):
        import open3d as o3d

        filtered = np.where((mask > 0) & np.isfinite(depth) & (depth > 0), depth, 0).astype(np.float32)
        color = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        if np.any(intrinsics.distortion):
            matrix = camera_matrix(intrinsics)
            mx, my = cv2.initUndistortRectifyMap(
                matrix,
                np.asarray(intrinsics.distortion),
                None,
                matrix,
                (intrinsics.width, intrinsics.height),
                cv2.CV_32FC1,
            )
            filtered = cv2.remap(filtered, mx, my, cv2.INTER_NEAREST)
            color = cv2.remap(color, mx, my, cv2.INTER_LINEAR)
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d.geometry.Image(np.ascontiguousarray(color)),
            o3d.geometry.Image(filtered),
            depth_scale=1.0,
            depth_trunc=1000.0,
            convert_rgb_to_intensity=False,
        )
        intrinsic = o3d.camera.PinholeCameraIntrinsic(
            intrinsics.width, intrinsics.height, intrinsics.fx, intrinsics.fy, intrinsics.cx, intrinsics.cy
        )
        self.volume.integrate(rgbd, intrinsic, np.linalg.inv(transform))
        self.integrated += 1

    def extract(self):
        mesh = self.volume.extract_triangle_mesh()
        mesh.remove_degenerate_triangles()
        mesh.remove_duplicated_triangles()
        if len(mesh.triangles) < 10:
            raise ValueError("Not enough consistent depth for a surface yet.")
        labels, counts, _ = mesh.cluster_connected_triangles()
        counts = np.asarray(counts)
        # Remove floating slivers, without closing unseen surfaces or inventing a bottom.
        mesh.remove_triangles_by_mask(counts[np.asarray(labels)] < max(10, int(counts.max() * 0.02)))
        mesh.remove_unreferenced_vertices()
        if len(mesh.triangles) > 200_000:
            mesh = mesh.simplify_quadric_decimation(200_000)
        mesh = mesh.filter_smooth_taubin(number_of_iterations=2)
        mesh.compute_vertex_normals()
        import open3d as o3d

        cloud = o3d.geometry.PointCloud()
        cloud.points = mesh.vertices
        cloud.colors = mesh.vertex_colors
        cloud.normals = mesh.vertex_normals
        return cloud, mesh
