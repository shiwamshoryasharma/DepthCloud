"""One globally posed object volume, with per-view silhouette validation."""

import hashlib
import io
import json

import cv2
import numpy as np

from depthcloud.reconstruction.geometry import CAMERA_TO_ISAAC, CAMERA_TO_THREE
from depthcloud.reconstruction.sfm import camera_intrinsics, reconstruct_cameras, shared_metric_scale
from depthcloud.reconstruction.surface import cloud_buffer, mesh_buffer
from depthcloud.reconstruction.visual_hull import color_surface, silhouette_agreement, silhouette_surface


def validate_agreement(metrics):
    values = np.asarray([m["silhouette_iou"] for m in metrics])
    if (
        len(values) < 3
        or not np.isfinite(values).all()
        or np.median(values) < 0.75
        or np.percentile(values, 10) < 0.55
    ):
        raise ValueError(
            "The shared mesh does not explain the captured silhouettes reliably. No inconsistent surface was published."
        )
    return {
        "median_silhouette_iou": float(np.median(values)),
        "p10_silhouette_iou": float(np.percentile(values, 10)),
    }


def build_multiview_mesh(dataset, models, options, update, check_cancel):
    import open3d as o3d

    from depthcloud.reconstruction.visual_hull import refine_silhouette

    if not dataset.sealed or dataset.count < 3:
        raise ValueError("A sealed set of at least 3 images is required.")
    # Filtering is separate from camera estimation: keep original RGB context.
    usable = set()
    rejected = []
    for i in range(dataset.count):
        check_cancel()
        update(
            "filtering",
            f"Preparing object silhouettes: {i + 1} / {dataset.count}",
            processed=i + 1,
            stage_total=dataset.count,
        )
        try:
            clean, filtered = refine_silhouette(dataset.image(i), dataset.image(i, "mask"))
            dataset.write_image(f"{i:04d}-clean.png", clean)
            dataset.write_image(f"{i:04d}-filtered.png", filtered)
            usable.add(i)
        except ValueError as exc:
            rejected.append({"id": str(i), "label": f"Capture {i + 1}", "reason": str(exc)})
    reconstruction, camera_quality = reconstruct_cameras(dataset, update, check_cancel)
    images = sorted((im for im in reconstruction.images.values() if im.has_pose), key=lambda im: im.name)
    views = []
    ratios = []
    depth_stats = []
    # Preserve full detail for small sets without multiplying mask RAM unboundedly.
    mask_pixels_per_view = 256 * 1024**2 / max(1, len(images))
    for image in images:
        check_cancel()
        index = int(image.name.split(".")[0])
        if index not in usable:
            continue
        intr = camera_intrinsics(image.camera)
        update(
            "depth_scale",
            f"Estimating one shared scale: {len(views) + 1} / {len(images)}",
            processed=len(views) + 1,
            stage_total=len(images),
            registered=len(images),
            rejected=dataset.count - len(images),
        )
        bgr = dataset.image(index)
        mask = cv2.imread(str(dataset.root / f"{index:04d}-clean.png"), cv2.IMREAD_GRAYSCALE)
        depth, _ = models.infer_depth(bgr, intr.fx)
        transform = np.eye(4)
        transform[:3] = image.cam_from_world().matrix()
        samples = []
        for feature in image.points2D:
            if not feature.has_point3D():
                continue
            point = reconstruction.points3D[feature.point3D_id]
            if point.error > 2:
                continue
            x, y = np.rint(feature.xy).astype(int)
            if not 0 <= x < depth.shape[1] or not 0 <= y < depth.shape[0]:
                continue
            z = (transform[:3, :3] @ point.xyz + transform[:3, 3])[2]
            prediction = float(depth[y, x])
            if z > 0 and 0.05 < prediction < 20:
                samples.append(prediction / z)
        if len(samples) >= 30:
            ratios.append(float(np.median(samples)))
        valid = (mask > 0) & np.isfinite(depth) & (depth > 0)
        if np.any(valid):
            depth_stats.append(
                {
                    "index": index,
                    "median_inferred_depth_m": float(np.median(depth[valid])),
                    "scale_anchors": len(samples),
                }
            )
        buffer = io.BytesIO()
        np.savez_compressed(buffer, depth=np.where(valid, depth, 0).astype(np.float32), mask=mask)
        dataset.write(f"{index:04d}-sfm-depth.npz", buffer.getvalue())
        ratio = min(1.0, 1280 / intr.width, np.sqrt(mask_pixels_per_view / (intr.width * intr.height)))
        width, height = int(intr.width * ratio), int(intr.height * ratio)
        mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
        views.append(
            {
                "index": index,
                "intrinsics": intr.scaled(width, height),
                "world_to_camera": transform,
                "mask": mask,
            }
        )
    scale, scale_mad = shared_metric_scale(ratios)
    if len(views) < 3:
        raise ValueError("Too few jointly registered object views to reconstruct.")
    # A single similarity transform establishes the world and scale for ALL
    # cameras. Never scale or rotate individual object surfaces to force a fit.
    first = views[0]["world_to_camera"].copy()
    for view in views:
        transform = view["world_to_camera"] @ np.linalg.inv(first)
        transform[:3, 3] *= scale
        view["world_to_camera"] = transform
    registered = {view["index"] for view in views}
    for index in sorted(usable - registered):
        rejected.append(
            {
                "id": str(index),
                "label": f"Capture {index + 1}",
                "reason": "Not in the selected connected camera reconstruction; disconnected sides are never overlaid.",
            }
        )
    update("shape_constraints", "Solving the common object volume from calibrated viewing cones.")
    mesh, quality = silhouette_surface(views, options.volume_resolution, update, check_cancel)
    update("shape_validation", "Checking the shared mesh against every registered object silhouette.")
    agreement = silhouette_agreement(mesh, views, check_cancel)
    dataset.write(
        "shape-validation.json",
        json.dumps({"silhouettes": agreement, "camera_quality": camera_quality}, allow_nan=False).encode(
            "utf-8"
        ),
    )
    quality.update(validate_agreement(agreement))
    update("surface_color", "Projecting observed colors onto the verified outer surface.")
    quality["colored_fraction"] = color_surface(mesh, views, dataset.image, check_cancel)
    quality.update(camera_quality)
    quality["metric_scale_relative_mad"] = scale_mad
    points = np.asarray(mesh.vertices)
    cloud = o3d.geometry.PointCloud()
    cloud.points, cloud.colors, cloud.normals = mesh.vertices, mesh.vertex_colors, mesh.vertex_normals
    result = {
        "points": len(points),
        "vertices": len(points),
        "triangles": len(mesh.triangles),
        "accepted": [str(v["index"]) for v in views],
        "rejected": rejected,
        "mesh_error": None,
        "dimensions_m": np.ptp(points, axis=0).tolist(),
        "bounds": [points.min(axis=0).tolist(), points.max(axis=0).tolist()],
        "units": "meters",
        "frame": "First jointly registered camera: +X right,+Y down,+Z forward",
        "camera_to_three": CAMERA_TO_THREE.tolist(),
        "camera_to_isaac": CAMERA_TO_ISAAC.tolist(),
        "estimated_geometry": True,
        "calibration_source": "sfm_estimate" if not dataset.data["calibration"] else "calibrated",
        "method": "COLMAP bundle adjustment / multi-view silhouette outer shape",
        "geometry_kind": "visual_hull",
        "quality": quality,
        "complete_coverage": False,
        "directions": [],
        "limitations": "Outer-shape estimate. Hidden concavities and unobserved surfaces are not measured. Dimensions use one inferred scale.",
    }
    check_cancel()
    dataset.check_space(32 * 1024**2)
    temporary = dataset.root / "surface-v2.tmp.ply"
    if not o3d.io.write_triangle_mesh(str(temporary), mesh):
        raise ValueError("Could not persist the reconstructed surface.")
    temporary.replace(dataset.root / "surface-v2.ply")
    dataset.data["report"] = {
        **result,
        "algorithm_version": 2,
        "silhouettes": agreement,
        "per_view_depth": depth_stats,
        "shared_scale_m_per_sfm_unit": scale,
        "poses": [
            {
                "index": v["index"],
                "matrix": np.linalg.inv(v["world_to_camera"]).tolist(),
                "intrinsics": v["intrinsics"].model_dump(),
            }
            for v in views
        ],
        "surface_sha256": hashlib.sha256((dataset.root / "surface-v2.ply").read_bytes()).hexdigest(),
    }
    dataset.refresh_bytes()
    dataset.save()
    return cloud, mesh, {"cloud": cloud_buffer(cloud), "mesh": mesh_buffer(mesh)}, result
