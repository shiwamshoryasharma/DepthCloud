import numpy as np

from depthcloud.reconstruction.geometry import project_depth
from depthcloud.vision.masking import choose_target, filter_mask_depth, foreground_mask
from depthcloud.workflow.schemas import Intrinsics


def process_observation(view, models, calibration, options, stage):
    h, w = view.bgr.shape[:2]
    intrinsics = calibration.scaled(w, h) if calibration is not None else None
    stage("depth_estimation", "Estimating depth from the original, unmasked RGB observation")
    depth, focal = models.infer_depth(view.bgr, intrinsics.fx if intrinsics else None)
    stage("detecting", "Detecting objects with the local DETR model")
    detections = models.detect(view.bgr, options.threshold)
    target, box, warning = choose_target(detections, view.roi_pixels, options.require_detection)
    stage("masking", "Separating foreground with GrabCut")
    mask = foreground_mask(view.bgr, box)
    raw_mask_count = int(np.count_nonzero(mask))
    removed = 0
    if options.depth_cleanup:
        mask, removed = filter_mask_depth(depth, mask, options.depth_outlier_strength)
    if intrinsics is None:
        intrinsics = Intrinsics(fx=focal, fy=focal, cx=(w - 1) / 2, cy=(h - 1) / 2,
                                width=w, height=h, source="depth_pro_estimate")
    stage("point_cloud_generation", "Projecting valid object depth into camera-space meters")
    points, colors = project_depth(depth, view.bgr, mask, intrinsics, options)
    if len(points) < 30:
        raise ValueError("Fewer than 30 valid object points. Adjust the ROI or depth range.")
    valid = (mask > 0) & np.isfinite(depth) & (depth >= options.depth_min) & (depth <= options.depth_max)
    values = depth[valid]
    return {"depth": depth, "mask": mask, "points": points, "colors": colors,
            "detections": detections, "target": target, "target_box": box, "warning": warning,
            "intrinsics": intrinsics.model_dump(), "focal_px": focal, "point_count": len(points),
            "depth_stats": {"minimum": float(values.min()), "maximum": float(values.max()),
                            "median": float(np.median(values)),
                            "valid_percent": float(valid.sum() / max(1, np.count_nonzero(mask)) * 100)},
            "depth_outliers_removed": removed, "raw_foreground_pixels": raw_mask_count,
            "mask_method": "OpenCV GrabCut + morphology + largest component; optional robust depth cleanup"}

