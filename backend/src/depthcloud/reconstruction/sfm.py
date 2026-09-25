"""Image-based camera reconstruction with joint bundle adjustment (COLMAP)."""

import hashlib
import json
import threading

import numpy as np

from depthcloud.workflow.schemas import Intrinsics


def camera_intrinsics(camera):
    params = camera.params
    if camera.model_name == "SIMPLE_RADIAL":
        f, cx, cy, k = params
        return Intrinsics(
            fx=f,
            fy=f,
            cx=cx,
            cy=cy,
            width=camera.width,
            height=camera.height,
            distortion=[k, 0, 0, 0, 0],
            source="sfm_estimate",
        )
    if camera.model_name in ("OPENCV", "FULL_OPENCV"):
        fx, fy, cx, cy = params[:4]
        return Intrinsics(
            fx=fx,
            fy=fy,
            cx=cx,
            cy=cy,
            width=camera.width,
            height=camera.height,
            distortion=params[4:].tolist(),
            source="sfm_estimate",
        )
    raise ValueError(f"Unsupported reconstructed camera model: {camera.model_name}")


def reconstruct_cameras(dataset, update, check_cancel):
    import pycolmap

    records = dataset.data["records"]
    signature = {
        "version": 3,
        "calibration": dataset.data["calibration"],
        "images": [
            (
                r["index"],
                (dataset.root / f"{r['index']:04d}.png").stat().st_size,
                (dataset.root / f"{r['index']:04d}.png").stat().st_mtime_ns,
            )
            for r in records
        ],
    }
    key = hashlib.sha256(json.dumps(signature, sort_keys=True).encode()).hexdigest()[:16]
    workspace = dataset.root / "sfm" / key
    cached = workspace / "selected"
    if (cached / "cameras.bin").exists() and (workspace / "summary.json").exists():
        update("camera_alignment", "Loading the verified camera reconstruction for this exact capture set.")
        return pycolmap.Reconstruction(cached), json.loads(
            (workspace / "summary.json").read_text(encoding="utf-8")
        )
    dataset.check_space(len(records) * 8 * 1024**2)
    workspace.mkdir(parents=True, exist_ok=True)
    database = workspace / "images.db"
    token = pycolmap.CancellationToken()
    finished = threading.Event()
    failure = []

    def watch():
        while not finished.wait(0.5):
            try:
                check_cancel()
                dataset.refresh_bytes()
                dataset.check_space(0)
            except Exception as exc:
                failure.append(exc)
                token.cancel()
                return

    watcher = threading.Thread(target=watch, daemon=True, name="depthcloud-sfm-cancellation")
    watcher.start()
    try:
        reader = {}
        calibration = dataset.data["calibration"]
        if calibration:
            intr = Intrinsics(**calibration)
            height, width = dataset.data["shape"]
            intr = intr.scaled(width, height)
            distortion = list(intr.distortion)
            if any(distortion[8:]):
                raise ValueError(
                    "SfM does not support this thin-prism/tilted sensor calibration. Use a standard calibrated camera model."
                )
            distortion = (distortion + [0] * 8)[:8]
            reader = {
                "camera_model": "FULL_OPENCV",
                "camera_params": ",".join(map(str, [intr.fx, intr.fy, intr.cx, intr.cy, *distortion])),
            }
        update(
            "camera_features",
            "Extracting image features for a shared camera model; depth is not used to infer poses.",
        )
        pycolmap.extract_features(
            database,
            dataset.root,
            image_names=[f"{r['index']:04d}.png" for r in records],
            camera_mode=pycolmap.CameraMode.SINGLE,
            reader_options=reader,
            extraction_options={"num_threads": 8, "max_image_size": 1280, "sift": {"max_num_features": 6000}},
            device=pycolmap.Device.cpu,
            cancellation_token=token,
        )
        check_cancel()
        update("camera_matching", "Verifying cross-image correspondences.")
        # Captures are ordered. Local/quadratic temporal neighbors reduce false
        # links between repeated scene details and scale to 1,000 images.
        pycolmap.match_sequential(
            database,
            matching_options={"num_threads": 8},
            pairing_options={"overlap": 10, "quadratic_overlap": True},
            device=pycolmap.Device.cpu,
            cancellation_token=token,
        )
        check_cancel()
        update("camera_alignment", "Jointly solving camera poses and 3D tracks with bundle adjustment.")
        maps = pycolmap.incremental_mapping(
            database,
            dataset.root,
            workspace,
            options={
                "num_threads": 8,
                "min_model_size": 3,
                "random_seed": 0,
                "mapper": {"init_min_tri_angle": 4.0},
                "ba_refine_focal_length": not bool(calibration),
                "ba_refine_extra_params": not bool(calibration),
            },
            cancellation_token=token,
        )
        check_cancel()
        valid = [r for r in maps.values() if r.num_reg_images() >= 3 and r.num_points3D() >= 100]
        if not valid:
            raise ValueError(
                "No connected image reconstruction with sufficient 3D tracks. Capture overlapping camera motion with a stationary textured scene."
            )
        selected = max(valid, key=lambda r: r.num_reg_images())
        error = float(selected.compute_mean_reprojection_error())
        if not np.isfinite(error) or error > 2.5:
            raise ValueError(f"Camera reprojection error {error:.2f}px is too high for reliable geometry.")
        summary = {
            "registered": selected.num_reg_images(),
            "scene_points": selected.num_points3D(),
            "mean_reprojection_px": error,
            "components": [r.num_reg_images() for r in maps.values()],
            "method": "COLMAP structure from motion and bundle adjustment",
        }
        cached.mkdir(exist_ok=True)
        selected.write(cached)
        (workspace / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
        return selected, summary
    finally:
        finished.set()
        watcher.join(2)
        dataset.refresh_bytes()
        if failure:
            raise failure[0]


def shared_metric_scale(frame_scales):
    values = np.asarray(frame_scales, float)
    values = values[np.isfinite(values) & (values > 0)]
    if len(values) < 3:
        raise ValueError("Too few triangulated depth anchors to estimate shared metric scale.")
    scale = float(np.median(values))
    relative_mad = float(np.median(np.abs(values / scale - 1)))
    if relative_mad > 0.25:
        raise ValueError(
            "Depth estimates disagree too much to establish a shared scale. Calibrate the camera and recapture."
        )
    return scale, relative_mad
