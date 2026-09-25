"""Latest-frame live reconstruction. Saved references are detection-only."""

import logging
import threading
import time
from collections import deque

import cv2
import numpy as np

from depthcloud.reconstruction.geometry import CAMERA_TO_ISAAC, CAMERA_TO_THREE, project_depth
from depthcloud.reconstruction.live_geometry import SurfaceVolume, validate_incremental_pose, view_direction
from depthcloud.reconstruction.registration import depth_feature_pose, make_cloud, register_clouds
from depthcloud.reconstruction.scene_tracking import SceneTracker, object_search_box
from depthcloud.reconstruction.surface import cloud_buffer, mesh_buffer
from depthcloud.vision.masking import choose_target, filter_mask_depth, foreground_mask
from depthcloud.vision.reference import ReferenceBank
from depthcloud.workflow.observations import Observation
from depthcloud.workflow.schemas import Intrinsics

logger = logging.getLogger("depthcloud.live")


class LiveScanner:
    def __init__(self, service, camera):
        self.service, self.camera = service, camera
        self.preview = None
        self.stop_event = threading.Event()

    def start(self, options):
        service = self.service
        with service.lock:
            service._require_idle()
            references = [service.store.get(reference_id) for reference_id in options.reference_ids]
            packet, generation = self.camera.latest_for_scan()
            service._invalidate()
            self.preview = None
            self.stop_event = threading.Event()
            service.cancel_event = self.stop_event
            service.busy = True
            service.started = time.monotonic()
            service.state, service.message = (
                "reference_detection",
                "Preparing every included object reference; saved pixels are not fused.",
            )
            service.live_status = {
                "reference_ids": [reference.id for reference in references],
                "reference_count": len(references),
                "scan_mode": options.scan_mode,
                "camera_pose": None,
                "camera_tracking": "initializing" if options.scan_mode == "moving_camera" else "not_used",
                "prepared_references": 0,
                "matched_reference": None,
                "preview_sequence": None,
                "processed": 0,
                "integrated": 0,
                "rejected": 0,
                "skipped": 0,
                "last_sequence": None,
                "tracking": "initializing",
                "last_error": None,
                "surface_updates": 0,
                "directions": [],
                "orientation": "Initial live camera defines front and image-up; not inferred semantic orientation.",
                "processing_fps": 0.0,
                "volume_resolution": options.volume_resolution,
                "voxel_size": None,
                "stopping": False,
                "geometry_only": options.allow_geometry_tracking and options.scan_mode == "rotating_object",
            }
            service.thread = threading.Thread(
                target=self._run,
                args=(references, options, packet.sequence_id, generation),
                daemon=True,
                name="depthcloud-live-scan",
            )
            service.thread.start()
            return service.status()

    def stop(self):
        with self.service.lock:
            if self.service.busy and self.service.live_status:
                self.stop_event.set()
                self.service.live_status["stopping"] = True
                self.service.message = (
                    "Stopping after the current operation; the generated surface will be retained."
                )
            return self.service.status()

    def _status(self, **values):
        with self.service.lock:
            self.service.live_status.update(values)

    def _message(self, state, message):
        with self.service.lock:
            self.service.state, self.service.message = state, message

    def _frame_data(self, packet, options, bank, intrinsics, tracking_only=False, prior_box=None, depth_prediction=None):
        pipeline = options.pipeline
        bgr = packet.bgr
        h, w = bgr.shape[:2]
        # Locate against ALL reference angles in the whole current image first.
        # Saved reference coordinates have no meaning in this new camera frame.
        match_error = None
        try:
            if tracking_only:
                raise ValueError('Waiting for camera pose before object association.')
            match = bank.locate(bgr, prior_box=prior_box) if prior_box is not None else bank.locate(bgr)
        except ValueError as exc:
            if options.scan_mode != "moving_camera":
                raise
            match, match_error = None, str(exc)
        box = match.box if match is not None else (0, 0, w, h)
        mask = match.mask if match is not None else np.zeros((h, w), np.uint8)
        target, warning = None, None
        if match is not None and pipeline.require_detection:
            detections = self.service.models.detect(bgr, pipeline.threshold)
            target, _, warning = choose_target(detections, box, True)
        if match is not None and not options.color_filter:
            x0, y0, x1, y1 = box
            pad = max(3, round(max(x1 - x0, y1 - y0) * .04))
            mask = foreground_mask(bgr, (max(0, x0 - pad), max(0, y0 - pad),
                                         min(w, x1 + pad), min(h, y1 + pad)))
        self._status(matched_reference=match.metadata() if match is not None else None)
        self._preview_frame(packet, mask, box)
        # Only current live RGB reaches depth inference; references never do.
        depth, focal = depth_prediction if depth_prediction is not None else self.service.models.infer_depth(bgr, intrinsics.fx if intrinsics else None)
        removed = 0
        if match is not None and pipeline.depth_cleanup:
            mask, removed = filter_mask_depth(depth, mask, pipeline.depth_outlier_strength)
        # Boundary pixels mix foreground/background and produce flying triangles.
        mask = cv2.erode(mask, np.ones((3, 3), np.uint8))
        valid = np.isfinite(depth) & (depth >= pipeline.depth_min) & (depth <= pipeline.depth_max)
        # Reject depth discontinuities before TSDF, without rewriting raw depth.
        safe = np.where(valid, depth, 0).astype(np.float32)
        local_max = cv2.dilate(safe, np.ones((3, 3), np.uint8))
        local_min = cv2.erode(safe, np.ones((3, 3), np.uint8))
        mask[(local_max - local_min) > np.maximum(0.015, safe * 0.035)] = 0
        mask[~valid] = 0
        intrinsics = intrinsics or Intrinsics(
            fx=focal, fy=focal, cx=(w - 1) / 2, cy=(h - 1) / 2, width=w, height=h, source="depth_pro_estimate"
        )
        if match is not None:
            points, colors = project_depth(depth, bgr, mask, intrinsics, pipeline)
        else:
            points, colors = np.empty((0, 3), np.float32), np.empty((0, 3), np.float32)
        if match is not None and len(points) < 100:
            raise ValueError("Too few clean object pixels. Tighten the reference crop and improve lighting.")
        view = Observation(
            str(packet.sequence_id),
            "Live frame",
            packet.sequence_id,
            packet.timestamp,
            bgr,
            {"x": box[0] / w, "y": box[1] / h, "width": (box[2] - box[0]) / w, "height": (box[3] - box[1]) / h},
            box,
            {},
        )
        view.result = {
            "depth": depth,
            "mask": mask,
            "points": points,
            "colors": colors,
            "intrinsics": intrinsics.model_dump(),
            "target": target,
            "warning": warning,
            "depth_outliers_removed": removed,
            "match_error": match_error,
        }
        self._preview_frame(packet, mask, box)
        return view, intrinsics

    def _preview_frame(self, packet, mask=None, box=None):
        overlay = (packet.bgr * .2).astype(np.uint8)
        if mask is not None:
            overlay[mask > 0] = packet.bgr[mask > 0]
        if box is not None:
            x0, y0, x1, y1 = box
            cv2.rectangle(overlay, (x0, y0), (x1, y1), (90, 220, 190), 2)
        ok, encoded = cv2.imencode(".jpg", overlay, [cv2.IMWRITE_JPEG_QUALITY, 75])
        if ok:
            with self.service.lock:
                self.preview = encoded.tobytes()
                self.service.live_status["preview_sequence"] = packet.sequence_id

    def _publish(self, volume, history, options, directions, intrinsics):
        cloud, mesh = volume.extract()
        points = np.asarray(cloud.points)
        buffers = {"cloud": cloud_buffer(cloud), "mesh": mesh_buffer(mesh)}
        with self.service.lock:
            service = self.service
            service.revision += 1
            service.cloud, service.mesh, service.buffers = cloud, mesh, buffers
            service.result = {
                "revision": service.revision,
                "points": len(points),
                "vertices": len(mesh.vertices),
                "triangles": len(mesh.triangles),
                "accepted": list(history),
                "rejected": [],
                "mesh_error": None,
                "dimensions_m": np.ptp(points, axis=0).tolist(),
                "bounds": [points.min(axis=0).tolist(), points.max(axis=0).tolist()],
                "units": "meters",
                "frame": ("Initial tracked scene camera: +X right,+Y down,+Z forward" if options.scan_mode == "moving_camera" else "First accepted live camera: +X right,+Y down,+Z forward"),
                "camera_to_three": CAMERA_TO_THREE.tolist(),
                "camera_to_isaac": CAMERA_TO_ISAAC.tolist(),
                "estimated_geometry": True,
                "calibration_source": intrinsics.source,
                "options": options.model_dump(),
                "method": "masked TSDF / marching cubes",
                "directions": sorted(directions),
            }
            service.live_status["surface_updates"] += 1

    def _run(self, references, options, last_sequence, generation):
        service = self.service
        volume = previous = None
        previous_pose = np.eye(4)
        scene_tracker = SceneTracker() if options.scan_mode == "moving_camera" else None
        anchor_center = anchor_diameter = None
        history = deque(maxlen=128)
        directions = set()
        started = time.monotonic()
        fatal = None
        try:
            bank = ReferenceBank(service.models.describe_objects)
            for reference in references:
                if self.stop_event.is_set():
                    return
                try:
                    # The user's crop defines reference foreground, not a DETR
                    # class box which can accidentally select a nearby object.
                    mask = foreground_mask(reference.bgr, reference.roi_pixels)
                    mask = bank.add(reference.id, reference.label, reference.bgr, mask)
                except ValueError as exc:
                    with service.lock:
                        reference.status, reference.error = "failed", str(exc)
                    raise ValueError(f'Reference "{reference.label}": {exc} Replace or uncheck this crop.') from exc
                with service.lock:
                    reference.result = {"detections": [], "target": None, "warning": None,
                                        "mask": mask, "reference_only": True}
                    reference.status, reference.error = "detection_reference", None
                    service.live_status["prepared_references"] += 1
            intrinsics = service.calibration.intrinsics
            while not self.stop_event.is_set():
                began = time.monotonic()
                packet, current_generation = self.camera.latest_for_scan()
                if current_generation != generation:
                    raise ValueError(
                        "Camera generation changed. Start a new scan to keep calibration and tracking consistent."
                    )
                if packet.sequence_id <= last_sequence:
                    self.stop_event.wait(0.05)
                    continue
                last_sequence = packet.sequence_id
                if time.monotonic() - packet.captured_monotonic > 2:
                    self._message("waiting_for_frame", "Waiting for a fresh camera frame.")
                    self.stop_event.wait(0.1)
                    continue
                self._status(last_sequence=last_sequence, matched_reference=None, tracking="matching", last_error=None)
                self._preview_frame(packet)
                with service.lock:
                    service.live_status["processed"] += 1
                try:
                    if scene_tracker is not None:
                        self._status(camera_tracking="estimating")
                    self._message("live_depth", "Matching the live object against all included reference angles.")
                    if intrinsics:
                        intrinsics = intrinsics.scaled(packet.bgr.shape[1], packet.bgr.shape[0])
                    view, intrinsics = self._frame_data(packet, options, bank, intrinsics, tracking_only=scene_tracker is not None)
                    if self.stop_event.is_set():
                        break
                    relative = np.eye(4)
                    pose = np.eye(4)
                    metrics = {"method": "first live frame"}
                    if scene_tracker is not None:
                        pose, scale, metrics = scene_tracker.estimate(view)
                        # Camera localization is independent of object visibility.
                        # Commit sparse scene evidence even when fusion is withheld.
                        scene_tracker.commit(view, pose, scale)
                        self._status(camera_pose=pose.tolist(), registration=metrics, camera_tracking="tracked")
                        search_box = object_search_box(anchor_center, anchor_diameter, pose, intrinsics) if anchor_center is not None else None
                        view, intrinsics = self._frame_data(packet, options, bank, intrinsics, prior_box=search_box,
                                                           depth_prediction=(view.result['depth'], intrinsics.fx))
                        if view.result["match_error"]:
                            raise ValueError("Camera tracked; mesh paused: " + view.result["match_error"])
                        # Raw model depth stays unchanged. Corrected depth/points
                        # are used only for fusion in the initial scene's scale.
                        view.result["fusion_depth"] = view.result["depth"] * scale
                        view.result["points"] = view.result["points"] * scale
                        world_points = view.result["points"] @ pose[:3, :3].T + pose[:3, 3]
                        center = np.median(world_points, axis=0)
                        if anchor_center is not None and np.linalg.norm(center - anchor_center) > max(.025, anchor_diameter * .35):
                            raise ValueError("Object moved relative to the scene, or its inferred depth changed inconsistently. Keep the object stationary and return to a tracked view.")
                        relative = np.linalg.inv(previous_pose) @ pose
                        rotation = float(np.degrees(np.arccos(np.clip((np.trace(relative[:3, :3]) - 1) / 2, -1, 1))))
                        translation = float(np.linalg.norm(relative[:3, 3]))
                        metrics.update(rotation_deg=rotation, camera_translation_m=translation)
                        self._status(camera_pose=pose.tolist(), registration=metrics)
                        if previous is not None and rotation < 1. and translation < max(.002, volume.voxel_size):
                            with service.lock:
                                service.live_status["skipped"] += 1
                            self._status(tracking="steady", last_error=None)
                            self._message("live_scanning", "Camera pose unchanged. Move the camera slowly around the stationary object.")
                            self.stop_event.wait(max(.02, options.interval_seconds - (time.monotonic() - began)))
                            continue
                    elif previous is not None:
                        initial, inliers = depth_feature_pose(
                            view, previous, min(0.015, options.pipeline.icp_distance)
                        )
                        if initial is None or inliers < options.min_feature_inliers:
                            if not options.allow_geometry_tracking:
                                raise ValueError(
                                    "Pose uncertain: too few object features. Add removable nonrepeating markers, rotate slowly, or enable ambiguous geometry-only tracking."
                                )
                            initial = np.eye(4)
                        strict = options.pipeline.model_copy(
                            update={
                                "min_fitness": max(0.65, options.pipeline.min_fitness),
                                "icp_distance": min(options.pipeline.icp_distance, 0.015),
                            }
                        )
                        relative, metrics = register_clouds(
                            make_cloud(view.result["points"], view.result["colors"]),
                            make_cloud(previous.result["points"], previous.result["colors"]),
                            strict,
                            initial,
                        )
                        motion = validate_incremental_pose(
                            relative, view.result["points"], previous.result["points"]
                        )
                        metrics.update(motion)
                        metrics["feature_inliers"] = inliers
                        pose = previous_pose @ relative
                        if motion["rotation_deg"] < 1.5 and motion["center_motion_m"] < max(
                            0.002, volume.voxel_size
                        ):
                            with service.lock:
                                service.live_status["skipped"] += 1
                            self._status(tracking="steady", last_error=None)
                            self._message(
                                "live_scanning", "Pose unchanged. Rotate slowly to reveal another surface."
                            )
                            self.stop_event.wait(
                                max(0.02, options.interval_seconds - (time.monotonic() - began))
                            )
                            continue
                    if volume is None:
                        initial_points = (view.result["points"] @ pose[:3, :3].T + pose[:3, 3]) if scene_tracker is not None else view.result["points"]
                        volume = SurfaceVolume(initial_points, options)
                        self._status(voxel_size=volume.voxel_size)
                    world = view.result["points"] @ pose[:3, :3].T + pose[:3, 3]
                    inside = ((world >= volume.origin) & (world <= volume.origin + volume.length)).all(axis=1)
                    if inside.mean() < 0.9:
                        raise ValueError(
                            "Tracked object left the fixed reconstruction volume. Return to the accepted pose or start a new scan."
                        )
                    volume.integrate(view.bgr, view.result.get("fusion_depth", view.result["depth"]), view.result["mask"], intrinsics, pose)
                    if scene_tracker is not None:
                        if anchor_center is None:
                            anchor_center = np.median(world, axis=0)
                            anchor_diameter = float(np.linalg.norm(np.ptp(world, axis=0)))
                    previous, previous_pose = view, pose
                    history.append(view.id)
                    directions.add(view_direction(pose))
                    self._status(
                        integrated=volume.integrated,
                        tracking="camera_tracked" if scene_tracker is not None else ("geometry_only" if options.allow_geometry_tracking else "tracked"),
                        camera_pose=pose.tolist(),
                        last_error=None,
                        directions=sorted(directions),
                        registration=metrics,
                    )
                    self._publish(volume, history, options, directions, intrinsics)
                    self._message(
                        "live_scanning",
                        ("Surface updated from tracked camera motion. Keep the object and background stationary; move through overlapping views." if scene_tracker is not None else "Surface updated. Rotate slowly; show top and bottom through overlapping views. Stop generation retains the mesh."),
                    )
                except ValueError as exc:
                    with service.lock:
                        service.live_status["rejected"] += 1
                    self._status(tracking="uncertain", last_error=str(exc))
                    if scene_tracker is not None and service.live_status["camera_tracking"] == "estimating":
                        self._status(camera_tracking="lost")
                    self._message("tracking_lost", str(exc))
                self._status(
                    processing_fps=round(
                        service.live_status["processed"] / max(0.01, time.monotonic() - started), 2
                    )
                )
                self.stop_event.wait(max(0.02, options.interval_seconds - (time.monotonic() - began)))
        except Exception as exc:
            logger.exception("Live generation stopped")
            fatal = str(exc)
        finally:
            with service.lock:
                service.busy = False
                service.elapsed_seconds = time.monotonic() - service.started
                service.state = "completed" if service.result else ("failed" if fatal else "ready")
                service.error = fatal
                service.message = (
                    fatal
                    or "Generation stopped. The last accepted surface is retained; reference set remains available."
                )
                service.progress = 100 if service.result else 0
                service.live_status["stopping"] = False
                service.live_status["tracking"] = "stopped"
                if options.scan_mode == "moving_camera":
                    service.live_status["camera_tracking"] = "stopped"
            # Native volume and previous full-size frame released when worker exits.
            volume = previous = None
