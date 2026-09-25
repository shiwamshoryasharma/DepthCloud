import io
import logging
import threading
import time

import numpy as np

from depthcloud.reconstruction.geometry import CAMERA_TO_ISAAC, CAMERA_TO_THREE
from depthcloud.reconstruction.registration import (
    depth_feature_pose,
    make_cloud,
    prepare_cloud,
    register_clouds,
)
from depthcloud.reconstruction.surface import cloud_buffer, fuse_clouds, mesh_buffer, reconstruct_mesh
from depthcloud.workflow.observations import ObservationStore
from depthcloud.workflow.processing import process_observation

logger = logging.getLogger("depthcloud.reconstruction")


class WorkflowBusy(RuntimeError):
    pass


class Cancelled(RuntimeError):
    pass


class ReconstructionService:
    def __init__(self, models, calibration, max_views=24, max_bytes=512 * 1024**2):
        self.models, self.calibration = models, calibration
        self.store = ObservationStore(max_views, max_bytes)
        self.lock = self.store.lock
        self.state = "idle"
        self.message = "Freeze a frame, select an object, and capture viewpoints."
        self.progress = 0
        self.error = None
        self.busy = False
        self.thread = None
        self.cancel_event = threading.Event()
        self.revision = 0
        self.result = None
        self.cloud = self.mesh = None
        self.buffers = {}
        self.elapsed_seconds = 0
        self.started = 0
        self.active_view = None
        self.live_status = None
        self.capture_status = None

    def _require_idle(self):
        if self.busy:
            raise WorkflowBusy("Reconstruction is busy. Cancel and wait for the current stage to finish before editing.")

    def _invalidate(self):
        self.live_status = None
        self.revision += 1
        self.result = self.cloud = self.mesh = None
        self.buffers.clear()
        self.error = None
        self.state = "ready" if self.store.views else "idle"

    def freeze(self, packet, camera):
        with self.lock:
            self._require_idle()
            return self.store.freeze(packet, camera)

    def capture(self, payload):
        with self.lock:
            self._require_idle()
            view = self.store.capture(payload)
            self._invalidate()
            self.message = "View captured. Move around the same rigid object with overlap, then capture another."
            logger.info("Captured observation %s from frame %s", view["id"], view["sequence_id"])
            return view

    def select(self, view_id, selected):
        with self.lock:
            self._require_idle()
            view = self.store.get(view_id)
            view.selected = selected
            self._invalidate()
            return view.metadata()

    def delete(self, view_id):
        with self.lock:
            self._require_idle()
            self.store.delete(view_id)
            self._invalidate()

    def reset(self):
        with self.lock:
            self._require_idle()
            self.store.reset()
            self._invalidate()
            self.progress = self.elapsed_seconds = 0
            self.active_view = None
            self.message = "Reconstruction reset. Loaded models and camera are retained."
            return self.status()

    def status(self):
        with self.lock:
            return {"session_id": self.store.session_id, "state": self.state, "message": self.message,
                    "busy": self.busy, "progress": self.progress, "error": self.error,
                    "active_view": self.active_view, "views": self.store.list(), "result": self.result,
                    "revision": self.revision, "memory_bytes": self.store.bytes_used(),
                    "memory_limit": self.store.max_bytes, "max_views": self.store.max_views,
                    "elapsed_seconds": round(time.monotonic() - self.started if self.busy else self.elapsed_seconds, 1),
                    "capture": dict(self.capture_status) if self.capture_status else None,
                    "calibration": self.calibration.status(), "live": dict(self.live_status) if self.live_status else None}

    def start(self, options):
        with self.lock:
            self._require_idle()
            selected = [v for v in self.store.views.values() if v.selected]
            if not selected:
                raise ValueError("Select at least one captured observation before reconstruction.")
            self._invalidate()
            for view in selected:
                view.result = {}
                view.error = None
                view.status = "queued"
            self.busy = True
            self.state, self.message, self.progress = "starting", "Preparing selected observations", 0
            self.started = time.monotonic()
            self.cancel_event = threading.Event()
            calibration = self.calibration.intrinsics
            self.thread = threading.Thread(target=self._run, args=(selected, options, calibration),
                                           daemon=True, name="depthcloud-reconstruction")
            self.thread.start()
            return self.status()

    def cancel(self):
        with self.lock:
            if self.busy:
                self.cancel_event.set()
                self.message = "Cancellation requested; waiting for the current native/model operation to finish."
            return self.status()

    def _check_cancel(self):
        if self.cancel_event.is_set():
            raise Cancelled("Reconstruction cancelled.")

    def _stage(self, state, message, progress=None, view=None):
        self._check_cancel()
        with self.lock:
            self.state, self.message = state, message
            if progress is not None:
                self.progress = progress
            if view is not None:
                self.active_view = view.id
                view.status = state
        logger.info("%s: %s", state, message)

    def _run(self, selected, options, calibration):
        accepted, registered, transforms, rejected = [], [], {}, []
        intrinsics_for_job = calibration
        try:
            for index, view in enumerate(selected):
                self._stage("depth_estimation", f"Processing {view.label}", round(index / len(selected) * 85), view)
                try:
                    data = process_observation(view, self.models, intrinsics_for_job, options,
                                               lambda state, message: self._stage(state, message, view=view))
                    if intrinsics_for_job is None:
                        from depthcloud.workflow.schemas import Intrinsics
                        intrinsics_for_job = Intrinsics(**data["intrinsics"])
                    with self.lock:
                        extra = sum(v.nbytes for v in data.values() if isinstance(v, np.ndarray))
                        if self.store.bytes_used() + extra > self.store.max_bytes:
                            raise ValueError("Session memory limit reached while processing. Use fewer/lower-resolution observations.")
                        view.result = data
                    cloud = prepare_cloud(make_cloud(data["points"], data["colors"]), options)
                    if not accepted:
                        transform = np.eye(4)
                        metrics = {"method": "Reference camera", "accepted": True, "fitness": 1.0,
                                   "overlap": 1.0, "rmse": 0.0, "correspondences": len(cloud.points)}
                    else:
                        self._stage("registration", f"Registering {view.label} against accepted observations", view=view)
                        errors = []
                        transform = metrics = None
                        # Try recent overlap first, then the original reference; never concatenate unregistered data.
                        for target in list(reversed(accepted))[:3]:
                            self._check_cancel()
                            initial, matches = depth_feature_pose(view, target, options.icp_distance)
                            target_cloud = make_cloud(target.result["points"], target.result["colors"])
                            try:
                                relative, metrics = register_clouds(cloud, target_cloud, options, initial)
                                if initial is None or matches < 8:
                                    raise ValueError("Ambiguous batch registration: too few object feature correspondences. Use live scanning with slow rotation and textured markers.")
                                transform = transforms[target.id] @ relative
                                metrics["feature_inliers"] = matches
                                metrics["target_view"] = target.id
                                break
                            except ValueError as exc:
                                errors.append(str(exc))
                        if transform is None:
                            raise ValueError(errors[-1] if errors else "No overlapping accepted view.")
                    cloud.transform(transform)
                    with self.lock:
                        view.result["registration"] = metrics
                        view.result["transform_to_world"] = transform.tolist()
                        view.status = "registered"
                    accepted.append(view)
                    registered.append(cloud)
                    transforms[view.id] = transform
                except Cancelled:
                    raise
                except Exception as exc:
                    logger.exception("Observation %s failed", view.id)
                    with self.lock:
                        view.error, view.status = str(exc), "failed"
                    rejected.append({"id": view.id, "label": view.label, "reason": str(exc)})
            if not accepted:
                raise ValueError("No selected observation produced usable registered geometry. Inspect the per-view errors.")
            self._stage("fusion", "Fusing only accepted views, downsampling and removing outliers", 88)
            cloud = fuse_clouds(registered, options)
            mesh, mesh_error = None, None
            if options.create_mesh:
                self._stage("mesh_generation", "Reconstructing a surface with ball pivoting", 94)
                try:
                    mesh = reconstruct_mesh(cloud, options)
                except Exception as exc:
                    logger.warning("Mesh failed; point cloud retained: %s", exc)
                    mesh_error = str(exc)
            self._check_cancel()
            points = np.asarray(cloud.points)
            metadata = {"revision": self.revision, "points": len(points),
                        "vertices": len(mesh.vertices) if mesh is not None else 0,
                        "triangles": len(mesh.triangles) if mesh is not None else 0,
                        "accepted": [v.id for v in accepted], "rejected": rejected,
                        "mesh_error": mesh_error, "dimensions_m": np.ptp(points, axis=0).tolist(),
                        "bounds": [points.min(axis=0).tolist(), points.max(axis=0).tolist()],
                        "units": "meters", "frame": "first accepted camera: +X right, +Y down, +Z forward",
                        "camera_to_three": CAMERA_TO_THREE.tolist(), "camera_to_isaac": CAMERA_TO_ISAAC.tolist(),
                        "options": options.model_dump(), "estimated_geometry": True,
                        "calibration_source": calibration.source if calibration else "depth_pro_estimate"}
            buffers = {"cloud": cloud_buffer(cloud)}
            if mesh is not None:
                buffers["mesh"] = mesh_buffer(mesh)
            with self.lock:
                self._check_cancel()
                self.cloud, self.mesh, self.result, self.buffers = cloud, mesh, metadata, buffers
                self.state, self.progress = "completed", 100
                self.message = f"Reconstructed {len(accepted)} view(s); rejected {len(rejected)}. Geometry is an estimate, not a metrology result."
                for view in accepted:
                    view.status = "completed"
        except Cancelled as exc:
            with self.lock:
                self.state, self.message = "cancelled", str(exc)
                for view in selected:
                    view.status, view.result = "cancelled", {}
        except Exception as exc:
            logger.exception("Reconstruction failed")
            with self.lock:
                self.state, self.error, self.message = "failed", str(exc), str(exc)
        finally:
            with self.lock:
                self.elapsed_seconds = time.monotonic() - self.started
                self.busy = False
                self.active_view = None

    def geometry(self, kind, revision):
        with self.lock:
            if self.result is None or revision != self.result["revision"]:
                raise ValueError("Geometry revision changed or no reconstruction exists.")
            if kind not in self.buffers:
                raise ValueError(f"No {kind} geometry is available.")
            return self.buffers[kind]

    def export(self, kind):
        # Serialize from stable native objects while edits are excluded.
        with self.lock:
            if self.result is None:
                raise ValueError("Reconstruct a selected observation before exporting.")
            output = io.BytesIO()
            if kind == "ply":
                points = np.asarray(self.cloud.points, dtype="<f4")
                colors = np.clip(np.asarray(self.cloud.colors) * 255, 0, 255).astype(np.uint8)
                normals = np.asarray(self.cloud.normals, dtype="<f4")
                output.write(("ply\nformat binary_little_endian 1.0\ncomment DepthCloud meters; camera x-right y-down z-forward\n"
                              f"element vertex {len(points)}\nproperty float x\nproperty float y\nproperty float z\n"
                              "property float nx\nproperty float ny\nproperty float nz\n"
                              "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n").encode())
                records = np.empty(len(points), dtype=[("position", "<f4", 3), ("normal", "<f4", 3), ("color", "u1", 3)])
                records["position"], records["normal"], records["color"] = points, normals, colors
                output.write(records.tobytes())
            elif kind == "obj":
                if self.mesh is None:
                    raise ValueError("Mesh unavailable; PLY point-cloud export is still available.")
                output.write(b"# DepthCloud: meters, +X right +Y down +Z forward\n")
                for point in np.asarray(self.mesh.vertices):
                    output.write(("v %.7g %.7g %.7g\n" % tuple(point)).encode())
                for triangle in np.asarray(self.mesh.triangles) + 1:
                    output.write(("f %d %d %d\n" % tuple(triangle)).encode())
            else:
                raise ValueError("Supported export formats: ply, obj.")
            return output.getvalue()

    def close(self):
        self.cancel()
        if self.thread is not None:
            self.thread.join(timeout=2)

