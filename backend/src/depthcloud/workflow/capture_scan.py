"""Collect stable object detections, seal the disk set, then reconstruct offline."""
import logging
import threading
import time

import cv2

from depthcloud.reconstruction.saved_scan import build_saved_mesh
from depthcloud.vision.masking import foreground_mask
from depthcloud.vision.reference import ReferenceBank
from depthcloud.vision.stability import StabilityGate, sharpness
from depthcloud.workflow.capture_dataset import CaptureDataset
from depthcloud.workflow.schemas import CaptureOptions
from depthcloud.workflow.service import Cancelled

logger = logging.getLogger('depthcloud.capture')


class CaptureScanner:
    def __init__(self, service, camera, root):
        self.service, self.camera = service, camera
        self.active = False
        try:
            self.dataset = CaptureDataset(root)
            if self.dataset.data:
                CaptureOptions(**self.dataset.data['options'])
                if not isinstance(self.dataset.data.get('references'), list):
                    raise ValueError('Invalid saved reference metadata.')
        except (ValueError, KeyError, TypeError, OSError) as exc:
            self.dataset = CaptureDataset(root, load=False)
            self._update('failed', f'Capture manifest cannot be recovered: {exc}. Discard the saved set to start again.')
            return
        self._update('ready' if self.dataset.sealed else 'paused' if self.dataset.data else 'empty',
                     'Saved capture set recovered; resume collection or reconstruct it.' if self.dataset.data else 'Capture reference crops to identify one object.')
        self._restore_surface()

    def _restore_surface(self):
        import hashlib

        import numpy as np
        import open3d as o3d

        from depthcloud.reconstruction.surface import cloud_buffer, mesh_buffer
        report=self.dataset.data.get('report', {}) if self.dataset.data else {}
        path=self.dataset.root/'surface-v2.ply'
        if report.get('algorithm_version') != 2 or not path.is_file():
            return
        if path.stat().st_size > 64*1024**2 or hashlib.sha256(path.read_bytes()).hexdigest() != report.get('surface_sha256'):
            self._update('ready','Saved mesh integrity check failed; rebuild the retained images.')
            return
        mesh=o3d.io.read_triangle_mesh(str(path))
        if not 10 <= len(mesh.triangles) <= 200000 or not np.isfinite(np.asarray(mesh.vertices)).all():
            self._update('ready','Saved mesh is invalid; rebuild the retained images.')
            return
        mesh.compute_vertex_normals()
        cloud=o3d.geometry.PointCloud()
        cloud.points,cloud.colors,cloud.normals=mesh.vertices,mesh.vertex_colors,mesh.vertex_normals
        result={k:v for k,v in report.items() if k not in ('poses','per_view_depth','silhouettes','surface_sha256')}
        self.service.revision+=1
        result['revision']=self.service.revision
        self.service.cloud,self.service.mesh,self.service.result=cloud,mesh,result
        self.service.buffers={'cloud':cloud_buffer(cloud),'mesh':mesh_buffer(mesh)}
        self.service.state='completed'
        self._update('completed','Restored the validated outer-shape estimate. Hidden concavities remain unmeasured.',
                     registered=len(result['accepted']),rejected=len(result['rejected']),reconstruction_quality=result['quality'])

    def _update(self, phase=None, message=None, **values):
        with self.service.lock:
            state = self.service.capture_status or {}
            state.update(values)
            if phase:
                state['phase'] = phase
            if message:
                state['message'] = message
            state.update(collected=self.dataset.count, sealed=self.dataset.sealed,
                         target=self.dataset.data['options']['target_frames'] if self.dataset.data else 1000,
                         disk_bytes=self.dataset.data['bytes'] if self.dataset.data else 0,
                         reference_count=len(self.dataset.data['options']['reference_ids']) if self.dataset.data else 0,
                         interval_seconds=1.25, active=self.active)
            self.service.capture_status = state
            if self.active:
                self.service.state = 'capture_' + state['phase']
                self.service.message = state.get('message', '')

    def _launch(self, action, args=()):
        self.service._require_idle()
        self.service._invalidate()
        self.active = self.service.busy = True
        self.service.cancel_event = threading.Event()
        self.service.started = time.monotonic()
        self._update('preparing', 'Preparing the saved capture workflow.', processed=0, registered=0, rejected=0, error=None)
        self.service.thread = threading.Thread(target=self._run, args=(action, args), daemon=True, name='depthcloud-capture-set')
        self.service.thread.start()
        return self.service.status()

    def start(self, options):
        with self.service.lock:
            self.service._require_idle()
            if self.dataset.data or self.dataset.manifest.exists():
                raise ValueError('A saved capture set exists. Resume it, build it, or discard it first.')
            references = [self.service.store.get(identifier) for identifier in options.reference_ids]
            packet, generation = self.camera.latest_for_scan()
            calibration = self.service.calibration.intrinsics
            self.dataset.create(options.model_dump(), [], calibration.model_dump() if calibration else None)
            return self._launch(self._collect, (options, references, packet.sequence_id, generation))

    def resume(self):
        with self.service.lock:
            self.service._require_idle()
            if not self.dataset.data or self.dataset.sealed:
                raise ValueError('There is no unfinished capture set to resume.')
            if not self.dataset.data['references']:
                raise ValueError('Reference preparation was interrupted. Discard this empty set and start again.')
            packet, generation = self.camera.latest_for_scan()
            if self.dataset.count and list(packet.bgr.shape[:2]) != self.dataset.data['shape']:
                raise ValueError('Resume requires the same camera resolution.')
            return self._launch(self._collect, (CaptureOptions(**self.dataset.data['options']), None, packet.sequence_id, generation))

    def build(self):
        with self.service.lock:
            self.service._require_idle()
            self.dataset.seal()
            return self._launch(self._build)

    def pause(self):
        with self.service.lock:
            if self.active:
                self.service.cancel_event.set()
                self._update(message='Stopping after the current operation; saved captures will be retained.')
            return self.service.status()

    def discard(self):
        with self.service.lock:
            self.service._require_idle()
            self.dataset.discard()
            self.service._invalidate()
            self.service.capture_status = None
            self._update('empty', 'Temporary capture set discarded. Your detection references are retained.')
            return self.service.status()

    def _prepare_bank(self, references):
        bank = ReferenceBank(self.service.models.describe_objects)
        if references is None:
            for i, reference in enumerate(self.dataset.data['references']):
                self.service._check_cancel()
                bgr = cv2.imread(str(self.dataset.root / f'reference-{i}.png'))
                mask = cv2.imread(str(self.dataset.root / f'reference-{i}-mask.png'), cv2.IMREAD_GRAYSCALE)
                if bgr is None or mask is None:
                    raise ValueError('Saved reference files are incomplete. Discard this set and start again.')
                bank.add(reference['id'], reference['label'], bgr, mask)
        else:
            metadata = []
            for i, reference in enumerate(references):
                self.service._check_cancel()
                mask = bank.add(reference.id, reference.label, reference.bgr, foreground_mask(reference.bgr, reference.roi_pixels))
                self.dataset.write_image(f'reference-{i}.png', reference.bgr)
                self.dataset.write_image(f'reference-{i}-mask.png', mask)
                metadata.append({'id': reference.id, 'label': reference.label})
                with self.service.lock:
                    reference.result = {'mask': mask, 'reference_only': True}
                    reference.status = 'detection_reference'
            self.dataset.data['references'] = metadata
            self.dataset.save()
        return bank

    def _collect(self, options, references, last_sequence, generation):
        bank = self._prepare_bank(references)
        gate = StabilityGate(options.minimum_sharpness, options.maximum_motion)
        next_attempt = 0.
        self._update('collecting', 'Move around the stationary object. Sharp, steady detections are saved every 1.25 seconds.', skipped=0)
        while not self.dataset.sealed:
            self.service._check_cancel()
            packet, current_generation = self.camera.latest_for_scan()
            if generation != current_generation:
                raise ValueError('Camera restarted. Captures are retained; resume with the same camera and scene.')
            if packet.sequence_id <= last_sequence or time.monotonic() - packet.captured_monotonic > 2:
                self.service.cancel_event.wait(.04)
                continue
            last_sequence = packet.sequence_id
            quality = gate.update(packet.bgr, packet.captured_monotonic)
            self._update(quality=quality)
            if not quality['stable'] or packet.captured_monotonic < next_attempt:
                self.service.cancel_event.wait(.025)
                continue
            next_attempt = packet.captured_monotonic + 1.25
            try:
                match = bank.locate(packet.bgr)
                x0, y0, x1, y1 = match.box
                object_focus = sharpness(packet.bgr[y0:y1, x0:x1])
                if object_focus < options.minimum_sharpness:
                    raise ValueError('Object is blurred; hold steady and improve lighting.')
            except ValueError as exc:
                self._update(message=str(exc), skipped=self.service.capture_status.get('skipped', 0) + 1)
                continue
            self.service._check_cancel()
            self.dataset.append(packet.bgr, match.mask, {'timestamp': packet.timestamp,
                'sequence': packet.sequence_id, 'match': match.metadata(), 'quality': quality,
                'object_sharpness': object_focus})
            self._update(message=f'Saved {self.dataset.count} / {options.target_frames}. Continue through overlapping angles.',
                         latest=self.dataset.count - 1, match=match.metadata())
        # No camera access after this point. References never become mesh inputs.
        self._build()

    def _build(self):
        options = CaptureOptions(**self.dataset.data['options'])
        cloud, mesh, buffers, result = build_saved_mesh(self.dataset, self.service.models, options,
                                                       self._update, self.service._check_cancel)
        self.service._check_cancel()
        with self.service.lock:
            self.service.revision += 1
            result['revision'] = self.service.revision
            self.service.cloud, self.service.mesh, self.service.buffers = cloud, mesh, buffers
            self.service.result = result
            self.service.progress = 100
        self._update('completed', f'Outer-shape estimate from {len(result["accepted"])} / {self.dataset.count} jointly aligned images. Hidden concavities are not measured.' if result.get('geometry_kind') == 'visual_hull' else f'Mesh generated from {len(result["accepted"])} / {self.dataset.count} saved images. Unseen surfaces remain incomplete.',
                     registered=len(result['accepted']), rejected=len(result['rejected']), directions=result['directions'], reconstruction_quality=result.get('quality'))

    def _run(self, action, args):
        try:
            action(*args)
        except Cancelled:
            self._update('ready' if self.dataset.sealed else 'paused', 'Stopped. Saved images are retained; resume collection or reconstruct when ready.')
        except Exception as exc:
            logger.exception('Saved capture workflow stopped')
            self._update('failed', str(exc), error=str(exc))
            with self.service.lock:
                self.service.error = str(exc)
        finally:
            with self.service.lock:
                self.active = self.service.busy = False
                self.service.elapsed_seconds = time.monotonic() - self.service.started
                self._update()
                self.service.state = 'completed' if self.service.result else 'ready'
