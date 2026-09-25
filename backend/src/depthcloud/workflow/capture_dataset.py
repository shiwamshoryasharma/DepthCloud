"""One recoverable, disk-backed capture set. No camera arrays retained in RAM."""
import json
import shutil
from pathlib import Path

import cv2
import numpy as np


class CaptureDataset:
    LIMIT = 16 * 1024**3
    RESERVE = 1024**3

    def __init__(self, root, load=True):
        self.root = Path(root).resolve()
        self.manifest = self.root / 'manifest.json'
        self.data = None
        if load and self.manifest.exists():
            self.data = json.loads(self.manifest.read_text(encoding='utf-8'))
            if not isinstance(self.data, dict):
                raise ValueError('Invalid capture manifest structure.')
            records = self.data.get('records', [])
            if self.data.get('version') != 1 or len(records) > 1000:
                raise ValueError('Unsupported or invalid capture manifest.')
            if any(record.get('index') != i for i, record in enumerate(records)):
                raise ValueError('Invalid capture frame indices.')
            if not 1 <= self.data['options']['target_frames'] <= 1000:
                raise ValueError('Invalid capture target.')
            self.refresh_bytes()

    @property
    def count(self):
        return len(self.data['records']) if self.data else 0

    @property
    def sealed(self):
        return bool(self.data and self.data['sealed'])

    def save(self):
        temporary = self.root / 'manifest.tmp'
        temporary.write_text(json.dumps(self.data, allow_nan=False), encoding='utf-8')
        temporary.replace(self.manifest)

    def create(self, options, references, calibration=None):
        if self.data:
            raise ValueError('A capture set already exists. Resume, reconstruct, or discard it first.')
        self.root.mkdir(parents=True, exist_ok=True)
        if not 1 <= options['target_frames'] <= 1000:
            raise ValueError('Capture target must be between 1 and 1000.')
        self.data = {'version': 1, 'options': options, 'references': references,
                     'calibration': calibration, 'records': [], 'sealed': False, 'bytes': 0}
        self.save()

    def check_space(self, size):
        if self.data['bytes'] + size > self.LIMIT:
            raise ValueError('Temporary scan reached its 16 GiB disk budget. Build the saved set or discard it.')
        if shutil.disk_usage(self.root).free < size + self.RESERVE:
            raise ValueError('Insufficient disk space; 1 GiB is reserved. Saved captures are retained.')

    def write(self, name, data):
        # Only internal generated basenames are accepted.
        if Path(name).name != name or '/' in name or '\\' in name:
            raise ValueError('Invalid capture filename.')
        path = self.root / name
        old = path.stat().st_size if path.exists() else 0
        self.check_space(max(0, len(data) - old))
        temporary = path.with_suffix(path.suffix + '.tmp')
        temporary.write_bytes(data)
        temporary.replace(path)
        self.data['bytes'] += len(data) - old

    def write_image(self, name, image):
        ok, encoded = cv2.imencode('.png', image)
        if not ok:
            raise ValueError('Could not encode captured image.')
        self.write(name, encoded.tobytes())

    def append(self, bgr, mask, metadata):
        if self.sealed:
            raise ValueError('Capture set is sealed; later camera images cannot modify it.')
        if self.count and metadata['timestamp'] - self.data['records'][-1]['timestamp'] < 1.25:
            raise ValueError('Accepted captures must be at least 1.25 seconds apart.')
        if mask.shape != bgr.shape[:2] or bgr.dtype != np.uint8:
            raise ValueError('Capture image/mask dimensions are inconsistent.')
        index = self.count
        if index and list(bgr.shape[:2]) != self.data['shape']:
            raise ValueError('Camera resolution changed. Reconstruct this set before starting another.')
        self.write_image(f'{index:04d}.png', bgr)
        self.write_image(f'{index:04d}-mask.png', mask)
        self.data['shape'] = list(bgr.shape[:2])
        self.data['records'].append({**metadata, 'index': index})
        self.data['sealed'] = self.count >= self.data['options']['target_frames']
        try:
            self.save()
        except Exception:
            self.data['records'].pop()
            self.data['sealed'] = False
            raise
        return index

    def seal(self):
        if self.count < 3:
            raise ValueError('Capture at least 3 overlapping images before reconstruction.')
        self.data['sealed'] = True
        self.save()

    def image(self, index, kind='original'):
        if not 0 <= index < self.count or kind not in ('original', 'mask', 'filtered'):
            raise ValueError('Captured frame is unavailable.')
        suffix = '' if kind == 'original' else '-' + kind
        image = cv2.imread(str(self.root / f'{index:04d}{suffix}.png'), cv2.IMREAD_UNCHANGED)
        if image is None:
            raise ValueError('Captured file is missing or incomplete.')
        return image

    def refresh_bytes(self):
        total = 0
        for path in self.root.rglob('*'):
            try:
                if path.is_file() and path.name != 'manifest.json':
                    total += path.stat().st_size
            except FileNotFoundError:
                continue  # Native database journals can disappear during census.
        if self.data:
            self.data['bytes'] = total
        return total

    def discard(self):
        # Verify EVERY resolved target before deleting anything, including the
        # generated SfM workspace. Never follow junctions/symlinks outside root.
        targets = list(self.root.rglob('*')) if self.root.exists() else []
        for path in targets:
            if path.is_symlink() or not path.resolve().is_relative_to(self.root):
                raise ValueError('Unexpected linked entry in capture directory; manual inspection required.')
        for path in sorted(targets, key=lambda p: len(p.parts), reverse=True):
            if path.is_dir():
                path.rmdir()
            else:
                path.unlink()
        self.data = None
