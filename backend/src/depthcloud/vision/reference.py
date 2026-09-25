"""Reference appearance excludes static background during object rotation."""

import cv2
import numpy as np


class ReferenceAppearance:
    def __init__(self, bgr, mask):
        if np.count_nonzero(mask) < 32:
            raise ValueError("Reference has too little foreground. Tighten the crop.")
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        self.histogram = cv2.calcHist([hsv], [0, 1], mask, [36, 16], [0, 180, 0, 256])
        # Smooth neighboring bins to tolerate exposure and color noise.
        self.histogram = cv2.GaussianBlur(self.histogram, (3, 3), 0.7)
        cv2.normalize(self.histogram, self.histogram, 0, 255, cv2.NORM_MINMAX)

    def segment(self, bgr, box):
        h, w = bgr.shape[:2]
        scale = min(1.0, 512 / max(h, w))
        image = cv2.resize(bgr, (round(w * scale), round(h * scale))) if scale < 1 else bgr
        sh, sw = image.shape[:2]
        x0, y0, x1, y1 = [round(v * scale) for v in box]
        x0, y0, x1, y1 = max(1, x0), max(1, y0), min(sw - 1, x1), min(sh - 1, y1)
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        probability = cv2.calcBackProject([hsv], [0, 1], self.histogram, [0, 180, 0, 256], 1)
        allowed = np.zeros((sh, sw), np.uint8)
        allowed[y0:y1, x0:x1] = 255
        candidate = ((probability > 16) & (allowed > 0)).astype(np.uint8) * 255
        candidate = cv2.morphologyEx(candidate, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        count, labels, stats, centers = cv2.connectedComponentsWithStats(candidate)
        if count < 2:
            raise ValueError(
                "Reference colors were lost. Keep the same object in the crop with steady lighting."
            )
        center = np.array([(x0 + x1) / 2, (y0 + y1) / 2])
        scores = stats[1:, cv2.CC_STAT_AREA] / (
            1 + np.linalg.norm(centers[1:] - center, axis=1) / max(4, x1 - x0)
        )
        component = 1 + int(np.argmax(scores))
        seed = (labels == component).astype(np.uint8) * 255
        gc = np.zeros((sh, sw), np.uint8)
        gc[allowed > 0] = cv2.GC_PR_BGD
        gc[seed > 0] = cv2.GC_PR_FGD
        core = cv2.erode(seed, np.ones((3, 3), np.uint8))
        gc[core > 0] = cv2.GC_FGD
        if np.count_nonzero(core) < 16:
            raise ValueError("Reference foreground is too small or occluded.")
        cv2.grabCut(image, gc, None, np.zeros((1, 65)), np.zeros((1, 65)), 2, cv2.GC_INIT_WITH_MASK)
        mask = np.where(
            ((gc == cv2.GC_FGD) | (gc == cv2.GC_PR_FGD)) & (probability > 8) & (allowed > 0), 255, 0
        ).astype(np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
        if scale < 1:
            mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
        if np.count_nonzero(mask) < 64:
            raise ValueError("Object mask is too small. Keep hands and background outside the selection.")
        return mask


class ReferenceMatch:
    def __init__(self, entry, score, mask, box):
        self.reference_id = entry['id']
        self.label = entry['label']
        self.score = float(score)
        self.mask = mask
        self.box = box
        self.method = 'foreground color and silhouette reference matching'

    def metadata(self):
        return {'reference_id': self.reference_id, 'label': self.label,
                'score': round(self.score, 3), 'box': list(self.box),
                'method': self.method}


def _silhouette(mask):
    x, y, w, h = cv2.boundingRect(mask)
    if w < 3 or h < 3:
        raise ValueError('Reference foreground is too small.')
    normalized = cv2.resize(mask[y:y + h, x:x + w], (48, 64), interpolation=cv2.INTER_NEAREST) > 0
    return normalized, w / h


def _box_iou(a, b):
    area = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))
    return area / max(1, (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - area)


class ReferenceBank:
    """Bounded full-frame appearance search against all angles of ONE object.

    Reference coordinates never constrain the search. These are appearance
    exemplars, not training data or posed reconstruction observations. Color and
    silhouette cannot establish identity between visually identical instances;
    ambiguous candidates are rejected, never chosen by area or reference order.
    """

    def __init__(self, describe=None):
        self.entries = []
        self.describe = describe

    def add(self, reference_id, label, bgr, mask):
        if len(self.entries) >= 24:
            raise ValueError('At most 24 object references are supported.')
        appearance = ReferenceAppearance(bgr, mask)
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        # Downweight colors common outside the selected foreground. Raw crop
        # histograms can otherwise learn the hand/table instead of the object.
        foreground = cv2.calcHist([hsv], [0, 1], mask, [36, 16], [0, 180, 0, 256])
        background_mask = cv2.bitwise_not(cv2.dilate(mask, np.ones((5, 5), np.uint8)))
        background = cv2.calcHist([hsv], [0, 1], background_mask, [36, 16], [0, 180, 0, 256])
        foreground = cv2.GaussianBlur(foreground, (3, 3), .7)
        background = cv2.GaussianBlur(background, (3, 3), .7)
        foreground /= max(1., float(foreground.sum()))
        background /= max(1., float(background.sum()))
        contrast = foreground / (foreground + 2 * background + 1e-6)
        contrast *= np.sqrt(foreground / max(1e-6, float(foreground.max())))
        cv2.normalize(contrast, contrast, 0, 255, cv2.NORM_MINMAX)
        appearance.histogram = contrast
        probability = cv2.calcBackProject([hsv], [0, 1], contrast, [0, 180, 0, 256], 1)
        filtered = ((mask > 0) & (probability > 40)).astype(np.uint8) * 255
        filtered = cv2.morphologyEx(filtered, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        _, labels, stats, _ = cv2.connectedComponentsWithStats(filtered)
        if len(stats) < 2:
            raise ValueError('Reference has no distinctive foreground. Use a contrasting background.')
        component = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        mask = np.where((labels == component) & (mask > 0), 255, 0).astype(np.uint8)
        if np.count_nonzero(mask) < 64:
            raise ValueError('Reference has too little distinctive foreground.')
        # When one saturated hue clearly dominates, exclude disconnected
        # neutral/skin contamination that GrabCut attached through the hand.
        hue_hist = cv2.calcHist([hsv], [0], ((mask > 0) & (hsv[:, :, 1] > 45)).astype(np.uint8), [36], [0, 180]).ravel()
        dominant = int(np.argmax(hue_hist)) * 5 + 2
        delta = np.abs(hsv[:, :, 0].astype(float) - dominant)
        similar = (np.minimum(delta, 180 - delta) < 18) & (hsv[:, :, 1] > 40) & (mask > 0)
        if np.count_nonzero(similar) > np.count_nonzero(mask) * .65:
            mask = similar.astype(np.uint8) * 255
            appearance = ReferenceAppearance(bgr, mask)
        silhouette, aspect = _silhouette(mask)
        descriptor = self.describe([_object_crop(bgr, mask)])[0] if self.describe else None
        values = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)[:, :, 2][mask > 0]
        low, high = np.percentile(values, [2, 98])
        self.entries.append({'id': reference_id, 'label': label, 'appearance': appearance,
                             'silhouette': silhouette, 'aspect': aspect, 'descriptor': descriptor,
                             'area_fraction': np.count_nonzero(mask) / mask.size,
                             'value_range': (max(0, low - 55), min(255, high + 55))})
        return mask

    def locate(self, bgr, prior_box=None):
        h, w = bgr.shape[:2]
        scale = min(1., 640 / max(h, w))
        image = cv2.resize(bgr, (round(w * scale), round(h * scale))) if scale < 1 else bgr
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        sh, sw = image.shape[:2]
        candidates = []
        kernel = np.ones((3, 3), np.uint8)
        for entry in self.entries:
            probability = cv2.calcBackProject([hsv], [0, 1], entry['appearance'].histogram, [0, 180, 0, 256], 1)
            low, high = entry['value_range']
            binary = ((probability > 40) & (hsv[:, :, 2] >= low) & (hsv[:, :, 2] <= high)).astype(np.uint8) * 255
            binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
            binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
            count, labels, stats, _ = cv2.connectedComponentsWithStats(binary)
            # Bounded work even with highly fragmented backgrounds.
            indices = sorted(range(1, count), key=lambda i: int(stats[i, cv2.CC_STAT_AREA]), reverse=True)[:32]
            for index in indices:
                x, y, bw, bh, area = map(int, stats[index])
                if area < max(64, entry['area_fraction'] * sh * sw * .12) or x <= 1 or y <= 1 or x + bw >= sw - 1 or y + bh >= sh - 1:
                    continue  # cropped objects cannot provide a complete silhouette
                component = np.where(labels[y:y + bh, x:x + bw] == index, 255, 0).astype(np.uint8)
                silhouette, aspect = _silhouette(component)
                agreement = np.count_nonzero(silhouette & entry['silhouette']) / max(1, np.count_nonzero(silhouette | entry['silhouette']))
                ratio = min(aspect / entry['aspect'], entry['aspect'] / aspect)
                if ratio < .55 or agreement < .50:
                    continue
                score = .65 * agreement + .35 * ratio
                if score < (.64 if self.describe else .74):
                    continue
                box = (x, y, x + bw, y + bh)
                if prior_box is not None:
                    px0, py0, px1, py1 = np.asarray(prior_box) * scale
                    overlap = max(0, min(x + bw, px1) - max(x, px0)) * max(0, min(y + bh, py1) - max(y, py0))
                    if overlap / (bw * bh) < .8:
                        continue
                candidates.append((score, entry, box, component))
        if not candidates:
            raise ValueError('No reference matches a complete object in this frame. Show an included angle with a clear silhouette.')
        if self.describe:
            # Batch candidates once, deduplicating proposals from overlapping
            # reference histograms. Score each crop against EVERY learned exemplar.
            unique = []
            for candidate in sorted(candidates, key=lambda item: item[0], reverse=True):
                if not any(_box_iou(candidate[2], item[2]) > .8 for item in unique):
                    unique.append(candidate)
            if len(unique) > 24:
                raise ValueError('Object identity is ambiguous: too many foreground candidates. Simplify the background.')
            crops = []
            for _, _, (x0, y0, x1, y1), component in unique:
                crops.append(_object_crop(image[y0:y1, x0:x1], component))
            descriptors = self.describe(crops)
            verified = []
            for candidate, descriptor in zip(unique, descriptors):
                score, _, box, component = candidate
                similarities = [float(descriptor @ entry['descriptor']) for entry in self.entries]
                index = int(np.argmax(similarities))
                similarity = similarities[index]
                # A validated camera pose and known world object volume provide
                # independent association evidence for appearance changes.
                if similarity >= (.76 if prior_box is not None else .82):
                    verified.append((.25 * score + .75 * similarity, self.entries[index], box, component))
            candidates = verified
            if not candidates:
                raise ValueError('No reference matches the visual appearance of these candidates. Add a clear example of this angle.')
        candidates.sort(key=lambda item: item[0], reverse=True)
        best = candidates[0]
        # Several angles matching the SAME candidate are normal. Distinct objects
        # with comparable appearance are not safe to fuse.
        if any(item[0] >= best[0] - (.04 if self.describe else .08) and _box_iou(item[2], best[2]) < .3 for item in candidates[1:]):
            raise ValueError('Object identity is ambiguous: multiple objects match the reference set. Isolate the target.')
        score, entry, box, component = best
        x0, y0, x1, y1 = box
        mask = np.zeros((sh, sw), np.uint8)
        mask[y0:y1, x0:x1] = component
        if scale < 1:
            mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
        x, y, bw, bh = cv2.boundingRect(mask)
        match = ReferenceMatch(entry, score, mask, (x, y, x + bw, y + bh))
        if self.describe:
            match.method = 'local DETR visual features, color and silhouette'
        return match


def _object_crop(bgr, mask):
    x, y, w, h = cv2.boundingRect(mask)
    crop = bgr[y:y + h, x:x + w].copy()
    crop[mask[y:y + h, x:x + w] == 0] = 127
    # Preserve aspect ratio on a neutral square, identical for both pathways.
    side = max(w, h)
    output = np.full((side, side, 3), 127, np.uint8)
    ox, oy = (side - w) // 2, (side - h) // 2
    output[oy:oy + h, ox:ox + w] = crop
    return cv2.resize(output, (224, 224))
