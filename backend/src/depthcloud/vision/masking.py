import cv2
import numpy as np


def choose_target(detections, roi_pixels, require_detection=False):
    """Select the detection most consistent with the explicit user ROI."""
    x0, y0, x1, y1 = roi_pixels
    roi_area = (x1 - x0) * (y1 - y0)
    matches = []
    for detection in detections:
        a, b, c, d = detection["box"]
        intersection = max(0, min(x1, c) - max(x0, a)) * max(0, min(y1, d) - max(y0, b))
        union = roi_area + max(0, c - a) * max(0, d - b) - intersection
        iou = intersection / max(1, union)
        # Both centers must agree. An edge-overlapping background detection must
        # not replace the user's central object with a different subject.
        centers_agree = (x0 <= (a + c) / 2 <= x1 and y0 <= (b + d) / 2 <= y1
                         and a <= (x0 + x1) / 2 <= c and b <= (y0 + y1) / 2 <= d)
        if iou >= 0.1 and centers_agree:
            matches.append((iou * detection["score"], detection))
    if not matches:
        if require_detection:
            raise ValueError("No DETR object above threshold matches this ROI. Adjust the ROI/threshold, or allow manual ROI fallback.")
        return None, roi_pixels, "No matching DETR class; foreground uses the explicit manual ROI."
    _, target = max(matches, key=lambda entry: entry[0])
    a, b, c, d = target["box"]
    # The selected ROI remains a hard exclusion boundary.
    box = (max(x0, round(a)), max(y0, round(b)), min(x1, round(c)), min(y1, round(d)))
    if box[2] - box[0] < 4 or box[3] - box[1] < 4:
        raise ValueError("Detected target is too small for foreground extraction.")
    warning = f"{len(matches)} overlapping detections; selected greatest ROI overlap/confidence." if len(matches) > 1 else None
    return target, box, warning


def foreground_mask(bgr, box):
    """GrabCut segmentation, separate from DETR; fail transparently if empty."""
    h, w = bgr.shape[:2]
    scale = min(1, 1024 / max(h, w))
    image = cv2.resize(bgr, (max(8, round(w * scale)), max(8, round(h * scale)))) if scale < 1 else bgr
    sh, sw = image.shape[:2]
    x0, y0, x1, y1 = [round(value * scale) for value in box]
    x0, y0 = max(1, x0), max(1, y0)
    x1, y1 = min(sw - 1, x1), min(sh - 1, y1)
    if x1 - x0 < 3 or y1 - y0 < 3:
        raise ValueError("Foreground ROI is too small.")
    labels = np.zeros((sh, sw), np.uint8)
    bg, fg = np.zeros((1, 65)), np.zeros((1, 65))
    cv2.grabCut(image, labels, (x0, y0, x1 - x0, y1 - y0), bg, fg, 5, cv2.GC_INIT_WITH_RECT)
    mask = np.where((labels == cv2.GC_FGD) | (labels == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    count, components, stats, _ = cv2.connectedComponentsWithStats(mask)
    if count < 2:
        raise ValueError("GrabCut found no foreground. Tighten the crop and use a contrasting background.")
    largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    mask = np.where(components == largest, 255, 0).astype(np.uint8)
    if scale < 1:
        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
    x0, y0, x1, y1 = box
    bounded = np.zeros_like(mask)
    bounded[y0:y1, x0:x1] = mask[y0:y1, x0:x1]
    if np.count_nonzero(bounded) < 32:
        raise ValueError("Insufficient foreground pixels for reconstruction.")
    return bounded


def depth_preview(depth, mask=None):
    valid = np.isfinite(depth) & (depth > 0)
    if mask is not None:
        valid &= mask > 0
    output = np.zeros((*depth.shape, 3), np.uint8)
    if valid.any():
        low, high = np.percentile(depth[valid], [2, 98])
        normalized = np.zeros(depth.shape, np.uint8)
        normalized[valid] = np.clip((depth[valid] - low) / max(1e-6, high - low) * 255, 0, 255).astype(np.uint8)
        color = cv2.applyColorMap(255 - normalized, cv2.COLORMAP_TURBO)
        output[valid] = color[valid]
    return output



def filter_mask_depth(depth, mask, strength=8):
    """Reject extreme masked depth tails using robust statistics, never alter raw depth.

    A minimum 5 cm / 10% median-depth tolerance prevents a flat surface's near-zero
    MAD from rejecting normal depth noise. Users can disable this heuristic.
    """
    valid = (mask > 0) & np.isfinite(depth) & (depth > 0)
    values = depth[valid]
    if len(values) < 32:
        raise ValueError("Too few finite foreground depth pixels.")
    median = float(np.median(values))
    sigma = 1.4826 * float(np.median(np.abs(values - median)))
    tolerance = max(.05, median * .1, strength * sigma)
    keep = valid & (np.abs(depth - median) <= tolerance)
    cleaned = np.where(keep, 255, 0).astype(np.uint8)
    if np.count_nonzero(cleaned) < 32:
        raise ValueError("Depth cleanup removed the usable foreground. Disable cleanup or adjust the ROI.")
    return cleaned, int(np.count_nonzero(mask) - np.count_nonzero(cleaned))

