import cv2
import numpy as np

from depthcloud.workflow.schemas import PreviewOptions


def transform_preview(bgr: np.ndarray, options: PreviewOptions):
    """Display-only processing. Source BGR remains unchanged for inference."""
    if options.mode == "rgb":
        return bgr
    if options.mode == "vivid":
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        hsv[:, :, 1] = np.clip(hsv[:, :, 1].astype(np.float32) * options.saturation, 0, 255).astype(np.uint8)
        color = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        return np.clip(color.astype(np.float32) * options.contrast + options.brightness, 0, 255).astype(np.uint8)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    if options.mode == "grayscale":
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    if options.mode == "mono":
        _, binary = cv2.threshold(gray, options.mono_threshold, 255, cv2.THRESH_BINARY)
        return cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
    if options.mode == "pixelated":
        h, w = bgr.shape[:2]
        small = cv2.resize(bgr, (max(1, w // options.block_size), max(1, h // options.block_size)), interpolation=cv2.INTER_AREA)
        return cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
    if options.mode == "pseudo_ir":
        palette = cv2.COLORMAP_INFERNO if options.ir_palette == "inferno" else cv2.COLORMAP_TURBO
        return cv2.applyColorMap(gray, palette)
    # Metallic inspection effect, not an inferred material classification.
    contrast = cv2.createCLAHE(clipLimit=2, tileGridSize=(8, 8)).apply(gray)
    return cv2.applyColorMap(contrast, cv2.COLORMAP_BONE)

