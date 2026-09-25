import gc
import logging
import threading
import time
from dataclasses import replace

import cv2
import numpy as np

from depthcloud.config import Settings

logger = logging.getLogger("depthcloud.models")


class ModelManager:
    """Lazy local models, serialized inference, nonblocking status inventory."""
    def __init__(self, settings: Settings):
        self.settings = settings
        self.device = {"type": "cpu", "name": "CPU", "cuda_available": False, "error": None}
        self._lock = threading.RLock()
        self._inference_lock = threading.RLock()
        self._initialized = False
        self._depth = self._transform = self._detector = self._processor = None
        self._states = {name: {"loaded": False, "loading": False, "error": None, "latency_ms": None}
                        for name in ("depth_pro", "detr")}

    def initialize_device(self):
        with self._lock:
            if self._initialized:
                return
            try:
                import torch

                available = torch.cuda.is_available()
                self.device = {"type": "cuda" if available else "cpu",
                               "name": torch.cuda.get_device_name(0) if available else "CPU",
                               "cuda_available": available, "error": None}
            except (ImportError, RuntimeError, OSError) as exc:
                logger.warning("CUDA probe failed; using CPU: %s", exc)
                self.device["error"] = str(exc)
            self._initialized = True

    def _state(self, name, **changes):
        with self._lock:
            self._states[name].update(changes)

    def _load_depth(self):
        if self._depth is not None:
            return
        import depth_pro
        import torch
        from depth_pro.depth_pro import DEFAULT_MONODEPTH_CONFIG_DICT
        path = self.settings.model_directory / "depth_pro.pt"
        if not path.is_file():
            raise FileNotFoundError(f"Local Depth Pro checkpoint is missing: {path}")
        self._state("depth_pro", loading=True, error=None)
        try:
            self.initialize_device()
            import torch
            torch.set_num_threads(self.settings.inference_cpu_threads)
            config = replace(DEFAULT_MONODEPTH_CONFIG_DICT, checkpoint_uri=str(path))
            precision = torch.float16 if self.device["cuda_available"] else torch.float32
            model, transform = depth_pro.create_model_and_transforms(
                config=config, device=torch.device(self.device["type"]), precision=precision)
            model.eval()
            self._depth, self._transform = model, transform
            self._state("depth_pro", loaded=True)
            logger.info("Loaded local Depth Pro on %s", self.device["type"])
        except Exception as exc:
            self._state("depth_pro", error=str(exc))
            raise RuntimeError(f"Depth Pro could not load: {exc}") from exc
        finally:
            self._state("depth_pro", loading=False)

    def _load_detector(self):
        if self._detector is not None:
            return
        from transformers import DetrForObjectDetection, DetrImageProcessor
        directory = self.settings.model_directory / "detr-resnet-50"
        self._state("detr", loading=True, error=None)
        try:
            self.initialize_device()
            import torch
            torch.set_num_threads(self.settings.inference_cpu_threads)
            processor = DetrImageProcessor.from_pretrained(str(directory), local_files_only=True)
            model, info = DetrForObjectDetection.from_pretrained(
                str(directory), local_files_only=True, output_loading_info=True)
            missing = [key for key in info.get("missing_keys", []) if not key.endswith("num_batches_tracked")]
            unexpected = [key for key in info.get("unexpected_keys", []) if not key.endswith("num_batches_tracked")]
            if missing or unexpected or info.get("mismatched_keys"):
                raise RuntimeError(f"Local DETR weights do not match the model: missing={missing[:5]}")
            model.to(self.device["type"]).eval()
            self._detector, self._processor = model, processor
            self._state("detr", loaded=True)
            logger.info("Loaded local DETR on %s", self.device["type"])
        except Exception as exc:
            self._state("detr", error=str(exc))
            raise RuntimeError(f"DETR could not load locally: {exc}") from exc
        finally:
            self._state("detr", loading=False)

    def infer_depth(self, bgr, focal_px=None):
        import torch
        with self._inference_lock:
            self._load_depth()
            began = time.perf_counter()
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            with torch.inference_mode():
                focal = torch.tensor(float(focal_px), device=self.device["type"]) if focal_px is not None else None
                prediction = self._depth.infer(self._transform(rgb), f_px=focal)
                depth = prediction["depth"].float().cpu().numpy()
                focal = float(prediction["focallength_px"].item())
            if depth.shape != bgr.shape[:2] or not np.isfinite(focal) or focal <= 0:
                raise RuntimeError("Depth Pro returned invalid dimensions or focal length.")
            self._state("depth_pro", latency_ms=round((time.perf_counter() - began) * 1000, 1))
            return depth, focal

    def detect(self, bgr, threshold):
        import torch
        with self._inference_lock:
            self._load_detector()
            began = time.perf_counter()
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            inputs = self._processor(images=rgb, return_tensors="pt").to(self.device["type"])
            with torch.inference_mode():
                outputs = self._detector(**inputs)
                target = torch.tensor([bgr.shape[:2]], device=self.device["type"])
                results = self._processor.post_process_object_detection(outputs, threshold=threshold, target_sizes=target)[0]
            detections = [{"label": self._detector.config.id2label.get(int(label), str(int(label))),
                           "score": round(float(score), 4), "box": box.float().cpu().tolist()}
                          for score, label, box in zip(results["scores"], results["labels"], results["boxes"])]
            self._state("detr", latency_ms=round((time.perf_counter() - began) * 1000, 1))
            return detections

    def describe_objects(self, crops):
        """Normalized visual descriptors from the existing local DETR backbone.

        No downloads or weight updates. Crop preparation is identical for saved
        exemplars and current candidates; full-frame RGB depth remains separate.
        """
        import torch
        import torch.nn.functional as functional

        if not crops or len(crops) > 24:
            raise ValueError("Descriptor batches must contain 1 to 24 object crops.")
        with self._inference_lock:
            self._load_detector()
            rgb = np.stack([cv2.cvtColor(cv2.resize(crop, (224, 224)), cv2.COLOR_BGR2RGB) for crop in crops])
            pixels = torch.from_numpy(rgb).to(self.device["type"], dtype=torch.float32).permute(0, 3, 1, 2) / 255
            mean = pixels.new_tensor(self._processor.image_mean)[None, :, None, None]
            std = pixels.new_tensor(self._processor.image_std)[None, :, None, None]
            pixels = (pixels - mean) / std
            mask = torch.ones((len(crops), 224, 224), dtype=torch.bool, device=pixels.device)
            with torch.inference_mode():
                maps = self._detector.model.backbone(pixels, mask)
                pooled = maps[-1][0].mean(dim=(-2, -1))
                return functional.normalize(pooled, dim=1).float().cpu().numpy()

    def release(self):
        with self._inference_lock:
            self._depth = self._transform = self._detector = self._processor = None
            for name in self._states:
                self._state(name, loaded=False)
            gc.collect()
            if self.device["cuda_available"]:
                import torch
                torch.cuda.empty_cache()

    def status(self):
        directory = self.settings.model_directory
        assets = {"depth_pro": [directory / "depth_pro.pt"],
                  "detr": [directory / "detr-resnet-50" / name
                           for name in ("config.json", "preprocessor_config.json", "model.safetensors")]}
        with self._lock:
            states = {name: dict(state) for name, state in self._states.items()}
            device = dict(self.device)
        allocated = reserved = 0
        if device["cuda_available"]:
            import torch
            allocated = torch.cuda.memory_allocated() / 1024**2
            reserved = torch.cuda.memory_reserved() / 1024**2
        return {"device": device, "models": {
            name: {"files_present": all(p.is_file() for p in paths), **states[name],
                   "inference_enabled": states[name]["loaded"],
                   "bytes": sum(p.stat().st_size for p in paths if p.is_file())}
            for name, paths in assets.items()},
            "vram_allocated_mb": round(allocated, 1), "vram_reserved_mb": round(reserved, 1),
            "phase": "reconstruction"}


