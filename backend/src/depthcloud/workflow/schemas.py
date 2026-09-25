import math
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class ROI(StrictModel):
    x: float = Field(ge=0, le=1)
    y: float = Field(ge=0, le=1)
    width: float = Field(gt=0, le=1)
    height: float = Field(gt=0, le=1)

    @model_validator(mode="after")
    def bounds(self):
        if self.x + self.width > 1.000001 or self.y + self.height > 1.000001:
            raise ValueError("ROI must stay inside the image.")
        return self

    def pixels(self, width, height):
        x0, y0 = int(self.x * width), int(self.y * height)
        x1 = min(width, round((self.x + self.width) * width))
        y1 = min(height, round((self.y + self.height) * height))
        if x1 - x0 < 4 or y1 - y0 < 4:
            raise ValueError("ROI must cover at least 4 by 4 native pixels.")
        return x0, y0, x1, y1


class CaptureView(StrictModel):
    snapshot_id: str = Field(min_length=1, max_length=64)
    roi: ROI
    label: str = Field(default="", max_length=64)
    replace_id: str | None = Field(default=None, max_length=64)


class ViewUpdate(StrictModel):
    selected: bool


class PreviewOptions(StrictModel):
    mode: Literal["rgb", "vivid", "mono", "chrome", "pixelated", "pseudo_ir", "grayscale"] = "rgb"
    saturation: float = Field(default=1.5, ge=0, le=3)
    contrast: float = Field(default=1.1, ge=0.2, le=3)
    brightness: float = Field(default=0, ge=-100, le=100)
    block_size: int = Field(default=12, ge=2, le=80)
    mono_threshold: int = Field(default=127, ge=0, le=255)
    ir_palette: Literal["inferno", "turbo"] = "inferno"


class Intrinsics(StrictModel):
    fx: float = Field(gt=0, le=100000)
    fy: float = Field(gt=0, le=100000)
    cx: float = Field(ge=0)
    cy: float = Field(ge=0)
    width: int = Field(gt=0, le=8192)
    height: int = Field(gt=0, le=8192)
    distortion: list[float] = Field(default_factory=lambda: [0, 0, 0, 0, 0], max_length=14)
    source: Literal["manual", "checkerboard", "depth_pro_estimate", "sfm_estimate"] = "manual"
    rms: float | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def valid_camera(self):
        if self.cx >= self.width or self.cy >= self.height:
            raise ValueError("Principal point must be inside the calibration image.")
        if len(self.distortion) not in (4, 5, 8, 12, 14):
            raise ValueError("Distortion requires 4, 5, 8, 12, or 14 coefficients.")
        return self

    def scaled(self, width, height):
        if not math.isclose(width / height, self.width / self.height, rel_tol=0.01):
            raise ValueError("Calibration aspect ratio differs from this observation. Recalibrate for the capture mode.")
        sx, sy = width / self.width, height / self.height
        return self.model_copy(update={"fx": self.fx * sx, "fy": self.fy * sy,
                                       "cx": self.cx * sx, "cy": self.cy * sy, "width": width, "height": height})


class PipelineOptions(StrictModel):
    threshold: float = Field(default=0.5, ge=0.05, le=0.99)
    require_detection: bool = False
    depth_min: float = Field(default=0.05, gt=0, le=100)
    depth_max: float = Field(default=10, gt=0, le=1000)
    stride: int = Field(default=2, ge=1, le=16)
    voxel_size: float = Field(default=0.005, ge=0.0005, le=0.1)
    icp_distance: float = Field(default=0.03, ge=0.001, le=0.5)
    icp_iterations: int = Field(default=60, ge=10, le=200)
    min_fitness: float = Field(default=0.35, ge=0.1, le=0.95)
    normal_radius: float = Field(default=0.02, ge=0.002, le=0.5)
    outlier_neighbors: int = Field(default=20, ge=5, le=100)
    outlier_std: float = Field(default=2, ge=0.5, le=5)
    depth_cleanup: bool = True
    depth_outlier_strength: float = Field(default=8, ge=2, le=20)
    create_mesh: bool = True
    mesh_radius_factor: float = Field(default=2.5, ge=1, le=10)

    @model_validator(mode="after")
    def valid_range(self):
        if self.depth_min >= self.depth_max:
            raise ValueError("Depth minimum must be less than maximum.")
        if self.normal_radius < self.voxel_size:
            raise ValueError("Normal radius must be at least the voxel size.")
        return self


class Checkerboard(StrictModel):
    columns: int = Field(default=9, ge=3, le=20)
    rows: int = Field(default=6, ge=3, le=20)
    square_size: float = Field(default=0.025, ge=0.001, le=1)



class LiveOptions(StrictModel):
    scan_mode: Literal["moving_camera", "rotating_object"] = "moving_camera"
    reference_ids: list[Annotated[str, Field(min_length=1, max_length=64)]] = Field(min_length=1, max_length=24)
    pipeline: PipelineOptions = Field(default_factory=PipelineOptions)
    interval_seconds: float = Field(default=.5, ge=.1, le=5)
    volume_resolution: int = Field(default=192, ge=64, le=256)
    color_filter: bool = True
    allow_geometry_tracking: bool = False
    min_feature_inliers: int = Field(default=8, ge=6, le=50)

    @model_validator(mode="after")
    def unique_references(self):
        if len(set(self.reference_ids)) != len(self.reference_ids):
            raise ValueError("Object references must be unique.")
        return self

class CaptureOptions(StrictModel):
    reconstruction_method: Literal["multiview", "depth_fusion"] = "multiview"
    reference_ids: list[Annotated[str, Field(min_length=1, max_length=64)]] = Field(min_length=1, max_length=24)
    target_frames: int = Field(default=1000, ge=3, le=1000)
    minimum_sharpness: float = Field(default=25, ge=5, le=200)
    maximum_motion: float = Field(default=80, ge=10, le=300)
    volume_resolution: int = Field(default=192, ge=64, le=256)
    pipeline: PipelineOptions = Field(default_factory=lambda: PipelineOptions(voxel_size=.002, normal_radius=.015, depth_outlier_strength=5))

    @model_validator(mode='after')
    def unique_references(self):
        if len(set(self.reference_ids)) != len(self.reference_ids):
            raise ValueError('Object references must be unique.')
        return self
