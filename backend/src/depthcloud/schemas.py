from typing import Literal

from pydantic import BaseModel, Field


class CameraStart(BaseModel):
    index: int = Field(default=0, ge=0, le=32)
    width: int = Field(default=1280, ge=160, le=3840)
    height: int = Field(default=720, ge=120, le=2160)
    fps: int = Field(default=30, ge=1, le=120)
    backend: Literal["auto", "dshow", "msmf", "any"] = "auto"


class BufferSettings(BaseModel):
    capacity: int = Field(default=10, ge=1, le=128)
    automatic: bool = False

