from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

BACKEND_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="DEPTHCLOUD_", env_file=BACKEND_ROOT / ".env", extra="ignore"
    )
    host: str = "127.0.0.1"
    port: int = Field(default=8765, ge=1024, le=65535)
    buffer_capacity: int = Field(default=10, ge=1, le=128)
    maximum_buffer_size: int = Field(default=128, ge=1, le=128)
    buffer_memory_mb: int = Field(default=128, ge=16, le=512)
    preview_fps: int = Field(default=24, ge=1, le=60)
    preview_width: int = Field(default=1280, ge=320, le=1920)
    jpeg_quality: int = Field(default=82, ge=40, le=95)
    camera_timeout: float = Field(default=8.0, ge=0.5, le=30)
    startup_timeout: float = Field(default=15.0, ge=1, le=60)
    worker_join_timeout: float = Field(default=1.5, ge=0.1, le=5)
    discovery_max_index: int = Field(default=5, ge=0, le=16)
    discovery_timeout: float = Field(default=3.0, ge=0.5, le=5)
    max_stream_clients: int = Field(default=4, ge=1, le=8)
    inference_cpu_threads: int = Field(default=8, ge=1, le=32)
    max_saved_views: int = Field(default=24, ge=1, le=48)
    session_memory_mb: int = Field(default=512, ge=64, le=2048)
    model_directory: Path = BACKEND_ROOT / ".models"
    allowed_origins: list[str] = [
        "http://localhost:5173", "http://127.0.0.1:5173",
        "http://localhost:8765", "http://127.0.0.1:8765",
    ]


