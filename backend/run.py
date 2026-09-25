"""Launch from the embedded runtime: python.bat run.py."""
import multiprocessing
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

if __name__ == "__main__":
    multiprocessing.freeze_support()
    import uvicorn

    from depthcloud.config import Settings  # pyright: ignore[reportMissingImports]

    config = Settings()
    uvicorn.run("depthcloud.main:app", host=config.host, port=config.port, log_level="info")

