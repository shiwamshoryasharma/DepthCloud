import sys
from types import SimpleNamespace

from depthcloud.config import Settings
from depthcloud.models.manager import ModelManager


def test_cpu_fallback_and_no_model_loading(monkeypatch, tmp_path):
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False)))
    manager = ModelManager(Settings(model_directory=tmp_path))
    manager.initialize_device()
    status = manager.status()
    assert status["device"]["type"] == "cpu"
    assert status["device"]["cuda_available"] is False
    assert all(not model["loaded"] for model in status["models"].values())
    assert all(not model["files_present"] for model in status["models"].values())


def test_local_model_inventory_distinguishes_files_from_loaded_models(tmp_path):
    (tmp_path / "depth_pro.pt").write_bytes(b"test inventory only")
    manager = ModelManager(Settings(model_directory=tmp_path))
    depth = manager.status()["models"]["depth_pro"]
    assert depth["files_present"]
    assert not depth["loaded"]
    assert not depth["inference_enabled"]

