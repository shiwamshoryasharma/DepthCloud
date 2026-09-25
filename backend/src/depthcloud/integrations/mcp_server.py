"""Read-only MCP tools proxy the running local API; no second camera/model owner."""
import httpx
from mcp.server import MCPServer
from mcp.types import ToolAnnotations

from depthcloud.config import Settings

server = MCPServer("DepthCloud")
read_only = ToolAnnotations(read_only_hint=True, destructive_hint=False, idempotent_hint=True, open_world_hint=False)
base_url = f"http://127.0.0.1:{Settings().port}"


def read(path):
    with httpx.Client(base_url=base_url, timeout=5, trust_env=False) as client:
        response = client.get(path)
        response.raise_for_status()
        return response.json()


@server.tool(annotations=read_only)
def get_system_status() -> dict:
    """Read actual runtime, device, model, and memory status."""
    return read("/api/system")


@server.tool(annotations=read_only)
def get_camera_status() -> dict:
    """Read live camera status without opening or changing the camera."""
    return read("/api/camera/status")


@server.tool(annotations=read_only)
def get_reconstruction_status() -> dict:
    """Read active job state, progress, views, and reconstruction quality."""
    return read("/api/reconstruction/state")


@server.tool(annotations=read_only)
def get_reconstruction_metadata() -> dict:
    """Read current reconstruction units, bounds, transforms, and confidence."""
    return read("/api/reconstruction/state").get("result") or {"available": False}


@server.tool(annotations=read_only)
def get_latest_point_cloud_info() -> dict:
    """Read point count, units and local PLY download path."""
    result = read("/api/reconstruction/state").get("result")
    return {"available": bool(result), "points": result["points"] if result else 0,
            "units": "meters", "export_path": "/api/reconstruction/export/ply" if result else None}


@server.tool(annotations=read_only)
def get_latest_mesh_info() -> dict:
    """Read mesh availability and local OBJ download path."""
    result = read("/api/reconstruction/state").get("result")
    available = bool(result and result["triangles"])
    return {"available": available, "triangles": result["triangles"] if result else 0,
            "export_path": "/api/reconstruction/export/obj" if available else None}


@server.tool(annotations=read_only)
def get_session_info() -> dict:
    """Read the selected-view session summary and bounded memory usage."""
    state = read("/api/reconstruction/state")
    return {key: state[key] for key in ("session_id", "views", "memory_bytes", "memory_limit", "max_views")}


@server.tool(annotations=read_only)
def get_isaac_transfer_metadata() -> dict:
    """Read the explicit reference-camera to Isaac coordinate conversion/export contract."""
    return read("/api/integration/isaac")

