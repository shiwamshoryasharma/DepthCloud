from depthcloud.reconstruction.geometry import CAMERA_TO_ISAAC, CAMERA_TO_THREE


def descriptor(state):
    result = state["result"]
    return {"integration": "isaac_sim", "connection": "not_connected", "transport": "local HTTP export",
            "session_id": state["session_id"], "available": result is not None,
            "units": "meters", "source_frame": "+X right, +Y down, +Z forward (reference camera)",
            "isaac_frame": "+X forward, +Y left, +Z up", "camera_to_isaac": CAMERA_TO_ISAAC.tolist(),
            "camera_to_three": CAMERA_TO_THREE.tolist(),
            "dimensions_m": result["dimensions_m"] if result else None,
            "object_to_reference": [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
            "exports": {"point_cloud": "/api/reconstruction/export/ply" if result else None,
                        "mesh": "/api/reconstruction/export/obj" if result and result["triangles"] else None},
            "note": "Apply camera_to_isaac once when importing exported geometry. No simulator connection or robot control is performed."}

