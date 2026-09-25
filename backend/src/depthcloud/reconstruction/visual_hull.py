"""One common shape constrained by calibrated silhouettes, never per-view meshes.

A visual hull is an outer-shape estimate: hidden concavities are not observable
from silhouettes. Camera reconstruction and shared scale are separate inputs.
"""

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter

from depthcloud.reconstruction.geometry import camera_matrix


def refine_silhouette(bgr, mask):
    """Recover foreground detail lost by detection's coarse color filter.

    Detection provides identity and seeds, not the final reconstruction edge.
    GrabCut can restore object-colored holes while retaining true apertures.
    Growth is confined to a small band around the detected object.
    """
    seed = np.where(mask > 0, 255, 0).astype(np.uint8)
    x, y, w, h = cv2.boundingRect(seed)
    if np.count_nonzero(seed) < 100:
        raise ValueError("Object mask is too small to refine.")
    pad = max(4, round(max(w, h) * 0.04))
    x0, y0, x1, y1 = (
        max(0, x - pad),
        max(0, y - pad),
        min(bgr.shape[1], x + w + pad),
        min(bgr.shape[0], y + h + pad),
    )
    image = np.ascontiguousarray(bgr[y0:y1, x0:x1])
    crop = seed[y0:y1, x0:x1]
    labels = np.full(crop.shape, cv2.GC_BGD, np.uint8)
    band = cv2.dilate(crop, np.ones((2 * pad + 1, 2 * pad + 1), np.uint8))
    labels[band > 0] = cv2.GC_PR_BGD
    contours, _ = cv2.findContours(crop, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros_like(crop)
    cv2.drawContours(filled, contours, -1, 255, cv2.FILLED)
    labels[filled > 0] = cv2.GC_PR_FGD
    core = cv2.erode(crop, np.ones((3, 3), np.uint8))
    labels[core > 0] = cv2.GC_FGD
    labels[[0, -1], :] = cv2.GC_BGD
    labels[:, [0, -1]] = cv2.GC_BGD
    cv2.grabCut(image, labels, None, np.zeros((1, 65)), np.zeros((1, 65)), 3, cv2.GC_INIT_WITH_MASK)
    selected = ((labels == cv2.GC_FGD) | (labels == cv2.GC_PR_FGD)).astype(np.uint8)
    count, components, stats, _ = cv2.connectedComponentsWithStats(selected)
    supported = [i for i in range(1, count) if np.count_nonzero((components == i) & (crop > 0)) >= 32]
    selected = np.isin(components, supported).astype(np.uint8) * 255
    ratio = np.count_nonzero(selected) / np.count_nonzero(crop)
    if not 0.75 <= ratio <= 1.5:
        raise ValueError("Refined foreground changed too much; inspect this detection mask.")
    clean = np.zeros_like(seed)
    clean[y0:y1, x0:x1] = selected
    filtered = np.full_like(bgr, 127)
    filtered[clean > 0] = bgr[clean > 0]
    return clean, filtered


def project(points, view):
    transform = view["world_to_camera"]
    camera = points @ transform[:3, :3].T + transform[:3, 3]
    intrinsics = view["intrinsics"]
    pixels, _ = cv2.projectPoints(
        camera, np.zeros(3), np.zeros(3), camera_matrix(intrinsics), np.asarray(intrinsics.distortion)
    )
    return pixels.reshape(-1, 2), camera[:, 2]


def object_bounds(views):
    directions, origins, heights = [], [], []
    for view in views:
        y, x = np.where(view["mask"] > 0)
        if len(x) < 100:
            raise ValueError("Object mask is too small for a shared shape.")
        intr = view["intrinsics"]
        pixel = np.array([[[np.mean(x), np.mean(y)]]], np.float64)
        ray = np.r_[cv2.undistortPoints(pixel, camera_matrix(intr), np.asarray(intr.distortion)).ravel(), 1.0]
        pose = np.linalg.inv(view["world_to_camera"])
        direction = pose[:3, :3] @ ray
        directions.append(direction / np.linalg.norm(direction))
        origins.append(pose[:3, 3])
        heights.append(float(np.ptp(y)) / intr.fy)
    directions, origins = np.asarray(directions), np.asarray(origins)
    perpendicular = np.eye(3)[None] - directions[:, :, None] * directions[:, None, :]
    matrix = perpendicular.sum(axis=0)
    if np.linalg.cond(matrix) > 500:
        raise ValueError(
            "Insufficient camera parallax to establish 3D thickness. Move around the object, not only toward it."
        )
    center = np.linalg.solve(matrix, np.einsum("nij,nj->i", perpendicular, origins))
    distances = np.einsum("ij,ij->i", center - origins, directions)
    if np.any(distances <= 0):
        raise ValueError("Camera rays do not establish a common visible object.")
    length = float(np.median(np.asarray(heights) * distances)) * 1.6
    if not np.isfinite(length) or length <= 0:
        raise ValueError("Invalid shared object bounds.")
    span = np.degrees(np.arccos(np.clip(directions @ directions.T, -1, 1))).max()
    return center - length / 2, length, float(span)


def silhouette_surface(views, resolution, update, check_cancel):
    import open3d as o3d

    origin, length, span = object_bounds(views)
    if span < 12:
        raise ValueError(
            f"Only {span:.1f} degrees of camera separation; insufficient side evidence for a 3D shape."
        )
    # Every calibrated view constrains the shape. A cluster of repeated front
    # views must never outvote a unique side silhouette.
    fields = []
    for view in views:
        mask = (view["mask"] > 0).astype(np.uint8)
        fields.append(
            cv2.distanceTransform(mask, cv2.DIST_L2, 3) - cv2.distanceTransform(1 - mask, cv2.DIST_L2, 3)
        )
    n = resolution
    spacing = length / (n - 1)
    volume = np.empty(n**3, np.float32)
    for begin in range(0, len(volume), 32768):
        check_cancel()
        indices = np.arange(begin, min(begin + 32768, len(volume)))
        points = origin + np.column_stack([indices % n, (indices // n) % n, indices // (n * n)]) * spacing
        smallest = np.full(len(indices), np.inf, np.float32)
        for view, field in zip(views, fields):
            pixels, z = project(points, view)
            x, y = pixels.T
            valid = (z > 0) & (x >= 0) & (y >= 0) & (x < field.shape[1] - 1) & (y < field.shape[0] - 1)
            distance = np.full(len(indices), -length, np.float32)
            # Sample continuous distance, preserving sub-pixel mask boundaries.
            px, py = x[valid], y[valid]
            ix, iy = np.floor(px).astype(int), np.floor(py).astype(int)
            wx, wy = px - ix, py - iy
            sampled = (
                (1 - wx) * (1 - wy) * field[iy, ix]
                + wx * (1 - wy) * field[iy, ix + 1]
                + (1 - wx) * wy * field[iy + 1, ix]
                + wx * wy * field[iy + 1, ix + 1]
            )
            distance[valid] = (sampled + 1.0) * z[valid] / view["intrinsics"].fx
            smallest = np.minimum(smallest, distance)
        volume[indices] = smallest
        update(
            "shape_constraints",
            "Solving one shared volume from calibrated silhouettes.",
            processed=int(indices[-1]) + 1,
            stage_total=n**3,
        )
    # Smooth the scalar field, not triangle vertices: vertex smoothing can
    # fold thin features through neighboring faces and create intersections.
    volume = gaussian_filter(volume.reshape((n, n, n)), sigma=0.65)

    if not np.any(volume > 0):
        raise ValueError("The calibrated silhouettes do not agree on a common shape; no mesh was published.")
    if any(
        np.any(face > 0)
        for face in (volume[0], volume[-1], volume[:, 0], volume[:, -1], volume[:, :, 0], volume[:, :, -1])
    ):
        raise ValueError("Shape reaches its search boundary; camera/mask consistency needs inspection.")
    mesh = o3d.t.geometry.TriangleMesh.create_isosurfaces(o3d.core.Tensor(volume)).to_legacy()
    mesh.vertices = o3d.utility.Vector3dVector(np.asarray(mesh.vertices) * spacing + origin)
    labels, counts, _ = mesh.cluster_connected_triangles()
    if not counts:
        raise ValueError("No supported shared surface was found.")
    largest = int(np.argmax(counts))
    mesh.remove_triangles_by_mask(np.asarray(labels) != largest)
    mesh.remove_unreferenced_vertices()
    # Flying Edges may leave near-coincident vertices at cell corners.
    # Weld at 0.2% of a voxel (micrometres here), far below surface resolution.
    mesh.merge_close_vertices(spacing * 0.002)
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_unreferenced_vertices()
    if len(mesh.triangles) > 200000:
        mesh = mesh.simplify_quadric_decimation(200000)
    if not mesh.is_watertight():
        raise ValueError(
            "The inferred surface has invalid topology. No self-intersecting mesh was published."
        )
    mesh.compute_vertex_normals()
    return mesh, {
        "watertight": True,
        "camera_span_deg": span,
        "voxel_size": spacing,
        "mask_consensus": 1.0,
    }


def silhouette_agreement(mesh, views, check_cancel):
    import open3d as o3d

    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
    results = []
    for view in views:
        check_cancel()
        intr = view["intrinsics"]
        scale = min(1.0, 320 / intr.width)
        width, height = round(intr.width * scale), round(intr.height * scale)
        matrix = camera_matrix(intr)
        target = cv2.undistort(view["mask"], matrix, np.asarray(intr.distortion))
        target = cv2.resize(target, (width, height), interpolation=cv2.INTER_NEAREST) > 0
        matrix[:2] *= scale
        rays = scene.create_rays_pinhole(matrix, view["world_to_camera"], width, height)
        predicted = np.isfinite(scene.cast_rays(rays)["t_hit"].numpy())
        union = np.count_nonzero(predicted | target)
        iou = float(np.count_nonzero(predicted & target) / max(1, union))
        results.append({"index": view["index"], "silhouette_iou": round(iou, 4)})
    return results


def color_surface(mesh, views, load_image, check_cancel):
    import open3d as o3d

    vertices, normals = np.asarray(mesh.vertices), np.asarray(mesh.vertex_normals)
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
    tolerance = max(float(np.linalg.norm(np.ptp(vertices, axis=0))) * 1e-4, 1e-7)
    colors = np.full((len(vertices), 3), 0.65)
    best = np.full(len(vertices), -np.inf)
    for view in views:
        check_cancel()
        pixels, z = project(vertices, view)
        x, y = np.rint(pixels).astype(int).T
        image = load_image(view["index"])
        image = cv2.resize(
            image, (view["intrinsics"].width, view["intrinsics"].height), interpolation=cv2.INTER_AREA
        )
        valid = (z > 0) & (x >= 0) & (y >= 0) & (x < image.shape[1]) & (y < image.shape[0])
        ids = np.where(valid)[0]
        valid[ids] &= view["mask"][y[ids], x[ids]] > 0
        camera = np.linalg.inv(view["world_to_camera"])[:3, 3]
        direction = camera - vertices
        distances = np.linalg.norm(direction, axis=1)
        direction /= np.maximum(distances[:, None], 1e-9)
        rays = np.column_stack([np.broadcast_to(camera, vertices.shape), -direction]).astype(np.float32)
        hit = scene.cast_rays(o3d.core.Tensor(rays))["t_hit"].numpy()
        valid &= hit >= distances - tolerance
        score = np.einsum("ij,ij->i", normals, direction)
        chosen = valid & (score > best) & (score > 0)
        colors[chosen] = image[y[chosen], x[chosen], ::-1] / 255
        best[chosen] = score[chosen]
    mesh.vertex_colors = o3d.utility.Vector3dVector(colors)
    return float(np.mean(np.isfinite(best)))
