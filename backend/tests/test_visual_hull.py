import numpy as np
import open3d as o3d
import pytest

from depthcloud.reconstruction.visual_hull import object_bounds, silhouette_agreement, silhouette_surface
from depthcloud.workflow.schemas import Intrinsics


def sphere_views():
    intr = Intrinsics(fx=100, fy=100, cx=63.5, cy=47.5, width=128, height=96)
    mesh = o3d.geometry.TriangleMesh.create_sphere(0.1)
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
    result = []
    for i, angle in enumerate(np.linspace(0, 2 * np.pi, 8, endpoint=False)):
        position = np.array([0.5 * np.sin(angle), 0.08, 0.5 * np.cos(angle)])
        forward = -position / np.linalg.norm(position)
        right = np.cross(forward, [0, -1, 0])
        right /= np.linalg.norm(right)
        down = np.cross(forward, right)
        pose = np.eye(4)
        pose[:3, :3] = np.array([right, down, forward])
        pose[:3, 3] = -pose[:3, :3] @ position
        rays = scene.create_rays_pinhole(
            np.array([[100, 0, 63.5], [0, 100, 47.5], [0, 0, 1.0]]), pose, 128, 96
        )
        mask = np.isfinite(scene.cast_rays(rays)["t_hit"].numpy()).astype(np.uint8) * 255
        result.append({"index": i, "intrinsics": intr, "world_to_camera": pose, "mask": mask})
    return result


def test_sides_form_one_shared_shape_instead_of_stacking_front_surfaces():
    views = sphere_views()
    mesh, quality = silhouette_surface(views, 48, lambda *a, **k: None, lambda: None)
    extents = np.ptp(np.asarray(mesh.vertices), axis=0)
    assert np.all((extents > 0.18) & (extents < 0.24)), extents
    assert quality["camera_span_deg"] > 140
    assert min(m["silhouette_iou"] for m in silhouette_agreement(mesh, views, lambda: None)) > 0.8


def test_same_camera_cannot_fabricate_3d_thickness():
    views = sphere_views()
    with pytest.raises(ValueError, match="parallax"):
        object_bounds([views[0]] * 6)


def test_contradictory_views_must_fail_surface_validation():
    views = sphere_views()
    mesh, _ = silhouette_surface(views, 48, lambda *a, **k: None, lambda: None)
    wrong = sphere_views()
    for view in wrong:
        view["world_to_camera"][:3, 3] += np.array([0.25, 0, 0])
    assert np.median([m["silhouette_iou"] for m in silhouette_agreement(mesh, wrong, lambda: None)]) < 0.15


def test_repeated_front_views_cannot_vote_away_unique_side_constraints():
    views = sphere_views()
    mesh, quality = silhouette_surface(views + [views[0]] * 80, 40, lambda *a, **k: None, lambda: None)
    assert np.all(np.ptp(np.asarray(mesh.vertices), axis=0) < 0.25)
    assert quality["mask_consensus"] == 1.0


def test_color_projection_does_not_paint_occluded_back_surface():
    from depthcloud.reconstruction.visual_hull import color_surface

    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(
        [
            [-0.4, -0.4, 1],
            [0, 0.4, 1],
            [0.4, -0.4, 1],
            [-0.2, -0.2, 2],
            [0, 0.2, 2],
            [0.2, -0.2, 2],
        ]
    )
    mesh.triangles = o3d.utility.Vector3iVector([[0, 1, 2], [3, 4, 5]])
    mesh.compute_vertex_normals()
    intr = Intrinsics(fx=100, fy=100, cx=63.5, cy=47.5, width=128, height=96)
    views = [
        {
            "index": 0,
            "intrinsics": intr,
            "world_to_camera": np.eye(4),
            "mask": np.full((96, 128), 255, np.uint8),
        }
    ]
    image = np.zeros((96, 128, 3), np.uint8)
    image[:, :, 0] = 255
    color_surface(mesh, views, lambda i: image, lambda: None)
    colors = np.asarray(mesh.vertex_colors)
    np.testing.assert_allclose(colors[3:], 0.65)
    assert colors[:3, 2].min() > 0.9


def test_silhouette_refinement_repairs_color_filter_holes_but_keeps_real_apertures():
    from depthcloud.reconstruction.visual_hull import refine_silhouette

    image = np.full((120, 160, 3), 210, np.uint8)
    image[20:100, 40:120] = [220, 50, 20]
    mask = np.zeros((120, 160), np.uint8)
    mask[20:100, 40:120] = 255
    mask[40:55, 60:75] = 0  # Object pixels incorrectly removed by color filtering.
    image[65:80, 85:100] = 210
    mask[65:80, 85:100] = 0  # Actual background aperture.
    refined, _ = refine_silhouette(image, mask)
    assert refined[47, 67] == 255
    assert refined[72, 92] == 0
    assert refined[5, 5] == 0
