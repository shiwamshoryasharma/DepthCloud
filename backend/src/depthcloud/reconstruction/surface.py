import numpy as np


def fuse_clouds(clouds, options):
    import open3d as o3d
    combined = o3d.geometry.PointCloud()
    for cloud in clouds:
        combined += cloud
    combined = combined.voxel_down_sample(options.voxel_size)
    if len(combined.points) >= options.outlier_neighbors + 2:
        combined, _ = combined.remove_statistical_outlier(options.outlier_neighbors, options.outlier_std)
    if len(combined.points) < 30:
        raise ValueError("Too few valid points remain after filtering. Adjust depth range, ROI or voxel size.")
    if len(combined.points) > 200_000:
        combined = combined.select_by_index(np.linspace(0, len(combined.points) - 1, 200_000, dtype=int))
    if not combined.has_normals():
        combined.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=options.normal_radius, max_nn=40))
        combined.orient_normals_towards_camera_location()
    combined.normalize_normals()
    return combined


def reconstruct_mesh(cloud, options):
    import open3d as o3d
    if len(cloud.points) < 100:
        raise ValueError("Mesh requires at least 100 filtered points. Point cloud remains available.")
    working = cloud
    if len(cloud.points) > 50_000:
        working = cloud.voxel_down_sample(options.voxel_size * 2)
    distances = np.asarray(working.compute_nearest_neighbor_distance())
    usable = distances[np.isfinite(distances) & (distances > 0)]
    if len(usable) == 0:
        raise ValueError("Mesh has no usable point spacing.")
    radius = max(options.voxel_size, float(np.median(usable))) * options.mesh_radius_factor
    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
        working, o3d.utility.DoubleVector([radius, radius * 2]))
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_unreferenced_vertices()
    if len(mesh.triangles) < 10:
        raise ValueError("Ball pivoting could not form a usable surface. Capture more overlapping views.")
    if len(mesh.triangles) > 200_000:
        mesh = mesh.simplify_quadric_decimation(200_000)
    mesh.compute_vertex_normals()
    return mesh


def cloud_buffer(cloud):
    return np.column_stack([np.asarray(cloud.points), np.asarray(cloud.colors),
                            np.asarray(cloud.normals)]).astype("<f4").tobytes()


def mesh_buffer(mesh):
    colors = np.asarray(mesh.vertex_colors)
    if len(colors) != len(mesh.vertices):
        colors = np.full((len(mesh.vertices), 3), 0.7)
    vertices = np.column_stack([np.asarray(mesh.vertices), colors,
                               np.asarray(mesh.vertex_normals)]).astype("<f4").tobytes()
    return vertices + np.asarray(mesh.triangles, dtype="<u4").tobytes()

