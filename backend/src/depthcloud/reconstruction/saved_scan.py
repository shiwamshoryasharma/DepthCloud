"""Reconstruct a sealed capture set, independently of the running camera."""
import io
from types import SimpleNamespace

import cv2
import numpy as np

from depthcloud.reconstruction.geometry import CAMERA_TO_ISAAC, CAMERA_TO_THREE, project_depth
from depthcloud.reconstruction.live_geometry import SurfaceVolume, view_direction
from depthcloud.reconstruction.scene_tracking import SceneTracker, solve_scene_pose
from depthcloud.reconstruction.surface import cloud_buffer, mesh_buffer
from depthcloud.vision.masking import filter_mask_depth
from depthcloud.workflow.schemas import Intrinsics


def refine_mask(bgr, mask):
    """Keep detection support, remove islands and mixed boundary pixels."""
    seed = np.where(mask > 0, 255, 0).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(seed)
    if count < 2:
        raise ValueError('No detected object pixels remain.')
    sizes = stats[1:, cv2.CC_STAT_AREA]
    keep = np.r_[False, sizes >= max(32, sizes.max() * .03)]
    clean = (keep[labels] * 255).astype(np.uint8)
    clean = cv2.erode(clean, np.ones((3, 3), np.uint8))
    if np.count_nonzero(clean) < 100:
        raise ValueError('Foreground mask is too small after boundary filtering.')
    filtered = np.full_like(bgr, 127)
    filtered[clean > 0] = bgr[clean > 0]
    return clean, filtered


def match_features(source, target, intrinsics):
    pairs = cv2.BFMatcher(cv2.NORM_L2 if source['descriptors'].dtype == np.float32 else cv2.NORM_HAMMING).knnMatch(source['descriptors'], target['descriptors'], k=2)
    unique = {}
    for pair in pairs:
        if len(pair) == 2 and pair[0].distance < .72 * pair[1].distance:
            match = pair[0]
            if match.trainIdx not in unique or match.distance < unique[match.trainIdx].distance:
                unique[match.trainIdx] = match
    matches = list(unique.values())
    a, b = [m.queryIdx for m in matches], [m.trainIdx for m in matches]
    return solve_scene_pose(target['points'][b], source['pixels'][a], source['points'][a, 2], intrinsics)


def npz_bytes(**arrays):
    output = io.BytesIO()
    np.savez_compressed(output, **arrays)
    return output.getvalue()


def load_arrays(path):
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def build_depth_mesh(dataset, models, options, update, check_cancel):
    """Two-pass depth/alignment followed by global pose optimization and TSDF.

    Depth Pro retains original RGB context: inference on cutout RGB destroys
    monocular scale/shape cues (verified on the preserved camera images).
    Masks are prepared first, then all background depth is excluded from fusion.
    Scene pixels constrain poses only. Full frames are streamed from disk.
    """
    import open3d as o3d

    if not dataset.sealed or dataset.count < 3:
        raise ValueError('Reconstruction needs a sealed set of at least 3 images.')
    records = tuple(dict(record) for record in dataset.data['records'])
    total = len(records)
    rejected, filtered_indices = [], []
    for record in records:
        check_cancel()
        index = record['index']
        update('filtering', f'Removing background: {index + 1} / {total}', processed=index + 1, stage_total=total, registered=0)
        try:
            mask, foreground = refine_mask(dataset.image(index), dataset.image(index, 'mask'))
            dataset.write_image(f'{index:04d}-filtered.png', foreground)
            dataset.write_image(f'{index:04d}-clean.png', mask)
            filtered_indices.append(index)
        except ValueError as exc:
            rejected.append({'id': str(index), 'label': f'Capture {index + 1}', 'reason': str(exc)})
    dataset.save()
    tracker = SceneTracker(max_keyframes=24, feature_method="sift")
    intrinsics = Intrinsics(**dataset.data['calibration']) if dataset.data['calibration'] else None
    graph = o3d.pipelines.registration.PoseGraph()
    nodes, node_by_id, signatures = [], {}, []
    loops = 0

    def edge(source_id, target_id, relative, source, target, uncertain):
        source_cloud, target_cloud = o3d.geometry.PointCloud(), o3d.geometry.PointCloud()
        source_cloud.points = o3d.utility.Vector3dVector(source['points'])
        target_cloud.points = o3d.utility.Vector3dVector(target['points'])
        information = o3d.pipelines.registration.get_information_matrix_from_point_clouds(
            source_cloud, target_cloud, .03, relative)
        graph.edges.append(o3d.pipelines.registration.PoseGraphEdge(source_id, target_id, relative, information, uncertain))

    for processed, index in enumerate(filtered_indices):
        check_cancel()
        update('depth_alignment', f'Depth and camera alignment: {processed + 1} / {len(filtered_indices)}',
               processed=processed + 1, stage_total=len(filtered_indices), registered=len(nodes), rejected=len(rejected), loop_closures=loops)
        try:
            bgr = dataset.image(index)
            mask = cv2.imread(str(dataset.root / f'{index:04d}-clean.png'), cv2.IMREAD_GRAYSCALE)
            h, w = bgr.shape[:2]
            if intrinsics:
                intrinsics = intrinsics.scaled(w, h)
            scene_depth, focal = models.infer_depth(bgr, intrinsics.fx if intrinsics else None)
            intrinsics = intrinsics or Intrinsics(fx=focal, fy=focal, cx=(w-1)/2, cy=(h-1)/2,
                                                 width=w, height=h, source='depth_pro_estimate')
            check_cancel()
            object_depth = scene_depth
            object_scale = 1.0
            valid = (mask > 0) & np.isfinite(object_depth) & (object_depth > .05)
            if valid.sum() < 100:
                raise ValueError('Too few valid object depth pixels.')
            view = SimpleNamespace(id=str(index), bgr=bgr, result={'depth': scene_depth, 'mask': mask, 'intrinsics': intrinsics.model_dump()})
            # SceneTracker uses background features outside the detected foreground.
            pose, scale, metrics = tracker.estimate(view)
            features = tracker.current
            scaled_features = {**features, 'points': features['points'] * scale}
            depth = object_depth * object_scale * scale
            mask, _ = filter_mask_depth(depth, mask, options.pipeline.depth_outlier_strength)
            points, _ = project_depth(depth, bgr, mask, intrinsics, options.pipeline)
            if len(points) < 100:
                raise ValueError('Too few clean object samples for fusion.')
            # No pose is committed until object-depth validation also succeeds.
            dataset.write(f'{index:04d}-features.npz', npz_bytes(points=scaled_features['points'], pixels=features['pixels'], descriptors=features['descriptors']))
            dataset.write(f'{index:04d}-depth.npz', npz_bytes(depth=np.where(mask > 0, depth, 0).astype(np.float32), mask=mask))
            parent = metrics.get('keyframe_id')
            node_id = len(nodes)
            graph.nodes.append(o3d.pipelines.registration.PoseGraphNode(pose))
            if parent is not None:
                parent_node = node_by_id[parent]
                target = load_arrays(dataset.root / f'{nodes[parent_node]["index"]:04d}-features.npz')
                edge(node_id, parent_node, np.linalg.inv(nodes[parent_node]['pose']) @ pose, scaled_features, target, False)
            histogram = cv2.calcHist([cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)], [0, 1], None, [16, 8], [0, 180, 0, 256]).ravel()
            histogram /= max(1., np.linalg.norm(histogram))
            # Bounded loop search; every candidate must pass independent PnP,
            # image-spread, reprojection and scene-depth consistency checks.
            if node_id >= 15 and node_id % 5 == 0:
                candidates = sorted(range(node_id - 10), key=lambda j: float(signatures[j] @ histogram), reverse=True)[:3]
                for candidate in candidates:
                    check_cancel()
                    target = load_arrays(dataset.root / f'{nodes[candidate]["index"]:04d}-features.npz')
                    try:
                        relative, correction, _ = match_features(scaled_features, target, intrinsics)
                        if not .95 < correction < 1.05:
                            continue
                        edge(node_id, candidate, relative, scaled_features, target, True)
                        loops += 1
                    except ValueError:
                        continue
            nodes.append({'index': index, 'pose': pose, 'scale': scale, 'metrics': metrics})
            node_by_id[str(index)] = node_id
            signatures.append(histogram)
            tracker.commit(view, pose, scale)
        except ValueError as exc:
            rejected.append({'id': str(index), 'label': f'Capture {index + 1}', 'reason': str(exc)})
            update('depth_alignment', str(exc), rejected=len(rejected))
    dataset.data['report'] = {'registered': len(nodes), 'total': total, 'rejected': rejected, 'loop_closures': loops}
    dataset.save()
    if len(nodes) < 3:
        raise ValueError(f'Only {len(nodes)} / {total} images aligned. At least 3 reliable overlapping views are required; captures are retained.')
    check_cancel()
    update('optimizing', f'Optimizing {len(nodes)} camera poses with {loops} verified loop closures.', registered=len(nodes), loop_closures=loops)
    o3d.pipelines.registration.global_optimization(graph,
        o3d.pipelines.registration.GlobalOptimizationLevenbergMarquardt(),
        o3d.pipelines.registration.GlobalOptimizationConvergenceCriteria(),
        o3d.pipelines.registration.GlobalOptimizationOption(max_correspondence_distance=.03, edge_prune_threshold=.25, reference_node=0))
    bounds, centers, diameters = [], [], []
    for node, optimized in zip(nodes, graph.nodes):
        check_cancel()
        node['pose'] = np.asarray(optimized.pose).copy()
        data = load_arrays(dataset.root / f'{node["index"]:04d}-depth.npz')
        points, _ = project_depth(data['depth'], dataset.image(node['index']), data['mask'], intrinsics, options.pipeline)
        world = points @ node['pose'][:3, :3].T + node['pose'][:3, 3]
        bounds.append(np.percentile(world, [1, 99], axis=0))
        centers.append(np.median(world, axis=0))
        diameters.append(float(np.linalg.norm(np.ptp(world, axis=0))))
    center = np.median(centers, axis=0)
    tolerance = max(.025, float(np.median(diameters)) * .4)
    accepted = [i for i, point in enumerate(centers) if np.linalg.norm(point - center) <= tolerance]
    if len(accepted) < 3:
        raise ValueError('Aligned object positions are inconsistent. Captures are retained for inspection.')
    for i, node in enumerate(nodes):
        if i not in accepted:
            rejected.append({'id': str(node['index']), 'label': f'Capture {node["index"] + 1}', 'reason': 'Object position inconsistent after pose optimization.'})
    combined = np.concatenate([bounds[i] for i in accepted])
    volume = SurfaceVolume(np.array([combined.min(axis=0), combined.max(axis=0)]), options)
    directions = set()
    for processed, i in enumerate(accepted):
        check_cancel()
        node = nodes[i]
        update('fusing', f'Fusing object surfaces: {processed + 1} / {len(accepted)}', processed=processed + 1, stage_total=len(accepted), registered=len(accepted), rejected=len(rejected))
        data = load_arrays(dataset.root / f'{node["index"]:04d}-depth.npz')
        volume.integrate(dataset.image(node['index']), data['depth'], data['mask'], intrinsics, node['pose'])
        directions.add(view_direction(node['pose']))
    check_cancel()
    update('meshing', 'Extracting the fused triangle surface.')
    cloud, mesh = volume.extract()
    points = np.asarray(cloud.points)
    result = {'points': len(points), 'vertices': len(mesh.vertices), 'triangles': len(mesh.triangles),
              'accepted': [str(nodes[i]['index']) for i in accepted], 'rejected': rejected,
              'mesh_error': None, 'dimensions_m': np.ptp(points, axis=0).tolist(),
              'bounds': [points.min(axis=0).tolist(), points.max(axis=0).tolist()], 'units': 'meters',
              'frame': 'First aligned camera: +X right,+Y down,+Z forward',
              'camera_to_three': CAMERA_TO_THREE.tolist(), 'camera_to_isaac': CAMERA_TO_ISAAC.tolist(),
              'estimated_geometry': True, 'calibration_source': intrinsics.source,
              'method': 'sealed capture set / scene PnP / pose graph / masked TSDF',
              'depth_input': 'original RGB context; foreground-only fusion',
              'directions': sorted(directions), 'loop_closures': loops, 'complete_coverage': False}
    # Captures and diagnostics survive restarts; the native mesh can be rebuilt.
    dataset.data['report'] = {**result, 'poses': [{'index': n['index'], 'matrix': n['pose'].tolist(), 'scale': n['scale']} for n in nodes]}
    dataset.save()
    return cloud, mesh, {'cloud': cloud_buffer(cloud), 'mesh': mesh_buffer(mesh)}, result


def build_saved_mesh(dataset, models, options, update, check_cancel):
    if options.reconstruction_method == 'depth_fusion':
        return build_depth_mesh(dataset, models, options, update, check_cancel)
    from depthcloud.reconstruction.multiview import build_multiview_mesh
    return build_multiview_mesh(dataset, models, options, update, check_cancel)
