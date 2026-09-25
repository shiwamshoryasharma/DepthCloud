import cv2
import numpy as np

from depthcloud.reconstruction.geometry import camera_matrix, rigid_transform
from depthcloud.workflow.schemas import Intrinsics


def make_cloud(points, colors):
    import open3d as o3d
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(np.asarray(points, dtype=np.float64))
    cloud.colors = o3d.utility.Vector3dVector(np.asarray(colors, dtype=np.float64))
    return cloud


def prepare_cloud(cloud, options):
    import open3d as o3d
    down = cloud.voxel_down_sample(options.voxel_size)
    if len(down.points) < 30:
        raise ValueError("At least 30 distinct points are required for registration.")
    down.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(
        radius=max(options.normal_radius, options.voxel_size * 3), max_nn=40))
    down.orient_normals_towards_camera_location()
    return down


def depth_feature_pose(source, target, distance):
    """ORB pixel correspondences lifted by actual depth; rigid 3D RANSAC."""
    orb = cv2.ORB_create(nfeatures=2500)
    kp1, ds1 = orb.detectAndCompute(cv2.cvtColor(source.bgr, cv2.COLOR_BGR2GRAY), source.result["mask"])
    kp2, ds2 = orb.detectAndCompute(cv2.cvtColor(target.bgr, cv2.COLOR_BGR2GRAY), target.result["mask"])
    if ds1 is None or ds2 is None:
        return None, 0
    pairs = cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(ds1, ds2, k=2)
    matches = [pair[0] for pair in pairs if len(pair) == 2 and pair[0].distance < 0.75 * pair[1].distance]
    if len(matches) < 8:
        return None, len(matches)
    xyz = []
    valid_all = []
    for observation, keypoints, indices in (
        (source, kp1, [m.queryIdx for m in matches]), (target, kp2, [m.trainIdx for m in matches])
    ):
        pixels = np.array([keypoints[i].pt for i in indices], dtype=np.float64)
        xy = np.rint(pixels).astype(int)
        depth = observation.result["depth"]
        xy[:, 0] = np.clip(xy[:, 0], 0, depth.shape[1] - 1)
        xy[:, 1] = np.clip(xy[:, 1], 0, depth.shape[0] - 1)
        z = depth[xy[:, 1], xy[:, 0]]
        valid = np.isfinite(z) & (z > 0) & (observation.result["mask"][xy[:, 1], xy[:, 0]] > 0)
        intrinsics = Intrinsics(**observation.result["intrinsics"])
        rays = cv2.undistortPoints(pixels.reshape(-1, 1, 2), camera_matrix(intrinsics),
                                  np.asarray(intrinsics.distortion)).reshape(-1, 2)
        xyz.append(np.column_stack([rays * z[:, None], z]))
        valid_all.append(valid)
    valid = valid_all[0] & valid_all[1]
    a, b = xyz[0][valid], xyz[1][valid]
    if len(a) < 8:
        return None, len(a)
    rng = np.random.default_rng(42)  # RANSAC sampling only, never synthetic geometry.
    best = np.zeros(len(a), dtype=bool)
    for _ in range(300):
        sample = rng.choice(len(a), 3, replace=False)
        try:
            transform = rigid_transform(a[sample], b[sample])
        except ValueError:
            continue
        error = np.linalg.norm(a @ transform[:3, :3].T + transform[:3, 3] - b, axis=1)
        inliers = error < distance
        if inliers.sum() > best.sum():
            best = inliers
    if best.sum() < 8 or best.mean() < 0.25:
        return None, int(best.sum())
    return rigid_transform(a[best], b[best]), int(best.sum())


def register_clouds(source, target, options, initial=None):
    import open3d as o3d
    reg = o3d.pipelines.registration
    a, b = prepare_cloud(source, options), prepare_cloud(target, options)
    method = "RGB/depth correspondences + ICP" if initial is not None else "FPFH global registration + ICP"
    if initial is None:
        radius = options.voxel_size * 5
        fa = reg.compute_fpfh_feature(a, o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=100))
        fb = reg.compute_fpfh_feature(b, o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=100))
        o3d.utility.random.seed(42)
        coarse = reg.registration_ransac_based_on_feature_matching(
            a, b, fa, fb, True, options.icp_distance * 2,
            reg.TransformationEstimationPointToPoint(False), 3,
            [reg.CorrespondenceCheckerBasedOnEdgeLength(0.9),
             reg.CorrespondenceCheckerBasedOnDistance(options.icp_distance * 2)],
            reg.RANSACConvergenceCriteria(20000, 0.995))
        if coarse.fitness < options.min_fitness / 2:
            raise ValueError(f"Insufficient coarse overlap ({coarse.fitness:.2f}). Capture closer, overlapping views.")
        initial = coarse.transformation
    refined = reg.registration_icp(a, b, options.icp_distance, initial,
                                  reg.TransformationEstimationPointToPlane(),
                                  reg.ICPConvergenceCriteria(max_iteration=options.icp_iterations))
    reverse = reg.evaluate_registration(b, a, options.icp_distance, np.linalg.inv(refined.transformation))
    count = len(refined.correspondence_set)
    accepted = (refined.fitness >= options.min_fitness and reverse.fitness >= options.min_fitness
                and refined.inlier_rmse <= options.icp_distance * 0.6 and count >= 20
                and np.isfinite(refined.transformation).all())
    metrics = {"method": method, "fitness": float(refined.fitness), "overlap": float(reverse.fitness),
               "rmse": float(refined.inlier_rmse), "correspondences": count, "accepted": bool(accepted)}
    if not accepted:
        raise ValueError(f"Registration rejected: fitness={refined.fitness:.2f}, reverse overlap={reverse.fitness:.2f}, "
                         f"RMSE={refined.inlier_rmse:.4f}m, matches={count}. More viewpoint overlap is required.")
    return np.asarray(refined.transformation), metrics

