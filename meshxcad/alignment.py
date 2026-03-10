"""Alignment utilities: ICP registration and spatial correspondence."""

import itertools
import numpy as np
from scipy.spatial import KDTree
from .gpu import AcceleratedKDTree as _AKDTree, svd as _gpu_svd, eigh as _gpu_eigh


def icp(source, target, max_iterations=50, tolerance=1e-6):
    """Iterative Closest Point alignment of source points to target points.

    Args:
        source: (N, 3) array of source points
        target: (M, 3) array of target points
        max_iterations: maximum ICP iterations
        tolerance: convergence threshold on mean point distance change

    Returns:
        aligned_source: (N, 3) transformed source points
        rotation: (3, 3) rotation matrix
        translation: (3,) translation vector
    """
    src = source.copy()
    R_total = np.eye(3)
    t_total = np.zeros(3)
    prev_error = float("inf")

    for _ in range(max_iterations):
        tree = _AKDTree(target)
        distances, indices = tree.query(src)
        mean_error = np.mean(distances)

        if abs(prev_error - mean_error) < tolerance:
            break
        prev_error = mean_error

        # Find centroids
        src_centroid = np.mean(src, axis=0)
        tgt_centroid = np.mean(target[indices], axis=0)

        # Center points
        src_centered = src - src_centroid
        tgt_centered = target[indices] - tgt_centroid

        # SVD for optimal rotation
        H = src_centered.T @ tgt_centered
        U, _, Vt = _gpu_svd(H)
        R = Vt.T @ U.T

        # Ensure proper rotation (det = +1)
        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = Vt.T @ U.T

        t = tgt_centroid - R @ src_centroid

        # Apply transform
        src = (R @ src.T).T + t
        R_total = R @ R_total
        t_total = R @ t_total + t

    return src, R_total, t_total


def find_correspondences(source_points, target_points, max_distance=None):
    """Find nearest-neighbor correspondences between point sets.

    Args:
        source_points: (N, 3) array
        target_points: (M, 3) array
        max_distance: optional maximum correspondence distance

    Returns:
        source_indices: indices into source_points
        target_indices: corresponding indices into target_points
        distances: distances between correspondences
    """
    tree = _AKDTree(target_points)
    distances, indices = tree.query(source_points)

    if max_distance is not None:
        mask = distances <= max_distance
        src_idx = np.where(mask)[0]
        tgt_idx = indices[mask]
        dists = distances[mask]
    else:
        src_idx = np.arange(len(source_points))
        tgt_idx = indices
        dists = distances

    return src_idx, tgt_idx, dists


def compute_displacement_field(source_points, target_points, source_normals=None):
    """Compute per-vertex displacement from source to target.

    If normals are provided, returns signed displacement along the normal.
    Otherwise returns the 3D displacement vectors.

    Args:
        source_points: (N, 3) source vertex positions
        target_points: (N, 3) corresponding target positions (same indexing)
        source_normals: optional (N, 3) vertex normals

    Returns:
        If normals given: (N,) signed scalar displacements
        Otherwise: (N, 3) displacement vectors
    """
    diff = target_points - source_points
    if source_normals is not None:
        return np.sum(diff * source_normals, axis=1)
    return diff


# ---------------------------------------------------------------------------
# Intelligent pre-alignment
# ---------------------------------------------------------------------------

def _pca_axes(points):
    """Return PCA eigenvectors sorted by descending eigenvalue."""
    centered = points - points.mean(axis=0)
    cov = centered.T @ centered / len(points)
    eigvals, eigvecs = _gpu_eigh(cov)
    order = np.argsort(eigvals)[::-1]
    return eigvecs[:, order], np.sqrt(np.maximum(eigvals[order], 0))


def _axis_rotation(axis, angle):
    """Rodrigues rotation matrix: rotate by *angle* radians about *axis*."""
    axis = axis / (np.linalg.norm(axis) + 1e-12)
    c, s = np.cos(angle), np.sin(angle)
    K = np.array([[0, -axis[2], axis[1]],
                  [axis[2], 0, -axis[0]],
                  [-axis[1], axis[0], 0]])
    return np.eye(3) * c + (1 - c) * np.outer(axis, axis) + s * K


def _mean_nn_distance(src, tgt_tree):
    """Mean nearest-neighbour distance from *src* to a pre-built KDTree."""
    dists, _ = tgt_tree.query(src)
    return float(np.mean(dists))


def estimate_scale(source, target):
    """Estimate uniform scale factor that maps *source* extent to *target*.

    Uses the ratio of bounding-box diagonals.
    """
    src_diag = np.linalg.norm(source.max(axis=0) - source.min(axis=0))
    tgt_diag = np.linalg.norm(target.max(axis=0) - target.min(axis=0))
    if src_diag < 1e-12:
        return 1.0
    return float(tgt_diag / src_diag)


def _pca_rotation_candidates(src_axes, tgt_axes):
    """Generate candidate rotations that map PCA axes of source to target.

    There are 24 possible axis permutations × sign flips that map one
    orthogonal frame onto another; we test them all and keep the rotation
    matrices that are proper (det = +1).
    """
    candidates = []
    perms = list(itertools.permutations(range(3)))
    signs = list(itertools.product([-1, 1], repeat=3))

    for perm in perms:
        for sgn in signs:
            # Build target frame with permuted/flipped axes
            T = np.column_stack([
                tgt_axes[:, perm[0]] * sgn[0],
                tgt_axes[:, perm[1]] * sgn[1],
                tgt_axes[:, perm[2]] * sgn[2],
            ])
            R = T @ np.linalg.inv(src_axes)
            if np.linalg.det(R) > 0:  # proper rotation only
                candidates.append(R)
    return candidates


def pre_align(source, target, subsample=2000):
    """Coarse alignment of *source* to *target* via scale + PCA orientation.

    Steps:
      1. Estimate and apply uniform scale.
      2. Centre both point clouds.
      3. Enumerate PCA axis permutations/flips (up to 24 candidates).
      4. For each, also try 3 additional in-plane rotations (45°, 90°, 135°)
         around the longest PCA axis — catches parts that are symmetric.
      5. Pick the candidate with the lowest mean nearest-neighbour distance.

    Args:
        source: (N, 3) point cloud to be aligned.
        target: (M, 3) reference point cloud.
        subsample: max points to use for scoring (speed).

    Returns:
        aligned: (N, 3) aligned source points.
        scale: float, uniform scale applied.
        R: (3, 3) rotation matrix applied.
        t: (3,) translation applied.
    """
    source = np.asarray(source, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)

    # 1. Estimate scale
    raw_scale = estimate_scale(source, target)

    # Test both scaled and unscaled; keep whichever is better
    scales_to_try = [raw_scale]
    if abs(raw_scale - 1.0) > 0.01:
        scales_to_try.append(1.0)

    tgt_center = target.mean(axis=0)
    tgt_c = target - tgt_center

    # Subsample for fast scoring
    rng = np.random.RandomState(0)
    if len(tgt_c) > subsample:
        tgt_sub = tgt_c[rng.choice(len(tgt_c), subsample, replace=False)]
    else:
        tgt_sub = tgt_c
    tgt_tree = _AKDTree(tgt_sub)

    best_scale = 1.0
    best_R = np.eye(3)
    best_src_center = source.mean(axis=0)
    best_err = float("inf")

    for scale in scales_to_try:
        scaled = source * scale
        src_center = scaled.mean(axis=0)
        src_c = scaled - src_center

        if len(src_c) > subsample:
            src_sub = src_c[rng.choice(len(src_c), subsample, replace=False)]
        else:
            src_sub = src_c

        # Score identity (no rotation) for this scale
        err_identity = _mean_nn_distance(src_sub, tgt_tree)
        if err_identity < best_err:
            best_err = err_identity
            best_R = np.eye(3)
            best_scale = scale
            best_src_center = src_center

        # 3. PCA frames
        src_axes, _ = _pca_axes(src_c)
        tgt_axes, _ = _pca_axes(tgt_c)

        base_candidates = _pca_rotation_candidates(src_axes, tgt_axes)

        # 4. Augment with in-plane twists around the primary target axis
        all_candidates = []
        primary_axis = tgt_axes[:, 0]
        twist_angles = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]
        for R_base in base_candidates:
            for angle in twist_angles:
                R_twist = _axis_rotation(primary_axis, angle)
                all_candidates.append(R_twist @ R_base)

        # 5. Score each candidate
        for R_cand in all_candidates:
            rotated = src_sub @ R_cand.T
            err = _mean_nn_distance(rotated, tgt_tree)
            if err < best_err:
                best_err = err
                best_R = R_cand
                best_scale = scale
                best_src_center = src_center

    # Build final transform
    scaled_final = source * best_scale
    src_c_final = scaled_final - best_src_center
    aligned = src_c_final @ best_R.T + tgt_center
    t = tgt_center - best_R @ best_src_center

    return aligned, best_scale, best_R, t


def full_align(source, target, max_icp_iterations=100):
    """Complete alignment pipeline: pre_align → ICP.

    Returns:
        aligned: (N, 3) aligned source points.
        scale: float, uniform scale applied.
        R: (3, 3) total rotation.
        t: (3,) total translation.
    """
    # Coarse alignment
    pre_aligned, scale, R_pre, t_pre = pre_align(source, target)

    # Fine ICP
    aligned, R_icp, t_icp = icp(pre_aligned, target,
                                  max_iterations=max_icp_iterations)

    # Compose transforms: final = R_icp @ (scale * source @ R_pre.T + t_pre) + t_icp
    R_total = R_icp @ R_pre
    t_total = R_icp @ t_pre + t_icp

    return aligned, scale, R_total, t_total
