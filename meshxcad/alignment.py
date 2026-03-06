"""Alignment utilities: ICP registration and spatial correspondence."""

import numpy as np
from scipy.spatial import KDTree


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
        tree = KDTree(target)
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
        U, _, Vt = np.linalg.svd(H)
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
    tree = KDTree(target_points)
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
