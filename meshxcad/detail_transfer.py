"""Core detail transfer: displacement field computation and application."""

import numpy as np
from scipy.spatial import KDTree
from scipy.interpolate import RBFInterpolator

from . import mesh_io, alignment


def compute_detail_displacement(plain_vertices, featured_vertices, plain_faces,
                                 outlier_percentile=95, max_icp_iterations=100):
    """Compute the displacement field that transforms plain geometry into featured.

    Algorithm:
    1. Intelligent pre-alignment (scale + PCA orientation search + ICP)
       in both directions, picking the lower-error result.
    2. For each plain vertex, find closest aligned featured vertex.
    3. Reject outlier correspondences (handles topology mismatches).
    4. Clamp displacement magnitudes to prevent wild deformations.

    Args:
        plain_vertices: (N, 3) vertices of the plain mesh
        featured_vertices: (M, 3) vertices of the featured mesh
        plain_faces: (F, 3) face indices of the plain mesh
        outlier_percentile: percentile threshold for outlier rejection
        max_icp_iterations: maximum ICP iterations

    Returns:
        displacements: (N, 3) per-vertex displacement vectors
    """
    def _try_candidate(aligned_featured):
        """Score a candidate by the final result distance (after displacement + outlier clamping)."""
        tree = KDTree(aligned_featured)
        distances, indices = tree.query(plain_vertices)
        disp = aligned_featured[indices] - plain_vertices
        mags = np.linalg.norm(disp, axis=1)
        if len(mags) > 0 and np.max(mags) > 0:
            thresh = np.percentile(mags, outlier_percentile)
            out = mags > thresh
            if np.any(out):
                s = np.ones(len(mags))
                s[out] = thresh / mags[out]
                disp = disp * s[:, None]
        result = plain_vertices + disp
        # Measure how close result is to the original (unaligned) featured mesh
        feat_tree = KDTree(featured_vertices)
        rd, _ = feat_tree.query(result)
        return float(np.mean(rd)), disp

    candidates = []

    # --- Strategy 1: pre_align forward rotation + ICP (no scale applied to geometry) ---
    pa_aligned, pa_scale, pa_R, pa_t = alignment.pre_align(
        featured_vertices, plain_vertices)
    feat_rotated = (featured_vertices - featured_vertices.mean(axis=0)) @ pa_R.T + plain_vertices.mean(axis=0)
    icp_result, _, _ = alignment.icp(feat_rotated, plain_vertices, max_iterations=max_icp_iterations)
    score1, disp1 = _try_candidate(icp_result)
    candidates.append((score1, disp1))

    # --- Strategy 2: ICP-only forward (handles already-aligned inputs) ---
    icp_fwd, _, _ = alignment.icp(
        featured_vertices, plain_vertices, max_iterations=max_icp_iterations
    )
    score2, disp2 = _try_candidate(icp_fwd)
    candidates.append((score2, disp2))

    # --- Strategy 3: ICP-only reverse ---
    icp_rev, R_icp_rev, t_icp_rev = alignment.icp(
        plain_vertices, featured_vertices, max_iterations=max_icp_iterations
    )
    icp_rev_featured = (np.linalg.inv(R_icp_rev) @ (featured_vertices - t_icp_rev).T).T
    score3, disp3 = _try_candidate(icp_rev_featured)
    candidates.append((score3, disp3))

    # Pick the best by final result quality
    candidates.sort(key=lambda x: x[0])
    return candidates[0][1]


def apply_displacement_to_mesh(vertices, faces, displacements):
    """Apply displacement vectors to mesh vertices.

    Args:
        vertices: (N, 3) vertex positions
        faces: (F, 3) face indices
        displacements: (N, 3) displacement per vertex

    Returns:
        new_vertices: (N, 3) displaced vertices
    """
    return vertices + displacements


def interpolate_displacement_field(
    known_points, known_displacements, query_points, smoothing=0.0
):
    """Interpolate a displacement field using RBF interpolation.

    This is used to transfer displacements computed on one mesh onto
    different geometry (e.g., from mesh vertices to CAD surface sample points).

    Args:
        known_points: (N, 3) points where displacement is known
        known_displacements: (N, 3) displacement vectors at known points
        query_points: (M, 3) points where displacement is needed
        smoothing: RBF smoothing parameter (0 = exact interpolation)

    Returns:
        interpolated: (M, 3) displacement vectors at query points
    """
    result = np.zeros((len(query_points), 3))
    for dim in range(3):
        rbf = RBFInterpolator(
            known_points,
            known_displacements[:, dim],
            smoothing=smoothing,
            kernel="thin_plate_spline",
        )
        result[:, dim] = rbf(query_points)
    return result


def transfer_mesh_detail_to_mesh(plain_mesh_verts, plain_mesh_faces,
                                  featured_mesh_verts, featured_mesh_faces):
    """Transfer detail from a featured mesh onto a plain mesh.

    Args:
        plain_mesh_verts: (N, 3) vertices of plain mesh
        plain_mesh_faces: (F, 3) faces of plain mesh
        featured_mesh_verts: (M, 3) vertices of featured mesh
        featured_mesh_faces: (G, 3) faces of featured mesh

    Returns:
        result_verts: (N, 3) modified vertices with detail applied
    """
    displacements = compute_detail_displacement(
        plain_mesh_verts, featured_mesh_verts, plain_mesh_faces
    )
    return apply_displacement_to_mesh(plain_mesh_verts, plain_mesh_faces, displacements)
