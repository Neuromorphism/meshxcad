"""Core detail transfer: displacement field computation and application."""

import numpy as np
from scipy.spatial import KDTree
from scipy.interpolate import RBFInterpolator

from . import mesh_io, alignment


def compute_detail_displacement(plain_vertices, featured_vertices, plain_faces,
                                 outlier_percentile=95, max_icp_iterations=100):
    """Compute the displacement field that transforms plain geometry into featured.

    Improved algorithm:
    1. Align using ICP with more iterations
    2. For each plain vertex, find closest featured vertex
    3. Reject outlier correspondences (handles topology mismatches)
    4. Clamp displacement magnitudes to prevent wild deformations

    Args:
        plain_vertices: (N, 3) vertices of the plain mesh
        featured_vertices: (M, 3) vertices of the featured mesh
        plain_faces: (F, 3) face indices of the plain mesh
        outlier_percentile: percentile threshold for outlier rejection
        max_icp_iterations: maximum ICP iterations

    Returns:
        displacements: (N, 3) per-vertex displacement vectors
    """
    # Try ICP alignment in both directions, pick the one with lower error
    aligned_fwd, R_fwd, t_fwd = alignment.icp(
        featured_vertices, plain_vertices, max_iterations=max_icp_iterations
    )
    fwd_tree = KDTree(aligned_fwd)
    fwd_dists, _ = fwd_tree.query(plain_vertices)
    fwd_error = np.mean(fwd_dists)

    # Also try aligning plain to featured (reverse direction)
    aligned_rev, R_rev, t_rev = alignment.icp(
        plain_vertices, featured_vertices, max_iterations=max_icp_iterations
    )
    # Invert: we need featured in plain's frame
    # If plain was moved to featured's frame by R_rev, t_rev:
    #   aligned_rev = R_rev @ plain + t_rev  (≈ featured)
    # So featured in plain's frame = R_rev^T @ (featured - t_rev)
    aligned_rev_featured = (np.linalg.inv(R_rev) @ (featured_vertices - t_rev).T).T
    rev_tree = KDTree(aligned_rev_featured)
    rev_dists, _ = rev_tree.query(plain_vertices)
    rev_error = np.mean(rev_dists)

    # Pick the better alignment
    if fwd_error <= rev_error:
        aligned_featured = aligned_fwd
    else:
        aligned_featured = aligned_rev_featured

    # Find correspondences
    tree = KDTree(aligned_featured)
    distances, indices = tree.query(plain_vertices)

    # Compute raw displacements
    displacements = aligned_featured[indices] - plain_vertices

    # Outlier rejection: clamp large displacements
    disp_mags = np.linalg.norm(displacements, axis=1)
    if len(disp_mags) > 0 and np.max(disp_mags) > 0:
        threshold = np.percentile(disp_mags, outlier_percentile)
        outliers = disp_mags > threshold
        if np.any(outliers):
            # Scale down outlier displacements to the threshold
            scale = np.ones(len(disp_mags))
            scale[outliers] = threshold / disp_mags[outliers]
            displacements = displacements * scale[:, None]

    return displacements


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
