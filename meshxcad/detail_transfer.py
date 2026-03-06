"""Core detail transfer: displacement field computation and application."""

import numpy as np
from scipy.spatial import KDTree
from scipy.interpolate import RBFInterpolator

from . import mesh_io, alignment


def compute_detail_displacement(plain_vertices, featured_vertices, plain_faces):
    """Compute the displacement field that transforms plain geometry into featured.

    The plain and featured meshes must already be roughly aligned (same pose).
    This finds correspondences and computes per-vertex displacement vectors.

    Args:
        plain_vertices: (N, 3) vertices of the plain mesh
        featured_vertices: (M, 3) vertices of the featured mesh
        plain_faces: (F, 3) face indices of the plain mesh

    Returns:
        displacements: (N, 3) per-vertex displacement vectors
    """
    # Align featured to plain using ICP
    aligned_featured, R, t = alignment.icp(featured_vertices, plain_vertices)

    # For each plain vertex, find closest featured vertex
    tree = KDTree(aligned_featured)
    distances, indices = tree.query(plain_vertices)

    # Displacement = where the featured surface is relative to the plain surface
    displacements = aligned_featured[indices] - plain_vertices

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
