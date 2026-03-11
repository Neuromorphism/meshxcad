"""Common mesh builder for revolve-based objects.

Provides the revolve engine, component combiners, and helpers
used by all object definitions.
"""

import math
import numpy as np


def revolve_profile(profile_rz, n_angular=48, close_top=True, close_bottom=True):
    """Revolve a 2D profile around the Z axis to create a mesh.

    Args:
        profile_rz: list of (radius, z) tuples from bottom to top
        n_angular: number of angular divisions
        close_top: add a cap at the top
        close_bottom: add a cap at the bottom

    Returns:
        vertices: (N, 3) array
        faces: (M, 3) array
    """
    n_profile = len(profile_rz)
    vertices = []
    faces = []

    # Generate ring vertices
    for r, z in profile_rz:
        for j in range(n_angular):
            angle = 2 * math.pi * j / n_angular
            vertices.append([r * math.cos(angle), r * math.sin(angle), z])

    vertices = np.array(vertices, dtype=np.float64)

    # Side faces
    for i in range(n_profile - 1):
        for j in range(n_angular):
            j_next = (j + 1) % n_angular
            p00 = i * n_angular + j
            p01 = i * n_angular + j_next
            p10 = (i + 1) * n_angular + j
            p11 = (i + 1) * n_angular + j_next
            faces.append([p00, p01, p10])
            faces.append([p01, p11, p10])

    # Bottom cap
    if close_bottom and profile_rz[0][0] > 0.01:
        center_idx = len(vertices)
        vertices = np.vstack([vertices, [[0, 0, profile_rz[0][1]]]])
        for j in range(n_angular):
            j_next = (j + 1) % n_angular
            faces.append([center_idx, j_next, j])

    # Top cap
    if close_top and profile_rz[-1][0] > 0.01:
        center_idx = len(vertices)
        top_start = (n_profile - 1) * n_angular
        vertices = np.vstack([vertices, [[0, 0, profile_rz[-1][1]]]])
        for j in range(n_angular):
            j_next = (j + 1) % n_angular
            faces.append([center_idx, top_start + j, top_start + j_next])

    return vertices, np.array(faces)


def make_torus(major_r, minor_r, z_center, n_angular=48, n_cross=12):
    """Create a torus mesh."""
    profile = []
    r_mid = major_r
    for i in range(n_cross + 1):
        angle = 2 * math.pi * i / n_cross
        r = r_mid + minor_r * math.cos(angle)
        z = z_center + minor_r * math.sin(angle)
        profile.append((r, z))
    return revolve_profile(profile, n_angular, close_top=False, close_bottom=False)


def make_cylinder(radius, z_bottom, z_top, n_angular=24, n_height=2):
    """Create a simple cylinder mesh."""
    profile = []
    for i in range(n_height + 1):
        z = z_bottom + (z_top - z_bottom) * i / n_height
        profile.append((radius, z))
    return revolve_profile(profile, n_angular)


def combine_meshes(mesh_list):
    """Combine multiple (vertices, faces) into a single mesh."""
    if not mesh_list:
        return np.zeros((0, 3)), np.zeros((0, 3), dtype=int)
    all_verts = []
    all_faces = []
    offset = 0
    for verts, faces in mesh_list:
        all_verts.append(verts)
        all_faces.append(faces + offset)
        offset += len(verts)
    return np.vstack(all_verts), np.vstack(all_faces)


def smooth_profile(points, n_output=40):
    """Interpolate a coarse profile into a smooth one with more points.

    Args:
        points: list of (radius, z) tuples (coarse)
        n_output: number of output points

    Returns:
        list of (radius, z) tuples (smooth)
    """
    from scipy.interpolate import CubicSpline
    rs = [p[0] for p in points]
    zs = [p[1] for p in points]
    t = np.linspace(0, 1, len(points))
    t_fine = np.linspace(0, 1, n_output)
    cs_r = CubicSpline(t, rs)
    cs_z = CubicSpline(t, zs)
    return [(max(float(cs_r(ti)), 0.01), float(cs_z(ti))) for ti in t_fine]


def lerp(a, b, t):
    """Linear interpolation."""
    return a + (b - a) * t
