"""Synthetic test geometry generation using pure numpy (no FreeCAD required).

Used for unit testing the alignment and detail transfer algorithms
without requiring a FreeCAD installation.
"""

import numpy as np


def make_cube_mesh(size=10.0, subdivisions=4):
    """Generate a triangulated cube mesh centered at origin.

    Args:
        size: side length
        subdivisions: number of subdivisions per edge

    Returns:
        vertices: (N, 3) array
        faces: (M, 3) array of triangle indices
    """
    half = size / 2
    vertices = []
    faces = []
    vert_offset = 0

    # Generate each face as a subdivided quad
    face_configs = [
        # (origin, u_axis, v_axis) for each face
        (np.array([-half, -half, half]), np.array([1, 0, 0]), np.array([0, 1, 0])),   # +Z
        (np.array([-half, -half, -half]), np.array([1, 0, 0]), np.array([0, 1, 0])),  # -Z
        (np.array([half, -half, -half]), np.array([0, 1, 0]), np.array([0, 0, 1])),   # +X
        (np.array([-half, -half, -half]), np.array([0, 1, 0]), np.array([0, 0, 1])),  # -X
        (np.array([-half, half, -half]), np.array([1, 0, 0]), np.array([0, 0, 1])),   # +Y
        (np.array([-half, -half, -half]), np.array([1, 0, 0]), np.array([0, 0, 1])),  # -Y
    ]

    n = subdivisions + 1
    for origin, u_ax, v_ax in face_configs:
        face_verts = []
        for j in range(n):
            for i in range(n):
                u = i / subdivisions
                v = j / subdivisions
                pt = origin + u * size * u_ax + v * size * v_ax
                face_verts.append(pt)
        vertices.extend(face_verts)

        for j in range(subdivisions):
            for i in range(subdivisions):
                idx = vert_offset + j * n + i
                # Two triangles per quad
                faces.append([idx, idx + 1, idx + n])
                faces.append([idx + 1, idx + n + 1, idx + n])

        vert_offset += n * n

    return np.array(vertices), np.array(faces)


def add_cube_face_pockets(vertices, faces, size=10.0, pocket_size=3.0, pocket_depth=1.0):
    """Add pocket features to a cube mesh by displacing vertices near face centers.

    Vertices within pocket_size/2 of each face center are pushed inward
    by pocket_depth along the face normal.

    Args:
        vertices: (N, 3) cube mesh vertices
        faces: (M, 3) face indices
        size: cube side length (must match original)
        pocket_size: width of square pocket region
        pocket_depth: how deep to push vertices inward

    Returns:
        modified_vertices: (N, 3) vertices with pockets applied
    """
    half = size / 2
    ph = pocket_size / 2
    result = vertices.copy()

    # Face definitions: (axis_index, axis_value, inward_sign)
    face_defs = [
        (2, half, -1),   # +Z face
        (2, -half, 1),   # -Z face
        (0, half, -1),   # +X face
        (0, -half, 1),   # -X face
        (1, half, -1),   # +Y face
        (1, -half, 1),   # -Y face
    ]

    for axis, val, sign in face_defs:
        # Find vertices on this face
        on_face = np.abs(result[:, axis] - val) < 1e-6
        # Find vertices within pocket region (check the other two axes)
        other_axes = [a for a in range(3) if a != axis]
        in_pocket = on_face.copy()
        for oa in other_axes:
            in_pocket &= np.abs(result[:, oa]) <= ph

        # Displace inward
        result[in_pocket, axis] += sign * pocket_depth

    return result


def make_sphere_mesh(radius=5.0, lat_divs=20, lon_divs=20):
    """Generate a triangulated sphere mesh centered at origin.

    Args:
        radius: sphere radius
        lat_divs: number of latitude divisions
        lon_divs: number of longitude divisions

    Returns:
        vertices: (N, 3) array
        faces: (M, 3) array
    """
    vertices = []
    faces = []

    for i in range(lat_divs + 1):
        theta = np.pi * i / lat_divs
        for j in range(lon_divs):
            phi = 2 * np.pi * j / lon_divs
            x = radius * np.sin(theta) * np.cos(phi)
            y = radius * np.sin(theta) * np.sin(phi)
            z = radius * np.cos(theta)
            vertices.append([x, y, z])

    vertices = np.array(vertices)

    for i in range(lat_divs):
        for j in range(lon_divs):
            p1 = i * lon_divs + j
            p2 = i * lon_divs + (j + 1) % lon_divs
            p3 = (i + 1) * lon_divs + j
            p4 = (i + 1) * lon_divs + (j + 1) % lon_divs
            faces.append([p1, p2, p3])
            faces.append([p2, p4, p3])

    return vertices, np.array(faces)


def add_sphere_dimples(vertices, radius=5.0, dimple_angle=30.0, dimple_depth=0.8):
    """Add dimple features at 6 cardinal points of a sphere mesh.

    Args:
        vertices: (N, 3) sphere mesh vertices
        radius: sphere radius
        dimple_angle: angular extent of each dimple in degrees
        dimple_depth: max depth of dimple

    Returns:
        modified_vertices: (N, 3) with dimples
    """
    result = vertices.copy()
    cos_threshold = np.cos(np.radians(dimple_angle))

    cardinal_dirs = np.array([
        [0, 0, 1], [0, 0, -1],
        [1, 0, 0], [-1, 0, 0],
        [0, 1, 0], [0, -1, 0],
    ], dtype=float)

    norms = np.linalg.norm(result, axis=1, keepdims=True)
    norms[norms == 0] = 1
    unit_verts = result / norms

    for d in cardinal_dirs:
        cos_angles = unit_verts @ d
        in_dimple = cos_angles > cos_threshold
        # Smooth displacement: cos-based falloff
        depth_factor = (cos_angles[in_dimple] - cos_threshold) / (1 - cos_threshold)
        displacement = dimple_depth * depth_factor
        # Push inward along vertex normal (toward center)
        result[in_dimple] -= displacement[:, None] * unit_verts[in_dimple]

    return result


def make_cylinder_mesh(radius=5.0, height=15.0, radial_divs=24, height_divs=20):
    """Generate a triangulated cylinder mesh centered at origin.

    Args:
        radius: cylinder radius
        height: cylinder height
        radial_divs: circumferential divisions
        height_divs: divisions along height

    Returns:
        vertices: (N, 3) array
        faces: (M, 3) array
    """
    vertices = []
    faces = []
    half_h = height / 2

    # Side vertices
    for i in range(height_divs + 1):
        z = -half_h + height * i / height_divs
        for j in range(radial_divs):
            angle = 2 * np.pi * j / radial_divs
            x = radius * np.cos(angle)
            y = radius * np.sin(angle)
            vertices.append([x, y, z])

    # Side faces
    for i in range(height_divs):
        for j in range(radial_divs):
            p1 = i * radial_divs + j
            p2 = i * radial_divs + (j + 1) % radial_divs
            p3 = (i + 1) * radial_divs + j
            p4 = (i + 1) * radial_divs + (j + 1) % radial_divs
            faces.append([p1, p2, p3])
            faces.append([p2, p4, p3])

    # Cap centers
    top_center_idx = len(vertices)
    vertices.append([0, 0, half_h])
    bot_center_idx = len(vertices)
    vertices.append([0, 0, -half_h])

    # Top cap faces
    top_ring_start = height_divs * radial_divs
    for j in range(radial_divs):
        p1 = top_ring_start + j
        p2 = top_ring_start + (j + 1) % radial_divs
        faces.append([top_center_idx, p1, p2])

    # Bottom cap faces
    for j in range(radial_divs):
        p1 = j
        p2 = (j + 1) % radial_divs
        faces.append([bot_center_idx, p2, p1])

    return np.array(vertices), np.array(faces)


def add_cylinder_grooves(vertices, radius=5.0, height=15.0,
                          groove_depth=0.8, groove_width=1.5, num_grooves=3):
    """Add circumferential groove features to a cylinder mesh.

    Args:
        vertices: (N, 3) cylinder vertices
        radius: cylinder radius
        height: cylinder height
        groove_depth: depth of grooves
        groove_width: width of grooves
        num_grooves: number of grooves

    Returns:
        modified_vertices: (N, 3) with grooves
    """
    result = vertices.copy()
    half_h = height / 2
    spacing = height / (num_grooves + 1)

    for i in range(num_grooves):
        z_center = -half_h + spacing * (i + 1)
        half_w = groove_width / 2

        # Find vertices on the barrel surface near this z
        r = np.sqrt(result[:, 0] ** 2 + result[:, 1] ** 2)
        on_barrel = np.abs(r - radius) < 1e-6
        in_groove_z = np.abs(result[:, 2] - z_center) <= half_w
        in_groove = on_barrel & in_groove_z

        # Push inward along radial direction
        if np.any(in_groove):
            radial_dir = np.zeros_like(result[in_groove])
            radial_dir[:, 0] = result[in_groove, 0]
            radial_dir[:, 1] = result[in_groove, 1]
            radial_norms = np.linalg.norm(radial_dir, axis=1, keepdims=True)
            radial_norms[radial_norms == 0] = 1
            radial_dir /= radial_norms

            # Smooth groove profile (cosine)
            z_dist = np.abs(result[in_groove, 2] - z_center)
            depth_factor = np.cos(np.pi * z_dist / groove_width)
            depth_factor = np.maximum(depth_factor, 0)

            result[in_groove] -= (groove_depth * depth_factor)[:, None] * radial_dir

    return result
