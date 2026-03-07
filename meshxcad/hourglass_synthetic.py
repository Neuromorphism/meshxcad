"""Synthetic hourglass geometry using pure numpy (no FreeCAD required).

Creates numpy mesh representations of both ornate and simple hourglasses
for testing the detail transfer pipeline without FreeCAD.
"""

import math
import numpy as np


# Shared parameters
GLASS_HEIGHT = 120.0
GLASS_BULB_RADIUS = 30.0
GLASS_NECK_RADIUS = 4.0
PLATE_THICKNESS = 8.0
PLATE_RADIUS = 45.0
NUM_PILLARS = 4
PILLAR_ORBIT_RADIUS = 38.0
PILLAR_RADIUS = 5.0
PILLAR_HEIGHT = GLASS_HEIGHT + 2 * PLATE_THICKNESS


def _revolve_profile(profile_rz, n_angular=48):
    """Revolve a 2D profile (r, z) pairs around the Z axis.

    Args:
        profile_rz: list of (radius, z_height) tuples
        n_angular: angular divisions

    Returns:
        vertices: (N, 3) array
        faces: (M, 3) array
    """
    n_profile = len(profile_rz)
    vertices = []
    faces = []

    for i, (r, z) in enumerate(profile_rz):
        for j in range(n_angular):
            angle = 2 * math.pi * j / n_angular
            x = r * math.cos(angle)
            y = r * math.sin(angle)
            vertices.append([x, y, z])

    vertices = np.array(vertices)

    for i in range(n_profile - 1):
        for j in range(n_angular):
            j_next = (j + 1) % n_angular
            p00 = i * n_angular + j
            p01 = i * n_angular + j_next
            p10 = (i + 1) * n_angular + j
            p11 = (i + 1) * n_angular + j_next
            faces.append([p00, p01, p10])
            faces.append([p01, p11, p10])

    return vertices, np.array(faces)


def _make_cylinder_mesh(radius, height, x, y, z_base, n_angular=16, n_height=8):
    """Create a cylinder mesh at a given position."""
    profile = []
    for i in range(n_height + 1):
        z = z_base + height * i / n_height
        profile.append((radius, z))

    verts, faces = _revolve_profile(profile, n_angular)
    verts[:, 0] += x
    verts[:, 1] += y
    return verts, faces


def _make_disk_mesh(radius, z, n_radial=8, n_angular=48):
    """Create a flat disk mesh at height z."""
    vertices = [[0, 0, z]]  # center
    faces = []

    for i in range(1, n_radial + 1):
        r = radius * i / n_radial
        for j in range(n_angular):
            angle = 2 * math.pi * j / n_angular
            vertices.append([r * math.cos(angle), r * math.sin(angle), z])

    vertices = np.array(vertices)

    # Center fan
    for j in range(n_angular):
        j_next = (j + 1) % n_angular
        faces.append([0, 1 + j, 1 + j_next])

    # Rings
    for i in range(n_radial - 1):
        for j in range(n_angular):
            j_next = (j + 1) % n_angular
            p00 = 1 + i * n_angular + j
            p01 = 1 + i * n_angular + j_next
            p10 = 1 + (i + 1) * n_angular + j
            p11 = 1 + (i + 1) * n_angular + j_next
            faces.append([p00, p01, p10])
            faces.append([p01, p11, p10])

    return vertices, np.array(faces)


def _glass_profile_simple(n_points=20):
    """Generate simple glass bulb profile (fewer control points, smooth).

    Classic hourglass shape: narrow waist, full round bulbs.
    """
    half = GLASS_HEIGHT / 2
    profile = []

    for i in range(n_points + 1):
        t = i / n_points
        z = -half + GLASS_HEIGHT * t
        z_norm = abs(z) / half  # 0 at center, 1 at ends

        # Classic hourglass: tight waist, round bulbs
        # Use a power curve that gives a pronounced pinch at center
        bulge = math.sin(math.pi * z_norm) ** 0.6
        r = GLASS_NECK_RADIUS + (GLASS_BULB_RADIUS - GLASS_NECK_RADIUS) * bulge

        profile.append((r, z))

    return profile


def _glass_profile_ornate(n_points=40):
    """Generate ornate glass bulb profile (more detail, shaped curves).

    Distinguished from simple by:
    - Sharper waist transition
    - Slightly flattened bulb tops (shoulder region)
    - More pronounced maximum diameter
    """
    half = GLASS_HEIGHT / 2
    profile = []

    for i in range(n_points + 1):
        t = i / n_points
        z = -half + GLASS_HEIGHT * t
        z_norm = abs(z) / half  # 0 at center, 1 at ends

        # Ornate profile: sharper waist, fuller bulbs, defined shoulder
        if z_norm < 0.08:
            # Very narrow neck region — sharp transition
            r = GLASS_NECK_RADIUS + 3 * (z_norm / 0.08) ** 0.4
        elif z_norm < 0.3:
            # Rapid flare to full bulb
            local_t = (z_norm - 0.08) / 0.22
            r_start = GLASS_NECK_RADIUS + 3
            r = r_start + (GLASS_BULB_RADIUS * 1.05 - r_start) * math.sin(math.pi * local_t / 2)
        elif z_norm < 0.7:
            # Full bulb region — nearly constant max radius
            local_t = (z_norm - 0.3) / 0.4
            r = GLASS_BULB_RADIUS * 1.05 * (1 - 0.03 * math.sin(math.pi * local_t))
        elif z_norm < 0.88:
            # Shoulder contraction
            local_t = (z_norm - 0.7) / 0.18
            r = GLASS_BULB_RADIUS * 1.05 * (1 - 0.35 * local_t ** 1.2)
        else:
            # Near plate: taper to meet plate
            local_t = (z_norm - 0.88) / 0.12
            r_start = GLASS_BULB_RADIUS * 0.7
            r_end = GLASS_NECK_RADIUS + 8
            r = r_start + (r_end - r_start) * local_t

        profile.append((r, z))

    return profile


def _pillar_profile_simple(n_points=10):
    """Simple straight cylindrical pillar profile."""
    z_base = -GLASS_HEIGHT / 2 - PLATE_THICKNESS
    profile = []
    for i in range(n_points + 1):
        z = z_base + PILLAR_HEIGHT * i / n_points
        profile.append((PILLAR_RADIUS, z))
    return profile


def _pillar_profile_ornate(n_points=60):
    """Ornate turned pillar profile with beads, coves, and vase shape."""
    z_base = -GLASS_HEIGHT / 2 - PLATE_THICKNESS
    r = PILLAR_RADIUS
    profile = []

    for i in range(n_points + 1):
        t = i / n_points
        z = z_base + PILLAR_HEIGHT * t

        # Base flare (0-5%)
        if t < 0.05:
            local_t = t / 0.05
            radius = r * 1.4 * (1 - 0.5 * local_t)
        # Base cove and bead (5-12%)
        elif t < 0.12:
            local_t = (t - 0.05) / 0.07
            radius = r * (0.7 + 0.5 * math.sin(math.pi * local_t))
        # Lower vase section (12-45%): tapers inward with a bead at 30%
        elif t < 0.45:
            mid_t = (t - 0.12) / 0.33
            # Overall vase taper
            base_r = r * (0.85 - 0.25 * math.sin(math.pi * mid_t))
            # Bead at ~30%
            bead = 0.15 * r * math.exp(-((mid_t - 0.55) ** 2) / 0.02)
            radius = base_r + bead
        # Center bulge/ring (45-55%)
        elif t < 0.55:
            local_t = (t - 0.45) / 0.1
            radius = r * (0.6 + 0.5 * math.sin(math.pi * local_t))
        # Upper vase section (55-88%): mirrors lower
        elif t < 0.88:
            mid_t = (t - 0.55) / 0.33
            base_r = r * (0.85 - 0.25 * math.sin(math.pi * (1 - mid_t)))
            bead = 0.15 * r * math.exp(-((mid_t - 0.45) ** 2) / 0.02)
            radius = base_r + bead
        # Top bead and cove (88-95%)
        elif t < 0.95:
            local_t = (t - 0.88) / 0.07
            radius = r * (0.7 + 0.5 * math.sin(math.pi * (1 - local_t)))
        # Top flare (95-100%)
        else:
            local_t = (t - 0.95) / 0.05
            radius = r * (0.7 + 0.7 * local_t)

        profile.append((max(radius, 1.0), z))

    return profile


def _plate_profile_simple(z_base):
    """Simple flat plate profile."""
    return [
        (PLATE_RADIUS, z_base),
        (PLATE_RADIUS, z_base + PLATE_THICKNESS),
    ]


def _plate_profile_ornate(z_base, n_points=20):
    """Ornate plate profile with stepped molding."""
    profile = []
    t = PLATE_THICKNESS

    # Build from bottom to top with molding steps
    steps = [
        (0.0, PLATE_RADIUS - 8),
        (1.5, PLATE_RADIUS - 8),
        (1.5, PLATE_RADIUS - 5),
        (3.0, PLATE_RADIUS - 5),
        (3.0, PLATE_RADIUS - 2.5),
        (4.5, PLATE_RADIUS),
        (t - 4.5, PLATE_RADIUS),
        (t - 3.0, PLATE_RADIUS - 2.5),
        (t - 3.0, PLATE_RADIUS - 5),
        (t - 1.5, PLATE_RADIUS - 5),
        (t - 1.5, PLATE_RADIUS - 8),
        (t, PLATE_RADIUS - 8),
    ]

    for dz, r in steps:
        profile.append((r, z_base + dz))

    return profile


def _combine_meshes(mesh_list):
    """Combine multiple (vertices, faces) tuples into one mesh."""
    all_verts = []
    all_faces = []
    offset = 0

    for verts, faces in mesh_list:
        all_verts.append(verts)
        all_faces.append(faces + offset)
        offset += len(verts)

    return np.vstack(all_verts), np.vstack(all_faces)


def make_simple_hourglass_mesh(n_angular=48):
    """Create simplified hourglass mesh (no decorative features).

    Returns:
        vertices: (N, 3) array
        faces: (M, 3) array
    """
    meshes = []

    # Glass body
    glass_profile = _glass_profile_simple()
    meshes.append(_revolve_profile(glass_profile, n_angular))

    # Bottom plate (cylinder approximation using two rings)
    z_bot = -GLASS_HEIGHT / 2 - PLATE_THICKNESS
    plate_profile = _plate_profile_simple(z_bot)
    meshes.append(_revolve_profile(plate_profile, n_angular))

    # Top plate
    z_top = GLASS_HEIGHT / 2
    plate_profile = _plate_profile_simple(z_top)
    meshes.append(_revolve_profile(plate_profile, n_angular))

    # Bottom plate caps
    meshes.append(_make_disk_mesh(PLATE_RADIUS, z_bot, n_angular=n_angular))
    meshes.append(_make_disk_mesh(PLATE_RADIUS, z_bot + PLATE_THICKNESS, n_angular=n_angular))
    meshes.append(_make_disk_mesh(PLATE_RADIUS, z_top, n_angular=n_angular))
    meshes.append(_make_disk_mesh(PLATE_RADIUS, z_top + PLATE_THICKNESS, n_angular=n_angular))

    # Pillars
    pillar_profile = _pillar_profile_simple()
    for i in range(NUM_PILLARS):
        angle = math.radians(45 + i * 90)
        x = PILLAR_ORBIT_RADIUS * math.cos(angle)
        y = PILLAR_ORBIT_RADIUS * math.sin(angle)
        pv, pf = _revolve_profile(pillar_profile, 12)
        pv[:, 0] += x
        pv[:, 1] += y
        meshes.append((pv, pf))

    return _combine_meshes(meshes)


def make_ornate_hourglass_mesh(n_angular=48):
    """Create ornate hourglass mesh (with decorative features).

    Returns:
        vertices: (N, 3) array
        faces: (M, 3) array
    """
    meshes = []

    # Ornate glass body
    glass_profile = _glass_profile_ornate()
    meshes.append(_revolve_profile(glass_profile, n_angular))

    # Ornate bottom plate
    z_bot = -GLASS_HEIGHT / 2 - PLATE_THICKNESS
    plate_profile = _plate_profile_ornate(z_bot)
    meshes.append(_revolve_profile(plate_profile, n_angular))

    # Ornate top plate
    z_top = GLASS_HEIGHT / 2
    plate_profile = _plate_profile_ornate(z_top)
    meshes.append(_revolve_profile(plate_profile, n_angular))

    # Plate caps (inner part)
    meshes.append(_make_disk_mesh(PLATE_RADIUS - 8, z_bot, n_angular=n_angular))
    meshes.append(_make_disk_mesh(PLATE_RADIUS - 8, z_bot + PLATE_THICKNESS, n_angular=n_angular))
    meshes.append(_make_disk_mesh(PLATE_RADIUS - 8, z_top, n_angular=n_angular))
    meshes.append(_make_disk_mesh(PLATE_RADIUS - 8, z_top + PLATE_THICKNESS, n_angular=n_angular))

    # Ornate turned pillars
    pillar_profile = _pillar_profile_ornate()
    for i in range(NUM_PILLARS):
        angle = math.radians(45 + i * 90)
        x = PILLAR_ORBIT_RADIUS * math.cos(angle)
        y = PILLAR_ORBIT_RADIUS * math.sin(angle)
        pv, pf = _revolve_profile(pillar_profile, 16)
        pv[:, 0] += x
        pv[:, 1] += y
        meshes.append((pv, pf))

    # Decorative rings
    ring_positions = [
        (-GLASS_HEIGHT / 2, GLASS_NECK_RADIUS + 6, GLASS_NECK_RADIUS + 10),  # bottom
        (0, GLASS_NECK_RADIUS + 1, GLASS_NECK_RADIUS + 4),                    # center
        (GLASS_HEIGHT / 2, GLASS_NECK_RADIUS + 6, GLASS_NECK_RADIUS + 10),   # top
    ]
    for z_pos, r_inner, r_outer in ring_positions:
        ring_profile = _make_torus_profile(r_inner, r_outer, z_pos)
        meshes.append(_revolve_profile(ring_profile, n_angular))

    # Finial on top
    finial_profile = _make_finial_profile()
    meshes.append(_revolve_profile(finial_profile, n_angular))

    return _combine_meshes(meshes)


def _make_torus_profile(r_inner, r_outer, z_center, n_cross=12):
    """Create a torus-like ring profile at z_center."""
    r_mid = (r_inner + r_outer) / 2
    r_tube = (r_outer - r_inner) / 2
    profile = []
    for i in range(n_cross + 1):
        angle = 2 * math.pi * i / n_cross
        r = r_mid + r_tube * math.cos(angle)
        z = z_center + r_tube * math.sin(angle)
        profile.append((r, z))
    return profile


def _make_finial_profile(n_points=15):
    """Create a finial profile for the top of the hourglass."""
    z_base = GLASS_HEIGHT / 2 + PLATE_THICKNESS
    finial_height = 15.0
    finial_radius = 6.0

    profile = [(0.1, z_base)]  # near-axis start

    for i in range(1, n_points + 1):
        t = i / n_points  # 0 to 1
        z = z_base + finial_height * t

        # Shaped profile: bulge then taper to point
        if t < 0.25:
            r = finial_radius * math.sin(math.pi * t / 0.5) * 0.8
        elif t < 0.5:
            local_t = (t - 0.25) / 0.25
            r = finial_radius * (0.8 + 0.2 * math.sin(math.pi * local_t))
        elif t < 0.75:
            local_t = (t - 0.5) / 0.25
            r = finial_radius * (1.0 - 0.4 * local_t)
        else:
            local_t = (t - 0.75) / 0.25
            r = finial_radius * 0.6 * (1 - local_t) ** 1.5

        profile.append((max(r, 0.2), z))

    profile.append((0.1, z_base + finial_height + 1))  # tip
    return profile
