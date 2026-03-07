"""Extended geometric operations for non-revolve objects.

Provides extrusion, boolean approximations, circular arrays, polygon
sweeps, and other operations needed for mechanical/structural parts.
"""

import math
import numpy as np
from .builder import combine_meshes, revolve_profile


def extrude_polygon(polygon_xy, height, n_height=2):
    """Extrude a 2D polygon along the Z axis.

    Args:
        polygon_xy: list of (x, y) tuples defining the polygon (CCW order)
        height: extrusion height (z=0 to z=height)
        n_height: number of height divisions

    Returns:
        vertices: (N, 3) array
        faces: (M, 3) array
    """
    n_pts = len(polygon_xy)
    vertices = []
    faces = []

    # Generate ring of vertices at each height level
    for i in range(n_height + 1):
        z = height * i / n_height
        for x, y in polygon_xy:
            vertices.append([x, y, z])

    # Side faces
    for i in range(n_height):
        for j in range(n_pts):
            j_next = (j + 1) % n_pts
            p00 = i * n_pts + j
            p01 = i * n_pts + j_next
            p10 = (i + 1) * n_pts + j
            p11 = (i + 1) * n_pts + j_next
            faces.append([p00, p10, p01])
            faces.append([p01, p10, p11])

    # Bottom cap (fan triangulation)
    bottom_center_idx = len(vertices)
    cx = np.mean([p[0] for p in polygon_xy])
    cy = np.mean([p[1] for p in polygon_xy])
    vertices.append([cx, cy, 0])
    for j in range(n_pts):
        j_next = (j + 1) % n_pts
        faces.append([bottom_center_idx, j_next, j])

    # Top cap
    top_center_idx = len(vertices)
    vertices.append([cx, cy, height])
    top_start = n_height * n_pts
    for j in range(n_pts):
        j_next = (j + 1) % n_pts
        faces.append([top_center_idx, top_start + j, top_start + j_next])

    return np.array(vertices, dtype=np.float64), np.array(faces)


def make_regular_polygon(n_sides, radius, center=(0, 0), start_angle=0):
    """Generate vertices of a regular polygon in 2D.

    Args:
        n_sides: number of sides
        radius: circumscribed radius
        center: (cx, cy) center point
        start_angle: rotation offset in radians

    Returns:
        list of (x, y) tuples
    """
    return [
        (center[0] + radius * math.cos(start_angle + 2 * math.pi * i / n_sides),
         center[1] + radius * math.sin(start_angle + 2 * math.pi * i / n_sides))
        for i in range(n_sides)
    ]


def make_star_polygon(n_points, outer_r, inner_r, center=(0, 0)):
    """Generate a star-shaped polygon.

    Args:
        n_points: number of star points
        outer_r: radius to star tips
        inner_r: radius to star valleys
        center: center point

    Returns:
        list of (x, y) tuples
    """
    pts = []
    for i in range(n_points * 2):
        angle = math.pi / 2 + 2 * math.pi * i / (n_points * 2)
        r = outer_r if i % 2 == 0 else inner_r
        pts.append((center[0] + r * math.cos(angle),
                     center[1] + r * math.sin(angle)))
    return pts


def circular_array(mesh_vf, n_copies, axis_z=True):
    """Replicate a mesh in a circular array around the Z axis.

    Args:
        mesh_vf: (vertices, faces) tuple
        n_copies: number of copies (including the original)
        axis_z: if True, rotate around Z axis

    Returns:
        combined (vertices, faces)
    """
    verts, faces = mesh_vf
    meshes = []
    for i in range(n_copies):
        angle = 2 * math.pi * i / n_copies
        cos_a, sin_a = math.cos(angle), math.sin(angle)
        rot = np.array([
            [cos_a, -sin_a, 0],
            [sin_a, cos_a, 0],
            [0, 0, 1],
        ])
        rotated_verts = verts @ rot.T
        meshes.append((rotated_verts, faces.copy()))
    return combine_meshes(meshes)


def make_cylinder_at(x, y, z_bottom, z_top, radius, n_angular=16, n_height=2):
    """Create a cylinder at an arbitrary XY position."""
    profile = []
    for i in range(n_height + 1):
        z = z_bottom + (z_top - z_bottom) * i / n_height
        profile.append((radius, z))

    verts, faces = revolve_profile(profile, n_angular)
    verts[:, 0] += x
    verts[:, 1] += y
    return verts, faces


def subtract_cylinders(base_verts, base_faces, holes):
    """Approximate boolean subtraction of cylinders from a mesh.

    Since true boolean is complex, we use vertex displacement:
    vertices inside any hole cylinder are pushed to the nearest hole wall.

    Args:
        base_verts: (N, 3) base mesh vertices
        base_faces: (M, 3) base mesh faces
        holes: list of (cx, cy, radius, z_bottom, z_top) hole definitions

    Returns:
        modified_verts: (N, 3) with vertices displaced away from holes
    """
    result = base_verts.copy()
    for cx, cy, radius, z_bot, z_top in holes:
        dx = result[:, 0] - cx
        dy = result[:, 1] - cy
        dist_xy = np.sqrt(dx ** 2 + dy ** 2)

        # Find vertices inside the cylinder
        inside = (dist_xy < radius) & (result[:, 2] >= z_bot) & (result[:, 2] <= z_top)

        if np.any(inside):
            # Push vertices outward to the cylinder wall
            scale = radius / np.maximum(dist_xy[inside], 0.01)
            result[inside, 0] = cx + dx[inside] * scale
            result[inside, 1] = cy + dy[inside] * scale

    return result


def make_rectangular_frame(width, height, depth, thickness, profile_pts=None):
    """Create a rectangular frame by sweeping a profile along a rectangle.

    If no profile is given, uses a simple rectangular cross-section.

    Args:
        width: outer width (X direction)
        height: outer height (Y direction)
        depth: depth of the frame (Z direction)
        thickness: frame member thickness
        profile_pts: optional cross-section profile as list of (offset, depth) tuples

    Returns:
        vertices, faces
    """
    # Build four sides as extruded rectangles, mitered at corners
    meshes = []
    hw, hh = width / 2, height / 2
    t = thickness

    if profile_pts is None:
        profile_pts = [(0, 0), (t, 0), (t, depth), (0, depth)]

    # Bottom rail
    n = 20
    for seg_idx, (start, end, is_x) in enumerate([
        ((-hw, -hh), (hw, -hh), True),   # bottom
        ((hw, -hh), (hw, hh), False),     # right
        ((hw, hh), (-hw, hh), True),      # top
        ((-hw, hh), (-hw, -hh), False),   # left
    ]):
        sx, sy = start
        ex, ey = end
        rail_verts = []
        rail_faces = []
        n_profile = len(profile_pts)

        for i in range(n + 1):
            frac = i / n
            px = sx + (ex - sx) * frac
            py = sy + (ey - sy) * frac

            for offset, d in profile_pts:
                if is_x:
                    # Rail along X: profile extends inward in Y, depth in Z
                    if sy < 0:
                        rail_verts.append([px, py + offset, d])
                    else:
                        rail_verts.append([px, py - offset, d])
                else:
                    if sx > 0:
                        rail_verts.append([px - offset, py, d])
                    else:
                        rail_verts.append([px + offset, py, d])

        rail_verts = np.array(rail_verts, dtype=np.float64)

        for i in range(n):
            for j in range(n_profile):
                j_next = (j + 1) % n_profile
                p00 = i * n_profile + j
                p01 = i * n_profile + j_next
                p10 = (i + 1) * n_profile + j
                p11 = (i + 1) * n_profile + j_next
                rail_faces.append([p00, p10, p01])
                rail_faces.append([p01, p10, p11])

        meshes.append((rail_verts, np.array(rail_faces)))

    return combine_meshes(meshes)


def make_involute_gear_profile(module, n_teeth, pressure_angle_deg=20):
    """Generate the 2D profile of an involute spur gear.

    Args:
        module: gear module (pitch diameter / number of teeth)
        n_teeth: number of teeth
        pressure_angle_deg: pressure angle in degrees

    Returns:
        list of (x, y) tuples defining the gear outer profile
    """
    pa = math.radians(pressure_angle_deg)
    pitch_r = module * n_teeth / 2
    base_r = pitch_r * math.cos(pa)
    addendum = module
    dedendum = 1.25 * module
    outer_r = pitch_r + addendum
    root_r = max(pitch_r - dedendum, 0.5)

    pts = []
    # For each tooth, generate the profile
    tooth_angle = 2 * math.pi / n_teeth

    for t_idx in range(n_teeth):
        base_angle = t_idx * tooth_angle

        # Root arc
        n_root = 3
        root_start = base_angle - tooth_angle * 0.35
        root_end = base_angle - tooth_angle * 0.15
        for i in range(n_root):
            a = root_start + (root_end - root_start) * i / (n_root - 1)
            pts.append((root_r * math.cos(a), root_r * math.sin(a)))

        # Rising flank (involute approximation)
        n_flank = 5
        for i in range(n_flank):
            frac = i / (n_flank - 1)
            r = root_r + (outer_r - root_r) * frac
            # Involute offset
            if r > base_r:
                inv_angle = math.sqrt((r / base_r) ** 2 - 1) - math.acos(base_r / r)
            else:
                inv_angle = 0
            a = base_angle - tooth_angle * 0.15 + inv_angle * 0.5
            pts.append((r * math.cos(a), r * math.sin(a)))

        # Tooth tip arc
        n_tip = 3
        tip_half = tooth_angle * 0.08
        tip_center = base_angle + tooth_angle * 0.0
        for i in range(n_tip):
            a = tip_center - tip_half + 2 * tip_half * i / (n_tip - 1)
            pts.append((outer_r * math.cos(a), outer_r * math.sin(a)))

        # Falling flank
        for i in range(n_flank):
            frac = 1 - i / (n_flank - 1)
            r = root_r + (outer_r - root_r) * frac
            if r > base_r:
                inv_angle = math.sqrt((r / base_r) ** 2 - 1) - math.acos(base_r / r)
            else:
                inv_angle = 0
            a = base_angle + tooth_angle * 0.15 - inv_angle * 0.5
            pts.append((r * math.cos(a), r * math.sin(a)))

        # Root arc (trailing)
        root_start2 = base_angle + tooth_angle * 0.15
        root_end2 = base_angle + tooth_angle * 0.35
        for i in range(n_root):
            a = root_start2 + (root_end2 - root_start2) * i / (n_root - 1)
            pts.append((root_r * math.cos(a), root_r * math.sin(a)))

    return pts


def make_knurled_surface(base_verts, radius, z_bottom, z_top,
                          n_knurls=24, knurl_depth=0.5):
    """Apply diamond knurling pattern to cylindrical vertices.

    Displaces vertices radially based on a diamond-pattern function.

    Args:
        base_verts: (N, 3) vertices to modify
        radius: cylinder radius
        z_bottom, z_top: Z range of the knurled section
        n_knurls: number of knurl ridges circumferentially
        knurl_depth: depth of knurl grooves

    Returns:
        modified_verts: (N, 3)
    """
    result = base_verts.copy()
    for i in range(len(result)):
        x, y, z = result[i]
        r = math.sqrt(x ** 2 + y ** 2)
        if abs(r - radius) < 1.0 and z_bottom <= z <= z_top:
            angle = math.atan2(y, x)
            # Diamond knurl: two crossing helical patterns
            z_norm = (z - z_bottom) / (z_top - z_bottom)
            pattern1 = math.sin(n_knurls * angle + z_norm * 20)
            pattern2 = math.sin(n_knurls * angle - z_norm * 20)
            knurl = max(pattern1, pattern2) * knurl_depth
            # Adjust radius
            new_r = radius + knurl
            if r > 0:
                result[i, 0] = x * new_r / r
                result[i, 1] = y * new_r / r
    return result
