"""Parametric cephalopod mesh generators.

Creates diverse octopus, squid, and Cthulhu-like meshes with tentacles
for stress-testing mesh-to-CAD reconstruction on non-axisymmetric organic shapes.
"""

import math
import numpy as np


def _make_tube(path_points, radii, n_angular=12):
    """Create a tube mesh along a 3D path with varying radius.

    Args:
        path_points: (N, 3) array of 3D points along the path
        radii: (N,) array of radii at each path point
        n_angular: number of angular divisions around the tube

    Returns:
        vertices: (M, 3) array
        faces: (K, 3) array
    """
    path = np.array(path_points, dtype=np.float64)
    radii = np.array(radii, dtype=np.float64)
    n_path = len(path)

    # Compute tangent vectors along the path
    tangents = np.zeros_like(path)
    tangents[0] = path[1] - path[0]
    tangents[-1] = path[-1] - path[-2]
    for i in range(1, n_path - 1):
        tangents[i] = path[i + 1] - path[i - 1]

    # Normalize tangents
    for i in range(n_path):
        norm = np.linalg.norm(tangents[i])
        if norm > 1e-12:
            tangents[i] /= norm

    # Build local coordinate frames using parallel transport
    # Start with an arbitrary normal
    t0 = tangents[0]
    if abs(t0[0]) < 0.9:
        up = np.array([1.0, 0.0, 0.0])
    else:
        up = np.array([0.0, 1.0, 0.0])

    normals = np.zeros_like(path)
    binormals = np.zeros_like(path)

    normals[0] = np.cross(t0, up)
    normals[0] /= np.linalg.norm(normals[0])
    binormals[0] = np.cross(t0, normals[0])

    # Parallel transport
    for i in range(1, n_path):
        b = np.cross(tangents[i - 1], tangents[i])
        b_norm = np.linalg.norm(b)
        if b_norm < 1e-12:
            normals[i] = normals[i - 1]
            binormals[i] = binormals[i - 1]
        else:
            b /= b_norm
            angle = math.acos(max(-1, min(1, np.dot(tangents[i - 1], tangents[i]))))
            cos_a, sin_a = math.cos(angle), math.sin(angle)
            n_prev = normals[i - 1]
            # Rodrigues rotation
            normals[i] = (n_prev * cos_a +
                          np.cross(b, n_prev) * sin_a +
                          b * np.dot(b, n_prev) * (1 - cos_a))
            norm = np.linalg.norm(normals[i])
            if norm > 1e-12:
                normals[i] /= norm
            binormals[i] = np.cross(tangents[i], normals[i])

    # Generate ring vertices
    vertices = []
    for i in range(n_path):
        center = path[i]
        n = normals[i]
        bn = binormals[i]
        r = radii[i]
        for j in range(n_angular):
            angle = 2 * math.pi * j / n_angular
            v = center + r * (math.cos(angle) * n + math.sin(angle) * bn)
            vertices.append(v)

    # Tip vertex (close the end)
    vertices.append(path[-1])
    tip_idx = len(vertices) - 1

    vertices = np.array(vertices, dtype=np.float64)

    # Side faces
    faces = []
    for i in range(n_path - 1):
        for j in range(n_angular):
            j_next = (j + 1) % n_angular
            p00 = i * n_angular + j
            p01 = i * n_angular + j_next
            p10 = (i + 1) * n_angular + j
            p11 = (i + 1) * n_angular + j_next
            faces.append([p00, p10, p01])
            faces.append([p01, p10, p11])

    # Bottom cap
    bottom_center_idx = len(vertices)
    vertices = np.vstack([vertices, [path[0]]])
    for j in range(n_angular):
        j_next = (j + 1) % n_angular
        faces.append([bottom_center_idx, j, j_next])

    # Top cap (taper to tip)
    top_ring_start = (n_path - 1) * n_angular
    for j in range(n_angular):
        j_next = (j + 1) % n_angular
        faces.append([tip_idx, top_ring_start + j_next, top_ring_start + j])

    return vertices, np.array(faces, dtype=np.int64)


def _combine(mesh_list):
    """Combine multiple (vertices, faces) into a single mesh."""
    if not mesh_list:
        return np.zeros((0, 3)), np.zeros((0, 3), dtype=np.int64)
    all_v, all_f = [], []
    offset = 0
    for v, f in mesh_list:
        all_v.append(v)
        all_f.append(f + offset)
        offset += len(v)
    return np.vstack(all_v), np.vstack(all_f)


def _make_sphere(center, radius, n_lat=12, n_lon=16):
    """Create a UV sphere mesh."""
    verts = []
    faces = []

    # Top pole
    verts.append(center + np.array([0, 0, radius]))

    # Latitude rings
    for i in range(1, n_lat):
        phi = math.pi * i / n_lat
        z = radius * math.cos(phi)
        r = radius * math.sin(phi)
        for j in range(n_lon):
            theta = 2 * math.pi * j / n_lon
            verts.append(center + np.array([r * math.cos(theta),
                                            r * math.sin(theta), z]))

    # Bottom pole
    verts.append(center + np.array([0, 0, -radius]))

    verts = np.array(verts, dtype=np.float64)

    # Top cap faces
    for j in range(n_lon):
        j_next = (j + 1) % n_lon
        faces.append([0, 1 + j, 1 + j_next])

    # Mid faces
    for i in range(n_lat - 2):
        for j in range(n_lon):
            j_next = (j + 1) % n_lon
            p00 = 1 + i * n_lon + j
            p01 = 1 + i * n_lon + j_next
            p10 = 1 + (i + 1) * n_lon + j
            p11 = 1 + (i + 1) * n_lon + j_next
            faces.append([p00, p10, p01])
            faces.append([p01, p10, p11])

    # Bottom cap faces
    bottom_idx = len(verts) - 1
    bottom_ring = 1 + (n_lat - 2) * n_lon
    for j in range(n_lon):
        j_next = (j + 1) % n_lon
        faces.append([bottom_idx, bottom_ring + j_next, bottom_ring + j])

    return verts, np.array(faces, dtype=np.int64)


def _tentacle_path(base_pos, direction, length, curl_amount=0.3,
                   n_points=20, wave_freq=1.5, wave_amp=0.0):
    """Generate a curling tentacle path.

    Args:
        base_pos: (3,) starting position
        direction: (3,) initial direction (will be normalized)
        length: total tentacle length
        curl_amount: how much the tentacle curls downward
        n_points: number of path points
        wave_freq: frequency of lateral wave
        wave_amp: amplitude of lateral wave

    Returns:
        (n_points, 3) array of path positions
    """
    base = np.array(base_pos, dtype=np.float64)
    d = np.array(direction, dtype=np.float64)
    d /= np.linalg.norm(d)

    # Build a local frame
    if abs(d[2]) < 0.9:
        up = np.array([0, 0, 1.0])
    else:
        up = np.array([1.0, 0, 0])
    lateral = np.cross(d, up)
    lateral /= np.linalg.norm(lateral)
    up = np.cross(lateral, d)

    points = [base.copy()]
    pos = base.copy()
    step = length / n_points

    for i in range(1, n_points):
        t = i / n_points

        # Curl: direction gradually tilts downward
        curl_dir = d * (1 - curl_amount * t) + np.array([0, 0, -1]) * curl_amount * t
        curl_dir /= np.linalg.norm(curl_dir)

        # Wave: sinusoidal lateral oscillation
        wave_offset = lateral * wave_amp * math.sin(wave_freq * 2 * math.pi * t)

        pos = pos + curl_dir * step + wave_offset * step
        points.append(pos.copy())

    return np.array(points)


def make_octopus(body_radius=15.0, n_tentacles=8, tentacle_length=40.0,
                 tentacle_base_radius=4.0, curl=0.4, wave=0.1,
                 n_angular=10, n_path_pts=16):
    """Generate an octopus mesh with body dome and tentacles.

    Returns:
        vertices: (N, 3) array
        faces: (M, 3) array
    """
    meshes = []

    # Body: oblate sphere (dome)
    body = _make_sphere(np.array([0, 0, body_radius * 0.3]),
                        body_radius, n_lat=14, n_lon=20)
    # Squash vertically to make dome-shaped
    v = body[0].copy()
    mask_top = v[:, 2] > body_radius * 0.3
    v[mask_top, 2] = body_radius * 0.3 + (v[mask_top, 2] - body_radius * 0.3) * 1.3
    mask_bot = v[:, 2] < body_radius * 0.3
    v[mask_bot, 2] = body_radius * 0.3 + (v[mask_bot, 2] - body_radius * 0.3) * 0.6
    meshes.append((v, body[1]))

    # Eyes: two small spheres
    eye_r = body_radius * 0.15
    eye_z = body_radius * 0.2
    for side in [-1, 1]:
        eye_pos = np.array([body_radius * 0.5 * side, body_radius * 0.4, eye_z])
        meshes.append(_make_sphere(eye_pos, eye_r, 6, 8))

    # Tentacles
    for i in range(n_tentacles):
        angle = 2 * math.pi * i / n_tentacles
        base = np.array([
            body_radius * 0.7 * math.cos(angle),
            body_radius * 0.7 * math.sin(angle),
            -body_radius * 0.2
        ])
        direction = np.array([math.cos(angle), math.sin(angle), -0.5])

        # Vary curl and wave per tentacle for natural look
        t_curl = curl * (0.7 + 0.6 * math.sin(angle * 2.3))
        t_wave = wave * (0.5 + 1.0 * math.cos(angle * 1.7))

        path = _tentacle_path(base, direction, tentacle_length,
                              curl_amount=t_curl, n_points=n_path_pts,
                              wave_freq=1.5 + 0.5 * math.sin(angle),
                              wave_amp=t_wave)

        # Tapering radius
        radii = np.array([tentacle_base_radius * (1.0 - 0.85 * (j / n_path_pts) ** 0.6)
                          for j in range(n_path_pts)])

        meshes.append(_make_tube(path, radii, n_angular))

    return _combine(meshes)


def make_squid(mantle_length=30.0, mantle_radius=8.0, n_arms=8,
               arm_length=25.0, n_tentacles=2, tentacle_length=45.0,
               fin_size=10.0, n_angular=10, n_path_pts=16):
    """Generate a squid mesh with elongated mantle, fins, arms, and two long tentacles.

    Returns:
        vertices: (N, 3) array
        faces: (M, 3) array
    """
    meshes = []

    # Mantle: elongated body along Z axis
    n_rings = 20
    mantle_pts = []
    for i in range(n_rings):
        t = i / (n_rings - 1)
        z = mantle_length * t
        # Tapered cylinder shape
        r = mantle_radius * (0.6 + 0.4 * math.sin(math.pi * t)) * (1.0 - 0.3 * t * t)
        mantle_pts.append((r, z))

    from meshxcad.objects.builder import revolve_profile
    meshes.append(revolve_profile(mantle_pts, n_angular=16, close_top=True, close_bottom=True))

    # Fins: two diamond-shaped fins at the top of the mantle
    for side in [-1, 1]:
        fin_verts = []
        fin_faces = []
        base_z = mantle_length * 0.7
        center_y = side * mantle_radius * 0.8
        # Diamond shape
        fin_verts.append([0, center_y, base_z])  # inner
        fin_verts.append([0, center_y + side * fin_size, base_z])  # outer tip
        fin_verts.append([0, center_y + side * fin_size * 0.5, base_z + fin_size * 0.6])  # top
        fin_verts.append([0, center_y + side * fin_size * 0.5, base_z - fin_size * 0.6])  # bottom
        fin_faces.append([0, 1, 2])
        fin_faces.append([0, 2, 1])  # double-sided
        fin_faces.append([0, 3, 1])
        fin_faces.append([0, 1, 3])  # double-sided
        meshes.append((np.array(fin_verts, dtype=np.float64),
                       np.array(fin_faces, dtype=np.int64)))

    # Short arms at bottom
    for i in range(n_arms):
        angle = 2 * math.pi * i / n_arms
        base = np.array([
            mantle_radius * 0.5 * math.cos(angle),
            mantle_radius * 0.5 * math.sin(angle),
            -2.0
        ])
        direction = np.array([math.cos(angle) * 0.3, math.sin(angle) * 0.3, -1.0])
        path = _tentacle_path(base, direction, arm_length,
                              curl_amount=0.3 + 0.2 * math.sin(angle * 3),
                              n_points=n_path_pts, wave_freq=2.0, wave_amp=0.05)
        radii = np.array([3.0 * (1.0 - 0.9 * (j / n_path_pts) ** 0.5)
                          for j in range(n_path_pts)])
        meshes.append(_make_tube(path, radii, n_angular))

    # Two long feeding tentacles
    for i in range(n_tentacles):
        angle = math.pi * i  # opposite sides
        base = np.array([
            mantle_radius * 0.3 * math.cos(angle),
            mantle_radius * 0.3 * math.sin(angle),
            -3.0
        ])
        direction = np.array([math.cos(angle) * 0.2, math.sin(angle) * 0.2, -1.0])
        path = _tentacle_path(base, direction, tentacle_length,
                              curl_amount=0.15, n_points=n_path_pts + 4,
                              wave_freq=1.0, wave_amp=0.08)

        n_pts = n_path_pts + 4
        # Thin stalk with club-shaped end
        radii = np.array([
            1.5 * (1.0 - 0.6 * (j / n_pts)) +
            2.0 * max(0, math.sin(math.pi * max(0, (j / n_pts - 0.7) / 0.3)))
            for j in range(n_pts)
        ])
        meshes.append(_make_tube(path, radii, n_angular))

    return _combine(meshes)


def make_cthulhu(body_height=35.0, body_radius=12.0, n_face_tentacles=6,
                 n_arm_tentacles=4, wing_span=25.0, n_angular=10, n_path_pts=14):
    """Generate a Cthulhu-like mesh: humanoid body, face tentacles, wings.

    Returns:
        vertices: (N, 3) array
        faces: (M, 3) array
    """
    meshes = []

    # Torso: barrel-shaped body
    torso_pts = []
    for i in range(16):
        t = i / 15
        z = body_height * t
        r = body_radius * (0.8 + 0.2 * math.sin(math.pi * t))
        torso_pts.append((r, z))
    from meshxcad.objects.builder import revolve_profile
    meshes.append(revolve_profile(torso_pts, n_angular=16))

    # Head: sphere on top
    head_r = body_radius * 0.8
    head_z = body_height + head_r * 0.5
    meshes.append(_make_sphere(np.array([0, 0, head_z]), head_r, 10, 14))

    # Face tentacles (beard)
    for i in range(n_face_tentacles):
        angle = -math.pi / 4 + (math.pi / 2) * i / max(1, n_face_tentacles - 1)
        base = np.array([
            head_r * 0.6 * math.cos(angle),
            head_r * 0.8,
            head_z - head_r * 0.3
        ])
        direction = np.array([
            math.cos(angle) * 0.2,
            0.6,
            -0.5
        ])
        path = _tentacle_path(base, direction, 20.0,
                              curl_amount=0.5 + 0.3 * math.sin(angle * 2),
                              n_points=n_path_pts, wave_freq=2.0,
                              wave_amp=0.15)
        radii = np.array([2.5 * (1.0 - 0.85 * (j / n_path_pts) ** 0.5)
                          for j in range(n_path_pts)])
        meshes.append(_make_tube(path, radii, 8))

    # Arms (two thick tentacle-arms, one per side)
    for side in [-1, 1]:
        base = np.array([side * body_radius * 0.9, 0, body_height * 0.75])
        direction = np.array([side * 0.8, 0.2, -0.3])
        path = _tentacle_path(base, direction, 30.0,
                              curl_amount=0.2, n_points=n_path_pts,
                              wave_freq=1.0, wave_amp=0.1)
        radii = np.array([4.0 * (1.0 - 0.7 * (j / n_path_pts) ** 0.4)
                          for j in range(n_path_pts)])
        meshes.append(_make_tube(path, radii, n_angular))

    # Arm tentacles (smaller, from lower body)
    for i in range(n_arm_tentacles):
        angle = 2 * math.pi * i / n_arm_tentacles
        base = np.array([
            body_radius * 0.6 * math.cos(angle),
            body_radius * 0.6 * math.sin(angle),
            body_height * 0.15
        ])
        direction = np.array([math.cos(angle), math.sin(angle), -0.7])
        path = _tentacle_path(base, direction, 18.0,
                              curl_amount=0.4, n_points=n_path_pts,
                              wave_freq=1.5, wave_amp=0.08)
        radii = np.array([2.5 * (1.0 - 0.85 * (j / n_path_pts) ** 0.5)
                          for j in range(n_path_pts)])
        meshes.append(_make_tube(path, radii, 8))

    # Wings: flat triangular surfaces
    for side in [-1, 1]:
        wing_base_z = body_height * 0.65
        wing_verts = np.array([
            [side * body_radius * 0.8, 0, wing_base_z],  # inner base
            [side * (body_radius + wing_span), 0, wing_base_z + wing_span * 0.3],  # outer tip top
            [side * (body_radius + wing_span * 0.7), 0, wing_base_z - wing_span * 0.2],  # outer tip bottom
            [side * body_radius * 0.9, 0, wing_base_z + wing_span * 0.15],  # inner top
            [side * body_radius * 0.9, 0, wing_base_z - wing_span * 0.1],  # inner bottom
            # Slight thickness for double-sided
            [side * body_radius * 0.8, 1.0, wing_base_z],
            [side * (body_radius + wing_span), 1.0, wing_base_z + wing_span * 0.3],
            [side * (body_radius + wing_span * 0.7), 1.0, wing_base_z - wing_span * 0.2],
        ], dtype=np.float64)
        wing_faces = np.array([
            [0, 1, 3], [0, 2, 1], [0, 4, 2],
            [5, 7, 6], [5, 6, 7], [5, 7, 4],  # back side
        ], dtype=np.int64)
        meshes.append((wing_verts, wing_faces))

    # Legs
    for side in [-1, 1]:
        base = np.array([side * body_radius * 0.4, 0, 0])
        direction = np.array([side * 0.1, 0, -1.0])
        path = _tentacle_path(base, direction, 20.0, curl_amount=0.1,
                              n_points=12, wave_freq=0.5, wave_amp=0.02)
        radii = np.array([5.0 * (1.0 - 0.5 * j / 12) for j in range(12)])
        meshes.append(_make_tube(path, radii, n_angular))

    return _combine(meshes)


def make_jellyfish(bell_radius=12.0, n_tentacles=12, tentacle_length=35.0,
                   n_angular=10, n_path_pts=18):
    """Generate a jellyfish mesh: dome bell with long trailing tentacles.

    Returns:
        vertices: (N, 3) array
        faces: (M, 3) array
    """
    meshes = []

    # Bell (dome): half-sphere with undulating edge
    bell_pts = []
    for i in range(16):
        t = i / 15
        z = bell_radius * 0.8 * t
        if t < 0.8:
            r = bell_radius * math.sin(math.pi * t / 0.8 * 0.5)
        else:
            # Undulating rim
            r = bell_radius * (1.0 - 0.3 * (t - 0.8) / 0.2)
        bell_pts.append((max(r, 0.5), z))

    from meshxcad.objects.builder import revolve_profile
    meshes.append(revolve_profile(bell_pts, n_angular=20))

    # Oral arms (4 thick, short frilly arms)
    for i in range(4):
        angle = 2 * math.pi * i / 4
        base = np.array([
            bell_radius * 0.3 * math.cos(angle),
            bell_radius * 0.3 * math.sin(angle),
            -1.0
        ])
        direction = np.array([math.cos(angle) * 0.15, math.sin(angle) * 0.15, -1.0])
        path = _tentacle_path(base, direction, tentacle_length * 0.4,
                              curl_amount=0.2, n_points=10,
                              wave_freq=3.0, wave_amp=0.2)
        radii = np.array([3.0 * (1.0 - 0.5 * j / 10) for j in range(10)])
        meshes.append(_make_tube(path, radii, 8))

    # Trailing tentacles
    for i in range(n_tentacles):
        angle = 2 * math.pi * i / n_tentacles
        base = np.array([
            bell_radius * 0.85 * math.cos(angle),
            bell_radius * 0.85 * math.sin(angle),
            bell_radius * 0.1
        ])
        direction = np.array([math.cos(angle) * 0.1, math.sin(angle) * 0.1, -1.0])
        path = _tentacle_path(base, direction, tentacle_length,
                              curl_amount=0.15 + 0.1 * math.sin(angle * 3),
                              n_points=n_path_pts,
                              wave_freq=2.0 + math.sin(angle),
                              wave_amp=0.12)
        radii = np.array([1.2 * (1.0 - 0.9 * (j / n_path_pts) ** 0.4)
                          for j in range(n_path_pts)])
        meshes.append(_make_tube(path, radii, 6))

    return _combine(meshes)


def make_kraken(body_radius=20.0, n_tentacles=10, tentacle_length=60.0,
                n_angular=10, n_path_pts=20):
    """Generate a kraken mesh: massive bulbous body with thick, long tentacles.

    Returns:
        vertices: (N, 3) array
        faces: (M, 3) array
    """
    meshes = []

    # Massive bulbous body
    body = _make_sphere(np.array([0, 0, body_radius * 0.5]),
                        body_radius, n_lat=16, n_lon=22)
    # Elongate vertically
    v = body[0].copy()
    v[:, 2] = v[:, 2] * 1.4
    meshes.append((v, body[1]))

    # Beak-like protrusion
    beak = _make_sphere(np.array([0, body_radius * 0.4, -body_radius * 0.3]),
                        body_radius * 0.3, 8, 10)
    meshes.append(beak)

    # Eyes
    for side in [-1, 1]:
        eye_pos = np.array([side * body_radius * 0.6, body_radius * 0.5,
                            body_radius * 0.4])
        meshes.append(_make_sphere(eye_pos, body_radius * 0.12, 6, 8))

    # Massive tentacles
    for i in range(n_tentacles):
        angle = 2 * math.pi * i / n_tentacles
        base = np.array([
            body_radius * 0.8 * math.cos(angle),
            body_radius * 0.8 * math.sin(angle),
            -body_radius * 0.5
        ])
        direction = np.array([
            math.cos(angle) * 0.6,
            math.sin(angle) * 0.6,
            -0.7
        ])

        t_curl = 0.3 + 0.2 * math.sin(angle * 2.7)
        t_wave = 0.15 + 0.1 * math.cos(angle * 1.3)
        t_length = tentacle_length * (0.8 + 0.4 * abs(math.sin(angle)))

        path = _tentacle_path(base, direction, t_length,
                              curl_amount=t_curl, n_points=n_path_pts,
                              wave_freq=1.2 + 0.5 * math.sin(angle),
                              wave_amp=t_wave)
        radii = np.array([5.0 * (1.0 - 0.85 * (j / n_path_pts) ** 0.5)
                          for j in range(n_path_pts)])
        meshes.append(_make_tube(path, radii, n_angular))

    return _combine(meshes)


# Convenience: all generators with parameters
GENERATORS = {
    "octopus_basic": lambda: make_octopus(),
    "octopus_many_arms": lambda: make_octopus(n_tentacles=12, tentacle_length=35),
    "octopus_curly": lambda: make_octopus(curl=0.7, wave=0.2, tentacle_length=30),
    "octopus_long": lambda: make_octopus(tentacle_length=60, body_radius=12),
    "octopus_stubby": lambda: make_octopus(tentacle_length=20, body_radius=20,
                                            tentacle_base_radius=6),
    "squid_basic": lambda: make_squid(),
    "squid_long": lambda: make_squid(mantle_length=45, tentacle_length=60),
    "squid_compact": lambda: make_squid(mantle_length=20, arm_length=15,
                                         tentacle_length=30),
    "cthulhu_basic": lambda: make_cthulhu(),
    "cthulhu_many_tentacles": lambda: make_cthulhu(n_face_tentacles=10,
                                                     n_arm_tentacles=6),
    "jellyfish_basic": lambda: make_jellyfish(),
    "jellyfish_many": lambda: make_jellyfish(n_tentacles=20, tentacle_length=45),
    "kraken_basic": lambda: make_kraken(),
    "kraken_massive": lambda: make_kraken(body_radius=25, n_tentacles=14,
                                           tentacle_length=80),
}
