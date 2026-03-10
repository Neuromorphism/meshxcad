"""Synthetic test models for tracing/segmentation testing.

Generates procedural STL-like meshes for:
  - Dead trees with branching structure
  - Humanoid figures
  - Various CIFAR-100-relevant objects

These are used by unit tests and for development/evaluation of the
tracing pipeline.
"""

import math
import numpy as np
from .objects.builder import combine_meshes, make_cylinder, revolve_profile


# ---------------------------------------------------------------------------
# Tree generation
# ---------------------------------------------------------------------------

def _branch_mesh(start, direction, length, radius_start, radius_end,
                  n_angular=8, n_height=4):
    """Generate a tapered cylinder (branch) along an arbitrary direction."""
    # Build in local frame, then rotate
    d = np.asarray(direction, dtype=np.float64)
    d = d / max(np.linalg.norm(d), 1e-12)

    # Local Z = direction
    if abs(d[2]) < 0.9:
        ref = np.array([0, 0, 1.0])
    else:
        ref = np.array([1, 0, 0.0])
    u = np.cross(d, ref)
    u = u / max(np.linalg.norm(u), 1e-12)
    v = np.cross(d, u)
    v = v / max(np.linalg.norm(v), 1e-12)

    verts = []
    faces = []
    angles = np.linspace(0, 2 * np.pi, n_angular, endpoint=False)

    for hi in range(n_height + 1):
        t = hi / n_height
        r = radius_start + (radius_end - radius_start) * t
        center = np.asarray(start) + d * length * t
        for ai in range(n_angular):
            pt = center + r * (math.cos(angles[ai]) * u + math.sin(angles[ai]) * v)
            verts.append(pt)

    verts = np.array(verts)

    # Faces
    for hi in range(n_height):
        for ai in range(n_angular):
            a_next = (ai + 1) % n_angular
            i0 = hi * n_angular + ai
            i1 = hi * n_angular + a_next
            i2 = (hi + 1) * n_angular + ai
            i3 = (hi + 1) * n_angular + a_next
            faces.append([i0, i2, i1])
            faces.append([i1, i2, i3])

    # Bottom cap
    bc = len(verts)
    verts = np.vstack([verts, np.asarray(start)[None, :]])
    for ai in range(n_angular):
        a_next = (ai + 1) % n_angular
        faces.append([bc, a_next, ai])

    # Top cap
    tc = len(verts)
    top_center = np.asarray(start) + d * length
    verts = np.vstack([verts, top_center[None, :]])
    top_base = n_height * n_angular
    for ai in range(n_angular):
        a_next = (ai + 1) % n_angular
        faces.append([tc, top_base + ai, top_base + a_next])

    return verts, np.array(faces, dtype=np.int64)


def make_dead_tree(trunk_height=10.0, trunk_radius=0.8,
                    n_primary_branches=4, n_secondary_per_primary=3,
                    seed=42):
    """Generate a dead tree mesh with trunk and branches.

    Returns (vertices, faces) numpy arrays.
    """
    rng = np.random.RandomState(seed)
    parts = []

    # Trunk — slight taper
    trunk = _branch_mesh(
        start=[0, 0, 0],
        direction=[0, 0, 1],
        length=trunk_height,
        radius_start=trunk_radius,
        radius_end=trunk_radius * 0.6,
        n_angular=12,
        n_height=8,
    )
    parts.append(trunk)

    # Primary branches
    for i in range(n_primary_branches):
        # Branch off at various heights
        t = 0.4 + 0.5 * (i / max(n_primary_branches - 1, 1))
        branch_start = np.array([0, 0, trunk_height * t])

        angle = 2 * math.pi * i / n_primary_branches + rng.uniform(-0.3, 0.3)
        elev = rng.uniform(0.3, 0.8)  # angle from horizontal
        dx = math.cos(angle) * math.cos(elev)
        dy = math.sin(angle) * math.cos(elev)
        dz = math.sin(elev)
        direction = np.array([dx, dy, dz])

        branch_length = trunk_height * rng.uniform(0.25, 0.45)
        branch_r_start = trunk_radius * rng.uniform(0.2, 0.35)
        branch_r_end = branch_r_start * 0.3

        branch = _branch_mesh(
            start=branch_start, direction=direction,
            length=branch_length,
            radius_start=branch_r_start, radius_end=branch_r_end,
            n_angular=8, n_height=5,
        )
        parts.append(branch)

        # Secondary branches off this primary
        for j in range(n_secondary_per_primary):
            st = 0.3 + 0.5 * (j / max(n_secondary_per_primary - 1, 1))
            sub_start = branch_start + direction * branch_length * st

            sub_angle = angle + rng.uniform(-1.0, 1.0)
            sub_elev = rng.uniform(0.1, 0.6)
            sub_dx = math.cos(sub_angle) * math.cos(sub_elev)
            sub_dy = math.sin(sub_angle) * math.cos(sub_elev)
            sub_dz = math.sin(sub_elev)
            sub_dir = np.array([sub_dx, sub_dy, sub_dz])

            sub_length = branch_length * rng.uniform(0.3, 0.6)
            sub_r = branch_r_start * st * rng.uniform(0.3, 0.5)

            sub_branch = _branch_mesh(
                start=sub_start, direction=sub_dir,
                length=sub_length,
                radius_start=max(sub_r, 0.02), radius_end=max(sub_r * 0.2, 0.01),
                n_angular=6, n_height=3,
            )
            parts.append(sub_branch)

    return combine_meshes(parts)


def make_dead_tree_gnarled(seed=123):
    """A gnarled dead tree with irregular branching."""
    return make_dead_tree(
        trunk_height=8.0, trunk_radius=1.0,
        n_primary_branches=5, n_secondary_per_primary=4,
        seed=seed,
    )


def make_dead_tree_tall(seed=456):
    """A tall slender dead tree with sparse branches."""
    return make_dead_tree(
        trunk_height=15.0, trunk_radius=0.5,
        n_primary_branches=3, n_secondary_per_primary=2,
        seed=seed,
    )


def make_dead_tree_stumpy(seed=789):
    """A short thick dead tree stump with heavy branches."""
    return make_dead_tree(
        trunk_height=5.0, trunk_radius=1.2,
        n_primary_branches=6, n_secondary_per_primary=2,
        seed=seed,
    )


# ---------------------------------------------------------------------------
# Winter tree generation (healthy leafless trees with deep branching)
# ---------------------------------------------------------------------------

def _grow_branches(rng, parts, start, direction, length, radius,
                   depth, max_depth, children_range=(2, 4),
                   spread=0.7, length_decay=0.65, radius_decay=0.55,
                   n_angular=8):
    """Recursively grow branches to build a large canopy structure.

    Parameters
    ----------
    depth : current recursion depth (0 = primary branches off trunk)
    max_depth : stop recursing beyond this depth
    children_range : (min, max) number of child branches per parent
    spread : controls how wide child branches splay outward (radians)
    length_decay : child length as fraction of parent length
    radius_decay : child radius as fraction of parent radius
    """
    d = np.asarray(direction, dtype=np.float64)
    d = d / max(np.linalg.norm(d), 1e-12)

    # Reduce angular segments for finer branches
    seg = max(n_angular - depth * 2, 4)

    branch = _branch_mesh(
        start=start, direction=d,
        length=length,
        radius_start=radius, radius_end=radius * 0.65,
        n_angular=seg, n_height=max(4 - depth, 2),
    )
    parts.append(branch)

    if depth >= max_depth:
        return

    n_children = rng.randint(children_range[0], children_range[1] + 1)
    for ci in range(n_children):
        # Place child along parent (30%-90% of the way up)
        t = 0.3 + 0.6 * (ci / max(n_children - 1, 1))
        child_start = np.asarray(start) + d * length * t

        # Child direction: parent direction + random perturbation
        perturb_angle = rng.uniform(0, 2 * math.pi)
        perturb_elev = rng.uniform(0.3, spread)

        # Build a local frame from parent direction
        if abs(d[2]) < 0.9:
            ref = np.array([0.0, 0.0, 1.0])
        else:
            ref = np.array([1.0, 0.0, 0.0])
        u = np.cross(d, ref)
        u = u / max(np.linalg.norm(u), 1e-12)
        v = np.cross(d, u)
        v = v / max(np.linalg.norm(v), 1e-12)

        child_d = (d * math.cos(perturb_elev)
                   + u * math.sin(perturb_elev) * math.cos(perturb_angle)
                   + v * math.sin(perturb_elev) * math.sin(perturb_angle))

        # Bias upward slightly for healthy growth
        child_d[2] = max(child_d[2], 0.15)
        child_d = child_d / max(np.linalg.norm(child_d), 1e-12)

        child_length = length * length_decay * rng.uniform(0.8, 1.2)
        child_radius = radius * radius_decay * rng.uniform(0.8, 1.2)

        if child_radius < 0.01 or child_length < 0.05:
            continue

        _grow_branches(
            rng, parts, child_start, child_d,
            child_length, child_radius,
            depth + 1, max_depth,
            children_range=children_range,
            spread=spread,
            length_decay=length_decay,
            radius_decay=radius_decay,
            n_angular=n_angular,
        )


def make_winter_tree(trunk_height=12.0, trunk_radius=0.9,
                     n_primary=5, max_depth=3,
                     children_range=(2, 3), spread=0.7, seed=100):
    """Generate a healthy leafless winter tree with deep recursive branching.

    Returns (vertices, faces) numpy arrays.
    """
    rng = np.random.RandomState(seed)
    parts = []

    # Trunk — healthy taper
    trunk = _branch_mesh(
        start=[0, 0, 0],
        direction=[0, 0, 1],
        length=trunk_height,
        radius_start=trunk_radius,
        radius_end=trunk_radius * 0.5,
        n_angular=12,
        n_height=10,
    )
    parts.append(trunk)

    # Primary branches distributed around the trunk
    for i in range(n_primary):
        t = 0.35 + 0.55 * (i / max(n_primary - 1, 1))
        branch_start = np.array([0.0, 0.0, trunk_height * t])

        angle = 2 * math.pi * i / n_primary + rng.uniform(-0.3, 0.3)
        elev = rng.uniform(0.3, 0.7)
        dx = math.cos(angle) * math.cos(elev)
        dy = math.sin(angle) * math.cos(elev)
        dz = math.sin(elev)
        direction = np.array([dx, dy, dz])

        branch_length = trunk_height * rng.uniform(0.3, 0.5)
        branch_radius = trunk_radius * rng.uniform(0.25, 0.4)

        _grow_branches(
            rng, parts, branch_start, direction,
            branch_length, branch_radius,
            depth=0, max_depth=max_depth,
            children_range=children_range,
            spread=spread,
        )

    # Crown branches from the very top
    crown_start = np.array([0.0, 0.0, trunk_height * 0.95])
    for i in range(3):
        angle = 2 * math.pi * i / 3 + rng.uniform(-0.4, 0.4)
        dx = math.cos(angle) * 0.3
        dy = math.sin(angle) * 0.3
        direction = np.array([dx, dy, 0.9])
        direction = direction / np.linalg.norm(direction)

        _grow_branches(
            rng, parts, crown_start, direction,
            trunk_height * 0.25, trunk_radius * 0.3,
            depth=0, max_depth=max_depth - 1,
            children_range=children_range,
            spread=spread * 0.8,
        )

    return combine_meshes(parts)


def make_winter_oak(seed=101):
    """A broad winter oak with wide spreading branches."""
    return make_winter_tree(
        trunk_height=10.0, trunk_radius=1.2,
        n_primary=6, max_depth=3,
        children_range=(2, 4), spread=0.8,
        seed=seed,
    )


def make_winter_elm(seed=102):
    """A tall winter elm with vase-shaped branching."""
    return make_winter_tree(
        trunk_height=14.0, trunk_radius=0.7,
        n_primary=4, max_depth=4,
        children_range=(2, 3), spread=0.55,
        seed=seed,
    )


def make_winter_maple(seed=103):
    """A winter maple with dense, rounded branching structure."""
    return make_winter_tree(
        trunk_height=11.0, trunk_radius=0.85,
        n_primary=5, max_depth=3,
        children_range=(3, 4), spread=0.65,
        seed=seed,
    )


# ---------------------------------------------------------------------------
# Humanoid generation
# ---------------------------------------------------------------------------

def _limb_segment(start, end, radius, n_angular=8, n_height=4):
    """Tapered cylinder limb between two points."""
    direction = np.asarray(end) - np.asarray(start)
    length = np.linalg.norm(direction)
    if length < 1e-12:
        direction = np.array([0, 0, 1.0])
        length = 0.01
    return _branch_mesh(
        start=start, direction=direction / length,
        length=length,
        radius_start=radius, radius_end=radius * 0.85,
        n_angular=n_angular, n_height=n_height,
    )


def _sphere_mesh(center, radius, n_lat=8, n_lon=12):
    """Simple UV sphere."""
    verts = []
    faces = []

    for i in range(n_lat + 1):
        theta = math.pi * i / n_lat
        for j in range(n_lon):
            phi = 2 * math.pi * j / n_lon
            x = center[0] + radius * math.sin(theta) * math.cos(phi)
            y = center[1] + radius * math.sin(theta) * math.sin(phi)
            z = center[2] + radius * math.cos(theta)
            verts.append([x, y, z])

    for i in range(n_lat):
        for j in range(n_lon):
            j_next = (j + 1) % n_lon
            i0 = i * n_lon + j
            i1 = i * n_lon + j_next
            i2 = (i + 1) * n_lon + j
            i3 = (i + 1) * n_lon + j_next
            if i > 0:
                faces.append([i0, i2, i1])
            if i < n_lat - 1:
                faces.append([i1, i2, i3])

    return np.array(verts), np.array(faces, dtype=np.int64)


def make_humanoid(height=1.8, arm_span_factor=1.0, seed=42):
    """Generate a simple humanoid figure mesh.

    Returns (vertices, faces).
    """
    h = height
    parts = []

    # Torso
    torso_h = h * 0.3
    torso_r = h * 0.12
    torso_bottom = h * 0.45
    torso = _branch_mesh(
        start=[0, 0, torso_bottom],
        direction=[0, 0, 1],
        length=torso_h,
        radius_start=torso_r * 1.1,
        radius_end=torso_r * 0.9,
        n_angular=12, n_height=6,
    )
    parts.append(torso)

    # Head
    head_r = h * 0.07
    head_center = [0, 0, torso_bottom + torso_h + head_r * 1.5]
    head = _sphere_mesh(head_center, head_r, n_lat=8, n_lon=12)
    parts.append(head)

    # Neck
    neck = _branch_mesh(
        start=[0, 0, torso_bottom + torso_h],
        direction=[0, 0, 1],
        length=head_r * 1.0,
        radius_start=torso_r * 0.4,
        radius_end=torso_r * 0.35,
        n_angular=8, n_height=2,
    )
    parts.append(neck)

    # Arms
    shoulder_y = torso_bottom + torso_h * 0.9
    arm_length = h * 0.27 * arm_span_factor
    upper_arm_r = h * 0.035
    forearm_r = h * 0.028

    for side in [-1, 1]:
        shoulder = [side * torso_r * 1.1, 0, shoulder_y]
        elbow = [side * (torso_r * 1.1 + arm_length * 0.5), 0, shoulder_y - arm_length * 0.15]
        wrist = [side * (torso_r * 1.1 + arm_length), 0, shoulder_y - arm_length * 0.4]

        upper_arm = _limb_segment(shoulder, elbow, upper_arm_r)
        parts.append(upper_arm)
        forearm = _limb_segment(elbow, wrist, forearm_r)
        parts.append(forearm)

    # Legs
    hip_y = torso_bottom
    leg_length = h * 0.45
    thigh_r = h * 0.05
    shin_r = h * 0.035

    for side in [-1, 1]:
        hip = [side * torso_r * 0.5, 0, hip_y]
        knee = [side * torso_r * 0.5, 0, hip_y - leg_length * 0.5]
        ankle = [side * torso_r * 0.5, 0, 0.05]

        thigh = _limb_segment(hip, knee, thigh_r)
        parts.append(thigh)
        shin = _limb_segment(knee, ankle, shin_r)
        parts.append(shin)

        # Foot
        foot = _branch_mesh(
            start=ankle, direction=[0, 1, 0],
            length=h * 0.08,
            radius_start=shin_r * 0.9, radius_end=shin_r * 0.7,
            n_angular=6, n_height=2,
        )
        parts.append(foot)

    return combine_meshes(parts)


def make_humanoid_stocky(seed=42):
    """Stocky humanoid with wider proportions."""
    return make_humanoid(height=1.6, arm_span_factor=1.2, seed=seed)


def make_humanoid_tall(seed=42):
    """Tall slender humanoid."""
    return make_humanoid(height=2.0, arm_span_factor=0.9, seed=seed)


# ---------------------------------------------------------------------------
# Additional test objects
# ---------------------------------------------------------------------------

def make_simple_chair(seat_size=0.4, seat_height=0.45, back_height=0.4):
    """Generate a simple chair mesh."""
    parts = []
    leg_r = 0.02
    half = seat_size / 2

    # Seat (flat box approximated as short cylinder)
    seat = _branch_mesh(
        start=[0, 0, seat_height - 0.02],
        direction=[0, 0, 1],
        length=0.04,
        radius_start=half * 1.1, radius_end=half * 1.1,
        n_angular=4, n_height=1,
    )
    parts.append(seat)

    # 4 legs
    for sx, sy in [(-1, -1), (-1, 1), (1, -1), (1, 1)]:
        leg = _branch_mesh(
            start=[sx * half * 0.9, sy * half * 0.9, 0],
            direction=[0, 0, 1],
            length=seat_height,
            radius_start=leg_r, radius_end=leg_r,
            n_angular=6, n_height=3,
        )
        parts.append(leg)

    # Backrest
    back = _branch_mesh(
        start=[0, -half * 0.9, seat_height],
        direction=[0, 0, 1],
        length=back_height,
        radius_start=half * 0.9, radius_end=half * 0.8,
        n_angular=4, n_height=3,
    )
    parts.append(back)

    return combine_meshes(parts)


def make_simple_bottle(height=0.3, body_radius=0.04, neck_radius=0.015):
    """Generate a simple bottle (revolve of solid)."""
    profile = np.array([
        [0.0, 0.0],
        [body_radius, 0.0],
        [body_radius, height * 0.6],
        [body_radius * 0.8, height * 0.65],
        [neck_radius, height * 0.7],
        [neck_radius, height * 0.95],
        [neck_radius * 1.2, height],
        [0.0, height],
    ])
    return revolve_profile(profile, n_angular=24, close_top=True, close_bottom=True)


def make_simple_table(top_size=0.8, top_height=0.75, leg_radius=0.03):
    """Generate a simple table mesh."""
    parts = []
    half = top_size / 2

    # Top
    top = _branch_mesh(
        start=[0, 0, top_height - 0.03],
        direction=[0, 0, 1],
        length=0.06,
        radius_start=half * 1.1, radius_end=half * 1.1,
        n_angular=4, n_height=1,
    )
    parts.append(top)

    # 4 legs
    for sx, sy in [(-1, -1), (-1, 1), (1, -1), (1, 1)]:
        leg = _branch_mesh(
            start=[sx * half * 0.85, sy * half * 0.85, 0],
            direction=[0, 0, 1],
            length=top_height,
            radius_start=leg_radius, radius_end=leg_radius,
            n_angular=6, n_height=4,
        )
        parts.append(leg)

    return combine_meshes(parts)


def make_simple_rocket(height=1.0, body_radius=0.1):
    """Generate a simple rocket mesh (cylinder + cone + fins)."""
    parts = []

    # Body cylinder
    body = _branch_mesh(
        start=[0, 0, 0],
        direction=[0, 0, 1],
        length=height * 0.7,
        radius_start=body_radius, radius_end=body_radius,
        n_angular=16, n_height=6,
    )
    parts.append(body)

    # Nose cone
    cone = _branch_mesh(
        start=[0, 0, height * 0.7],
        direction=[0, 0, 1],
        length=height * 0.3,
        radius_start=body_radius, radius_end=0.001,
        n_angular=16, n_height=4,
    )
    parts.append(cone)

    # 4 fins
    for i in range(4):
        angle = 2 * math.pi * i / 4
        dx = math.cos(angle)
        dy = math.sin(angle)
        fin_base = [dx * body_radius, dy * body_radius, 0]
        fin_tip = [dx * body_radius * 2.5, dy * body_radius * 2.5, height * 0.15]
        fin = _limb_segment(fin_base, fin_tip, body_radius * 0.3, n_angular=4)
        parts.append(fin)

    return combine_meshes(parts)


# ---------------------------------------------------------------------------
# Master list of all test models
# ---------------------------------------------------------------------------

ALL_TEST_MODELS = {
    "dead_tree": make_dead_tree,
    "dead_tree_gnarled": make_dead_tree_gnarled,
    "dead_tree_tall": make_dead_tree_tall,
    "dead_tree_stumpy": make_dead_tree_stumpy,
    "winter_tree": make_winter_tree,
    "winter_oak": make_winter_oak,
    "winter_elm": make_winter_elm,
    "winter_maple": make_winter_maple,
    "humanoid": make_humanoid,
    "humanoid_stocky": make_humanoid_stocky,
    "humanoid_tall": make_humanoid_tall,
    "chair": make_simple_chair,
    "bottle": make_simple_bottle,
    "table": make_simple_table,
    "rocket": make_simple_rocket,
}


def get_test_model(name):
    """Get a test model by name. Returns (vertices, faces)."""
    fn = ALL_TEST_MODELS.get(name)
    if fn is None:
        raise ValueError(f"Unknown test model: {name}. Available: {list(ALL_TEST_MODELS.keys())}")
    return fn()
