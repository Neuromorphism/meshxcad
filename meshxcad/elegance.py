"""CAD elegance scoring and adversarial loops for CAD quality.

This module implements two adversarial loops:

**Loop 1: CAD vs Mesh Discriminator**
    A critic tries to tell apart a CAD-generated mesh from a raw/scanned mesh.
    The attacker evolves the CAD program to produce meshes the critic cannot
    distinguish from real mesh data.

**Loop 2: Elegance Tournament**
    A critic judges which of two CadPrograms is more elegant (better design
    practice).  The attacker tries to produce a program that fools the critic
    into picking it as the more elegant one, despite having been auto-generated.

Both loops draw on research-backed CAD quality criteria:

    Academic:  Camba et al. (2023) graph-based complexity, Company et al. (2015)
               six quality dimensions, Johnson et al. (2018) geometric complexity
    Industry:  ASME Y14.5 design intent, GrabCAD community quality standards
    Practice:  GrabCAD top-1000 patterns — symmetry exploitation, operation
               economy, functional accuracy, parametric intelligence

Elegance criteria (from research + GrabCAD analysis):
    - Conciseness: fewer operations to achieve the same geometry
    - Symmetry exploitation: use mirror/pattern instead of duplicating ops
    - Operation hierarchy: primitives > extrude/revolve > sweep > booleans
    - Design intent: ops reflect functional purpose, not arbitrary decomposition
    - Feature tree depth: shallow dependency chains preferred
    - Reference stability: operations relative to origin/axes, not arbitrary edges
    - No redundancy: no overlapping operations, no ops that cancel each other
    - Mesh quality: good triangle quality in the output (aspect ratio, normals)
    - Completeness: all target geometry is covered
    - Parametric readability: a human could read the program and understand it
"""

import math
import copy
import numpy as np
from dataclasses import dataclass, field
from typing import Optional

from scipy.spatial import KDTree

from .gpu import AcceleratedKDTree as _AKDTree

from .general_align import (
    hausdorff_distance, surface_distance_map, _compute_vertex_normals,
    _compute_face_normals, _face_areas, _vertex_adjacency,
    surface_area, volume_diff, curvature_histogram_diff,
)
from .cad_program import (
    CadOp, CadProgram, OP_COSTS, ALPHA, BETA, GAMMA,
    _eval_op, find_program_gaps, add_operation, refine_operation,
    remove_operation, simplify_program, initial_program,
)
from .objects.builder import combine_meshes


# ============================================================================
# Elegance scoring criteria (research-backed)
# ============================================================================

# Operation hierarchy: lower tier = more elegant use of that operation.
# Based on GrabCAD analysis: top models use primitives + extrude/revolve
# efficiently, minimize boolean complexity.
OP_TIER = {
    # Tier 1 — fundamental primitives (most elegant when appropriate)
    "sphere": 1, "cylinder": 1, "profiled_cylinder": 1, "box": 1, "cone": 1,
    # Tier 2 — profile-based generation (clean parametric intent)
    "extrude": 2, "revolve": 2, "torus": 2,
    # Tier 3 — path-based (higher complexity, but sometimes necessary)
    "sweep": 3,
    # Tier 4 — modifiers (elegant when used sparingly)
    "translate": 4, "scale": 4, "rotate": 4, "mirror": 4,
    # Tier 5 — boolean/destructive (least elegant, avoid if possible)
    "subtract_cylinder": 5, "union": 5,
}

# Symmetry operations — using these instead of duplicating is elegant
SYMMETRY_OPS = {"mirror", "rotate"}

# Operations that indicate good parametric thinking
PARAMETRIC_OPS = {"revolve", "extrude", "sweep", "mirror"}

# Maximum recommended ops for different shape complexities
# (from GrabCAD analysis: top single-part models average 5-15 features)
MAX_ELEGANT_OPS = {
    "simple": 3,     # sphere, cylinder, box shapes
    "moderate": 7,   # revolved/extruded shapes with a few features
    "complex": 12,   # multi-feature mechanical parts
}


# ============================================================================
# Elegance scoring functions
# ============================================================================

def score_conciseness(program):
    """Score how concise the program is (fewer ops = better).

    Based on Camba et al. "conciseness" dimension:
    No redundant/fragmented operations.

    Returns:
        float: 0.0 (terrible) to 1.0 (perfect conciseness)
    """
    n = program.n_enabled()
    if n == 0:
        return 0.0
    # 1 op is perfect, decays with more ops
    # Calibrated: 1 op → 1.0, 5 ops → 0.7, 10 → 0.5, 20 → 0.33
    return 1.0 / (1.0 + 0.1 * (n - 1) ** 1.3)


def score_op_hierarchy(program):
    """Score preference for simpler operation types.

    GrabCAD pattern: top models use primitives + extrude/revolve efficiently.
    Research: "direct_features / (direct_features + boolean_combos)" higher is better.

    Returns:
        float: 0.0 to 1.0 (higher = better hierarchy usage)
    """
    enabled = [op for op in program.operations if op.enabled]
    if not enabled:
        return 0.0

    tier_scores = []
    for op in enabled:
        tier = OP_TIER.get(op.op_type, 5)
        # Tier 1 → 1.0, Tier 2 → 0.8, Tier 3 → 0.6, Tier 4 → 0.4, Tier 5 → 0.2
        tier_scores.append(1.0 - (tier - 1) * 0.2)

    return float(np.mean(tier_scores))


def score_symmetry_exploitation(program, target_v):
    """Score whether the program uses symmetry ops where geometry is symmetric.

    GrabCAD insight: top models exploit symmetry via patterns/mirror rather
    than duplicating geometry manually.

    Returns:
        float: 0.0 to 1.0
    """
    target_v = np.asarray(target_v)
    if len(target_v) == 0:
        return 0.5

    enabled = [op for op in program.operations if op.enabled]
    if not enabled:
        return 0.0

    # Check if target has symmetry planes
    centroid = target_v.mean(axis=0)
    centered = target_v - centroid

    symmetry_axes = []
    for axis_idx in range(3):
        reflected = centered.copy()
        reflected[:, axis_idx] *= -1
        # Compare point clouds (approximate symmetry check)
        tree = _AKDTree(centered)
        dists, _ = tree.query(reflected)
        bbox_diag = np.linalg.norm(centered.max(0) - centered.min(0))
        rel_error = dists.mean() / max(bbox_diag, 1e-8)
        if rel_error < 0.05:  # 5% threshold → symmetric
            symmetry_axes.append(axis_idx)

    if not symmetry_axes:
        return 0.8  # no symmetry to exploit → neutral/good

    # Check if program uses symmetry ops
    has_symmetry_op = any(op.op_type in SYMMETRY_OPS for op in enabled)

    # Count generator ops (non-modifier, non-symmetry)
    generators = [op for op in enabled
                  if op.op_type not in SYMMETRY_OPS
                  and op.op_type not in ("translate", "scale", "union")]
    n_generators = len(generators)

    if has_symmetry_op:
        return 1.0  # using symmetry — great
    elif n_generators > 1 and len(symmetry_axes) > 0:
        # Multiple generators where mirror would suffice
        return 0.3
    else:
        return 0.6


def score_no_redundancy(program, target_v, target_f):
    """Score whether every operation contributes to the final result.

    Camba et al. "conciseness": unique_operations / total_operations.
    An op is redundant if removing it barely changes the output.

    Returns:
        float: 0.0 to 1.0 (1.0 = no redundancy)
    """
    enabled = [i for i, op in enumerate(program.operations) if op.enabled]
    if len(enabled) <= 1:
        return 1.0

    base_cost = program.total_cost(target_v, target_f)
    if not np.isfinite(base_cost):
        return 0.0

    redundant = 0
    for idx in enabled:
        trial = program.copy()
        trial.operations[idx].enabled = False
        trial.invalidate_cache()
        cost_without = trial.total_cost(target_v, target_f)
        # If removing it barely hurts accuracy (< 1% of base cost), it's redundant
        if cost_without < base_cost * 1.01:
            redundant += 1

    return 1.0 - redundant / len(enabled)


def score_feature_tree_depth(program):
    """Score the dependency chain depth of the program.

    Research: "Horizontal modeling" minimizes long chains. Shallower = better.
    GrabCAD: top models have well-organized flat feature trees.

    For our CadProgram, depth = longest chain of modifiers applied to
    a single generator.

    Returns:
        float: 0.0 to 1.0
    """
    enabled = [op for op in program.operations if op.enabled]
    if not enabled:
        return 0.0

    # Count consecutive modifier chains
    max_chain = 0
    current_chain = 0
    modifiers = {"translate", "scale", "rotate", "mirror",
                 "subtract_cylinder", "union"}

    for op in enabled:
        if op.op_type in modifiers:
            current_chain += 1
        else:
            max_chain = max(max_chain, current_chain)
            current_chain = 0
    max_chain = max(max_chain, current_chain)

    # 0 modifiers in a row → 1.0, 1 → 0.9, 2 → 0.7, 3+ → decay
    return max(0.0, 1.0 - max_chain * 0.15)


def score_parameter_economy(program):
    """Score whether parameters are used economically.

    Research: fewer driver parameters = clearer design intent.
    GrabCAD: 2-5 key parameters control overall shape.

    Returns:
        float: 0.0 to 1.0
    """
    total_params = program.total_param_count()
    n_ops = program.n_enabled()
    if n_ops == 0:
        return 0.0

    # Average params per op — ideal is 2-4 (center + 1-2 dimensions)
    avg_params = total_params / n_ops
    if avg_params <= 4:
        return 1.0
    elif avg_params <= 8:
        return 0.8
    elif avg_params <= 15:
        return 0.5
    else:
        return 0.3


def score_origin_anchoring(program):
    """Score whether operations reference the origin (0,0,0).

    Research: "Missing anchor to origin" is a CAD anti-pattern (-10 penalty).
    GrabCAD: centralized model origin is a "must-have" quality marker.

    Returns:
        float: 0.0 to 1.0
    """
    enabled = [op for op in program.operations if op.enabled]
    if not enabled:
        return 0.0

    # Check if at least one generator is near origin
    generators = [op for op in enabled
                  if op.op_type not in ("translate", "scale", "rotate",
                                         "mirror", "subtract_cylinder", "union")]
    if not generators:
        return 0.5

    has_origin_ref = False
    for op in generators:
        center = op.params.get("center", [0, 0, 0])
        if isinstance(center, (list, np.ndarray)):
            dist = np.linalg.norm(np.asarray(center))
        else:
            dist = abs(center)
        if dist < 0.01:
            has_origin_ref = True
            break

    return 1.0 if has_origin_ref else 0.4


def score_mesh_quality(program):
    """Score the triangle quality of the program's output mesh.

    Research (Sorgente 2023): aspect ratio, skewness, element quality.
    ANSYS: good mesh has aspect ratio < 5 for 95% of elements.

    Returns:
        float: 0.0 to 1.0
    """
    cad_v, cad_f = program.evaluate()
    if len(cad_f) == 0:
        return 0.0

    v = np.asarray(cad_v, dtype=np.float64)
    f = np.asarray(cad_f)

    # Compute per-triangle aspect ratio
    v0, v1, v2 = v[f[:, 0]], v[f[:, 1]], v[f[:, 2]]
    e0 = np.linalg.norm(v1 - v0, axis=1)
    e1 = np.linalg.norm(v2 - v1, axis=1)
    e2 = np.linalg.norm(v0 - v2, axis=1)
    max_edge = np.maximum(np.maximum(e0, e1), e2)

    # Semi-perimeter and area via Heron's formula
    s = (e0 + e1 + e2) / 2.0
    area_sq = s * (s - e0) * (s - e1) * (s - e2)
    area_sq = np.maximum(area_sq, 0)
    areas = np.sqrt(area_sq)

    # Aspect ratio = longest_edge * s / (2 * area)
    # For equilateral triangle, this equals sqrt(3) ≈ 1.73
    safe_areas = np.where(areas > 1e-12, areas, 1.0)
    aspect = np.where(areas > 1e-12, max_edge * s / (2 * safe_areas), 100.0)

    # Score: fraction of triangles with aspect < 5
    good_frac = float(np.mean(aspect < 5.0))

    # Also check for degenerate triangles
    degen_frac = float(np.mean(areas < 1e-10))

    return good_frac * (1.0 - degen_frac)


def score_normal_consistency(program):
    """Score whether output mesh has consistent face normals.

    Research: "Normal consistency: All face normals point same direction" → pass.
    We check consistency (all inward OR all outward) rather than requiring outward,
    and skip degenerate (zero-area) faces.

    Returns:
        float: 0.0 to 1.0
    """
    cad_v, cad_f = program.evaluate()
    if len(cad_f) == 0:
        return 0.0

    v = np.asarray(cad_v, dtype=np.float64)
    f = np.asarray(cad_f)

    face_normals = _compute_face_normals(v, f)
    fn_mag = np.linalg.norm(face_normals, axis=1)
    valid = fn_mag > 1e-8  # skip degenerate faces

    if valid.sum() < 2:
        return 0.5

    face_centers = (v[f[:, 0]] + v[f[:, 1]] + v[f[:, 2]]) / 3.0
    centroid = v.mean(axis=0)
    outward_dir = face_centers[valid] - centroid
    outward_norm = np.linalg.norm(outward_dir, axis=1, keepdims=True)
    outward_norm[outward_norm < 1e-12] = 1e-12
    outward_dir = outward_dir / outward_norm

    dots = np.sum(face_normals[valid] * outward_dir, axis=1)

    # Consistent means they all agree on direction (all inward or all outward)
    outward_frac = float(np.mean(dots > 0))
    consistency = max(outward_frac, 1.0 - outward_frac)  # best of either direction

    # Also factor in fraction of non-degenerate faces
    valid_frac = float(valid.mean())

    return consistency * valid_frac


def score_op_diversity(program):
    """Score whether the program uses varied operation types appropriately.

    Anti-pattern: using 10 spheres when 1 sphere + 1 mirror would work.
    Good: each op type appears at most a few times.

    Returns:
        float: 0.0 to 1.0
    """
    enabled = [op for op in program.operations if op.enabled]
    if len(enabled) <= 1:
        return 1.0

    from collections import Counter
    type_counts = Counter(op.op_type for op in enabled)
    n_types = len(type_counts)
    n_ops = len(enabled)

    # Ratio of unique types to total ops
    diversity = n_types / n_ops

    # Penalize excessive repetition of any type
    max_repeat = max(type_counts.values())
    repeat_penalty = 1.0 if max_repeat <= 2 else 0.8 if max_repeat <= 4 else 0.5

    return min(1.0, diversity * 1.5) * repeat_penalty


def score_watertightness(program):
    """Score whether the output mesh is likely watertight.

    Research: "is_edge_manifold AND is_vertex_manifold AND NOT is_self_intersecting"
    Euler-Poincare: V - E + F = 2(1 - g) for genus g manifold.

    Returns:
        float: 0.0 to 1.0
    """
    cad_v, cad_f = program.evaluate()
    if len(cad_f) == 0:
        return 0.0

    f = np.asarray(cad_f)
    n_v = len(cad_v)

    # Build edge set and check manifoldness
    edge_face_count = {}
    for tri_idx in range(len(f)):
        for i in range(3):
            e = tuple(sorted([f[tri_idx, i], f[tri_idx, (i + 1) % 3]]))
            edge_face_count[e] = edge_face_count.get(e, 0) + 1

    n_e = len(edge_face_count)
    n_f = len(f)

    # Boundary edges (shared by only 1 face)
    boundary = sum(1 for c in edge_face_count.values() if c == 1)
    # Non-manifold edges (shared by 3+ faces)
    nonmanifold = sum(1 for c in edge_face_count.values() if c > 2)

    boundary_score = 1.0 if boundary == 0 else max(0.0, 1.0 - boundary / n_e)
    manifold_score = 1.0 if nonmanifold == 0 else max(0.0, 1.0 - nonmanifold / n_e)

    # Euler check: V - E + F should be 2 for a closed manifold (genus 0)
    euler = n_v - n_e + n_f
    euler_score = 1.0 if euler == 2 else 0.7 if abs(euler - 2) <= 2 else 0.3

    return boundary_score * 0.4 + manifold_score * 0.3 + euler_score * 0.3


# ============================================================================
# Feature fidelity — medium-small feature preservation
# ============================================================================

def _decompose_features(vertices, faces):
    """Decompose a mesh into connected components and classify by size.

    Returns list of dicts sorted by vertex count descending:
        [{label, verts_idx, center, radius, frac, size_class}, ...]

    size_class: 'large' (>15%), 'medium' (2-15%), 'small' (0.5-2%), 'tiny' (<0.5%)
    """
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import connected_components

    n = len(vertices)
    if n == 0:
        return []

    rows = np.concatenate([faces[:, 0], faces[:, 1], faces[:, 2]])
    cols = np.concatenate([faces[:, 1], faces[:, 2], faces[:, 0]])
    data = np.ones(len(rows))
    adj = csr_matrix((data, (rows, cols)), shape=(n, n))
    n_comp, labels = connected_components(adj, directed=False)

    # Adaptive large-feature threshold: for meshes with few components,
    # only the single largest (if it dominates) should be "large".
    # This ensures chair legs, rocket fins, etc. are treated as medium.
    if n_comp <= 3:
        large_thresh = 0.50
    elif n_comp <= 8:
        large_thresh = 0.30
    else:
        large_thresh = 0.15

    features = []
    for c in range(n_comp):
        mask = labels == c
        idx = np.where(mask)[0]
        pts = vertices[idx]
        center = pts.mean(axis=0)
        radius = float(np.max(np.linalg.norm(pts - center, axis=1)))
        frac = len(idx) / n

        if frac > large_thresh:
            size_class = "large"
        elif frac > 0.02:
            size_class = "medium"
        elif frac > 0.005:
            size_class = "small"
        else:
            size_class = "tiny"

        features.append({
            "label": c,
            "verts_idx": idx,
            "center": center,
            "radius": radius,
            "frac": frac,
            "size_class": size_class,
        })

    features.sort(key=lambda f: len(f["verts_idx"]), reverse=True)
    return features


def score_feature_fidelity(program, target_v, target_f):
    """Score how well the CAD output preserves medium-small features.

    Focuses on features that are 0.5%-15% of the total mesh — the scale
    of fingers, individual branches, chair legs, rocket fins, etc.

    Measures two things per feature:
      1. Coverage: fraction of feature vertices within a tight distance
         threshold of the CAD output
      2. Spatial presence: whether the CAD output has geometry near the
         feature's centroid at all

    Returns:
        float: 0.0 (no features preserved) to 1.0 (all features well-covered)
    """
    cad_v, cad_f = program.evaluate()
    if len(cad_v) == 0:
        return 0.0

    target_v = np.asarray(target_v, dtype=np.float64)
    target_f = np.asarray(target_f)

    features = _decompose_features(target_v, target_f)

    # Filter to medium and small features only
    ms_features = [f for f in features if f["size_class"] in ("medium", "small")]
    if not ms_features:
        # No medium-small features to evaluate — return neutral score
        return 0.8

    bbox_diag = float(np.linalg.norm(target_v.max(0) - target_v.min(0)))
    if bbox_diag < 1e-12:
        return 0.0

    # Tight coverage threshold: 5% of bounding box diagonal
    coverage_thresh = bbox_diag * 0.05

    cad_tree = _AKDTree(cad_v)

    weighted_score = 0.0
    total_weight = 0.0

    for feat in ms_features:
        feat_pts = target_v[feat["verts_idx"]]

        # Weight: medium features count 3x, small features count 1x
        w = 3.0 if feat["size_class"] == "medium" else 1.0

        # 1. Coverage: what fraction of feature vertices are close to CAD?
        dists, _ = cad_tree.query(feat_pts)
        coverage = float(np.mean(dists < coverage_thresh))

        # 2. Spatial presence: is there CAD geometry near the feature center?
        center_dist, _ = cad_tree.query(feat["center"].reshape(1, 3))
        # Presence decays: if CAD is within the feature's own radius, full credit
        presence_thresh = max(feat["radius"] * 1.5, coverage_thresh)
        presence = float(math.exp(-center_dist[0] / presence_thresh))

        # Combined per-feature score
        feat_score = coverage * 0.7 + presence * 0.3

        weighted_score += w * feat_score
        total_weight += w

    if total_weight < 1e-12:
        return 0.8

    return float(weighted_score / total_weight)


def score_accuracy(program, target_v, target_f):
    """Score how well the program's mesh matches the target.

    Returns:
        float: 0.0 to 1.0 (1.0 = perfect match)
    """
    cad_v, cad_f = program.evaluate()
    if len(cad_v) == 0:
        return 0.0

    hd = hausdorff_distance(cad_v, target_v)
    mean_sym = hd["mean_symmetric"]

    bbox_diag = float(np.linalg.norm(
        np.asarray(target_v).max(0) - np.asarray(target_v).min(0)))
    if bbox_diag > 1e-12:
        rel_dist = mean_sym / bbox_diag
    else:
        rel_dist = mean_sym

    # Exponential decay: 0 distance → 1.0, larger → decays smoothly
    # (avoids hard clip to 0.0 that trapped the optimizer)
    import math as _math
    return _math.exp(-rel_dist * 3.0)


# ============================================================================
# Composite elegance score
# ============================================================================

# Weights from academic rubrics (Company et al. 2015, adapted)
ELEGANCE_WEIGHTS = {
    "accuracy":            0.18,  # completeness: does it match the target?
    "feature_fidelity":    0.10,  # medium-small feature preservation
    "conciseness":         0.11,  # fewer ops = more elegant
    "op_hierarchy":        0.07,  # prefer primitives over booleans
    "symmetry":            0.07,  # exploit symmetry
    "no_redundancy":       0.09,  # every op contributes
    "tree_depth":          0.06,  # shallow dependency chains
    "param_economy":       0.05,  # economical parameter usage
    "origin_anchoring":    0.04,  # anchored to origin
    "mesh_quality":        0.07,  # good triangle quality
    "normal_consistency":  0.04,  # consistent normals
    "op_diversity":        0.05,  # varied but not repetitive ops
    "watertightness":      0.07,  # closed manifold output
}


def compute_elegance_score(program, target_v, target_f):
    """Compute the full elegance score for a CadProgram.

    Returns:
        dict with per-dimension scores (0-1) and weighted total (0-1).
        Higher = more elegant.
    """
    target_v = np.asarray(target_v, dtype=np.float64)
    target_f = np.asarray(target_f)

    scores = {
        "accuracy":           score_accuracy(program, target_v, target_f),
        "feature_fidelity":   score_feature_fidelity(program, target_v, target_f),
        "conciseness":        score_conciseness(program),
        "op_hierarchy":       score_op_hierarchy(program),
        "symmetry":           score_symmetry_exploitation(program, target_v),
        "no_redundancy":      score_no_redundancy(program, target_v, target_f),
        "tree_depth":         score_feature_tree_depth(program),
        "param_economy":      score_parameter_economy(program),
        "origin_anchoring":   score_origin_anchoring(program),
        "mesh_quality":       score_mesh_quality(program),
        "normal_consistency": score_normal_consistency(program),
        "op_diversity":       score_op_diversity(program),
        "watertightness":     score_watertightness(program),
    }

    total = sum(scores[k] * ELEGANCE_WEIGHTS[k] for k in ELEGANCE_WEIGHTS)

    return {
        "scores": scores,
        "total": float(total),
        "n_ops": program.n_enabled(),
        "complexity": program.total_complexity(),
        "program_summary": program.summary(),
    }


# ============================================================================
# Loop 1: CAD vs Mesh Discriminator
# ============================================================================

# Discriminator features — metrics that differ between CAD-generated meshes
# and raw/scanned meshes.  Based on research: CAD meshes have regularities
# that scanned meshes lack.

def _edge_length_regularity(v, f):
    """Measure how regular edge lengths are.  CAD meshes have very uniform
    edge lengths; scanned meshes are irregular."""
    f = np.asarray(f)
    v = np.asarray(v, dtype=np.float64)
    if len(f) == 0:
        return 0.0
    edges = np.concatenate([
        np.linalg.norm(v[f[:, 1]] - v[f[:, 0]], axis=1),
        np.linalg.norm(v[f[:, 2]] - v[f[:, 1]], axis=1),
        np.linalg.norm(v[f[:, 0]] - v[f[:, 2]], axis=1),
    ])
    if edges.mean() < 1e-12:
        return 0.0
    return float(edges.std() / edges.mean())  # CV: low for CAD, high for mesh


def _face_area_regularity(v, f):
    """Coefficient of variation of triangle areas.  CAD = uniform, mesh = varied."""
    areas = _face_areas(v, f)
    if len(areas) == 0 or areas.mean() < 1e-12:
        return 0.0
    return float(areas.std() / areas.mean())


def _normal_smoothness(v, f):
    """Average angle between adjacent face normals.  CAD tends to have sharp
    transitions at feature edges but smooth patches; scanned meshes have
    noisy normals everywhere."""
    normals = _compute_face_normals(v, f)
    f_arr = np.asarray(f)
    if len(f_arr) < 2:
        return 0.0

    # Build face adjacency via shared edges
    edge_to_faces = {}
    for fi in range(len(f_arr)):
        for i in range(3):
            e = tuple(sorted([f_arr[fi, i], f_arr[fi, (i + 1) % 3]]))
            edge_to_faces.setdefault(e, []).append(fi)

    angles = []
    for faces_list in edge_to_faces.values():
        if len(faces_list) == 2:
            dot = np.clip(np.dot(normals[faces_list[0]], normals[faces_list[1]]),
                          -1, 1)
            angles.append(math.acos(dot))

    if not angles:
        return 0.0
    return float(np.std(angles))  # CAD: bimodal (0° smooth + 90° features); mesh: noisy


def _vertex_valence_regularity(v, f):
    """Standard deviation of vertex valence.  CAD meshes have regular valence
    (mostly 6 for internal vertices); scanned meshes vary widely."""
    f_arr = np.asarray(f)
    n_v = len(v)
    valence = np.zeros(n_v, dtype=int)
    for i in range(3):
        np.add.at(valence, f_arr[:, i], 1)
    active = valence[valence > 0]
    if len(active) == 0:
        return 0.0
    return float(active.std() / max(active.mean(), 1e-8))


def _curvature_bimodality(v, f):
    """Measure whether curvature distribution is bimodal (flat + sharp edges)
    vs continuous (scanned surface).  CAD meshes are bimodal."""
    from .general_align import _vertex_curvature
    curv = _vertex_curvature(v, f)
    if len(curv) < 10:
        return 0.0
    # Kurtosis: bimodal distributions tend to have negative excess kurtosis
    mu = curv.mean()
    std = curv.std()
    if std < 1e-12:
        return 0.0
    kurt = float(np.mean(((curv - mu) / std) ** 4) - 3.0)
    return kurt  # negative = bimodal (CAD-like), positive = peaked (scan-like)


def _symmetry_score(v):
    """Measure degree of reflective symmetry.  CAD models tend to be highly
    symmetric; scanned objects less so."""
    v = np.asarray(v, dtype=np.float64)
    if len(v) < 4:
        return 0.0

    centroid = v.mean(axis=0)
    centered = v - centroid
    tree = KDTree(centered)

    best_sym = 0.0
    bbox_diag = np.linalg.norm(centered.max(0) - centered.min(0))
    if bbox_diag < 1e-8:
        return 1.0

    for axis_idx in range(3):
        reflected = centered.copy()
        reflected[:, axis_idx] *= -1
        dists, _ = tree.query(reflected)
        rel = 1.0 - dists.mean() / bbox_diag
        best_sym = max(best_sym, rel)

    return float(best_sym)


def _planarity_fraction(v, f):
    """Fraction of triangles that lie on planar patches.  CAD meshes have
    large planar regions; scanned meshes do not."""
    normals = _compute_face_normals(v, f)
    f_arr = np.asarray(f)
    if len(f_arr) < 4:
        return 0.0

    # Build face adjacency
    edge_to_faces = {}
    for fi in range(len(f_arr)):
        for i in range(3):
            e = tuple(sorted([f_arr[fi, i], f_arr[fi, (i + 1) % 3]]))
            edge_to_faces.setdefault(e, []).append(fi)

    # A face is "planar" if all its neighbors have nearly the same normal
    planar_count = 0
    threshold = math.cos(math.radians(5))  # < 5° deviation
    for fi in range(len(f_arr)):
        neighbor_normals = []
        for i in range(3):
            e = tuple(sorted([f_arr[fi, i], f_arr[fi, (i + 1) % 3]]))
            for nf in edge_to_faces.get(e, []):
                if nf != fi:
                    neighbor_normals.append(normals[nf])
        if not neighbor_normals:
            continue
        dots = [np.dot(normals[fi], nn) for nn in neighbor_normals]
        if all(d > threshold for d in dots):
            planar_count += 1

    return float(planar_count / len(f_arr))


DISCRIMINATOR_FEATURES = [
    ("edge_regularity",      _edge_length_regularity),
    ("area_regularity",      _face_area_regularity),
    ("normal_smoothness",    _normal_smoothness),
    ("valence_regularity",   _vertex_valence_regularity),
    ("curvature_bimodality", _curvature_bimodality),
    ("symmetry",             _symmetry_score),
    ("planarity",            _planarity_fraction),
]


def compute_discriminator_features(v, f):
    """Extract all discriminator features from a mesh.

    Returns:
        dict: feature_name → float value
    """
    v = np.asarray(v, dtype=np.float64)
    f = np.asarray(f)
    features = {}
    for name, fn in DISCRIMINATOR_FEATURES:
        try:
            if name == "symmetry":
                features[name] = fn(v)
            else:
                features[name] = fn(v, f)
        except Exception:
            features[name] = 0.0
    return features


def discriminate_cad_vs_mesh(v, f):
    """Score how "CAD-like" vs "mesh-like" a mesh appears.

    Returns:
        float: 0.0 = definitely raw mesh, 1.0 = definitely CAD
    """
    feats = compute_discriminator_features(v, f)

    # CAD indicators (based on research):
    # - low edge/area irregularity (regular tessellation)
    # - bimodal curvature (flat patches + sharp features)
    # - high symmetry
    # - high planarity fraction
    # - low valence irregularity

    cad_signals = 0.0
    total_weight = 0.0

    # Edge regularity: CAD < 0.3, mesh > 0.5
    w = 1.5
    reg = feats.get("edge_regularity", 0.5)
    cad_signals += w * max(0, 1.0 - reg * 2.0)
    total_weight += w

    # Area regularity: CAD < 0.5, mesh > 1.0
    w = 1.0
    areg = feats.get("area_regularity", 0.5)
    cad_signals += w * max(0, 1.0 - areg)
    total_weight += w

    # Valence regularity: CAD < 0.3, mesh > 0.5
    w = 1.0
    vreg = feats.get("valence_regularity", 0.5)
    cad_signals += w * max(0, 1.0 - vreg * 2.0)
    total_weight += w

    # Symmetry: CAD > 0.8, mesh < 0.6
    w = 1.5
    sym = feats.get("symmetry", 0.5)
    cad_signals += w * sym
    total_weight += w

    # Planarity: CAD > 0.5, mesh < 0.2
    w = 1.0
    plan = feats.get("planarity", 0.3)
    cad_signals += w * plan
    total_weight += w

    # Normal smoothness std: CAD has bimodal (high std), mesh has uniform noise (low std)
    w = 0.5
    ns = feats.get("normal_smoothness", 0.3)
    cad_signals += w * min(1.0, ns * 2.0)
    total_weight += w

    return float(np.clip(cad_signals / total_weight, 0.0, 1.0))


def run_cad_vs_mesh_loop(target_v, target_f, max_rounds=20, max_ops=10):
    """Loop 1: Evolve a CadProgram so the critic cannot distinguish its mesh
    from a raw/scanned mesh.

    The CRITIC (RED team) uses discriminator features to score how CAD-like
    the output mesh is.

    The ATTACKER (BLUE team) mutates the CadProgram to:
      1. Maintain accuracy (match target)
      2. Reduce CAD-likeness (fool the critic)

    The attacker's strategies:
      - Add controlled noise to parameters (break regularity)
      - Use non-uniform subdivisions
      - Jitter vertex positions post-generation
      - Merge small ops into fewer ops (reduce regularity)

    Returns:
        dict with final program, history, discriminator scores
    """
    target_v = np.asarray(target_v, dtype=np.float64)
    target_f = np.asarray(target_f)

    program = initial_program(target_v, target_f)
    history = []

    for rnd in range(max_rounds):
        cad_v, cad_f = program.evaluate()
        if len(cad_v) == 0:
            break

        # CRITIC: how CAD-like is this mesh?
        cad_score = discriminate_cad_vs_mesh(cad_v, cad_f)
        accuracy = score_accuracy(program, target_v, target_f)
        features = compute_discriminator_features(cad_v, cad_f)

        # ATTACKER: try mutations to reduce CAD-likeness while keeping accuracy
        best_program = program
        best_objective = cad_score - accuracy * 0.5  # want low cad_score, high accuracy
        best_action = "held"

        mutations = _generate_anti_cad_mutations(program, features, target_v, target_f)

        for action_name, mutant in mutations:
            mut_v, mut_f = mutant.evaluate()
            if len(mut_v) == 0:
                continue
            mut_cad_score = discriminate_cad_vs_mesh(mut_v, mut_f)
            mut_accuracy = score_accuracy(mutant, target_v, target_f)

            # Accept if: less CAD-like AND accuracy didn't tank
            objective = mut_cad_score - mut_accuracy * 0.5
            if objective < best_objective and mut_accuracy > 0.3:
                best_objective = objective
                best_program = mutant
                best_action = action_name

        program = best_program

        history.append({
            "round": rnd,
            "cad_score": round(cad_score, 4),
            "accuracy": round(accuracy, 4),
            "action": best_action,
            "n_ops": program.n_enabled(),
            "features": {k: round(v, 4) for k, v in features.items()},
        })

    # Also refine for accuracy using the standard gap-filling
    for _ in range(5):
        gaps = find_program_gaps(program, target_v, target_f)
        if not gaps or gaps[0].residual_score < 0.01:
            break
        for gap in gaps[:2]:
            if gap.action == "add" and program.n_enabled() < max_ops:
                p = program.copy()
                add_operation(p, gap)
                if score_accuracy(p, target_v, target_f) > score_accuracy(program, target_v, target_f):
                    program = p

    final_v, final_f = program.evaluate()
    return {
        "program": program,
        "history": history,
        "cad_vertices": final_v,
        "cad_faces": final_f,
        "final_cad_score": discriminate_cad_vs_mesh(final_v, final_f) if len(final_v) > 0 else 1.0,
        "final_accuracy": score_accuracy(program, target_v, target_f),
    }


def _generate_anti_cad_mutations(program, features, target_v, target_f):
    """Generate CadProgram mutations that reduce CAD-likeness.

    Strategies based on what discriminators detect:
    - high edge regularity → perturb subdivision counts
    - high symmetry → slight asymmetric offsets
    - high planarity → add slight curvature
    - regular valence → merge or split operations
    """
    mutations = []

    # Strategy 1: Perturb subdivision parameters (break edge regularity)
    if features.get("edge_regularity", 1) < 0.4:
        p = program.copy()
        rng = np.random.RandomState(42)
        for op in p.operations:
            if op.enabled and "divs" in op.params:
                base = op.params["divs"]
                op.params["divs"] = base + rng.randint(-3, 4)
                op.params["divs"] = max(6, op.params["divs"])
            if op.enabled and "h_divs" in op.params:
                base = op.params["h_divs"]
                op.params["h_divs"] = base + rng.randint(-2, 3)
                op.params["h_divs"] = max(3, op.params["h_divs"])
        p.invalidate_cache()
        mutations.append(("perturb_divs", p))

    # Strategy 2: Break symmetry with small parameter offsets
    if features.get("symmetry", 0) > 0.7:
        p = program.copy()
        for op in p.operations:
            if not op.enabled:
                continue
            center = op.params.get("center")
            if center is not None:
                c = np.asarray(center, dtype=np.float64)
                bbox_diag = np.linalg.norm(
                    target_v.max(0) - target_v.min(0))
                offset = np.random.RandomState(123).randn(3) * bbox_diag * 0.005
                op.params["center"] = (c + offset).tolist()
        p.invalidate_cache()
        mutations.append(("break_symmetry", p))

    # Strategy 3: Refine toward target (improve accuracy, may change mesh structure)
    gaps = find_program_gaps(program, target_v, target_f, max_gaps=2)
    for gap in gaps[:1]:
        if gap.nearest_program_op >= 0:
            p = program.copy()
            refine_operation(p, gap.nearest_program_op, target_v, target_f)
            mutations.append(("refine_accuracy", p))

    # Strategy 4: Add small operations to create irregularity
    if program.n_enabled() < 8 and features.get("planarity", 0) > 0.4:
        p = program.copy()
        center = target_v.mean(axis=0)
        bbox = target_v.max(0) - target_v.min(0)
        tiny_r = float(bbox.min()) * 0.02
        p.operations.append(CadOp("sphere", {
            "center": (center + np.random.RandomState(77).randn(3) * bbox * 0.3).tolist(),
            "radius": tiny_r,
            "divs": 7,  # odd number → irregular
        }))
        p.invalidate_cache()
        mutations.append(("add_irregularity", p))

    # Strategy 5: Simplify (fewer ops can sometimes look less "manufactured")
    p = program.copy()
    simplify_program(p, target_v, target_f)
    mutations.append(("simplify", p))

    # Strategy 5b: Try batch h_divs for all cylinders (match mesh density)
    cyl_disc = [(i, op) for i, op in enumerate(program.operations)
                if op.enabled and op.op_type in ("cylinder", "cone")]
    if cyl_disc:
        for trial_hdivs in [2, 3, 4]:
            p = program.copy()
            changed = False
            for i, op in cyl_disc:
                if p.operations[i].params.get("h_divs", 10) != trial_hdivs:
                    p.operations[i].params["h_divs"] = trial_hdivs
                    changed = True
            if changed:
                p.invalidate_cache()
                mutations.append((f"batch_hdivs_{trial_hdivs}", p))

    # Strategy 6: Add secondary primitive at largest gap
    if program.n_enabled() < 10:
        gaps = find_program_gaps(program, target_v, target_f, max_gaps=3)
        for i, gap in enumerate(gaps):
            if gap.action == "add" and gap.residual_score > 0.3:
                p = program.copy()
                add_operation(p, gap)
                # Refine the newly added op
                refine_operation(p, len(p.operations) - 1,
                                 target_v, target_f, max_iter=15)
                mutations.append((f"add_gap_primitive_{i}", p))

    # Strategy 7: Add subtract_cylinder for concavities
    if program.n_enabled() < 8:
        cad_v, cad_f = program.evaluate()
        if len(cad_v) > 0:
            _try_subtract_mutations(
                program, cad_v, target_v, target_f, mutations)

    # Strategy 8: Feature-targeted fill (same as elegance strategy 14,
    # also in the discriminator loop so it gets tried in both passes)
    if program.n_enabled() < 10:
        _try_feature_targeted_fill(
            program, target_v, target_f, mutations,
            np.random.RandomState(42))

    return mutations


def _try_subtract_mutations(program, cad_v, target_v, target_f, mutations):
    """Try adding subtract_cylinder ops where the CAD mesh overshoots."""
    tree_target = _AKDTree(target_v)
    dists_from_cad, _ = tree_target.query(cad_v)

    # Find vertices where CAD extends beyond target (overshoot)
    threshold = max(np.percentile(dists_from_cad, 85), 1e-6)
    overshoot = dists_from_cad > threshold
    if not np.any(overshoot):
        return

    over_pts = cad_v[overshoot]
    if len(over_pts) < 5:
        return

    # PCA on overshoot region to find axis
    center = over_pts.mean(axis=0)
    centered = over_pts - center
    cov = centered.T @ centered / len(centered)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    spread = np.sqrt(np.maximum(eigvals[order], 0))

    # If elongated, try subtracting a cylinder along principal axis
    if spread[0] > 1e-8 and spread[0] / max(spread[1], 1e-8) > 1.5:
        axis = eigvecs[:, order[0]]
        radii = np.linalg.norm(
            centered - np.outer(centered @ axis, axis), axis=1)
        p = program.copy()
        p.operations.append(CadOp("subtract_cylinder", {
            "center": center.tolist(),
            "axis": axis.tolist(),
            "radius": float(np.median(radii) * 1.2),
            "height": float(spread[0] * 2.5),
        }))
        p.invalidate_cache()
        mutations.append(("subtract_overshoot", p))


# ============================================================================
# Loop 2: Elegance Tournament
# ============================================================================

def compare_elegance(prog_a, prog_b, target_v, target_f):
    """CRITIC: Judge which of two CadPrograms is more elegant.

    Uses all 12 elegance dimensions.  Returns positive if A is more elegant,
    negative if B is.

    Returns:
        dict with:
            winner: "A" or "B" or "tie"
            margin: float (how much better the winner is)
            a_scores: per-dimension scores for A
            b_scores: per-dimension scores for B
    """
    a_eval = compute_elegance_score(prog_a, target_v, target_f)
    b_eval = compute_elegance_score(prog_b, target_v, target_f)

    diff = a_eval["total"] - b_eval["total"]

    if abs(diff) < 0.005:
        winner = "tie"
    elif diff > 0:
        winner = "A"
    else:
        winner = "B"

    return {
        "winner": winner,
        "margin": abs(diff),
        "a_total": a_eval["total"],
        "b_total": b_eval["total"],
        "a_scores": a_eval["scores"],
        "b_scores": b_eval["scores"],
        "a_summary": a_eval["program_summary"],
        "b_summary": b_eval["program_summary"],
    }


def _mutate_for_elegance(program, target_v, target_f, rng):
    """Generate mutations that might improve elegance score.

    Draws on GrabCAD top-model patterns:
    - Prefer fewer, cleaner operations
    - Exploit symmetry
    - Replace boolean combos with primitives
    - Improve origin anchoring
    """
    mutations = []
    acc = score_accuracy(program, target_v, target_f)

    # Strategy 1: Simplify (remove redundant ops)
    p = program.copy()
    simplify_program(p, target_v, target_f)
    if p.n_enabled() < program.n_enabled():
        mutations.append(("simplify", p))

    # Strategy 2: Replace multiple generators with one + mirror
    enabled = [i for i, op in enumerate(program.operations) if op.enabled]
    generators = [(i, program.operations[i]) for i in enabled
                  if program.operations[i].op_type not in
                  ("translate", "scale", "rotate", "mirror",
                   "subtract_cylinder", "union")]

    if len(generators) >= 2:
        # Try keeping just the first generator + mirror
        p = program.copy()
        for idx, _ in generators[1:]:
            p.operations[idx].enabled = False
        # Add a mirror op
        p.operations.append(CadOp("mirror", {
            "normal": [1, 0, 0], "point": [0, 0, 0],
        }))
        p.invalidate_cache()
        mirror_acc = score_accuracy(p, target_v, target_f)
        if mirror_acc > 0.4:
            mutations.append(("mirror_replace", p))

    # Strategy 3: Upgrade op type (e.g., replace sphere+translate with sphere at center)
    for i, op in enumerate(program.operations):
        if not op.enabled:
            continue
        if op.op_type == "sphere" and "center" not in op.params:
            p = program.copy()
            p.operations[i].params["center"] = [0, 0, 0]
            p.invalidate_cache()
            mutations.append(("anchor_to_origin", p))
            break

    # Strategy 4: Replace subtract_cylinder with simpler primitive
    for i, op in enumerate(program.operations):
        if op.enabled and op.op_type == "subtract_cylinder":
            p = program.copy()
            p.operations[i].enabled = False
            p.invalidate_cache()
            rm_acc = score_accuracy(p, target_v, target_f)
            if rm_acc > 0.5:
                mutations.append(("remove_boolean", p))

    # Strategy 4b: Upgrade cylinder → profiled_cylinder for tapered shapes
    for i, op in enumerate(program.operations):
        if op.enabled and op.op_type == "cylinder":
            from .cad_program import _make_candidate_op
            try:
                # Get the target vertices near this cylinder
                cad_v_temp, _ = program.evaluate()
                if len(cad_v_temp) == 0:
                    continue
                # Use profiled_cylinder fitting on the full target
                pc_op = _make_candidate_op("profiled_cylinder", target_v)
                if pc_op is not None:
                    p = program.copy()
                    p.operations[i] = pc_op
                    p.invalidate_cache()
                    mutations.append(("upgrade_to_profiled", p))
                    break
            except Exception:
                pass

    # Strategy 5: Refine existing ops for better accuracy without adding more
    # Skip if accuracy is already very high (expensive and unlikely to help)
    if acc < 0.99:
        for i, op in enumerate(program.operations):
            if op.enabled and op.op_type in ("sphere", "cylinder", "profiled_cylinder",
                                               "box", "cone", "revolve"):
                p = program.copy()
                refine_operation(p, i, target_v, target_f, max_iter=15)
                mutations.append(("refine_" + op.op_type, p))
                break

    # Strategy 6: Random parameter perturbation (explore new territory)
    p = program.copy()
    for op in p.operations:
        if not op.enabled:
            continue
        for key, val in list(op.params.items()):
            if isinstance(val, (int, float)) and key != "divs":
                op.params[key] = val * (1.0 + rng.uniform(-0.05, 0.05))
        break  # only perturb first op
    p.invalidate_cache()
    mutations.append(("perturb_params", p))

    # Strategy 7: Add an operation to fill gaps (if accuracy is low)
    if acc < 0.95 and program.n_enabled() < 8:
        gaps = find_program_gaps(program, target_v, target_f, max_gaps=1)
        if gaps:
            p = program.copy()
            add_operation(p, gaps[0])
            mutations.append(("add_for_accuracy", p))

    # Strategy 8: Add + refine gap primitive (more thorough than strategy 7)
    if acc < 0.99 and program.n_enabled() < 10:
        gaps = find_program_gaps(program, target_v, target_f, max_gaps=2)
        for i, gap in enumerate(gaps):
            if gap.action == "add":
                p = program.copy()
                add_operation(p, gap)
                refine_operation(p, len(p.operations) - 1,
                                 target_v, target_f, max_iter=10)
                mutations.append((f"add_refined_gap_{i}", p))

    # Strategy 9: Multi-gap fill — add 2-3 primitives at once
    if acc < 0.6 and program.n_enabled() < 4:
        gaps = find_program_gaps(program, target_v, target_f, max_gaps=3)
        if len(gaps) >= 2:
            p = program.copy()
            for gap in gaps:
                add_operation(p, gap)
            # Refine all new ops
            n_old = program.n_enabled()
            for j in range(len(p.operations) - len(gaps), len(p.operations)):
                refine_operation(p, j, target_v, target_f, max_iter=10)
            mutations.append(("multi_gap_fill", p))

    # Strategy 10: Subtract cylinder for holes/concavities
    if acc < 0.8 and program.n_enabled() < 5:
        cad_v, cad_f = program.evaluate()
        if len(cad_v) > 0:
            _try_subtract_for_elegance(
                program, cad_v, target_v, target_f, mutations, rng)

    # Strategy 11: Add torus ring at gap (common decorative/structural feature)
    if acc < 0.8 and program.n_enabled() < 5:
        _try_torus_ring(program, target_v, target_f, mutations)

    # Strategy 12: Refine all ops (not just first) for multi-op programs
    if 2 <= program.n_enabled() <= 8:
        p = program.copy()
        for i, op in enumerate(p.operations):
            if op.enabled:
                refine_operation(p, i, target_v, target_f, max_iter=10)
        mutations.append(("refine_all_ops", p))

    # Strategy 12b: Try batch h_divs for all cylinder ops at once
    # (matching target mesh density is usually the biggest single improvement)
    cyl_ops = [(i, op) for i, op in enumerate(program.operations)
               if op.enabled and op.op_type in ("cylinder", "cone")]
    if cyl_ops:
        for trial_hdivs in [2, 3, 4]:
            p = program.copy()
            changed = False
            for i, op in cyl_ops:
                if p.operations[i].params.get("h_divs", 10) != trial_hdivs:
                    p.operations[i].params["h_divs"] = trial_hdivs
                    changed = True
            if changed:
                p.invalidate_cache()
                mutations.append((f"batch_hdivs_{trial_hdivs}", p))

    # Strategy 12b2: Intensive refinement of single-op programs
    # For single-revolve or single-profiled_cylinder programs, try re-fitting
    if program.n_enabled() <= 2:
        for i, op in enumerate(program.operations):
            if op.enabled and op.op_type in ("revolve", "profiled_cylinder"):
                p = program.copy()
                refine_operation(p, i, target_v, target_f, max_iter=30)
                mutations.append(("intensive_refine", p))
                break

    # Strategy 12c: Per-component profiled_cylinder upgrade
    # For programs with cylinder ops, try upgrading each to profiled_cylinder
    if acc < 0.99:
        for i, op in enumerate(program.operations):
            if (op.enabled and op.op_type == "cylinder"
                    and op.params.get("height", 0) > 0):
                from .cad_program import _make_candidate_op
                try:
                    # Collect target vertices near this cylinder
                    cad_v_temp, _ = program.evaluate()
                    if len(cad_v_temp) == 0:
                        continue
                    op_mesh = _eval_op(op, [])
                    if op_mesh is None:
                        continue
                    op_v, _ = op_mesh
                    tree = _AKDTree(op_v)
                    bbox_diag = float(np.linalg.norm(
                        target_v.max(0) - target_v.min(0)))
                    d, _ = tree.query(target_v)
                    near_mask = d < bbox_diag * 0.15
                    near_v = target_v[near_mask]
                    if len(near_v) < 10:
                        continue
                    pc_op = _make_candidate_op("profiled_cylinder", near_v)
                    if pc_op is not None:
                        p = program.copy()
                        p.operations[i] = pc_op
                        p.invalidate_cache()
                        mutations.append((f"upgrade_cyl_{i}_to_profiled", p))
                        break  # Only try one per round
                except Exception:
                    pass

    # Strategy 13: Add scaled copy at offset (cover more of the target)
    if acc < 0.7 and program.n_enabled() == 1:
        _try_scaled_copy(program, target_v, target_f, mutations, rng)

    # Strategy 14: Feature-targeted fill — find the worst-covered
    # medium/small feature and add a primitive fitted to its vertices
    if program.n_enabled() < 10:
        _try_feature_targeted_fill(
            program, target_v, target_f, mutations, rng)

    # Strategy 17: Intersection fillet — detect uncovered blend regions
    # between fitted primitives and add fillet ops to fill them
    if program.n_enabled() >= 2 and acc < 0.995:
        _try_intersection_fillet(program, target_v, target_f, mutations)

    return mutations


def _try_subtract_for_elegance(program, cad_v, target_v, target_f,
                                mutations, rng):
    """Try subtract_cylinder where CAD overshoots target."""
    tree_target = _AKDTree(target_v)
    dists, _ = tree_target.query(cad_v)

    threshold = max(np.percentile(dists, 80), 1e-6)
    over_mask = dists > threshold
    if np.sum(over_mask) < 10:
        return

    over_pts = cad_v[over_mask]
    center = over_pts.mean(axis=0)
    centered = over_pts - center

    if len(centered) < 4:
        return

    cov = centered.T @ centered / len(centered)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    spread = np.sqrt(np.maximum(eigvals[order], 0))

    if spread[0] < 1e-8:
        return

    axis = eigvecs[:, order[0]]
    radii = np.linalg.norm(
        centered - np.outer(centered @ axis, axis), axis=1)

    p = program.copy()
    p.operations.append(CadOp("subtract_cylinder", {
        "center": center.tolist(),
        "axis": axis.tolist(),
        "radius": float(np.median(radii) * 1.1),
        "height": float(spread[0] * 2.2),
    }))
    p.invalidate_cache()
    mutations.append(("subtract_for_shape", p))


def _try_torus_ring(program, target_v, target_f, mutations):
    """Try adding a torus ring at a gap region."""
    gaps = find_program_gaps(program, target_v, target_f, max_gaps=2)
    for gap in gaps:
        if gap.region_radius < 0.5:
            continue
        # Check if the gap region is ring-like (circular cross-section)
        center = np.asarray(gap.region_center)
        # Use the gap's suggested op — if it's a torus already, great
        if gap.suggested_op == "torus":
            p = program.copy()
            add_operation(p, gap)
            refine_operation(p, len(p.operations) - 1,
                             target_v, target_f, max_iter=15)
            mutations.append(("add_torus_ring", p))
            break
        # Otherwise try fitting a torus at the gap center
        # Estimate major_r from gap radius, minor_r from residual
        p = program.copy()
        p.operations.append(CadOp("torus", {
            "center": center.tolist(),
            "major_r": float(gap.region_radius * 0.8),
            "minor_r": float(gap.region_radius * 0.15),
            "z_center": float(center[2]),
        }))
        p.invalidate_cache()
        refine_operation(p, len(p.operations) - 1,
                         target_v, target_f, max_iter=15)
        mutations.append(("add_torus_ring", p))
        break


def _try_scaled_copy(program, target_v, target_f, mutations, rng):
    """Add a scaled/translated copy of the primary primitive."""
    op = None
    for o in program.operations:
        if o.enabled and o.op_type in ("sphere", "cylinder", "box", "cone"):
            op = o
            break
    if op is None:
        return

    # Find where the target extends beyond current coverage
    gaps = find_program_gaps(program, target_v, target_f, max_gaps=1)
    if not gaps:
        return
    gap = gaps[0]

    p = program.copy()
    # Add a copy of the same primitive type at the gap center
    new_params = copy.deepcopy(op.params)
    new_params["center"] = np.asarray(gap.region_center).tolist()
    # Scale to fit gap region
    bbox_diag = float(np.linalg.norm(target_v.max(0) - target_v.min(0)))
    scale_factor = gap.region_radius / max(bbox_diag * 0.5, 1e-8)
    if "radius" in new_params:
        new_params["radius"] = float(new_params["radius"] * scale_factor)
    if "dimensions" in new_params:
        dims = np.asarray(new_params["dimensions"]) * scale_factor
        new_params["dimensions"] = dims.tolist()

    p.operations.append(CadOp(op.op_type, new_params))
    p.invalidate_cache()
    refine_operation(p, len(p.operations) - 1,
                     target_v, target_f, max_iter=15)
    mutations.append(("add_scaled_copy", p))


def _try_feature_targeted_fill(program, target_v, target_f, mutations, rng):
    """Find worst-covered medium/small features and add primitives for them.

    Unlike gap-filling (which uses distance percentiles), this directly
    decomposes the target into connected components, measures per-feature
    coverage, and adds a primitive fitted to the worst-covered feature's
    own vertices.
    """
    cad_v, cad_f = program.evaluate()
    if len(cad_v) == 0:
        return

    target_v = np.asarray(target_v, dtype=np.float64)
    features = _decompose_features(target_v, target_f)

    ms_features = [f for f in features if f["size_class"] in ("medium", "small")]
    if not ms_features:
        return

    bbox_diag = float(np.linalg.norm(target_v.max(0) - target_v.min(0)))
    coverage_thresh = bbox_diag * 0.05

    cad_tree = _AKDTree(cad_v)

    # Score each feature by coverage
    scored = []
    for feat in ms_features:
        feat_pts = target_v[feat["verts_idx"]]
        dists, _ = cad_tree.query(feat_pts)
        coverage = float(np.mean(dists < coverage_thresh))
        scored.append((coverage, feat))

    scored.sort(key=lambda x: x[0])

    # Try adding a primitive for the worst-covered features (up to 2)
    from .reconstruct import fit_sphere, fit_cylinder, fit_box, fit_profiled_cylinder
    for coverage, feat in scored[:2]:
        if coverage > 0.7:
            break  # already well-covered

        feat_pts = target_v[feat["verts_idx"]]
        center = feat["center"]
        radius = feat["radius"]

        if len(feat_pts) < 4:
            continue

        # Try sphere and cylinder, pick whichever fits better
        best_op = None
        best_residual = float("inf")

        try:
            sp = fit_sphere(feat_pts)
            if sp["residual"] < best_residual:
                best_residual = sp["residual"]
                best_op = CadOp("sphere", {
                    "center": sp["center"].tolist(),
                    "radius": sp["radius"],
                })
        except Exception:
            pass

        try:
            cy = fit_cylinder(feat_pts)
            if cy["residual"] < best_residual:
                best_residual = cy["residual"]
                best_op = CadOp("cylinder", {
                    "center": cy["center"].tolist(),
                    "axis": cy["axis"].tolist(),
                    "radius": cy["radius"],
                    "height": cy["height"],
                })
        except Exception:
            pass

        try:
            pc = fit_profiled_cylinder(feat_pts)
            if pc["residual"] < best_residual and pc.get("taper_ratio", 0) > 0.05:
                best_residual = pc["residual"]
                best_op = CadOp("profiled_cylinder", {
                    "center": pc["center"].tolist(),
                    "axis": pc["axis"].tolist(),
                    "height": pc["height"],
                    "radii": pc["radii"],
                    "heights": pc["heights"],
                })
        except Exception:
            pass

        try:
            bx = fit_box(feat_pts)
            if bx["residual"] < best_residual:
                best_residual = bx["residual"]
                best_op = CadOp("box", {
                    "center": bx["center"].tolist(),
                    "dimensions": bx["dimensions"].tolist(),
                })
        except Exception:
            pass

        if best_op is None:
            continue

        p = program.copy()
        p.operations.append(best_op)
        p.invalidate_cache()
        refine_operation(p, len(p.operations) - 1,
                         target_v, target_f, max_iter=30)
        mutations.append(("feature_fill", p))

    # Also try a multi-feature fill: add primitives for the 2-3 worst
    # features at once (amortizes the conciseness penalty across more
    # feature coverage)
    worst = [(cov, feat) for cov, feat in scored if cov < 0.7]
    if len(worst) >= 2 and program.n_enabled() < 4:
        p = program.copy()
        added = 0
        for _, feat in worst[:3]:
            feat_pts = target_v[feat["verts_idx"]]
            if len(feat_pts) < 4:
                continue
            try:
                cy = fit_cylinder(feat_pts)
                op = CadOp("cylinder", {
                    "center": cy["center"].tolist(),
                    "axis": cy["axis"].tolist(),
                    "radius": cy["radius"],
                    "height": cy["height"],
                })
            except Exception:
                try:
                    sp = fit_sphere(feat_pts)
                    op = CadOp("sphere", {
                        "center": sp["center"].tolist(),
                        "radius": sp["radius"],
                    })
                except Exception:
                    continue
            p.operations.append(op)
            added += 1
        if added >= 2:
            p.invalidate_cache()
            for j in range(len(p.operations) - added, len(p.operations)):
                refine_operation(p, j, target_v, target_f, max_iter=15)
            mutations.append(("multi_feature_fill", p))


def _try_intersection_fillet(program, target_v, target_f, mutations):
    """Detect uncovered blend regions between primitives and add fillet ops.

    Uses detect_intersection_fillets() to find gaps at primitive intersections,
    then fits fillet CadOps to fill them. This is a post-fitting strategy:
    primitives are already in place, and fillets are added to cover the
    transition zones that primitives can't reach.
    """
    try:
        from .segmentation import detect_intersection_fillets, fit_fillet_op
    except ImportError:
        return

    # Check if program already has fillets — avoid stacking
    n_fillets = sum(1 for op in program.operations
                    if op.enabled and op.op_type == "fillet")
    if n_fillets >= 3:
        return

    try:
        fillets = detect_intersection_fillets(program, target_v, target_f)
    except Exception:
        return

    if not fillets:
        return

    # Sort by number of uncovered vertices (most impactful first)
    fillets.sort(key=lambda f: f["n_vertices"], reverse=True)

    for fl in fillets[:2]:  # Try at most 2 fillets per round
        if fl["n_vertices"] < 10:
            continue

        try:
            fillet_op = fit_fillet_op(fl, target_v, target_f)
        except Exception:
            continue

        if fillet_op is None:
            continue

        p = program.copy()
        p.operations.append(fillet_op)
        p.invalidate_cache()

        # Quick refine
        try:
            refine_operation(p, len(p.operations) - 1,
                             target_v, target_f, max_iter=10)
        except Exception:
            pass

        mutations.append(("intersection_fillet", p))


def run_elegance_tournament(target_v, target_f, max_rounds=30, n_contestants=4):
    """Loop 2: Tournament where CadPrograms compete on elegance.

    Architecture:
    - A population of CadPrograms (contestants) compete pairwise.
    - The CRITIC judges each pair using compare_elegance().
    - The ATTACKER mutates programs to fool the critic into picking them.
    - Each round: mutate all contestants, keep those the critic prefers.
    - Programs evolve toward both accuracy AND elegance simultaneously.

    Returns:
        dict with champion program, history, final scores
    """
    target_v = np.asarray(target_v, dtype=np.float64)
    target_f = np.asarray(target_f)

    # Initialize population with different seeds
    population = []
    base = initial_program(target_v, target_f)
    population.append(base)

    # Create variants by perturbing the initial program
    for i in range(n_contestants - 1):
        p = base.copy()
        rng_init = np.random.RandomState(42 + i)
        for op in p.operations:
            if not op.enabled:
                continue
            for key, val in list(op.params.items()):
                if isinstance(val, (int, float)) and key not in ("divs", "h_divs"):
                    op.params[key] = val * (1.0 + rng_init.uniform(-0.15, 0.15))
        p.invalidate_cache()
        population.append(p)

    history = []
    rng = np.random.RandomState(0)

    for rnd in range(max_rounds):
        # --- ATTACK phase: generate mutants for each contestant ---
        candidates = []
        for prog in population:
            mutations = _mutate_for_elegance(prog, target_v, target_f, rng)
            # Include original
            candidates.append(("original", prog))
            for name, mut in mutations:
                candidates.append((name, mut))

        # --- CRITIC phase: score all candidates ---
        scored = []
        for name, cand in candidates:
            elegance = compute_elegance_score(cand, target_v, target_f)
            scored.append((elegance["total"], name, cand, elegance))

        scored.sort(key=lambda x: x[0], reverse=True)

        # Keep top N
        new_pop = []
        seen_summaries = set()
        for total, name, cand, eleg in scored:
            summary = cand.summary()
            if summary not in seen_summaries:
                new_pop.append(cand)
                seen_summaries.add(summary)
            if len(new_pop) >= n_contestants:
                break

        # Ensure population doesn't shrink
        while len(new_pop) < n_contestants and scored:
            new_pop.append(scored[len(new_pop) % len(scored)][2].copy())

        population = new_pop

        # --- Record ---
        champion = scored[0]
        history.append({
            "round": rnd,
            "champion_score": round(champion[0], 4),
            "champion_action": champion[1],
            "champion_summary": champion[2].summary(),
            "champion_n_ops": champion[2].n_enabled(),
            "champion_accuracy": round(champion[3]["scores"]["accuracy"], 4),
            "champion_conciseness": round(champion[3]["scores"]["conciseness"], 4),
            "population_scores": [round(s[0], 4) for s in scored[:n_contestants]],
        })

        # Pairwise comparison between top 2 for the log
        if len(population) >= 2:
            comp = compare_elegance(population[0], population[1], target_v, target_f)
            history[-1]["top2_comparison"] = {
                "winner": comp["winner"],
                "margin": round(comp["margin"], 4),
            }

    # Final champion
    final_scores = []
    for prog in population:
        eleg = compute_elegance_score(prog, target_v, target_f)
        final_scores.append((eleg["total"], prog, eleg))
    final_scores.sort(key=lambda x: x[0], reverse=True)

    champion = final_scores[0]
    champ_v, champ_f = champion[1].evaluate()

    return {
        "program": champion[1],
        "elegance_score": champion[0],
        "elegance_details": champion[2],
        "history": history,
        "cad_vertices": champ_v,
        "cad_faces": champ_f,
        "population": [(s, p.summary()) for s, p, _ in final_scores],
    }
