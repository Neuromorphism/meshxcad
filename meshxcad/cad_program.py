"""CAD program abstraction: represent a CAD model as a sequence of parametric
operations and evolve it adversarially toward both accuracy and elegance.

A CadProgram is a list of CadOps that, when evaluated sequentially, produce
a triangle mesh.  The adversarial loop operates at the program level:

  RED team:  finds spatial regions where the program's mesh doesn't match
             the target, and classifies what operation is missing.
  BLUE team: mutates the program (add/remove/refine/simplify operations)
             to close gaps while keeping the program short.

The total cost balances accuracy and elegance:
    total_cost = mesh_distance * (1.0 + alpha * complexity + beta * n_ops)
"""

import math
import copy
import json
import numpy as np
from dataclasses import dataclass, field, asdict
from typing import Optional

from scipy.spatial import KDTree
from .gpu import AcceleratedKDTree as _AKDTree, covariance_pca as _gpu_pca, eigh as _gpu_eigh

from .general_align import hausdorff_distance, surface_distance_map
from .reconstruct import (
    classify_mesh, fit_sphere, fit_cylinder, fit_cone, fit_box,
    fit_profiled_cylinder, fit_revolve_profile, _rotation_between,
)
from .objects.builder import revolve_profile, combine_meshes, make_torus
from .objects.operations import (
    extrude_polygon, sweep_along_path, make_regular_polygon,
)
from .synthetic import make_sphere_mesh, make_cylinder_mesh


# ---------------------------------------------------------------------------
# Operation vocabulary
# ---------------------------------------------------------------------------

OP_COSTS = {
    "sphere": 1.0,
    "cylinder": 1.0,
    "profiled_cylinder": 1.5,
    "cone": 1.5,
    "box": 1.0,
    "torus": 1.5,
    "revolve": 2.0,
    "extrude": 1.5,
    "sweep": 3.0,
    "fillet": 1.5,
    "translate": 0.2,
    "scale": 0.3,
    "rotate": 0.3,
    "union": 0.5,
    "subtract_cylinder": 1.5,
    "mirror": 0.5,
}

# Elegance penalty weights
ALPHA = 0.02   # per-unit complexity cost
BETA = 0.05    # per-operation count
GAMMA = 0.001  # per-parameter


# ---------------------------------------------------------------------------
# Mesh complexity and adaptive op budget
# ---------------------------------------------------------------------------

def mesh_complexity(vertices, faces):
    """Estimate geometric complexity of a target mesh.

    Returns a dict with:
        complexity: float 0-1 (0=simple sphere, 1=maximally complex)
        n_components: connected component count
        curvature_entropy: entropy of discrete curvature distribution
        op_budget: recommended maximum operation count

    The complexity drives the op budget: simple shapes get few ops,
    complex shapes get many.  Principle: as few ops as possible, as many
    as necessary.
    """
    import math as _math
    vertices = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces)

    n_verts = len(vertices)
    n_faces = len(faces)

    # Component count
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import connected_components
    n = len(vertices)
    if n_faces > 0 and n > 0:
        rows = np.concatenate([faces[:, 0], faces[:, 1], faces[:, 2]])
        cols = np.concatenate([faces[:, 1], faces[:, 2], faces[:, 0]])
        adj = csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n))
        n_comp, _ = connected_components(adj, directed=False)
    else:
        n_comp = 1

    # Curvature entropy (vectorised — no Python loops over faces)
    curvature_entropy = 0.0
    if n_verts > 3 and n_faces > 0:
        try:
            angle_sum = np.zeros(n_verts)
            # Compute angles at each vertex of each face in bulk
            for k in range(3):
                i_idx = faces[:, k]
                j_idx = faces[:, (k + 1) % 3]
                l_idx = faces[:, (k + 2) % 3]
                v1 = vertices[j_idx] - vertices[i_idx]  # (n_faces, 3)
                v2 = vertices[l_idx] - vertices[i_idx]  # (n_faces, 3)
                len1 = np.linalg.norm(v1, axis=1)
                len2 = np.linalg.norm(v2, axis=1)
                valid = (len1 > 1e-12) & (len2 > 1e-12)
                cos_a = np.zeros(n_faces)
                cos_a[valid] = np.clip(
                    np.einsum('ij,ij->i', v1[valid], v2[valid])
                    / (len1[valid] * len2[valid]),
                    -1, 1)
                angles = np.zeros(n_faces)
                angles[valid] = np.arccos(cos_a[valid])
                np.add.at(angle_sum, i_idx, angles)
            curvature = 2 * _math.pi - angle_sum
            n_bins = min(30, max(5, n_verts // 10))
            hist, _ = np.histogram(curvature, bins=n_bins, density=True)
            hist = hist[hist > 0]
            if len(hist) > 0:
                p = hist / hist.sum()
                curvature_entropy = float(-np.sum(p * np.log2(p + 1e-12)))
                max_entropy = _math.log2(n_bins)
                curvature_entropy = curvature_entropy / max(max_entropy, 1e-8)
        except Exception:
            curvature_entropy = 0.5

    # Bounding box aspect spread
    bbox = vertices.max(0) - vertices.min(0)
    bbox_sorted = np.sort(bbox)[::-1]
    if bbox_sorted[0] > 1e-12:
        ratios = bbox_sorted / bbox_sorted[0]
        aspect_spread = float(np.std(ratios))
    else:
        aspect_spread = 0.0

    # Edge entropy
    edge_entropy = 0.0
    if n_faces > 0:
        try:
            edges = np.concatenate([
                np.linalg.norm(vertices[faces[:, 1]] - vertices[faces[:, 0]], axis=1),
                np.linalg.norm(vertices[faces[:, 2]] - vertices[faces[:, 1]], axis=1),
                np.linalg.norm(vertices[faces[:, 0]] - vertices[faces[:, 2]], axis=1),
            ])
            n_bins = min(20, max(5, len(edges) // 20))
            hist, _ = np.histogram(edges, bins=n_bins, density=True)
            hist = hist[hist > 0]
            if len(hist) > 0:
                p = hist / hist.sum()
                edge_entropy = float(-np.sum(p * np.log2(p + 1e-12)))
                edge_entropy = edge_entropy / max(_math.log2(n_bins), 1e-8)
        except Exception:
            edge_entropy = 0.5

    # Composite complexity
    import math as _math
    comp_factor = min(1.0, _math.log2(1 + n_comp) / 5.0)
    vert_factor = min(1.0, _math.log2(1 + n_verts) / 13.0)

    complexity = (
        0.30 * comp_factor +
        0.30 * curvature_entropy +
        0.15 * edge_entropy +
        0.15 * vert_factor +
        0.10 * min(1.0, aspect_spread * 3)
    )
    complexity = max(0.0, min(1.0, complexity))

    # Op budget: 4 (simple) to 60 (complex)
    base_ops = 4
    max_budget = 60
    op_budget = int(base_ops + complexity * (max_budget - base_ops))
    op_budget = max(op_budget, min(n_comp, max_budget))

    return {
        "complexity": round(complexity, 4),
        "n_components": n_comp,
        "curvature_entropy": round(curvature_entropy, 4),
        "edge_entropy": round(edge_entropy, 4),
        "aspect_spread": round(aspect_spread, 4),
        "vertex_count": n_verts,
        "op_budget": op_budget,
    }


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class CadOp:
    """A single parametric CAD operation."""
    op_type: str
    params: dict
    enabled: bool = True

    @property
    def complexity_cost(self):
        return OP_COSTS.get(self.op_type, 1.0)

    @property
    def param_count(self):
        count = 0
        for v in self.params.values():
            if isinstance(v, (list, np.ndarray)):
                count += len(v)
            else:
                count += 1
        return count

    def to_dict(self):
        d = {"op_type": self.op_type, "enabled": self.enabled}
        params = {}
        for k, v in self.params.items():
            if isinstance(v, np.ndarray):
                params[k] = v.tolist()
            else:
                params[k] = v
        d["params"] = params
        return d


@dataclass
class ProgramGap:
    """RED team output: a spatial region where the program fails."""
    region_center: np.ndarray
    region_radius: float
    residual_score: float
    suggested_op: str
    suggested_params: dict
    nearest_program_op: int  # -1 if none
    action: str  # "add", "refine", "remove"


class CadProgram:
    """An ordered list of parametric operations that produces a mesh."""

    def __init__(self, operations=None):
        self.operations: list[CadOp] = operations or []
        self._cached_mesh = None
        self._cache_valid = False

    def invalidate_cache(self):
        self._cache_valid = False
        self._cached_mesh = None

    def evaluate(self):
        """Execute all enabled ops, return (vertices, faces)."""
        if self._cache_valid and self._cached_mesh is not None:
            return self._cached_mesh

        meshes = []
        for op in self.operations:
            if not op.enabled:
                continue
            result = _eval_op(op, meshes)
            if result is not None:
                meshes.append(result)

        if not meshes:
            return np.zeros((0, 3)), np.zeros((0, 3), dtype=int)

        v, f = combine_meshes(meshes)
        self._cached_mesh = (v, f)
        self._cache_valid = True
        return v, f

    def total_complexity(self):
        """Sum of complexity costs of enabled ops."""
        return sum(op.complexity_cost for op in self.operations if op.enabled)

    def total_param_count(self):
        return sum(op.param_count for op in self.operations if op.enabled)

    def n_enabled(self):
        return sum(1 for op in self.operations if op.enabled)

    def elegance_penalty(self):
        return (ALPHA * self.total_complexity() +
                BETA * self.n_enabled() +
                GAMMA * self.total_param_count())

    def total_cost(self, target_v, target_f):
        """Accuracy * (1 + elegance_penalty).  Lower is better."""
        cad_v, cad_f = self.evaluate()
        if len(cad_v) == 0:
            return float('inf')
        hd = hausdorff_distance(cad_v, target_v)
        distance = hd["mean_symmetric"]
        bbox_diag = float(np.linalg.norm(
            np.asarray(target_v).max(axis=0) - np.asarray(target_v).min(axis=0)))
        if bbox_diag > 1e-12:
            distance /= bbox_diag
        return distance * (1.0 + self.elegance_penalty())

    def copy(self):
        new_ops = [CadOp(op.op_type, copy.deepcopy(op.params), op.enabled)
                   for op in self.operations]
        return CadProgram(new_ops)

    def to_dict(self):
        return {
            "operations": [op.to_dict() for op in self.operations],
            "n_ops": self.n_enabled(),
            "complexity": round(self.total_complexity(), 2),
        }

    def summary(self):
        """Human-readable one-line summary."""
        ops = [op.op_type for op in self.operations if op.enabled]
        return f"{len(ops)} ops: {' → '.join(ops)}"

    @classmethod
    def from_dict(cls, d):
        ops = []
        for od in d["operations"]:
            params = {}
            for k, v in od["params"].items():
                if isinstance(v, list):
                    params[k] = np.array(v)
                else:
                    params[k] = v
            ops.append(CadOp(od["op_type"], params, od.get("enabled", True)))
        return cls(ops)


# ---------------------------------------------------------------------------
# Operation evaluators
# ---------------------------------------------------------------------------

def _eval_fillet_op(p):
    """Evaluate a fillet CadOp.

    A fillet is a blend surface with roughly triangular cross-section
    extruded along a path (the intersection curve of two primitives).

    The cross-section has three edges:
    1. Edge on primitive A surface (outward/radial direction)
    2. Edge on primitive B surface (upward/perpendicular direction)
    3. The blend edge connecting them (optionally concave)

    Params:
        path: (K, 3) 3D path points along the intersection curve
        radius: fillet radius (size of cross-section)
        concavity: 0.0 = flat/chamfer, 1.0 = fully concave (circular blend)
        closed: whether path forms a closed loop (default True)
        n_cross: cross-section resolution (default 6)
        up_dir: (3,) direction toward primitive B (default [0,0,1])
    """
    path = np.asarray(p.get("path", [[0, 0, 0], [1, 0, 0]]), dtype=np.float64)
    radius = float(p.get("radius", 0.1))
    concavity = float(p.get("concavity", 0.5))
    closed = bool(p.get("closed", True))
    n_cross = int(p.get("n_cross", 6))
    up_dir = np.asarray(p.get("up_dir", [0, 0, 1]), dtype=np.float64)
    n_path = len(path)

    if n_path < 2:
        return np.zeros((0, 3)), np.zeros((0, 3), dtype=int)

    # Compute path center for radial direction computation
    path_center = path.mean(axis=0)

    # Normalize up direction
    up_len = np.linalg.norm(up_dir)
    if up_len > 1e-12:
        up_dir = up_dir / up_len
    else:
        up_dir = np.array([0, 0, 1.0])

    verts = []
    for i in range(n_path):
        # Tangent along the path
        if closed:
            tangent = path[(i + 1) % n_path] - path[(i - 1) % n_path]
        else:
            if i == 0:
                tangent = path[1] - path[0]
            elif i == n_path - 1:
                tangent = path[-1] - path[-2]
            else:
                tangent = path[i + 1] - path[i - 1]
        t_len = np.linalg.norm(tangent)
        if t_len < 1e-12:
            tangent = np.array([0, 0, 1.0])
        else:
            tangent /= t_len

        # Radial direction: outward from path center, perpendicular to tangent
        radial = path[i] - path_center
        # Remove tangent component
        radial -= np.dot(radial, tangent) * tangent
        r_len = np.linalg.norm(radial)
        if r_len < 1e-12:
            # Fallback: use cross product of tangent and up
            radial = np.cross(tangent, up_dir)
            r_len = np.linalg.norm(radial)
            if r_len < 1e-12:
                radial = np.array([1, 0, 0.0])
            else:
                radial /= r_len
        else:
            radial /= r_len

        # Cross-section: parametric curve from (path + radius*radial)
        # to (path + radius*up_dir), with optional concavity
        for j in range(n_cross):
            t_param = j / max(n_cross - 1, 1)  # 0 to 1

            # Linear interpolation endpoints
            start_pt = path[i] + radius * radial
            end_pt = path[i] + radius * up_dir

            # Linear blend
            pt = start_pt * (1.0 - t_param) + end_pt * t_param

            # Concave inward displacement (toward path[i])
            # Maximum at t=0.5, zero at endpoints
            blend = concavity * radius * math.sin(t_param * math.pi) * 0.4
            toward_path = path[i] - pt
            tp_len = np.linalg.norm(toward_path)
            if tp_len > 1e-12:
                toward_path /= tp_len
            pt += toward_path * blend

            verts.append(pt)

    verts = np.array(verts, dtype=np.float64)

    # Build faces
    faces = []
    n_segs = n_path if closed else n_path - 1
    for i in range(n_segs):
        i_next = (i + 1) % n_path
        for j in range(n_cross - 1):
            a = i * n_cross + j
            b = i * n_cross + j + 1
            c = i_next * n_cross + j
            d = i_next * n_cross + j + 1
            faces.append([a, c, b])
            faces.append([b, c, d])

    if len(faces) == 0:
        return np.zeros((0, 3)), np.zeros((0, 3), dtype=int)

    return verts, np.array(faces, dtype=np.int64)


def _eval_op(op, existing_meshes):
    """Evaluate a single CadOp, return (vertices, faces) or None."""
    p = op.params
    t = op.op_type

    if t == "sphere":
        v, f = make_sphere_mesh(
            radius=p.get("radius", 1.0),
            lat_divs=int(p.get("divs", 20)),
            lon_divs=int(p.get("divs", 20)))
        center = np.asarray(p.get("center", [0, 0, 0]), dtype=np.float64)
        return v + center, f

    if t == "cylinder":
        v, f = make_cylinder_mesh(
            radius=p.get("radius", 1.0),
            height=p.get("height", 1.0),
            radial_divs=int(p.get("divs", 24)),
            height_divs=int(p.get("h_divs", 10)))
        axis = np.asarray(p.get("axis", [0, 0, 1]), dtype=np.float64)
        center = np.asarray(p.get("center", [0, 0, 0]), dtype=np.float64)
        # Align to axis
        z = np.array([0.0, 0.0, 1.0])
        axis_n = axis / (np.linalg.norm(axis) + 1e-12)
        R = _rotation_between(z, axis_n)
        v = v @ R.T + center
        return v, f

    if t == "profiled_cylinder":
        h = p.get("height", 1.0)
        radii = p.get("radii", [1.0, 1.0])
        heights = p.get("heights", [i / max(len(radii) - 1, 1)
                                     for i in range(len(radii))])
        n_angular = int(p.get("divs", 24))
        # Build revolve profile from (radius, z) pairs
        profile = []
        for ri, hi in zip(radii, heights):
            r = max(float(ri), 0.005)
            z = float(hi) * h
            profile.append((r, z))
        if len(profile) < 2:
            profile = [(max(radii[0], 0.005), 0.0),
                       (max(radii[-1], 0.005), h)]
        v, f = revolve_profile(profile, n_angular)
        center = np.asarray(p.get("center", [0, 0, 0]), dtype=np.float64)
        axis = np.asarray(p.get("axis", [0, 0, 1]), dtype=np.float64)
        # Align to axis
        z_ax = np.array([0.0, 0.0, 1.0])
        axis_n = axis / (np.linalg.norm(axis) + 1e-12)
        R = _rotation_between(z_ax, axis_n)
        v = v @ R.T + center
        return v, f

    if t == "cone":
        h = p.get("height", 1.0)
        base_r = p.get("base_radius", 1.0)
        top_r = p.get("top_radius", 0.1)
        n_h = int(p.get("h_divs", 10))
        profile = [(base_r + (top_r - base_r) * i / n_h,
                     i / n_h * h) for i in range(n_h + 1)]
        profile = [(max(r, 0.01), z) for r, z in profile]
        v, f = revolve_profile(profile, int(p.get("divs", 24)))
        center = np.asarray(p.get("center", [0, 0, 0]), dtype=np.float64)
        return v + center, f

    if t == "box":
        from .reconstruct import _make_box_mesh
        dims = np.asarray(p.get("dimensions", [1, 1, 1]), dtype=np.float64)
        v, f = _make_box_mesh(dims / 2, n_subdiv=p.get("subdivs", 3))
        center = np.asarray(p.get("center", [0, 0, 0]), dtype=np.float64)
        return v + center, f

    if t == "torus":
        v, f = make_torus(
            major_r=p.get("major_r", 5.0),
            minor_r=p.get("minor_r", 1.0),
            z_center=p.get("z_center", 0.0),
            n_angular=p.get("divs", 24),
            n_cross=p.get("cross_divs", 12))
        center = np.asarray(p.get("center", [0, 0, 0]), dtype=np.float64)
        return v + center, f

    if t == "revolve":
        profile = p.get("profile")
        if profile is None:
            # Build from separate radii/heights params if available
            radii = p.get("radii", [1.0, 1.0])
            heights = p.get("heights", [0.0, 1.0])
            profile = list(zip(radii, heights))
        # Ensure all radii are positive
        profile = [(max(float(r), 0.005), float(z)) for r, z in profile]
        v, f = revolve_profile(profile, int(p.get("divs", 24)))
        center = np.asarray(p.get("center", [0, 0, 0]), dtype=np.float64)
        axis = p.get("axis")
        if axis is not None:
            axis = np.asarray(axis, dtype=np.float64)
            z_ax = np.array([0.0, 0.0, 1.0])
            axis_n = axis / (np.linalg.norm(axis) + 1e-12)
            R = _rotation_between(z_ax, axis_n)
            v = v @ R.T
        return v + center, f

    if t == "extrude":
        polygon = p.get("polygon", [(-1, -1), (1, -1), (1, 1), (-1, 1)])
        height = p.get("height", 1.0)
        v, f = extrude_polygon(polygon, height)
        center = np.asarray(p.get("center", [0, 0, 0]), dtype=np.float64)
        return v + center, f

    if t == "sweep":
        profile = np.asarray(p.get("profile", [[1, 0], [0, 1], [-1, 0], [0, -1]]))
        path = np.asarray(p.get("path", [[0, 0, 0], [0, 0, 1]]))
        v, f = sweep_along_path(profile, path, n_profile=p.get("divs", 16))
        return v, f

    if t == "fillet":
        v, f = _eval_fillet_op(p)
        return v, f

    # --- Modifiers: operate on accumulated meshes ---
    if t == "translate":
        if not existing_meshes:
            return None
        offset = np.asarray(p.get("offset", [0, 0, 0]), dtype=np.float64)
        v, f = existing_meshes[-1]
        return v + offset, f

    if t == "scale":
        if not existing_meshes:
            return None
        factors = np.asarray(p.get("factors", [1, 1, 1]), dtype=np.float64)
        center = np.asarray(p.get("center", [0, 0, 0]), dtype=np.float64)
        v, f = existing_meshes[-1]
        return center + (v - center) * factors, f

    if t == "rotate":
        if not existing_meshes:
            return None
        axis = np.asarray(p.get("axis", [0, 0, 1]), dtype=np.float64)
        angle = math.radians(p.get("angle_deg", 0))
        center = np.asarray(p.get("center", [0, 0, 0]), dtype=np.float64)
        v, f = existing_meshes[-1]
        c, s = math.cos(angle), math.sin(angle)
        ax = axis / (np.linalg.norm(axis) + 1e-12)
        K = np.array([[0, -ax[2], ax[1]], [ax[2], 0, -ax[0]], [-ax[1], ax[0], 0]])
        R = np.eye(3) * c + (1 - c) * np.outer(ax, ax) + s * K
        return center + (v - center) @ R.T, f

    if t == "subtract_cylinder":
        if not existing_meshes:
            return None
        # Approximate subtraction: remove vertices inside the cylinder
        v, f = combine_meshes(existing_meshes)
        cyl_center = np.asarray(p.get("center", [0, 0, 0]), dtype=np.float64)
        cyl_axis = np.asarray(p.get("axis", [0, 0, 1]), dtype=np.float64)
        cyl_axis = cyl_axis / (np.linalg.norm(cyl_axis) + 1e-12)
        cyl_r = p.get("radius", 1.0)
        cyl_h = p.get("height", 1.0)
        # Project vertices onto axis
        rel = v - cyl_center
        proj = rel @ cyl_axis
        radial = rel - np.outer(proj, cyl_axis)
        r_dist = np.linalg.norm(radial, axis=1)
        inside = (r_dist < cyl_r) & (proj > -cyl_h / 2) & (proj < cyl_h / 2)
        # Push inside vertices outward to cylinder surface
        v_out = v.copy()
        if np.any(inside):
            r_dist_in = r_dist[inside]
            r_dist_in[r_dist_in < 1e-8] = 1e-8
            scale = cyl_r / r_dist_in
            v_out[inside] = cyl_center + np.outer(proj[inside], cyl_axis) + \
                radial[inside] * scale[:, None]
        return v_out, f

    if t == "mirror":
        if not existing_meshes:
            return None
        v, f = combine_meshes(existing_meshes)
        normal = np.asarray(p.get("normal", [1, 0, 0]), dtype=np.float64)
        normal = normal / (np.linalg.norm(normal) + 1e-12)
        point = np.asarray(p.get("point", [0, 0, 0]), dtype=np.float64)
        d = np.dot(v - point, normal)
        mirrored = v - 2 * np.outer(d, normal)
        return combine_meshes([(v, f), (mirrored, f)])

    if t == "union":
        # No-op structurally — combine_meshes handles it
        return None

    return None


# ---------------------------------------------------------------------------
# RED team: find program gaps
# ---------------------------------------------------------------------------

def find_program_gaps(program, target_v, target_f, max_gaps=5):
    """Find regions where the program's mesh doesn't match the target.

    Returns a list of ProgramGap objects sorted by residual_score descending.
    """
    cad_v, cad_f = program.evaluate()
    target_v = np.asarray(target_v, dtype=np.float64)

    if len(cad_v) == 0:
        center = target_v.mean(axis=0)
        return [ProgramGap(
            region_center=center,
            region_radius=float(np.linalg.norm(target_v.max(0) - target_v.min(0))),
            residual_score=100.0,
            suggested_op="sphere",
            suggested_params={},
            nearest_program_op=-1,
            action="add",
        )]

    # Compute per-vertex distances from target to CAD
    tree_cad = _AKDTree(cad_v)
    dists_to_cad, _ = tree_cad.query(target_v)

    # Find high-residual regions via percentile threshold
    threshold = max(np.percentile(dists_to_cad, 80), 1e-6)
    high_mask = dists_to_cad > threshold
    if not np.any(high_mask):
        return []

    high_pts = target_v[high_mask]
    high_dists = dists_to_cad[high_mask]

    # Simple clustering: divide into spatial clusters via KDTree partitioning
    gaps = _cluster_gaps(high_pts, high_dists, target_v, target_f,
                          program, max_gaps)
    gaps.sort(key=lambda g: g.residual_score, reverse=True)
    return gaps[:max_gaps]


def _cluster_gaps(high_pts, high_dists, target_v, target_f,
                  program, max_clusters):
    """Cluster high-residual points into gap regions."""
    if len(high_pts) < 3:
        return [ProgramGap(
            region_center=high_pts.mean(axis=0) if len(high_pts) > 0
                          else np.zeros(3),
            region_radius=1.0,
            residual_score=float(high_dists.mean()) if len(high_dists) > 0 else 0,
            suggested_op="sphere",
            suggested_params={},
            nearest_program_op=-1,
            action="add",
        )]

    # K-means-style clustering (lightweight, no sklearn dependency)
    n_clusters = min(max_clusters, max(1, len(high_pts) // 20))
    rng = np.random.RandomState(42)
    centers = high_pts[rng.choice(len(high_pts), n_clusters, replace=False)]

    for _ in range(10):
        # Assign
        dist_to_centers = np.linalg.norm(
            high_pts[:, None, :] - centers[None, :, :], axis=2)
        labels = dist_to_centers.argmin(axis=1)
        # Update
        for k in range(n_clusters):
            mask = labels == k
            if np.any(mask):
                centers[k] = high_pts[mask].mean(axis=0)

    gaps = []
    for k in range(n_clusters):
        mask = labels == k
        if not np.any(mask):
            continue
        cluster_pts = high_pts[mask]
        cluster_dists = high_dists[mask]
        center = cluster_pts.mean(axis=0)
        radius = float(np.max(np.linalg.norm(cluster_pts - center, axis=1)))

        # Classify what operation would fill this gap
        suggested_op, suggested_params = _classify_gap(
            cluster_pts, center, radius, target_v, target_f)

        # Find nearest existing program op
        nearest_op = _find_nearest_op(center, program)

        action = "refine" if nearest_op >= 0 and radius < 2.0 else "add"

        gaps.append(ProgramGap(
            region_center=center,
            region_radius=radius,
            residual_score=float(cluster_dists.mean()),
            suggested_op=suggested_op,
            suggested_params=suggested_params,
            nearest_program_op=nearest_op,
            action=action,
        ))

    return gaps


def _classify_gap(points, center, radius, target_v, target_f):
    """Classify what geometric feature a gap region needs."""
    pts = np.asarray(points, dtype=np.float64)
    centered = pts - center

    if len(pts) < 4:
        return "sphere", {"center": center.tolist(), "radius": float(radius)}

    # PCA to understand gap shape
    cov = centered.T @ centered / len(pts)
    eigvals, eigvecs = _gpu_eigh(cov)
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]

    spread = np.sqrt(np.maximum(eigvals, 0))
    if spread[0] < 1e-8:
        return "sphere", {"center": center.tolist(), "radius": float(radius)}

    # Elongation ratio
    elongation = spread[0] / max(spread[1], 1e-8)

    # Flatness ratio
    flatness = spread[1] / max(spread[2], 1e-8) if spread[2] > 1e-8 else 1.0

    # Radial check: are points roughly circular in cross-section?
    radii = np.linalg.norm(centered[:, :2], axis=1)
    circularity = 1.0 - float(radii.std() / max(radii.mean(), 1e-8))

    if elongation > 3.0:
        # Elongated: cylinder or sweep
        axis = eigvecs[:, order[0]]
        return "cylinder", {
            "center": center.tolist(),
            "axis": axis.tolist(),
            "radius": float(np.median(radii)),
            "height": float(spread[0] * 2),
        }

    if circularity > 0.8 and flatness < 2.0:
        # Roughly spherical
        return "sphere", {
            "center": center.tolist(),
            "radius": float(np.median(np.linalg.norm(centered, axis=1))),
        }

    if flatness > 3.0:
        # Flat: likely a box face or extrusion
        return "box", {
            "center": center.tolist(),
            "dimensions": (spread * 2).tolist(),
        }

    if circularity > 0.6:
        # Somewhat circular, ring-like: torus
        return "torus", {
            "center": center.tolist(),
            "major_r": float(radii.mean()),
            "minor_r": float(spread[2]),
        }

    # Default: sphere
    return "sphere", {
        "center": center.tolist(),
        "radius": float(radius),
    }


def _find_nearest_op(point, program):
    """Find the program operation whose output is nearest to point."""
    best_idx = -1
    best_dist = float('inf')

    meshes = []
    for i, op in enumerate(program.operations):
        if not op.enabled:
            continue
        result = _eval_op(op, meshes)
        if result is not None:
            v, f = result
            meshes.append((v, f))
            centroid = v.mean(axis=0)
            d = float(np.linalg.norm(centroid - point))
            if d < best_dist:
                best_dist = d
                best_idx = i

    return best_idx


# ---------------------------------------------------------------------------
# BLUE team: program mutations
# ---------------------------------------------------------------------------

def add_operation(program, gap):
    """Add a new operation to the program based on the gap analysis."""
    op = CadOp(
        op_type=gap.suggested_op,
        params=copy.deepcopy(gap.suggested_params),
    )
    program.operations.append(op)
    program.invalidate_cache()


def refine_operation(program, op_index, target_v, target_f, max_iter=30):
    """Refine an operation's numeric parameters to reduce distance.

    Uses multi-scale coordinate descent: starts with large perturbations
    (±50%) to escape bad initial fits (e.g. wrong radius), then narrows
    to fine adjustments.  Runs multiple passes for max_iter iterations.
    """
    if op_index < 0 or op_index >= len(program.operations):
        return

    op = program.operations[op_index]
    original_params = copy.deepcopy(op.params)
    best_cost = program.total_cost(target_v, target_f)

    # Multi-scale: coarse → fine delta fractions
    scalar_deltas = [0.5, -0.5, 0.3, -0.3, 0.15, -0.15, 0.05, -0.05]
    vector_deltas = [1.0, -1.0, 0.5, -0.5, 0.2, -0.2, 0.05, -0.05]

    # Skip discrete parameters that shouldn't be continuously refined
    skip_keys = {"divs", "h_divs", "subdivs", "cross_divs"}

    for iteration in range(max_iter):
        improved_any = False

        for key, val in list(original_params.items()):
            if key in skip_keys:
                continue
            if isinstance(val, (int, float)):
                for delta_frac in scalar_deltas:
                    trial = val * (1.0 + delta_frac)
                    if isinstance(val, int):
                        trial = int(round(trial))
                    op.params[key] = trial
                    program.invalidate_cache()
                    cost = program.total_cost(target_v, target_f)
                    if cost < best_cost:
                        best_cost = cost
                        original_params[key] = trial
                        val = trial  # update for next delta
                        improved_any = True
                    else:
                        op.params[key] = original_params[key]
                        program.invalidate_cache()

            elif isinstance(val, list) and all(
                    isinstance(x, (int, float)) for x in val):
                for i in range(len(val)):
                    for delta in vector_deltas:
                        trial_val = list(original_params[key])
                        scale = abs(trial_val[i]) if abs(trial_val[i]) > 0.01 else 1.0
                        trial_val[i] += delta * scale
                        op.params[key] = trial_val
                        program.invalidate_cache()
                        cost = program.total_cost(target_v, target_f)
                        if cost < best_cost:
                            best_cost = cost
                            original_params[key] = list(trial_val)
                            improved_any = True
                        else:
                            op.params[key] = list(original_params[key])
                            program.invalidate_cache()

        if not improved_any:
            break


def refine_operation_diff(program, op_index, target_v, target_f, max_iter=50):
    """Refine an operation's parameters using gradient-based optimization.

    Uses automatic differentiation (via PyTorch) or finite-difference
    gradients to minimize the total cost w.r.t. continuous parameters.
    This is generally more efficient than coordinate descent for operations
    with many coupled parameters (e.g., sweep paths, revolve profiles).

    Falls back to the standard coordinate-descent refine_operation() if
    the optim module is unavailable.

    Args:
        program: CadProgram instance
        op_index: index of the operation to refine
        target_v: (N, 3) target mesh vertices
        target_f: (M, 3) target mesh faces
        max_iter: maximum gradient descent iterations

    Returns:
        float: final total cost
    """
    try:
        from .optim import DifferentiableRefiner
        refiner = DifferentiableRefiner(max_iter=max_iter)
        return refiner.refine(program, op_index, target_v, target_f)
    except Exception:
        # Fallback to coordinate descent
        refine_operation(program, op_index, target_v, target_f,
                         max_iter=max_iter)
        return program.total_cost(target_v, target_f)


def remove_operation(program, op_index):
    """Disable an operation and check if total cost improves."""
    if op_index < 0 or op_index >= len(program.operations):
        return
    program.operations[op_index].enabled = False
    program.invalidate_cache()


def simplify_program(program, target_v, target_f):
    """Remove operations whose removal improves or barely worsens total cost.

    Also removes disabled operations from the list.
    """
    # First: physically remove disabled ops
    program.operations = [op for op in program.operations if op.enabled]
    program.invalidate_cache()

    if len(program.operations) <= 1:
        return

    baseline = program.total_cost(target_v, target_f)

    # Try removing each op (from least complex to most)
    ops_by_cost = sorted(range(len(program.operations)),
                         key=lambda i: program.operations[i].complexity_cost)

    removed = set()
    for idx in ops_by_cost:
        if idx in removed:
            continue
        op = program.operations[idx]
        op.enabled = False
        program.invalidate_cache()
        cost_without = program.total_cost(target_v, target_f)

        # Allow removal if cost increases by less than the elegance savings
        elegance_saved = (ALPHA * op.complexity_cost + BETA +
                          GAMMA * op.param_count)
        if cost_without <= baseline + elegance_saved * 0.5:
            removed.add(idx)
            baseline = cost_without
        else:
            op.enabled = True
            program.invalidate_cache()

    # Clean up
    program.operations = [op for i, op in enumerate(program.operations)
                          if i not in removed]
    program.invalidate_cache()


# ---------------------------------------------------------------------------
# Initial program construction
# ---------------------------------------------------------------------------

def _make_candidate_op(shape, target_v):
    """Build a CadOp for a given shape type fitted to the target vertices."""
    if shape == "sphere":
        params = fit_sphere(target_v)
        return CadOp("sphere", {
            "center": params["center"].tolist(),
            "radius": params["radius"],
        })
    elif shape == "cylinder":
        params = fit_cylinder(target_v)
        op_params = {
            "center": params["center"].tolist(),
            "axis": params["axis"].tolist(),
            "radius": params["radius"],
            "height": params["height"],
        }
        # Use detected cross-section divs if available
        if "best_divs" in params:
            op_params["divs"] = params["best_divs"]
        # Estimate height subdivisions from target vertex density
        h_divs = _estimate_height_divs(target_v, params["axis"],
                                        params["center"], params["height"])
        if h_divs is not None:
            op_params["h_divs"] = h_divs
        return CadOp("cylinder", op_params)
    elif shape == "profiled_cylinder":
        params = fit_profiled_cylinder(target_v)
        # Only use profiled if actually tapered (taper_ratio > 5%)
        if params.get("taper_ratio", 0) < 0.05:
            return None
        op_params = {
            "center": params["center"].tolist(),
            "axis": params["axis"].tolist(),
            "height": params["height"],
            "radii": params["radii"],
            "heights": params["heights"],
        }
        return CadOp("profiled_cylinder", op_params)
    elif shape == "cone":
        params = fit_cone(target_v)
        # Center is the base of the cone, not the apex
        # The cone profile goes from z=0 (base_radius) to z=height (top_radius)
        axis = params["axis"]
        apex = params["apex"]
        h = params["height"]
        # Base is at apex - axis * (height * slope_direction)
        # Simpler: compute base from target vertices along the axis
        tv = np.asarray(target_v, dtype=np.float64)
        center = tv.mean(axis=0)
        proj = np.dot(tv - center, axis)
        base_pos = center + axis * float(proj.min())
        return CadOp("cone", {
            "center": base_pos.tolist(),
            "base_radius": params["base_radius"],
            "top_radius": max(params["top_radius"], 0.01),
            "height": params["height"],
        })
    elif shape == "auto_revolve":
        params = fit_revolve_profile(target_v)
        if params["residual"] > 100:
            return None
        # Store as radii/heights for coordinate descent refinement
        radii = [r for r, z in params["profile"]]
        heights = [z for r, z in params["profile"]]
        op_params = {
            "radii": radii,
            "heights": heights,
            "center": params["center"].tolist(),
            "axis": params["axis"].tolist(),
        }
        return CadOp("revolve", op_params)
    elif shape == "box":
        params = fit_box(target_v)
        return CadOp("box", {
            "center": params["center"].tolist(),
            "dimensions": params["dimensions"].tolist(),
        })
    return None


def _estimate_height_divs(vertices, axis, center, height):
    """Estimate number of height subdivisions from vertex height distribution.

    For low-poly meshes, the vertices are arranged at discrete height levels.
    We detect these levels and return the number of gaps between them.

    Returns:
        int or None: estimated h_divs, or None if can't determine
    """
    v = np.asarray(vertices, dtype=np.float64)
    axis = np.asarray(axis, dtype=np.float64)
    axis = axis / (np.linalg.norm(axis) + 1e-12)
    center = np.asarray(center, dtype=np.float64)

    # Project onto axis
    proj = np.dot(v - center, axis)

    if height < 1e-8:
        return None

    # Cluster unique height levels (relative to height)
    tol = height * 0.05  # 5% of height
    sorted_proj = np.sort(proj)
    levels = [sorted_proj[0]]
    for p in sorted_proj[1:]:
        if p - levels[-1] > tol:
            levels.append(p)

    n_levels = len(levels)
    if n_levels < 2 or n_levels > 20:
        return None

    # h_divs = number of gaps between levels
    h_divs = n_levels - 1
    return max(2, h_divs)


def initial_program(target_v, target_f):
    """Create a CadProgram that covers the target mesh.

    Uses adaptive complexity-based op budget and multiple strategies:
    1. Per-component fitting for multi-component meshes (trees, humanoids)
    2. Single best primitive (sphere, cylinder, revolve, box, cone)
    3. Axial segmentation (split along principal axis)
    4. Geometric segmentation (convexity/SDF/skeleton decomposition)

    Picks whichever strategy gives the highest accuracy.
    """
    from .elegance import score_accuracy

    target_v = np.asarray(target_v, dtype=np.float64)
    target_f = np.asarray(target_f)

    # Compute mesh complexity and adaptive op budget
    mc = mesh_complexity(target_v, target_f)
    budget = mc["op_budget"]

    # Decompose into connected components
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import connected_components

    n = len(target_v)
    rows = np.concatenate([target_f[:, 0], target_f[:, 1], target_f[:, 2]])
    cols = np.concatenate([target_f[:, 1], target_f[:, 2], target_f[:, 0]])
    adj = csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n))
    n_comp, labels = connected_components(adj, directed=False)

    # Gather component info, skip tiny components
    min_verts = max(4, int(n * 0.002))  # at least 0.2% of mesh
    comps = []
    for c in range(n_comp):
        idx = np.where(labels == c)[0]
        if len(idx) < min_verts:
            continue
        comps.append(idx)

    # Sort by size descending
    comps.sort(key=lambda x: len(x), reverse=True)

    # Strategy 1: For meshes with many components, fit per-component primitives
    if len(comps) >= 4:
        ops = []
        # Fit best primitive to each significant component using
        # Hausdorff-based accuracy (not fitting residual) as criterion
        # Cap at budget but with a performance ceiling for highly-branching
        # meshes where mutation evaluation becomes expensive.
        performance_cap = 35 if len(comps) > 50 else budget
        max_ops = min(len(comps), budget, performance_cap)
        for comp_idx in comps[:max_ops]:
            comp_v = target_v[comp_idx]
            best_op = None
            best_acc = -1.0
            for shape in ("profiled_cylinder", "cylinder", "box", "sphere"):
                try:
                    op = _make_candidate_op(shape, comp_v)
                    if op is None:
                        continue
                    # Evaluate against the component's own vertices
                    test_prog = CadProgram([op])
                    acc = score_accuracy(test_prog, comp_v,
                                         np.zeros((0, 3), dtype=np.int64))
                    if acc > best_acc:
                        best_acc = acc
                        best_op = op
                except Exception:
                    pass
            if best_op is not None:
                ops.append(best_op)

        if ops:
            prog = CadProgram(ops)
            acc = score_accuracy(prog, target_v, target_f)
            # Also try the single-primitive approach and pick the better one
            best_single = _best_single_primitive(target_v, target_f)
            if best_single is not None:
                single_acc = score_accuracy(best_single, target_v, target_f)
                if single_acc > acc:
                    return best_single
            return prog

    # Strategy 2: For few-component meshes, try single primitives and
    # also axial segmentation (split along principal axis into 2-4 slices,
    # fit a cylinder per slice — good for tapered/varied-radius shapes)
    best_single = _best_single_primitive(target_v, target_f)
    best_axial = _axial_segmented_program(target_v, target_f)
    best_geom = _segmented_program(target_v, target_f, budget=budget)

    # Refine single-primitive revolve/profiled_cylinder before comparing
    if best_single is not None:
        for si, sop in enumerate(best_single.operations):
            if sop.enabled and sop.op_type in ("revolve", "profiled_cylinder"):
                refine_operation(best_single, si, target_v, target_f,
                                 max_iter=15)
                break

    from .elegance import score_accuracy
    candidates = []
    if best_single is not None:
        candidates.append((score_accuracy(best_single, target_v, target_f),
                           best_single))
    if best_axial is not None:
        candidates.append((score_accuracy(best_axial, target_v, target_f),
                           best_axial))
    if best_geom is not None:
        candidates.append((score_accuracy(best_geom, target_v, target_f),
                           best_geom))

    if not candidates:
        return CadProgram([_make_candidate_op("sphere", target_v)])

    candidates.sort(key=lambda x: x[0], reverse=True)
    return candidates[0][1]


def _axial_segmented_program(target_v, target_f, n_segments=None):
    """Split the mesh along its principal axis into segments, fit each.

    Useful for tapered shapes (bottles, rockets) where a single cylinder
    can't capture varying radius.  Tries multiple segment counts (3-8)
    and returns the best.
    """
    from .elegance import score_accuracy

    v = np.asarray(target_v, dtype=np.float64)
    center = v.mean(axis=0)
    centered = v - center

    # PCA for principal axis
    cov = centered.T @ centered / len(v)
    eigvals, eigvecs = _gpu_eigh(cov)
    order = np.argsort(eigvals)[::-1]
    axis = eigvecs[:, order[0]]
    if axis[2] < 0:
        axis = -axis

    # Project onto axis
    proj = centered @ axis
    p_min, p_max = proj.min(), proj.max()
    span = p_max - p_min

    if span < 1e-8:
        return None

    if n_segments is not None:
        seg_counts = [n_segments]
    else:
        seg_counts = [3, 4, 5, 6, 8, 10, 12, 15, 20]

    best_prog = None
    best_acc = -1.0

    for n_seg in seg_counts:
        ops = []
        boundaries = np.linspace(p_min, p_max, n_seg + 1)
        prog = _build_segmented_ops(v, proj, boundaries, span)
        if prog is not None and prog.n_enabled() >= 2:
            acc = score_accuracy(prog, target_v, target_f)
            if acc > best_acc:
                best_acc = acc
                best_prog = prog

    return best_prog


def _build_segmented_ops(v, proj, boundaries, span):
    """Build ops from axial segmentation boundaries.

    Uses Hausdorff-based accuracy (not fitting residual) to pick
    the best primitive per segment, same as the component-aware strategy.
    """
    from .elegance import score_accuracy

    ops = []
    n_segments = len(boundaries) - 1
    for seg_i in range(n_segments):
        lo, hi = boundaries[seg_i], boundaries[seg_i + 1]
        # Expand slightly to avoid gaps
        lo_exp = lo - span * 0.02
        hi_exp = hi + span * 0.02
        mask = (proj >= lo_exp) & (proj <= hi_exp)
        seg_v = v[mask]
        if len(seg_v) < 4:
            continue

        # Try each primitive and pick best by accuracy against segment
        best_op = None
        best_acc = -1.0
        empty_f = np.zeros((0, 3), dtype=np.int64)

        for shape in ("profiled_cylinder", "cylinder", "box", "cone"):
            try:
                op = _make_candidate_op(shape, seg_v)
                if op is None:
                    continue
                test_prog = CadProgram([op])
                acc = score_accuracy(test_prog, seg_v, empty_f)
                if acc > best_acc:
                    best_acc = acc
                    best_op = op
            except Exception:
                pass

        if best_op is not None:
            ops.append(best_op)

    if len(ops) < 2:
        return None
    return CadProgram(ops)


def _segmented_program(target_v, target_f, budget=20):
    """Build a CadProgram by geometrically segmenting the mesh.

    Unlike connected-component decomposition (which only works for
    disconnected parts) or axial segmentation (which only works for
    elongated shapes), geometric segmentation decomposes a single
    connected mesh into regions that are each well-described by a
    single CAD primitive.

    For example, a chair becomes: 4 leg-cylinders + 1 seat-box + 1 back,
    each fitted independently for high accuracy.
    """
    from .elegance import score_accuracy
    from .segmentation import segment_mesh

    try:
        segments = segment_mesh(target_v, target_f, strategy="auto")
    except Exception:
        return None

    if not segments or len(segments) < 2:
        return None

    # Sort by vertex count descending (largest segments first)
    segments.sort(key=lambda s: len(s.vertices), reverse=True)

    ops = []
    empty_f = np.zeros((0, 3), dtype=np.int64)

    for seg in segments[:budget]:
        seg_v = seg.vertices
        if len(seg_v) < 4:
            continue

        # Skip fillet/blend segments — these are transition surfaces
        # between primitives that mask the underlying sharp intersections.
        # Including fillet geometry in primitive fitting distorts the fit.
        # Fillets can be covered later by gap-fill ops if needed.
        if getattr(seg, 'is_fillet', False):
            continue

        # Choose fitting strategy based on segment classification
        if seg.cad_action == "revolve":
            # Segment is rotationally symmetric — try auto_revolve first
            shapes = ("auto_revolve", "profiled_cylinder", "cylinder", "sphere")
        elif seg.cad_action == "extrude":
            shapes = ("box", "cylinder", "profiled_cylinder")
        elif seg.cad_action == "sweep":
            shapes = ("profiled_cylinder", "cylinder", "auto_revolve")
        else:
            # loft / freeform — try everything
            shapes = ("cylinder", "box", "sphere", "profiled_cylinder")

        # Skip expensive fits for tiny segments
        if len(seg_v) < 30:
            shapes = tuple(s for s in shapes
                           if s not in ("auto_revolve", "profiled_cylinder"))
            if not shapes:
                shapes = ("cylinder", "sphere")

        best_op = None
        best_acc = -1.0

        for shape in shapes:
            try:
                op = _make_candidate_op(shape, seg_v)
                if op is None:
                    continue
                test_prog = CadProgram([op])
                acc = score_accuracy(test_prog, seg_v, empty_f)
                if acc > best_acc:
                    best_acc = acc
                    best_op = op
            except Exception:
                pass

        if best_op is not None:
            # Refine revolve/profiled_cylinder ops for better initial fit
            if best_op.op_type in ("revolve", "profiled_cylinder") and len(seg_v) >= 20:
                try:
                    test_prog = CadProgram([best_op])
                    refine_operation(test_prog, 0, seg_v, empty_f, max_iter=5)
                    best_op = test_prog.operations[0]
                except Exception:
                    pass
            ops.append(best_op)

    if len(ops) < 2:
        return None
    return CadProgram(ops)


def _best_single_primitive(target_v, target_f):
    """Try all four primitive types, return program with best accuracy."""
    from .elegance import score_accuracy

    candidates = []
    for shape in ("sphere", "cylinder", "profiled_cylinder", "auto_revolve", "cone", "box"):
        try:
            op = _make_candidate_op(shape, target_v)
            if op is not None:
                prog = CadProgram([op])
                acc = score_accuracy(prog, target_v, target_f)
                candidates.append((acc, prog))
        except Exception:
            pass
    if not candidates:
        return None
    candidates.sort(key=lambda x: x[0], reverse=True)
    return candidates[0][1]


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def run_cad_program_loop(target_v, target_f,
                          max_rounds=30,
                          max_ops=12,
                          distance_threshold=0.01):
    """Evolve a CAD program toward accuracy and elegance.

    Args:
        target_v: (N, 3) target mesh vertices
        target_f: (M, 3) target mesh faces
        max_rounds: maximum evolution rounds
        max_ops:    maximum allowed operations
        distance_threshold: stop when normalized distance drops below this

    Returns:
        dict with:
            program      — final CadProgram
            history      — list of per-round dicts
            cad_vertices — final mesh vertices
            cad_faces    — final mesh faces
    """
    target_v = np.asarray(target_v, dtype=np.float64)
    target_f = np.asarray(target_f)

    program = initial_program(target_v, target_f)
    best_cost = program.total_cost(target_v, target_f)
    history = []

    for round_num in range(max_rounds):
        # RED: find gaps
        gaps = find_program_gaps(program, target_v, target_f)

        if not gaps:
            history.append(_round_record(round_num, program, best_cost,
                                          "converged", []))
            break

        if gaps[0].residual_score < distance_threshold:
            history.append(_round_record(round_num, program, best_cost,
                                          "threshold_reached", []))
            break

        # BLUE: generate candidates
        candidates = []

        # 1. Try adding operations for top gaps
        for gap in gaps[:3]:
            if gap.action == "add" and program.n_enabled() < max_ops:
                p = program.copy()
                add_operation(p, gap)
                candidates.append(("add:" + gap.suggested_op, p))

        # 2. Try refining existing operations
        for gap in gaps[:2]:
            if gap.nearest_program_op >= 0:
                p = program.copy()
                refine_operation(p, gap.nearest_program_op, target_v, target_f)
                candidates.append(("refine:" + str(gap.nearest_program_op), p))

        # 3. Try removing each operation
        for i in range(len(program.operations)):
            if program.operations[i].enabled:
                p = program.copy()
                remove_operation(p, i)
                candidates.append(("remove:" + str(i), p))

        # 4. Try simplification
        p = program.copy()
        simplify_program(p, target_v, target_f)
        candidates.append(("simplify", p))

        # Pick best candidate
        if not candidates:
            history.append(_round_record(round_num, program, best_cost,
                                          "no_candidates", []))
            break

        best_candidate = None
        best_candidate_cost = best_cost
        best_action = "held"

        for action_name, cand in candidates:
            cost = cand.total_cost(target_v, target_f)
            if cost < best_candidate_cost:
                best_candidate_cost = cost
                best_candidate = cand
                best_action = action_name

        if best_candidate is not None:
            program = best_candidate
            best_cost = best_candidate_cost

        history.append(_round_record(round_num, program, best_cost,
                                      best_action,
                                      [(g.suggested_op, g.residual_score)
                                       for g in gaps[:3]]))

    # Final simplification pass
    simplify_program(program, target_v, target_f)
    best_cost = program.total_cost(target_v, target_f)

    cad_v, cad_f = program.evaluate()
    return {
        "program": program,
        "history": history,
        "cad_vertices": cad_v,
        "cad_faces": cad_f,
        "total_cost": best_cost,
        "n_ops": program.n_enabled(),
        "complexity": program.total_complexity(),
    }


def _round_record(round_num, program, cost, action, gaps):
    return {
        "round": round_num,
        "n_ops": program.n_enabled(),
        "complexity": round(program.total_complexity(), 2),
        "total_cost": round(cost, 6),
        "action": action,
        "program": program.summary(),
        "top_gaps": [(op, round(s, 4)) for op, s in gaps],
    }
