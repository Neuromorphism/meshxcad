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

from .general_align import hausdorff_distance, surface_distance_map
from .reconstruct import (
    classify_mesh, fit_sphere, fit_cylinder, fit_cone, fit_box,
    _rotation_between,
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
    "cone": 1.5,
    "box": 1.0,
    "torus": 1.5,
    "revolve": 2.0,
    "extrude": 1.5,
    "sweep": 3.0,
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

def _eval_op(op, existing_meshes):
    """Evaluate a single CadOp, return (vertices, faces) or None."""
    p = op.params
    t = op.op_type

    if t == "sphere":
        v, f = make_sphere_mesh(
            radius=p.get("radius", 1.0),
            lat_divs=p.get("divs", 20),
            lon_divs=p.get("divs", 20))
        center = np.asarray(p.get("center", [0, 0, 0]), dtype=np.float64)
        return v + center, f

    if t == "cylinder":
        v, f = make_cylinder_mesh(
            radius=p.get("radius", 1.0),
            height=p.get("height", 1.0),
            radial_divs=p.get("divs", 24),
            height_divs=p.get("h_divs", 10))
        axis = np.asarray(p.get("axis", [0, 0, 1]), dtype=np.float64)
        center = np.asarray(p.get("center", [0, 0, 0]), dtype=np.float64)
        # Align to axis
        z = np.array([0.0, 0.0, 1.0])
        axis_n = axis / (np.linalg.norm(axis) + 1e-12)
        R = _rotation_between(z, axis_n)
        v = v @ R.T + center
        return v, f

    if t == "cone":
        h = p.get("height", 1.0)
        base_r = p.get("base_radius", 1.0)
        top_r = p.get("top_radius", 0.1)
        n_h = p.get("h_divs", 10)
        profile = [(base_r + (top_r - base_r) * i / n_h,
                     i / n_h * h) for i in range(n_h + 1)]
        profile = [(max(r, 0.01), z) for r, z in profile]
        v, f = revolve_profile(profile, p.get("divs", 24))
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
        profile = p.get("profile", [(1, 0), (1, 1)])
        v, f = revolve_profile(profile, p.get("divs", 24))
        center = np.asarray(p.get("center", [0, 0, 0]), dtype=np.float64)
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
    tree_cad = KDTree(cad_v)
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
    eigvals, eigvecs = np.linalg.eigh(cov)
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

    Uses coordinate-descent on each parameter.
    """
    if op_index < 0 or op_index >= len(program.operations):
        return

    op = program.operations[op_index]
    original_params = copy.deepcopy(op.params)
    best_cost = program.total_cost(target_v, target_f)

    # Try perturbing each numeric parameter
    for key, val in list(original_params.items()):
        if isinstance(val, (int, float)):
            for delta_frac in [0.1, -0.1, 0.05, -0.05, 0.2, -0.2]:
                trial = val * (1.0 + delta_frac)
                if isinstance(val, int):
                    trial = int(round(trial))
                op.params[key] = trial
                program.invalidate_cache()
                cost = program.total_cost(target_v, target_f)
                if cost < best_cost:
                    best_cost = cost
                    original_params[key] = trial
                else:
                    op.params[key] = original_params[key]
                    program.invalidate_cache()

        elif isinstance(val, list) and all(isinstance(x, (int, float)) for x in val):
            for i in range(len(val)):
                for delta in [0.1, -0.1, 0.3, -0.3]:
                    trial_val = list(original_params[key])
                    trial_val[i] += delta * abs(trial_val[i]) if abs(trial_val[i]) > 0.01 else delta
                    op.params[key] = trial_val
                    program.invalidate_cache()
                    cost = program.total_cost(target_v, target_f)
                    if cost < best_cost:
                        best_cost = cost
                        original_params[key] = list(trial_val)
                    else:
                        op.params[key] = list(original_params[key])
                        program.invalidate_cache()


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

def initial_program(target_v, target_f):
    """Create a 1-operation CadProgram from mesh classification."""
    target_v = np.asarray(target_v, dtype=np.float64)
    target_f = np.asarray(target_f)

    classification = classify_mesh(target_v, target_f)
    shape = classification["shape_type"]

    if shape == "sphere":
        params = fit_sphere(target_v)
        op = CadOp("sphere", {
            "center": params["center"].tolist(),
            "radius": params["radius"],
        })

    elif shape == "cylinder":
        params = fit_cylinder(target_v)
        op = CadOp("cylinder", {
            "center": params["center"].tolist(),
            "axis": params["axis"].tolist(),
            "radius": params["radius"],
            "height": params["height"],
        })

    elif shape == "cone":
        params = fit_cone(target_v)
        op = CadOp("cone", {
            "center": params["apex"].tolist(),
            "base_radius": params["base_radius"],
            "top_radius": max(params["top_radius"], 0.01),
            "height": params["height"],
        })

    elif shape == "box":
        params = fit_box(target_v)
        op = CadOp("box", {
            "center": params["center"].tolist(),
            "dimensions": params["dimensions"].tolist(),
        })

    elif shape in ("revolve", "extrude", "sweep", "freeform"):
        # Start with the best-fit primitive and let the loop refine
        sphere = fit_sphere(target_v)
        cyl = fit_cylinder(target_v)
        # Pick whichever has lower residual
        if cyl["residual"] < sphere["residual"]:
            op = CadOp("cylinder", {
                "center": cyl["center"].tolist(),
                "axis": cyl["axis"].tolist(),
                "radius": cyl["radius"],
                "height": cyl["height"],
            })
        else:
            op = CadOp("sphere", {
                "center": sphere["center"].tolist(),
                "radius": sphere["radius"],
            })
    else:
        sphere = fit_sphere(target_v)
        op = CadOp("sphere", {
            "center": sphere["center"].tolist(),
            "radius": sphere["radius"],
        })

    return CadProgram([op])


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
