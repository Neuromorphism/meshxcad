"""Diffusion-based strategy selection for CAD program synthesis.

Applies the principles of denoising diffusion models to the problem of
finding a CAD action sequence that reconstructs a target mesh.

Analogy to image diffusion:

    Stable Diffusion                  CAD Diffusion
    ─────────────────                 ──────────────
    Pixel space (H×W×3)              Program space (N_ops × D_params)
    Forward: image → noise           Forward: valid program → random ops
    Reverse: noise → image           Reverse: random ops → valid program
    U-Net predicts noise ε           Score function predicts action noise
    Text conditioning (CLIP)         Mesh conditioning (geometric features)
    Classifier-free guidance         Elegance-guided denoising
    Timestep schedule T→0            Refinement schedule (coarse→fine)

Just as most pixel configurations are noise and only a tiny manifold
represents real images, most CAD operation sequences are nonsensical —
only a tiny manifold represents programs that produce coherent geometry
matching a target.

The key insight: we don't need a trained neural network.  The existing
cost function (accuracy × elegance) serves as our *score function* —
the gradient of the log-probability under the data distribution.  We
perform Langevin-dynamics-style sampling guided by this score, with a
noise schedule that moves from exploration (high noise, random ops) to
exploitation (low noise, parameter refinement).

Architecture:
    1. MeshConditioner: extracts a geometric feature vector from the
       target mesh (analogous to CLIP text embeddings).
    2. ProgramEmbedding: maps CadProgram ↔ continuous vector space.
    3. NoiseSchedule: cosine/linear schedule controlling noise variance
       at each diffusion timestep.
    4. DiffusionStrategySelector: the main reverse-process loop that
       starts from a noisy program and iteratively denoises it toward
       a valid reconstruction.
"""

import math
import copy
import numpy as np
from dataclasses import dataclass, field
from typing import Optional

from .cad_program import (
    CadOp, CadProgram, OP_COSTS, mesh_complexity,
    find_program_gaps, add_operation, refine_operation,
    remove_operation, simplify_program, initial_program,
    _eval_op,
)
from .general_align import hausdorff_distance
from .objects.builder import combine_meshes
from .gpu import AcceleratedKDTree as _AKDTree, eigh as _gpu_eigh


# ============================================================================
# Operation vocabulary for embedding
# ============================================================================

# Canonical ordering of op types for embedding indices
OP_TYPES = list(OP_COSTS.keys())
OP_TYPE_TO_IDX = {op: i for i, op in enumerate(OP_TYPES)}
N_OP_TYPES = len(OP_TYPES)

# Maximum parameters per operation (padded)
MAX_PARAMS_PER_OP = 16

# Maximum operations in a program embedding
MAX_OPS = 20

# Embedding dimension per operation: one-hot type + params + enabled flag
OP_EMBED_DIM = N_OP_TYPES + MAX_PARAMS_PER_OP + 1

# Full program embedding dimension
PROGRAM_EMBED_DIM = MAX_OPS * OP_EMBED_DIM


# ============================================================================
# Mesh conditioning — the "CLIP embedding" analog
# ============================================================================

# Number of features in the mesh conditioning vector
MESH_CONDITION_DIM = 32


def extract_mesh_features(vertices, faces):
    """Extract a fixed-length geometric feature vector from a mesh.

    This is analogous to the CLIP text embedding in stable diffusion:
    it conditions the denoising process on *what we want to produce*.

    Features capture the geometric "fingerprint" of the target:
        - Bounding box dimensions and aspect ratios
        - PCA eigenvalue spectrum (shape elongation/flatness)
        - Radial distribution statistics (circularity)
        - Curvature statistics
        - Component count and vertex density
        - Symmetry indicators

    Returns:
        np.ndarray of shape (MESH_CONDITION_DIM,), normalized to [0, 1].
    """
    v = np.asarray(vertices, dtype=np.float64)
    f = np.asarray(faces)

    features = np.zeros(MESH_CONDITION_DIM)

    if len(v) < 4:
        return features

    # --- Bounding box features (6 dims) ---
    bbox_min = v.min(axis=0)
    bbox_max = v.max(axis=0)
    bbox_size = bbox_max - bbox_min
    bbox_diag = float(np.linalg.norm(bbox_size))
    if bbox_diag < 1e-12:
        return features

    # Normalized dimensions (sorted descending for rotation invariance)
    sorted_dims = np.sort(bbox_size)[::-1] / bbox_diag
    features[0:3] = sorted_dims

    # Aspect ratios
    features[3] = sorted_dims[1] / max(sorted_dims[0], 1e-12)  # medium/long
    features[4] = sorted_dims[2] / max(sorted_dims[0], 1e-12)  # short/long
    features[5] = sorted_dims[2] / max(sorted_dims[1], 1e-12)  # short/medium

    # --- PCA eigenvalue spectrum (4 dims) ---
    center = v.mean(axis=0)
    centered = v - center
    cov = centered.T @ centered / len(v)
    eigvals, eigvecs = _gpu_eigh(cov)
    eigvals = np.sort(np.maximum(eigvals, 0))[::-1]
    spread = np.sqrt(eigvals)
    total_spread = spread.sum()
    if total_spread > 1e-12:
        features[6:9] = spread / total_spread  # normalized eigenvalues
    # Elongation ratio normalized to [0,1] via sigmoid-like mapping
    raw_elongation = spread[0] / max(spread[2], 1e-12)
    features[9] = 1.0 - 1.0 / (1.0 + raw_elongation * 0.2)  # maps [0,∞) → [0,1)

    # --- Radial distribution (4 dims) ---
    radii = np.linalg.norm(centered, axis=1)
    if radii.max() > 1e-12:
        norm_radii = radii / radii.max()
        features[10] = float(np.mean(norm_radii))
        features[11] = float(np.std(norm_radii))
        features[12] = float(np.median(norm_radii))
        # Circularity: how constant is the radius?
        features[13] = 1.0 - min(float(norm_radii.std() / max(norm_radii.mean(), 1e-12)), 1.0)

    # --- Cross-sectional circularity along principal axis (3 dims) ---
    proj = centered @ eigvecs[:, np.argsort(eigvals)[::-1][0]]
    n_slices = 5
    slice_bounds = np.linspace(proj.min(), proj.max(), n_slices + 1)
    circularities = []
    for s in range(n_slices):
        mask = (proj >= slice_bounds[s]) & (proj < slice_bounds[s + 1])
        if mask.sum() < 4:
            continue
        slice_pts = centered[mask]
        # Project onto the plane perpendicular to principal axis
        local_2d = slice_pts @ eigvecs[:, np.argsort(eigvals)[::-1][1:3]]
        r = np.linalg.norm(local_2d, axis=1)
        if r.mean() > 1e-12:
            circularities.append(1.0 - float(r.std() / r.mean()))
    if circularities:
        features[14] = float(np.mean(circularities))
        features[15] = float(np.min(circularities))
        features[16] = float(np.std(circularities))

    # --- Curvature statistics (4 dims) ---
    if len(f) > 0:
        # Approximate discrete curvature via face normal variation
        # (lightweight: sample up to 2000 faces)
        sample_n = min(len(f), 2000)
        rng = np.random.RandomState(0)
        sample_idx = rng.choice(len(f), sample_n, replace=False) if len(f) > sample_n else np.arange(len(f))
        sample_faces = f[sample_idx]

        # Face normals
        v0 = v[sample_faces[:, 0]]
        v1 = v[sample_faces[:, 1]]
        v2 = v[sample_faces[:, 2]]
        normals = np.cross(v1 - v0, v2 - v0)
        norms = np.linalg.norm(normals, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-12)
        normals = normals / norms

        # Normal direction entropy (binned into octants)
        octant = (normals > 0).astype(int)
        octant_idx = octant[:, 0] * 4 + octant[:, 1] * 2 + octant[:, 2]
        hist = np.bincount(octant_idx, minlength=8).astype(float)
        hist = hist / max(hist.sum(), 1)
        hist = hist[hist > 0]
        entropy = -float(np.sum(hist * np.log(hist + 1e-12)))
        features[17] = entropy / np.log(8)  # normalized to [0, 1]

        # Face area statistics
        areas = norms.flatten() * 0.5
        if areas.sum() > 1e-12:
            norm_areas = areas / areas.sum()
            area_entropy = -float(np.sum(norm_areas * np.log(norm_areas + 1e-12)))
            features[18] = min(area_entropy / np.log(max(len(areas), 2)), 1.0)
        features[19] = float(areas.std() / max(areas.mean(), 1e-12))

        # Mean face normal deviation from nearest axis
        axis_dots = np.abs(normals)  # dot with x,y,z axes
        max_axis_alignment = axis_dots.max(axis=1)
        features[20] = float(np.mean(max_axis_alignment))

    # --- Topology features (4 dims) ---
    mc = mesh_complexity(v, f)
    features[21] = min(mc["complexity"], 1.0)
    features[22] = min(mc["n_components"] / 10.0, 1.0)
    features[23] = min(len(v) / 10000.0, 1.0)  # vertex density proxy
    features[24] = min(mc.get("curvature_entropy", 0) / 3.0, 1.0)

    # --- Symmetry indicators (4 dims) ---
    # Check reflection symmetry about each principal plane
    for ax_i in range(3):
        reflected = centered.copy()
        reflected[:, ax_i] = -reflected[:, ax_i]
        tree = _AKDTree(centered)
        dists, _ = tree.query(reflected)
        sym_score = 1.0 - min(float(np.mean(dists)) / bbox_diag, 1.0)
        features[25 + ax_i] = sym_score

    # Rotational symmetry indicator (check 90-degree rotation about principal axis)
    rot_axis = eigvecs[:, np.argsort(eigvals)[::-1][0]]
    angle = np.pi / 2
    c, s = np.cos(angle), np.sin(angle)
    K = np.array([[0, -rot_axis[2], rot_axis[1]],
                  [rot_axis[2], 0, -rot_axis[0]],
                  [-rot_axis[1], rot_axis[0], 0]])
    R = np.eye(3) + s * K + (1 - c) * (K @ K)
    rotated = (R @ centered.T).T
    tree = _AKDTree(centered)
    dists, _ = tree.query(rotated)
    features[28] = 1.0 - min(float(np.mean(dists)) / bbox_diag, 1.0)

    # --- Volume/surface-area ratio (2 dims) ---
    if len(f) > 0:
        v0 = v[f[:, 0]]
        v1 = v[f[:, 1]]
        v2 = v[f[:, 2]]
        cross = np.cross(v1 - v0, v2 - v0)
        sa = float(np.sum(np.linalg.norm(cross, axis=1)) * 0.5)
        # Signed volume via divergence theorem
        vol = abs(float(np.sum(v0 * cross) / 6.0))
        if sa > 1e-12:
            # Isoperimetric ratio: sphere = 1.0, complex = lower
            iso_ratio = (36 * np.pi * vol**2) / (sa**3)
            features[29] = min(iso_ratio ** (1/3), 1.0)
        if bbox_diag > 1e-12:
            features[30] = min(sa / (bbox_diag**2), 1.0)

    # --- Op-type affinity hints (1 dim) ---
    # How "revolvable" is this mesh? (high circularity + elongation)
    # Revolvability: high circularity * elongation (both already in [0,1])
    features[31] = features[14] * features[9]

    return np.clip(features, 0.0, 1.0)


# ============================================================================
# Program embedding — map CadProgram ↔ continuous vector
# ============================================================================

def embed_op(op):
    """Embed a single CadOp as a continuous vector.

    Returns np.ndarray of shape (OP_EMBED_DIM,).
    """
    vec = np.zeros(OP_EMBED_DIM)

    # One-hot operation type
    idx = OP_TYPE_TO_IDX.get(op.op_type, 0)
    vec[idx] = 1.0

    # Flatten parameters into fixed-length vector
    param_vec = _flatten_params(op.params)
    n = min(len(param_vec), MAX_PARAMS_PER_OP)
    vec[N_OP_TYPES:N_OP_TYPES + n] = param_vec[:n]

    # Enabled flag
    vec[-1] = 1.0 if op.enabled else 0.0

    return vec


def embed_program(program):
    """Embed a CadProgram as a continuous vector.

    Returns np.ndarray of shape (PROGRAM_EMBED_DIM,).
    """
    vec = np.zeros(PROGRAM_EMBED_DIM)
    for i, op in enumerate(program.operations[:MAX_OPS]):
        start = i * OP_EMBED_DIM
        vec[start:start + OP_EMBED_DIM] = embed_op(op)
    return vec


def unembed_program(vec, reference_program=None):
    """Reconstruct a CadProgram from a continuous vector.

    Because the mapping from continuous space back to discrete ops is
    lossy, we use the closest valid op type (argmax of one-hot) and
    reconstruct parameters from the continuous values.

    Args:
        vec: np.ndarray of shape (PROGRAM_EMBED_DIM,)
        reference_program: optional CadProgram whose parameter names
            guide reconstruction.

    Returns:
        CadProgram
    """
    ops = []
    for i in range(MAX_OPS):
        start = i * OP_EMBED_DIM
        op_vec = vec[start:start + OP_EMBED_DIM]

        # Check if this slot is populated (any type has weight > 0.1)
        type_weights = op_vec[:N_OP_TYPES]
        if type_weights.max() < 0.1:
            continue

        # Decode operation type (argmax)
        op_idx = int(np.argmax(type_weights))
        op_type = OP_TYPES[op_idx]

        # Decode enabled flag
        enabled = bool(op_vec[-1] > 0.5)

        # Decode parameters
        param_vec = op_vec[N_OP_TYPES:N_OP_TYPES + MAX_PARAMS_PER_OP]

        # If we have a reference program with the same op at this position,
        # use its parameter names/structure
        if (reference_program is not None
                and i < len(reference_program.operations)
                and reference_program.operations[i].op_type == op_type):
            params = _unflatten_params(
                param_vec, reference_program.operations[i].params)
        else:
            params = _unflatten_params_generic(param_vec, op_type)

        ops.append(CadOp(op_type, params, enabled))

    return CadProgram(ops) if ops else CadProgram()


def _flatten_params(params):
    """Flatten a parameter dict into a 1-D array."""
    values = []
    for k in sorted(params.keys()):
        v = params[k]
        if isinstance(v, np.ndarray):
            values.extend(v.flatten().tolist())
        elif isinstance(v, (list, tuple)):
            for item in v:
                if isinstance(item, (list, tuple, np.ndarray)):
                    values.extend(np.asarray(item).flatten().tolist())
                else:
                    values.append(float(item))
        elif isinstance(v, (int, float)):
            values.append(float(v))
    return np.array(values, dtype=np.float64)


def _unflatten_params(param_vec, reference_params):
    """Reconstruct params dict using reference structure."""
    params = {}
    idx = 0
    for k in sorted(reference_params.keys()):
        ref_v = reference_params[k]
        if isinstance(ref_v, np.ndarray):
            n = ref_v.size
            if idx + n <= len(param_vec):
                params[k] = param_vec[idx:idx + n].reshape(ref_v.shape)
            else:
                params[k] = ref_v.copy()
            idx += n
        elif isinstance(ref_v, (list, tuple)):
            if all(isinstance(item, (int, float)) for item in ref_v):
                n = len(ref_v)
                if idx + n <= len(param_vec):
                    params[k] = list(param_vec[idx:idx + n])
                else:
                    params[k] = list(ref_v)
                idx += n
            else:
                # Nested list (e.g., profile points)
                flat = []
                for item in ref_v:
                    if isinstance(item, (list, tuple)):
                        flat.extend(item)
                    else:
                        flat.append(item)
                n = len(flat)
                if idx + n <= len(param_vec):
                    reconstructed = []
                    fi = idx
                    for item in ref_v:
                        if isinstance(item, (list, tuple)):
                            l = len(item)
                            reconstructed.append(list(param_vec[fi:fi + l]))
                            fi += l
                        else:
                            reconstructed.append(float(param_vec[fi]))
                            fi += 1
                    params[k] = reconstructed
                else:
                    params[k] = copy.deepcopy(ref_v)
                idx += n
        elif isinstance(ref_v, (int, float)):
            if idx < len(param_vec):
                if isinstance(ref_v, int):
                    params[k] = int(round(param_vec[idx]))
                else:
                    params[k] = float(param_vec[idx])
            else:
                params[k] = ref_v
            idx += 1
        else:
            params[k] = copy.deepcopy(ref_v)
    return params


def _unflatten_params_generic(param_vec, op_type):
    """Reconstruct params from vector without a reference, using op-type defaults."""
    # Default parameter templates for each op type
    templates = {
        "sphere": {"center": np.zeros(3), "radius": 1.0},
        "cylinder": {"center": np.zeros(3), "axis": np.array([0, 0, 1.0]),
                      "radius": 1.0, "height": 2.0},
        "box": {"center": np.zeros(3), "dimensions": np.ones(3)},
        "cone": {"center": np.zeros(3), "axis": np.array([0, 0, 1.0]),
                 "radius_bottom": 1.0, "radius_top": 0.5, "height": 2.0},
        "torus": {"center": np.zeros(3), "major_r": 2.0, "minor_r": 0.5},
        "translate": {"offset": np.zeros(3)},
        "scale": {"center": np.zeros(3), "factors": np.ones(3)},
        "rotate": {"center": np.zeros(3), "axis": np.array([0, 0, 1.0]),
                   "angle_deg": 0.0},
        "mirror": {"point": np.zeros(3), "normal": np.array([1, 0, 0.0])},
        "revolve": {"profile_rz": [(1.0, 0.0), (1.0, 1.0)], "n_angular": 32},
        "extrude": {"polygon": [(0, 0), (1, 0), (0.5, 1.0)], "height": 1.0},
        "sweep": {"profile": [(0, 0), (1, 0), (0.5, 1.0)],
                  "path": [(0, 0, 0), (0, 0, 1)]},
        "union": {},
        "subtract_cylinder": {"center": np.zeros(3), "axis": np.array([0, 0, 1.0]),
                              "radius": 0.5, "height": 2.0},
        "fillet": {"radius": 0.2},
        "profiled_cylinder": {"center": np.zeros(3), "axis": np.array([0, 0, 1.0]),
                              "radii": [1.0, 1.0], "heights": [0.0, 1.0]},
    }

    template = templates.get(op_type, {"center": np.zeros(3), "radius": 1.0})
    return _unflatten_params(param_vec, template)


# ============================================================================
# Noise schedule — controls diffusion strength at each timestep
# ============================================================================

@dataclass
class NoiseSchedule:
    """Diffusion noise schedule.

    Supports linear, cosine, and sqrt schedules.  The schedule defines
    alpha_bar(t) — the cumulative signal-retention at timestep t:

        x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * ε

    At t=0: alpha_bar=1 (pure signal, no noise).
    At t=T: alpha_bar≈0 (pure noise).
    """
    num_timesteps: int = 50
    schedule_type: str = "cosine"  # "linear", "cosine", "sqrt"
    beta_start: float = 0.0001
    beta_end: float = 0.02

    def __post_init__(self):
        self._alpha_bars = self._compute_alpha_bars()

    def _compute_alpha_bars(self):
        T = self.num_timesteps
        if self.schedule_type == "linear":
            betas = np.linspace(self.beta_start, self.beta_end, T)
            alphas = 1.0 - betas
            return np.cumprod(alphas)
        elif self.schedule_type == "cosine":
            # Cosine schedule from Nichol & Dhariwal (2021)
            steps = np.arange(T + 1) / T
            alpha_bars = np.cos((steps + 0.008) / 1.008 * (np.pi / 2)) ** 2
            alpha_bars = alpha_bars / alpha_bars[0]
            return alpha_bars[1:]
        elif self.schedule_type == "sqrt":
            alpha_bars = 1.0 - np.sqrt(np.linspace(0, 1, T))
            return np.maximum(alpha_bars, 1e-6)
        else:
            raise ValueError(f"Unknown schedule: {self.schedule_type}")

    def alpha_bar(self, t):
        """Cumulative signal retention at timestep t (0-indexed)."""
        t = max(0, min(t, self.num_timesteps - 1))
        return float(self._alpha_bars[t])

    def noise_level(self, t):
        """Noise standard deviation at timestep t."""
        return math.sqrt(1.0 - self.alpha_bar(t))

    def signal_level(self, t):
        """Signal retention at timestep t."""
        return math.sqrt(self.alpha_bar(t))


# ============================================================================
# Forward process — add noise to a valid program
# ============================================================================

def forward_diffuse(program, t, schedule, rng=None):
    """Add noise to a CadProgram at diffusion timestep t.

    This is the forward process q(x_t | x_0):
        x_t = sqrt(alpha_bar_t) * embed(x_0) + sqrt(1-alpha_bar_t) * ε

    Returns:
        (noisy_vec, noise_vec, clean_vec) — all np.ndarray of shape
        (PROGRAM_EMBED_DIM,).
    """
    if rng is None:
        rng = np.random.RandomState()

    clean_vec = embed_program(program)
    noise = rng.randn(PROGRAM_EMBED_DIM)

    signal = schedule.signal_level(t)
    noise_scale = schedule.noise_level(t)

    noisy = signal * clean_vec + noise_scale * noise
    return noisy, noise, clean_vec


# ============================================================================
# Score function — approximates ∇ log p(x | mesh)
# ============================================================================

def score_program(program, target_v, target_f, mesh_features=None):
    """Evaluate how well a program matches the target mesh.

    Returns a scalar score (higher = better) that combines:
        - Geometric accuracy (Hausdorff distance)
        - Elegance penalty (operation count, complexity, param count)
        - Mesh conditioning alignment

    This serves as the negative energy function -E(x) whose gradient
    ∇_x(-E) = ∇_x log p(x|mesh) guides the reverse diffusion.
    """
    cad_v, cad_f = program.evaluate()
    if len(cad_v) == 0:
        return -100.0

    target_v = np.asarray(target_v, dtype=np.float64)
    hd = hausdorff_distance(cad_v, target_v)
    distance = hd["mean_symmetric"]

    bbox_diag = float(np.linalg.norm(
        target_v.max(axis=0) - target_v.min(axis=0)))
    if bbox_diag > 1e-12:
        distance /= bbox_diag

    # Base score: negative distance (lower distance = higher score)
    score = -distance

    # Elegance penalty
    score -= program.elegance_penalty()

    # Mesh conditioning bonus: reward programs whose geometric features
    # align with the target mesh features
    if mesh_features is not None:
        cad_features = extract_mesh_features(cad_v, cad_f)
        feature_similarity = 1.0 - float(np.mean(np.abs(
            cad_features - mesh_features)))
        score += 0.1 * feature_similarity

    return score


# ============================================================================
# Reverse diffusion strategies — the "denoising" operations
# ============================================================================

# Strategy vocabulary: each strategy is a denoising "move" that can be
# applied at a given noise level.  Strategies are tiered by the noise
# level at which they are most effective.

@dataclass
class DiffusionStrategy:
    """A single denoising strategy (one step of the reverse process)."""
    name: str
    tier: int  # 1=coarse (high noise), 2=medium, 3=fine (low noise)
    description: str = ""

    def apply(self, program, target_v, target_f, mesh_features,
              noise_level, rng):
        """Apply this strategy to produce candidate programs.

        Args:
            program: current CadProgram
            target_v, target_f: target mesh
            mesh_features: conditioning vector
            noise_level: current noise level (0=clean, 1=pure noise)
            rng: numpy RandomState

        Returns:
            list of (name, CadProgram) candidates
        """
        raise NotImplementedError


class StructuralRewrite(DiffusionStrategy):
    """Tier 1: Major structural changes — replace/reorder entire operation types.

    Active at high noise levels (early in reverse process).
    Analogous to the first denoising steps that establish coarse structure.
    """

    def __init__(self):
        super().__init__(
            name="structural_rewrite",
            tier=1,
            description="Replace operation types or reorder program structure",
        )

    def apply(self, program, target_v, target_f, mesh_features,
              noise_level, rng):
        candidates = []

        # Strategy 1a: Replace the weakest operation with a better-fitting type
        if program.n_enabled() > 0:
            gaps = find_program_gaps(program, target_v, target_f, max_gaps=3)
            for gap in gaps:
                if gap.action == "add" and program.n_enabled() < MAX_OPS:
                    p = program.copy()
                    add_operation(p, gap)
                    candidates.append(("struct:add_" + gap.suggested_op, p))

        # Strategy 1b: Try completely different initialization strategies
        # with noise-scaled exploration
        try:
            alt = initial_program(target_v, target_f)
            # Blend: at high noise, prefer the alternative; at low noise,
            # prefer the current program
            if noise_level > 0.5:
                candidates.append(("struct:reinit", alt))
            else:
                # Merge: keep current ops, add best ops from alternative
                merged = program.copy()
                for op in alt.operations[:3]:
                    if merged.n_enabled() < MAX_OPS:
                        merged.operations.append(
                            CadOp(op.op_type, copy.deepcopy(op.params)))
                        merged.invalidate_cache()
                candidates.append(("struct:merge", merged))
        except Exception:
            pass

        # Strategy 1c: Random op-type substitution (exploration)
        if program.n_enabled() > 0 and noise_level > 0.3:
            p = program.copy()
            # Pick a random enabled op and change its type
            enabled_indices = [i for i, op in enumerate(p.operations) if op.enabled]
            if enabled_indices:
                idx = rng.choice(enabled_indices)
                new_type = OP_TYPES[rng.randint(N_OP_TYPES)]
                old_params = p.operations[idx].params
                p.operations[idx] = CadOp(new_type, copy.deepcopy(old_params))
                p.invalidate_cache()
                candidates.append(("struct:swap_type", p))

        return candidates


class TopologicalAdjust(DiffusionStrategy):
    """Tier 2: Add/remove operations to match target topology.

    Active at medium noise levels.  Analogous to mid-stage denoising
    where major features emerge but details are still forming.
    """

    def __init__(self):
        super().__init__(
            name="topological_adjust",
            tier=2,
            description="Add/remove operations to match target structure",
        )

    def apply(self, program, target_v, target_f, mesh_features,
              noise_level, rng):
        candidates = []

        gaps = find_program_gaps(program, target_v, target_f, max_gaps=5)

        # Add operations for uncovered regions
        for gap in gaps:
            if gap.action == "add" and program.n_enabled() < MAX_OPS:
                p = program.copy()
                add_operation(p, gap)
                candidates.append(("topo:add_" + gap.suggested_op, p))

        # Remove operations that hurt more than they help
        for i in range(len(program.operations)):
            if program.operations[i].enabled:
                p = program.copy()
                remove_operation(p, i)
                candidates.append(("topo:remove_" + str(i), p))

        # Simplify: remove operations whose removal barely worsens cost
        p = program.copy()
        simplify_program(p, target_v, target_f)
        candidates.append(("topo:simplify", p))

        # Use mesh features to suggest operation types
        if mesh_features is not None and program.n_enabled() < MAX_OPS:
            suggested_ops = _suggest_ops_from_features(mesh_features)
            for op_type, params in suggested_ops[:2]:
                p = program.copy()
                p.operations.append(CadOp(op_type, params))
                p.invalidate_cache()
                candidates.append(("topo:feature_" + op_type, p))

        return candidates


class ParameterRefine(DiffusionStrategy):
    """Tier 3: Fine-tune operation parameters.

    Active at low noise levels (late in reverse process).
    Analogous to the final denoising steps that sharpen fine details.
    """

    def __init__(self):
        super().__init__(
            name="parameter_refine",
            tier=3,
            description="Fine-tune operation parameters for accuracy",
        )

    def apply(self, program, target_v, target_f, mesh_features,
              noise_level, rng):
        candidates = []

        # Refine each enabled operation's parameters
        gaps = find_program_gaps(program, target_v, target_f, max_gaps=3)

        for gap in gaps[:2]:
            if gap.nearest_program_op >= 0:
                p = program.copy()
                refine_operation(p, gap.nearest_program_op, target_v, target_f,
                                 max_iter=max(5, int(20 * (1 - noise_level))))
                candidates.append(("refine:gap_" + str(gap.nearest_program_op), p))

        # Stochastic parameter perturbation (Langevin step)
        if program.n_enabled() > 0:
            p = program.copy()
            _langevin_step(p, noise_level, rng)
            candidates.append(("refine:langevin", p))

        # Coordinate descent on all parameters
        for i, op in enumerate(program.operations):
            if op.enabled:
                p = program.copy()
                refine_operation(p, i, target_v, target_f,
                                 max_iter=max(3, int(10 * (1 - noise_level))))
                candidates.append(("refine:op_" + str(i), p))

        return candidates


class ElegancePolish(DiffusionStrategy):
    """Tier 3: Improve program elegance without sacrificing accuracy.

    Active at very low noise levels (final polishing).
    Analogous to the last few denoising steps that add subtle quality.
    """

    def __init__(self):
        super().__init__(
            name="elegance_polish",
            tier=3,
            description="Improve program elegance (fewer ops, cleaner design)",
        )

    def apply(self, program, target_v, target_f, mesh_features,
              noise_level, rng):
        candidates = []

        # Try removing each op and check if accuracy is maintained
        for i in range(len(program.operations)):
            if program.operations[i].enabled:
                p = program.copy()
                p.operations[i].enabled = False
                p.invalidate_cache()
                candidates.append(("polish:disable_" + str(i), p))

        # Try merging similar operations
        merged = _try_merge_ops(program, target_v, target_f)
        if merged is not None:
            candidates.append(("polish:merge", merged))

        # Simplification pass
        p = program.copy()
        simplify_program(p, target_v, target_f)
        candidates.append(("polish:simplify", p))

        return candidates


# ============================================================================
# Helper functions for strategies
# ============================================================================

def _suggest_ops_from_features(mesh_features):
    """Use mesh conditioning features to suggest likely operation types.

    Analogous to how the text embedding in stable diffusion biases
    the denoising toward certain image structures.
    """
    suggestions = []

    # High circularity + elongation → revolve or profiled_cylinder
    # features[9] is sigmoid-mapped: >0.3 means raw elongation >~2
    # features[14] is cross-sectional circularity
    if mesh_features[14] > 0.4 and mesh_features[9] > 0.3:
        suggestions.append(("revolve", {
            "profile_rz": [(1.0, 0.0), (1.5, 0.5), (1.0, 1.0)],
            "n_angular": 32,
        }))
        suggestions.append(("profiled_cylinder", {
            "center": np.zeros(3),
            "axis": np.array([0, 0, 1.0]),
            "radii": [1.0, 1.5, 1.0],
            "heights": [0.0, 0.5, 1.0],
        }))

    # High circularity + low elongation → sphere or torus
    if mesh_features[13] > 0.8 and mesh_features[9] < 0.5:
        suggestions.append(("sphere", {
            "center": np.zeros(3),
            "radius": 1.0,
        }))
        suggestions.append(("torus", {
            "center": np.zeros(3),
            "major_r": 2.0,
            "minor_r": 0.5,
        }))

    # Low circularity + high flatness → box or extrude
    if mesh_features[13] < 0.5 and mesh_features[20] > 0.8:
        suggestions.append(("box", {
            "center": np.zeros(3),
            "dimensions": np.ones(3),
        }))
        suggestions.append(("extrude", {
            "polygon": [(0, 0), (1, 0), (1, 1), (0, 1)],
            "height": 1.0,
        }))

    # High elongation → cylinder
    # features[9] > 0.67 means raw elongation > ~5
    if mesh_features[9] > 0.67:
        suggestions.append(("cylinder", {
            "center": np.zeros(3),
            "axis": np.array([0, 0, 1.0]),
            "radius": 1.0,
            "height": 5.0,
        }))

    # High symmetry → add mirror
    for ax in range(3):
        if mesh_features[25 + ax] > 0.9:
            normal = np.zeros(3)
            normal[ax] = 1.0
            suggestions.append(("mirror", {
                "point": np.zeros(3),
                "normal": normal,
            }))

    return suggestions


def _langevin_step(program, noise_level, rng):
    """Apply a Langevin dynamics step to program parameters.

    Perturbs numeric parameters with noise proportional to the current
    noise level.  This is the stochastic component of the reverse
    diffusion process.
    """
    step_size = noise_level * 0.1  # Scale perturbation by noise level
    for op in program.operations:
        if not op.enabled:
            continue
        for k, v in op.params.items():
            if isinstance(v, np.ndarray) and v.dtype in (np.float64, np.float32):
                perturbation = rng.randn(*v.shape) * step_size * np.maximum(
                    np.abs(v), 0.1)
                op.params[k] = v + perturbation
            elif isinstance(v, float):
                perturbation = rng.randn() * step_size * max(abs(v), 0.1)
                op.params[k] = v + perturbation
            elif isinstance(v, list) and all(isinstance(x, (int, float)) for x in v):
                for j in range(len(v)):
                    if isinstance(v[j], float):
                        v[j] += rng.randn() * step_size * max(abs(v[j]), 0.1)
    program.invalidate_cache()


def _try_merge_ops(program, target_v, target_f):
    """Try to merge similar adjacent operations into one."""
    if program.n_enabled() < 2:
        return None

    best_merged = None
    best_cost = program.total_cost(target_v, target_f)

    enabled = [(i, op) for i, op in enumerate(program.operations) if op.enabled]
    for j in range(len(enabled) - 1):
        idx_a, op_a = enabled[j]
        idx_b, op_b = enabled[j + 1]

        if op_a.op_type != op_b.op_type:
            continue

        # Try disabling the second op (simplest merge)
        p = program.copy()
        p.operations[idx_b].enabled = False
        p.invalidate_cache()
        cost = p.total_cost(target_v, target_f)
        if cost < best_cost:
            best_cost = cost
            best_merged = p

    return best_merged


# ============================================================================
# DiffusionStrategySelector — the main reverse-process loop
# ============================================================================

# Default strategy set
DEFAULT_STRATEGIES = [
    StructuralRewrite(),
    TopologicalAdjust(),
    ParameterRefine(),
    ElegancePolish(),
]


@dataclass
class DiffusionConfig:
    """Configuration for the diffusion strategy selector."""
    num_timesteps: int = 50
    schedule_type: str = "cosine"
    n_candidates_per_step: int = 8
    patience: int = 5
    guidance_scale: float = 2.0  # Classifier-free guidance strength
    temperature: float = 1.0     # Sampling temperature
    seed: int = 42


@dataclass
class DiffusionStep:
    """Record of one reverse-diffusion step."""
    timestep: int
    noise_level: float
    strategy_name: str
    score_before: float
    score_after: float
    improved: bool
    n_ops: int


def run_diffusion_strategy(target_v, target_f, config=None):
    """Run the diffusion-based strategy selector.

    This is the main entry point.  It performs the reverse diffusion
    process: starting from a noisy/random program and iteratively
    denoising it toward a valid CAD reconstruction of the target mesh.

    The process has three phases (mapped to noise level):

    Phase 1 — Structure (high noise, t=T..2T/3):
        Establish coarse program structure.  Try different operation
        types, program topologies, and initialization strategies.
        Analogous to the first denoising steps that establish the
        overall composition of an image.

    Phase 2 — Topology (medium noise, t=2T/3..T/3):
        Add/remove operations to match target topology.  Fine-tune
        which operations are needed and where they go.
        Analogous to mid-stage denoising where major features emerge.

    Phase 3 — Refinement (low noise, t=T/3..0):
        Fine-tune parameters and polish elegance.  Small, precise
        adjustments to achieve the final accuracy.
        Analogous to the last denoising steps that sharpen details.

    Args:
        target_v: (N, 3) target mesh vertices
        target_f: (M, 3) target mesh faces
        config: DiffusionConfig or None for defaults

    Returns:
        dict with:
            program:      final CadProgram
            history:      list of DiffusionStep records
            cad_vertices: final mesh vertices
            cad_faces:    final mesh faces
            mesh_features: conditioning vector used
            total_cost:   final total_cost
    """
    if config is None:
        config = DiffusionConfig()

    target_v = np.asarray(target_v, dtype=np.float64)
    target_f = np.asarray(target_f)

    rng = np.random.RandomState(config.seed)
    schedule = NoiseSchedule(
        num_timesteps=config.num_timesteps,
        schedule_type=config.schedule_type,
    )
    strategies = DEFAULT_STRATEGIES

    # --- Extract mesh conditioning features ---
    mesh_features = extract_mesh_features(target_v, target_f)

    # --- Initialize: start from the best initial program ---
    # (In a trained model, we'd start from pure noise.  Here we start
    # from a reasonable initialization and use diffusion to improve it.)
    program = initial_program(target_v, target_f)
    best_score = score_program(program, target_v, target_f, mesh_features)
    history = []
    no_improvement_count = 0

    # --- Reverse diffusion loop ---
    for t_rev in range(config.num_timesteps):
        # Map reverse timestep to forward timestep
        # t_rev=0 → t=T-1 (high noise), t_rev=T-1 → t=0 (clean)
        t = config.num_timesteps - 1 - t_rev
        noise_level = schedule.noise_level(t)

        # Select strategies appropriate for this noise level
        active_strategies = _select_strategies(strategies, noise_level)

        # Generate candidates from all active strategies
        all_candidates = []
        for strategy in active_strategies:
            try:
                candidates = strategy.apply(
                    program, target_v, target_f, mesh_features,
                    noise_level, rng)
                all_candidates.extend(
                    [(strategy.name, name, cand) for name, cand in candidates])
            except Exception:
                continue

        if not all_candidates:
            history.append(DiffusionStep(
                timestep=t, noise_level=noise_level,
                strategy_name="none", score_before=best_score,
                score_after=best_score, improved=False,
                n_ops=program.n_enabled(),
            ))
            no_improvement_count += 1
            if no_improvement_count >= config.patience:
                break
            continue

        # Score all candidates
        scored = []
        for strat_name, cand_name, cand in all_candidates:
            try:
                s = score_program(cand, target_v, target_f, mesh_features)
                # Classifier-free guidance: amplify the conditioning signal
                # score = (1 + w) * conditional_score - w * unconditional_score
                unconditional_s = score_program(cand, target_v, target_f, None)
                guided_s = ((1 + config.guidance_scale) * s
                            - config.guidance_scale * unconditional_s)
                scored.append((guided_s, strat_name, cand_name, cand))
            except Exception:
                continue

        if not scored:
            no_improvement_count += 1
            if no_improvement_count >= config.patience:
                break
            continue

        # Select best candidate (with optional temperature sampling)
        scored.sort(key=lambda x: x[0], reverse=True)

        if config.temperature > 0 and len(scored) > 1:
            # Softmax sampling with temperature
            scores_arr = np.array([s[0] for s in scored[:config.n_candidates_per_step]])
            scores_arr = scores_arr - scores_arr.max()  # numerical stability
            probs = np.exp(scores_arr / max(config.temperature, 0.01))
            probs = probs / probs.sum()
            chosen_idx = rng.choice(len(probs), p=probs)
        else:
            chosen_idx = 0

        chosen_score, chosen_strat, chosen_name, chosen_program = \
            scored[chosen_idx]

        # Accept or reject (Metropolis-Hastings style)
        # At high noise levels, accept more liberally (exploration)
        # At low noise levels, only accept improvements (exploitation)
        accept_threshold = -0.05 * noise_level  # More permissive at high noise
        improvement = chosen_score - best_score

        if improvement > accept_threshold:
            program = chosen_program
            best_score = chosen_score
            no_improvement_count = 0
            improved = True
        else:
            no_improvement_count += 1
            improved = False

        history.append(DiffusionStep(
            timestep=t, noise_level=round(noise_level, 4),
            strategy_name=f"{chosen_strat}:{chosen_name}",
            score_before=round(best_score - improvement if improved else best_score, 6),
            score_after=round(best_score, 6),
            improved=improved,
            n_ops=program.n_enabled(),
        ))

        if no_improvement_count >= config.patience:
            break

    # --- Final polishing ---
    simplify_program(program, target_v, target_f)
    final_score = score_program(program, target_v, target_f, mesh_features)

    cad_v, cad_f = program.evaluate()

    return {
        "program": program,
        "history": history,
        "cad_vertices": cad_v,
        "cad_faces": cad_f,
        "mesh_features": mesh_features,
        "total_cost": program.total_cost(target_v, target_f),
        "diffusion_score": final_score,
        "n_ops": program.n_enabled(),
        "n_steps": len(history),
        "n_improved": sum(1 for s in history if s.improved),
    }


def _select_strategies(strategies, noise_level):
    """Select which strategies are active at the current noise level.

    Tier mapping:
        noise > 0.6  → Tier 1 (structural) + Tier 2 (topological)
        0.3 < noise ≤ 0.6 → Tier 2 (topological) + Tier 3 (refinement)
        noise ≤ 0.3  → Tier 3 (refinement + elegance polish)

    All tiers are always considered but with different probabilities;
    lower-tier strategies dominate at their preferred noise level.
    """
    active = []
    for s in strategies:
        if noise_level > 0.6:
            if s.tier <= 2:
                active.append(s)
        elif noise_level > 0.3:
            if s.tier >= 2:
                active.append(s)
        else:
            if s.tier >= 3:
                active.append(s)

    # Always include at least one strategy
    if not active:
        active = strategies[:]

    return active


# ============================================================================
# Comparison runner: diffusion vs classical
# ============================================================================

def compare_approaches(target_v, target_f, diffusion_config=None,
                       classical_max_rounds=30):
    """Run both diffusion and classical approaches and compare results.

    Useful for benchmarking the diffusion approach against the existing
    RED/BLUE team evolution loop.

    Returns:
        dict with results from both approaches and comparison metrics.
    """
    from .cad_program import run_cad_program_loop

    target_v = np.asarray(target_v, dtype=np.float64)
    target_f = np.asarray(target_f)

    # Run classical approach
    classical = run_cad_program_loop(target_v, target_f,
                                     max_rounds=classical_max_rounds)

    # Run diffusion approach
    diffusion = run_diffusion_strategy(target_v, target_f,
                                        config=diffusion_config)

    # Compare
    c_cost = classical["total_cost"]
    d_cost = diffusion["total_cost"]

    return {
        "classical": {
            "total_cost": c_cost,
            "n_ops": classical["n_ops"],
            "complexity": classical["complexity"],
            "program_summary": classical["program"].summary(),
        },
        "diffusion": {
            "total_cost": d_cost,
            "n_ops": diffusion["n_ops"],
            "diffusion_score": diffusion["diffusion_score"],
            "n_steps": diffusion["n_steps"],
            "n_improved": diffusion["n_improved"],
            "program_summary": diffusion["program"].summary(),
        },
        "comparison": {
            "cost_ratio": d_cost / max(c_cost, 1e-12),
            "diffusion_better": d_cost < c_cost,
            "cost_improvement": c_cost - d_cost,
            "ops_diff": diffusion["n_ops"] - classical["n_ops"],
        },
    }
