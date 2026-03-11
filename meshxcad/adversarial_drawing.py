"""Adversarial hardening for the Drawing -> CAD pipeline.

Three strategies systematically stress-test the interpreter:
  A. adversarial_generator_loop   -- increasingly complex generated drawings
  B. adversarial_perturbation_loop -- image-level perturbations on known-good drawings
  C. adversarial_mutation_loop     -- CAD-level edge-case mutations

Plus FailureDatabase for structured tracking and run_adversarial_suite() to
orchestrate all strategies and produce a combined report.
"""

import json
import logging
import math
import os
import tempfile
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

from .cad_program import CadOp, CadProgram
from .drawing import render_drawing_sheet, render_orthographic
from .drawing_compare import compare_drawings
from .drawing_to_cad import drawing_to_cad
from .general_align import hausdorff_distance
from .synthetic import make_sphere_mesh, make_cylinder_mesh, make_cube_mesh

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Failure tracking
# ---------------------------------------------------------------------------

FAILURE_TYPES = [
    "missing_feature",
    "extra_feature",
    "wrong_dimension",
    "wrong_shape_type",
    "wrong_orientation",
    "scale_error",
    "missing_view",
    "json_parse_error",
    "hallucination",
    "symmetry_error",
    "empty_result",
]


@dataclass
class DrawingFailure:
    """A single recorded failure from adversarial testing."""
    drawing_image: np.ndarray
    expected_program: dict       # serialised CadProgram
    actual_program: dict         # serialised CadProgram (or None)
    failure_type: str            # from FAILURE_TYPES
    severity: float              # 0-1
    details: dict = field(default_factory=dict)
    round_info: dict = field(default_factory=dict)


class FailureDatabase:
    """Structured collection of pipeline failures."""

    def __init__(self):
        self.failures: list[DrawingFailure] = []

    def add_failure(self, failure: DrawingFailure):
        self.failures.append(failure)

    def cluster_by_type(self) -> dict:
        """Group failures by failure_type."""
        clusters = {}
        for f in self.failures:
            clusters.setdefault(f.failure_type, []).append(f)
        return clusters

    def worst_failure_types(self, top_n: int = 5) -> list:
        """Return the top-N failure types ranked by mean severity."""
        clusters = self.cluster_by_type()
        ranked = []
        for ftype, items in clusters.items():
            mean_sev = float(np.mean([it.severity for it in items]))
            ranked.append({
                "failure_type": ftype,
                "count": len(items),
                "mean_severity": mean_sev,
                "max_severity": float(max(it.severity for it in items)),
            })
        ranked.sort(key=lambda x: x["mean_severity"], reverse=True)
        return ranked[:top_n]

    def summary(self) -> dict:
        """Overall statistics."""
        if not self.failures:
            return {"total": 0, "by_type": {}, "mean_severity": 0.0}
        severities = [f.severity for f in self.failures]
        by_type = {}
        for f in self.failures:
            by_type[f.failure_type] = by_type.get(f.failure_type, 0) + 1
        return {
            "total": len(self.failures),
            "by_type": by_type,
            "mean_severity": float(np.mean(severities)),
            "max_severity": float(max(severities)),
            "worst_types": self.worst_failure_types(),
        }

    def save(self, path: str):
        """Save to JSON (images stored as shape-only references)."""
        records = []
        for f in self.failures:
            rec = {
                "failure_type": f.failure_type,
                "severity": f.severity,
                "details": f.details,
                "round_info": f.round_info,
                "expected_program": f.expected_program,
                "actual_program": f.actual_program,
                "image_shape": list(f.drawing_image.shape) if f.drawing_image is not None else None,
            }
            records.append(rec)
        with open(path, "w") as fh:
            json.dump({"failures": records, "summary": self.summary()}, fh, indent=2)

    @classmethod
    def load(cls, path: str) -> "FailureDatabase":
        """Load from JSON (images are placeholders)."""
        with open(path) as fh:
            data = json.load(fh)
        db = cls()
        for rec in data.get("failures", []):
            shape = rec.get("image_shape")
            if shape:
                img = np.zeros(shape, dtype=np.uint8)
            else:
                img = np.zeros((1, 1), dtype=np.uint8)
            db.add_failure(DrawingFailure(
                drawing_image=img,
                expected_program=rec.get("expected_program"),
                actual_program=rec.get("actual_program"),
                failure_type=rec.get("failure_type", "empty_result"),
                severity=rec.get("severity", 0.0),
                details=rec.get("details", {}),
                round_info=rec.get("round_info", {}),
            ))
        return db


# ---------------------------------------------------------------------------
# CadProgram generators (for Strategy A)
# ---------------------------------------------------------------------------

def _make_box_program(width=20.0, depth=15.0, height=10.0):
    """Simple box CadProgram."""
    return CadProgram([CadOp("box", {
        "center": [0, 0, height / 2],
        "dimensions": [width, depth, height],
        "subdivisions": 4,
    })])


def _make_cylinder_program(radius=8.0, height=20.0):
    """Simple cylinder CadProgram."""
    return CadProgram([CadOp("cylinder", {
        "center": [0, 0, height / 2],
        "axis": [0, 0, 1],
        "radius": radius,
        "height": height,
        "radial_divs": 24,
        "height_divs": 10,
    })])


def _make_sphere_program(radius=10.0):
    """Simple sphere CadProgram."""
    return CadProgram([CadOp("sphere", {
        "center": [0, 0, 0],
        "radius": radius,
        "divs": 20,
    })])


def _add_holes_to_program(program, n_holes=1, body_radius=8.0, body_height=20.0):
    """Add subtract_cylinder holes to an existing program."""
    rng = np.random.RandomState(42)
    for i in range(n_holes):
        hole_r = body_radius * rng.uniform(0.05, 0.25)
        angle = 2 * math.pi * i / max(n_holes, 1)
        offset_r = body_radius * 0.5
        cx = offset_r * math.cos(angle)
        cy = offset_r * math.sin(angle)
        program.operations.append(CadOp("subtract_cylinder", {
            "center": [cx, cy, body_height / 2],
            "axis": [0, 0, 1],
            "radius": hole_r,
            "height": body_height * 1.5,
        }))
    program.invalidate_cache()
    return program


def _make_composition_program(rng, n_parts=2):
    """Multi-primitive composition."""
    ops = []
    for i in range(n_parts):
        ptype = rng.choice(["box", "cylinder", "sphere"])
        offset = [float(rng.uniform(-15, 15)),
                  float(rng.uniform(-15, 15)),
                  float(rng.uniform(0, 20))]
        if ptype == "box":
            s = float(rng.uniform(5, 15))
            ops.append(CadOp("box", {
                "center": offset,
                "dimensions": [s, s * rng.uniform(0.5, 1.5),
                               s * rng.uniform(0.5, 1.5)],
                "subdivisions": 4,
            }))
        elif ptype == "cylinder":
            r = float(rng.uniform(3, 10))
            h = float(rng.uniform(5, 25))
            ops.append(CadOp("cylinder", {
                "center": offset,
                "axis": [0, 0, 1],
                "radius": r, "height": h,
                "radial_divs": 24, "height_divs": 10,
            }))
        else:
            r = float(rng.uniform(3, 10))
            ops.append(CadOp("sphere", {
                "center": offset,
                "radius": r, "divs": 20,
            }))
    return CadProgram(ops)


def _make_revolve_program(rng):
    """Revolve profile with features."""
    n_pts = rng.randint(4, 8)
    profile = []
    for i in range(n_pts):
        r = float(rng.uniform(3, 15))
        z = float(i * rng.uniform(5, 15))
        profile.append([r, z])
    # Ensure valid (positive radii, ascending z)
    profile.sort(key=lambda p: p[1])
    for p in profile:
        p[0] = max(p[0], 0.5)
    return CadProgram([CadOp("revolve", {
        "center": [0, 0, 0],
        "profile": profile,
        "divs": 48,
    })])


def generate_program_for_round(round_num: int, rng=None) -> tuple:
    """Generate a CadProgram appropriate for a given difficulty round.

    Returns (program, complexity_label).
    """
    if rng is None:
        rng = np.random.RandomState(round_num)

    if round_num <= 10:
        # Single primitives with varying dimensions
        ptype = rng.choice(["box", "cylinder", "sphere"])
        scale = 5.0 + round_num * 2
        if ptype == "box":
            prog = _make_box_program(scale, scale * 0.8, scale * 0.6)
        elif ptype == "cylinder":
            prog = _make_cylinder_program(scale * 0.4, scale)
        else:
            prog = _make_sphere_program(scale * 0.5)
        return prog, "single_primitive"

    elif round_num <= 20:
        # Primitives with holes
        n_holes = 1 + (round_num - 11) // 3  # 1-4 holes
        r = 10.0
        h = 25.0
        prog = _make_cylinder_program(r, h)
        _add_holes_to_program(prog, n_holes, r, h)
        return prog, "primitive_with_holes"

    elif round_num <= 30:
        # Multi-primitive compositions
        n_parts = 2 + (round_num - 21) // 3
        prog = _make_composition_program(rng, n_parts)
        return prog, "multi_primitive"

    elif round_num <= 40:
        # Revolve profiles with features
        prog = _make_revolve_program(rng)
        if round_num > 35:
            _add_holes_to_program(prog, 2, 10.0, 30.0)
        return prog, "revolve_profile"

    else:
        # Complex assemblies from catalogs
        return None, "catalog_object"


def _get_catalog_program(round_num: int):
    """Fetch a catalog object and wrap as CadProgram-equivalent mesh.

    Returns (vertices, faces, label) or None.
    """
    try:
        from .objects.catalog import OBJECT_CATALOG, make_ornate
        names = list(OBJECT_CATALOG.keys())
        idx = (round_num - 41) % len(names)
        name = names[idx]
        v, f = make_ornate(name)
        return v, f, f"catalog:{name}"
    except Exception as e:
        logger.warning("Failed to load catalog object: %s", e)
        return None


# ---------------------------------------------------------------------------
# Image perturbation functions (for Strategy B)
# ---------------------------------------------------------------------------

def apply_gaussian_noise(image: np.ndarray, sigma: float) -> np.ndarray:
    """Add Gaussian noise to image."""
    noise = np.random.RandomState(42).normal(0, sigma, image.shape)
    noisy = image.astype(np.float64) + noise
    return np.clip(noisy, 0, 255).astype(np.uint8)


def apply_resolution_change(image: np.ndarray, target_size: int) -> np.ndarray:
    """Downsample then upsample back to original size."""
    h, w = image.shape[:2]
    pil = Image.fromarray(image)
    small = pil.resize((target_size, target_size), Image.BILINEAR)
    restored = small.resize((w, h), Image.BILINEAR)
    return np.array(restored)


def apply_rotation(image: np.ndarray, angle_deg: float) -> np.ndarray:
    """Rotate image by a small angle (simulating scan skew)."""
    pil = Image.fromarray(image)
    rotated = pil.rotate(angle_deg, resample=Image.BILINEAR,
                         expand=False, fillcolor=255)
    return np.array(rotated)


def apply_line_weight(image: np.ndarray, mode: str = "dilate") -> np.ndarray:
    """Erode or dilate edge pixels to vary line weight."""
    pil = Image.fromarray(image)
    if mode == "dilate":
        filtered = pil.filter(ImageFilter.MinFilter(3))
    else:
        filtered = pil.filter(ImageFilter.MaxFilter(3))
    return np.array(filtered)


def apply_salt_and_pepper(image: np.ndarray, density: float = 0.01) -> np.ndarray:
    """Add salt-and-pepper noise."""
    rng = np.random.RandomState(42)
    out = image.copy()
    n_pixels = image.size
    n_salt = int(n_pixels * density / 2)
    n_pepper = int(n_pixels * density / 2)

    flat = out.reshape(-1)
    salt_idx = rng.choice(len(flat), n_salt, replace=False)
    pepper_idx = rng.choice(len(flat), n_pepper, replace=False)
    flat[salt_idx] = 255
    flat[pepper_idx] = 0
    return out


def apply_gaussian_blur(image: np.ndarray, sigma: float) -> np.ndarray:
    """Apply Gaussian blur."""
    pil = Image.fromarray(image)
    radius = max(1, int(sigma))
    blurred = pil.filter(ImageFilter.GaussianBlur(radius=radius))
    return np.array(blurred)


def apply_contrast_reduction(image: np.ndarray, factor: float = 0.5) -> np.ndarray:
    """Reduce contrast by blending toward mean gray."""
    mean_val = image.mean()
    adjusted = mean_val + factor * (image.astype(np.float64) - mean_val)
    return np.clip(adjusted, 0, 255).astype(np.uint8)


def apply_partial_crop(image: np.ndarray, fraction: float = 0.1) -> np.ndarray:
    """Remove edges by filling with white (simulating partial crop)."""
    h, w = image.shape[:2]
    margin_h = int(h * fraction)
    margin_w = int(w * fraction)
    out = image.copy()
    fill = 255
    if out.ndim == 3:
        out[:margin_h, :, :] = fill
        out[-margin_h:, :, :] = fill
        out[:, :margin_w, :] = fill
        out[:, -margin_w:, :] = fill
    else:
        out[:margin_h, :] = fill
        out[-margin_h:, :] = fill
        out[:, :margin_w] = fill
        out[:, -margin_w:] = fill
    return out


# Registry of all perturbation functions with parameter sets
PERTURBATIONS = {
    "gaussian_noise": [
        ("sigma_5", lambda img: apply_gaussian_noise(img, 5)),
        ("sigma_10", lambda img: apply_gaussian_noise(img, 10)),
        ("sigma_20", lambda img: apply_gaussian_noise(img, 20)),
        ("sigma_40", lambda img: apply_gaussian_noise(img, 40)),
    ],
    "resolution_change": [
        ("256px", lambda img: apply_resolution_change(img, 256)),
        ("128px", lambda img: apply_resolution_change(img, 128)),
        ("64px", lambda img: apply_resolution_change(img, 64)),
    ],
    "rotation": [
        ("neg1", lambda img: apply_rotation(img, -1)),
        ("pos1", lambda img: apply_rotation(img, 1)),
        ("neg3", lambda img: apply_rotation(img, -3)),
        ("pos3", lambda img: apply_rotation(img, 3)),
        ("neg5", lambda img: apply_rotation(img, -5)),
        ("pos5", lambda img: apply_rotation(img, 5)),
    ],
    "line_weight": [
        ("erode", lambda img: apply_line_weight(img, "erode")),
        ("dilate", lambda img: apply_line_weight(img, "dilate")),
    ],
    "salt_and_pepper": [
        ("density_01", lambda img: apply_salt_and_pepper(img, 0.01)),
        ("density_05", lambda img: apply_salt_and_pepper(img, 0.05)),
    ],
    "gaussian_blur": [
        ("sigma_1", lambda img: apply_gaussian_blur(img, 1)),
        ("sigma_2", lambda img: apply_gaussian_blur(img, 2)),
        ("sigma_3", lambda img: apply_gaussian_blur(img, 3)),
    ],
    "contrast_reduction": [
        ("factor_07", lambda img: apply_contrast_reduction(img, 0.7)),
        ("factor_05", lambda img: apply_contrast_reduction(img, 0.5)),
        ("factor_03", lambda img: apply_contrast_reduction(img, 0.3)),
    ],
    "partial_crop": [
        ("crop_10pct", lambda img: apply_partial_crop(img, 0.10)),
        ("crop_20pct", lambda img: apply_partial_crop(img, 0.20)),
    ],
}


# ---------------------------------------------------------------------------
# Interpreter invocation helper
# ---------------------------------------------------------------------------

def _interpret_drawing_image(interpreter, image: np.ndarray,
                             views_hint=None):
    """Save image to temp file, run interpreter, return DrawingSpec or None."""
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        tmp_path = tmp.name
        Image.fromarray(image).save(tmp_path)
    try:
        spec = interpreter.interpret_drawing(tmp_path, views_hint=views_hint)
        return spec
    except Exception as e:
        logger.warning("Interpreter failed: %s", e)
        return None
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


def _classify_failure(mesh_dist_norm: float, program1, program2,
                      spec) -> tuple:
    """Classify a failure and return (failure_type, severity).

    Returns (None, 0.0) if no failure.
    """
    if spec is None:
        return "json_parse_error", 1.0

    if program2 is None:
        return "empty_result", 1.0

    v2, f2 = program2.evaluate()
    if len(v2) == 0:
        return "empty_result", 1.0

    severity = min(1.0, mesh_dist_norm)

    if severity < 0.05:
        return None, 0.0  # success

    # Try to classify the failure mode
    ops1 = {op.op_type for op in program1.operations if op.enabled}
    ops2 = {op.op_type for op in program2.operations if op.enabled}

    if not ops2:
        return "empty_result", severity

    # Different primary shape
    prim_types = {"sphere", "cylinder", "box", "cone", "revolve", "extrude"}
    shapes1 = ops1 & prim_types
    shapes2 = ops2 & prim_types
    if shapes1 and shapes2 and shapes1 != shapes2:
        return "wrong_shape_type", severity

    # Missing features (subtractions in original not in reconstruction)
    subs1 = sum(1 for op in program1.operations
                if op.enabled and op.op_type == "subtract_cylinder")
    subs2 = sum(1 for op in program2.operations
                if op.enabled and op.op_type == "subtract_cylinder")
    if subs1 > subs2:
        return "missing_feature", severity
    if subs2 > subs1 + 1:
        return "hallucination", severity

    # Scale error (dimensions far off but shape correct)
    if severity > 0.3 and shapes1 == shapes2:
        return "scale_error", severity

    # Dimension error
    if severity > 0.1:
        return "wrong_dimension", severity

    return "wrong_dimension", severity


# ---------------------------------------------------------------------------
# Strategy A: Adversarial Generator Loop
# ---------------------------------------------------------------------------

def adversarial_generator_loop(interpreter, n_rounds=50, output_dir=None,
                               views=("front", "side", "top"),
                               image_size=512):
    """Generate increasingly hard drawings that challenge the interpreter.

    Args:
        interpreter: DrawingInterpreter instance.
        n_rounds: total rounds (1-50).
        output_dir: optional directory for saving intermediates.
        views: view types for rendering.
        image_size: render resolution.

    Returns:
        dict with results, failure_db, and statistics.
    """
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    failure_db = FailureDatabase()
    results = []
    stats_by_level = {}

    for round_num in range(1, n_rounds + 1):
        logger.info("Generator round %d/%d", round_num, n_rounds)
        round_result = {"round": round_num, "status": "unknown"}

        try:
            program, level = generate_program_for_round(round_num)

            # For catalog rounds, use pre-made meshes
            if program is None and level == "catalog_object":
                cat_result = _get_catalog_program(round_num)
                if cat_result is None:
                    round_result["status"] = "skipped"
                    results.append(round_result)
                    continue
                v1, f1, label = cat_result
                round_result["label"] = label
            else:
                v1, f1 = program.evaluate()
                round_result["label"] = level

            if len(v1) == 0:
                round_result["status"] = "empty_source"
                results.append(round_result)
                continue

            # Render drawing
            drawing = render_drawing_sheet(v1, f1, views, image_size)

            if output_dir:
                Image.fromarray(drawing).save(
                    os.path.join(output_dir, f"gen_round_{round_num:03d}.png"))

            # Interpret
            spec = _interpret_drawing_image(interpreter, drawing,
                                            views_hint=list(views))

            if spec is None:
                failure_db.add_failure(DrawingFailure(
                    drawing_image=drawing,
                    expected_program=program.to_dict() if program else {},
                    actual_program=None,
                    failure_type="json_parse_error",
                    severity=1.0,
                    round_info={"strategy": "generator", "round": round_num,
                                "level": level},
                ))
                round_result["status"] = "interpret_failed"
                _update_level_stats(stats_by_level, level, False)
                results.append(round_result)
                continue

            # Reconstruct
            program2 = drawing_to_cad(spec)
            v2, f2 = program2.evaluate()

            if len(v2) == 0:
                failure_db.add_failure(DrawingFailure(
                    drawing_image=drawing,
                    expected_program=program.to_dict() if program else {},
                    actual_program=program2.to_dict(),
                    failure_type="empty_result",
                    severity=1.0,
                    round_info={"strategy": "generator", "round": round_num,
                                "level": level},
                ))
                round_result["status"] = "empty_reconstruction"
                _update_level_stats(stats_by_level, level, False)
                results.append(round_result)
                continue

            # Compare meshes
            hd = hausdorff_distance(v1, v2)
            bbox_diag = float(np.linalg.norm(
                v1.max(axis=0) - v1.min(axis=0)))
            mesh_dist_norm = hd["mean_symmetric"] / max(bbox_diag, 1e-6)

            round_result["mesh_dist_norm"] = float(mesh_dist_norm)
            round_result["hausdorff"] = float(hd["hausdorff"])

            ftype, severity = _classify_failure(
                mesh_dist_norm, program if program else CadProgram(), program2,
                spec)

            if ftype is not None:
                failure_db.add_failure(DrawingFailure(
                    drawing_image=drawing,
                    expected_program=program.to_dict() if program else {},
                    actual_program=program2.to_dict(),
                    failure_type=ftype,
                    severity=severity,
                    details={"mesh_dist_norm": mesh_dist_norm},
                    round_info={"strategy": "generator", "round": round_num,
                                "level": level},
                ))
                round_result["status"] = f"failure:{ftype}"
                _update_level_stats(stats_by_level, level, False)
            else:
                round_result["status"] = "success"
                _update_level_stats(stats_by_level, level, True)

        except Exception as e:
            logger.error("Generator round %d failed: %s", round_num, e)
            round_result["status"] = f"error:{e}"
            _update_level_stats(stats_by_level, level if 'level' in dir() else "unknown", False)

        results.append(round_result)

    return {
        "strategy": "generator",
        "results": results,
        "failure_db": failure_db,
        "stats_by_level": stats_by_level,
        "n_rounds": n_rounds,
        "n_successes": sum(1 for r in results if r["status"] == "success"),
        "n_failures": sum(1 for r in results if r["status"].startswith("failure")),
    }


def _update_level_stats(stats, level, success):
    if level not in stats:
        stats[level] = {"total": 0, "success": 0, "fail": 0}
    stats[level]["total"] += 1
    if success:
        stats[level]["success"] += 1
    else:
        stats[level]["fail"] += 1


# ---------------------------------------------------------------------------
# Strategy B: Adversarial Perturbation Loop
# ---------------------------------------------------------------------------

def adversarial_perturbation_loop(interpreter, test_programs,
                                  n_perturbations=8,
                                  views=("front", "side", "top"),
                                  image_size=512, output_dir=None):
    """Perturb known-good drawings to find interpreter breaking points.

    Args:
        interpreter: DrawingInterpreter instance.
        test_programs: list of CadProgram objects to test.
        n_perturbations: max perturbation categories to test (1-8).
        views: view types for rendering.
        image_size: render resolution.
        output_dir: optional directory for saving intermediates.

    Returns:
        dict with results and failure_db.
    """
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    failure_db = FailureDatabase()
    results = []
    perturbation_categories = list(PERTURBATIONS.keys())[:n_perturbations]

    for prog_idx, program in enumerate(test_programs):
        logger.info("Perturbation test: program %d/%d",
                     prog_idx + 1, len(test_programs))

        try:
            v1, f1 = program.evaluate()
            if len(v1) == 0:
                continue

            # Render clean drawing
            drawing_clean = render_drawing_sheet(v1, f1, views, image_size)

            # Interpret clean drawing as baseline
            spec_clean = _interpret_drawing_image(interpreter, drawing_clean,
                                                  views_hint=list(views))
            if spec_clean is None:
                continue

            prog_clean = drawing_to_cad(spec_clean)
            v_clean, f_clean = prog_clean.evaluate()

            for cat_name in perturbation_categories:
                variants = PERTURBATIONS[cat_name]
                for var_name, perturb_fn in variants:
                    label = f"prog{prog_idx}_{cat_name}_{var_name}"
                    logger.debug("  perturbation: %s", label)

                    try:
                        perturbed = perturb_fn(drawing_clean)

                        if output_dir:
                            Image.fromarray(perturbed).save(
                                os.path.join(output_dir, f"perturb_{label}.png"))

                        spec_p = _interpret_drawing_image(
                            interpreter, perturbed, views_hint=list(views))

                        if spec_p is None:
                            failure_db.add_failure(DrawingFailure(
                                drawing_image=perturbed,
                                expected_program=program.to_dict(),
                                actual_program=None,
                                failure_type="json_parse_error",
                                severity=1.0,
                                round_info={"strategy": "perturbation",
                                            "program_idx": prog_idx,
                                            "category": cat_name,
                                            "variant": var_name},
                            ))
                            results.append({
                                "label": label, "status": "interpret_failed"})
                            continue

                        prog_p = drawing_to_cad(spec_p)
                        v_p, f_p = prog_p.evaluate()

                        if len(v_p) == 0:
                            failure_db.add_failure(DrawingFailure(
                                drawing_image=perturbed,
                                expected_program=program.to_dict(),
                                actual_program=prog_p.to_dict(),
                                failure_type="empty_result",
                                severity=1.0,
                                round_info={"strategy": "perturbation",
                                            "program_idx": prog_idx,
                                            "category": cat_name,
                                            "variant": var_name},
                            ))
                            results.append({
                                "label": label, "status": "empty_result"})
                            continue

                        # Compare perturbed interpretation to original mesh
                        hd = hausdorff_distance(v1, v_p)
                        bbox_diag = float(np.linalg.norm(
                            v1.max(axis=0) - v1.min(axis=0)))
                        dist_norm = hd["mean_symmetric"] / max(bbox_diag, 1e-6)

                        # Also compare to clean interpretation
                        if len(v_clean) > 0:
                            hd_vs_clean = hausdorff_distance(v_clean, v_p)
                            drift = hd_vs_clean["mean_symmetric"] / max(bbox_diag, 1e-6)
                        else:
                            drift = dist_norm

                        ftype, severity = _classify_failure(
                            dist_norm, program, prog_p, spec_p)

                        entry = {
                            "label": label,
                            "dist_norm": float(dist_norm),
                            "drift_from_clean": float(drift),
                        }

                        if ftype is not None:
                            failure_db.add_failure(DrawingFailure(
                                drawing_image=perturbed,
                                expected_program=program.to_dict(),
                                actual_program=prog_p.to_dict(),
                                failure_type=ftype,
                                severity=severity,
                                details={"dist_norm": dist_norm,
                                         "drift": drift},
                                round_info={"strategy": "perturbation",
                                            "program_idx": prog_idx,
                                            "category": cat_name,
                                            "variant": var_name},
                            ))
                            entry["status"] = f"failure:{ftype}"
                        else:
                            entry["status"] = "success"

                        results.append(entry)

                    except Exception as e:
                        logger.error("Perturbation %s failed: %s", label, e)
                        results.append({"label": label,
                                        "status": f"error:{e}"})

        except Exception as e:
            logger.error("Program %d evaluation failed: %s", prog_idx, e)

    return {
        "strategy": "perturbation",
        "results": results,
        "failure_db": failure_db,
        "n_programs": len(test_programs),
        "n_perturbation_types": len(perturbation_categories),
        "n_successes": sum(1 for r in results if r.get("status") == "success"),
        "n_failures": sum(1 for r in results
                          if r.get("status", "").startswith("failure")),
    }


# ---------------------------------------------------------------------------
# Strategy C: Adversarial Mutation Loop
# ---------------------------------------------------------------------------

def _mutate_tiny_features(program, rng):
    """Scale holes down to 1% of body."""
    p = program.copy()
    for op in p.operations:
        if op.op_type == "subtract_cylinder" and op.enabled:
            op.params["radius"] = op.params.get("radius", 1.0) * 0.01
    p.invalidate_cache()
    return p, "tiny_features"


def _mutate_extreme_aspect(program, rng):
    """Stretch to extreme aspect ratio."""
    p = program.copy()
    axis = rng.choice([0, 1, 2])
    factor = rng.choice([10.0, 0.1])
    for op in p.operations:
        if op.enabled:
            if "dimensions" in op.params:
                dims = list(op.params["dimensions"])
                dims[axis] *= factor
                op.params["dimensions"] = dims
            elif "radius" in op.params and "height" in op.params:
                if axis == 2:
                    op.params["height"] = op.params["height"] * factor
                else:
                    op.params["radius"] = op.params["radius"] * factor
    p.invalidate_cache()
    return p, "extreme_aspect"


def _mutate_many_features(program, rng):
    """Add 8+ holes."""
    p = program.copy()
    body_r = 10.0
    body_h = 20.0
    for op in p.operations:
        if op.enabled and op.op_type in ("cylinder", "box"):
            if "radius" in op.params:
                body_r = op.params["radius"]
            if "height" in op.params:
                body_h = op.params["height"]
            break
    _add_holes_to_program(p, 8, body_r, body_h)
    return p, "many_features"


def _mutate_thin_walls(program, rng):
    """Subtract cylinder nearly as big as body."""
    p = program.copy()
    for op in p.operations:
        if op.enabled and op.op_type in ("cylinder", "sphere"):
            r = op.params.get("radius", 10.0)
            h = op.params.get("height", 20.0)
            center = list(op.params.get("center", [0, 0, h / 2]))
            p.operations.append(CadOp("subtract_cylinder", {
                "center": center,
                "axis": [0, 0, 1],
                "radius": r * 0.95,
                "height": h * 1.5,
            }))
            break
    p.invalidate_cache()
    return p, "thin_walls"


def _mutate_unusual_orientation(program, rng):
    """Rotate entire mesh 45 degrees."""
    p = program.copy()
    p.operations.append(CadOp("rotate", {
        "axis": [1, 1, 0],
        "angle_deg": 45,
        "center": [0, 0, 0],
    }))
    p.invalidate_cache()
    return p, "unusual_orientation"


MUTATION_FUNCTIONS = [
    _mutate_tiny_features,
    _mutate_extreme_aspect,
    _mutate_many_features,
    _mutate_thin_walls,
    _mutate_unusual_orientation,
]


def adversarial_mutation_loop(interpreter, base_programs,
                              n_mutations=20,
                              views=("front", "side", "top"),
                              image_size=512, output_dir=None):
    """Mutate known-good CadPrograms to find edge cases.

    Args:
        interpreter: DrawingInterpreter instance.
        base_programs: list of CadProgram objects.
        n_mutations: total mutations to generate.
        views: view types for rendering.
        image_size: render resolution.
        output_dir: optional directory for saving intermediates.

    Returns:
        dict with results and failure_db.
    """
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    failure_db = FailureDatabase()
    results = []
    rng = np.random.RandomState(42)

    mutation_count = 0
    for prog_idx, base_prog in enumerate(base_programs):
        if mutation_count >= n_mutations:
            break

        for mut_fn in MUTATION_FUNCTIONS:
            if mutation_count >= n_mutations:
                break

            label = f"prog{prog_idx}_{mut_fn.__name__}"
            logger.info("Mutation %d/%d: %s", mutation_count + 1,
                        n_mutations, label)

            try:
                mutated, mut_type = mut_fn(base_prog, rng)
                v1, f1 = mutated.evaluate()

                if len(v1) == 0:
                    results.append({"label": label, "mutation": mut_type,
                                    "status": "empty_source"})
                    mutation_count += 1
                    continue

                drawing = render_drawing_sheet(v1, f1, views, image_size)

                if output_dir:
                    Image.fromarray(drawing).save(
                        os.path.join(output_dir, f"mut_{label}.png"))

                spec = _interpret_drawing_image(interpreter, drawing,
                                                views_hint=list(views))

                if spec is None:
                    failure_db.add_failure(DrawingFailure(
                        drawing_image=drawing,
                        expected_program=mutated.to_dict(),
                        actual_program=None,
                        failure_type="json_parse_error",
                        severity=1.0,
                        round_info={"strategy": "mutation",
                                    "program_idx": prog_idx,
                                    "mutation": mut_type},
                    ))
                    results.append({"label": label, "mutation": mut_type,
                                    "status": "interpret_failed"})
                    mutation_count += 1
                    continue

                prog2 = drawing_to_cad(spec)
                v2, f2 = prog2.evaluate()

                if len(v2) == 0:
                    failure_db.add_failure(DrawingFailure(
                        drawing_image=drawing,
                        expected_program=mutated.to_dict(),
                        actual_program=prog2.to_dict(),
                        failure_type="empty_result",
                        severity=1.0,
                        round_info={"strategy": "mutation",
                                    "program_idx": prog_idx,
                                    "mutation": mut_type},
                    ))
                    results.append({"label": label, "mutation": mut_type,
                                    "status": "empty_result"})
                    mutation_count += 1
                    continue

                hd = hausdorff_distance(v1, v2)
                bbox_diag = float(np.linalg.norm(
                    v1.max(axis=0) - v1.min(axis=0)))
                dist_norm = hd["mean_symmetric"] / max(bbox_diag, 1e-6)

                ftype, severity = _classify_failure(
                    dist_norm, mutated, prog2, spec)

                entry = {"label": label, "mutation": mut_type,
                         "dist_norm": float(dist_norm)}

                if ftype is not None:
                    failure_db.add_failure(DrawingFailure(
                        drawing_image=drawing,
                        expected_program=mutated.to_dict(),
                        actual_program=prog2.to_dict(),
                        failure_type=ftype,
                        severity=severity,
                        details={"dist_norm": dist_norm},
                        round_info={"strategy": "mutation",
                                    "program_idx": prog_idx,
                                    "mutation": mut_type},
                    ))
                    entry["status"] = f"failure:{ftype}"
                else:
                    entry["status"] = "success"

                results.append(entry)

            except Exception as e:
                logger.error("Mutation %s failed: %s", label, e)
                results.append({"label": label, "mutation": mut_fn.__name__,
                                "status": f"error:{e}"})

            mutation_count += 1

    return {
        "strategy": "mutation",
        "results": results,
        "failure_db": failure_db,
        "n_mutations": mutation_count,
        "n_successes": sum(1 for r in results if r.get("status") == "success"),
        "n_failures": sum(1 for r in results
                          if r.get("status", "").startswith("failure")),
    }


# ---------------------------------------------------------------------------
# Top-level suite
# ---------------------------------------------------------------------------

def run_adversarial_suite(interpreter, output_dir,
                          generator_rounds=50,
                          perturbation_programs=None,
                          n_perturbations=8,
                          mutation_programs=None,
                          n_mutations=20,
                          views=("front", "side", "top"),
                          image_size=512):
    """Run all three adversarial strategies and produce a combined report.

    Args:
        interpreter: DrawingInterpreter instance (reused across all strategies).
        output_dir: root directory for all output.
        generator_rounds: rounds for strategy A.
        perturbation_programs: list of CadProgram for strategy B. If None,
            generates simple defaults.
        n_perturbations: perturbation categories for strategy B.
        mutation_programs: list of CadProgram for strategy C. If None,
            generates simple defaults.
        n_mutations: total mutations for strategy C.
        views: view types.
        image_size: render resolution.

    Returns:
        dict with combined report.
    """
    os.makedirs(output_dir, exist_ok=True)
    start_time = time.time()

    # Default test programs if not provided
    if perturbation_programs is None:
        perturbation_programs = [
            _make_box_program(),
            _make_cylinder_program(),
            _make_sphere_program(),
        ]
    if mutation_programs is None:
        mutation_programs = [
            _make_box_program(),
            _make_cylinder_program(),
        ]

    # Strategy A
    logger.info("=== Strategy A: Adversarial Generator Loop ===")
    gen_dir = os.path.join(output_dir, "generator")
    gen_result = adversarial_generator_loop(
        interpreter, n_rounds=generator_rounds,
        output_dir=gen_dir, views=views, image_size=image_size)

    # Strategy B
    logger.info("=== Strategy B: Adversarial Perturbation Loop ===")
    perturb_dir = os.path.join(output_dir, "perturbation")
    perturb_result = adversarial_perturbation_loop(
        interpreter, perturbation_programs,
        n_perturbations=n_perturbations,
        views=views, image_size=image_size, output_dir=perturb_dir)

    # Strategy C
    logger.info("=== Strategy C: Adversarial Mutation Loop ===")
    mut_dir = os.path.join(output_dir, "mutation")
    mut_result = adversarial_mutation_loop(
        interpreter, mutation_programs,
        n_mutations=n_mutations,
        views=views, image_size=image_size, output_dir=mut_dir)

    # Combine failure databases
    combined_db = FailureDatabase()
    for db in [gen_result["failure_db"],
               perturb_result["failure_db"],
               mut_result["failure_db"]]:
        for f in db.failures:
            combined_db.add_failure(f)

    elapsed = time.time() - start_time

    # Save combined failure DB
    combined_db.save(os.path.join(output_dir, "failures.json"))

    report = {
        "elapsed_seconds": elapsed,
        "generator": {
            "n_rounds": gen_result["n_rounds"],
            "n_successes": gen_result["n_successes"],
            "n_failures": gen_result["n_failures"],
            "stats_by_level": gen_result["stats_by_level"],
        },
        "perturbation": {
            "n_programs": perturb_result["n_programs"],
            "n_perturbation_types": perturb_result["n_perturbation_types"],
            "n_successes": perturb_result["n_successes"],
            "n_failures": perturb_result["n_failures"],
        },
        "mutation": {
            "n_mutations": mut_result["n_mutations"],
            "n_successes": mut_result["n_successes"],
            "n_failures": mut_result["n_failures"],
        },
        "combined_failures": combined_db.summary(),
    }

    # Save report
    with open(os.path.join(output_dir, "report.json"), "w") as fh:
        json.dump(report, fh, indent=2)

    logger.info("Adversarial suite completed in %.1fs. Total failures: %d",
                elapsed, combined_db.summary()["total"])

    return {
        "report": report,
        "generator_result": gen_result,
        "perturbation_result": perturb_result,
        "mutation_result": mut_result,
        "combined_failure_db": combined_db,
    }
