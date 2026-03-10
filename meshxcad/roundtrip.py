"""CAD ↔ Drawing roundtrip verification.

Tests fidelity by going CAD₁ → Drawing₁ → interpret → CAD₂ → Drawing₂
and comparing at each stage.
"""

import logging
import numpy as np

from .cad_program import CadProgram
from .drawing import render_drawing_sheet
from .drawing_compare import compare_drawings
from .drawing_to_cad import drawing_to_cad
from .general_align import hausdorff_distance

logger = logging.getLogger(__name__)


def roundtrip_test(program, interpreter, views=("front", "side", "top"),
                   image_size=512, optimize_sweeps=0) -> dict:
    """Full roundtrip: CAD₁ → Drawing₁ → CAD₂ → Drawing₂.

    Args:
        program: CadProgram to test.
        interpreter: DrawingInterpreter instance.
        views: view types for rendering.
        image_size: render resolution.
        optimize_sweeps: if >0, optimise CAD₂ against Drawing₁.

    Returns dict with roundtrip metrics.
    """
    # Step 1: Evaluate program → mesh₁
    v1, f1 = program.evaluate()
    if len(v1) == 0:
        return {"error": "program produced empty mesh"}

    # Step 2: Render to drawing₁
    drawing1 = render_drawing_sheet(v1, f1, views, image_size)

    # Step 3: Interpret drawing₁ → spec₂
    # Save drawing to temp file for the interpreter
    import tempfile, os
    from PIL import Image
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        tmp_path = tmp.name
        Image.fromarray(drawing1).save(tmp_path)

    try:
        spec2 = interpreter.interpret_drawing(tmp_path,
                                               views_hint=list(views))
    finally:
        os.unlink(tmp_path)

    # Step 4: Build CAD₂ from spec
    program2 = drawing_to_cad(spec2)

    # Step 5: Evaluate CAD₂ → mesh₂
    v2, f2 = program2.evaluate()
    if len(v2) == 0:
        return {
            "error": "reconstructed program produced empty mesh",
            "drawing_1": drawing1,
            "spec_2": spec2.to_dict(),
        }

    # Step 6: Render drawing₂
    drawing2 = render_drawing_sheet(v2, f2, views, image_size)

    # Step 7: Compare
    # Mesh comparison
    mesh_dist = hausdorff_distance(v1, v2)
    bbox_diag = np.linalg.norm(v1.max(axis=0) - v1.min(axis=0))
    mesh_norm = mesh_dist["mean_symmetric"] / max(bbox_diag, 1e-6)

    # Drawing comparison
    gray1 = np.mean(drawing1, axis=2).astype(np.uint8) if drawing1.ndim == 3 else drawing1
    gray2 = np.mean(drawing2, axis=2).astype(np.uint8) if drawing2.ndim == 3 else drawing2
    draw_metrics = compare_drawings(gray1, gray2)

    # Program comparison
    ops1 = [op.op_type for op in program.operations if op.enabled]
    ops2 = [op.op_type for op in program2.operations if op.enabled]
    op_match = _op_type_similarity(ops1, ops2)

    # Combined score
    roundtrip_score = (
        0.4 * max(0, 1.0 - mesh_norm) +
        0.3 * draw_metrics["pixel_iou"] +
        0.2 * (draw_metrics["edge_precision"] + draw_metrics["edge_recall"]) / 2 +
        0.1 * op_match
    )

    return {
        "mesh_hausdorff": mesh_dist["hausdorff"],
        "mesh_mean_symmetric": mesh_dist["mean_symmetric"],
        "mesh_hausdorff_normalized": float(mesh_norm),
        "drawing_chamfer": draw_metrics["chamfer_distance"],
        "drawing_iou": draw_metrics["pixel_iou"],
        "drawing_precision": draw_metrics["edge_precision"],
        "drawing_recall": draw_metrics["edge_recall"],
        "program_op_match": op_match,
        "roundtrip_score": float(roundtrip_score),
        "drawing_1": drawing1,
        "drawing_2": drawing2,
        "program_1_ops": ops1,
        "program_2_ops": ops2,
        "spec_2": spec2.to_dict(),
    }


def batch_roundtrip(programs, interpreter, labels=None,
                    views=("front", "side", "top"), image_size=512) -> dict:
    """Run roundtrip on multiple objects, report statistics.

    Args:
        programs: list of (label, CadProgram) or just CadProgram objects.
        interpreter: DrawingInterpreter instance.
        labels: optional list of names.

    Returns aggregated results.
    """
    if labels is None:
        labels = [f"object_{i}" for i in range(len(programs))]

    results = []
    for label, prog in zip(labels, programs):
        logger.info("Roundtrip: %s", label)
        r = roundtrip_test(prog, interpreter, views, image_size)
        r["label"] = label
        results.append(r)
        if "error" not in r:
            logger.info("  score=%.3f mesh_dist=%.4f draw_iou=%.3f",
                         r["roundtrip_score"],
                         r["mesh_hausdorff_normalized"],
                         r["drawing_iou"])
        else:
            logger.warning("  FAILED: %s", r["error"])

    # Aggregate
    valid = [r for r in results if "error" not in r]
    if valid:
        scores = [r["roundtrip_score"] for r in valid]
        mesh_dists = [r["mesh_hausdorff_normalized"] for r in valid]
        ious = [r["drawing_iou"] for r in valid]
        worst_idx = int(np.argmin(scores))
        best_idx = int(np.argmax(scores))
    else:
        scores = mesh_dists = ious = []
        worst_idx = best_idx = 0

    return {
        "results": results,
        "n_tested": len(results),
        "n_succeeded": len(valid),
        "mean_roundtrip_score": float(np.mean(scores)) if scores else 0.0,
        "mean_mesh_distance": float(np.mean(mesh_dists)) if mesh_dists else 0.0,
        "mean_drawing_iou": float(np.mean(ious)) if ious else 0.0,
        "worst_object": valid[worst_idx]["label"] if valid else None,
        "best_object": valid[best_idx]["label"] if valid else None,
    }


def _op_type_similarity(ops1, ops2):
    """Fraction of op types that match between two programs."""
    if not ops1 and not ops2:
        return 1.0
    if not ops1 or not ops2:
        return 0.0
    set1 = set(ops1)
    set2 = set(ops2)
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    return intersection / union if union > 0 else 1.0
