#!/usr/bin/env python3
"""Evaluate mesh-to-CAD reconstruction quality on all chess king models.

Runs the full reconstruct_cad pipeline on each king, measures quality,
and reports results.  Used to develop and refine CAD strategies for
organic/ornate shapes.

Usage:
    python -m dev_models.chess_kings.evaluate_kings          # all kings
    python -m dev_models.chess_kings.evaluate_kings staunton # one king
"""

import sys
import os
import time
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import numpy as np
from dev_models.chess_kings import load_king, load_all_kings, list_kings
from meshxcad.stl_io import write_binary_stl
from meshxcad.reconstruct import reconstruct_cad, _measure_quality
from meshxcad.general_align import hausdorff_distance


OUT_DIR = os.path.join(os.path.dirname(__file__), "results")


def evaluate_single(name, target_v, target_f, out_dir=None):
    """Evaluate reconstruction for a single king.

    Returns dict with metrics.
    """
    t0 = time.time()

    # Run reconstruction
    result = reconstruct_cad(target_v, target_f)

    elapsed = time.time() - t0

    cad_v = result["cad_vertices"]
    cad_f = result["cad_faces"]

    # Detailed distance metrics
    if len(cad_v) > 0:
        hd = hausdorff_distance(cad_v, target_v)
        bbox_diag = float(np.linalg.norm(
            target_v.max(axis=0) - target_v.min(axis=0)))

        # Accuracy: 1 - normalized mean symmetric distance
        accuracy = max(0.0, 1.0 - hd["mean_symmetric"] / max(bbox_diag, 1e-12) * 5)

        # Also compute forward/backward coverage
        forward_coverage = max(0.0, 1.0 - hd["mean_a_to_b"] / max(bbox_diag, 1e-12) * 5)
        backward_coverage = max(0.0, 1.0 - hd["mean_b_to_a"] / max(bbox_diag, 1e-12) * 5)
    else:
        accuracy = 0.0
        forward_coverage = 0.0
        backward_coverage = 0.0
        hd = {}

    metrics = {
        "name": name,
        "shape_type": result["shape_type"],
        "accuracy": round(accuracy, 4),
        "quality": result.get("quality", 0.0),
        "forward_coverage": round(forward_coverage, 4),
        "backward_coverage": round(backward_coverage, 4),
        "target_verts": len(target_v),
        "target_faces": len(target_f),
        "cad_verts": len(cad_v),
        "cad_faces": len(cad_f),
        "elapsed_sec": round(elapsed, 2),
        "hausdorff": round(hd.get("hausdorff", 999.0), 4),
        "mean_symmetric": round(hd.get("mean_symmetric", 999.0), 4),
    }

    # Save outputs if requested
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        write_binary_stl(os.path.join(out_dir, "target.stl"), target_v, target_f)
        if len(cad_v) > 0:
            write_binary_stl(os.path.join(out_dir, "cad_output.stl"), cad_v, cad_f)
        with open(os.path.join(out_dir, "metrics.json"), "w") as f:
            json.dump(metrics, f, indent=2)

    return metrics


def evaluate_all(names=None, save=True):
    """Evaluate all (or specified) kings.

    Returns list of metric dicts.
    """
    if names is None:
        names = list_kings()

    results = []
    print(f"Evaluating {len(names)} chess king models...")
    print(f"{'#':>2} {'Name':20s} {'Type':10s} {'Accuracy':>8} {'Quality':>8} "
          f"{'Verts':>6} {'Time':>6}")
    print("-" * 70)

    for i, name in enumerate(names, 1):
        try:
            v, f = load_king(name, normalize=True)
        except Exception as e:
            print(f"{i:2d} {name:20s} LOAD ERROR: {e}")
            continue

        out_dir = os.path.join(OUT_DIR, name) if save else None
        m = evaluate_single(name, v, f, out_dir=out_dir)
        results.append(m)

        status = "OK" if m["accuracy"] >= 0.99 else "  "
        print(f"{i:2d} {name:20s} {m['shape_type']:10s} {m['accuracy']:8.4f} "
              f"{m['quality']:8.3f} {m['cad_verts']:6d} {m['elapsed_sec']:6.1f}s {status}")

    # Summary
    if results:
        accs = [r["accuracy"] for r in results]
        print("\n" + "=" * 70)
        print(f"Mean accuracy: {np.mean(accs):.4f}")
        print(f"Min accuracy:  {np.min(accs):.4f} ({results[np.argmin(accs)]['name']})")
        print(f"Max accuracy:  {np.max(accs):.4f} ({results[np.argmax(accs)]['name']})")
        print(f">=99%: {sum(1 for a in accs if a >= 0.99)}/{len(accs)}")
        print(f">=95%: {sum(1 for a in accs if a >= 0.95)}/{len(accs)}")
        print(f">=90%: {sum(1 for a in accs if a >= 0.90)}/{len(accs)}")

        if save:
            summary_path = os.path.join(OUT_DIR, "summary.json")
            os.makedirs(OUT_DIR, exist_ok=True)
            with open(summary_path, "w") as f:
                json.dump({
                    "results": results,
                    "mean_accuracy": round(float(np.mean(accs)), 4),
                    "n_at_99": sum(1 for a in accs if a >= 0.99),
                    "n_total": len(accs),
                }, f, indent=2)
            print(f"\nResults saved to {OUT_DIR}/")

    return results


if __name__ == "__main__":
    names = sys.argv[1:] if len(sys.argv) > 1 else None
    evaluate_all(names=names)
