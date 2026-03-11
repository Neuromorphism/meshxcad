#!/usr/bin/env python3
"""Evaluate mesh-to-CAD reconstruction on cephalopod/organic models.

Runs reconstruct_cad on diverse non-axisymmetric organic shapes to identify
weaknesses in the reconstruction pipeline and measure improvement.

Usage:
    python -m dev_models.cephalopods.evaluate_cephalopods              # all models
    python -m dev_models.cephalopods.evaluate_cephalopods octopus_basic # one model
"""

import sys
import os
import time
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import numpy as np
from dev_models.cephalopods import load_model, list_models
from meshxcad.stl_io import write_binary_stl
from meshxcad.reconstruct import reconstruct_cad
from meshxcad.general_align import hausdorff_distance


OUT_DIR = os.path.join(os.path.dirname(__file__), "results")


def evaluate_single(name, target_v, target_f, out_dir=None):
    """Evaluate reconstruction for a single model.

    Returns dict with metrics.
    """
    t0 = time.time()

    result = reconstruct_cad(target_v, target_f)

    elapsed = time.time() - t0

    cad_v = result["cad_vertices"]
    cad_f = result["cad_faces"]

    if len(cad_v) > 0:
        hd = hausdorff_distance(cad_v, target_v)
        bbox_diag = float(np.linalg.norm(
            target_v.max(axis=0) - target_v.min(axis=0)))

        accuracy = max(0.0, 1.0 - hd["mean_symmetric"] / max(bbox_diag, 1e-12) * 5)
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

    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        write_binary_stl(os.path.join(out_dir, "target.stl"), target_v, target_f)
        if len(cad_v) > 0:
            write_binary_stl(os.path.join(out_dir, "cad_output.stl"), cad_v, cad_f)
        with open(os.path.join(out_dir, "metrics.json"), "w") as f:
            json.dump(metrics, f, indent=2)

    return metrics


def evaluate_all(names=None, save=True, skip_large=True):
    """Evaluate all (or specified) models.

    Args:
        names: list of model names (None = all)
        save: save results to disk
        skip_large: skip models with >100k vertices (very slow)

    Returns list of metric dicts.
    """
    if names is None:
        names = list_models()

    results = []
    print(f"Evaluating {len(names)} cephalopod/organic models...")
    print(f"{'#':>2} {'Name':25s} {'Type':10s} {'Accuracy':>8} {'Quality':>8} "
          f"{'Verts':>8} {'Time':>7}")
    print("-" * 80)

    for i, name in enumerate(names, 1):
        try:
            v, f = load_model(name, normalize=True)
        except Exception as e:
            print(f"{i:2d} {name:25s} LOAD ERROR: {e}")
            continue

        if skip_large and len(v) > 100000:
            print(f"{i:2d} {name:25s} SKIPPED (too large: {len(v)} verts)")
            continue

        out_dir = os.path.join(OUT_DIR, name) if save else None
        m = evaluate_single(name, v, f, out_dir=out_dir)
        results.append(m)

        status = "OK" if m["accuracy"] >= 0.99 else ("  " if m["accuracy"] >= 0.90 else "!!")
        print(f"{i:2d} {name:25s} {m['shape_type']:10s} {m['accuracy']:8.4f} "
              f"{m['quality']:8.3f} {m['cad_verts']:8d} {m['elapsed_sec']:7.1f}s {status}")

    # Summary
    if results:
        accs = [r["accuracy"] for r in results]
        print("\n" + "=" * 80)
        print(f"Models evaluated: {len(results)}")
        print(f"Mean accuracy: {np.mean(accs):.4f}")
        print(f"Min accuracy:  {np.min(accs):.4f} ({results[np.argmin(accs)]['name']})")
        print(f"Max accuracy:  {np.max(accs):.4f} ({results[np.argmax(accs)]['name']})")
        print(f">=99%: {sum(1 for a in accs if a >= 0.99)}/{len(accs)}")
        print(f">=95%: {sum(1 for a in accs if a >= 0.95)}/{len(accs)}")
        print(f">=90%: {sum(1 for a in accs if a >= 0.90)}/{len(accs)}")
        print(f">=80%: {sum(1 for a in accs if a >= 0.80)}/{len(accs)}")
        print(f">=50%: {sum(1 for a in accs if a >= 0.50)}/{len(accs)}")

        if save:
            summary_path = os.path.join(OUT_DIR, "summary.json")
            os.makedirs(OUT_DIR, exist_ok=True)
            with open(summary_path, "w") as f:
                json.dump({
                    "results": results,
                    "mean_accuracy": round(float(np.mean(accs)), 4),
                    "n_at_99": sum(1 for a in accs if a >= 0.99),
                    "n_at_90": sum(1 for a in accs if a >= 0.90),
                    "n_total": len(accs),
                }, f, indent=2)
            print(f"\nResults saved to {OUT_DIR}/")

    return results


if __name__ == "__main__":
    names = sys.argv[1:] if len(sys.argv) > 1 else None
    evaluate_all(names=names)
