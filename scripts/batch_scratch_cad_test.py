#!/usr/bin/env python3
"""Batch from-scratch CAD creation test for test models (trees, humanoids, etc.).

For each test model, runs the full optimise pipeline to fit a CadProgram
from scratch, then exports the target mesh, CAD output mesh, comparison
render, and metrics.
"""

import sys
import os
import time
import json
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from meshxcad.test_models import ALL_TEST_MODELS
from meshxcad.stl_io import write_binary_stl
from meshxcad.__main__ import optimise
from meshxcad.render import render_comparison

BASE_DIR = os.path.join(os.path.dirname(__file__), "..", "output_complex")

MAX_SWEEPS = 10
ROUNDS = 5
PATIENCE = 3


def run_single(name, target_v, target_f, out_dir):
    os.makedirs(out_dir, exist_ok=True)

    # Save target mesh
    write_binary_stl(os.path.join(out_dir, "target.stl"), target_v, target_f)

    # Run from-scratch optimisation
    t0 = time.time()
    result = optimise(
        target_v, target_f,
        max_sweeps=MAX_SWEEPS,
        rounds=ROUNDS,
        patience=PATIENCE,
        verbose=False,
    )
    elapsed = time.time() - t0

    # Extract CAD mesh
    prog_obj = result["_program_obj"]
    cad_v, cad_f = prog_obj.evaluate()

    if len(cad_v) > 0:
        write_binary_stl(os.path.join(out_dir, "cad_output.stl"), cad_v, cad_f)

    # Save program JSON
    with open(os.path.join(out_dir, "program.json"), "w") as f:
        json.dump({
            "program": result["program"],
            "initial": result["initial"],
            "final": result["final"],
            "converged": result["converged"],
            "total_sweeps": result["total_sweeps"],
            "elapsed_sec": result["elapsed_sec"],
        }, f, indent=2)

    # Render comparison
    try:
        panels = [(target_v, target_f)]
        labels = ["Target Mesh"]
        if len(cad_v) > 0:
            panels.append((cad_v, cad_f))
            labels.append("CAD Result")
        render_comparison(
            panels, labels,
            os.path.join(out_dir, "comparison.png"),
            title=name.replace("_", " ").title(),
        )
    except Exception as e:
        print(f"    Render failed: {e}")

    final = result["final"]
    return {
        "name": name,
        "accuracy": final["accuracy"],
        "elegance": final["elegance"],
        "cad_score": final["cad_score"],
        "n_ops": final["n_ops"],
        "program": final["program_summary"],
        "sweeps": result["total_sweeps"],
        "converged": result["converged"],
        "time": elapsed,
        "target_verts": len(target_v),
        "cad_verts": len(cad_v) if len(cad_v) > 0 else 0,
    }


def main():
    os.makedirs(BASE_DIR, exist_ok=True)
    print("=" * 70)
    print("TEST MODELS — FROM-SCRATCH CAD CREATION")
    print("=" * 70)

    models = list(ALL_TEST_MODELS.keys())
    results = []

    for idx, name in enumerate(models, 1):
        print(f"\n[{idx}/{len(models)}] {name}...")
        fn = ALL_TEST_MODELS[name]
        target_v, target_f = fn()
        print(f"    mesh: {len(target_v)}v / {len(target_f)}f")

        r = run_single(name, target_v, target_f, os.path.join(BASE_DIR, name))
        results.append(r)

        print(f"    {r['program']}  acc={r['accuracy']:.3f}  "
              f"eleg={r['elegance']:.3f}  cad={r['cad_score']:.3f}  "
              f"({r['time']:.1f}s, {r['sweeps']} sweeps)")

    # Summary table
    print("\n" + "=" * 70)
    print(f"{'Model':<22} {'Accuracy':>8} {'Elegance':>8} {'CAD':>6} "
          f"{'Ops':>4} {'Program':<20}")
    print("-" * 70)

    total_acc = 0
    total_eleg = 0
    for r in results:
        total_acc += r["accuracy"]
        total_eleg += r["elegance"]
        print(f"{r['name']:<22} {r['accuracy']:>8.3f} {r['elegance']:>8.3f} "
              f"{r['cad_score']:>6.3f} {r['n_ops']:>4} {r['program']:<20}")

    avg_acc = total_acc / len(results)
    avg_eleg = total_eleg / len(results)
    print("-" * 70)
    print(f"{'AVERAGE':<22} {avg_acc:>8.3f} {avg_eleg:>8.3f}")
    print(f"\nTotal models: {len(results)}")

    # Write summary
    with open(os.path.join(BASE_DIR, "scratch_cad_summary.txt"), "w") as f:
        f.write("Model,Accuracy,Elegance,CAD_Score,N_Ops,Program,Sweeps,"
                "Converged,Time_s,Target_Verts,CAD_Verts\n")
        for r in results:
            f.write(f"{r['name']},{r['accuracy']:.4f},{r['elegance']:.4f},"
                    f"{r['cad_score']:.4f},{r['n_ops']},{r['program']},"
                    f"{r['sweeps']},{r['converged']},{r['time']:.1f},"
                    f"{r['target_verts']},{r['cad_verts']}\n")


if __name__ == "__main__":
    main()
