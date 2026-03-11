#!/usr/bin/env python3
"""Quick accuracy test for the spiral queen reconstruction.

Runs the current reconstruct_cad pipeline and prints accuracy metrics.
Used for iterative development.

Usage:
    python3 dev_models/chess_queens/test_spiral_accuracy.py
"""

import sys
import os
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import numpy as np
from dev_models.chess_queens import load_queen
from meshxcad.reconstruct import reconstruct_cad
from meshxcad.general_align import hausdorff_distance


def run_test(verbose=True):
    print("Loading spiral_queen...")
    v, f = load_queen("spiral", normalize=True)
    print(f"  vertices={len(v)}, faces={len(f)}")

    bbox_diag = float(np.linalg.norm(v.max(axis=0) - v.min(axis=0)))
    print(f"  bbox_diag={bbox_diag:.3f}")

    print("Running reconstruct_cad...")
    t0 = time.time()
    result = reconstruct_cad(v, f)
    elapsed = time.time() - t0

    cad_v = result["cad_vertices"]
    cad_f = result["cad_faces"]
    shape_type = result["shape_type"]
    quality = result.get("quality", 0.0)

    print(f"  shape_type={shape_type}, quality={quality:.4f}, elapsed={elapsed:.2f}s")

    if len(cad_v) > 0:
        hd = hausdorff_distance(cad_v, v)
        accuracy = max(0.0, 1.0 - hd["mean_symmetric"] / bbox_diag * 5)
        fwd = max(0.0, 1.0 - hd["mean_a_to_b"] / bbox_diag * 5)
        bwd = max(0.0, 1.0 - hd["mean_b_to_a"] / bbox_diag * 5)
        print(f"\n  mean_symmetric={hd['mean_symmetric']:.4f}")
        print(f"  accuracy       ={accuracy:.4f}  {'OK >=99%' if accuracy >= 0.99 else 'NEEDS IMPROVEMENT'}")
        print(f"  fwd_coverage   ={fwd:.4f}")
        print(f"  bwd_coverage   ={bwd:.4f}")
        print(f"  hausdorff_max  ={hd['hausdorff']:.4f}")
        return accuracy
    else:
        print("  ERROR: empty CAD output")
        return 0.0


if __name__ == "__main__":
    acc = run_test()
    sys.exit(0 if acc >= 0.99 else 1)
