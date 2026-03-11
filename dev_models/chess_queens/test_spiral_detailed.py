#!/usr/bin/env python3
"""Detailed accuracy test comparing reconstruction strategies on spiral queen."""

import sys
import os
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import numpy as np
from dev_models.chess_queens import load_queen
from meshxcad.general_align import hausdorff_distance

# Patch to expose internal strategies
import meshxcad.reconstruct as _rec


def accuracy_from_hd(hd, bbox_diag):
    return max(0.0, 1.0 - hd["mean_symmetric"] / bbox_diag * 5)


def test_pure_revolve(v, f, bbox_diag):
    """Test pure revolve profile (no surface asymmetry)."""
    r_xy = np.sqrt(v[:, 0]**2 + v[:, 1]**2)
    z_min = float(v[:, 2].min())
    z_max = float(v[:, 2].max())
    z_range = z_max - z_min

    fn = _rec._build_revolve_profile(v, r_xy, z_min, z_max, z_range, percentile=50)

    # Project: each vertex gets r(z) only, no angular information
    cv = v.copy()
    for i in range(len(v)):
        x, y, z = v[i]
        r_actual = np.sqrt(x*x + y*y)
        if r_actual < 1e-8:
            continue
        r_prof = fn(0.0, z)  # pure revolve ignores theta
        scale = r_prof / r_actual
        cv[i, 0] = x * scale
        cv[i, 1] = y * scale

    hd = hausdorff_distance(cv, v)
    return accuracy_from_hd(hd, bbox_diag), hd["mean_symmetric"]


def test_surface_profile(v, f, bbox_diag):
    """Test 2D surface profile (captures angular variation)."""
    r_xy = np.sqrt(v[:, 0]**2 + v[:, 1]**2)
    a_xy = np.arctan2(v[:, 1], v[:, 0])
    z_min = float(v[:, 2].min())
    z_max = float(v[:, 2].max())
    z_range = z_max - z_min

    fn = _rec._build_surface_profile(v, r_xy, a_xy, z_min, z_max, z_range)

    cv = v.copy()
    for i in range(len(v)):
        x, y, z = v[i]
        r_actual = np.sqrt(x*x + y*y)
        if r_actual < 1e-8:
            continue
        theta = np.arctan2(y, x)
        r_prof = fn(theta, z)
        scale = r_prof / r_actual
        cv[i, 0] = x * scale
        cv[i, 1] = y * scale

    hd = hausdorff_distance(cv, v)
    return accuracy_from_hd(hd, bbox_diag), hd["mean_symmetric"]


def run():
    print("=" * 60)
    print("Spiral Queen — Strategy Comparison")
    print("=" * 60)

    v, f = load_queen("spiral", normalize=True)
    bbox_diag = float(np.linalg.norm(v.max(axis=0) - v.min(axis=0)))
    print(f"vertices={len(v)}, faces={len(f)}, bbox_diag={bbox_diag:.3f}\n")

    print("Strategy 1: Pure revolve (r depends only on z)")
    t0 = time.time()
    acc_pr, ms_pr = test_pure_revolve(v, f, bbox_diag)
    print(f"  accuracy={acc_pr:.4f}  mean_sym={ms_pr:.4f}  [{time.time()-t0:.2f}s]")

    print("Strategy 2: Surface profile (r depends on theta AND z)")
    t0 = time.time()
    acc_sp, ms_sp = test_surface_profile(v, f, bbox_diag)
    print(f"  accuracy={acc_sp:.4f}  mean_sym={ms_sp:.4f}  [{time.time()-t0:.2f}s]")

    print("\nStrategy 3: Full reconstruct_cad pipeline")
    from meshxcad.reconstruct import reconstruct_cad
    t0 = time.time()
    result = reconstruct_cad(v, f)
    cad_v = result["cad_vertices"]
    hd = hausdorff_distance(cad_v, v)
    acc_full = accuracy_from_hd(hd, bbox_diag)
    print(f"  accuracy={acc_full:.4f}  mean_sym={hd['mean_symmetric']:.4f}  [{time.time()-t0:.2f}s]")
    print(f"  shape_type={result['shape_type']}, quality={result.get('quality', 0):.4f}")

    print("\n--- Summary ---")
    print(f"  pure revolve:  {acc_pr:.4f} {'OK' if acc_pr >= 0.99 else 'FAIL'}")
    print(f"  surface prof:  {acc_sp:.4f} {'OK' if acc_sp >= 0.99 else 'FAIL'}")
    print(f"  full pipeline: {acc_full:.4f} {'OK' if acc_full >= 0.99 else 'FAIL'}")
    print()
    return acc_full


if __name__ == "__main__":
    acc = run()
    sys.exit(0 if acc >= 0.99 else 1)
