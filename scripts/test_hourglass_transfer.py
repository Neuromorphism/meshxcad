#!/usr/bin/env python3
"""End-to-end test: transfer ornate hourglass detail onto simple hourglass.

This script:
1. Generates simple and ornate hourglass meshes (synthetic, no FreeCAD needed)
2. Exports both as STL files
3. Runs detail transfer: ornate mesh detail → simple mesh
4. Renders before/after comparison
5. Computes quantitative metrics (distance to objective)
"""

import sys
import os
import time
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from meshxcad.hourglass_synthetic import (
    make_simple_hourglass_mesh,
    make_ornate_hourglass_mesh,
)
from meshxcad.stl_io import write_binary_stl, read_binary_stl
from meshxcad.detail_transfer import transfer_mesh_detail_to_mesh
from meshxcad.alignment import find_correspondences
from meshxcad.render import render_mesh, render_comparison

output_dir = os.path.join(os.path.dirname(__file__), "..", "hourglass")
stl_dir = os.path.join(output_dir, "stl")
render_dir = os.path.join(output_dir, "renders")
os.makedirs(stl_dir, exist_ok=True)
os.makedirs(render_dir, exist_ok=True)


def main():
    print("=" * 60)
    print("HOURGLASS DETAIL TRANSFER TEST")
    print("=" * 60)

    # ----------------------------------------------------------------
    # Step 1: Generate meshes
    # ----------------------------------------------------------------
    print("\n[1/5] Generating synthetic hourglass meshes...")
    simple_v, simple_f = make_simple_hourglass_mesh(n_angular=48)
    ornate_v, ornate_f = make_ornate_hourglass_mesh(n_angular=48)
    print(f"  Simple: {len(simple_v)} verts, {len(simple_f)} faces")
    print(f"  Ornate: {len(ornate_v)} verts, {len(ornate_f)} faces")

    # ----------------------------------------------------------------
    # Step 2: Export as STL
    # ----------------------------------------------------------------
    print("\n[2/5] Exporting STL files...")
    simple_stl = os.path.join(stl_dir, "simple_hourglass.stl")
    ornate_stl = os.path.join(stl_dir, "ornate_hourglass.stl")
    write_binary_stl(simple_stl, simple_v, simple_f)
    write_binary_stl(ornate_stl, ornate_v, ornate_f)
    print(f"  Simple STL: {os.path.getsize(simple_stl):,} bytes")
    print(f"  Ornate STL: {os.path.getsize(ornate_stl):,} bytes")

    # Verify round-trip
    simple_v2, simple_f2 = read_binary_stl(simple_stl)
    ornate_v2, ornate_f2 = read_binary_stl(ornate_stl)
    print(f"  STL round-trip: simple={len(simple_v2)} verts, ornate={len(ornate_v2)} verts")

    # ----------------------------------------------------------------
    # Step 3: Detail transfer — ornate mesh detail → simple mesh
    # ----------------------------------------------------------------
    print("\n[3/5] Transferring ornate detail onto simple mesh...")
    t0 = time.time()
    result_v = transfer_mesh_detail_to_mesh(
        simple_v, simple_f, ornate_v, ornate_f
    )
    elapsed = time.time() - t0
    print(f"  Transfer completed in {elapsed:.2f}s")

    # Save result
    result_stl = os.path.join(stl_dir, "transferred_hourglass.stl")
    write_binary_stl(result_stl, result_v, simple_f)
    print(f"  Result STL: {os.path.getsize(result_stl):,} bytes")

    # ----------------------------------------------------------------
    # Step 4: Quantitative evaluation
    # ----------------------------------------------------------------
    print("\n[4/5] Evaluating transfer quality...")

    # Distance from result to ornate (the objective)
    _, _, dists_to_ornate = find_correspondences(result_v, ornate_v)
    mean_dist = np.mean(dists_to_ornate)
    max_dist = np.max(dists_to_ornate)
    median_dist = np.median(dists_to_ornate)

    # Distance from simple (input) to ornate (baseline — how far we had to go)
    _, _, baseline_dists = find_correspondences(simple_v, ornate_v)
    baseline_mean = np.mean(baseline_dists)

    # Improvement ratio
    improvement = (1 - mean_dist / baseline_mean) * 100

    print(f"  Baseline (simple→ornate) mean distance:  {baseline_mean:.3f} mm")
    print(f"  Result (transferred→ornate) mean distance: {mean_dist:.3f} mm")
    print(f"  Result median distance:                     {median_dist:.3f} mm")
    print(f"  Result max distance:                        {max_dist:.3f} mm")
    print(f"  Improvement: {improvement:.1f}%")

    # Check if transfer actually helped
    if mean_dist < baseline_mean:
        print("  ✓ Transfer moved simple mesh CLOSER to ornate mesh")
    else:
        print("  ✗ Transfer did not improve alignment — needs investigation")

    # ----------------------------------------------------------------
    # Step 5: Render comparison
    # ----------------------------------------------------------------
    print("\n[5/5] Rendering comparison images...")

    render_comparison(
        [(simple_v, simple_f), (ornate_v, ornate_f), (result_v, simple_f)],
        ["Simple (Input)", "Ornate (Objective)", "Transfer Result"],
        os.path.join(render_dir, "hourglass_transfer_comparison.png"),
        title="Hourglass Detail Transfer: Simple + Ornate Detail → Result",
    )

    # Also render the result on its own
    render_mesh(
        result_v, simple_f,
        os.path.join(render_dir, "transferred_hourglass.png"),
        title="Transfer Result: Simple Mesh + Ornate Detail",
    )

    # Render front-view overlay comparison (before/after)
    render_comparison(
        [(simple_v, simple_f), (result_v, simple_f)],
        ["Before (Simple)", "After (Detail Transfer)"],
        os.path.join(render_dir, "hourglass_before_after.png"),
        title="Before vs After Detail Transfer",
    )

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Input:      simple hourglass ({len(simple_v)} verts)")
    print(f"  Detail src: ornate hourglass ({len(ornate_v)} verts)")
    print(f"  Output:     transferred mesh  ({len(result_v)} verts)")
    print(f"  Mean distance to objective: {mean_dist:.3f} mm")
    print(f"  Improvement over baseline:  {improvement:.1f}%")
    print(f"\n  STL files in:   {stl_dir}")
    print(f"  Renders in:     {render_dir}")
    print("=" * 60)

    return mean_dist < baseline_mean


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
