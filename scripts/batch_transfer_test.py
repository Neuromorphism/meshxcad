#!/usr/bin/env python3
"""Batch detail transfer test across all 19 decorative objects + hourglass.

For each object:
1. Generate simple (plain) and ornate (featured) meshes
2. Export both as STL
3. Transfer ornate detail onto simple mesh
4. Measure distance improvement
5. Render 3-way comparison (simple, ornate, result)
6. Generate summary report
"""

import sys
import os
import time
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from meshxcad.objects.catalog import list_objects, make_simple, make_ornate
from meshxcad.hourglass_synthetic import (
    make_simple_hourglass_mesh,
    make_ornate_hourglass_mesh,
)
from meshxcad.stl_io import write_binary_stl
from meshxcad.detail_transfer import transfer_mesh_detail_to_mesh
from meshxcad.alignment import find_correspondences
from meshxcad.render import render_comparison

BASE_DIR = os.path.join(os.path.dirname(__file__), "..", "output")


def run_single_test(name, simple_v, simple_f, ornate_v, ornate_f, output_dir):
    """Run detail transfer test on a single object.

    Returns dict with metrics.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Export STL
    write_binary_stl(os.path.join(output_dir, "simple.stl"), simple_v, simple_f)
    write_binary_stl(os.path.join(output_dir, "ornate.stl"), ornate_v, ornate_f)

    # Transfer
    t0 = time.time()
    result_v = transfer_mesh_detail_to_mesh(simple_v, simple_f, ornate_v, ornate_f)
    elapsed = time.time() - t0

    # Save result
    write_binary_stl(os.path.join(output_dir, "transferred.stl"), result_v, simple_f)

    # Metrics
    _, _, baseline_dists = find_correspondences(simple_v, ornate_v)
    _, _, result_dists = find_correspondences(result_v, ornate_v)

    baseline_mean = float(np.mean(baseline_dists))
    result_mean = float(np.mean(result_dists))
    improvement = (1 - result_mean / baseline_mean) * 100 if baseline_mean > 0 else 0

    # Render comparison
    try:
        render_comparison(
            [(simple_v, simple_f), (ornate_v, ornate_f), (result_v, simple_f)],
            ["Simple", "Ornate (Target)", "Transfer Result"],
            os.path.join(output_dir, "comparison.png"),
            title=name.replace("_", " ").title(),
        )
    except Exception as e:
        print(f"    Render failed: {e}")

    return {
        "name": name,
        "simple_verts": len(simple_v),
        "ornate_verts": len(ornate_v),
        "baseline_mean": baseline_mean,
        "result_mean": result_mean,
        "improvement_pct": improvement,
        "time_s": elapsed,
        "success": result_mean < baseline_mean,
    }


def main():
    os.makedirs(BASE_DIR, exist_ok=True)

    print("=" * 70)
    print("BATCH DETAIL TRANSFER TEST — 20 DECORATIVE OBJECTS")
    print("=" * 70)

    results = []

    # All 19 catalog objects
    objects = list_objects()
    total = len(objects) + 1  # +1 for hourglass

    for idx, name in enumerate(objects, 1):
        print(f"\n[{idx}/{total}] {name}...")
        simple_v, simple_f = make_simple(name)
        ornate_v, ornate_f = make_ornate(name)
        output_dir = os.path.join(BASE_DIR, name)
        r = run_single_test(name, simple_v, simple_f, ornate_v, ornate_f, output_dir)
        results.append(r)
        status = "OK" if r["success"] else "FAIL"
        print(f"    {status}: {r['improvement_pct']:.1f}% improvement "
              f"({r['baseline_mean']:.2f} -> {r['result_mean']:.2f}) "
              f"in {r['time_s']:.2f}s")

    # Hourglass
    print(f"\n[{total}/{total}] hourglass...")
    sv, sf = make_simple_hourglass_mesh(n_angular=48)
    ov, of_ = make_ornate_hourglass_mesh(n_angular=48)
    output_dir = os.path.join(BASE_DIR, "hourglass")
    r = run_single_test("hourglass", sv, sf, ov, of_, output_dir)
    results.append(r)
    status = "OK" if r["success"] else "FAIL"
    print(f"    {status}: {r['improvement_pct']:.1f}% improvement "
          f"({r['baseline_mean']:.2f} -> {r['result_mean']:.2f}) "
          f"in {r['time_s']:.2f}s")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"{'Object':<22} {'Baseline':>10} {'Result':>10} {'Improv%':>10} {'Status':>8}")
    print("-" * 62)

    successes = 0
    total_improvement = 0
    for r in results:
        status = "OK" if r["success"] else "FAIL"
        if r["success"]:
            successes += 1
        total_improvement += r["improvement_pct"]
        print(f"{r['name']:<22} {r['baseline_mean']:>10.2f} {r['result_mean']:>10.2f} "
              f"{r['improvement_pct']:>9.1f}% {status:>8}")

    avg_improvement = total_improvement / len(results)
    print("-" * 62)
    print(f"{'AVERAGE':<22} {'':>10} {'':>10} {avg_improvement:>9.1f}%")
    print(f"\nPass rate: {successes}/{len(results)} objects improved")
    print(f"Output directory: {BASE_DIR}")

    # Save summary as text
    summary_path = os.path.join(BASE_DIR, "summary.txt")
    with open(summary_path, "w") as f:
        f.write("Object,Baseline,Result,Improvement%,Status\n")
        for r in results:
            f.write(f"{r['name']},{r['baseline_mean']:.4f},{r['result_mean']:.4f},"
                    f"{r['improvement_pct']:.2f},{'PASS' if r['success'] else 'FAIL'}\n")
    print(f"Summary saved: {summary_path}")

    return successes == len(results)


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
