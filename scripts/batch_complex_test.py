#!/usr/bin/env python3
"""Batch transfer test for 10 complex (non-revolve) objects."""

import sys
import os
import time
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from meshxcad.objects.complex_catalog import list_complex_objects, make_complex_simple, make_complex_ornate
from meshxcad.stl_io import write_binary_stl
from meshxcad.detail_transfer import transfer_mesh_detail_to_mesh
from meshxcad.alignment import find_correspondences
from meshxcad.render import render_comparison

BASE_DIR = os.path.join(os.path.dirname(__file__), "..", "output_complex")


def run_single(name, sv, sf, ov, of_, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    write_binary_stl(os.path.join(out_dir, "simple.stl"), sv, sf)
    write_binary_stl(os.path.join(out_dir, "ornate.stl"), ov, of_)

    t0 = time.time()
    rv = transfer_mesh_detail_to_mesh(sv, sf, ov, of_)
    elapsed = time.time() - t0

    write_binary_stl(os.path.join(out_dir, "transferred.stl"), rv, sf)

    _, _, bd = find_correspondences(sv, ov)
    _, _, rd = find_correspondences(rv, ov)
    bm, rm = float(np.mean(bd)), float(np.mean(rd))
    imp = (1 - rm / bm) * 100 if bm > 0 else 0

    try:
        render_comparison(
            [(sv, sf), (ov, of_), (rv, sf)],
            ["Simple", "Ornate (Target)", "Transfer Result"],
            os.path.join(out_dir, "comparison.png"),
            title=name.replace("_", " ").title(),
        )
    except Exception as e:
        print(f"    Render failed: {e}")

    return {"name": name, "baseline": bm, "result": rm,
            "improvement": imp, "time": elapsed, "ok": rm < bm}


def main():
    os.makedirs(BASE_DIR, exist_ok=True)
    print("=" * 70)
    print("COMPLEX OBJECTS — BATCH DETAIL TRANSFER TEST")
    print("=" * 70)

    objects = list_complex_objects()
    results = []

    for idx, name in enumerate(objects, 1):
        print(f"\n[{idx}/{len(objects)}] {name}...")
        sv, sf = make_complex_simple(name)
        ov, of_ = make_complex_ornate(name)
        r = run_single(name, sv, sf, ov, of_, os.path.join(BASE_DIR, name))
        results.append(r)
        s = "OK" if r["ok"] else "FAIL"
        print(f"    {s}: {r['improvement']:.1f}% ({r['baseline']:.2f} -> {r['result']:.2f}) in {r['time']:.2f}s")

    print("\n" + "=" * 70)
    print(f"{'Object':<24} {'Baseline':>10} {'Result':>10} {'Improv%':>10} {'Status':>8}")
    print("-" * 64)

    ok_count = 0
    total_imp = 0
    for r in results:
        s = "OK" if r["ok"] else "FAIL"
        ok_count += r["ok"]
        total_imp += r["improvement"]
        print(f"{r['name']:<24} {r['baseline']:>10.2f} {r['result']:>10.2f} "
              f"{r['improvement']:>9.1f}% {s:>8}")

    avg = total_imp / len(results)
    print("-" * 64)
    print(f"{'AVERAGE':<24} {'':>10} {'':>10} {avg:>9.1f}%")
    print(f"\nPass rate: {ok_count}/{len(results)}")

    with open(os.path.join(BASE_DIR, "summary.txt"), "w") as f:
        f.write("Object,Baseline,Result,Improvement%,Status\n")
        for r in results:
            f.write(f"{r['name']},{r['baseline']:.4f},{r['result']:.4f},"
                    f"{r['improvement']:.2f},{'PASS' if r['ok'] else 'FAIL'}\n")

    return ok_count == len(results)


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
