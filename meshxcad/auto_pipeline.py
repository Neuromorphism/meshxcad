"""Automated mesh-to-CAD pipeline that chains all available capabilities.

Two modes:
    1. Mesh-only: given just a target mesh, build the best CAD program
       from scratch using all available analysis and fitting tools.
    2. CAD+Mesh: given an existing CAD program plus a target mesh, add
       detail — fill gaps, add missing features, detect fillets, refine.

The pipeline runs these phases in order, each phase only proceeding if
it improves the result:

    Phase 1  Analyse     — classify shape, estimate complexity, set budget
    Phase 2  Initialise  — build starting program (or load existing)
    Phase 3  Coevolve    — alternating discriminator + elegance sweeps
    Phase 4  Segment     — find uncovered regions in the target mesh
    Phase 5  Fill gaps   — fit new primitives to uncovered segments
    Phase 6  Refine      — gradient-free parameter tuning per operation
    Phase 7  Fillets     — detect and add intersection blend surfaces
    Phase 8  Final sweep — one last coevolution pass to clean up
"""

import copy
import time
import numpy as np
from typing import Optional


def auto_pipeline(target_v, target_f, *,
                  initial_cad=None,
                  thoroughness="normal",
                  verbose=True):
    """Run the full automated mesh-to-CAD pipeline.

    Args:
        target_v: (N, 3) target mesh vertices
        target_f: (M, 3) target mesh faces
        initial_cad: optional starting CadProgram (dict or CadProgram).
                     If None, builds from scratch. If provided, adds detail.
        thoroughness: "quick", "normal", or "thorough"
        verbose: print progress

    Returns:
        dict with:
            program: CadProgram dict
            _program_obj: live CadProgram instance
            accuracy: float
            elegance: dict
            phases: list of phase logs
            elapsed_sec: total wall time
    """
    from .cad_program import (CadProgram, initial_program, refine_operation,
                              mesh_complexity, _make_candidate_op)
    from .elegance import score_accuracy, compute_elegance_score

    target_v = np.asarray(target_v, dtype=np.float64)
    target_f = np.asarray(target_f)
    t_start = time.time()

    # Thoroughness presets
    presets = {
        "quick":    {"sweeps": 3,  "rounds": 3,  "patience": 2,
                     "refine_iter": 15, "gap_passes": 0, "final_sweep": False},
        "normal":   {"sweeps": 15, "rounds": 5,  "patience": 3,
                     "refine_iter": 30, "gap_passes": 1, "final_sweep": True},
        "thorough": {"sweeps": 30, "rounds": 10, "patience": 5,
                     "refine_iter": 50, "gap_passes": 2, "final_sweep": True},
    }
    cfg = presets.get(thoroughness, presets["normal"])
    phases = []

    def _log(phase_name, msg):
        if verbose:
            print(f"  [{phase_name}] {msg}")

    # ------------------------------------------------------------------
    # Phase 1: Analyse
    # ------------------------------------------------------------------
    phase_start = time.time()
    mc = mesh_complexity(target_v, target_f)
    budget = mc["op_budget"]
    complexity = mc["complexity"]

    phase_rec = {
        "phase": "analyse",
        "complexity": round(complexity, 3),
        "op_budget": budget,
        "n_components": mc["n_components"],
        "elapsed": round(time.time() - phase_start, 2),
    }
    phases.append(phase_rec)
    if verbose:
        print(f"Phase 1 — Analyse")
        _log("analyse", f"complexity={complexity:.3f}, budget={budget} ops, "
             f"components={mc['n_components']}")

    # ------------------------------------------------------------------
    # Phase 2: Initialise
    # ------------------------------------------------------------------
    phase_start = time.time()
    if initial_cad is not None:
        # Load existing program
        if isinstance(initial_cad, dict):
            prog = CadProgram.from_dict(initial_cad)
        else:
            prog = initial_cad
        mode = "refine"
    else:
        # Build from scratch
        prog = initial_program(target_v, target_f)
        mode = "scratch"

    acc = score_accuracy(prog, target_v, target_f)
    phase_rec = {
        "phase": "initialise",
        "mode": mode,
        "accuracy": round(acc, 4),
        "n_ops": prog.n_enabled(),
        "summary": prog.summary(),
        "elapsed": round(time.time() - phase_start, 2),
    }
    phases.append(phase_rec)
    if verbose:
        print(f"Phase 2 — Initialise ({mode})")
        _log("init", f"acc={acc:.4f}, {prog.summary()}")

    # ------------------------------------------------------------------
    # Phase 3: Coevolve (main optimisation)
    # ------------------------------------------------------------------
    phase_start = time.time()
    from .coevolution import (
        ObjectState, TechniqueLibrary,
        _run_discriminator_pass, _run_elegance_pass,
    )

    state = ObjectState.__new__(ObjectState)
    state.name = "target"
    state.target_v = target_v
    state.target_f = target_f
    state.program = prog
    state.elegance = 0.0
    state.cad_score = 1.0
    state.accuracy = 0.0
    state.initial_accuracy = 0.0
    state.history = []
    state._update_scores()
    state.initial_accuracy = state.accuracy

    library = TechniqueLibrary()
    rng = np.random.RandomState(42)
    no_improvement = 0
    n_sweeps = 0

    if verbose:
        print(f"Phase 3 — Coevolve (max {cfg['sweeps']} sweeps)")

    for sweep in range(cfg["sweeps"]):
        d_improved = _run_discriminator_pass(state, library, rng, cfg["rounds"])
        e_improved = _run_elegance_pass(state, library, rng, cfg["rounds"])
        library.advance_generation()
        n_sweeps += 1

        if d_improved or e_improved:
            no_improvement = 0
        else:
            no_improvement += 1
            if no_improvement >= cfg["patience"]:
                break

        if verbose:
            d_tag = "+" if d_improved else "."
            e_tag = "+" if e_improved else "."
            _log("coevolve", f"sweep {sweep:2d} [{d_tag}{e_tag}] "
                 f"acc={state.accuracy:.4f} eleg={state.elegance:.3f} "
                 f"ops={state.program.n_enabled()}")

    prog = state.program
    acc_after_coevolve = score_accuracy(prog, target_v, target_f)
    phase_rec = {
        "phase": "coevolve",
        "sweeps": n_sweeps,
        "accuracy": round(acc_after_coevolve, 4),
        "n_ops": prog.n_enabled(),
        "elapsed": round(time.time() - phase_start, 2),
    }
    phases.append(phase_rec)

    # ------------------------------------------------------------------
    # Phase 4+5: Segment uncovered regions and fill gaps
    # ------------------------------------------------------------------
    for gap_pass in range(cfg["gap_passes"]):
        phase_start = time.time()
        acc_before_gaps = score_accuracy(prog, target_v, target_f)
        added = _fill_gaps(prog, target_v, target_f, budget, verbose)

        if added > 0:
            acc_after_gaps = score_accuracy(prog, target_v, target_f)
            if verbose:
                print(f"Phase 4 — Fill gaps (pass {gap_pass + 1})")
                _log("gaps", f"added {added} ops, "
                     f"acc {acc_before_gaps:.4f} → {acc_after_gaps:.4f}")
        else:
            acc_after_gaps = acc_before_gaps
            if verbose:
                print(f"Phase 4 — Fill gaps (pass {gap_pass + 1}): no gaps found")

        phase_rec = {
            "phase": f"fill_gaps_{gap_pass + 1}",
            "ops_added": added,
            "accuracy": round(acc_after_gaps, 4),
            "n_ops": prog.n_enabled(),
            "elapsed": round(time.time() - phase_start, 2),
        }
        phases.append(phase_rec)

    # ------------------------------------------------------------------
    # Phase 6: Refine all operations
    # ------------------------------------------------------------------
    phase_start = time.time()
    acc_before_refine = score_accuracy(prog, target_v, target_f)

    if verbose:
        print(f"Phase 5 — Refine ({cfg['refine_iter']} iterations per op)")

    for i, op in enumerate(prog.operations):
        if op.enabled:
            refine_operation(prog, i, target_v, target_f,
                             max_iter=cfg["refine_iter"])

    acc_after_refine = score_accuracy(prog, target_v, target_f)
    phase_rec = {
        "phase": "refine",
        "accuracy_before": round(acc_before_refine, 4),
        "accuracy_after": round(acc_after_refine, 4),
        "elapsed": round(time.time() - phase_start, 2),
    }
    phases.append(phase_rec)
    if verbose:
        _log("refine", f"acc {acc_before_refine:.4f} → {acc_after_refine:.4f}")

    # ------------------------------------------------------------------
    # Phase 7: Detect and add fillets
    # ------------------------------------------------------------------
    phase_start = time.time()
    fillets_added = 0
    if prog.n_enabled() >= 2:
        fillets_added = _add_fillets(prog, target_v, target_f, verbose)

    phase_rec = {
        "phase": "fillets",
        "fillets_added": fillets_added,
        "accuracy": round(score_accuracy(prog, target_v, target_f), 4),
        "elapsed": round(time.time() - phase_start, 2),
    }
    phases.append(phase_rec)

    # ------------------------------------------------------------------
    # Phase 8: Final coevolution sweep to clean up
    # ------------------------------------------------------------------
    if cfg["final_sweep"]:
        phase_start = time.time()
        acc_before_final = score_accuracy(prog, target_v, target_f)

        if verbose:
            print(f"Phase 7 — Final sweep")

        state.program = prog
        state._update_scores()
        # Short focused pass
        for sweep in range(min(5, cfg["sweeps"])):
            d_improved = _run_discriminator_pass(state, library, rng,
                                                  cfg["rounds"])
            e_improved = _run_elegance_pass(state, library, rng, cfg["rounds"])
            library.advance_generation()
            if not d_improved and not e_improved:
                break

        prog = state.program
        acc_after_final = score_accuracy(prog, target_v, target_f)
        phase_rec = {
            "phase": "final_sweep",
            "accuracy_before": round(acc_before_final, 4),
            "accuracy_after": round(acc_after_final, 4),
            "elapsed": round(time.time() - phase_start, 2),
        }
        phases.append(phase_rec)
        if verbose:
            _log("final", f"acc {acc_before_final:.4f} → {acc_after_final:.4f}")

    # ------------------------------------------------------------------
    # Final scoring
    # ------------------------------------------------------------------
    total_elapsed = time.time() - t_start
    final_acc = score_accuracy(prog, target_v, target_f)
    final_eleg = compute_elegance_score(prog, target_v, target_f)

    if verbose:
        print(f"\nResult: {prog.summary()}")
        print(f"  accuracy={final_acc:.4f}  elegance={final_eleg['total']:.4f}")
        print(f"  {total_elapsed:.1f}s total")

    return {
        "program": prog.to_dict(),
        "_program_obj": prog,
        "accuracy": round(final_acc, 4),
        "elegance": {k: round(v, 4) for k, v in final_eleg.items()
                     if isinstance(v, (int, float))},
        "phases": phases,
        "elapsed_sec": round(total_elapsed, 1),
        "mode": mode,
        "techniques": library.summary(),
    }


# ---------------------------------------------------------------------------
# Gap-filling: find uncovered target regions and add primitives
# ---------------------------------------------------------------------------

def _fill_gaps(program, target_v, target_f, budget, verbose):
    """Find uncovered target regions and try to fill them with new ops.

    Returns number of operations added.
    """
    from .cad_program import CadProgram, _make_candidate_op, _eval_op
    from .elegance import score_accuracy
    from scipy.spatial import KDTree

    if program.n_enabled() >= budget:
        return 0

    # Evaluate current program mesh
    cad_v, cad_f = program.evaluate()
    if len(cad_v) == 0:
        return 0

    # Find uncovered target vertices
    tree = KDTree(cad_v)
    dists, _ = tree.query(target_v)

    bbox_diag = float(np.linalg.norm(target_v.max(0) - target_v.min(0)))
    threshold = bbox_diag * 0.05  # 5% of bbox diagonal
    uncovered_mask = dists > threshold
    uncovered_verts = target_v[uncovered_mask]

    if len(uncovered_verts) < 10:
        return 0

    # Cluster uncovered vertices into groups
    clusters = _cluster_vertices(uncovered_verts, max_clusters=min(5, budget - program.n_enabled()))
    if not clusters:
        return 0

    added = 0
    acc_before = score_accuracy(program, target_v, target_f)

    for cluster_verts in clusters:
        if len(cluster_verts) < 8:
            continue
        if program.n_enabled() >= budget:
            break

        # Try fitting primitives to this cluster
        best_op = None
        best_acc = acc_before

        for shape in ["cylinder", "box", "sphere", "cone"]:
            try:
                op = _make_candidate_op(shape, cluster_verts)
                if op is None:
                    continue
                # Test if adding this op improves accuracy
                program.operations.append(op)
                program.invalidate_cache()
                test_acc = score_accuracy(program, target_v, target_f)
                program.operations.pop()
                program.invalidate_cache()

                if test_acc > best_acc + 0.001:
                    best_acc = test_acc
                    best_op = op
            except Exception:
                continue

        if best_op is not None:
            program.operations.append(best_op)
            program.invalidate_cache()
            added += 1
            acc_before = best_acc
            if verbose:
                print(f"    + {best_op.op_type} (acc → {best_acc:.4f})")

    return added


def _cluster_vertices(vertices, max_clusters=5):
    """Simple spatial clustering of vertices into groups."""
    if len(vertices) < 8:
        return [vertices]

    from scipy.spatial import KDTree

    # Use farthest-point sampling to pick cluster seeds
    n = len(vertices)
    k = min(max_clusters, max(1, n // 20))

    # Pick seeds via farthest-point
    seeds = [0]
    dists = np.full(n, np.inf)
    for _ in range(k - 1):
        d = np.linalg.norm(vertices - vertices[seeds[-1]], axis=1)
        dists = np.minimum(dists, d)
        seeds.append(int(np.argmax(dists)))

    # Assign each vertex to nearest seed
    seed_pts = vertices[seeds]
    tree = KDTree(seed_pts)
    _, labels = tree.query(vertices)

    clusters = []
    for i in range(k):
        mask = labels == i
        if mask.sum() >= 5:
            clusters.append(vertices[mask])

    return clusters


# ---------------------------------------------------------------------------
# Fillet detection and addition
# ---------------------------------------------------------------------------

def _add_fillets(program, target_v, target_f, verbose):
    """Detect intersection fillets and add them if they improve accuracy."""
    from .segmentation import detect_intersection_fillets, fit_fillet_op
    from .elegance import score_accuracy

    try:
        fillets = detect_intersection_fillets(program, target_v, target_f)
    except Exception:
        return 0

    if not fillets:
        return 0

    if verbose:
        print(f"Phase 6 — Fillets: {len(fillets)} candidate(s)")

    acc_before = score_accuracy(program, target_v, target_f)
    added = 0

    for fl in fillets:
        try:
            fillet_op = fit_fillet_op(fl, target_v, target_f)
        except Exception:
            continue
        if fillet_op is None:
            continue

        # Only add if it improves accuracy
        program.operations.append(fillet_op)
        program.invalidate_cache()
        acc_after = score_accuracy(program, target_v, target_f)

        if acc_after > acc_before + 0.0005:
            added += 1
            acc_before = acc_after
            if verbose:
                print(f"    + fillet (acc → {acc_after:.4f})")
        else:
            # Remove: didn't help
            program.operations.pop()
            program.invalidate_cache()

    return added
