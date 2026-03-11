"""Automated mesh-to-CAD pipeline that chains all available capabilities.

Two modes:
    1. Mesh-only: given just a target mesh, build the best CAD program
       from scratch using all available analysis and fitting tools.
    2. CAD+Mesh: given an existing CAD program plus a target mesh, add
       detail — fill gaps, add missing features, detect fillets, refine.

The pipeline runs these phases in order, each phase only proceeding if
it improves the result:

    Phase 1   Analyse       — classify shape, estimate complexity, set budget
    Phase 2   Initialise    — compete initialisers: basic, reconstruct_cad,
                              tracing, and diffusion strategy
    Phase 3   Coevolve      — alternating discriminator + elegance sweeps
    Phase 3b  Diffusion     — diffusion-based strategy refinement pass
    Phase 4   Segment       — find uncovered regions (learned strategy)
    Phase 5   Fill gaps     — fit new primitives to uncovered segments
    Phase 6   Refine        — gradient-based parameter tuning per operation
    Phase 6b  Profile refine— adaptive revolve/extrude profile refinement
    Phase 7   Fillets       — detect and add intersection blend surfaces
    Phase 8   Final sweep   — one last coevolution pass to clean up
    Phase 8b  Mesh polish   — laplacian smoothing, feature edge sharpening,
                              hole filling on the output mesh
    Phase 9   Record        — save experience for federated learning
"""

import copy
import time
import numpy as np
from typing import Optional
from scipy.spatial import KDTree
from .gpu import AcceleratedKDTree as _AKDTree


# ---------------------------------------------------------------------------
# Federated learning: load canonical weights at module import time
# ---------------------------------------------------------------------------

def _load_federation_weights():
    """Load learned weights from federated canonical checkpoint if available."""
    try:
        from .federation import load_canonical
        load_canonical()
        return True
    except Exception:
        return False

_federation_loaded = _load_federation_weights()


def auto_pipeline(target_v, target_f, *,
                  initial_cad=None,
                  thoroughness="normal",
                  verbose=True,
                  overrides=None):
    """Run the full automated mesh-to-CAD pipeline.

    Args:
        target_v: (N, 3) target mesh vertices
        target_f: (M, 3) target mesh faces
        initial_cad: optional starting CadProgram (dict or CadProgram).
                     If None, builds from scratch. If provided, adds detail.
        thoroughness: "quick", "normal", or "thorough"
        verbose: print progress
        overrides: optional dict of config overrides.  Supported keys:
            sweeps, rounds, patience, refine_iter, gap_passes, final_sweep,
            diffusion_steps, use_reconstruct, no_fillets, no_profile_refine,
            no_polish, no_diffusion, segmentation_strategy, seed.

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
                              refine_operation_diff,
                              mesh_complexity, _make_candidate_op)
    from .elegance import score_accuracy, compute_elegance_score

    target_v = np.asarray(target_v, dtype=np.float64)
    target_f = np.asarray(target_f)
    t_start = time.time()

    # Thoroughness presets
    presets = {
        "quick":    {"sweeps": 3,  "rounds": 3,  "patience": 2,
                     "refine_iter": 15, "gap_passes": 0, "final_sweep": False,
                     "diffusion_steps": 0, "use_reconstruct": False},
        "normal":   {"sweeps": 15, "rounds": 5,  "patience": 3,
                     "refine_iter": 30, "gap_passes": 1, "final_sweep": True,
                     "diffusion_steps": 25, "use_reconstruct": True},
        "thorough": {"sweeps": 30, "rounds": 10, "patience": 5,
                     "refine_iter": 50, "gap_passes": 2, "final_sweep": True,
                     "diffusion_steps": 50, "use_reconstruct": True},
    }
    cfg = presets.get(thoroughness, presets["normal"])

    # Apply CLI overrides on top of the thoroughness preset
    if overrides:
        for key in ("sweeps", "rounds", "patience", "refine_iter",
                     "gap_passes", "diffusion_steps"):
            if overrides.get(key) is not None:
                cfg[key] = overrides[key]
        if overrides.get("final_sweep") is not None:
            cfg["final_sweep"] = overrides["final_sweep"]
        if overrides.get("use_reconstruct") is not None:
            cfg["use_reconstruct"] = overrides["use_reconstruct"]
        if overrides.get("no_diffusion"):
            cfg["diffusion_steps"] = 0
        if overrides.get("no_fillets"):
            cfg["_no_fillets"] = True
        if overrides.get("no_profile_refine"):
            cfg["_no_profile_refine"] = True
        if overrides.get("no_polish"):
            cfg["_no_polish"] = True
        if overrides.get("segmentation_strategy"):
            cfg["_seg_strategy"] = overrides["segmentation_strategy"]
        if overrides.get("seed") is not None:
            cfg["_seed"] = overrides["seed"]
    phases = []

    def _log(phase_name, msg):
        if verbose:
            print(f"  [{phase_name}] {msg}")

    # ------------------------------------------------------------------
    # Phase 1: Analyse — classify shape, estimate complexity, set budget
    # ------------------------------------------------------------------
    phase_start = time.time()
    mc = mesh_complexity(target_v, target_f)
    budget = mc["op_budget"]
    complexity = mc["complexity"]

    # Shape classification informs Phase 2 initialisation strategy
    shape_info = None
    if cfg["use_reconstruct"]:
        try:
            from .reconstruct import classify_mesh
            shape_info = classify_mesh(target_v, target_f)
        except Exception:
            pass

    phase_rec = {
        "phase": "analyse",
        "complexity": round(complexity, 3),
        "op_budget": budget,
        "n_components": mc["n_components"],
        "shape_type": shape_info["shape_type"] if shape_info else None,
        "shape_confidence": round(shape_info["confidence"], 3) if shape_info else None,
        "elapsed": round(time.time() - phase_start, 2),
    }
    phases.append(phase_rec)
    if verbose:
        print(f"Phase 1 — Analyse")
        shape_str = ""
        if shape_info:
            shape_str = (f", shape={shape_info['shape_type']}"
                         f" ({shape_info['confidence']:.0%})")
        _log("analyse", f"complexity={complexity:.3f}, budget={budget} ops, "
             f"components={mc['n_components']}{shape_str}")

    # ------------------------------------------------------------------
    # Phase 2: Initialise — compete multiple initialization strategies
    # ------------------------------------------------------------------
    phase_start = time.time()
    init_strategies = {}

    if initial_cad is not None:
        # Load existing program
        if isinstance(initial_cad, dict):
            prog = CadProgram.from_dict(initial_cad)
        else:
            prog = initial_cad
        mode = "refine"
        acc = score_accuracy(prog, target_v, target_f)
        init_strategies["provided"] = (prog, acc)
    else:
        mode = "scratch"

        # Strategy A: basic initial_program (always)
        prog_basic = initial_program(target_v, target_f)
        acc_basic = score_accuracy(prog_basic, target_v, target_f)
        init_strategies["basic"] = (prog_basic, acc_basic)

        # Strategy B: reconstruct_cad — shape-aware reconstruction
        if cfg["use_reconstruct"] and shape_info is not None:
            try:
                from .reconstruct import reconstruct_cad
                recon = reconstruct_cad(target_v, target_f,
                                        shape_type=shape_info["shape_type"])
                if recon and recon.get("cad_vertices") is not None:
                    # Convert reconstruction result to a CadProgram
                    recon_prog = _reconstruction_to_program(
                        recon, target_v, target_f)
                    if recon_prog is not None:
                        acc_recon = score_accuracy(recon_prog, target_v,
                                                   target_f)
                        init_strategies["reconstruct"] = (recon_prog, acc_recon)
            except Exception:
                pass

        # Strategy C: tracing reconstruction (segment → revolve/extrude/sweep)
        if cfg["use_reconstruct"]:
            try:
                from .tracing import trace_reconstruct
                trace_result = trace_reconstruct(target_v, target_f)
                if (trace_result and trace_result.get("quality", 0) > 0.3
                        and trace_result.get("cad_vertices") is not None
                        and len(trace_result["cad_vertices"]) > 0):
                    # Convert trace result to a CadProgram via segments
                    trace_prog = _tracing_to_program(
                        trace_result, target_v, target_f)
                    if trace_prog is not None:
                        acc_trace = score_accuracy(trace_prog, target_v,
                                                    target_f)
                        init_strategies["tracing"] = (trace_prog, acc_trace)
            except Exception:
                pass

        # Strategy D: diffusion strategy (for normal/thorough)
        if cfg["diffusion_steps"] > 0:
            try:
                from .diffusion_strategy import (
                    run_diffusion_strategy, DiffusionConfig)
                diff_cfg = DiffusionConfig(
                    num_timesteps=min(15, cfg["diffusion_steps"]),
                    patience=3,
                    n_candidates_per_step=4,
                )
                diff_result = run_diffusion_strategy(
                    target_v, target_f, config=diff_cfg)
                diff_prog = diff_result["program"]
                acc_diff = score_accuracy(diff_prog, target_v, target_f)
                init_strategies["diffusion"] = (diff_prog, acc_diff)
            except Exception:
                pass

        # Pick the best initialiser
        best_strategy = max(init_strategies,
                            key=lambda k: init_strategies[k][1])
        prog, acc = init_strategies[best_strategy]

    if mode == "refine":
        best_strategy = "provided"
        acc = init_strategies["provided"][1]

    phase_rec = {
        "phase": "initialise",
        "mode": mode,
        "strategy": best_strategy,
        "candidates": {k: round(v[1], 4) for k, v in init_strategies.items()},
        "accuracy": round(acc, 4),
        "n_ops": prog.n_enabled(),
        "summary": prog.summary(),
        "elapsed": round(time.time() - phase_start, 2),
    }
    phases.append(phase_rec)
    if verbose:
        print(f"Phase 2 — Initialise ({mode})")
        if len(init_strategies) > 1:
            for name, (_, a) in sorted(init_strategies.items(),
                                        key=lambda x: -x[1][1]):
                tag = " ◀" if name == best_strategy else ""
                _log("init", f"  {name}: acc={a:.4f}{tag}")
        _log("init", f"best={best_strategy}, acc={acc:.4f}, {prog.summary()}")

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
    rng = np.random.RandomState(cfg.get("_seed", 42))
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
    # Phase 3b: Diffusion refinement pass (normal/thorough only)
    # Only keep the result if it actually improves overall accuracy.
    # ------------------------------------------------------------------
    if cfg["diffusion_steps"] > 0:
        phase_start = time.time()
        acc_before_diff = acc_after_coevolve
        diff_prog, acc_after_diff, diff_info = _run_diffusion_refinement(
            prog, target_v, target_f, cfg, verbose)

        # Only adopt the diffusion result if it beats what coevolution produced
        adopted = acc_after_diff > acc_before_diff
        if adopted:
            prog = diff_prog
        else:
            acc_after_diff = acc_before_diff

        phase_rec = {
            "phase": "diffusion_refine",
            "accuracy_before": round(acc_before_diff, 4),
            "accuracy_after": round(acc_after_diff, 4),
            "adopted": adopted,
            "steps_improved": diff_info.get("n_improved", 0),
            "elapsed": round(time.time() - phase_start, 2),
        }
        phases.append(phase_rec)
        if verbose:
            tag = "adopted" if adopted else "kept coevolve"
            _log("diffusion", f"acc {acc_before_diff:.4f} → "
                 f"{acc_after_diff:.4f} ({tag}, "
                 f"{diff_info.get('n_improved', 0)} improvements)")

    # ------------------------------------------------------------------
    # Phase 4+5: Segment uncovered regions and fill gaps
    # ------------------------------------------------------------------
    for gap_pass in range(cfg["gap_passes"]):
        phase_start = time.time()
        acc_before_gaps = score_accuracy(prog, target_v, target_f)
        added = _fill_gaps(prog, target_v, target_f, budget, verbose,
                           seg_strategy=cfg.get("_seg_strategy"))

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
            # Use gradient-based refinement (autodiff or finite-diff),
            # falling back to coordinate descent if unavailable
            refine_operation_diff(prog, i, target_v, target_f,
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
    # Phase 6b: Adaptive profile refinement (revolve/extrude operations)
    # ------------------------------------------------------------------
    if thoroughness != "quick" and not cfg.get("_no_profile_refine"):
        phase_start = time.time()
        acc_before_profile = score_accuracy(prog, target_v, target_f)
        profile_improved = _refine_profiles(prog, target_v, target_f, verbose)
        acc_after_profile = score_accuracy(prog, target_v, target_f)

        # Rollback if profile refinement made things worse
        if acc_after_profile < acc_before_profile - 0.001:
            # Re-run the gradient refinement to restore
            for i, op in enumerate(prog.operations):
                if op.enabled:
                    refine_operation_diff(prog, i, target_v, target_f,
                                          max_iter=5)
            acc_after_profile = score_accuracy(prog, target_v, target_f)

        phase_rec = {
            "phase": "profile_refine",
            "accuracy_before": round(acc_before_profile, 4),
            "accuracy_after": round(acc_after_profile, 4),
            "ops_refined": profile_improved,
            "elapsed": round(time.time() - phase_start, 2),
        }
        phases.append(phase_rec)
        if verbose and profile_improved > 0:
            _log("profile", f"refined {profile_improved} ops, "
                 f"acc {acc_before_profile:.4f} → {acc_after_profile:.4f}")

    # ------------------------------------------------------------------
    # Phase 7: Detect and add fillets
    # ------------------------------------------------------------------
    phase_start = time.time()
    fillets_added = 0
    if prog.n_enabled() >= 2 and not cfg.get("_no_fillets"):
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
    # Phase 8b: Mesh polish — smooth, sharpen edges, fill holes
    # ------------------------------------------------------------------
    if thoroughness != "quick" and not cfg.get("_no_polish"):
        phase_start = time.time()
        acc_before_polish = score_accuracy(prog, target_v, target_f)
        polish_info = _polish_output_mesh(prog, target_v, target_f, verbose)
        acc_after_polish = score_accuracy(prog, target_v, target_f)

        phase_rec = {
            "phase": "mesh_polish",
            "accuracy_before": round(acc_before_polish, 4),
            "accuracy_after": round(acc_after_polish, 4),
            "steps_applied": polish_info.get("steps_applied", []),
            "elapsed": round(time.time() - phase_start, 2),
        }
        phases.append(phase_rec)
        if verbose and polish_info.get("steps_applied"):
            _log("polish", f"acc {acc_before_polish:.4f} → "
                 f"{acc_after_polish:.4f} "
                 f"({', '.join(polish_info['steps_applied'])})")

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

    # ------------------------------------------------------------------
    # Phase 9: Record experience for federated learning
    # ------------------------------------------------------------------
    _record_pipeline_experience(
        phases, final_acc, final_eleg, prog, target_v, target_f,
        mode, best_strategy if mode == "scratch" else "provided",
        library,
    )

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
# Reconstruction result → CadProgram converter
# ---------------------------------------------------------------------------

def _reconstruction_to_program(recon_result, target_v, target_f):
    """Convert a reconstruct_cad result into a CadProgram.

    The reconstruction module returns raw mesh vertices/faces and shape
    parameters.  We translate those into CadOp operations that the
    pipeline can further refine.
    """
    from .cad_program import CadProgram, CadOp

    shape = recon_result.get("shape_type", "freeform")
    params = recon_result.get("params", {})

    if shape == "freeform":
        return None  # Can't represent freeform as parametric ops

    ops = []

    if shape == "sphere":
        center = params.get("center", [0, 0, 0])
        radius = params.get("radius", 1.0)
        ops.append(CadOp("sphere", {
            "center": list(center) if hasattr(center, '__iter__') else [0, 0, 0],
            "radius": float(radius),
        }))

    elif shape == "cylinder":
        center = params.get("center", [0, 0, 0])
        radius = params.get("radius", 1.0)
        height = params.get("height", 2.0)
        axis = params.get("axis", [0, 0, 1])
        ops.append(CadOp("cylinder", {
            "center": list(center) if hasattr(center, '__iter__') else [0, 0, 0],
            "radius": float(radius),
            "height": float(height),
            "axis": list(axis) if hasattr(axis, '__iter__') else [0, 0, 1],
        }))

    elif shape == "cone":
        center = params.get("center", [0, 0, 0])
        radius = params.get("radius", 1.0)
        height = params.get("height", 2.0)
        ops.append(CadOp("cone", {
            "center": list(center) if hasattr(center, '__iter__') else [0, 0, 0],
            "radius": float(radius),
            "height": float(height),
        }))

    elif shape == "box":
        center = params.get("center", [0, 0, 0])
        half_dims = params.get("half_dims", [1, 1, 1])
        ops.append(CadOp("box", {
            "center": list(center) if hasattr(center, '__iter__') else [0, 0, 0],
            "half_dims": list(half_dims) if hasattr(half_dims, '__iter__') else [1, 1, 1],
        }))

    elif shape in ("revolve", "extrude", "sweep", "composite"):
        # For complex shapes, fall back to initial_program which
        # already handles these well via its own classification
        return None

    else:
        return None

    if not ops:
        return None

    try:
        prog = CadProgram(ops)
        # Verify it produces valid geometry
        v, f = prog.evaluate()
        if len(v) == 0:
            return None
        return prog
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Diffusion refinement: use diffusion strategies to improve an existing program
# ---------------------------------------------------------------------------

def _run_diffusion_refinement(prog, target_v, target_f, cfg, verbose):
    """Run a diffusion-based refinement pass on an existing program.

    Unlike the full diffusion initialiser, this takes an already-good
    program and applies targeted diffusion strategies (ParameterRefine,
    ElegancePolish) to squeeze out more accuracy.

    Returns (improved_program, accuracy, info_dict).
    """
    from .elegance import score_accuracy

    try:
        from .diffusion_strategy import (
            DiffusionConfig, DiffusionStep,
            ParameterRefine, ElegancePolish, TopologicalAdjust,
            score_program, extract_mesh_features, NoiseSchedule,
        )
    except ImportError:
        acc = score_accuracy(prog, target_v, target_f)
        return prog, acc, {"n_improved": 0}

    rng = np.random.RandomState(123)
    mesh_features = extract_mesh_features(target_v, target_f)

    # Focus on refinement-tier strategies (low noise)
    strategies = [ParameterRefine(), ElegancePolish(), TopologicalAdjust()]

    best_prog = copy.deepcopy(prog)
    best_score = score_program(best_prog, target_v, target_f, mesh_features)
    n_improved = 0
    n_steps = min(cfg["diffusion_steps"], 30)

    schedule = NoiseSchedule(num_timesteps=n_steps, schedule_type="cosine")

    for t_rev in range(n_steps):
        t = n_steps - 1 - t_rev
        noise_level = schedule.noise_level(t)

        # At this stage we want low-noise refinement
        effective_noise = noise_level * 0.3  # Scale down — we're refining

        all_candidates = []
        for strategy in strategies:
            # Only use strategies appropriate for current noise level
            if effective_noise > 0.6 and strategy.tier > 2:
                continue
            if effective_noise <= 0.3 and strategy.tier < 3:
                continue

            try:
                candidates = strategy.apply(
                    best_prog, target_v, target_f, mesh_features,
                    effective_noise, rng)
                all_candidates.extend(candidates)
            except Exception:
                continue

        # If tier filtering excluded everything, try all strategies
        if not all_candidates:
            for strategy in strategies:
                try:
                    candidates = strategy.apply(
                        best_prog, target_v, target_f, mesh_features,
                        effective_noise, rng)
                    all_candidates.extend(candidates)
                except Exception:
                    continue

        for name, cand in all_candidates:
            try:
                s = score_program(cand, target_v, target_f, mesh_features)
                if s > best_score:
                    best_prog = cand
                    best_score = s
                    n_improved += 1
            except Exception:
                continue

    acc = score_accuracy(best_prog, target_v, target_f)
    return best_prog, acc, {"n_improved": n_improved, "n_steps": n_steps}


# ---------------------------------------------------------------------------
# Gap-filling: find uncovered target regions and add primitives
# ---------------------------------------------------------------------------

def _fill_gaps(program, target_v, target_f, budget, verbose,
               seg_strategy=None):
    """Find uncovered target regions and try to fill them with new ops.

    Uses learned segmentation strategy selection when available to pick
    the best decomposition approach for the uncovered region, and tries
    advanced shape types (revolve, extrude, profiled_cylinder) in addition
    to basic primitives.

    Args:
        seg_strategy: optional segmentation strategy override
            (auto/skeleton/sdf/convexity/projection/normal_cluster)

    Returns number of operations added.
    """
    from .cad_program import CadProgram, _make_candidate_op, _eval_op
    from .elegance import score_accuracy

    if program.n_enabled() >= budget:
        return 0

    # Evaluate current program mesh
    cad_v, cad_f = program.evaluate()
    if len(cad_v) == 0:
        return 0

    # Find uncovered target vertices
    tree = _AKDTree(cad_v)
    dists, _ = tree.query(target_v)

    bbox_diag = float(np.linalg.norm(target_v.max(0) - target_v.min(0)))
    threshold = bbox_diag * 0.05  # 5% of bbox diagonal
    uncovered_mask = dists > threshold
    uncovered_verts = target_v[uncovered_mask]

    if len(uncovered_verts) < 10:
        return 0

    # Use learned segmentation strategy if available for smarter clustering
    uncovered_faces = _extract_uncovered_faces(target_f, uncovered_mask)

    # Cluster uncovered vertices into groups
    clusters = _cluster_vertices(uncovered_verts,
                                 max_clusters=min(5, budget - program.n_enabled()))
    if not clusters:
        return 0

    added = 0
    acc_before = score_accuracy(program, target_v, target_f)

    # Extended shape types: try advanced shapes alongside basic primitives
    basic_shapes = ["cylinder", "box", "sphere", "cone"]
    advanced_shapes = ["profiled_cylinder", "revolve", "extrude"]

    for cluster_verts in clusters:
        if len(cluster_verts) < 8:
            continue
        if program.n_enabled() >= budget:
            break

        # Try fitting primitives to this cluster
        best_op = None
        best_acc = acc_before

        # Try basic shapes first (fast)
        for shape in basic_shapes:
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

        # Try advanced shapes if basic shapes didn't find a great fit
        if best_op is None or best_acc < acc_before + 0.01:
            for shape in advanced_shapes:
                try:
                    op = _make_candidate_op(shape, cluster_verts)
                    if op is None:
                        continue
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


def _extract_uncovered_faces(faces, uncovered_mask):
    """Extract faces where at least 2 vertices are uncovered."""
    if len(faces) == 0:
        return np.array([], dtype=np.int64).reshape(0, 3)
    uncov_count = uncovered_mask[faces].sum(axis=1)
    return faces[uncov_count >= 2]


def _cluster_vertices(vertices, max_clusters=5):
    """Simple spatial clustering of vertices into groups."""
    if len(vertices) < 8:
        return [vertices]


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
    tree = _AKDTree(seed_pts)
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


# ---------------------------------------------------------------------------
# Tracing reconstruction → CadProgram converter
# ---------------------------------------------------------------------------

def _tracing_to_program(trace_result, target_v, target_f):
    """Convert a trace_reconstruct result into a CadProgram.

    The tracing module segments the mesh and reconstructs each segment
    via revolve/extrude/sweep/loft.  We translate the per-segment
    reconstructions into CadOp operations.
    """
    from .cad_program import CadProgram, CadOp

    segments = trace_result.get("segments", [])
    if not segments:
        return None

    ops = []
    for seg in segments:
        action = seg.cad_action if hasattr(seg, 'cad_action') else "freeform"

        if action == "revolve" and seg.profile is not None and len(seg.profile) >= 2:
            profile = np.asarray(seg.profile).tolist()
            # Sort by Z
            profile.sort(key=lambda p: p[1])
            ops.append(CadOp("revolve", {
                "profile_rz": profile,
                "n_angular": min(48, max(16, len(seg.vertices) // 10)),
            }))

        elif action == "extrude" and seg.profile is not None and len(seg.profile) >= 3:
            profile_2d = np.asarray(seg.profile).tolist()
            v = seg.vertices
            proj = (v - seg.centroid) @ seg.primary_axis
            height = float(proj.max() - proj.min())
            ops.append(CadOp("extrude", {
                "polygon": profile_2d,
                "height": max(height, 0.01),
            }))

        elif action == "sweep" and seg.path is not None and len(seg.path) >= 2:
            # Sweep ops are expensive — only add if confident
            if seg.quality is not None and seg.quality > 0.4:
                radii = np.linalg.norm(seg.vertices - seg.centroid, axis=1)
                r = float(np.median(radii)) * 0.5
                ops.append(CadOp("cylinder", {
                    "center": seg.centroid.tolist(),
                    "radius": r,
                    "height": float(np.linalg.norm(seg.path[-1] - seg.path[0])),
                    "axis": (seg.primary_axis / max(
                        np.linalg.norm(seg.primary_axis), 1e-12)).tolist(),
                }))

        # Skip freeform/loft — they don't map to parametric CadOps

    if not ops:
        return None

    try:
        prog = CadProgram(ops)
        v, f = prog.evaluate()
        if len(v) == 0:
            return None
        return prog
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Adaptive profile refinement for revolve/extrude operations
# ---------------------------------------------------------------------------

def _refine_profiles(program, target_v, target_f, verbose):
    """Apply adaptive profile refinement to revolve and extrude operations.

    For revolve operations: uses adaptive_revolve to rebuild per-ring
    radii from the target mesh, then updates the operation's profile.

    For extrude operations: uses adaptive_extrude to rebuild per-height
    cross-sections from the target mesh.

    Returns the number of operations that were improved.
    """
    from .elegance import score_accuracy
    import copy

    improved_count = 0

    for i, op in enumerate(program.operations):
        if not op.enabled:
            continue

        if op.op_type == "revolve":
            improved_count += _refine_revolve_op(
                program, i, target_v, target_f, verbose)

        elif op.op_type == "profiled_cylinder":
            improved_count += _refine_revolve_op(
                program, i, target_v, target_f, verbose)

    return improved_count


def _refine_revolve_op(program, op_idx, target_v, target_f, verbose):
    """Refine a revolve/profiled_cylinder op using adaptive profile tools.

    Tries several profile refinement strategies and keeps the best one.
    """
    from .elegance import score_accuracy
    import copy

    op = program.operations[op_idx]
    acc_before = score_accuracy(program, target_v, target_f)
    best_acc = acc_before
    best_params = copy.deepcopy(op.params)

    # Strategy 1: Refine profile radii from mesh
    try:
        from .revolve_align import refine_profile_radii, extract_radial_profile

        profile = op.params.get("profile_rz") or op.params.get("radii")
        if profile is not None:
            profile_rz = np.asarray(profile)
            if profile_rz.ndim == 2 and profile_rz.shape[1] == 2:
                refined = refine_profile_radii(profile_rz, target_v, blend=0.7)
                old_profile = op.params.get("profile_rz")
                op.params["profile_rz"] = refined.tolist()
                program.invalidate_cache()
                acc_test = score_accuracy(program, target_v, target_f)
                if acc_test > best_acc + 0.0005:
                    best_acc = acc_test
                    best_params = copy.deepcopy(op.params)
                else:
                    op.params["profile_rz"] = old_profile
                    program.invalidate_cache()
    except Exception:
        op.params = copy.deepcopy(best_params)
        program.invalidate_cache()

    # Strategy 2: Insert detail points where mesh has fine features
    try:
        from .revolve_align import insert_profile_detail

        profile = op.params.get("profile_rz")
        if profile is not None:
            profile_rz = np.asarray(profile)
            if profile_rz.ndim == 2 and profile_rz.shape[1] == 2:
                detailed = insert_profile_detail(profile_rz, target_v,
                                                  threshold=0.3)
                old_profile = op.params.get("profile_rz")
                op.params["profile_rz"] = detailed.tolist()
                program.invalidate_cache()
                acc_test = score_accuracy(program, target_v, target_f)
                if acc_test > best_acc + 0.0005:
                    best_acc = acc_test
                    best_params = copy.deepcopy(op.params)
                else:
                    op.params["profile_rz"] = old_profile
                    program.invalidate_cache()
    except Exception:
        op.params = copy.deepcopy(best_params)
        program.invalidate_cache()

    # Restore best params
    op.params = best_params
    program.invalidate_cache()

    improved = best_acc > acc_before + 0.0005
    if improved and verbose:
        print(f"    profile {op.op_type}[{op_idx}]: "
              f"acc {acc_before:.4f} → {best_acc:.4f}")
    return 1 if improved else 0


# ---------------------------------------------------------------------------
# Mesh quality polish: smooth, sharpen edges, fill holes
# ---------------------------------------------------------------------------

def _polish_output_mesh(program, target_v, target_f, verbose):
    """Apply mesh quality improvements to the output CAD mesh.

    Runs three steps (each only kept if it improves accuracy):
    1. Laplacian smoothing biased toward target mesh
    2. Feature edge sharpening where target has sharp edges
    3. Surface hole filling

    These operate on the evaluated mesh vertices, not on the CadProgram
    operations, so they're applied as a final post-processing step.

    Returns info dict with steps applied.
    """
    from .elegance import score_accuracy

    cad_v, cad_f = program.evaluate()
    if len(cad_v) == 0:
        return {"steps_applied": []}

    steps_applied = []
    best_v = cad_v.copy()
    best_acc = score_accuracy(program, target_v, target_f)

    # Step 1: Laplacian smoothing toward target
    try:
        from .general_align import laplacian_smooth_toward
        smoothed = laplacian_smooth_toward(
            best_v, cad_f, target_v,
            iterations=3, lam=0.3, target_weight=0.4)
        # Test improvement by temporarily overriding the cache
        acc_smooth = _score_mesh_accuracy(smoothed, cad_f, target_v)
        if acc_smooth > best_acc + 0.0005:
            best_v = smoothed
            best_acc = acc_smooth
            steps_applied.append("smooth")
    except Exception:
        pass

    # Step 2: Feature edge sharpening
    try:
        from .general_align import feature_edge_transfer
        sharpened, n_sharp = feature_edge_transfer(
            best_v, cad_f, target_v, target_f,
            dihedral_threshold_deg=35.0)
        if n_sharp > 0:
            acc_sharp = _score_mesh_accuracy(sharpened, cad_f, target_v)
            if acc_sharp > best_acc + 0.0005:
                best_v = sharpened
                best_acc = acc_sharp
                steps_applied.append(f"sharpen({n_sharp} edges)")
    except Exception:
        pass

    # Step 3: Fill surface holes
    try:
        from .general_align import fill_surface_holes
        filled_v, filled_f = fill_surface_holes(best_v, cad_f)
        if len(filled_f) > len(cad_f):
            acc_fill = _score_mesh_accuracy(filled_v, filled_f, target_v)
            if acc_fill > best_acc + 0.0005:
                best_v = filled_v
                cad_f = filled_f
                best_acc = acc_fill
                steps_applied.append("fill_holes")
    except Exception:
        pass

    # Apply the polished mesh back to the program's cache if improved
    if steps_applied:
        program._polished_mesh = (best_v, cad_f)

    return {"steps_applied": steps_applied, "accuracy": round(best_acc, 4)}


def _score_mesh_accuracy(cad_v, cad_f, target_v):
    """Score accuracy of a raw mesh against target (without going through
    CadProgram evaluation).  Used for mesh polish steps.
    """
    tree = _AKDTree(cad_v)
    dists, _ = tree.query(target_v)
    bbox_diag = float(np.linalg.norm(target_v.max(0) - target_v.min(0)))
    if bbox_diag < 1e-12:
        return 0.0
    mean_dist = float(np.mean(dists))
    return max(0.0, 1.0 - mean_dist / bbox_diag * 10)


# ---------------------------------------------------------------------------
# Federated learning: record experience from this pipeline run
# ---------------------------------------------------------------------------

def _record_pipeline_experience(phases, final_acc, final_eleg, prog,
                                target_v, target_f, mode, init_strategy,
                                technique_library):
    """Record experience from this pipeline run for federated learning.

    Captures which initialization strategy worked best, which coevolution
    techniques were effective, and overall pipeline performance.  This
    feeds into the federation system so future runs benefit from this
    session's learnings.
    """
    try:
        from .federation import record_experience, auto_save_shard
        from .optim import mesh_features

        features = mesh_features(target_v, target_f)
        feature_vec = [float(v) for v in features.values()]

        # Record which init strategy won
        record_experience(
            "segmentation",
            features=feature_vec,
            strategy=init_strategy,
            quality=float(final_acc),
        )

        # Record technique effectiveness from coevolution
        tech_summary = technique_library.summary() if technique_library else {}
        if isinstance(tech_summary, dict):
            for tech_name, stats in tech_summary.items():
                if isinstance(stats, dict) and stats.get("successes", 0) > 0:
                    record_experience(
                        "fixer",
                        fixer_name=tech_name,
                        improvement=float(stats.get("avg_improvement", 0)),
                        attempts=int(stats.get("attempts", 0)),
                        successes=int(stats.get("successes", 0)),
                    )

        # Auto-save experience shard (non-blocking, best-effort)
        try:
            auto_save_shard()
        except Exception:
            pass

    except Exception:
        pass  # Federation is optional — never fail the pipeline
