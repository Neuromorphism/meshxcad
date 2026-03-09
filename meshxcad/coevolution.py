"""Alternating coevolution runner for CAD elegance.

Runs Loop 1 (CAD-vs-Mesh discriminator) and Loop 2 (Elegance Tournament)
alternately on a shared set of mesh targets.  Each loop's improvements feed
into the next iteration:

  - Loop 1 teaches the program to produce meshes that look natural.
  - Loop 2 teaches the program to be elegant (fewer ops, clean design).
  - Techniques that improve one loop are carried forward to the other.

The runner stops when neither loop produces improvement across a full
sweep of all objects.
"""

import time
import json
import os
import copy
import numpy as np

from .cad_program import (
    CadOp, CadProgram, initial_program, refine_operation,
    add_operation, simplify_program, find_program_gaps,
)
from .elegance import (
    compute_elegance_score, compute_discriminator_features,
    discriminate_cad_vs_mesh, score_accuracy,
    run_cad_vs_mesh_loop, run_elegance_tournament,
    compare_elegance, _mutate_for_elegance, _generate_anti_cad_mutations,
    ELEGANCE_WEIGHTS,
)
from .synthetic import make_sphere_mesh, make_cylinder_mesh, make_cube_mesh
from .objects.builder import make_torus, revolve_profile, combine_meshes
from .objects.operations import extrude_polygon, make_regular_polygon, sweep_along_path


# ============================================================================
# Object gallery: diverse targets from all catalogs
# ============================================================================

def _make_target_sphere():
    v, f = make_sphere_mesh(radius=5.0, lat_divs=20, lon_divs=20)
    return v, f, "sphere"


def _make_target_cylinder():
    v, f = make_cylinder_mesh(radius=3.0, height=10.0, radial_divs=24, height_divs=15)
    return v, f, "cylinder"


def _make_target_cube():
    v, f = make_cube_mesh(size=8.0, subdivisions=5)
    return v, f, "cube"


def _make_target_torus():
    v, f = make_torus(major_r=6.0, minor_r=2.0, z_center=0.0,
                       n_angular=32, n_cross=16)
    return v, f, "torus"


def _make_target_vase():
    from .objects.catalog import make_ornate
    v, f = make_ornate("classical_vase")
    return v, f, "vase"


def _make_target_goblet():
    from .objects.catalog import make_ornate
    v, f = make_ornate("goblet")
    return v, f, "goblet"


def _make_target_candlestick():
    from .objects.catalog import make_ornate
    v, f = make_ornate("candlestick")
    return v, f, "candlestick"


def _make_target_bell():
    from .objects.catalog import make_ornate
    v, f = make_ornate("bell")
    return v, f, "bell"


def _make_target_column():
    from .objects.catalog import make_ornate
    v, f = make_ornate("column")
    return v, f, "column"


def _make_target_bowl():
    from .objects.catalog import make_ornate
    v, f = make_ornate("decorative_bowl")
    return v, f, "bowl"


def _make_target_spinning_top():
    from .objects.catalog import make_ornate
    v, f = make_ornate("spinning_top")
    return v, f, "spinning_top"


def _make_target_door_knob():
    from .objects.catalog import make_ornate
    v, f = make_ornate("door_knob")
    return v, f, "door_knob"


def _make_target_gear():
    from .objects.complex_catalog import make_complex_ornate
    v, f = make_complex_ornate("spur_gear")
    return v, f, "gear"


def _make_target_bracket():
    from .objects.complex_catalog import make_complex_ornate
    v, f = make_complex_ornate("shelf_bracket")
    return v, f, "bracket"


def _make_target_hex_nut():
    from .objects.complex_catalog import make_complex_ornate
    v, f = make_complex_ornate("hex_nut")
    return v, f, "hex_nut"


def _make_target_star_knob():
    from .objects.complex_catalog import make_complex_ornate
    v, f = make_complex_ornate("star_knob")
    return v, f, "star_knob"


def _make_target_hex_extrude():
    """Hexagonal extrusion — tests extrude path."""
    poly = make_regular_polygon(6, radius=4.0)
    v, f = extrude_polygon(poly, height=12.0)
    return v, f, "hex_extrude"


def _make_target_cone():
    """Truncated cone — tests revolve-like shape."""
    profile = [(4.0, 0.0), (3.5, 3.0), (2.5, 6.0), (1.5, 9.0), (0.5, 12.0)]
    v, f = revolve_profile(profile, n_angular=32)
    return v, f, "cone"


def _make_target_bent_tube():
    """Curved tube — tests sweep path."""
    t = np.linspace(0, np.pi, 30)
    path = np.column_stack([8.0 * np.sin(t), np.zeros(30), 8.0 * (1 - np.cos(t))])
    profile = np.column_stack([
        2.0 * np.cos(np.linspace(0, 2 * np.pi, 16, endpoint=False)),
        2.0 * np.sin(np.linspace(0, 2 * np.pi, 16, endpoint=False)),
    ])
    v, f = sweep_along_path(profile, path, n_profile=16)
    return v, f, "bent_tube"


def _make_target_wine_decanter():
    from .objects.catalog import make_ornate
    v, f = make_ornate("wine_decanter")
    return v, f, "wine_decanter"


def _make_target_fluted_column():
    from .objects.complex_catalog import make_complex_ornate
    v, f = make_complex_ornate("fluted_column")
    return v, f, "fluted_column"


def _make_target_lattice_panel():
    from .objects.complex_catalog import make_complex_ornate
    v, f = make_complex_ornate("lattice_panel")
    return v, f, "lattice_panel"


def _make_target_pipe_flange():
    from .objects.complex_catalog import make_complex_ornate
    v, f = make_complex_ornate("pipe_flange")
    return v, f, "pipe_flange"


def _make_target_chess_king():
    from .objects.catalog import make_ornate
    v, f = make_ornate("chess_king")
    return v, f, "chess_king"


def _make_target_castellated_ring():
    from .objects.complex_catalog import make_complex_ornate
    v, f = make_complex_ornate("castellated_ring")
    return v, f, "castellated_ring"


# All targets — ordered roughly by difficulty (primitives first, complex last)
TARGET_GENERATORS = [
    _make_target_sphere,
    _make_target_cylinder,
    _make_target_cube,
    _make_target_torus,
    _make_target_cone,
    _make_target_hex_extrude,
    _make_target_bowl,
    _make_target_spinning_top,
    _make_target_door_knob,
    _make_target_bell,
    _make_target_vase,
    _make_target_goblet,
    _make_target_candlestick,
    _make_target_column,
    _make_target_wine_decanter,
    _make_target_bent_tube,
    _make_target_gear,
    _make_target_bracket,
    _make_target_hex_nut,
    _make_target_star_knob,
]


# ============================================================================
# Technique library — learned strategies from both loops
# ============================================================================

class TechniqueLibrary:
    """Tracks which mutation strategies work, across both loops.

    Each technique has:
      - name: human-readable identifier
      - source: "discriminator" or "elegance"
      - attempts: how many times tried
      - successes: how many times it improved the objective
      - total_improvement: sum of improvement amounts
    """

    def __init__(self):
        self.techniques = {}
        self._generation = 0

    def record(self, technique_name, source, improved, improvement_amount=0.0):
        key = f"{source}:{technique_name}"
        if key not in self.techniques:
            self.techniques[key] = {
                "name": technique_name,
                "source": source,
                "attempts": 0,
                "successes": 0,
                "total_improvement": 0.0,
                "first_seen_gen": self._generation,
                "last_success_gen": -1,
            }
        t = self.techniques[key]
        t["attempts"] += 1
        if improved:
            t["successes"] += 1
            t["total_improvement"] += improvement_amount
            t["last_success_gen"] = self._generation

    def advance_generation(self):
        self._generation += 1

    @property
    def generation(self):
        return self._generation

    def success_rate(self, key):
        t = self.techniques.get(key)
        if not t or t["attempts"] == 0:
            return 0.0
        return t["successes"] / t["attempts"]

    def best_techniques(self, source=None, top_n=10):
        """Return top techniques sorted by success rate."""
        items = list(self.techniques.values())
        if source:
            items = [t for t in items if t["source"] == source]
        items = [t for t in items if t["attempts"] >= 1]
        items.sort(key=lambda t: (
            t["successes"] / max(t["attempts"], 1),
            t["total_improvement"],
        ), reverse=True)
        return items[:top_n]

    def stale_techniques(self, lookback=3):
        """Techniques that haven't succeeded in recent generations."""
        cutoff = self._generation - lookback
        return [
            t for t in self.techniques.values()
            if t["last_success_gen"] < cutoff and t["attempts"] >= 2
        ]

    def summary(self):
        active = [t for t in self.techniques.values() if t["attempts"] > 0]
        if not active:
            return {"total": 0, "effective": 0, "stale": 0}
        effective = sum(1 for t in active if t["successes"] > 0)
        stale = len(self.stale_techniques())
        return {
            "total": len(active),
            "effective": effective,
            "stale": stale,
            "generation": self._generation,
            "top_5": [{
                "name": t["name"],
                "source": t["source"],
                "rate": round(t["successes"] / max(t["attempts"], 1) * 100, 1),
                "attempts": t["attempts"],
            } for t in self.best_techniques(top_n=5)],
        }

    def to_dict(self):
        return {
            "generation": self._generation,
            "techniques": self.techniques,
        }


# ============================================================================
# Per-object state
# ============================================================================

class ObjectState:
    """Tracks the evolving CadProgram for a single target object."""

    def __init__(self, name, target_v, target_f):
        self.name = name
        self.target_v = np.asarray(target_v, dtype=np.float64)
        self.target_f = np.asarray(target_f)
        self.program = initial_program(self.target_v, self.target_f)
        self.elegance = 0.0
        self.cad_score = 1.0
        self.accuracy = 0.0
        self.initial_accuracy = 0.0
        self.history = []
        self._update_scores()
        self.initial_accuracy = self.accuracy

    def _update_scores(self):
        cad_v, cad_f = self.program.evaluate()
        if len(cad_v) == 0:
            self.elegance = 0.0
            self.cad_score = 1.0
            self.accuracy = 0.0
            return
        eleg = compute_elegance_score(self.program, self.target_v, self.target_f)
        self.elegance = eleg["total"]
        self.accuracy = eleg["scores"]["accuracy"]
        self.cad_score = discriminate_cad_vs_mesh(cad_v, cad_f)

    def record(self, loop_type, action, old_elegance, old_cad, old_acc):
        self.history.append({
            "loop": loop_type,
            "action": action,
            "elegance_before": round(old_elegance, 4),
            "elegance_after": round(self.elegance, 4),
            "cad_score_before": round(old_cad, 4),
            "cad_score_after": round(self.cad_score, 4),
            "accuracy_before": round(old_acc, 4),
            "accuracy_after": round(self.accuracy, 4),
            "n_ops": self.program.n_enabled(),
        })


# ============================================================================
# Alternating loop runner
# ============================================================================

MIN_IMPROVEMENT_THRESHOLD = 0.002  # ignore changes smaller than this
ACCURACY_FLOOR = 0.40  # never accept mutations that drop accuracy below this


def _joint_score(elegance, cad_score, accuracy):
    """Unified objective: high elegance + high accuracy + low cad-likeness.

    Both loops use this same objective so they cannot undo each other's work.
    """
    # Elegance and accuracy are 0-1 (higher is better)
    # cad_score is 0-1 (lower is better — less detectable as CAD)
    return elegance * 0.5 + accuracy * 0.3 + (1.0 - cad_score) * 0.2


def _run_discriminator_pass(state, library, rng, rounds_per_object=5):
    """Run Loop 1 (discriminator) on a single object.

    Uses joint objective: accepts mutations that improve the combined score
    of elegance + accuracy + low-cad-likeness.

    Returns True if any improvement was made.
    """
    improved_any = False

    for _ in range(rounds_per_object):
        old_eleg = state.elegance
        old_cad = state.cad_score
        old_acc = state.accuracy

        cad_v, cad_f = state.program.evaluate()
        if len(cad_v) == 0:
            break

        features = compute_discriminator_features(cad_v, cad_f)
        mutations = _generate_anti_cad_mutations(
            state.program, features, state.target_v, state.target_f)

        best_program = state.program
        best_joint = _joint_score(state.elegance, state.cad_score, state.accuracy)
        best_action = "held"

        for action_name, mutant in mutations:
            mut_v, mut_f = mutant.evaluate()
            if len(mut_v) == 0:
                continue
            mut_cad = discriminate_cad_vs_mesh(mut_v, mut_f)
            mut_eleg = compute_elegance_score(
                mutant, state.target_v, state.target_f)
            mut_acc = mut_eleg["scores"]["accuracy"]
            mut_joint = _joint_score(mut_eleg["total"], mut_cad, mut_acc)

            acc_floor = max(ACCURACY_FLOOR, state.initial_accuracy - 0.05)
            if (mut_joint > best_joint
                    and mut_acc >= acc_floor):
                best_joint = mut_joint
                best_program = mutant
                best_action = action_name

        old_joint = _joint_score(old_eleg, old_cad, old_acc)
        improved = (best_action != "held"
                    and (best_joint - old_joint) >= MIN_IMPROVEMENT_THRESHOLD)

        if improved:
            imp_amount = best_joint - old_joint
            state.program = best_program
            state._update_scores()
            improved_any = True
        else:
            best_action = "held"
            imp_amount = 0.0

        library.record(best_action, "discriminator", improved, abs(imp_amount))
        state.record("discriminator", best_action, old_eleg, old_cad, old_acc)

        if not improved:
            break  # no point continuing if stuck

    return improved_any


def _run_elegance_pass(state, library, rng, rounds_per_object=5):
    """Run Loop 2 (elegance tournament) on a single object.

    Uses joint objective: accepts mutations that improve the combined score,
    not just elegance alone.

    Returns True if any improvement was made.
    """
    improved_any = False

    for _ in range(rounds_per_object):
        old_eleg = state.elegance
        old_cad = state.cad_score
        old_acc = state.accuracy

        mutations = _mutate_for_elegance(
            state.program, state.target_v, state.target_f, rng)

        best_program = state.program
        best_joint = _joint_score(state.elegance, state.cad_score, state.accuracy)
        best_action = "held"

        for action_name, mutant in mutations:
            mut_v, mut_f = mutant.evaluate()
            if len(mut_v) == 0:
                continue
            mut_cad = discriminate_cad_vs_mesh(mut_v, mut_f)
            mut_eleg = compute_elegance_score(
                mutant, state.target_v, state.target_f)
            mut_acc = mut_eleg["scores"]["accuracy"]
            mut_joint = _joint_score(mut_eleg["total"], mut_cad, mut_acc)

            acc_floor = max(ACCURACY_FLOOR, state.initial_accuracy - 0.05)
            if (mut_joint > best_joint
                    and mut_acc >= acc_floor):
                best_joint = mut_joint
                best_program = mutant
                best_action = action_name

        old_joint = _joint_score(old_eleg, old_cad, old_acc)
        improved = (best_action != "held"
                    and (best_joint - old_joint) >= MIN_IMPROVEMENT_THRESHOLD)

        if improved:
            imp_amount = best_joint - old_joint
            state.program = best_program
            state._update_scores()
            improved_any = True
        else:
            best_action = "held"
            imp_amount = 0.0

        library.record(best_action, "elegance", improved, abs(imp_amount))
        state.record("elegance", best_action, old_eleg, old_cad, old_acc)

        if not improved:
            break

    return improved_any


def run_coevolution(target_generators=None,
                     max_sweeps=20,
                     rounds_per_object=5,
                     patience=3,
                     output_dir="/tmp/coevolution"):
    """Run alternating Loop 1 / Loop 2 on all objects until convergence.

    Each "sweep" runs both loops on every object.  Stops when `patience`
    consecutive sweeps produce no improvement on any object.

    Args:
        target_generators: list of callables returning (v, f, name).
            Defaults to TARGET_GENERATORS.
        max_sweeps: maximum number of full sweeps (each sweep = both loops
            on all objects).
        rounds_per_object: how many inner rounds each loop gets per object.
        patience: stop after this many sweeps with no improvement.
        output_dir: where to save results.

    Returns:
        dict with per-object results, technique library, and convergence info.
    """
    if target_generators is None:
        target_generators = TARGET_GENERATORS

    os.makedirs(output_dir, exist_ok=True)
    start_time = time.time()

    # Initialize all objects
    objects = {}
    print(f"Initializing {len(target_generators)} target objects...")
    for gen_fn in target_generators:
        try:
            v, f, name = gen_fn()
            objects[name] = ObjectState(name, v, f)
            eleg = objects[name].elegance
            acc = objects[name].accuracy
            cad = objects[name].cad_score
            print(f"  {name:20s}  elegance={eleg:.3f}  accuracy={acc:.3f}  "
                  f"cad_score={cad:.3f}  ops={objects[name].program.n_enabled()}")
        except Exception as e:
            print(f"  {name:20s}  FAILED: {e}")

    if not objects:
        return {"error": "No objects initialized"}

    library = TechniqueLibrary()
    rng = np.random.RandomState(42)

    sweep_results = []
    no_improvement_count = 0

    for sweep in range(max_sweeps):
        sweep_start = time.time()
        sweep_improved = False
        loop1_improved = 0
        loop2_improved = 0
        loop1_total = 0
        loop2_total = 0

        # --- Loop 1: Discriminator pass on all objects ---
        print(f"\n{'='*70}")
        print(f"SWEEP {sweep} — Loop 1: CAD vs Mesh Discriminator")
        print(f"{'='*70}")

        for name, state in objects.items():
            loop1_total += 1
            old_scores = (state.elegance, state.cad_score, state.accuracy)

            did_improve = _run_discriminator_pass(
                state, library, rng, rounds_per_object)

            if did_improve:
                loop1_improved += 1
                sweep_improved = True

            status = "IMPROVED" if did_improve else "held"
            print(f"  [{status:8s}] {name:20s}  "
                  f"cad={old_scores[1]:.3f}→{state.cad_score:.3f}  "
                  f"eleg={old_scores[0]:.3f}→{state.elegance:.3f}  "
                  f"acc={old_scores[2]:.3f}→{state.accuracy:.3f}  "
                  f"ops={state.program.n_enabled()}")

        # --- Loop 2: Elegance pass on all objects ---
        print(f"\nSWEEP {sweep} — Loop 2: Elegance Tournament")
        print(f"{'-'*70}")

        for name, state in objects.items():
            loop2_total += 1
            old_scores = (state.elegance, state.cad_score, state.accuracy)

            did_improve = _run_elegance_pass(
                state, library, rng, rounds_per_object)

            if did_improve:
                loop2_improved += 1
                sweep_improved = True

            status = "IMPROVED" if did_improve else "held"
            print(f"  [{status:8s}] {name:20s}  "
                  f"eleg={old_scores[0]:.3f}→{state.elegance:.3f}  "
                  f"acc={old_scores[2]:.3f}→{state.accuracy:.3f}  "
                  f"cad={old_scores[1]:.3f}→{state.cad_score:.3f}  "
                  f"ops={state.program.n_enabled()}")

        library.advance_generation()

        # --- Sweep summary ---
        elapsed = time.time() - sweep_start
        total_elapsed = time.time() - start_time

        sweep_record = {
            "sweep": sweep,
            "loop1_improved": loop1_improved,
            "loop1_total": loop1_total,
            "loop2_improved": loop2_improved,
            "loop2_total": loop2_total,
            "any_improved": sweep_improved,
            "elapsed_sec": round(elapsed, 1),
            "total_elapsed_sec": round(total_elapsed, 1),
        }
        sweep_results.append(sweep_record)

        print(f"\n  Sweep {sweep} summary: "
              f"L1={loop1_improved}/{loop1_total} "
              f"L2={loop2_improved}/{loop2_total} "
              f"({elapsed:.1f}s, total {total_elapsed:.0f}s)")

        # Technique library status
        lib_summary = library.summary()
        print(f"  Techniques: {lib_summary['effective']}/{lib_summary['total']} "
              f"effective, {lib_summary['stale']} stale")
        if lib_summary.get("top_5"):
            for t in lib_summary["top_5"][:3]:
                print(f"    {t['source']:13s} {t['name']:25s} "
                      f"{t['rate']:5.1f}% ({t['attempts']} attempts)")

        # Check convergence
        if sweep_improved:
            no_improvement_count = 0
        else:
            no_improvement_count += 1
            print(f"  No improvement ({no_improvement_count}/{patience})")

        if no_improvement_count >= patience:
            print(f"\n  CONVERGED after {patience} sweeps with no improvement.")
            break

        # Save checkpoint
        _save_checkpoint(output_dir, objects, library, sweep_results)

    # ========================================================================
    # Final report
    # ========================================================================
    total_elapsed = time.time() - start_time

    print(f"\n{'='*70}")
    print(f"COEVOLUTION COMPLETE")
    print(f"{'='*70}")
    print(f"  Sweeps: {len(sweep_results)}")
    print(f"  Total time: {total_elapsed:.1f}s")
    print(f"  Objects: {len(objects)}")

    # Per-object final scores
    print(f"\n  Final scores:")
    obj_results = {}
    for name, state in sorted(objects.items()):
        print(f"    {name:20s}  elegance={state.elegance:.3f}  "
              f"accuracy={state.accuracy:.3f}  cad={state.cad_score:.3f}  "
              f"ops={state.program.n_enabled()}  "
              f"program={state.program.summary()}")
        obj_results[name] = {
            "elegance": round(state.elegance, 4),
            "accuracy": round(state.accuracy, 4),
            "cad_score": round(state.cad_score, 4),
            "n_ops": state.program.n_enabled(),
            "program_summary": state.program.summary(),
            "program": state.program.to_dict(),
            "history": state.history,
        }

    # Technique library final
    print(f"\n  Technique library:")
    lib_summary = library.summary()
    print(f"    Total: {lib_summary['total']}")
    print(f"    Effective: {lib_summary['effective']}")
    print(f"    Stale: {lib_summary['stale']}")
    if lib_summary.get("top_5"):
        print(f"    Top techniques:")
        for t in lib_summary["top_5"]:
            print(f"      {t['source']:13s} {t['name']:25s} "
                  f"{t['rate']:5.1f}% ({t['attempts']} attempts)")

    # Convergence info
    sweeps_with_improvement = sum(1 for s in sweep_results if s["any_improved"])

    result = {
        "objects": obj_results,
        "technique_library": library.to_dict(),
        "technique_summary": lib_summary,
        "sweep_results": sweep_results,
        "convergence": {
            "total_sweeps": len(sweep_results),
            "sweeps_with_improvement": sweeps_with_improvement,
            "converged": no_improvement_count >= patience,
            "patience": patience,
            "total_elapsed_sec": round(total_elapsed, 1),
        },
        "aggregate": {
            "mean_elegance": round(float(np.mean([
                s.elegance for s in objects.values()])), 4),
            "mean_accuracy": round(float(np.mean([
                s.accuracy for s in objects.values()])), 4),
            "mean_cad_score": round(float(np.mean([
                s.cad_score for s in objects.values()])), 4),
            "total_ops": sum(s.program.n_enabled() for s in objects.values()),
        },
    }

    _save_checkpoint(output_dir, objects, library, sweep_results)

    # Save final results
    with open(os.path.join(output_dir, "final_results.json"), "w") as f:
        json.dump({k: v for k, v in result.items()
                   if k != "technique_library"}, f, indent=2)

    return result


def _save_checkpoint(output_dir, objects, library, sweep_results):
    """Save intermediate results to disk."""
    checkpoint = {
        "objects": {
            name: {
                "elegance": round(state.elegance, 4),
                "accuracy": round(state.accuracy, 4),
                "cad_score": round(state.cad_score, 4),
                "n_ops": state.program.n_enabled(),
                "program_summary": state.program.summary(),
            }
            for name, state in objects.items()
        },
        "technique_summary": library.summary(),
        "sweeps": sweep_results,
    }
    with open(os.path.join(output_dir, "checkpoint.json"), "w") as f:
        json.dump(checkpoint, f, indent=2)


# ============================================================================
# CLI entry point
# ============================================================================

if __name__ == "__main__":
    run_coevolution()
