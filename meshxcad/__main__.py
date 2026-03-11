"""meshxcad CLI — optimise a CAD program to match a target mesh.

Usage:
    python -m meshxcad mesh.stl                      # auto-fit from scratch
    python -m meshxcad mesh.stl -c existing.json     # refine existing program
    python -m meshxcad mesh.stl -c rough_draft.step  # use STEP as starting CAD
    python -m meshxcad part.step                     # STEP file as target mesh
    python -m meshxcad mesh.stl -o out/              # custom output dir
    python -m meshxcad mesh.stl --sweeps 20          # more optimisation rounds
    python -m meshxcad mesh.stl --fast               # quick 1-sweep check

    # Full-control add-detail to STEP:
    python -m meshxcad auto mesh.stl -c plain.step --sweeps 30 --rounds 10
    python -m meshxcad auto mesh.stl -c plain.step --gap-passes 3 --no-fillets
    python -m meshxcad auto mesh.stl -c plain.step -o result.step --render

The tool:
  1. Loads a target mesh (STL / OBJ / PLY / STEP / IGES)
  2. Optionally loads a starting CAD (JSON program or STEP/IGES file)
  3. Runs the full pipeline: analyse, initialise, coevolve, diffusion refine,
     segment + fill gaps, per-op refine, profile refine, fillets, final sweep,
     mesh polish — each phase configurable via CLI flags
  4. Writes: optimised CadProgram JSON, output mesh (STL or STEP), and a summary
"""

import argparse
import json
import os
import sys
import time

import numpy as np


# ---------------------------------------------------------------------------
# File format helpers
# ---------------------------------------------------------------------------

_STEP_EXTS = {".step", ".stp", ".iges", ".igs"}
_MESH_EXTS = {".stl", ".obj", ".ply"}


def _is_step_file(filepath):
    """Check if a file is a STEP/IGES CAD file."""
    return os.path.splitext(filepath)[1].lower() in _STEP_EXTS


# ---------------------------------------------------------------------------
# Mesh loading (STL / OBJ / PLY / STEP / IGES)
# ---------------------------------------------------------------------------

def load_mesh(filepath):
    """Load a mesh from STL, OBJ, PLY, STEP, or IGES.  Returns (vertices, faces)."""
    ext = os.path.splitext(filepath)[1].lower()

    if ext == ".stl":
        from .stl_io import read_binary_stl
        return read_binary_stl(filepath)

    if ext == ".obj":
        return _read_obj(filepath)

    if ext == ".ply":
        return _read_ply(filepath)

    if ext in _STEP_EXTS:
        from .step_io import read_step
        return read_step(filepath)

    raise ValueError(
        f"Unsupported mesh format: {ext}  "
        f"(use .stl, .obj, .ply, .step, .stp, .iges, .igs)")


def _read_obj(filepath):
    """Minimal Wavefront OBJ reader — vertices + triangulated faces."""
    verts = []
    faces = []
    with open(filepath) as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            if parts[0] == "v" and len(parts) >= 4:
                verts.append([float(parts[1]), float(parts[2]), float(parts[3])])
            elif parts[0] == "f":
                # OBJ faces are 1-indexed, may have v/vt/vn format
                idx = []
                for p in parts[1:]:
                    idx.append(int(p.split("/")[0]) - 1)
                # Triangulate n-gon as fan
                for i in range(1, len(idx) - 1):
                    faces.append([idx[0], idx[i], idx[i + 1]])
    if not verts:
        raise ValueError(f"No vertices found in {filepath}")
    return np.array(verts, dtype=np.float64), np.array(faces, dtype=np.int64)


def _read_ply(filepath):
    """Minimal ASCII/binary PLY reader."""
    with open(filepath, "rb") as f:
        # Parse header
        header_lines = []
        while True:
            line = f.readline().decode("ascii", errors="replace").strip()
            header_lines.append(line)
            if line == "end_header":
                break

        n_verts = 0
        n_faces = 0
        is_binary_le = False
        is_binary_be = False
        for line in header_lines:
            if line.startswith("element vertex"):
                n_verts = int(line.split()[-1])
            elif line.startswith("element face"):
                n_faces = int(line.split()[-1])
            elif "binary_little_endian" in line:
                is_binary_le = True
            elif "binary_big_endian" in line:
                is_binary_be = True

        if is_binary_le or is_binary_be:
            endian = "<" if is_binary_le else ">"
            # Read vertices (assume x y z as first 3 floats per vertex)
            # Count float properties for vertex
            vprop_count = sum(1 for l in header_lines
                              if l.startswith("property float")
                              or l.startswith("property double"))
            if vprop_count == 0:
                vprop_count = 3
            vdata = np.frombuffer(
                f.read(n_verts * vprop_count * 4),
                dtype=f"{endian}f4").reshape(n_verts, vprop_count)
            verts = vdata[:, :3].astype(np.float64)

            faces = []
            for _ in range(n_faces):
                count = np.frombuffer(f.read(1), dtype=np.uint8)[0]
                idx = np.frombuffer(f.read(count * 4), dtype=f"{endian}i4")
                for i in range(1, len(idx) - 1):
                    faces.append([idx[0], idx[i], idx[i + 1]])
        else:
            # ASCII mode
            lines = f.read().decode("ascii", errors="replace").splitlines()
            offset = 0
            verts = np.zeros((n_verts, 3), dtype=np.float64)
            for i in range(n_verts):
                parts = lines[offset + i].split()
                verts[i] = [float(parts[0]), float(parts[1]), float(parts[2])]
            offset += n_verts

            faces = []
            for i in range(n_faces):
                parts = lines[offset + i].split()
                count = int(parts[0])
                idx = [int(parts[j + 1]) for j in range(count)]
                for j in range(1, len(idx) - 1):
                    faces.append([idx[0], idx[j], idx[j + 1]])

    if n_verts == 0:
        raise ValueError(f"No vertices found in {filepath}")
    return np.array(verts, dtype=np.float64), np.array(faces, dtype=np.int64)


# ---------------------------------------------------------------------------
# STEP → initial CadProgram
# ---------------------------------------------------------------------------

def _load_cad_from_step(step_path, target_v, target_f, quiet=False):
    """Load a STEP file and build an initial CadProgram from its mesh.

    The STEP solid is tessellated, then ``initial_program`` fits the best
    primitive(s).  The coevolution loop can then refine from there.

    Returns:
        CadProgram instance (not a dict — optimise() handles both)
    """
    from .step_io import read_step
    from .cad_program import initial_program, refine_operation

    cad_v, cad_f = read_step(step_path)
    if not quiet:
        print(f"  STEP mesh: {len(cad_v)} vertices, {len(cad_f)} faces")

    # Build a CadProgram that best fits the STEP mesh
    prog = initial_program(cad_v, cad_f)

    # Refine it against the *target* mesh so it starts closer
    for i in range(len(prog.operations)):
        if prog.operations[i].enabled:
            refine_operation(prog, i, target_v, target_f, max_iter=20)

    if not quiet:
        print(f"  initial fit: {prog.summary()}")

    return prog


# ---------------------------------------------------------------------------
# Single-object optimiser (wraps coevolution machinery for one mesh)
# ---------------------------------------------------------------------------

def optimise(target_v, target_f, *,
             initial_cad=None,
             max_sweeps=15,
             rounds=5,
             patience=3,
             verbose=True):
    """Run the full optimisation pipeline on a single mesh.

    Args:
        target_v: (N, 3) target vertices
        target_f: (M, 3) target faces
        initial_cad: optional CadProgram (dict or CadProgram instance)
        max_sweeps: max alternating sweeps
        rounds: inner mutation rounds per sweep
        patience: stop after this many no-improvement sweeps
        verbose: print progress

    Returns:
        dict with program, scores, history, timing
    """
    from .cad_program import CadProgram, initial_program, refine_operation
    from .elegance import (
        compute_elegance_score, discriminate_cad_vs_mesh,
        _mutate_for_elegance, _generate_anti_cad_mutations,
        compute_discriminator_features, score_accuracy,
    )
    from .coevolution import (
        _joint_score, ObjectState, TechniqueLibrary,
        _run_discriminator_pass, _run_elegance_pass,
        MIN_IMPROVEMENT_THRESHOLD, ACCURACY_FLOOR,
    )

    target_v = np.asarray(target_v, dtype=np.float64)
    target_f = np.asarray(target_f)
    start = time.time()

    # --- Build or load the starting program ---
    if initial_cad is not None:
        if isinstance(initial_cad, dict):
            prog = CadProgram.from_dict(initial_cad)
        else:
            prog = initial_cad
    else:
        prog = initial_program(target_v, target_f)

    # --- Initial scores ---
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

    init_scores = {
        "elegance": round(state.elegance, 4),
        "accuracy": round(state.accuracy, 4),
        "cad_score": round(state.cad_score, 4),
        "n_ops": state.program.n_enabled(),
        "program": state.program.summary(),
    }

    if verbose:
        print(f"Initial:  {init_scores['program']}")
        print(f"          elegance={init_scores['elegance']:.3f}  "
              f"accuracy={init_scores['accuracy']:.3f}  "
              f"cad={init_scores['cad_score']:.3f}")

    # --- Run alternating sweeps ---
    library = TechniqueLibrary()
    rng = np.random.RandomState(42)
    no_improvement = 0
    sweep_log = []

    for sweep in range(max_sweeps):
        sweep_start = time.time()
        improved = False

        # Loop 1: discriminator
        old = (state.elegance, state.accuracy, state.cad_score,
               state.program.n_enabled())
        d_improved = _run_discriminator_pass(state, library, rng, rounds)
        if d_improved:
            improved = True

        # Loop 2: elegance
        e_improved = _run_elegance_pass(state, library, rng, rounds)
        if e_improved:
            improved = True

        library.advance_generation()
        elapsed = time.time() - sweep_start

        sweep_rec = {
            "sweep": sweep,
            "L1": d_improved,
            "L2": e_improved,
            "elegance": round(state.elegance, 4),
            "accuracy": round(state.accuracy, 4),
            "cad_score": round(state.cad_score, 4),
            "n_ops": state.program.n_enabled(),
            "elapsed": round(elapsed, 1),
        }
        sweep_log.append(sweep_rec)

        if verbose:
            d_tag = "+" if d_improved else "."
            e_tag = "+" if e_improved else "."
            print(f"  sweep {sweep:2d}  [{d_tag}{e_tag}]  "
                  f"eleg={state.elegance:.3f}  acc={state.accuracy:.3f}  "
                  f"cad={state.cad_score:.3f}  ops={state.program.n_enabled()}  "
                  f"({elapsed:.1f}s)")

        if improved:
            no_improvement = 0
        else:
            no_improvement += 1
            if no_improvement >= patience:
                if verbose:
                    print(f"  converged (patience={patience})")
                break

    total_time = time.time() - start

    # --- Final scores ---
    final_scores = {
        "elegance": round(state.elegance, 4),
        "accuracy": round(state.accuracy, 4),
        "cad_score": round(state.cad_score, 4),
        "n_ops": state.program.n_enabled(),
        "program_summary": state.program.summary(),
    }

    if verbose:
        print(f"\nResult:   {final_scores['program_summary']}")
        print(f"          elegance={final_scores['elegance']:.3f}  "
              f"accuracy={final_scores['accuracy']:.3f}  "
              f"cad={final_scores['cad_score']:.3f}")
        print(f"          {total_time:.1f}s total")

    # Technique summary
    lib = library.summary()

    return {
        "program": state.program.to_dict(),
        "initial": init_scores,
        "final": final_scores,
        "sweeps": sweep_log,
        "techniques": lib,
        "converged": no_improvement >= patience,
        "total_sweeps": len(sweep_log),
        "elapsed_sec": round(total_time, 1),
        "_program_obj": state.program,  # live object for mesh export
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _run_drawing_mode(args):
    """Handle the 'drawing' subcommand: image → CAD via vision model."""
    from .drawing import render_drawing_sheet
    from .drawing_compare import compare_drawings
    from .drawing_spec import DrawingSpec
    from .drawing_to_cad import drawing_to_cad
    from .vision import DrawingInterpreter
    from .cad_program import CadProgram
    from PIL import Image as PILImage

    if not os.path.isfile(args.drawing):
        print(f"Error: drawing file not found: {args.drawing}", file=sys.stderr)
        sys.exit(1)

    out_dir = args.output or os.path.splitext(args.drawing)[0] + "_cad"
    os.makedirs(out_dir, exist_ok=True)

    if not args.quiet:
        print(f"Loading drawing: {args.drawing}")

    # Parse views
    views = tuple(args.views.split(",")) if args.views else None

    # Load vision model and interpret drawing
    if not args.quiet:
        print("Loading vision model...")
    interpreter = DrawingInterpreter(
        model_path=args.model,
        quantize=args.quantize or "auto",
    )

    if not args.quiet:
        print("Interpreting drawing...")
    spec = interpreter.interpret_drawing(args.drawing, views_hint=views)

    # Save spec
    spec_path = os.path.join(out_dir, "drawing_spec.json")
    with open(spec_path, "w") as f:
        f.write(spec.to_json())
    if not args.quiet:
        print(f"  Type: {spec.object_type}, Symmetry: {spec.symmetry}")
        print(f"  Dimensions: {len(spec.dimensions)}")
        print(f"  Spec saved to {spec_path}")

    # Build initial CAD program
    program = drawing_to_cad(spec)
    if not args.quiet:
        print(f"  Initial program: {program.summary()}")

    # Optionally optimise against the drawing image
    sweeps = args.sweeps
    rounds = args.rounds
    if args.fast:
        sweeps = 3
        rounds = 3

    if sweeps > 0:
        if not args.quiet:
            print(f"\nOptimising ({sweeps} sweeps, {rounds} rounds)...")

        drawing_img = np.array(PILImage.open(args.drawing).convert("L"))
        program = _optimise_against_drawing(
            program, drawing_img,
            views=views or ("front", "side", "top"),
            max_sweeps=sweeps,
            rounds=rounds,
            patience=args.patience,
            verbose=not args.quiet,
        )

    # Output
    prog_path = os.path.join(out_dir, "program.json")
    with open(prog_path, "w") as f:
        json.dump(program.to_dict(), f, indent=2)

    cad_v, cad_f = program.evaluate()
    if len(cad_v) > 0:
        stl_path = os.path.join(out_dir, "output.stl")
        from .stl_io import write_binary_stl
        write_binary_stl(stl_path, cad_v, cad_f)

        # Render comparison
        sheet = render_drawing_sheet(cad_v, cad_f,
                                     views or ("front", "side", "top"), 512)
        comp_path = os.path.join(out_dir, "rendered.png")
        PILImage.fromarray(sheet).save(comp_path)

        if not args.quiet:
            print(f"\nOutput:")
            print(f"  {prog_path}")
            print(f"  {stl_path}")
            print(f"  {comp_path}")


def _optimise_against_drawing(program, drawing_img, views, max_sweeps,
                               rounds, patience, verbose):
    """Run coevolution-style optimisation using drawing comparison as accuracy."""
    from .drawing import render_drawing_sheet
    from .drawing_compare import compare_drawings
    from .cad_program import CadProgram
    from .elegance import compute_elegance_score
    import copy

    best_program = program.copy()
    best_score = -1.0

    no_improve = 0
    for sweep in range(max_sweeps):
        cad_v, cad_f = program.evaluate()
        if len(cad_v) == 0:
            break

        rendered = render_drawing_sheet(cad_v, cad_f, views, 512)
        # Convert to grayscale for comparison
        if rendered.ndim == 3:
            rendered_gray = np.mean(rendered, axis=2).astype(np.uint8)
        else:
            rendered_gray = rendered

        metrics = compare_drawings(drawing_img, rendered_gray)
        max_dist = max(drawing_img.shape) * 0.5
        chamfer_score = max(0, 1.0 - metrics["chamfer_distance"] / max_dist)
        score = (0.5 * chamfer_score +
                 0.3 * metrics["pixel_iou"] +
                 0.2 * (metrics["edge_precision"] + metrics["edge_recall"]) / 2)

        if verbose:
            print(f"  Sweep {sweep+1}: score={score:.3f} "
                  f"(chamfer={metrics['chamfer_distance']:.1f}, "
                  f"iou={metrics['pixel_iou']:.3f})")

        if score > best_score + 0.002:
            best_score = score
            best_program = program.copy()
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                if verbose:
                    print(f"  Converged after {sweep+1} sweeps")
                break

        # Simple mutation: try random parameter perturbations
        program = _mutate_program(best_program)

    return best_program


def _mutate_program(program):
    """Simple random perturbation of program parameters."""
    import copy
    prog = program.copy()
    rng = np.random.default_rng()
    for op in prog.operations:
        if not op.enabled:
            continue
        for key, val in op.params.items():
            if isinstance(val, (int, float)) and key not in ("divs", "subdivisions",
                                                               "radial_divs", "height_divs"):
                op.params[key] = val * (1.0 + rng.normal(0, 0.05))
    return prog


# ---------------------------------------------------------------------------
# Individual tool subcommands (for agent / programmatic use)
# ---------------------------------------------------------------------------

def _run_classify(args):
    """Classify mesh shape type."""
    from .reconstruct import classify_mesh
    target_v, target_f = load_mesh(args.mesh)
    result = classify_mesh(target_v, target_f)
    if getattr(args, 'json', False):
        print(json.dumps(result, indent=2, default=str))
    else:
        print(f"Shape type: {result['shape_type']}")
        print(f"Confidence: {result['confidence']}")
        print(f"All scores: {result['all_scores']}")


def _run_segment(args):
    """Segment mesh into CAD-friendly regions."""
    from .segmentation import segment_mesh
    target_v, target_f = load_mesh(args.mesh)
    strategy = getattr(args, 'strategy', 'auto')
    segments = segment_mesh(target_v, target_f, strategy=strategy)

    if getattr(args, 'json', False):
        result = []
        for seg in segments:
            result.append({
                "id": seg.segment_id,
                "vertices": len(seg.vertices),
                "faces": len(seg.faces),
                "action": seg.cad_action,
                "quality": round(seg.quality, 3),
                "is_fillet": seg.is_fillet,
                "label": seg.label,
                "centroid": seg.centroid.tolist(),
            })
        print(json.dumps(result, indent=2))
    else:
        n_fillets = sum(1 for s in segments if s.is_fillet)
        print(f"Segments: {len(segments)}  (fillets: {n_fillets})")
        for seg in segments:
            fillet_tag = " [FILLET]" if seg.is_fillet else ""
            print(f"  seg {seg.segment_id:>2}: {len(seg.vertices):>4}v "
                  f"{len(seg.faces):>4}f  action={seg.cad_action:<8} "
                  f"quality={seg.quality:.2f}{fillet_tag}")


def _run_fit(args):
    """Fit primitive shapes to mesh."""
    from .cad_program import _make_candidate_op, CadProgram
    from .elegance import score_accuracy
    from .reconstruct import (fit_sphere, fit_cylinder, fit_cone, fit_box,
                              fit_profiled_cylinder, fit_revolve_profile)

    target_v, target_f = load_mesh(args.mesh)
    target_v = np.asarray(target_v, dtype=np.float64)
    target_f = np.asarray(target_f)

    if args.shape:
        shapes = [args.shape]
    else:
        shapes = ["sphere", "cylinder", "profiled_cylinder",
                  "auto_revolve", "cone", "box"]

    results = []
    for shape in shapes:
        try:
            op = _make_candidate_op(shape, target_v)
            if op is None:
                continue
            prog = CadProgram([op])
            acc = score_accuracy(prog, target_v, target_f)
            results.append({
                "shape": shape,
                "op_type": op.op_type,
                "accuracy": round(acc, 4),
                "params": op.to_dict()["params"],
            })
        except Exception as e:
            results.append({"shape": shape, "error": str(e)})

    results.sort(key=lambda r: r.get("accuracy", 0), reverse=True)

    if getattr(args, 'json', False):
        print(json.dumps(results, indent=2, default=str))
    else:
        for r in results:
            if "error" in r:
                print(f"  {r['shape']}: ERROR - {r['error']}")
            else:
                print(f"  {r['shape']} ({r['op_type']}): accuracy={r['accuracy']}")


def _run_profile(args):
    """Fit variable-radius profile along axis."""
    from .reconstruct import fit_profiled_cylinder, fit_revolve_profile

    target_v, _ = load_mesh(args.mesh)

    pc = fit_profiled_cylinder(target_v, n_sections=args.sections)
    rv = fit_revolve_profile(target_v, n_slices=args.sections)

    result = {
        "profiled_cylinder": {
            "height": pc["height"],
            "taper_ratio": pc.get("taper_ratio", 0),
            "radii": pc["radii"],
            "heights": pc["heights"],
            "residual": pc["residual"],
        },
        "revolve_profile": {
            "height": rv["height"],
            "profile": rv["profile"],
            "residual": rv["residual"],
        },
    }

    if getattr(args, 'json', False):
        print(json.dumps(result, indent=2, default=str))
    else:
        print(f"Profiled cylinder: taper={pc.get('taper_ratio', 0):.3f}, "
              f"residual={pc['residual']:.4f}")
        print(f"  radii: {[round(r, 2) for r in pc['radii']]}")
        print(f"Revolve profile: residual={rv['residual']:.4f}")
        print(f"  profile: {[(round(r, 2), round(z, 2)) for r, z in rv['profile'][:10]]}")


def _run_reconstruct(args):
    """Auto-reconstruct CAD from mesh."""
    from .reconstruct import reconstruct_cad
    from .stl_io import write_binary_stl

    target_v, target_f = load_mesh(args.mesh)
    result = reconstruct_cad(target_v, target_f,
                              shape_type=getattr(args, 'shape', None))

    print(f"Shape type: {result['shape_type']}")
    print(f"Quality:    {result['quality']}")
    print(f"Vertices:   {len(result['cad_vertices'])}")
    print(f"Faces:      {len(result['cad_faces'])}")

    if args.output:
        write_binary_stl(args.output, result["cad_vertices"], result["cad_faces"])
        print(f"Saved to:   {args.output}")


def _run_score(args):
    """Score a CadProgram against a target mesh."""
    from .cad_program import CadProgram
    from .elegance import compute_elegance_score, score_accuracy
    from .elegance import score_feature_fidelity

    target_v, target_f = load_mesh(args.mesh)

    with open(args.program) as f:
        prog_data = json.load(f)
    if "program" in prog_data and "operations" in prog_data["program"]:
        prog = CadProgram.from_dict(prog_data["program"])
    elif "operations" in prog_data:
        prog = CadProgram.from_dict(prog_data)
    else:
        print("Error: cannot find operations in program JSON", file=sys.stderr)
        sys.exit(1)

    acc = score_accuracy(prog, target_v, target_f)
    eleg = compute_elegance_score(prog, target_v, target_f)
    fid = score_feature_fidelity(prog, target_v, target_f)

    result = {
        "accuracy": round(acc, 4),
        "elegance": round(eleg["total"], 4),
        "feature_fidelity": round(fid, 4),
        "n_ops": prog.n_enabled(),
        "program": prog.summary(),
        "scores": {k: round(v, 4) for k, v in eleg["scores"].items()},
    }
    print(json.dumps(result, indent=2))


def _run_refine(args):
    """Refine a CadProgram against a target mesh."""
    from .cad_program import CadProgram, refine_operation
    from .elegance import score_accuracy

    target_v, target_f = load_mesh(args.mesh)
    target_v = np.asarray(target_v, dtype=np.float64)
    target_f = np.asarray(target_f)

    with open(args.program) as f:
        prog_data = json.load(f)
    if "program" in prog_data and "operations" in prog_data["program"]:
        prog = CadProgram.from_dict(prog_data["program"])
    elif "operations" in prog_data:
        prog = CadProgram.from_dict(prog_data)
    else:
        print("Error: cannot find operations in program JSON", file=sys.stderr)
        sys.exit(1)

    acc_before = score_accuracy(prog, target_v, target_f)
    print(f"Before: accuracy={acc_before:.4f}, {prog.summary()}")

    for i, op in enumerate(prog.operations):
        if op.enabled:
            refine_operation(prog, i, target_v, target_f,
                           max_iter=args.iterations)

    acc_after = score_accuracy(prog, target_v, target_f)
    print(f"After:  accuracy={acc_after:.4f}, {prog.summary()}")

    out_path = args.output or args.program
    with open(out_path, "w") as f:
        json.dump(prog.to_dict(), f, indent=2)
    print(f"Saved to: {out_path}")


def _run_complexity(args):
    """Estimate mesh complexity and recommend op budget."""
    from .cad_program import mesh_complexity
    target_v, target_f = load_mesh(args.mesh)
    mc = mesh_complexity(target_v, target_f)

    if getattr(args, 'json', False):
        result = {k: round(v, 4) if isinstance(v, float) else v
                  for k, v in mc.items()}
        print(json.dumps(result, indent=2))
    else:
        print(f"Complexity:  {mc['complexity']:.3f}")
        print(f"Components:  {mc['n_components']}")
        print(f"Curvature H: {mc['curvature_entropy']:.3f}")
        print(f"Op budget:   {mc['op_budget']}")


def _run_auto(args):
    """Run the full automated pipeline."""
    from .auto_pipeline import auto_pipeline
    from .cad_program import CadProgram

    if not os.path.isfile(args.mesh):
        print(f"Error: mesh file not found: {args.mesh}", file=sys.stderr)
        sys.exit(1)

    # Load target mesh
    if not args.quiet:
        print(f"Loading {args.mesh}...")
    target_v, target_f = load_mesh(args.mesh)
    if not args.quiet:
        print(f"  {len(target_v)} vertices, {len(target_f)} faces")

    # Load existing CAD if provided
    initial_cad = None
    if args.cad:
        if not os.path.isfile(args.cad):
            print(f"Error: CAD file not found: {args.cad}", file=sys.stderr)
            sys.exit(1)

        if not args.quiet:
            print(f"Loading starting CAD from {args.cad}...")

        deflection = getattr(args, "deflection", 0.1)
        if _is_step_file(args.cad):
            initial_cad = _load_cad_from_step(args.cad, target_v, target_f,
                                               quiet=args.quiet)
        else:
            with open(args.cad) as f:
                cad_data = json.load(f)
            if "operations" in cad_data:
                initial_cad = cad_data
            elif "program" in cad_data and "operations" in cad_data["program"]:
                initial_cad = cad_data["program"]
            else:
                print("Error: JSON must contain 'operations' key",
                      file=sys.stderr)
                sys.exit(1)

            if not args.quiet:
                prog = CadProgram.from_dict(initial_cad)
                print(f"  Starting from: {prog.summary()}")

    # Determine thoroughness
    if args.fast:
        thoroughness = "quick"
    elif args.thorough:
        thoroughness = "thorough"
    else:
        thoroughness = "normal"

    # Build overrides dict from CLI arguments
    overrides = {}
    if getattr(args, "sweeps", None) is not None:
        overrides["sweeps"] = args.sweeps
    if getattr(args, "rounds", None) is not None:
        overrides["rounds"] = args.rounds
    if getattr(args, "patience", None) is not None:
        overrides["patience"] = args.patience
    if getattr(args, "refine_iter", None) is not None:
        overrides["refine_iter"] = args.refine_iter
    if getattr(args, "gap_passes", None) is not None:
        overrides["gap_passes"] = args.gap_passes
    if getattr(args, "diffusion_steps", None) is not None:
        overrides["diffusion_steps"] = args.diffusion_steps
    if getattr(args, "no_diffusion", False):
        overrides["no_diffusion"] = True
    if getattr(args, "no_fillets", False):
        overrides["no_fillets"] = True
    if getattr(args, "no_profile_refine", False):
        overrides["no_profile_refine"] = True
    if getattr(args, "no_polish", False):
        overrides["no_polish"] = True
    if getattr(args, "no_final_sweep", False):
        overrides["final_sweep"] = False
    if getattr(args, "seg_strategy", None) is not None:
        overrides["segmentation_strategy"] = args.seg_strategy
    if getattr(args, "seed", None) is not None:
        overrides["seed"] = args.seed

    if not args.quiet:
        mode = "adding detail" if initial_cad else "from scratch"
        override_str = ""
        if overrides:
            override_items = [f"{k}={v}" for k, v in overrides.items()]
            override_str = f", overrides: {', '.join(override_items)}"
        print(f"\nAuto pipeline ({mode}, {thoroughness}{override_str})")
        print("=" * 50)

    result = auto_pipeline(
        target_v, target_f,
        initial_cad=initial_cad,
        thoroughness=thoroughness,
        verbose=not args.quiet,
        overrides=overrides or None,
    )

    # Output: if -o looks like a file (has extension), write there directly;
    # otherwise treat it as a directory.
    out_dir = None
    out_file = None
    if args.output:
        ext = os.path.splitext(args.output)[1].lower()
        if ext in (_MESH_EXTS | _STEP_EXTS | {".json"}):
            out_file = args.output
            out_dir = os.path.dirname(args.output) or "."
        else:
            out_dir = args.output
    else:
        base = os.path.splitext(os.path.basename(args.mesh))[0]
        out_dir = os.path.join(os.path.dirname(args.mesh) or ".", f"{base}_cad")
    os.makedirs(out_dir, exist_ok=True)

    # Write program JSON — alongside the CAD file with matching name when a
    # specific output file was requested (e.g. result.step → result.json).
    if out_file and out_file.endswith(".json"):
        program_path = out_file
    elif out_file:
        program_path = os.path.splitext(out_file)[0] + ".json"
    else:
        program_path = os.path.join(out_dir, "program.json")
    program_out = {
        "program": result["program"],
        "accuracy": result["accuracy"],
        "elegance": result["elegance"],
        "phases": result["phases"],
        "elapsed_sec": result["elapsed_sec"],
        "mode": result["mode"],
        "techniques": result["techniques"],
    }
    with open(program_path, "w") as f:
        json.dump(program_out, f, indent=2)

    # Write output mesh (STL or STEP depending on output extension)
    prog_obj = result["_program_obj"]
    cad_v, cad_f = prog_obj.evaluate()
    mesh_path = None
    if len(cad_v) > 0 and not args.json_only:
        if out_file and os.path.splitext(out_file)[1].lower() in (_MESH_EXTS | _STEP_EXTS):
            mesh_path = out_file
        else:
            mesh_path = os.path.join(out_dir, "output.stl")

        mesh_ext = os.path.splitext(mesh_path)[1].lower()
        if mesh_ext in _STEP_EXTS:
            from .step_io import write_step
            write_step(mesh_path, cad_v, cad_f)
        else:
            from .stl_io import write_binary_stl
            write_binary_stl(mesh_path, cad_v, cad_f)

    # Render comparison image if requested
    if getattr(args, "render", False) and len(cad_v) > 0:
        try:
            from .render import render_comparison, HAS_MPL
            if HAS_MPL:
                comp_path = os.path.join(out_dir, "comparison.png")
                meshes = [(target_v, target_f), (cad_v, cad_f)]
                labels = ["Target Mesh", "CAD Result"]
                render_comparison(meshes, labels, comp_path,
                                  title="MeshXCAD Auto Pipeline Result")
                if not args.quiet:
                    print(f"  {comp_path}")
            elif not args.quiet:
                print("  (matplotlib not available, skipping comparison image)")
        except Exception as e:
            if not args.quiet:
                print(f"  (comparison image failed: {e})")

    if not args.quiet:
        print(f"\nOutput:")
        print(f"  {program_path}")
        if mesh_path:
            print(f"  {mesh_path}")
    else:
        summary = {
            "program_json": program_path,
            "mesh_stl": mesh_path,
            "accuracy": result["accuracy"],
            "elegance": result["elegance"].get("total", 0),
            "n_ops": result["_program_obj"].n_enabled(),
        }
        print(json.dumps(summary))


def _run_gpu_info(args):
    """Show GPU backend status."""
    from .gpu import gpu_info
    info = gpu_info()
    if getattr(args, 'json', False):
        print(json.dumps(info, indent=2))
    else:
        print(f"Backend:       {info['backend']}")
        print(f"GPU available: {info['gpu_available']}")
        if info.get('device_name'):
            print(f"Device:        {info['device_name']}")
        if info.get('gpu_memory_total_mb'):
            print(f"GPU memory:    {info.get('gpu_memory_free_mb', '?')} / "
                  f"{info['gpu_memory_total_mb']} MB")
        if info['forced_cpu']:
            print(f"Note: GPU forced off via MESHXCAD_CPU=1")
        if not info['gpu_available']:
            print(f"\nTo enable GPU acceleration, install one of:")
            print(f"  pip install cupy-cuda12x    # CuPy (recommended)")
            print(f"  pip install torch           # PyTorch with CUDA")
            print(f"  pip install numba           # Numba CUDA")


def _run_detect_fillets(args):
    """Detect intersection fillets and optionally add them to the program."""
    from .cad_program import CadProgram
    from .segmentation import detect_intersection_fillets, fit_fillet_op
    from .elegance import score_accuracy

    target_v, target_f = load_mesh(args.mesh)
    target_v = np.asarray(target_v, dtype=np.float64)
    target_f = np.asarray(target_f)

    with open(args.program) as f:
        prog_data = json.load(f)
    if "program" in prog_data and "operations" in prog_data["program"]:
        prog = CadProgram.from_dict(prog_data["program"])
    elif "operations" in prog_data:
        prog = CadProgram.from_dict(prog_data)
    else:
        print("Error: cannot find operations in program JSON", file=sys.stderr)
        sys.exit(1)

    acc_before = score_accuracy(prog, target_v, target_f)
    print(f"Program: {prog.summary()}")
    print(f"Accuracy: {acc_before:.4f}")
    print()

    fillets = detect_intersection_fillets(prog, target_v, target_f)
    if not fillets:
        print("No intersection fillets detected.")
        return

    print(f"Detected {len(fillets)} fillet region(s):")
    for i, fl in enumerate(fillets):
        print(f"  [{i}] ops ({fl['op_a']}, {fl['op_b']}): "
              f"{fl['n_vertices']} vertices, "
              f"concavity={fl['concavity']:.3f}, "
              f"zone_radius={fl['zone_radius']:.3f}")

    if args.add:
        for fl in fillets:
            fillet_op = fit_fillet_op(fl, target_v, target_f)
            if fillet_op:
                prog.operations.append(fillet_op)
                prog.invalidate_cache()
                print(f"  Added fillet op: closed={fillet_op.params['closed']}, "
                      f"radius={fillet_op.params['radius']:.4f}")

        acc_after = score_accuracy(prog, target_v, target_f)
        print(f"\nAccuracy after fillets: {acc_after:.4f} "
              f"(delta={acc_after - acc_before:+.4f})")
        print(f"Program: {prog.summary()}")

        out_path = args.output or args.program
        with open(out_path, "w") as f_out:
            json.dump(prog.to_dict(), f_out, indent=2)
        print(f"Saved to: {out_path}")


def main():
    parser = argparse.ArgumentParser(
        prog="meshxcad",
        description="Optimise a CAD program to match a target mesh.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
auto pipeline (recommended):
  python -m meshxcad auto part.stl                # full auto, all capabilities
  python -m meshxcad auto part.stl -c draft.json  # add detail to existing CAD
  python -m meshxcad auto part.stl -c plain.step  # add detail from STEP file
  python -m meshxcad auto part.stl --thorough     # thorough mode
  python -m meshxcad auto part.stl --fast         # quick mode

  fine-grained control (add detail to STEP):
  python -m meshxcad auto part.stl -c plain.step --sweeps 30 --rounds 10
  python -m meshxcad auto part.stl -c plain.step --gap-passes 3 --refine-iter 50
  python -m meshxcad auto part.stl -c plain.step --no-fillets --no-polish
  python -m meshxcad auto part.stl -c plain.step --seg-strategy skeleton
  python -m meshxcad auto part.stl -c plain.step --diffusion-steps 50
  python -m meshxcad auto part.stl -c plain.step -o result.step --render
  python -m meshxcad auto part.stl -c draft.json --seed 123 --patience 5

legacy single-pass:
  python -m meshxcad part.stl                     # coevolution only
  python -m meshxcad part.stl -c program.json     # refine existing
  python -m meshxcad part.stl --fast              # quick single sweep

individual tools:
  python -m meshxcad classify part.stl            # classify shape type
  python -m meshxcad complexity part.stl          # mesh complexity + op budget
  python -m meshxcad segment part.stl             # segment into regions
  python -m meshxcad fit part.stl                 # fit primitives
  python -m meshxcad profile part.stl             # fit profiled cylinder
  python -m meshxcad reconstruct part.stl         # reconstruct CAD mesh
  python -m meshxcad score part.stl program.json  # score a program
  python -m meshxcad refine part.stl program.json # refine a program
  python -m meshxcad detect-fillets part.stl prog.json --add

drawing mode:
  python -m meshxcad drawing input.png            # interpret drawing → CAD

gpu:
  python -m meshxcad gpu                          # show GPU backend status
""",
    )

    subparsers = parser.add_subparsers(dest="command")

    # --- Tool subcommands (for agent / programmatic use) ---

    # classify: identify shape type
    classify_p = subparsers.add_parser("classify",
                                        help="Classify mesh shape type")
    classify_p.add_argument("mesh", help="Input mesh file")
    classify_p.add_argument("--json", action="store_true",
                             help="Output as JSON")

    # complexity: estimate mesh complexity and recommended op budget
    complex_p = subparsers.add_parser("complexity",
                                       help="Estimate mesh complexity and op budget")
    complex_p.add_argument("mesh", help="Input mesh file")
    complex_p.add_argument("--json", action="store_true",
                            help="Output as JSON")

    # segment: decompose mesh into CAD-friendly regions
    seg_p = subparsers.add_parser("segment",
                                   help="Segment mesh into CAD-friendly regions")
    seg_p.add_argument("mesh", help="Input mesh file")
    seg_p.add_argument("--strategy", default="auto",
                        choices=["auto", "skeleton", "sdf", "convexity",
                                 "projection", "normal_cluster"],
                        help="Segmentation strategy (default: auto)")
    seg_p.add_argument("--json", action="store_true",
                        help="Output as JSON")

    # fit: fit primitives to mesh
    fit_p = subparsers.add_parser("fit",
                                   help="Fit primitive shapes to mesh")
    fit_p.add_argument("mesh", help="Input mesh file")
    fit_p.add_argument("--shape", default=None,
                        choices=["sphere", "cylinder", "cone", "box",
                                 "profiled_cylinder", "auto_revolve"],
                        help="Specific shape to fit (default: try all)")
    fit_p.add_argument("--json", action="store_true",
                        help="Output as JSON")

    # profile: fit profiled cylinder / revolve profile
    profile_p = subparsers.add_parser("profile",
                                       help="Fit variable-radius profile along axis")
    profile_p.add_argument("mesh", help="Input mesh file")
    profile_p.add_argument("--sections", type=int, default=12,
                            help="Number of cross-section samples (default: 12)")
    profile_p.add_argument("--json", action="store_true",
                            help="Output as JSON")

    # reconstruct: full auto-reconstruction
    recon_p = subparsers.add_parser("reconstruct",
                                     help="Auto-reconstruct CAD from mesh")
    recon_p.add_argument("mesh", help="Input mesh file")
    recon_p.add_argument("-o", "--output", default=None,
                          help="Output STL path")
    recon_p.add_argument("--shape", default=None,
                          help="Override shape type")

    # score: evaluate a program against a target
    score_p = subparsers.add_parser("score",
                                     help="Score a CadProgram against a target mesh")
    score_p.add_argument("mesh", help="Target mesh file")
    score_p.add_argument("program", help="CadProgram JSON file")

    # refine: refine a program against a target
    refine_p = subparsers.add_parser("refine",
                                      help="Refine a CadProgram against a target mesh")
    refine_p.add_argument("mesh", help="Target mesh file")
    refine_p.add_argument("program", help="CadProgram JSON file")
    refine_p.add_argument("-o", "--output", default=None,
                           help="Output JSON path (default: overwrite)")
    refine_p.add_argument("--iterations", type=int, default=30,
                           help="Max refinement iterations per op (default: 30)")

    # detect-fillets: detect intersection fillets in a fitted program
    fillet_p = subparsers.add_parser("detect-fillets",
                                      help="Detect fillet/blend regions between fitted primitives")
    fillet_p.add_argument("mesh", help="Target mesh file")
    fillet_p.add_argument("program", help="CadProgram JSON file")
    fillet_p.add_argument("--add", action="store_true",
                           help="Add fillet ops to the program and save")
    fillet_p.add_argument("-o", "--output", default=None,
                           help="Output JSON path (default: overwrite if --add)")

    # auto: full automated pipeline
    auto_p = subparsers.add_parser("auto",
                                    help="Full automated pipeline using all capabilities",
                                    formatter_class=argparse.RawDescriptionHelpFormatter,
                                    epilog="""\
examples — adding detail to an existing STEP file:
  python -m meshxcad auto target.stl -c plain.step
  python -m meshxcad auto target.stl -c plain.step --sweeps 30 --rounds 10
  python -m meshxcad auto target.stl -c plain.step --thorough --gap-passes 3
  python -m meshxcad auto target.stl -c draft.json --no-fillets --no-polish
  python -m meshxcad auto target.stl -c plain.step -o result.step --render
  python -m meshxcad auto target.stl -c plain.step --seg-strategy skeleton
  python -m meshxcad auto target.stl -c plain.step --diffusion-steps 50
""")
    auto_p.add_argument("mesh", help="Target mesh file (STL/OBJ/PLY/STEP/IGES)")
    auto_p.add_argument("-c", "--cad", default=None,
                         help="Starting CAD: JSON program or STEP/IGES file. "
                              "If provided, adds detail to this program.")
    auto_p.add_argument("-o", "--output", default=None,
                         help="Output path: directory, or file with extension "
                              "(.stl/.step/.stp/.json)")
    # Thoroughness presets
    auto_p.add_argument("--fast", action="store_true",
                         help="Quick mode: fewer sweeps, no gap-filling")
    auto_p.add_argument("--thorough", action="store_true",
                         help="Thorough mode: more sweeps, multiple gap-fill passes")
    # Fine-grained coevolution control
    auto_p.add_argument("--sweeps", type=int, default=None,
                         help="Max coevolution sweeps (overrides preset)")
    auto_p.add_argument("-r", "--rounds", type=int, default=None,
                         help="Inner mutation rounds per sweep (overrides preset)")
    auto_p.add_argument("--patience", type=int, default=None,
                         help="Stop after N no-improvement sweeps (overrides preset)")
    # Refinement control
    auto_p.add_argument("--refine-iter", type=int, default=None,
                         help="Max refinement iterations per operation "
                              "(overrides preset)")
    # Gap filling
    auto_p.add_argument("--gap-passes", type=int, default=None,
                         help="Number of gap-filling passes (overrides preset; "
                              "0 to disable)")
    auto_p.add_argument("--seg-strategy", default=None,
                         choices=["auto", "skeleton", "sdf", "convexity",
                                  "projection", "normal_cluster"],
                         help="Segmentation strategy for gap-fill regions "
                              "(default: auto)")
    # Diffusion
    auto_p.add_argument("--diffusion-steps", type=int, default=None,
                         help="Number of diffusion refinement steps "
                              "(overrides preset; 0 to disable)")
    auto_p.add_argument("--no-diffusion", action="store_true",
                         help="Disable diffusion refinement entirely")
    # Phase toggles
    auto_p.add_argument("--no-fillets", action="store_true",
                         help="Skip fillet detection and addition phase")
    auto_p.add_argument("--no-profile-refine", action="store_true",
                         help="Skip adaptive profile refinement phase")
    auto_p.add_argument("--no-polish", action="store_true",
                         help="Skip mesh polishing phase (smoothing, "
                              "edge sharpening, hole filling)")
    auto_p.add_argument("--no-final-sweep", action="store_true",
                         help="Skip the final coevolution cleanup sweep")
    # STEP tessellation
    auto_p.add_argument("--deflection", type=float, default=0.1,
                         help="Tessellation linear deflection for STEP/IGES input "
                              "(smaller = finer, default: 0.1)")
    # Output options
    auto_p.add_argument("-q", "--quiet", action="store_true",
                         help="Suppress progress output (machine-readable JSON)")
    auto_p.add_argument("--json-only", action="store_true",
                         help="Only output JSON (no mesh STL/STEP)")
    auto_p.add_argument("--render", action="store_true",
                         help="Generate comparison image (requires matplotlib)")
    # Reproducibility
    auto_p.add_argument("--seed", type=int, default=None,
                         help="Random seed for reproducibility (default: 42)")

    # gpu: show backend status
    gpu_p = subparsers.add_parser("gpu",
                                   help="Show GPU acceleration backend status")
    gpu_p.add_argument("--json", action="store_true",
                        help="Output as JSON")

    # Drawing subcommand
    draw_parser = subparsers.add_parser("drawing",
                                         help="Interpret a mechanical drawing → CAD")
    draw_parser.add_argument("drawing",
                              help="Drawing image file (PNG, JPG)")
    draw_parser.add_argument("--model", default=None,
                              help="Vision model path or HF model ID")
    draw_parser.add_argument("--views", default=None,
                              help="Comma-separated view types (front,side,top)")
    draw_parser.add_argument("--quantize", default=None,
                              help="Model quantisation: 4bit, 8bit, none, auto")
    draw_parser.add_argument("-o", "--output", default=None,
                              help="Output directory")
    draw_parser.add_argument("--sweeps", type=int, default=15,
                              help="Max optimisation sweeps (default: 15)")
    draw_parser.add_argument("-r", "--rounds", type=int, default=5,
                              help="Inner rounds per sweep (default: 5)")
    draw_parser.add_argument("--patience", type=int, default=3,
                              help="Stop after N no-improvement sweeps (default: 3)")
    draw_parser.add_argument("--fast", action="store_true",
                              help="Quick mode (3 sweeps, 3 rounds)")
    draw_parser.add_argument("-q", "--quiet", action="store_true",
                              help="Suppress progress output")

    # Check if the first arg is a known subcommand — if so, don't add
    # the top-level positional 'mesh' arg (it conflicts with subparsers).
    _known_subcommands = {
        "auto", "drawing", "classify", "complexity", "segment", "fit",
        "profile", "reconstruct", "score", "refine", "detect-fillets", "gpu",
    }
    _is_subcommand = (len(sys.argv) > 1 and sys.argv[1] in _known_subcommands)

    if not _is_subcommand:
        # Original mesh arguments (backward compatible, legacy single-pass)
        parser.add_argument("mesh", nargs="?", default=None,
                            help="Target mesh file (STL/OBJ/PLY/STEP/IGES)")
        parser.add_argument("-c", "--cad", default=None,
                            help="Starting CAD: JSON program or STEP/IGES file")
        parser.add_argument("-o", "--output", default=None,
                            help="Output directory (default: <mesh>_cad/)")
        parser.add_argument("--sweeps", type=int, default=15,
                            help="Max optimisation sweeps (default: 15)")
        parser.add_argument("-r", "--rounds", type=int, default=5,
                            help="Inner rounds per sweep (default: 5)")
        parser.add_argument("--patience", type=int, default=3,
                            help="Stop after N no-improvement sweeps (default: 3)")
        parser.add_argument("--fast", action="store_true",
                            help="Quick mode: 1 sweep, 3 rounds")
        parser.add_argument("-q", "--quiet", action="store_true",
                            help="Suppress progress output")
        parser.add_argument("--json-only", action="store_true",
                            help="Only output JSON (no mesh STL)")

    args = parser.parse_args()

    # --- GPU backend status at startup ---
    _quiet = getattr(args, "quiet", False)
    if not _quiet:
        from .gpu import get_backend, is_gpu_available, gpu_selftest
        _be = get_backend()
        if is_gpu_available():
            _ok, _detail = gpu_selftest()
            if _ok:
                print(f"GPU backend: {_be}  \u2713 selftest passed")
            else:
                print(f"GPU backend: {_be}  \u2717 selftest FAILED ({_detail})")
        else:
            _forced = os.environ.get("MESHXCAD_CPU", "").lower() in (
                "1", "true", "yes")
            if _forced:
                print("GPU backend: cpu (forced via MESHXCAD_CPU)")
            else:
                print("GPU backend: cpu (no GPU library found)")

    # Dispatch to subcommands
    _subcommand_dispatch = {
        "auto":           _run_auto,
        "drawing":        _run_drawing_mode,
        "classify":       _run_classify,
        "complexity":     _run_complexity,
        "segment":        _run_segment,
        "fit":            _run_fit,
        "profile":        _run_profile,
        "reconstruct":    _run_reconstruct,
        "score":          _run_score,
        "refine":         _run_refine,
        "detect-fillets": _run_detect_fillets,
        "gpu":            _run_gpu_info,
    }

    if args.command in _subcommand_dispatch:
        _subcommand_dispatch[args.command](args)
        return

    # --- Validate inputs ---
    if not os.path.isfile(args.mesh):
        print(f"Error: mesh file not found: {args.mesh}", file=sys.stderr)
        sys.exit(1)

    if args.cad and not os.path.isfile(args.cad):
        print(f"Error: CAD file not found: {args.cad}", file=sys.stderr)
        sys.exit(1)

    # --- Output: file or directory ---
    out_dir = None
    out_file = None
    if args.output:
        ext = os.path.splitext(args.output)[1].lower()
        if ext in (_MESH_EXTS | _STEP_EXTS | {".json"}):
            out_file = args.output
            out_dir = os.path.dirname(args.output) or "."
        else:
            out_dir = args.output
    else:
        base = os.path.splitext(os.path.basename(args.mesh))[0]
        out_dir = os.path.join(os.path.dirname(args.mesh) or ".", f"{base}_cad")

    os.makedirs(out_dir, exist_ok=True)

    # --- Load mesh ---
    if not args.quiet:
        print(f"Loading {args.mesh}...")
    target_v, target_f = load_mesh(args.mesh)
    if not args.quiet:
        print(f"  {len(target_v)} vertices, {len(target_f)} faces")

    # --- Load existing CAD program if provided ---
    initial_cad = None
    if args.cad:
        if not args.quiet:
            print(f"Loading CAD from {args.cad}...")

        if _is_step_file(args.cad):
            # STEP/IGES → tessellate → fit initial program from the mesh
            initial_cad = _load_cad_from_step(args.cad, target_v, target_f,
                                               quiet=args.quiet)
        else:
            # JSON program
            with open(args.cad) as f:
                cad_data = json.load(f)
            if "operations" in cad_data:
                initial_cad = cad_data
            elif "program" in cad_data:
                initial_cad = cad_data["program"]
            else:
                print("Error: JSON must contain 'operations' or 'program' key",
                      file=sys.stderr)
                sys.exit(1)
            if not args.quiet:
                from .cad_program import CadProgram
                prog = CadProgram.from_dict(initial_cad)
                print(f"  {prog.summary()}")

    # --- Fast mode overrides ---
    sweeps = args.sweeps
    rounds = args.rounds
    if args.fast:
        sweeps = 1
        rounds = 3

    # --- Run optimisation ---
    if not args.quiet:
        print(f"\nOptimising (max {sweeps} sweeps, {rounds} rounds/sweep)...")

    result = optimise(
        target_v, target_f,
        initial_cad=initial_cad,
        max_sweeps=sweeps,
        rounds=rounds,
        patience=args.patience,
        verbose=not args.quiet,
    )

    # --- Write outputs ---
    # 1. CadProgram JSON — alongside the CAD file with matching name when a
    # specific output file was requested (e.g. result.step → result.json).
    if out_file and out_file.endswith(".json"):
        program_path = out_file
    elif out_file:
        program_path = os.path.splitext(out_file)[0] + ".json"
    else:
        program_path = os.path.join(out_dir, "program.json")
    program_out = {
        "program": result["program"],
        "initial": result["initial"],
        "final": result["final"],
        "converged": result["converged"],
        "total_sweeps": result["total_sweeps"],
        "elapsed_sec": result["elapsed_sec"],
        "techniques": result["techniques"],
        "sweeps": result["sweeps"],
    }
    with open(program_path, "w") as f:
        json.dump(program_out, f, indent=2)

    # 2. Output mesh (STL or STEP depending on output extension)
    if not args.json_only:
        prog_obj = result["_program_obj"]
        cad_v, cad_f = prog_obj.evaluate()
        if len(cad_v) > 0:
            if out_file and os.path.splitext(out_file)[1].lower() in (_MESH_EXTS | _STEP_EXTS):
                mesh_path = out_file
            else:
                mesh_path = os.path.join(out_dir, "output.stl")

            mesh_ext = os.path.splitext(mesh_path)[1].lower()
            if mesh_ext in _STEP_EXTS:
                from .step_io import write_step
                write_step(mesh_path, cad_v, cad_f)
            else:
                from .stl_io import write_binary_stl
                write_binary_stl(mesh_path, cad_v, cad_f)
        else:
            mesh_path = None
    else:
        mesh_path = None

    # --- Summary ---
    if not args.quiet:
        print(f"\nOutput:")
        print(f"  {program_path}")
        if mesh_path:
            print(f"  {mesh_path}")
    else:
        # Machine-readable: print JSON to stdout
        summary = {
            "program_json": program_path,
            "mesh_stl": mesh_path,
            **result["final"],
        }
        print(json.dumps(summary))


if __name__ == "__main__":
    main()
