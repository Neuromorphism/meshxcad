"""meshxcad CLI — optimise a CAD program to match a target mesh.

Usage:
    python -m meshxcad mesh.stl                      # auto-fit from scratch
    python -m meshxcad mesh.stl -c existing.json     # refine existing program
    python -m meshxcad mesh.stl -c rough_draft.step  # use STEP as starting CAD
    python -m meshxcad part.step                     # STEP file as target mesh
    python -m meshxcad mesh.stl -o out/              # custom output dir
    python -m meshxcad mesh.stl --sweeps 20          # more optimisation rounds
    python -m meshxcad mesh.stl --fast               # quick 1-sweep check

The tool:
  1. Loads a target mesh (STL / OBJ / PLY / STEP / IGES)
  2. Optionally loads a starting CAD (JSON program or STEP/IGES file)
  3. Runs the coevolution loop (discriminator + elegance) until convergence
  4. Writes: optimised CadProgram JSON, output mesh STL, and a summary
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


def main():
    parser = argparse.ArgumentParser(
        prog="meshxcad",
        description="Optimise a CAD program to match a target mesh.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
examples:
  python -m meshxcad part.stl                     # auto-fit
  python -m meshxcad part.step                    # STEP as target
  python -m meshxcad part.stl -c program.json     # refine existing
  python -m meshxcad part.stl -c draft.step       # STEP as starting CAD
  python -m meshxcad part.stl -o results/         # output dir
  python -m meshxcad part.stl --fast              # quick single sweep
  python -m meshxcad part.stl --sweeps 30 -r 10   # thorough

drawing mode:
  python -m meshxcad drawing input.png            # interpret drawing → CAD
  python -m meshxcad drawing input.png --fast     # quick mode
""",
    )

    subparsers = parser.add_subparsers(dest="command")

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

    # Original mesh arguments (backward compatible)
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

    # Dispatch to drawing mode if subcommand
    if args.command == "drawing":
        _run_drawing_mode(args)
        return

    # --- Validate inputs ---
    if not os.path.isfile(args.mesh):
        print(f"Error: mesh file not found: {args.mesh}", file=sys.stderr)
        sys.exit(1)

    if args.cad and not os.path.isfile(args.cad):
        print(f"Error: CAD file not found: {args.cad}", file=sys.stderr)
        sys.exit(1)

    # --- Output directory ---
    if args.output:
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
    # 1. CadProgram JSON
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

    # 2. Output mesh STL
    if not args.json_only:
        prog_obj = result["_program_obj"]
        cad_v, cad_f = prog_obj.evaluate()
        if len(cad_v) > 0:
            mesh_path = os.path.join(out_dir, "output.stl")
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
