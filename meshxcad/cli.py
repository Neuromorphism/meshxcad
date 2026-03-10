"""Command-line interface for MeshXCAD detail transfer."""

import argparse
import os
import sys
import time


def _print_diagnostics(plain_verts, detail_verts, label="Diagnostics"):
    """Print bounding box and scale info to help debug alignment issues."""
    import numpy as np

    p_min, p_max = plain_verts.min(axis=0), plain_verts.max(axis=0)
    d_min, d_max = detail_verts.min(axis=0), detail_verts.max(axis=0)
    p_diag = float(np.linalg.norm(p_max - p_min))
    d_diag = float(np.linalg.norm(d_max - d_min))

    print(f"  {label}:")
    print(f"    Plain  bbox: ({p_min[0]:.2f},{p_min[1]:.2f},{p_min[2]:.2f}) "
          f"to ({p_max[0]:.2f},{p_max[1]:.2f},{p_max[2]:.2f})  "
          f"diag={p_diag:.4f}")
    print(f"    Detail bbox: ({d_min[0]:.2f},{d_min[1]:.2f},{d_min[2]:.2f}) "
          f"to ({d_max[0]:.2f},{d_max[1]:.2f},{d_max[2]:.2f})  "
          f"diag={d_diag:.4f}")
    if p_diag > 1e-12:
        print(f"    Scale ratio (detail/plain): {d_diag/p_diag:.4f}")


def _render_comparison_image(plain_verts, plain_faces,
                              detail_verts, detail_faces,
                              result_verts, result_faces,
                              output_path):
    """Render a 3-panel comparison image: Plain CAD | Detail Mesh | Result."""
    from .render import render_comparison, HAS_MPL

    if not HAS_MPL:
        print("  matplotlib not available, skipping comparison image")
        return

    base, _ = os.path.splitext(output_path)
    image_path = base + "_comparison.png"

    meshes = [
        (plain_verts, plain_faces),
        (detail_verts, detail_faces),
        (result_verts, result_faces),
    ]
    labels = ["Plain CAD", "Detail Mesh", "Result"]

    render_comparison(meshes, labels, image_path,
                      title="MeshXCAD Detail Transfer Comparison")


def _load_detail_mesh(detail_path):
    """Load a detail mesh from any supported format."""
    ext = os.path.splitext(detail_path)[1].lower()
    if ext == ".stl":
        from .stl_io import read_binary_stl
        return read_binary_stl(detail_path)
    else:
        from . import mesh_io
        fc_mesh = mesh_io.load_mesh(detail_path)
        return mesh_io.mesh_to_numpy(fc_mesh)


def _load_plain_mesh(plain_path, deflection=0.05):
    """Load a plain model as vertices/faces, tessellating CAD if needed."""
    ext = os.path.splitext(plain_path)[1].lower()
    cad_exts = {".step", ".stp", ".iges", ".igs", ".fcstd"}
    mesh_exts = {".stl", ".obj", ".ply"}

    if ext in cad_exts:
        from . import cad_io, mesh_io
        doc = cad_io.load_cad(plain_path)
        shape = cad_io.cad_to_shape(doc)
        fc_mesh = cad_io.shape_to_mesh(shape, linear_deflection=deflection)
        verts, faces = mesh_io.mesh_to_numpy(fc_mesh)
        cad_io.close_document(doc)
        return verts, faces
    elif ext in mesh_exts:
        from .stl_io import read_binary_stl
        return read_binary_stl(plain_path)
    else:
        raise ValueError(f"Unsupported plain model format: {ext}")


def cmd_transfer(args):
    """Run detail transfer from a detailed mesh onto a plain CAD model.

    Default behaviour is iterative: the result is fed back as input
    until convergence.  Use --single-pass for legacy one-shot mode.
    When --plain is omitted, an initial shape is auto-reconstructed
    from the detail mesh (scratch-start mode).
    """
    detail_path = args.detail
    output_path = args.output
    output_ext = os.path.splitext(output_path)[1].lower()
    render = not args.no_render

    # --- Load detail mesh (always needed) ---
    print(f"Loading detail mesh: {detail_path}")
    detail_verts, detail_faces = _load_detail_mesh(detail_path)
    print(f"  Loaded: {len(detail_verts)} vertices, {len(detail_faces)} faces")

    # --- Scratch-start mode (no --plain) ---
    if args.plain is None:
        # Handle --no-tracing flag
        if getattr(args, "no_tracing", False):
            args.use_tracing = False
        _transfer_scratch(detail_verts, detail_faces, output_path, args, render)
        return

    plain_path = args.plain
    plain_ext = os.path.splitext(plain_path)[1].lower()
    detail_ext = os.path.splitext(detail_path)[1].lower()

    cad_exts = {".step", ".stp", ".iges", ".igs", ".fcstd"}
    mesh_exts = {".stl", ".obj", ".ply"}
    plain_is_cad = plain_ext in cad_exts
    detail_is_mesh = detail_ext in mesh_exts

    # STEP B-rep output: use the brep_transfer pipeline (single-pass only)
    if output_ext in (".step", ".stp") and plain_is_cad and detail_is_mesh:
        _transfer_to_step_brep(plain_path, detail_path, output_path, args)
        return

    # --- Load plain model ---
    print(f"Loading plain model: {plain_path}")
    plain_verts, plain_faces = _load_plain_mesh(plain_path, args.deflection)
    print(f"  Loaded/tessellated: {len(plain_verts)} vertices, {len(plain_faces)} faces")

    _print_diagnostics(plain_verts, detail_verts)

    # --- Single-pass or iterative ---
    if args.single_pass:
        _transfer_single_pass(plain_verts, plain_faces,
                              detail_verts, detail_faces,
                              output_path, args, render)
    else:
        _transfer_iterative(plain_verts, plain_faces,
                            detail_verts, detail_faces,
                            output_path, args, render)


def _transfer_scratch(detail_verts, detail_faces, output_path, args, render):
    """Scratch-start: trace-reconstruct initial shape, then iterate."""
    from .iterative_transfer import scratch_transfer

    base, _ = os.path.splitext(output_path)
    output_dir = base + "_iterations" if render else None

    # Use tracing for the initial reconstruction
    use_tracing = getattr(args, "use_tracing", True)
    template_name = getattr(args, "template", None)

    t0 = time.time()
    result = scratch_transfer(
        detail_verts, detail_faces,
        max_iterations=args.max_iterations,
        min_improvement=args.min_improvement,
        patience=args.patience,
        use_vision=args.use_vision,
        vision_model=args.vision_model,
        output_dir=output_dir,
        render=render,
        use_tracing=use_tracing,
        template_name=template_name,
    )
    elapsed = time.time() - t0

    print(f"\nScratch-start complete in {elapsed:.2f}s")
    print(f"  Initial shape: {result['initial_shape_type']} "
          f"(quality={result['initial_quality']:.4f})")
    if result.get("template_name"):
        print(f"  Template: {result['template_name']}")
    if result.get("n_segments"):
        print(f"  Segments: {result['n_segments']}")
    print(f"  Iterations: {result['iterations']}")
    print(f"  Final mean distance: {result['distances'][-1]:.6f}")
    print(f"  Converged: {result['converged']}")

    _save_output(result["result_verts"], result["result_faces"], output_path)
    print(f"Saved: {output_path}")

    if render:
        _render_comparison_image(
            result["result_verts"], result["result_faces"],
            detail_verts, detail_faces,
            result["result_verts"], result["result_faces"],
            output_path,
        )


def cmd_trace(args):
    """Trace-reconstruct a mesh into CAD-like geometry (no iteration)."""
    from .tracing import trace_reconstruct_file

    print(f"Loading mesh: {args.input}")
    t0 = time.time()
    result = trace_reconstruct_file(
        args.input, args.output,
        template_name=args.template,
    )
    elapsed = time.time() - t0

    print(f"  Quality: {result['quality']:.4f}")
    print(f"  Segments: {result['n_segments']}")
    if result.get("template_name"):
        print(f"  Template: {result['template_name']}")
    print(f"  Done in {elapsed:.2f}s")
    print(f"Saved: {args.output}")

    if not args.no_render:
        from .stl_io import read_binary_stl
        from .render import render_comparison, HAS_MPL
        if HAS_MPL:
            input_v, input_f = read_binary_stl(args.input)
            output_v, output_f = read_binary_stl(args.output)
            base, _ = os.path.splitext(args.output)
            image_path = base + "_comparison.png"
            render_comparison(
                [(input_v, input_f), (output_v, output_f)],
                ["Input Mesh", "Traced CAD"],
                image_path,
                title="Tracing Reconstruction",
            )


def _transfer_iterative(plain_verts, plain_faces,
                          detail_verts, detail_faces,
                          output_path, args, render):
    """Default iterative transfer with convergence detection."""
    from .iterative_transfer import iterative_transfer
    import numpy as np

    base, _ = os.path.splitext(output_path)
    output_dir = base + "_iterations" if render else None

    print("Starting iterative detail transfer...")
    t0 = time.time()
    result = iterative_transfer(
        plain_verts, plain_faces,
        detail_verts, detail_faces,
        max_iterations=args.max_iterations,
        min_improvement=args.min_improvement,
        patience=args.patience,
        use_vision=args.use_vision,
        vision_model=args.vision_model,
        output_dir=output_dir,
        render=render,
    )
    elapsed = time.time() - t0

    # Summary
    from .alignment import find_correspondences
    _, _, bd = find_correspondences(plain_verts, detail_verts)
    bm = float(np.mean(bd))
    final_dist = result["distances"][-1]
    if bm > 0:
        imp = (1 - final_dist / bm) * 100
        print(f"\nIterative transfer complete in {elapsed:.2f}s")
        print(f"  Iterations: {result['iterations']}")
        print(f"  Distance: {bm:.4f} -> {final_dist:.4f} ({imp:.1f}% improvement)")
        print(f"  Converged: {result['converged']}")
    else:
        print(f"\nIterative transfer complete in {elapsed:.2f}s "
              f"({result['iterations']} iterations)")

    _save_output(result["result_verts"], result["result_faces"], output_path)
    print(f"Saved: {output_path}")

    if render:
        _render_comparison_image(plain_verts, plain_faces,
                                 detail_verts, detail_faces,
                                 result["result_verts"], result["result_faces"],
                                 output_path)


def _transfer_single_pass(plain_verts, plain_faces,
                           detail_verts, detail_faces,
                           output_path, args, render):
    """Legacy single-pass transfer (no iteration)."""
    from . import detail_transfer
    from .alignment import find_correspondences
    import numpy as np

    print("Transferring detail (single pass, with pre-alignment)...")
    t0 = time.time()
    result_verts = detail_transfer.transfer_mesh_detail_to_mesh(
        plain_verts, plain_faces, detail_verts, detail_faces
    )
    elapsed = time.time() - t0
    print(f"  Done in {elapsed:.2f}s")

    _, _, bd = find_correspondences(plain_verts, detail_verts)
    _, _, rd = find_correspondences(result_verts, detail_verts)
    bm, rm = float(np.mean(bd)), float(np.mean(rd))
    if bm > 0:
        imp = (1 - rm / bm) * 100
        print(f"  Distance: {bm:.4f} -> {rm:.4f} ({imp:.1f}% improvement)")

    _save_output(result_verts, plain_faces, output_path)
    print(f"Saved: {output_path}")

    if render:
        _render_comparison_image(plain_verts, plain_faces,
                                 detail_verts, detail_faces,
                                 result_verts, plain_faces,
                                 output_path)


def _transfer_to_step_brep(plain_cad_path, detail_mesh_path, output_path, args):
    """Transfer detail from mesh onto STEP model, producing a B-rep STEP file."""
    from .brep_transfer import transfer_to_step
    from .step_io import read_step, _read_step_shape, _tessellate_shape
    from .stl_io import read_binary_stl

    print(f"Loading plain CAD: {plain_cad_path}")
    print(f"Loading detail mesh: {detail_mesh_path}")

    plain_verts, plain_faces = read_step(plain_cad_path,
                                          linear_deflection=args.deflection)
    detail_verts, detail_faces = read_binary_stl(detail_mesh_path)

    _print_diagnostics(plain_verts, detail_verts)

    print("Analysing differences and building CAD operations...")
    t0 = time.time()
    result = transfer_to_step(
        plain_cad_path, detail_mesh_path, output_path,
        linear_deflection=args.deflection,
    )
    elapsed = time.time() - t0

    print(f"  Done in {elapsed:.2f}s — {result['n_operations']} B-rep operations applied")
    print(f"Saved STEP B-rep: {output_path}")

    if not args.no_render:
        try:
            result_shape = _read_step_shape(output_path)
            result_verts, result_faces = _tessellate_shape(
                result_shape, args.deflection, 0.5)
        except Exception:
            result_verts, result_faces = plain_verts, plain_faces

        _render_comparison_image(plain_verts, plain_faces,
                                 detail_verts, detail_faces,
                                 result_verts, result_faces,
                                 output_path)


def _save_output(verts, faces, output_path):
    """Save result mesh to the specified output format."""
    ext = os.path.splitext(output_path)[1].lower()
    if ext == ".stl":
        from .stl_io import write_binary_stl
        write_binary_stl(output_path, verts, faces)
    elif ext in (".step", ".stp"):
        from .step_io import write_step
        write_step(output_path, verts, faces)
    elif ext == ".fcstd":
        from . import mesh_io, cad_io
        fc_mesh = mesh_io.numpy_to_mesh(verts, faces)
        doc = cad_io.new_document("MeshXCAD_Result")
        mesh_obj = doc.addObject("Mesh::Feature", "TransferredDetail")
        mesh_obj.Mesh = fc_mesh
        doc.recompute()
        cad_io.save_cad(doc, output_path)
        cad_io.close_document(doc)
    else:
        print(f"Warning: unknown extension '{ext}', writing as STL")
        from .stl_io import write_binary_stl
        write_binary_stl(output_path, verts, faces)


def main():
    parser = argparse.ArgumentParser(
        prog="meshxcad",
        description="MeshXCAD — bidirectional detail transfer between meshes and CAD models",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # transfer command
    p_transfer = subparsers.add_parser(
        "transfer",
        help="Transfer detail from a detailed mesh onto a plain model",
    )
    p_transfer.add_argument(
        "--plain", default=None,
        help="Path to the plain model (.step, .stp, .fcstd, or .stl). "
             "If omitted, an initial shape is auto-reconstructed from the "
             "detail mesh (scratch-start mode).",
    )
    p_transfer.add_argument(
        "--detail", required=True,
        help="Path to the detailed/featured mesh (.stl)",
    )
    p_transfer.add_argument(
        "--output", required=True,
        help="Output path (.stl, .step, .stp, or .fcstd)",
    )
    p_transfer.add_argument(
        "--deflection", type=float, default=0.05,
        help="Tessellation linear deflection for CAD models (default: 0.05)",
    )
    p_transfer.add_argument(
        "--no-render", action="store_true", default=False,
        help="Skip generating comparison images",
    )

    # Iteration control
    p_transfer.add_argument(
        "--single-pass", action="store_true", default=False,
        help="Disable iterative refinement (legacy single-pass mode)",
    )
    p_transfer.add_argument(
        "--max-iterations", type=int, default=20,
        help="Maximum number of refinement iterations (default: 20)",
    )
    p_transfer.add_argument(
        "--min-improvement", type=float, default=0.001,
        help="Minimum fractional improvement to continue iterating (default: 0.001)",
    )
    p_transfer.add_argument(
        "--patience", type=int, default=3,
        help="Stop after this many iterations with no improvement (default: 3)",
    )

    # Vision LLM guidance
    p_transfer.add_argument(
        "--use-vision", action="store_true", default=False,
        help="Use a vision LLM to guide iterative refinement. "
             "Requires LOCAL_OPENAI_KEY and LOCAL_OPENAI_URL env vars.",
    )
    p_transfer.add_argument(
        "--vision-model", type=str, default=None,
        help="Model name for the vision LLM (default: from LOCAL_OPENAI_MODEL "
             "env var, or 'gpt-4o')",
    )

    # Tracing/template control
    p_transfer.add_argument(
        "--use-tracing", action="store_true", default=True,
        help="Use tracing-based reconstruction for scratch-start (default: True)",
    )
    p_transfer.add_argument(
        "--no-tracing", action="store_true", default=False,
        help="Disable tracing; use basic reconstruct_cad for scratch-start",
    )
    p_transfer.add_argument(
        "--template", type=str, default=None,
        help="Object template name to guide segmentation (e.g. 'chair', 'dead_tree')",
    )

    p_transfer.set_defaults(func=cmd_transfer)

    # trace command — standalone tracing reconstruction
    p_trace = subparsers.add_parser(
        "trace",
        help="Trace-reconstruct a mesh into CAD-like geometry via segmentation",
    )
    p_trace.add_argument(
        "--input", required=True,
        help="Path to the input mesh (.stl)",
    )
    p_trace.add_argument(
        "--output", required=True,
        help="Output path (.stl)",
    )
    p_trace.add_argument(
        "--template", type=str, default=None,
        help="Object template name (e.g. 'chair', 'dead_tree', 'bicycle')",
    )
    p_trace.add_argument(
        "--no-render", action="store_true", default=False,
        help="Skip generating comparison image",
    )
    p_trace.set_defaults(func=cmd_trace)

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        sys.exit(1)

    args.func(args)


if __name__ == "__main__":
    main()
