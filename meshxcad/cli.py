"""Command-line interface for MeshXCAD detail transfer."""

import argparse
import os
import sys
import time


def cmd_transfer(args):
    """Run detail transfer from a detailed mesh onto a plain CAD model."""
    plain_path = args.plain
    detail_path = args.detail
    output_path = args.output

    plain_ext = os.path.splitext(plain_path)[1].lower()
    detail_ext = os.path.splitext(detail_path)[1].lower()

    # Determine which direction: mesh->CAD or CAD->mesh
    cad_exts = {".step", ".stp", ".iges", ".igs", ".fcstd"}
    mesh_exts = {".stl", ".obj", ".ply"}

    plain_is_cad = plain_ext in cad_exts
    detail_is_mesh = detail_ext in mesh_exts

    if plain_is_cad and detail_is_mesh:
        _transfer_mesh_detail_to_cad(plain_path, detail_path, output_path, args)
    elif plain_ext in mesh_exts and detail_ext in mesh_exts:
        _transfer_mesh_to_mesh(plain_path, detail_path, output_path, args)
    else:
        print(f"Error: unsupported file combination: {plain_ext} + {detail_ext}")
        print("Supported: plain=.step/.stp/.fcstd + detail=.stl")
        print("       or: plain=.stl + detail=.stl")
        sys.exit(1)


def _transfer_mesh_detail_to_cad(plain_cad_path, detail_mesh_path, output_path, args):
    """Transfer detail from STL mesh onto a STEP/CAD model."""
    from . import cad_io, mesh_io, detail_transfer
    from .stl_io import read_binary_stl

    print(f"Loading plain CAD: {plain_cad_path}")
    doc = cad_io.load_cad(plain_cad_path)
    shape = cad_io.cad_to_shape(doc)
    plain_fc_mesh = cad_io.shape_to_mesh(
        shape, linear_deflection=args.deflection
    )
    plain_verts, plain_faces = mesh_io.mesh_to_numpy(plain_fc_mesh)
    cad_io.close_document(doc)
    print(f"  Tessellated: {len(plain_verts)} vertices, {len(plain_faces)} faces")

    print(f"Loading detail mesh: {detail_mesh_path}")
    detail_ext = os.path.splitext(detail_mesh_path)[1].lower()
    if detail_ext == ".stl":
        detail_verts, detail_faces = read_binary_stl(detail_mesh_path)
    else:
        detail_fc_mesh = mesh_io.load_mesh(detail_mesh_path)
        detail_verts, detail_faces = mesh_io.mesh_to_numpy(detail_fc_mesh)
    print(f"  Loaded: {len(detail_verts)} vertices, {len(detail_faces)} faces")

    print("Transferring detail...")
    t0 = time.time()
    result_verts = detail_transfer.transfer_mesh_detail_to_mesh(
        plain_verts, plain_faces, detail_verts, detail_faces
    )
    elapsed = time.time() - t0
    print(f"  Done in {elapsed:.2f}s")

    # Compute improvement metric
    from .alignment import find_correspondences
    import numpy as np
    _, _, bd = find_correspondences(plain_verts, detail_verts)
    _, _, rd = find_correspondences(result_verts, detail_verts)
    bm, rm = float(np.mean(bd)), float(np.mean(rd))
    if bm > 0:
        imp = (1 - rm / bm) * 100
        print(f"  Distance: {bm:.4f} -> {rm:.4f} ({imp:.1f}% improvement)")

    _save_output(result_verts, plain_faces, output_path)
    print(f"Saved: {output_path}")


def _transfer_mesh_to_mesh(plain_path, detail_path, output_path, args):
    """Transfer detail between two STL meshes."""
    from .stl_io import read_binary_stl
    from . import detail_transfer
    import numpy as np

    print(f"Loading plain mesh: {plain_path}")
    plain_verts, plain_faces = read_binary_stl(plain_path)
    print(f"  Loaded: {len(plain_verts)} vertices, {len(plain_faces)} faces")

    print(f"Loading detail mesh: {detail_path}")
    detail_verts, detail_faces = read_binary_stl(detail_path)
    print(f"  Loaded: {len(detail_verts)} vertices, {len(detail_faces)} faces")

    print("Transferring detail...")
    t0 = time.time()
    result_verts = detail_transfer.transfer_mesh_detail_to_mesh(
        plain_verts, plain_faces, detail_verts, detail_faces
    )
    elapsed = time.time() - t0
    print(f"  Done in {elapsed:.2f}s")

    from .alignment import find_correspondences
    _, _, bd = find_correspondences(plain_verts, detail_verts)
    _, _, rd = find_correspondences(result_verts, detail_verts)
    bm, rm = float(np.mean(bd)), float(np.mean(rd))
    if bm > 0:
        imp = (1 - rm / bm) * 100
        print(f"  Distance: {bm:.4f} -> {rm:.4f} ({imp:.1f}% improvement)")

    _save_output(result_verts, plain_faces, output_path)
    print(f"Saved: {output_path}")


def _save_output(verts, faces, output_path):
    """Save result mesh to the specified output format."""
    ext = os.path.splitext(output_path)[1].lower()
    if ext == ".stl":
        from .stl_io import write_binary_stl
        write_binary_stl(output_path, verts, faces)
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
        # Default to STL
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
        "--plain", required=True,
        help="Path to the plain model (.step, .stp, .fcstd, or .stl)",
    )
    p_transfer.add_argument(
        "--detail", required=True,
        help="Path to the detailed/featured mesh (.stl)",
    )
    p_transfer.add_argument(
        "--output", required=True,
        help="Output path (.stl or .fcstd)",
    )
    p_transfer.add_argument(
        "--deflection", type=float, default=0.05,
        help="Tessellation linear deflection for CAD models (default: 0.05)",
    )
    p_transfer.set_defaults(func=cmd_transfer)

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        sys.exit(1)

    args.func(args)


if __name__ == "__main__":
    main()
