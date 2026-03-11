"""STEP file I/O — read STEP/STP/IGES files via OpenCASCADE (OCP).

Tessellates the CAD solid into a triangle mesh for use with the
meshxcad optimisation pipeline.
"""

import numpy as np


def read_step(filepath, linear_deflection=0.1, angular_deflection=0.5):
    """Read a STEP or IGES file and tessellate it to a triangle mesh.

    Args:
        filepath: path to .step, .stp, or .iges file
        linear_deflection: mesh resolution (smaller = finer, default 0.1)
        angular_deflection: angular resolution in radians (default 0.5)

    Returns:
        vertices: (N, 3) float64 array
        faces: (M, 3) int64 array of triangle indices
    """
    ext = filepath.rsplit(".", 1)[-1].lower() if "." in filepath else ""

    if ext in ("step", "stp"):
        shape = _read_step_shape(filepath)
    elif ext in ("iges", "igs"):
        shape = _read_iges_shape(filepath)
    else:
        raise ValueError(
            f"Unsupported CAD format: .{ext}  (use .step, .stp, .iges, .igs)")

    return _tessellate_shape(shape, linear_deflection, angular_deflection)


def _read_step_shape(filepath):
    """Read a STEP file and return the OCC shape."""
    from OCP.STEPControl import STEPControl_Reader
    from OCP.IFSelect import IFSelect_RetDone

    reader = STEPControl_Reader()
    status = reader.ReadFile(filepath)
    if status != IFSelect_RetDone:
        raise IOError(f"Failed to read STEP file: {filepath} (status={status})")

    reader.TransferRoots()
    shape = reader.OneShape()
    return shape


def _read_iges_shape(filepath):
    """Read an IGES file and return the OCC shape."""
    from OCP.IGESControl import IGESControl_Reader
    from OCP.IFSelect import IFSelect_RetDone

    reader = IGESControl_Reader()
    status = reader.ReadFile(filepath)
    if status != IFSelect_RetDone:
        raise IOError(f"Failed to read IGES file: {filepath} (status={status})")

    reader.TransferRoots()
    shape = reader.OneShape()
    return shape


def _tessellate_shape(shape, linear_deflection, angular_deflection):
    """Tessellate an OCC shape into (vertices, faces) arrays."""
    from OCP.BRepMesh import BRepMesh_IncrementalMesh
    from OCP.TopExp import TopExp_Explorer
    from OCP.TopAbs import TopAbs_FACE
    from OCP.BRep import BRep_Tool
    from OCP.TopLoc import TopLoc_Location
    from OCP.TopoDS import TopoDS

    # Tessellate the shape
    BRepMesh_IncrementalMesh(shape, linear_deflection, False,
                             angular_deflection, True)

    all_verts = []
    all_faces = []
    vert_offset = 0

    explorer = TopExp_Explorer(shape, TopAbs_FACE)
    while explorer.More():
        face = TopoDS.Face_s(explorer.Current())
        location = TopLoc_Location()
        triangulation = BRep_Tool.Triangulation_s(face, location)

        if triangulation is not None:
            n_nodes = triangulation.NbNodes()
            n_tris = triangulation.NbTriangles()

            # Extract vertices
            trsf = location.Transformation()
            for i in range(1, n_nodes + 1):
                node = triangulation.Node(i)
                pt = node.Transformed(trsf)
                all_verts.append([pt.X(), pt.Y(), pt.Z()])

            # Extract triangles (1-indexed in OCC)
            for i in range(1, n_tris + 1):
                tri = triangulation.Triangle(i)
                i1, i2, i3 = tri.Get()
                all_faces.append([
                    i1 - 1 + vert_offset,
                    i2 - 1 + vert_offset,
                    i3 - 1 + vert_offset,
                ])

            vert_offset += n_nodes

        explorer.Next()

    if not all_verts:
        raise ValueError(f"No geometry found in shape (empty tessellation)")

    vertices = np.array(all_verts, dtype=np.float64)
    faces = np.array(all_faces, dtype=np.int64)

    # Deduplicate coincident vertices (from shared edges between faces)
    vertices, faces = _deduplicate(vertices, faces)

    return vertices, faces


def _deduplicate(vertices, faces, tol=1e-8):
    """Merge coincident vertices within tolerance."""
    # Round to tolerance to group nearby vertices
    rounded = np.round(vertices / tol) * tol
    unique, inverse = np.unique(rounded, axis=0, return_inverse=True)
    new_faces = inverse[faces]
    return unique, new_faces


def write_step(filepath, vertices, faces):
    """Write a triangle mesh as a STEP file (as a sewed shell).

    This creates a TopoDS_Shell from the triangles and writes it as STEP.
    Useful for exporting optimised CAD meshes back to STEP format.

    Args:
        filepath: output .step path
        vertices: (N, 3) array
        faces: (M, 3) array of triangle indices
    """
    from OCP.STEPControl import STEPControl_Writer, STEPControl_AsIs
    from OCP.IFSelect import IFSelect_RetDone
    from OCP.BRepBuilderAPI import BRepBuilderAPI_MakePolygon, BRepBuilderAPI_Sewing
    from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeFace
    from OCP.gp import gp_Pnt
    from OCP.TopoDS import TopoDS

    sewing = BRepBuilderAPI_Sewing(1e-6)

    for tri in faces:
        v0 = vertices[tri[0]]
        v1 = vertices[tri[1]]
        v2 = vertices[tri[2]]

        wire = BRepBuilderAPI_MakePolygon(
            gp_Pnt(*v0), gp_Pnt(*v1), gp_Pnt(*v2), True)
        face_maker = BRepBuilderAPI_MakeFace(wire.Wire(), True)
        if face_maker.IsDone():
            sewing.Add(face_maker.Face())

    sewing.Perform()
    shape = sewing.SewedShape()

    writer = STEPControl_Writer()
    writer.Transfer(shape, STEPControl_AsIs)
    status = writer.Write(filepath)
    if status != IFSelect_RetDone:
        raise IOError(f"Failed to write STEP file: {filepath}")
