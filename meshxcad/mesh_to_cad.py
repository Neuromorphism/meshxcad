"""Mesh-to-CAD detail transfer: modify a parametric CAD body using mesh detail."""

import numpy as np

from . import cad_io, mesh_io, alignment, detail_transfer


def transfer_detail(plain_cad_path, featured_mesh_path, output_path,
                    linear_deflection=0.05, smoothing=0.0):
    """Add detail from a featured mesh onto a plain CAD model.

    Workflow:
    1. Load the plain CAD and tessellate it to get surface sample points
    2. Load the featured mesh
    3. Compute displacement field from plain CAD surface to featured mesh surface
    4. Apply displacement to the CAD tessellation and create a new mesh from it
       (True parametric modification of CAD requires feature recognition which is
        a future enhancement; for now we output a high-quality mesh)
    5. Save result as FreeCAD document with a Mesh feature, or as STL

    Args:
        plain_cad_path: path to plain FreeCAD document
        featured_mesh_path: path to featured STL mesh
        output_path: path to write result (.FCStd or .stl)
        linear_deflection: tessellation quality for the plain CAD
        smoothing: RBF smoothing parameter for displacement interpolation

    Returns:
        output_path
    """
    # Load and tessellate the plain CAD
    doc = cad_io.load_cad(plain_cad_path)
    shape = cad_io.cad_to_shape(doc)
    plain_fc_mesh = cad_io.shape_to_mesh(shape, linear_deflection=linear_deflection)
    plain_verts, plain_faces = mesh_io.mesh_to_numpy(plain_fc_mesh)
    cad_io.close_document(doc)

    # Load the featured mesh
    featured_fc_mesh = mesh_io.load_mesh(featured_mesh_path)
    featured_verts, featured_faces = mesh_io.mesh_to_numpy(featured_fc_mesh)

    # Compute displacement and apply
    result_verts = detail_transfer.transfer_mesh_detail_to_mesh(
        plain_verts, plain_faces, featured_verts, featured_faces
    )

    # Save result
    if output_path.endswith(".FCStd"):
        _save_as_freecad(result_verts, plain_faces, output_path)
    else:
        result_mesh = mesh_io.numpy_to_mesh(result_verts, plain_faces)
        mesh_io.save_mesh(result_mesh, output_path)

    return output_path


def transfer_detail_numpy(plain_verts, plain_faces, featured_verts, featured_faces,
                           smoothing=0.0):
    """Transfer detail using numpy arrays directly.

    Args:
        plain_verts: (N, 3) plain CAD tessellation vertices
        plain_faces: (F, 3) plain CAD tessellation faces
        featured_verts: (M, 3) featured mesh vertices
        featured_faces: (G, 3) featured mesh faces
        smoothing: RBF smoothing for displacement interpolation

    Returns:
        result_verts: (N, 3) modified vertices
    """
    return detail_transfer.transfer_mesh_detail_to_mesh(
        plain_verts, plain_faces, featured_verts, featured_faces
    )


def _save_as_freecad(vertices, faces, filepath):
    """Save a mesh result as a FreeCAD document with a Mesh::Feature object."""
    import FreeCAD
    import Mesh

    fc_mesh = mesh_io.numpy_to_mesh(vertices, faces)
    doc = cad_io.new_document("MeshXCAD_Result")
    mesh_obj = doc.addObject("Mesh::Feature", "TransferredDetail")
    mesh_obj.Mesh = fc_mesh
    doc.recompute()
    cad_io.save_cad(doc, filepath)
    cad_io.close_document(doc)
