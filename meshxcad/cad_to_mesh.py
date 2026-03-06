"""CAD-to-Mesh detail transfer: apply CAD features onto a plain mesh."""

import numpy as np

from . import cad_io, mesh_io, alignment, detail_transfer


def transfer_detail(plain_mesh_path, featured_cad_path, output_path,
                    linear_deflection=0.05):
    """Add detail from a featured CAD model onto a plain mesh.

    Workflow:
    1. Load the plain mesh
    2. Load the featured CAD and tessellate it
    3. Compute displacement field (featured CAD surface - plain mesh surface)
    4. Apply displacements to the plain mesh
    5. Save result

    Args:
        plain_mesh_path: path to plain STL mesh
        featured_cad_path: path to featured FreeCAD document
        output_path: path to write the resulting STL mesh
        linear_deflection: tessellation quality for CAD→mesh conversion

    Returns:
        output_path
    """
    # Load plain mesh
    plain_fc_mesh = mesh_io.load_mesh(plain_mesh_path)
    plain_verts, plain_faces = mesh_io.mesh_to_numpy(plain_fc_mesh)

    # Load and tessellate the featured CAD
    doc = cad_io.load_cad(featured_cad_path)
    shape = cad_io.cad_to_shape(doc)
    featured_fc_mesh = cad_io.shape_to_mesh(shape, linear_deflection=linear_deflection)
    featured_verts, featured_faces = mesh_io.mesh_to_numpy(featured_fc_mesh)
    cad_io.close_document(doc)

    # Compute and apply displacement
    result_verts = detail_transfer.transfer_mesh_detail_to_mesh(
        plain_verts, plain_faces, featured_verts, featured_faces
    )

    # Build output mesh and save
    result_mesh = mesh_io.numpy_to_mesh(result_verts, plain_faces)
    mesh_io.save_mesh(result_mesh, output_path)
    return output_path


def transfer_detail_numpy(plain_verts, plain_faces, featured_verts, featured_faces):
    """Transfer detail using numpy arrays directly (no file I/O).

    Args:
        plain_verts: (N, 3) plain mesh vertices
        plain_faces: (F, 3) plain mesh faces
        featured_verts: (M, 3) featured mesh vertices (from CAD tessellation)
        featured_faces: (G, 3) featured mesh faces

    Returns:
        result_verts: (N, 3) modified mesh vertices
    """
    return detail_transfer.transfer_mesh_detail_to_mesh(
        plain_verts, plain_faces, featured_verts, featured_faces
    )
