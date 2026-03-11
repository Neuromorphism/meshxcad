"""Mesh I/O and basic operations using FreeCAD's Mesh module."""

import numpy as np


def load_mesh(filepath):
    """Load a mesh from STL file. Returns FreeCAD Mesh object."""
    import Mesh
    return Mesh.Mesh(filepath)


def save_mesh(mesh, filepath):
    """Save a FreeCAD Mesh object to STL file."""
    mesh.write(filepath)


def mesh_to_numpy(mesh):
    """Convert FreeCAD Mesh to numpy arrays of vertices and faces.

    Returns:
        vertices: (N, 3) array of vertex positions
        faces: (M, 3) array of triangle vertex indices
    """
    points = np.array([[p.x, p.y, p.z] for p in mesh.Points])
    facets = np.array([[f.PointIndices[0], f.PointIndices[1], f.PointIndices[2]]
                       for f in mesh.Facets])
    return points, facets


def numpy_to_mesh(vertices, faces):
    """Convert numpy arrays back to a FreeCAD Mesh.

    Args:
        vertices: (N, 3) array of vertex positions
        faces: (M, 3) array of triangle vertex indices

    Returns:
        FreeCAD Mesh object
    """
    import Mesh
    facets = []
    for f in faces:
        v0 = vertices[f[0]]
        v1 = vertices[f[1]]
        v2 = vertices[f[2]]
        facets.append([
            (float(v0[0]), float(v0[1]), float(v0[2])),
            (float(v1[0]), float(v1[1]), float(v1[2])),
            (float(v2[0]), float(v2[1]), float(v2[2])),
        ])
    return Mesh.Mesh(facets)


def compute_face_normals(vertices, faces):
    """Compute per-face normals for a triangle mesh."""
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]
    normals = np.cross(v1 - v0, v2 - v0)
    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return normals / norms


def compute_vertex_normals(vertices, faces):
    """Compute per-vertex normals by averaging adjacent face normals."""
    face_normals = compute_face_normals(vertices, faces)
    vertex_normals = np.zeros_like(vertices)
    for i in range(3):
        np.add.at(vertex_normals, faces[:, i], face_normals)
    norms = np.linalg.norm(vertex_normals, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return vertex_normals / norms
