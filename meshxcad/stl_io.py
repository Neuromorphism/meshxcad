"""Pure-numpy STL I/O — no FreeCAD required.

Supports binary STL read/write for standalone testing.
"""

import struct
import numpy as np


def write_binary_stl(filepath, vertices, faces):
    """Write a binary STL file.

    Args:
        filepath: output path
        vertices: (N, 3) array of vertex positions
        faces: (M, 3) array of triangle vertex indices
    """
    n_triangles = len(faces)

    with open(filepath, "wb") as f:
        # 80-byte header
        f.write(b"\x00" * 80)
        # Triangle count
        f.write(struct.pack("<I", n_triangles))

        for face in faces:
            v0 = vertices[face[0]]
            v1 = vertices[face[1]]
            v2 = vertices[face[2]]

            # Compute normal
            edge1 = v1 - v0
            edge2 = v2 - v0
            normal = np.cross(edge1, edge2)
            norm_len = np.linalg.norm(normal)
            if norm_len > 0:
                normal = normal / norm_len

            # Normal vector
            f.write(struct.pack("<fff", *normal))
            # Three vertices
            f.write(struct.pack("<fff", *v0))
            f.write(struct.pack("<fff", *v1))
            f.write(struct.pack("<fff", *v2))
            # Attribute byte count
            f.write(struct.pack("<H", 0))


def read_binary_stl(filepath):
    """Read a binary STL file.

    Returns:
        vertices: (N, 3) array (deduplicated)
        faces: (M, 3) array of triangle vertex indices
    """
    with open(filepath, "rb") as f:
        header = f.read(80)
        n_triangles = struct.unpack("<I", f.read(4))[0]

        raw_verts = []
        for _ in range(n_triangles):
            normal = struct.unpack("<fff", f.read(12))
            v0 = struct.unpack("<fff", f.read(12))
            v1 = struct.unpack("<fff", f.read(12))
            v2 = struct.unpack("<fff", f.read(12))
            attr = struct.unpack("<H", f.read(2))
            raw_verts.extend([v0, v1, v2])

    raw_verts = np.array(raw_verts)

    # Deduplicate vertices
    unique_verts, inverse = np.unique(raw_verts, axis=0, return_inverse=True)
    faces = inverse.reshape(-1, 3)

    return unique_verts, faces
