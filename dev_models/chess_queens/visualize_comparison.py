#!/usr/bin/env python3
"""Render side-by-side comparison of original STL vs CAD reconstruction.

Generates a PNG image with two 3D views:
  Left:  Original target mesh (duggan_black_queen.stl)
  Right: CAD reconstruction output

Usage:
    python3 dev_models/chess_queens/visualize_comparison.py
"""

import sys
import os
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless backend
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from meshxcad.stl_io import read_binary_stl, write_binary_stl
from meshxcad.reconstruct import reconstruct_cad
from meshxcad.general_align import hausdorff_distance


def compute_face_normals(vertices, faces):
    """Compute per-face normals for shading."""
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]
    normals = np.cross(v1 - v0, v2 - v0)
    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    norms[norms < 1e-12] = 1.0
    return normals / norms


def render_mesh(ax, vertices, faces, title, color_base, elev=25, azim=-60):
    """Render a mesh on a matplotlib 3D axis with face shading."""
    v = np.asarray(vertices, dtype=np.float64)
    f = np.asarray(faces, dtype=int)

    # Subsample faces for rendering performance (max ~15000 triangles)
    max_faces = 15000
    if len(f) > max_faces:
        idx = np.linspace(0, len(f) - 1, max_faces, dtype=int)
        f_render = f[idx]
    else:
        f_render = f

    # Compute face normals for Lambertian shading
    fn = compute_face_normals(v, f_render)
    light_dir = np.array([0.3, -0.5, 0.8])
    light_dir /= np.linalg.norm(light_dir)
    intensity = np.clip(np.dot(fn, light_dir), 0.15, 1.0)

    # Build polygon collection
    triangles = v[f_render]
    # Face colors based on shading
    base = np.array(matplotlib.colors.to_rgb(color_base))
    face_colors = np.outer(intensity, base)
    face_colors = np.clip(face_colors, 0, 1)
    # Add alpha column
    face_colors = np.column_stack([face_colors, np.ones(len(face_colors))])

    poly = Poly3DCollection(triangles, linewidths=0.0, edgecolors='none')
    poly.set_facecolor(face_colors)
    ax.add_collection3d(poly)

    # Set axis limits
    center = v.mean(axis=0)
    span = max(v.max(axis=0) - v.min(axis=0)) * 0.55
    ax.set_xlim(center[0] - span, center[0] + span)
    ax.set_ylim(center[1] - span, center[1] + span)
    ax.set_zlim(center[2] - span, center[2] + span)

    ax.set_title(title, fontsize=13, fontweight='bold', pad=10)
    ax.view_init(elev=elev, azim=azim)
    ax.set_axis_off()


def compute_error_colors(target_v, cad_v, target_f, bbox_diag):
    """Compute per-face error coloring for the error map."""
    from scipy.spatial import KDTree

    tree_cad = KDTree(cad_v)
    dists, _ = tree_cad.query(target_v)

    # Per-vertex error normalized by bbox_diag
    err_norm = dists / bbox_diag

    # Per-face error = mean of vertex errors
    face_err = np.mean(err_norm[target_f], axis=1)
    return face_err


def render_error_map(ax, vertices, faces, face_errors, title, elev=25, azim=-60):
    """Render a mesh colored by per-face reconstruction error."""
    v = np.asarray(vertices, dtype=np.float64)
    f = np.asarray(faces, dtype=int)

    max_faces = 15000
    if len(f) > max_faces:
        idx = np.linspace(0, len(f) - 1, max_faces, dtype=int)
        f_render = f[idx]
        err_render = face_errors[idx]
    else:
        f_render = f
        err_render = face_errors

    triangles = v[f_render]

    # Colormap: green (0 error) → yellow → red (high error)
    cmap = plt.cm.RdYlGn_r
    # Normalize errors to [0, 1] with saturation at 5% of bbox_diag
    err_clipped = np.clip(err_render / 0.02, 0, 1)
    face_colors = cmap(err_clipped)

    poly = Poly3DCollection(triangles, linewidths=0.0, edgecolors='none')
    poly.set_facecolor(face_colors)
    ax.add_collection3d(poly)

    center = v.mean(axis=0)
    span = max(v.max(axis=0) - v.min(axis=0)) * 0.55
    ax.set_xlim(center[0] - span, center[0] + span)
    ax.set_ylim(center[1] - span, center[1] + span)
    ax.set_zlim(center[2] - span, center[2] + span)

    ax.set_title(title, fontsize=13, fontweight='bold', pad=10)
    ax.view_init(elev=elev, azim=azim)
    ax.set_axis_off()


def main():
    stl_path = os.path.join(os.path.dirname(__file__), "duggan_black_queen.stl")
    out_png = os.path.join(os.path.dirname(__file__), "duggan_black_queen_comparison.png")

    print("Loading duggan_black_queen.stl...")
    target_v, target_f = read_binary_stl(stl_path)
    bbox_diag = float(np.linalg.norm(target_v.max(axis=0) - target_v.min(axis=0)))
    print(f"  {len(target_v)} verts, {len(target_f)} faces, bbox_diag={bbox_diag:.2f}")

    print("Running reconstruct_cad...")
    t0 = time.time()
    result = reconstruct_cad(target_v, target_f)
    elapsed = time.time() - t0
    cad_v = result["cad_vertices"]
    cad_f = result["cad_faces"]
    quality = result.get("quality", 0.0)
    shape_type = result["shape_type"]
    print(f"  shape_type={shape_type}, quality={quality:.4f}, elapsed={elapsed:.1f}s")

    # Compute accuracy
    hd = hausdorff_distance(cad_v, target_v)
    accuracy = max(0.0, 1.0 - hd["mean_symmetric"] / bbox_diag * 5)
    print(f"  accuracy={accuracy:.4f}")

    # Compute per-face errors for error map
    face_errors = compute_error_colors(target_v, cad_v, target_f, bbox_diag)

    # Create figure with 3 panels (2 views + error map)
    fig = plt.figure(figsize=(20, 7))
    fig.patch.set_facecolor('#1a1a2e')

    # Panel 1: Original STL
    ax1 = fig.add_subplot(131, projection='3d', facecolor='#16213e')
    render_mesh(ax1, target_v, target_f,
                f"Original STL\n({len(target_v):,} verts)",
                color_base='#4cc9f0', elev=20, azim=-55)

    # Panel 2: CAD Reconstruction
    ax2 = fig.add_subplot(132, projection='3d', facecolor='#16213e')
    render_mesh(ax2, cad_v, cad_f,
                f"CAD Reconstruction\n({len(cad_v):,} verts, {shape_type})",
                color_base='#f72585', elev=20, azim=-55)

    # Panel 3: Error map on original mesh
    ax3 = fig.add_subplot(133, projection='3d', facecolor='#16213e')
    render_error_map(ax3, target_v, target_f, face_errors,
                     f"Error Map\n(accuracy={accuracy*100:.1f}%)",
                     elev=20, azim=-55)

    # Title
    fig.suptitle(
        f"Duggan Black Queen — STL vs CAD Reconstruction",
        fontsize=16, fontweight='bold', color='white', y=0.98
    )

    # Stats annotation
    stats_text = (
        f"Mean symmetric dist: {hd['mean_symmetric']:.4f}  |  "
        f"Hausdorff max: {hd['hausdorff']:.4f}  |  "
        f"Accuracy: {accuracy*100:.2f}%"
    )
    fig.text(0.5, 0.02, stats_text, ha='center', fontsize=11,
             color='#e0e0e0', family='monospace')

    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    plt.savefig(out_png, dpi=150, bbox_inches='tight',
                facecolor=fig.get_facecolor(), edgecolor='none')
    print(f"\nSaved → {out_png}")
    plt.close()


if __name__ == "__main__":
    main()
