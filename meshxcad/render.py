"""Simple matplotlib-based renderer for mesh visualization.

Produces multi-view renderings (front, side, top, perspective) of meshes
so we can visually compare models without a full 3D viewer.
"""

import numpy as np
import os

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    HAS_MPL = True
except ImportError:
    HAS_MPL = False


def render_mesh(vertices, faces, output_path, title="Mesh", figsize=(16, 12)):
    """Render a mesh from multiple viewpoints and save as an image.

    Produces a 2x2 grid: front view, side view, perspective, and top view.

    Args:
        vertices: (N, 3) array of vertex positions
        faces: (M, 3) array of triangle indices
        output_path: path to save the rendered image (PNG)
        title: title for the figure
        figsize: figure size in inches
    """
    if not HAS_MPL:
        print("matplotlib not available, skipping render")
        return

    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)

    fig = plt.figure(figsize=figsize)
    fig.suptitle(title, fontsize=16, fontweight="bold")

    views = [
        ("Front View", 0, 0),
        ("Side View", 0, 90),
        ("Perspective", 25, 45),
        ("Top View", 90, 0),
    ]

    # Subsample faces for rendering performance
    max_faces = 5000
    if len(faces) > max_faces:
        indices = np.random.RandomState(42).choice(len(faces), max_faces, replace=False)
        render_faces = faces[indices]
    else:
        render_faces = faces

    # Compute bounds for consistent axis limits
    margin = 10
    x_range = [vertices[:, 0].min() - margin, vertices[:, 0].max() + margin]
    y_range = [vertices[:, 1].min() - margin, vertices[:, 1].max() + margin]
    z_range = [vertices[:, 2].min() - margin, vertices[:, 2].max() + margin]

    # Make ranges equal for proper aspect ratio
    max_range = max(x_range[1] - x_range[0], y_range[1] - y_range[0], z_range[1] - z_range[0])
    mid_x = sum(x_range) / 2
    mid_y = sum(y_range) / 2
    mid_z = sum(z_range) / 2

    for idx, (view_name, elev, azim) in enumerate(views):
        ax = fig.add_subplot(2, 2, idx + 1, projection="3d")
        ax.set_title(view_name, fontsize=12)
        ax.view_init(elev=elev, azim=azim)

        # Build triangles for rendering
        triangles = vertices[render_faces]
        collection = Poly3DCollection(triangles, alpha=0.6, linewidth=0.1,
                                       edgecolor="gray")
        collection.set_facecolor("steelblue")
        ax.add_collection3d(collection)

        ax.set_xlim(mid_x - max_range / 2, mid_x + max_range / 2)
        ax.set_ylim(mid_y - max_range / 2, mid_y + max_range / 2)
        ax.set_zlim(mid_z - max_range / 2, mid_z + max_range / 2)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Rendered: {output_path}")


def render_comparison(meshes, labels, output_path, title="Comparison",
                       figsize=(20, 6)):
    """Render multiple meshes side by side for comparison.

    Args:
        meshes: list of (vertices, faces) tuples
        labels: list of labels for each mesh
        output_path: path to save the image
        title: figure title
        figsize: figure size
    """
    if not HAS_MPL:
        print("matplotlib not available, skipping render")
        return

    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)

    n = len(meshes)
    fig = plt.figure(figsize=figsize)
    fig.suptitle(title, fontsize=16, fontweight="bold")

    # Find global bounds
    all_verts = np.vstack([v for v, _ in meshes])
    margin = 10
    max_range = max(
        all_verts[:, 0].max() - all_verts[:, 0].min(),
        all_verts[:, 1].max() - all_verts[:, 1].min(),
        all_verts[:, 2].max() - all_verts[:, 2].min(),
    ) + 2 * margin
    mid = all_verts.mean(axis=0)

    colors = ["steelblue", "coral", "mediumseagreen", "goldenrod"]

    for i, ((verts, faces), label) in enumerate(zip(meshes, labels)):
        ax = fig.add_subplot(1, n, i + 1, projection="3d")
        ax.set_title(label, fontsize=12)
        ax.view_init(elev=25, azim=45)

        max_faces = 4000
        if len(faces) > max_faces:
            idx = np.random.RandomState(42).choice(len(faces), max_faces, replace=False)
            render_faces = faces[idx]
        else:
            render_faces = faces

        triangles = verts[render_faces]
        collection = Poly3DCollection(triangles, alpha=0.6, linewidth=0.1,
                                       edgecolor="gray")
        collection.set_facecolor(colors[i % len(colors)])
        ax.add_collection3d(collection)

        ax.set_xlim(mid[0] - max_range / 2, mid[0] + max_range / 2)
        ax.set_ylim(mid[1] - max_range / 2, mid[1] + max_range / 2)
        ax.set_zlim(mid[2] - max_range / 2, mid[2] + max_range / 2)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Rendered: {output_path}")
