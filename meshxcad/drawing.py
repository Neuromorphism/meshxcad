"""Orthographic wireframe renderer for engineering-drawing-style views.

Renders a triangle mesh as 2D line drawings with hidden-line removal,
producing the standard front/side/top views used in mechanical drawings.
"""

import numpy as np
from PIL import Image, ImageDraw


# ---------------------------------------------------------------------------
# Projection helpers
# ---------------------------------------------------------------------------

VIEW_DIRS = {
    "front": np.array([0.0, -1.0, 0.0]),   # look along -Y, project onto XZ
    "back":  np.array([0.0,  1.0, 0.0]),
    "right": np.array([1.0,  0.0, 0.0]),    # look along +X, project onto YZ
    "left":  np.array([-1.0, 0.0, 0.0]),
    "side":  np.array([1.0,  0.0, 0.0]),    # alias for right
    "top":   np.array([0.0,  0.0, 1.0]),    # look along +Z, project onto XY
    "bottom": np.array([0.0, 0.0, -1.0]),
}


def _view_matrix(view):
    """Return (view_dir, right, up) for a named view or (elev, azim) tuple."""
    if isinstance(view, str):
        view_dir = VIEW_DIRS[view].copy()
    else:
        elev, azim = np.radians(view[0]), np.radians(view[1])
        view_dir = np.array([
            np.cos(elev) * np.sin(azim),
            -np.cos(elev) * np.cos(azim),
            np.sin(elev),
        ])

    # Build right/up from view_dir
    world_up = np.array([0.0, 0.0, 1.0])
    if abs(np.dot(view_dir, world_up)) > 0.99:
        world_up = np.array([0.0, 1.0, 0.0])
    right = np.cross(world_up, view_dir)
    right /= np.linalg.norm(right)
    up = np.cross(view_dir, right)
    up /= np.linalg.norm(up)
    return view_dir, right, up


def _project_vertices(vertices, view):
    """Orthographic projection to 2D.

    Returns (xy, depth) where xy is (N, 2) and depth is (N,).
    """
    view_dir, right, up = _view_matrix(view)
    x = vertices @ right
    y = vertices @ up
    depth = vertices @ view_dir
    return np.column_stack([x, y]), depth


# ---------------------------------------------------------------------------
# Edge extraction
# ---------------------------------------------------------------------------

def _build_edge_face_map(faces):
    """Build mapping from edge (sorted pair) -> list of face indices."""
    edge_faces = {}
    for fi, (a, b, c) in enumerate(faces):
        for e in [(min(a, b), max(a, b)),
                  (min(b, c), max(b, c)),
                  (min(a, c), max(a, c))]:
            edge_faces.setdefault(e, []).append(fi)
    return edge_faces


def _face_normals(vertices, faces):
    """Compute per-face normals (not normalised — magnitude = 2*area)."""
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]
    normals = np.cross(v1 - v0, v2 - v0)
    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    norms[norms < 1e-12] = 1.0
    return normals / norms


def _classify_edges(normals, facing, edge_faces):
    """Classify edges into silhouette/feature/boundary candidates.

    Args:
        normals: per-face unit normals.
        facing: per-face dot product with view direction.
        edge_faces: dict edge -> face indices.

    Returns list of (vertex_a, vertex_b) candidate edges.
    """
    feature_angle = np.radians(25.0)
    # Small positive threshold: treat nearly edge-on faces as front-facing
    threshold = 0.01
    candidates = []
    for (a, b), flist in edge_faces.items():
        if len(flist) == 1:
            if facing[flist[0]] <= threshold:
                candidates.append((a, b))
        elif len(flist) == 2:
            f0, f1 = flist
            front0 = facing[f0] <= threshold
            front1 = facing[f1] <= threshold
            if front0 != front1:
                candidates.append((a, b))
            elif front0 and front1:
                cos_angle = np.dot(normals[f0], normals[f1])
                cos_angle = np.clip(cos_angle, -1.0, 1.0)
                if np.arccos(cos_angle) > feature_angle:
                    candidates.append((a, b))
    return candidates


def extract_visible_edges(vertices, faces, view):
    """Hidden-line removal: return only visible edges as 2D segments.

    Algorithm:
    1. Classify faces as front/back-facing.
    2. Classify edges: silhouette, feature (sharp), boundary.
    3. Test visible candidate edges against a z-buffer for occlusion.

    Returns list of ((x1, y1), (x2, y2)) in normalised coordinates [0, 1].
    """
    faces = np.asarray(faces)
    vertices = np.asarray(vertices, dtype=np.float64)
    view_dir, right, up = _view_matrix(view)

    normals = _face_normals(vertices, faces)
    facing = normals @ view_dir  # >0 = back-facing, <0 = front-facing

    edge_faces = _build_edge_face_map(faces)

    candidate_edges = _classify_edges(normals, facing, edge_faces)

    # If no edges found, normals may be inconsistent — try flipped convention
    if not candidate_edges:
        candidate_edges = _classify_edges(normals, -facing, edge_faces)

    if not candidate_edges:
        return []

    # Project to 2D
    xy, depth = _project_vertices(vertices, view)

    # Compute bounding box for normalisation
    if len(xy) == 0:
        return []
    xy_min = xy.min(axis=0)
    xy_max = xy.max(axis=0)
    span = xy_max - xy_min
    span[span < 1e-12] = 1.0
    margin = 0.05
    scale = (1.0 - 2 * margin) / span

    def norm_pt(idx):
        p = (xy[idx] - xy_min) * scale + margin
        return (float(p[0]), float(1.0 - p[1]))  # flip Y for image coords

    # Z-buffer occlusion test: build a simple depth buffer from front faces
    # and test edge midpoints against it
    front_faces = faces[facing <= 0]
    zbuf_size = 256
    zbuf = np.full((zbuf_size, zbuf_size), np.inf)

    if len(front_faces) > 0:
        for fi in range(len(front_faces)):
            tri_idx = front_faces[fi]
            pts_2d = np.array([norm_pt(tri_idx[k]) for k in range(3)])
            pts_depth = np.array([depth[tri_idx[k]] for k in range(3)])
            _raster_tri_zbuf(zbuf, pts_2d, pts_depth, zbuf_size)

    # Test each candidate edge — sample midpoint
    visible_edges = []
    depth_tolerance = 0.02 * (depth.max() - depth.min()) if depth.max() > depth.min() else 0.1
    for a, b in candidate_edges:
        pa = norm_pt(a)
        pb = norm_pt(b)
        # Sample a few points along the edge
        visible = True
        for t in (0.3, 0.5, 0.7):
            sx = pa[0] * (1 - t) + pb[0] * t
            sy = pa[1] * (1 - t) + pb[1] * t
            sd = depth[a] * (1 - t) + depth[b] * t
            ix = int(sx * zbuf_size)
            iy = int(sy * zbuf_size)
            ix = max(0, min(zbuf_size - 1, ix))
            iy = max(0, min(zbuf_size - 1, iy))
            if zbuf[iy, ix] < sd - depth_tolerance:
                visible = False
                break
        if visible:
            visible_edges.append((pa, pb))

    return visible_edges


def _raster_tri_zbuf(zbuf, pts_2d, pts_depth, size):
    """Rasterise a single triangle into the z-buffer (min depth wins)."""
    # Convert normalised coords to pixel coords
    px = (pts_2d[:, 0] * size).astype(int)
    py = (pts_2d[:, 1] * size).astype(int)

    # Bounding box
    x_min = max(0, min(px))
    x_max = min(size - 1, max(px))
    y_min = max(0, min(py))
    y_max = min(size - 1, max(py))

    if x_min > x_max or y_min > y_max:
        return

    # Barycentric rasterisation
    x0, y0 = float(px[0]), float(py[0])
    x1, y1 = float(px[1]), float(py[1])
    x2, y2 = float(px[2]), float(py[2])
    d0, d1, d2 = pts_depth

    denom = (y1 - y2) * (x0 - x2) + (x2 - x1) * (y0 - y2)
    if abs(denom) < 1e-10:
        return

    inv_denom = 1.0 / denom
    for yy in range(y_min, y_max + 1):
        for xx in range(x_min, x_max + 1):
            w0 = ((y1 - y2) * (xx - x2) + (x2 - x1) * (yy - y2)) * inv_denom
            w1 = ((y2 - y0) * (xx - x2) + (x0 - x2) * (yy - y2)) * inv_denom
            w2 = 1.0 - w0 - w1
            if w0 >= -0.01 and w1 >= -0.01 and w2 >= -0.01:
                z = w0 * d0 + w1 * d1 + w2 * d2
                if z < zbuf[yy, xx]:
                    zbuf[yy, xx] = z


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def render_orthographic(vertices, faces, view="front", image_size=512) -> np.ndarray:
    """Render mesh as a 2D orthographic view with hidden-line removal.

    Args:
        vertices: (N, 3) mesh vertices.
        faces: (M, 3) triangle indices.
        view: "front", "side", "top", or (elevation_deg, azimuth_deg).
        image_size: output dimension in pixels.

    Returns:
        (H, W) uint8 image — black lines on white background.
    """
    edges = extract_visible_edges(vertices, faces, view)

    img = Image.new("L", (image_size, image_size), 255)
    draw = ImageDraw.Draw(img)

    for (x1, y1), (x2, y2) in edges:
        px1 = int(x1 * image_size)
        py1 = int(y1 * image_size)
        px2 = int(x2 * image_size)
        py2 = int(y2 * image_size)
        draw.line([(px1, py1), (px2, py2)], fill=0, width=2)

    return np.array(img)


def render_drawing_sheet(vertices, faces, views=("front", "side", "top"),
                         image_size=512) -> np.ndarray:
    """Multi-view engineering drawing on one sheet.

    Arranges views in standard third-angle projection layout:
    - Row 0: [front, side]
    - Row 1: [top, (empty or isometric)]

    Returns: (H, W, 3) RGB uint8 image.
    """
    n_views = len(views)
    if n_views <= 2:
        cols, rows = n_views, 1
    elif n_views <= 4:
        cols, rows = 2, 2
    else:
        cols = 3
        rows = (n_views + 2) // 3

    cell = image_size
    sheet_w = cols * cell
    sheet_h = rows * cell

    sheet = Image.new("RGB", (sheet_w, sheet_h), (255, 255, 255))
    draw = ImageDraw.Draw(sheet)

    for i, view in enumerate(views):
        col = i % cols
        row = i // cols
        x_off = col * cell
        y_off = row * cell

        single = render_orthographic(vertices, faces, view, cell)
        view_img = Image.fromarray(single).convert("RGB")
        sheet.paste(view_img, (x_off, y_off))

        # Draw border
        draw.rectangle([x_off, y_off, x_off + cell - 1, y_off + cell - 1],
                       outline=(180, 180, 180), width=1)

        # Label
        label = view if isinstance(view, str) else f"({view[0]}°, {view[1]}°)"
        draw.text((x_off + 5, y_off + 5), label.upper(), fill=(120, 120, 120))

    return np.array(sheet)
