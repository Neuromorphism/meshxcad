"""Orthographic wireframe renderer for engineering-drawing-style views.

Renders a triangle mesh as 2D line drawings with hidden-line removal,
producing the standard front/side/top views used in mechanical drawings.

Dimension annotations follow ASME Y14.5-2018:
  - Unidirectional (all text reads left-to-right)
  - Filled arrowheads with 3:1 length-to-width ratio
  - Extension lines with visible gap from object outline
  - Extension lines overshoot dimension lines
  - Diameter symbol (⌀) prefix for diameters
  - Radius symbol (R) prefix for radii
  - Text centered in a break in the dimension line
  - Third-angle projection layout
"""

import math

import numpy as np
from PIL import Image, ImageDraw, ImageFont


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
# Dimension annotation helpers
# ---------------------------------------------------------------------------

def _compute_mesh_dimensions(vertices, faces):
    """Analyze the mesh bounding box and detect cylindrical shapes.

    Returns a dict with:
        width, height, depth: bounding box extents in mesh units (mm).
        is_cylindrical: True if object appears cylindrical (circular cross-section
                        when viewed from top).
        diameter: if cylindrical, the diameter of the circular cross-section.
    """
    vertices = np.asarray(vertices, dtype=np.float64)
    bb_min = vertices.min(axis=0)
    bb_max = vertices.max(axis=0)
    extents = bb_max - bb_min  # (X-width, Y-depth, Z-height)

    dims = {
        "width": float(extents[0]),
        "depth": float(extents[1]),
        "height": float(extents[2]),
        "is_cylindrical": False,
        "diameter": None,
    }

    # Detect cylindrical shape: check if XY cross-section is roughly circular.
    # Compare bounding-box aspect in XY and check how close vertices lie
    # to a circle in the XY plane.
    xy_extent_x = extents[0]
    xy_extent_y = extents[1]
    if xy_extent_x > 1e-6 and xy_extent_y > 1e-6:
        aspect = min(xy_extent_x, xy_extent_y) / max(xy_extent_x, xy_extent_y)
        if aspect > 0.85:
            # Check how closely vertices fit a circle in XY
            cx = (bb_min[0] + bb_max[0]) / 2
            cy = (bb_min[1] + bb_max[1]) / 2
            radii = np.sqrt((vertices[:, 0] - cx) ** 2 + (vertices[:, 1] - cy) ** 2)
            r_max = radii.max()
            if r_max > 1e-6:
                # Check that a majority of vertices sit close to the max
                # radius (characteristic of cylinders / revolved shapes).
                # Cubes and other prismatic shapes have vertices spread across
                # a wider range of radii.
                near_rim = np.sum(radii > 0.90 * r_max)
                ratio = near_rim / len(radii)
                if ratio > 0.5:
                    dims["is_cylindrical"] = True
                    dims["diameter"] = float(2.0 * r_max)

    return dims


def _get_view_dimensions(dims, view):
    """Determine which dimension annotations to draw for a given view.

    Per ASME Y14.5, dimensions placed outside the view with extension lines.

    Returns a list of dicts, each with:
        orientation: 'horizontal' or 'vertical'
        value: the dimension value in mm
        label: 'width', 'height', 'depth', 'diameter', or 'radius'
    """
    view_name = view if isinstance(view, str) else None
    annotations = []

    if view_name == "front":
        if dims["is_cylindrical"]:
            annotations.append({"orientation": "horizontal", "value": dims["diameter"],
                                "label": "diameter"})
        else:
            annotations.append({"orientation": "horizontal", "value": dims["width"],
                                "label": "width"})
        annotations.append({"orientation": "vertical", "value": dims["height"],
                            "label": "height"})
    elif view_name in ("side", "right", "left"):
        if dims["is_cylindrical"]:
            annotations.append({"orientation": "horizontal", "value": dims["diameter"],
                                "label": "diameter"})
        else:
            annotations.append({"orientation": "horizontal", "value": dims["depth"],
                                "label": "depth"})
        annotations.append({"orientation": "vertical", "value": dims["height"],
                            "label": "height"})
    elif view_name == "top":
        annotations.append({"orientation": "horizontal", "value": dims["width"],
                            "label": "width"})
        if dims["is_cylindrical"]:
            annotations.append({"orientation": "vertical", "value": dims["diameter"],
                                "label": "diameter"})
        else:
            annotations.append({"orientation": "vertical", "value": dims["depth"],
                                "label": "depth"})
    else:
        annotations.append({"orientation": "horizontal", "value": dims["width"],
                            "label": "width"})
        annotations.append({"orientation": "vertical", "value": dims["height"],
                            "label": "height"})

    return annotations


# ---------------------------------------------------------------------------
# ASME Y14.5-2018 compliant dimension annotation
# ---------------------------------------------------------------------------

# ASME Y14.5 arrowhead: filled, 3:1 length-to-width ratio
_ARROW_LENGTH = 8
_ARROW_HALF_WIDTH = _ARROW_LENGTH / 3.0

# Extension line visible gap from object outline (ASME Y14.5 §1.7.4)
_EXT_GAP = 3
# Extension line overshoot past dimension line
_EXT_OVERSHOOT = 5
# Offset of first dimension line from object outline
_DIM_OFFSET = 30
# Thin line width for extension/dimension lines (ASME visible thin)
_THIN_LINE = 1


def _asme_format_value(value, label):
    """Format a dimension value per ASME Y14.5.

    - Diameter: ⌀XX.X  (using the Unicode diameter symbol U+2300)
    - Radius: RXX.X
    - Linear: XX.X
    - Omit trailing zeros after one decimal (ASME convention: show to
      the precision of measurement, minimum one decimal).
    """
    # Format to 1 decimal place; drop unnecessary trailing zero only
    # if the value is an integer
    if value == int(value) and value >= 1:
        text = f"{value:.0f}"
    else:
        text = f"{value:.1f}"

    if label == "diameter":
        return f"\u2300{text}"
    elif label == "radius":
        return f"R{text}"
    return text


def _draw_arrowhead(draw, tip, direction, fill=0):
    """Draw a filled arrowhead per ASME Y14.5 (3:1 aspect ratio).

    Args:
        tip: (x, y) pixel position of the arrow tip.
        direction: unit (dx, dy) pointing toward the tip.
    """
    dx, dy = direction
    px, py = -dy, dx  # perpendicular
    bx = tip[0] - dx * _ARROW_LENGTH
    by = tip[1] - dy * _ARROW_LENGTH
    points = [
        (tip[0], tip[1]),
        (bx + px * _ARROW_HALF_WIDTH, by + py * _ARROW_HALF_WIDTH),
        (bx - px * _ARROW_HALF_WIDTH, by - py * _ARROW_HALF_WIDTH),
    ]
    draw.polygon(points, fill=fill)


def _load_font(size=11):
    """Try to load a clean sans-serif font; fall back to default."""
    for path in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        "/usr/share/fonts/TTF/DejaVuSans.ttf",
    ):
        try:
            return ImageFont.truetype(path, size)
        except (IOError, OSError):
            continue
    return ImageFont.load_default()


def _add_dimension_annotations(draw, edges_2d, dimensions, view,
                                image_size, vertices, faces):
    """Draw ASME Y14.5 dimension annotations.

    Per ASME Y14.5-2018:
    - Extension lines start with a visible gap from the object outline.
    - Extension lines extend slightly past the dimension line.
    - Filled arrowheads with 3:1 length:width ratio terminate dimension lines.
    - Dimension text is unidirectional (reads left-to-right), centered in a
      break in the dimension line.
    - Diameter dimensions use the ⌀ symbol prefix.
    - Radius dimensions use the R prefix.
    """
    if not dimensions or not edges_2d:
        return

    # Object bounding box in pixel coords
    all_pts = []
    for (x1, y1), (x2, y2) in edges_2d:
        all_pts.append((x1 * image_size, y1 * image_size))
        all_pts.append((x2 * image_size, y2 * image_size))
    pts = np.array(all_pts)
    obj_left = pts[:, 0].min()
    obj_right = pts[:, 0].max()
    obj_top = pts[:, 1].min()
    obj_bottom = pts[:, 1].max()

    ink = 0  # black
    font = _load_font(11)

    for ann in dimensions:
        value = ann["value"]
        if value is None or value < 1e-6:
            continue
        label = ann["label"]
        text = _asme_format_value(value, label)
        orientation = ann["orientation"]

        # Measure text
        bbox = draw.textbbox((0, 0), text, font=font)
        tw = bbox[2] - bbox[0]
        th = bbox[3] - bbox[1]

        if orientation == "horizontal":
            _draw_horizontal_dim(draw, font, text, tw, th,
                                 obj_left, obj_right, obj_bottom,
                                 image_size, ink)
        elif orientation == "vertical":
            _draw_vertical_dim(draw, font, text, tw, th,
                               obj_top, obj_bottom, obj_right,
                               image_size, ink)


def _draw_horizontal_dim(draw, font, text, tw, th,
                          x_left, x_right, obj_bottom, image_size, ink):
    """Draw a horizontal dimension line below the object (ASME Y14.5)."""
    dim_y = min(obj_bottom + _DIM_OFFSET, image_size - 14)
    # Clamp extension line start so the gap is visible
    ext_start_y = obj_bottom + _EXT_GAP

    # Extension lines (vertical) — gap from object, overshoot past dim line
    draw.line([(x_left, ext_start_y), (x_left, dim_y + _EXT_OVERSHOOT)],
              fill=ink, width=_THIN_LINE)
    draw.line([(x_right, ext_start_y), (x_right, dim_y + _EXT_OVERSHOOT)],
              fill=ink, width=_THIN_LINE)

    # Dimension line with text break in the centre
    text_cx = (x_left + x_right) / 2
    text_pad = 3  # padding around text in the break
    break_left = text_cx - tw / 2 - text_pad
    break_right = text_cx + tw / 2 + text_pad

    # Draw dimension line in two segments around the text break
    if break_left > x_left + _ARROW_LENGTH:
        draw.line([(x_left, dim_y), (break_left, dim_y)],
                  fill=ink, width=_THIN_LINE)
    if break_right < x_right - _ARROW_LENGTH:
        draw.line([(break_right, dim_y), (x_right, dim_y)],
                  fill=ink, width=_THIN_LINE)

    # Arrowheads (pointing inward)
    _draw_arrowhead(draw, (x_left, dim_y), (1, 0), fill=ink)
    _draw_arrowhead(draw, (x_right, dim_y), (-1, 0), fill=ink)

    # Text — unidirectional, centred in the break (ASME Y14.5 §1.7.6)
    tx = text_cx - tw / 2
    ty = dim_y - th / 2 - 1
    draw.rectangle([tx - 2, ty - 1, tx + tw + 2, ty + th + 1], fill=255)
    draw.text((tx, ty), text, fill=ink, font=font)


def _draw_vertical_dim(draw, font, text, tw, th,
                        y_top, y_bottom, obj_right, image_size, ink):
    """Draw a vertical dimension line to the right of the object (ASME Y14.5).

    Per ASME Y14.5 unidirectional dimensioning, vertical dimension text
    reads left-to-right (horizontal), centred in a break in the dim line.
    """
    dim_x = min(obj_right + _DIM_OFFSET, image_size - 14)
    ext_start_x = obj_right + _EXT_GAP

    # Extension lines (horizontal)
    draw.line([(ext_start_x, y_top), (dim_x + _EXT_OVERSHOOT, y_top)],
              fill=ink, width=_THIN_LINE)
    draw.line([(ext_start_x, y_bottom), (dim_x + _EXT_OVERSHOOT, y_bottom)],
              fill=ink, width=_THIN_LINE)

    # Dimension line with text break
    text_cy = (y_top + y_bottom) / 2
    text_pad = 3
    break_top = text_cy - th / 2 - text_pad
    break_bottom = text_cy + th / 2 + text_pad

    if break_top > y_top + _ARROW_LENGTH:
        draw.line([(dim_x, y_top), (dim_x, break_top)],
                  fill=ink, width=_THIN_LINE)
    if break_bottom < y_bottom - _ARROW_LENGTH:
        draw.line([(dim_x, break_bottom), (dim_x, y_bottom)],
                  fill=ink, width=_THIN_LINE)

    # Arrowheads (pointing inward)
    _draw_arrowhead(draw, (dim_x, y_top), (0, 1), fill=ink)
    _draw_arrowhead(draw, (dim_x, y_bottom), (0, -1), fill=ink)

    # Text — unidirectional (reads L-to-R), centred in break
    tx = dim_x - tw / 2
    ty = text_cy - th / 2 - 1
    draw.rectangle([tx - 2, ty - 1, tx + tw + 2, ty + th + 1], fill=255)
    draw.text((tx, ty), text, fill=ink, font=font)


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def render_orthographic(vertices, faces, view="front", image_size=512,
                        annotate=False) -> np.ndarray:
    """Render mesh as a 2D orthographic view with hidden-line removal.

    Args:
        vertices: (N, 3) mesh vertices.
        faces: (M, 3) triangle indices.
        view: "front", "side", "top", or (elevation_deg, azimuth_deg).
        image_size: output dimension in pixels.
        annotate: if True, add dimension annotations (lines, arrows, text).

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

    if annotate:
        dims = _compute_mesh_dimensions(vertices, faces)
        annotations = _get_view_dimensions(dims, view)
        _add_dimension_annotations(draw, edges, annotations, view,
                                    image_size, vertices, faces)

    return np.array(img)


def render_drawing_sheet(vertices, faces, views=("front", "side", "top"),
                         image_size=512, annotate=False) -> np.ndarray:
    """Multi-view engineering drawing on one sheet.

    Arranges views in standard third-angle projection layout:
    - Row 0: [front, side]
    - Row 1: [top, (empty or isometric)]

    Args:
        vertices: (N, 3) mesh vertices.
        faces: (M, 3) triangle indices.
        views: sequence of view names or (elev, azim) tuples.
        image_size: cell size in pixels.
        annotate: if True, add dimension annotations to each view.

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

        single = render_orthographic(vertices, faces, view, cell,
                                         annotate=annotate)
        view_img = Image.fromarray(single).convert("RGB")
        sheet.paste(view_img, (x_off, y_off))

        # Draw border
        draw.rectangle([x_off, y_off, x_off + cell - 1, y_off + cell - 1],
                       outline=(180, 180, 180), width=1)

        # Label
        label = view if isinstance(view, str) else f"({view[0]}°, {view[1]}°)"
        draw.text((x_off + 5, y_off + 5), label.upper(), fill=(120, 120, 120))

    return np.array(sheet)
