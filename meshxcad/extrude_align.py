"""Extrude-to-mesh alignment: 12 features for moving CAD extrude geometry
toward mesh geometry by analyzing cross-sections and adapting profiles.

Given a CAD model built from extrusions and a target mesh, these tools let you:
1.  Extract 2D cross-sections from a mesh at arbitrary Z heights.
2.  Compare an extrude's polygon profile against the mesh cross-section.
3.  Fit primitive shapes (circle, ellipse, rectangle, regular polygon)
    to a mesh cross-section.
4.  Fit a smooth spline contour to an irregular cross-section.
5.  Compute a per-vertex offset (inward/outward) to resize a profile.
6.  Blend/morph between two profiles at different heights.
7.  Taper an extrude so the top and bottom profiles differ linearly.
8.  Twist an extrude (progressive rotation along the extrusion axis).
9.  Boolean-union two 2D profiles into a single merged outline.
10. Boolean-difference one 2D profile from another.
11. Adaptively rebuild an extrude by sampling the mesh at multiple
    Z-slices and extruding slice-by-slice.
12. Score a full extrude against its corresponding mesh region and
    return an ordered list of suggested manipulations.
"""

import math
import numpy as np
from scipy.spatial import ConvexHull, KDTree
from scipy.interpolate import CubicSpline


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _polygon_area_signed(poly):
    """Signed area of a 2D polygon (positive if CCW)."""
    poly = np.asarray(poly)
    x, y = poly[:, 0], poly[:, 1]
    return 0.5 * float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))


def _polygon_centroid(poly):
    """Centroid of a 2D polygon."""
    poly = np.asarray(poly)
    n = len(poly)
    if n == 0:
        return np.zeros(2)
    cx = float(np.mean(poly[:, 0]))
    cy = float(np.mean(poly[:, 1]))
    return np.array([cx, cy])


def _order_polygon_ccw(points):
    """Order 2D points into a CCW polygon via angular sort around centroid."""
    pts = np.asarray(points, dtype=np.float64)
    if len(pts) < 3:
        return pts
    c = _polygon_centroid(pts)
    angles = np.arctan2(pts[:, 1] - c[1], pts[:, 0] - c[0])
    order = np.argsort(angles)
    return pts[order]


def _resample_polygon(poly, n_out):
    """Resample a closed polygon to *n_out* equi-spaced points."""
    poly = np.asarray(poly, dtype=np.float64)
    # Close the loop
    closed = np.vstack([poly, poly[0:1]])
    diffs = np.diff(closed, axis=0)
    seg_lens = np.linalg.norm(diffs, axis=1)
    cum = np.concatenate([[0], np.cumsum(seg_lens)])
    total = cum[-1]
    if total < 1e-12:
        return poly[:n_out] if len(poly) >= n_out else poly

    targets = np.linspace(0, total, n_out, endpoint=False)
    out = np.zeros((n_out, 2))
    for i, t in enumerate(targets):
        idx = np.searchsorted(cum, t, side="right") - 1
        idx = max(0, min(idx, len(poly) - 1))
        nxt = (idx + 1) % len(poly)
        seg = seg_lens[idx]
        frac = (t - cum[idx]) / seg if seg > 1e-12 else 0.0
        out[i] = poly[idx] * (1 - frac) + poly[nxt] * frac
    return out


# =========================================================================
# 1. extract_cross_section
# =========================================================================

def extract_cross_section(vertices, faces, z_height, tolerance=None):
    """Slice a triangle mesh at *z_height* and return the 2D cross-section.

    For each triangle that straddles the plane z = z_height, we compute the
    two intersection points on the triangle edges, yielding line segments.
    The segments are then chained into an ordered polygon.

    Args:
        vertices: (N, 3) mesh vertices
        faces:    (M, 3) triangle indices
        z_height: Z value of the slicing plane
        tolerance: optional thickness band (default: 1e-6)

    Returns:
        polygon: (K, 2) array of ordered (x, y) points, or empty array
    """
    if tolerance is None:
        tolerance = 1e-6

    verts = np.asarray(vertices, dtype=np.float64)
    tris = np.asarray(faces)

    segments = []
    for tri in tris:
        v = verts[tri]  # (3, 3)
        below = v[:, 2] < z_height - tolerance
        above = v[:, 2] > z_height + tolerance

        # We need exactly 2 intersections (one vertex on each side)
        n_below = int(np.sum(below))
        n_above = int(np.sum(above))
        # If all on one side, no intersection
        if n_below == 3 or n_above == 3:
            continue
        # If all coincident with the plane, skip (degenerate)
        if n_below == 0 and n_above == 0:
            continue

        pts = []
        for i in range(3):
            j = (i + 1) % 3
            z0, z1 = v[i, 2], v[j, 2]
            if (z0 - z_height) * (z1 - z_height) < 0:
                # Edge crosses the plane
                t = (z_height - z0) / (z1 - z0)
                p = v[i] + t * (v[j] - v[i])
                pts.append(p[:2])
            elif abs(z0 - z_height) <= tolerance:
                pts.append(v[i, :2].copy())
            elif abs(z1 - z_height) <= tolerance:
                pts.append(v[j, :2].copy())

        # Deduplicate very close points
        if len(pts) >= 2:
            unique = [pts[0]]
            for p in pts[1:]:
                if np.linalg.norm(p - unique[-1]) > tolerance:
                    unique.append(p)
            if len(unique) >= 2:
                segments.append((unique[0], unique[1]))

    if not segments:
        return np.zeros((0, 2))

    # Chain segments into an ordered polygon
    polygon = _chain_segments(segments, tolerance * 10)
    if len(polygon) < 3:
        # Fallback: just collect all unique points and order them
        all_pts = []
        for a, b in segments:
            all_pts.append(a)
            all_pts.append(b)
        all_pts = np.array(all_pts)
        if len(all_pts) < 3:
            return all_pts
        return _order_polygon_ccw(all_pts)

    return np.array(polygon)


def _chain_segments(segments, tol):
    """Chain line segments into a polygon by matching endpoints."""
    if not segments:
        return []
    remaining = list(range(len(segments)))
    chain = [segments[0][0], segments[0][1]]
    remaining.remove(0)

    max_iter = len(segments) * 2
    for _ in range(max_iter):
        if not remaining:
            break
        tail = chain[-1]
        found = False
        for idx in remaining:
            a, b = segments[idx]
            if np.linalg.norm(a - tail) < tol:
                chain.append(b)
                remaining.remove(idx)
                found = True
                break
            elif np.linalg.norm(b - tail) < tol:
                chain.append(a)
                remaining.remove(idx)
                found = True
                break
        if not found:
            break

    return chain


# =========================================================================
# 2. compare_cross_sections
# =========================================================================

def compare_cross_sections(profile_a, profile_b, n_sample=64):
    """Compare two 2D profiles and return similarity metrics.

    Both profiles are resampled to *n_sample* equi-arc-length points, then
    compared via Hausdorff distance, mean point distance, and area ratio.

    Args:
        profile_a: (K, 2) first profile (e.g. extrude polygon)
        profile_b: (L, 2) second profile (e.g. mesh cross-section)
        n_sample:  resample count

    Returns:
        dict with keys:
            hausdorff   — max min-distance (worst-case mismatch)
            mean_dist   — average min-distance
            area_ratio  — area_a / area_b (1.0 = same area)
            iou         — approximate intersection-over-union (convex)
    """
    a = _resample_polygon(np.asarray(profile_a), n_sample)
    b = _resample_polygon(np.asarray(profile_b), n_sample)

    tree_b = KDTree(b)
    dists_a, _ = tree_b.query(a)
    tree_a = KDTree(a)
    dists_b, _ = tree_a.query(b)

    hausdorff = float(max(np.max(dists_a), np.max(dists_b)))
    mean_dist = float((np.mean(dists_a) + np.mean(dists_b)) / 2)

    area_a = abs(_polygon_area_signed(a))
    area_b = abs(_polygon_area_signed(b))
    area_ratio = area_a / area_b if area_b > 1e-12 else float("inf")

    # Convex IoU approximation
    try:
        combined = np.vstack([a, b])
        hull_union = ConvexHull(combined)
        union_area = hull_union.volume  # In 2D, volume = area
        hull_a = ConvexHull(a)
        hull_b = ConvexHull(b)
        inter_area = hull_a.volume + hull_b.volume - union_area
        iou = max(inter_area, 0) / union_area if union_area > 1e-12 else 0.0
    except Exception:
        iou = 0.0

    return {
        "hausdorff": hausdorff,
        "mean_dist": mean_dist,
        "area_ratio": area_ratio,
        "iou": iou,
    }


# =========================================================================
# 3. fit_primitive_to_cross_section
# =========================================================================

def fit_primitive_to_cross_section(points):
    """Fit several primitive shapes to a 2D point cloud and return the best.

    Candidates: circle, axis-aligned rectangle, regular polygons (3–8 sides).

    Args:
        points: (K, 2) array

    Returns:
        dict with keys:
            best        — name of best-fit primitive
            params      — dict of primitive parameters
            polygon     — (M, 2) fitted polygon vertices
            residual    — mean radial residual of best fit
            all_fits    — list of (name, residual) for every candidate
    """
    pts = np.asarray(points, dtype=np.float64)
    cx, cy = float(np.mean(pts[:, 0])), float(np.mean(pts[:, 1]))
    centered = pts - np.array([cx, cy])
    radii = np.linalg.norm(centered, axis=1)
    mean_r = float(np.mean(radii))

    fits = []

    # --- circle ---
    circ_residual = float(np.mean(np.abs(radii - mean_r)))
    n_circ = 48
    circ_poly = np.array([
        [cx + mean_r * math.cos(2 * math.pi * i / n_circ),
         cy + mean_r * math.sin(2 * math.pi * i / n_circ)]
        for i in range(n_circ)
    ])
    fits.append(("circle", circ_residual, {"center": (cx, cy), "radius": mean_r}, circ_poly))

    # --- axis-aligned bounding rectangle ---
    xmin, xmax = float(pts[:, 0].min()), float(pts[:, 0].max())
    ymin, ymax = float(pts[:, 1].min()), float(pts[:, 1].max())
    rect_poly = np.array([[xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax]])
    # Residual: mean distance from each point to nearest edge
    rect_dists = np.minimum(
        np.minimum(pts[:, 0] - xmin, xmax - pts[:, 0]),
        np.minimum(pts[:, 1] - ymin, ymax - pts[:, 1]),
    )
    rect_residual = float(np.mean(np.abs(rect_dists)))
    fits.append(("rectangle", rect_residual,
                 {"center": (cx, cy), "width": xmax - xmin, "height": ymax - ymin},
                 rect_poly))

    # --- regular polygons 3–8 ---
    for n_sides in range(3, 9):
        best_res = float("inf")
        best_angle = 0.0
        for start_deg in range(0, 360 // n_sides, 5):
            start_a = math.radians(start_deg)
            poly = np.array([
                [cx + mean_r * math.cos(start_a + 2 * math.pi * i / n_sides),
                 cy + mean_r * math.sin(start_a + 2 * math.pi * i / n_sides)]
                for i in range(n_sides)
            ])
            tree = KDTree(poly)
            dists, _ = tree.query(pts)
            res = float(np.mean(dists))
            if res < best_res:
                best_res = res
                best_angle = start_a
                best_poly = poly
        fits.append((f"polygon_{n_sides}", best_res,
                     {"center": (cx, cy), "radius": mean_r,
                      "n_sides": n_sides, "start_angle": best_angle},
                     best_poly))

    fits.sort(key=lambda x: x[1])
    best_name, best_res, best_params, best_poly = fits[0]
    return {
        "best": best_name,
        "params": best_params,
        "polygon": best_poly,
        "residual": best_res,
        "all_fits": [(f[0], f[1]) for f in fits],
    }


# =========================================================================
# 4. fit_ellipse_to_cross_section
# =========================================================================

def fit_ellipse_to_cross_section(points):
    """Fit an axis-aligned ellipse to 2D points via least-squares.

    Minimizes algebraic distance for:  (x-cx)^2/a^2 + (y-cy)^2/b^2 = 1

    Args:
        points: (K, 2)

    Returns:
        dict: center (cx, cy), semi_axes (a, b), residual, polygon (N, 2)
    """
    pts = np.asarray(points, dtype=np.float64)
    cx, cy = float(np.mean(pts[:, 0])), float(np.mean(pts[:, 1]))
    dx = pts[:, 0] - cx
    dy = pts[:, 1] - cy

    # Fit: x'^2/a^2 + y'^2/b^2 = 1  =>  solve for 1/a^2, 1/b^2
    A = np.column_stack([dx ** 2, dy ** 2])
    b_vec = np.ones(len(pts))
    result, _, _, _ = np.linalg.lstsq(A, b_vec, rcond=None)
    inv_a2, inv_b2 = result
    a = 1.0 / math.sqrt(max(abs(inv_a2), 1e-12))
    b = 1.0 / math.sqrt(max(abs(inv_b2), 1e-12))

    # Residual: distance from each point to the nearest point on the ellipse
    angles = np.arctan2(dy, dx)
    ellipse_r = a * b / np.sqrt((b * np.cos(angles)) ** 2 + (a * np.sin(angles)) ** 2)
    actual_r = np.sqrt(dx ** 2 + dy ** 2)
    residual = float(np.mean(np.abs(actual_r - ellipse_r)))

    # Generate polygon
    n_pts = 64
    t = np.linspace(0, 2 * math.pi, n_pts, endpoint=False)
    polygon = np.column_stack([cx + a * np.cos(t), cy + b * np.sin(t)])

    return {
        "center": (cx, cy),
        "semi_axes": (float(a), float(b)),
        "residual": residual,
        "polygon": polygon,
    }


# =========================================================================
# 5. offset_profile
# =========================================================================

def offset_profile(polygon, distance):
    """Offset a 2D polygon inward (negative) or outward (positive).

    Uses vertex-normal offsetting for simplicity.

    Args:
        polygon:  (K, 2) ordered polygon
        distance: offset distance (positive = outward, negative = inward)

    Returns:
        offset_polygon: (K, 2)
    """
    poly = np.asarray(polygon, dtype=np.float64)
    n = len(poly)
    if n < 3:
        return poly.copy()

    # Determine polygon winding (sign of signed area)
    # For CCW polygon: outward normal is rotate-right (CW 90°): (dy, -dx)
    # For CW polygon: outward normal is rotate-left (CCW 90°): (-dy, dx)
    area_sign = 1.0 if _polygon_area_signed(poly) > 0 else -1.0

    normals = np.zeros_like(poly)
    for i in range(n):
        prev_i = (i - 1) % n
        next_i = (i + 1) % n
        edge_prev = poly[i] - poly[prev_i]
        edge_next = poly[next_i] - poly[i]
        # Outward normals: for CCW, rotate 90° CW: (dy, -dx)
        n_prev = np.array([edge_prev[1], -edge_prev[0]]) * area_sign
        n_next = np.array([edge_next[1], -edge_next[0]]) * area_sign
        n_prev_len = np.linalg.norm(n_prev)
        n_next_len = np.linalg.norm(n_next)
        if n_prev_len > 1e-12:
            n_prev /= n_prev_len
        if n_next_len > 1e-12:
            n_next /= n_next_len
        normals[i] = (n_prev + n_next)
        nlen = np.linalg.norm(normals[i])
        if nlen > 1e-12:
            normals[i] /= nlen

    return poly + normals * distance


# =========================================================================
# 6. blend_profiles
# =========================================================================

def blend_profiles(profile_a, profile_b, t, n_sample=64):
    """Linearly blend two 2D profiles.

    Both profiles are resampled to the same number of points, then linearly
    interpolated:  result = (1-t)*A + t*B.

    Args:
        profile_a: (K, 2) first profile (t=0)
        profile_b: (L, 2) second profile (t=1)
        t:         blend factor in [0, 1]
        n_sample:  resample count

    Returns:
        blended: (n_sample, 2)
    """
    a = _resample_polygon(np.asarray(profile_a), n_sample)
    b = _resample_polygon(np.asarray(profile_b), n_sample)
    return (1 - t) * a + t * b


# =========================================================================
# 7. taper_extrude
# =========================================================================

def taper_extrude(polygon_xy, height, scale_top=1.0, n_height=10):
    """Extrude a 2D polygon with linear taper from bottom to top.

    At the bottom (z=0) the profile is used as-is.  At the top (z=height)
    it is scaled by *scale_top* relative to its centroid.

    Args:
        polygon_xy: list of (x, y) tuples
        height:     extrusion height
        scale_top:  scale factor at the top (1.0 = no taper)
        n_height:   number of height divisions

    Returns:
        vertices: (N, 3), faces: (M, 3)
    """
    poly = np.asarray(polygon_xy, dtype=np.float64)
    centroid = _polygon_centroid(poly)
    n_pts = len(poly)

    vertices = []
    for i in range(n_height + 1):
        frac = i / n_height
        z = height * frac
        scale = 1.0 + (scale_top - 1.0) * frac
        for pt in poly:
            scaled = centroid + (pt - centroid) * scale
            vertices.append([scaled[0], scaled[1], z])

    faces = []
    for i in range(n_height):
        for j in range(n_pts):
            j_next = (j + 1) % n_pts
            p00 = i * n_pts + j
            p01 = i * n_pts + j_next
            p10 = (i + 1) * n_pts + j
            p11 = (i + 1) * n_pts + j_next
            faces.append([p00, p10, p01])
            faces.append([p01, p10, p11])

    # Bottom cap
    bot_cx, bot_cy = centroid
    bot_center = len(vertices)
    vertices.append([bot_cx, bot_cy, 0])
    for j in range(n_pts):
        j_next = (j + 1) % n_pts
        faces.append([bot_center, j_next, j])

    # Top cap
    top_center = len(vertices)
    top_cx = centroid[0]
    top_cy = centroid[1]
    vertices.append([top_cx, top_cy, height])
    top_start = n_height * n_pts
    for j in range(n_pts):
        j_next = (j + 1) % n_pts
        faces.append([top_center, top_start + j, top_start + j_next])

    return np.array(vertices, dtype=np.float64), np.array(faces)


# =========================================================================
# 8. twist_extrude
# =========================================================================

def twist_extrude(polygon_xy, height, total_twist_deg=90.0, n_height=20):
    """Extrude with progressive rotation (twist) along Z.

    Args:
        polygon_xy:      list of (x, y)
        height:          extrusion height
        total_twist_deg: total rotation in degrees from bottom to top
        n_height:        height divisions

    Returns:
        vertices: (N, 3), faces: (M, 3)
    """
    poly = np.asarray(polygon_xy, dtype=np.float64)
    centroid = _polygon_centroid(poly)
    n_pts = len(poly)

    vertices = []
    for i in range(n_height + 1):
        frac = i / n_height
        z = height * frac
        angle = math.radians(total_twist_deg * frac)
        cos_a, sin_a = math.cos(angle), math.sin(angle)
        for pt in poly:
            dx, dy = pt[0] - centroid[0], pt[1] - centroid[1]
            rx = centroid[0] + dx * cos_a - dy * sin_a
            ry = centroid[1] + dx * sin_a + dy * cos_a
            vertices.append([rx, ry, z])

    faces = []
    for i in range(n_height):
        for j in range(n_pts):
            j_next = (j + 1) % n_pts
            p00 = i * n_pts + j
            p01 = i * n_pts + j_next
            p10 = (i + 1) * n_pts + j
            p11 = (i + 1) * n_pts + j_next
            faces.append([p00, p10, p01])
            faces.append([p01, p10, p11])

    # Bottom cap
    bot_center = len(vertices)
    vertices.append([centroid[0], centroid[1], 0])
    for j in range(n_pts):
        j_next = (j + 1) % n_pts
        faces.append([bot_center, j_next, j])

    # Top cap
    top_center = len(vertices)
    vertices.append([centroid[0], centroid[1], height])
    top_start = n_height * n_pts
    for j in range(n_pts):
        j_next = (j + 1) % n_pts
        faces.append([top_center, top_start + j, top_start + j_next])

    return np.array(vertices, dtype=np.float64), np.array(faces)


# =========================================================================
# 9. boolean_union_profiles
# =========================================================================

def boolean_union_profiles(profile_a, profile_b):
    """Approximate boolean union of two convex 2D profiles.

    Returns the convex hull of the combined point sets.  For non-convex
    inputs this is an outer approximation.

    Args:
        profile_a: (K, 2)
        profile_b: (L, 2)

    Returns:
        union_polygon: (M, 2) ordered CCW
    """
    combined = np.vstack([np.asarray(profile_a), np.asarray(profile_b)])
    if len(combined) < 3:
        return combined
    hull = ConvexHull(combined)
    return combined[hull.vertices]


# =========================================================================
# 10. boolean_difference_profiles
# =========================================================================

def boolean_difference_profiles(outer_profile, inner_profile, n_sample=128):
    """Approximate boolean difference: outer minus inner.

    Points of the outer profile that fall inside the inner profile are
    pushed outward to the inner boundary.  The result is the outer polygon
    with a concavity carved where the inner profile overlaps.

    For non-overlapping profiles the outer is returned unchanged.

    Args:
        outer_profile: (K, 2)
        inner_profile: (L, 2)
        n_sample:      resample count for the outer profile

    Returns:
        result_polygon: (M, 2)
    """
    outer = _resample_polygon(np.asarray(outer_profile), n_sample)
    inner = np.asarray(inner_profile, dtype=np.float64)

    if len(inner) < 3:
        return outer

    # Point-in-polygon test using winding number (simplified ray-casting)
    inside_mask = _points_in_polygon(outer, inner)
    if not np.any(inside_mask):
        return outer

    # For points inside the inner polygon, project them onto the inner
    # boundary (nearest point on inner polygon edges)
    result = outer.copy()
    inner_closed = np.vstack([inner, inner[0:1]])
    for i in np.where(inside_mask)[0]:
        pt = outer[i]
        best_dist = float("inf")
        best_proj = pt.copy()
        for j in range(len(inner)):
            a = inner_closed[j]
            b = inner_closed[j + 1]
            proj = _project_point_segment(pt, a, b)
            d = np.linalg.norm(pt - proj)
            if d < best_dist:
                best_dist = d
                best_proj = proj
        result[i] = best_proj

    return result


def _points_in_polygon(points, polygon):
    """Ray-casting point-in-polygon test."""
    pts = np.asarray(points)
    poly = np.asarray(polygon)
    n = len(poly)
    inside = np.zeros(len(pts), dtype=bool)
    for i in range(n):
        j = (i + 1) % n
        xi, yi = poly[i]
        xj, yj = poly[j]
        cond1 = (yi > pts[:, 1]) != (yj > pts[:, 1])
        slope = (xj - xi) * (pts[:, 1] - yi) / (yj - yi + 1e-30) + xi
        cond2 = pts[:, 0] < slope
        inside ^= (cond1 & cond2)
    return inside


def _project_point_segment(p, a, b):
    """Project point p onto segment a-b."""
    ab = b - a
    ap = p - a
    t = np.dot(ap, ab) / (np.dot(ab, ab) + 1e-30)
    t = max(0.0, min(1.0, t))
    return a + t * ab


# =========================================================================
# 11. adaptive_extrude
# =========================================================================

def adaptive_extrude(vertices, faces, z_bottom, z_top, n_slices=10, n_profile=32):
    """Build a mesh by sampling cross-sections from a target mesh at
    multiple Z heights and lofting between them.

    This adaptively reconstructs an extrude whose profile varies along Z
    to match the target mesh geometry.

    Args:
        vertices:  (N, 3) target mesh vertices
        faces:     (M, 3) target mesh faces
        z_bottom:  starting Z
        z_top:     ending Z
        n_slices:  number of Z-slices to sample
        n_profile: number of points per cross-section ring

    Returns:
        loft_vertices: (P, 3)
        loft_faces:    (Q, 3)
    """
    z_values = np.linspace(z_bottom, z_top, n_slices)
    rings = []

    for z in z_values:
        cs = extract_cross_section(vertices, faces, z)
        if len(cs) < 3:
            # Fallback: use nearby vertices projected to 2D
            near = np.abs(vertices[:, 2] - z) < (z_top - z_bottom) / (2 * n_slices)
            pts_2d = vertices[near, :2]
            if len(pts_2d) < 3:
                # Use a tiny circle as placeholder
                t = np.linspace(0, 2 * math.pi, n_profile, endpoint=False)
                ring = np.column_stack([0.01 * np.cos(t), 0.01 * np.sin(t)])
            else:
                ring = _resample_polygon(_order_polygon_ccw(pts_2d), n_profile)
        else:
            ring = _resample_polygon(_order_polygon_ccw(cs), n_profile)
        rings.append(ring)

    # Build loft mesh from rings
    all_verts = []
    all_faces = []
    for si, (ring, z) in enumerate(zip(rings, z_values)):
        for pt in ring:
            all_verts.append([pt[0], pt[1], z])

    for si in range(n_slices - 1):
        for j in range(n_profile):
            j_next = (j + 1) % n_profile
            p00 = si * n_profile + j
            p01 = si * n_profile + j_next
            p10 = (si + 1) * n_profile + j
            p11 = (si + 1) * n_profile + j_next
            all_faces.append([p00, p01, p10])
            all_faces.append([p01, p11, p10])

    # Bottom cap
    bot_center = len(all_verts)
    bpts = np.array(all_verts[:n_profile])
    all_verts.append([float(np.mean(bpts[:, 0])), float(np.mean(bpts[:, 1])), z_values[0]])
    for j in range(n_profile):
        j_next = (j + 1) % n_profile
        all_faces.append([bot_center, j_next, j])

    # Top cap
    top_center = len(all_verts)
    top_start = (n_slices - 1) * n_profile
    tpts = np.array(all_verts[top_start:top_start + n_profile])
    all_verts.append([float(np.mean(tpts[:, 0])), float(np.mean(tpts[:, 1])), z_values[-1]])
    for j in range(n_profile):
        j_next = (j + 1) % n_profile
        all_faces.append([top_center, top_start + j, top_start + j_next])

    return np.array(all_verts, dtype=np.float64), np.array(all_faces)


# =========================================================================
# 12. suggest_extrude_adjustments
# =========================================================================

def suggest_extrude_adjustments(cad_profile, mesh_vertices, mesh_faces,
                                 z_bottom, z_top, n_slices=5):
    """Analyze how a CAD extrude's profile compares to the corresponding
    mesh region and suggest ranked adjustments.

    Samples the mesh at several Z heights, compares each cross-section
    to the CAD profile, and returns an ordered list of suggested
    manipulations with parameters.

    Args:
        cad_profile:  (K, 2) the extrude's 2D polygon
        mesh_vertices: (N, 3) target mesh
        mesh_faces:    (M, 3) target mesh faces
        z_bottom:      extrude start Z
        z_top:         extrude end Z
        n_slices:      number of Z-slices to compare

    Returns:
        list of dicts, each with:
            action     — name of the suggested manipulation
            priority   — lower is more important
            params     — suggested parameters for the action
            reason     — human-readable explanation
    """
    cad = np.asarray(cad_profile, dtype=np.float64)
    suggestions = []

    z_values = np.linspace(z_bottom, z_top, max(n_slices, 2))
    slice_metrics = []
    mesh_slices = []

    for z in z_values:
        cs = extract_cross_section(mesh_vertices, mesh_faces, z)
        if len(cs) < 3:
            continue
        metrics = compare_cross_sections(cad, cs)
        slice_metrics.append(metrics)
        mesh_slices.append(cs)

    if not slice_metrics:
        return [{"action": "no_data", "priority": 0,
                 "params": {}, "reason": "No valid cross-sections found in mesh"}]

    # Aggregate metrics
    avg_hausdorff = float(np.mean([m["hausdorff"] for m in slice_metrics]))
    avg_mean_dist = float(np.mean([m["mean_dist"] for m in slice_metrics]))
    avg_area_ratio = float(np.mean([m["area_ratio"] for m in slice_metrics]))
    avg_iou = float(np.mean([m["iou"] for m in slice_metrics]))

    # 1. Scale mismatch → offset_profile
    if abs(avg_area_ratio - 1.0) > 0.1:
        scale_factor = math.sqrt(avg_area_ratio)
        cad_area = abs(_polygon_area_signed(cad))
        approx_perimeter = math.sqrt(cad_area) * 4  # rough
        offset_dist = (scale_factor - 1.0) * math.sqrt(cad_area / math.pi)
        suggestions.append({
            "action": "offset_profile",
            "priority": 1,
            "params": {"distance": round(offset_dist, 4)},
            "reason": f"Area ratio {avg_area_ratio:.2f} — profile needs "
                      f"{'expanding' if avg_area_ratio < 1 else 'shrinking'}",
        })

    # 2. Shape mismatch → fit_primitive
    if avg_mean_dist > 0.5:
        if mesh_slices:
            mid = mesh_slices[len(mesh_slices) // 2]
            fit = fit_primitive_to_cross_section(mid)
            suggestions.append({
                "action": "fit_primitive_to_cross_section",
                "priority": 2,
                "params": {"best_fit": fit["best"], "fit_params": fit["params"]},
                "reason": f"Mean distance {avg_mean_dist:.2f} — try {fit['best']} shape",
            })

    # 3. Taper detection → taper_extrude
    if len(slice_metrics) >= 2:
        first_area = abs(_polygon_area_signed(
            _resample_polygon(mesh_slices[0], 64))) if len(mesh_slices[0]) >= 3 else 0
        last_area = abs(_polygon_area_signed(
            _resample_polygon(mesh_slices[-1], 64))) if len(mesh_slices[-1]) >= 3 else 0
        if first_area > 1e-6 and last_area > 1e-6:
            taper_ratio = math.sqrt(last_area / first_area)
            if abs(taper_ratio - 1.0) > 0.05:
                suggestions.append({
                    "action": "taper_extrude",
                    "priority": 3,
                    "params": {"scale_top": round(taper_ratio, 4)},
                    "reason": f"Cross-section area changes bottom→top "
                              f"(ratio {taper_ratio:.2f})",
                })

    # 4. Twist detection
    if len(mesh_slices) >= 2:
        c0 = _polygon_centroid(mesh_slices[0])
        angles_bottom = np.arctan2(mesh_slices[0][:, 1] - c0[1],
                                    mesh_slices[0][:, 0] - c0[0])
        c_last = _polygon_centroid(mesh_slices[-1])
        angles_top = np.arctan2(mesh_slices[-1][:, 1] - c_last[1],
                                 mesh_slices[-1][:, 0] - c_last[0])
        # Compare mean angle offsets (approximate twist)
        mean_angle_diff = float(np.mean(angles_top)) - float(np.mean(angles_bottom))
        twist_deg = math.degrees(mean_angle_diff)
        if abs(twist_deg) > 3.0:
            suggestions.append({
                "action": "twist_extrude",
                "priority": 4,
                "params": {"total_twist_deg": round(twist_deg, 2)},
                "reason": f"Detected ~{twist_deg:.1f}° angular shift bottom→top",
            })

    # 5. Ellipse fit if circle doesn't fit well
    if mesh_slices:
        mid = mesh_slices[len(mesh_slices) // 2]
        ell = fit_ellipse_to_cross_section(mid)
        a, b = ell["semi_axes"]
        eccentricity = abs(a - b) / max(a, b) if max(a, b) > 1e-6 else 0
        if eccentricity > 0.15:
            suggestions.append({
                "action": "fit_ellipse",
                "priority": 5,
                "params": ell,
                "reason": f"Elliptical cross-section (eccentricity {eccentricity:.2f})",
            })

    # 6. Spline fit for non-primitive shapes
    if avg_mean_dist > 1.0:
        suggestions.append({
            "action": "fit_spline_profile",
            "priority": 6,
            "params": {"n_control": min(len(mesh_slices[0]), 24) if mesh_slices else 12},
            "reason": f"High mean distance ({avg_mean_dist:.2f}) — try spline contour",
        })

    # 7. Adaptive extrude as last resort
    if avg_iou < 0.7:
        suggestions.append({
            "action": "adaptive_extrude",
            "priority": 7,
            "params": {"n_slices": n_slices * 2, "z_bottom": z_bottom, "z_top": z_top},
            "reason": f"Low IoU ({avg_iou:.2f}) — rebuild extrude adaptively",
        })

    # 8. Blend profiles if cross-sections vary
    if len(mesh_slices) >= 2:
        vary = np.std([abs(_polygon_area_signed(_resample_polygon(s, 32)))
                       for s in mesh_slices if len(s) >= 3])
        if vary > 0.5:
            suggestions.append({
                "action": "blend_profiles",
                "priority": 8,
                "params": {"n_slices": len(mesh_slices)},
                "reason": f"Cross-section area varies (std={vary:.2f}) — use blended loft",
            })

    # 9. Boolean union if mesh has protrusions
    if avg_hausdorff > 2.0 and avg_area_ratio < 1.0:
        suggestions.append({
            "action": "boolean_union_profiles",
            "priority": 9,
            "params": {},
            "reason": "Mesh extends beyond CAD profile — add material via union",
        })

    # 10. Boolean difference if mesh has indentations
    if avg_hausdorff > 2.0 and avg_area_ratio > 1.0:
        suggestions.append({
            "action": "boolean_difference_profiles",
            "priority": 10,
            "params": {},
            "reason": "CAD extends beyond mesh — remove material via difference",
        })

    suggestions.sort(key=lambda s: s["priority"])
    return suggestions


# =========================================================================
# Convenience: fit_spline_profile (referenced by suggestions)
# =========================================================================

def fit_spline_profile(points, n_control=16, n_output=64):
    """Fit a smooth closed cubic spline to a 2D point cloud.

    Args:
        points:    (K, 2) unordered or ordered 2D points
        n_control: number of control points to retain
        n_output:  output polygon resolution

    Returns:
        polygon: (n_output, 2) smooth spline contour
    """
    pts = _order_polygon_ccw(np.asarray(points, dtype=np.float64))
    if len(pts) < 4:
        return pts

    # Subsample to n_control points
    control = _resample_polygon(pts, n_control)

    # Parameterize by arc length
    diffs = np.diff(control, axis=0)
    seg_lens = np.linalg.norm(diffs, axis=1)
    # Add closing segment length
    close_len = np.linalg.norm(control[0] - control[-1])
    all_lens = np.append(seg_lens, close_len)
    cum = np.concatenate([[0], np.cumsum(all_lens)])
    total = cum[-1]
    if total < 1e-12:
        return control

    t_ctrl = cum[:-1] / total  # parameter for each control point (0..1 exclusive)

    # For periodic spline, we need t strictly increasing and values that
    # match at t=0 and t=1.  Extend with the wrap-around point.
    t_ext = np.append(t_ctrl, 1.0)
    t_out = np.linspace(0, 1, n_output, endpoint=False)

    result = np.zeros((n_output, 2))
    for dim in range(2):
        vals = np.append(control[:, dim], control[0, dim])
        cs = CubicSpline(t_ext, vals, bc_type="periodic")
        result[:, dim] = cs(t_out)

    return result


# =========================================================================
# Differentiators (moved from adversarial_loop for standard reuse)
# =========================================================================

def _convex_hull_area(points_2d):
    """Approximate convex hull area of 2D points."""
    if len(points_2d) < 3:
        return 0.0
    c = points_2d.mean(axis=0)
    angles = np.arctan2(points_2d[:, 1] - c[1], points_2d[:, 0] - c[0])
    order = np.argsort(angles)
    pts = points_2d[order]
    n = len(pts)
    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += pts[i, 0] * pts[j, 1] - pts[j, 0] * pts[i, 1]
    return abs(area) / 2


def cross_section_contour_diff(cad_v, cad_f, mesh_v, mesh_f, n_slices=10):
    """Compare cross-section contour areas at multiple Z heights."""
    cad_v, mesh_v = np.asarray(cad_v), np.asarray(mesh_v)
    z_min = max(cad_v[:, 2].min(), mesh_v[:, 2].min())
    z_max = min(cad_v[:, 2].max(), mesh_v[:, 2].max())
    if z_max <= z_min:
        return 0.0
    z_vals = np.linspace(z_min + 0.05 * (z_max - z_min),
                          z_max - 0.05 * (z_max - z_min), n_slices)

    total_diff = 0.0
    for z in z_vals:
        tol = (z_max - z_min) / (n_slices * 2)
        cad_near = cad_v[np.abs(cad_v[:, 2] - z) < tol]
        mesh_near = mesh_v[np.abs(mesh_v[:, 2] - z) < tol]
        if len(cad_near) < 3 or len(mesh_near) < 3:
            continue
        cad_2d = cad_near[:, :2]
        mesh_2d = mesh_near[:, :2]
        cad_area = _convex_hull_area(cad_2d)
        mesh_area = _convex_hull_area(mesh_2d)
        total = cad_area + mesh_area
        if total > 1e-6:
            total_diff += abs(cad_area - mesh_area) / total
    return float(total_diff / n_slices * 100)


def z_profile_area_diff(cad_v, cad_f, mesh_v, mesh_f, n_slices=20):
    """Compare total cross-sectional 'area' at each Z height using vertex spread."""
    cad_v, mesh_v = np.asarray(cad_v), np.asarray(mesh_v)
    z_min = max(cad_v[:, 2].min(), mesh_v[:, 2].min())
    z_max = min(cad_v[:, 2].max(), mesh_v[:, 2].max())
    if z_max <= z_min:
        return 0.0

    z_vals = np.linspace(z_min, z_max, n_slices + 2)[1:-1]
    tol = (z_max - z_min) / (n_slices * 2)
    total_diff = 0.0

    for z in z_vals:
        cad_near = cad_v[np.abs(cad_v[:, 2] - z) < tol, :2]
        mesh_near = mesh_v[np.abs(mesh_v[:, 2] - z) < tol, :2]
        if len(cad_near) < 2 or len(mesh_near) < 2:
            continue
        cad_spread = np.prod(np.ptp(cad_near, axis=0) + 1e-12)
        mesh_spread = np.prod(np.ptp(mesh_near, axis=0) + 1e-12)
        total = cad_spread + mesh_spread
        total_diff += abs(cad_spread - mesh_spread) / total

    return float(total_diff / n_slices * 100)


# =========================================================================
# Fixers (moved from adversarial_loop for standard reuse)
# =========================================================================

def fix_cross_section_contour(cad_v, cad_f, mesh_v, mesh_f):
    """Per-Z-slice radial + XY centroid correction."""
    v = np.asarray(cad_v, dtype=np.float64).copy()
    mv = np.asarray(mesh_v)
    z_min = max(v[:, 2].min(), mv[:, 2].min())
    z_max = min(v[:, 2].max(), mv[:, 2].max())
    n_slices = 15
    z_vals = np.linspace(z_min, z_max, n_slices + 2)[1:-1]
    tol = (z_max - z_min) / (n_slices * 2)

    for z in z_vals:
        cad_mask = np.abs(v[:, 2] - z) < tol
        mesh_mask = np.abs(mv[:, 2] - z) < tol
        if np.sum(cad_mask) < 3 or np.sum(mesh_mask) < 3:
            continue
        # Shift XY centroid toward mesh centroid
        cad_c = v[cad_mask, :2].mean(axis=0)
        mesh_c = mv[mesh_mask, :2].mean(axis=0)
        shift = (mesh_c - cad_c) * 0.5
        v[cad_mask, 0] += shift[0]
        v[cad_mask, 1] += shift[1]
        # Radial scale correction
        cad_r = np.linalg.norm(v[cad_mask, :2] - mesh_c, axis=1)
        mesh_r = np.linalg.norm(mv[mesh_mask, :2] - mesh_c, axis=1)
        if cad_r.mean() > 1e-6:
            s = np.clip(mesh_r.mean() / cad_r.mean(), 0.5, 2.0)
            v[cad_mask, :2] = mesh_c + (v[cad_mask, :2] - mesh_c) * s
    return v


def fix_z_profile_area(cad_v, cad_f, mesh_v, mesh_f):
    """Per-Z-slice XY spread correction."""
    v = np.asarray(cad_v, dtype=np.float64).copy()
    mv = np.asarray(mesh_v)
    z_min = max(v[:, 2].min(), mv[:, 2].min())
    z_max = min(v[:, 2].max(), mv[:, 2].max())
    n_slices = 20
    z_vals = np.linspace(z_min, z_max, n_slices + 2)[1:-1]
    tol = (z_max - z_min) / (n_slices * 2)

    for z in z_vals:
        cad_mask = np.abs(v[:, 2] - z) < tol
        mesh_mask = np.abs(mv[:, 2] - z) < tol
        if np.sum(cad_mask) < 2 or np.sum(mesh_mask) < 2:
            continue
        for ax in [0, 1]:
            cad_span = np.ptp(v[cad_mask, ax])
            mesh_span = np.ptp(mv[mesh_mask, ax])
            if cad_span > 1e-6:
                s = np.clip(mesh_span / cad_span, 0.5, 2.0)
                center = v[cad_mask, ax].mean()
                v[cad_mask, ax] = center + (v[cad_mask, ax] - center) * s
    return v


# =========================================================================
# Sweep-along-path alignment (path-based extrusion fitting)
# =========================================================================

def extract_mesh_skeleton(vertices, faces, n_points=20):
    """Extract a centerline skeleton from a mesh by slicing along its
    principal axis and connecting slice centroids.

    Uses PCA to find the longest axis, then samples centroids at
    regular intervals along that axis.

    Args:
        vertices:  (N, 3) mesh vertices
        faces:     (M, 3) mesh faces
        n_points:  number of skeleton points

    Returns:
        skeleton: (n_points, 3) ordered path through the mesh center
        axis_direction: (3,) unit vector of the principal axis
    """
    verts = np.asarray(vertices, dtype=np.float64)
    centroid = verts.mean(axis=0)
    centered = verts - centroid

    # PCA to find principal axis
    cov = centered.T @ centered / len(verts)
    eigvals, eigvecs = np.linalg.eigh(cov)
    # Largest eigenvalue = principal axis
    axis = eigvecs[:, np.argmax(eigvals)]
    if axis[2] < 0:
        axis = -axis

    # Project all vertices onto this axis
    projections = centered @ axis
    t_min, t_max = float(projections.min()), float(projections.max())

    t_values = np.linspace(t_min, t_max, n_points)
    band = (t_max - t_min) / (n_points * 2)

    skeleton = []
    for t in t_values:
        mask = np.abs(projections - t) <= band
        if np.any(mask):
            local_centroid = verts[mask].mean(axis=0)
        else:
            # Interpolate from centroid along axis
            local_centroid = centroid + axis * t
        skeleton.append(local_centroid)

    return np.array(skeleton, dtype=np.float64), axis


def extract_sweep_cross_section(vertices, faces, point, tangent,
                                 radius=None, n_output=32):
    """Extract a 2D cross-section of a mesh perpendicular to a direction
    at a given point.

    Collects vertices near the cutting plane, projects them onto the
    plane defined by (point, tangent), and returns a 2D profile.

    Args:
        vertices:  (N, 3) mesh vertices
        faces:     (M, 3) mesh faces
        point:     (3,) point on the cutting plane
        tangent:   (3,) normal to the cutting plane (path tangent)
        radius:    max distance from point to include (None = auto)
        n_output:  resample output polygon to this many points

    Returns:
        profile_2d: (n_output, 2) cross-section in local frame
        frame:      dict with 'normal' and 'binormal' unit vectors
    """
    verts = np.asarray(vertices, dtype=np.float64)
    point = np.asarray(point, dtype=np.float64)
    tangent = np.asarray(tangent, dtype=np.float64)
    tangent = tangent / max(np.linalg.norm(tangent), 1e-12)

    # Signed distance of each vertex from the cutting plane
    disp = verts - point
    signed_dist = disp @ tangent

    # Auto-determine band thickness
    if radius is None:
        span = np.ptp(verts @ tangent)
        band = span / 40
    else:
        band = radius * 0.5

    mask = np.abs(signed_dist) <= band
    if np.sum(mask) < 3:
        # Widen band
        band *= 3
        mask = np.abs(signed_dist) <= band
    if np.sum(mask) < 3:
        return np.zeros((0, 2)), {"normal": np.zeros(3), "binormal": np.zeros(3)}

    near_pts = verts[mask]

    # Build local 2D frame perpendicular to tangent
    ref = np.array([0.0, 0.0, 1.0])
    if abs(np.dot(tangent, ref)) > 0.9:
        ref = np.array([1.0, 0.0, 0.0])
    normal = np.cross(tangent, ref)
    normal /= max(np.linalg.norm(normal), 1e-12)
    binormal = np.cross(tangent, normal)

    # Project to 2D
    local = near_pts - point
    u = local @ normal
    v_coord = local @ binormal

    pts_2d = np.column_stack([u, v_coord])

    # Order by angle and resample
    c2d = pts_2d.mean(axis=0)
    angles = np.arctan2(pts_2d[:, 1] - c2d[1], pts_2d[:, 0] - c2d[0])
    order = np.argsort(angles)
    ordered = pts_2d[order]

    if len(ordered) >= n_output:
        ordered = _resample_polygon(ordered, n_output)

    return ordered, {"normal": normal, "binormal": binormal}


def fit_sweep_to_mesh(vertices, faces, n_path_points=20, n_profile=24):
    """Automatically fit a sweep (profile + path) to a target mesh.

    Extracts a skeleton path through the mesh, then samples cross-sections
    along the path to determine the profile shape.

    Args:
        vertices:       (N, 3) target mesh
        faces:          (M, 3) target mesh faces
        n_path_points:  number of path points
        n_profile:      profile resolution

    Returns:
        dict with:
            path:     (M, 3) fitted path
            profile:  (K, 2) representative 2D profile
            profiles: list of (K, 2) per-path-point cross-sections
            scales:   (M,) scale factor at each path point
    """
    verts = np.asarray(vertices, dtype=np.float64)
    faces_arr = np.asarray(faces)

    # Extract skeleton
    skeleton, axis = extract_mesh_skeleton(verts, faces_arr, n_path_points)

    from meshxcad.objects.operations import compute_frenet_frames
    tangents, normals, binormals = compute_frenet_frames(skeleton)

    # Extract cross-sections at each path point
    profiles = []
    scales = []
    for i in range(len(skeleton)):
        cs, _ = extract_sweep_cross_section(
            verts, faces_arr, skeleton[i], tangents[i],
            n_output=n_profile)
        if len(cs) >= 3:
            # Measure scale as RMS distance from centroid
            c = cs.mean(axis=0)
            scale = float(np.sqrt(np.mean(np.sum((cs - c) ** 2, axis=1))))
            profiles.append(cs - c)  # centered
            scales.append(scale)
        else:
            profiles.append(None)
            scales.append(0.0)

    # Find representative profile (median scale, non-None)
    valid_indices = [i for i, p in enumerate(profiles) if p is not None]
    if not valid_indices:
        # Fallback: circular profile
        t = np.linspace(0, 2 * np.pi, n_profile, endpoint=False)
        circle = np.column_stack([np.cos(t), np.sin(t)])
        return {"path": skeleton, "profile": circle,
                "profiles": profiles, "scales": np.ones(len(skeleton))}

    valid_scales = [scales[i] for i in valid_indices]
    median_scale = float(np.median(valid_scales))
    # Pick the profile closest to median scale
    best_idx = valid_indices[
        int(np.argmin([abs(scales[i] - median_scale) for i in valid_indices]))]
    representative = profiles[best_idx]

    # Normalize scales relative to the representative
    if median_scale > 1e-12:
        norm_scales = np.array(scales) / median_scale
    else:
        norm_scales = np.ones(len(skeleton))

    return {
        "path": skeleton,
        "profile": representative,
        "profiles": profiles,
        "scales": norm_scales,
    }


def compare_sweep_to_mesh(sweep_v, sweep_f, mesh_v, mesh_f):
    """Compare a sweep-generated mesh to a target mesh.

    Returns quality metrics: Hausdorff distance, mean surface distance,
    volume overlap estimate, and cross-section alignment.

    Args:
        sweep_v, sweep_f: sweep mesh
        mesh_v, mesh_f:   target mesh

    Returns:
        dict with hausdorff, mean_dist, coverage, quality_score (0-100)
    """
    from scipy.spatial import KDTree
    sv = np.asarray(sweep_v, dtype=np.float64)
    mv = np.asarray(mesh_v, dtype=np.float64)

    tree_m = KDTree(mv)
    d_s2m, _ = tree_m.query(sv)
    tree_s = KDTree(sv)
    d_m2s, _ = tree_s.query(mv)

    hausdorff = float(max(np.max(d_s2m), np.max(d_m2s)))
    mean_dist = float((np.mean(d_s2m) + np.mean(d_m2s)) / 2)

    # Coverage: fraction of mesh vertices within 1 median-distance of sweep
    median_d = float(np.median(d_m2s))
    threshold = max(median_d * 2, 0.5)
    coverage = float(np.mean(d_m2s < threshold))

    # Quality score: 100 = perfect, 0 = terrible
    # Based on inverse of normalized mean distance
    mesh_span = float(np.max(np.ptp(mv, axis=0)))
    if mesh_span > 1e-12:
        normalized_dist = mean_dist / mesh_span
        quality = max(0.0, min(100.0, 100 * (1 - normalized_dist * 10)))
    else:
        quality = 0.0

    return {
        "hausdorff": hausdorff,
        "mean_dist": mean_dist,
        "coverage": coverage,
        "quality_score": round(quality, 1),
    }


def refine_sweep_path(path, mesh_vertices, mesh_faces, iterations=3):
    """Refine a sweep path so it better tracks the mesh centerline.

    At each path point, shifts toward the centroid of nearby mesh vertices
    while maintaining smoothness.

    Args:
        path:           (M, 3) current path
        mesh_vertices:  (N, 3) target mesh
        mesh_faces:     (F, 3) target mesh faces
        iterations:     number of refinement passes

    Returns:
        refined_path: (M, 3)
    """
    from meshxcad.objects.operations import compute_frenet_frames
    p = np.asarray(path, dtype=np.float64).copy()
    mv = np.asarray(mesh_vertices, dtype=np.float64)
    n = len(p)

    for _ in range(iterations):
        tangents, _, _ = compute_frenet_frames(p)

        for i in range(1, n - 1):  # keep endpoints fixed
            # Find mesh vertices near this path point
            disp = mv - p[i]
            dists = np.linalg.norm(disp, axis=1)
            # Use vertices within a local neighborhood
            spacing = np.linalg.norm(p[min(i+1, n-1)] - p[max(i-1, 0)]) / 2
            radius = max(spacing * 1.5, 0.5)
            mask = dists < radius

            if np.sum(mask) >= 3:
                local_centroid = mv[mask].mean(axis=0)
                # Move path point toward centroid, but constrained
                # to not move along the tangent (preserve path parameter)
                shift = local_centroid - p[i]
                # Remove tangent component to stay on the "same" path station
                tangent_comp = np.dot(shift, tangents[i]) * tangents[i]
                lateral_shift = shift - tangent_comp
                p[i] += lateral_shift * 0.5

        # Light smoothing pass
        smoothed = p.copy()
        for i in range(1, n - 1):
            smoothed[i] = 0.5 * p[i] + 0.25 * (p[i-1] + p[i+1])
        p = smoothed

    return p


def refine_sweep_profile(profile, path, mesh_vertices, mesh_faces,
                          n_samples=5):
    """Refine a sweep profile by comparing cross-sections at multiple
    path points to the actual mesh cross-sections.

    Args:
        profile:        (K, 2) current profile
        path:           (M, 3) sweep path
        mesh_vertices:  (N, 3) target mesh
        mesh_faces:     (F, 3) target mesh faces
        n_samples:      number of path points to sample

    Returns:
        refined_profile: (K, 2)
    """
    from meshxcad.objects.operations import compute_frenet_frames
    prof = np.asarray(profile, dtype=np.float64).copy()
    path_arr = np.asarray(path, dtype=np.float64)
    n_path = len(path_arr)
    n_pts = len(prof)

    tangents, _, _ = compute_frenet_frames(path_arr)

    # Sample evenly along path
    indices = np.linspace(1, n_path - 2, min(n_samples, n_path - 2),
                          dtype=int)

    accumulated = np.zeros_like(prof)
    count = 0
    for idx in indices:
        cs, _ = extract_sweep_cross_section(
            mesh_vertices, mesh_faces,
            path_arr[idx], tangents[idx],
            n_output=n_pts)
        if len(cs) >= 3:
            # Center the cross-section
            cs_centered = cs - cs.mean(axis=0)
            # Accumulate
            if len(cs_centered) == n_pts:
                accumulated += cs_centered
                count += 1

    if count > 0:
        average_cs = accumulated / count
        # Blend current profile toward average cross-section
        refined = prof * 0.4 + average_cs * 0.6
        return refined

    return prof


def adaptive_sweep_extrude(mesh_vertices, mesh_faces,
                            n_path=20, n_profile=24):
    """Automatically generate a sweep mesh that approximates a target mesh.

    Extracts a skeleton path, fits cross-sections, and sweeps to create
    a clean CAD-style mesh.

    Args:
        mesh_vertices: (N, 3) target mesh
        mesh_faces:    (M, 3) target mesh faces
        n_path:        number of path points
        n_profile:     profile resolution

    Returns:
        vertices: (P, 3) sweep mesh vertices
        faces:    (Q, 3) sweep mesh faces
        fit_info: dict with path, profile, scales, quality
    """
    from meshxcad.objects.operations import sweep_along_path

    # Step 1: Fit sweep parameters to mesh
    fit = fit_sweep_to_mesh(mesh_vertices, mesh_faces,
                            n_path_points=n_path, n_profile=n_profile)

    # Step 2: Refine path
    refined_path = refine_sweep_path(
        fit["path"], mesh_vertices, mesh_faces, iterations=3)

    # Step 3: Refine profile
    refined_profile = refine_sweep_profile(
        fit["profile"], refined_path, mesh_vertices, mesh_faces,
        n_samples=min(5, n_path - 2))

    # Step 4: Build scale function from extracted scales
    scales = fit["scales"]
    def scale_fn(t):
        idx = min(int(t * (len(scales) - 1)), len(scales) - 1)
        return float(scales[idx])

    # Step 5: Generate sweep mesh
    sweep_v, sweep_f = sweep_along_path(
        refined_profile, refined_path,
        n_profile=n_profile, scale_fn=scale_fn)

    # Step 6: Measure quality
    quality = compare_sweep_to_mesh(sweep_v, sweep_f,
                                     mesh_vertices, mesh_faces)

    return sweep_v, sweep_f, {
        "path": refined_path,
        "profile": refined_profile,
        "scales": scales,
        "quality": quality,
    }


def detect_sweep_candidate(vertices, faces):
    """Detect whether a mesh looks like it could be a sweep (elongated
    shape with consistent cross-sections).

    Args:
        vertices: (N, 3) mesh vertices
        faces:    (M, 3) mesh faces

    Returns:
        dict with:
            is_sweep:     bool — True if mesh appears sweep-like
            elongation:   ratio of longest to shortest axis
            consistency:  0-1 cross-section consistency score
            axis:         (3,) principal axis direction
            confidence:   0-1 overall confidence
    """
    verts = np.asarray(vertices, dtype=np.float64)

    # PCA
    centroid = verts.mean(axis=0)
    centered = verts - centroid
    cov = centered.T @ centered / len(verts)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]

    axis = eigvecs[:, 0]
    if axis[2] < 0:
        axis = -axis

    # Elongation: ratio of largest to second-largest eigenvalue
    if eigvals[1] > 1e-12:
        elongation = float(np.sqrt(eigvals[0] / eigvals[1]))
    else:
        elongation = float("inf")

    # Cross-section consistency: sample slices and compare areas
    projections = centered @ axis
    t_min, t_max = projections.min(), projections.max()
    n_slices = 10
    t_values = np.linspace(t_min + 0.1 * (t_max - t_min),
                           t_max - 0.1 * (t_max - t_min), n_slices)
    band = (t_max - t_min) / (n_slices * 2)

    areas = []
    for t in t_values:
        mask = np.abs(projections - t) <= band
        if np.sum(mask) < 3:
            continue
        local = verts[mask]
        # Project onto plane perpendicular to principal axis
        local_centered = local - local.mean(axis=0)
        # Use the two minor eigenvectors as the cross-section plane
        u_coords = local_centered @ eigvecs[:, 1]
        v_coords = local_centered @ eigvecs[:, 2]
        area_proxy = float((np.ptp(u_coords) + 1e-12) *
                           (np.ptp(v_coords) + 1e-12))
        areas.append(area_proxy)

    if len(areas) >= 3:
        areas_arr = np.array(areas)
        mean_area = float(areas_arr.mean())
        if mean_area > 1e-12:
            consistency = float(1.0 - min(areas_arr.std() / mean_area, 1.0))
        else:
            consistency = 0.0
    else:
        consistency = 0.0

    # A sweep candidate has elongation > 1.5 and good consistency
    is_sweep = elongation > 1.5 and consistency > 0.3
    confidence = min(1.0, (elongation - 1.0) / 3.0) * consistency

    return {
        "is_sweep": is_sweep,
        "elongation": round(elongation, 2),
        "consistency": round(consistency, 3),
        "axis": axis,
        "confidence": round(confidence, 3),
    }


def suggest_sweep_adjustments(sweep_v, sweep_f, mesh_v, mesh_f,
                               path=None, profile=None):
    """Analyze a sweep mesh vs a target mesh and suggest improvements.

    Args:
        sweep_v, sweep_f: current sweep mesh
        mesh_v, mesh_f:   target mesh
        path:             (M, 3) current path (optional)
        profile:          (K, 2) current profile (optional)

    Returns:
        list of dicts with action, priority, params, reason
    """
    metrics = compare_sweep_to_mesh(sweep_v, sweep_f, mesh_v, mesh_f)
    suggestions = []

    # 1. Poor coverage
    if metrics["coverage"] < 0.7:
        suggestions.append({
            "action": "refine_sweep_path",
            "priority": 1,
            "params": {"iterations": 5},
            "reason": f"Low coverage ({metrics['coverage']:.1%}) — "
                      f"path may not follow mesh centerline",
        })

    # 2. High mean distance
    if metrics["mean_dist"] > 1.0:
        suggestions.append({
            "action": "refine_sweep_profile",
            "priority": 2,
            "params": {"n_samples": 8},
            "reason": f"Mean distance {metrics['mean_dist']:.2f} — "
                      f"profile shape needs adjustment",
        })

    # 3. High Hausdorff
    if metrics["hausdorff"] > 5.0:
        suggestions.append({
            "action": "adaptive_sweep_extrude",
            "priority": 3,
            "params": {"n_path": 30, "n_profile": 32},
            "reason": f"Hausdorff {metrics['hausdorff']:.2f} — "
                      f"full adaptive rebuild recommended",
        })

    # 4. Low quality
    if metrics["quality_score"] < 50:
        suggestions.append({
            "action": "fit_sweep_to_mesh",
            "priority": 4,
            "params": {"n_path_points": 25, "n_profile": 32},
            "reason": f"Quality score {metrics['quality_score']} — "
                      f"re-fit sweep parameters from scratch",
        })

    # 5. Path needs more detail
    if path is not None and len(path) < 15:
        suggestions.append({
            "action": "fit_sweep_to_mesh",
            "priority": 5,
            "params": {"n_path_points": 30},
            "reason": f"Path has only {len(path)} points — "
                      f"increase resolution for better fit",
        })

    # 6. Profile may need higher resolution
    if profile is not None and len(profile) < 20:
        suggestions.append({
            "action": "refine_sweep_profile",
            "priority": 6,
            "params": {"n_samples": 10},
            "reason": f"Profile has only {len(profile)} points — "
                      f"increase resolution",
        })

    suggestions.sort(key=lambda s: s["priority"])
    return suggestions if suggestions else [{
        "action": "none",
        "priority": 0,
        "params": {},
        "reason": f"Sweep fits well (quality={metrics['quality_score']})",
    }]


# =========================================================================
# Extended extrude/sweep differentiators (round 2)
# =========================================================================

def taper_consistency_diff(cad_v, cad_f, mesh_v, mesh_f, n_slices=10):
    """Compare how cross-section area varies along Z (taper consistency)."""
    def _taper_profile(v, f, n):
        v = np.asarray(v, dtype=np.float64)
        z_min, z_max = float(v[:, 2].min()), float(v[:, 2].max())
        if z_max - z_min < 1e-6:
            return np.ones(n)
        z_vals = np.linspace(z_min + 0.1 * (z_max - z_min),
                             z_max - 0.1 * (z_max - z_min), n)
        areas = []
        for z in z_vals:
            cs = extract_cross_section(v, f, z)
            if len(cs) >= 3:
                areas.append(_convex_hull_area(cs))
            else:
                areas.append(0.0)
        arr = np.array(areas)
        total = arr.sum()
        return arr / max(total, 1e-12)

    tp_cad = _taper_profile(cad_v, cad_f, n_slices)
    tp_mesh = _taper_profile(mesh_v, mesh_f, n_slices)
    return float(np.sum(np.abs(tp_cad - tp_mesh)) * 50)


def sweep_path_deviation(cad_v, cad_f, mesh_v, mesh_f, n_points=15):
    """Compare skeleton centerlines of two meshes."""
    skel_cad, _ = extract_mesh_skeleton(cad_v, cad_f, n_points=n_points)
    skel_mesh, _ = extract_mesh_skeleton(mesh_v, mesh_f, n_points=n_points)

    if len(skel_cad) == 0 or len(skel_mesh) == 0:
        return 0.0

    tree = KDTree(skel_mesh)
    dists, _ = tree.query(skel_cad)
    bbox_diag = float(np.linalg.norm(
        np.asarray(mesh_v).max(axis=0) - np.asarray(mesh_v).min(axis=0)))
    return float(np.mean(dists) / max(bbox_diag, 1e-12) * 100)


def profile_circularity_diff(cad_v, cad_f, mesh_v, mesh_f, n_slices=8):
    """Compare cross-section circularity at multiple heights."""
    def _circularity_profile(v, f, n):
        v = np.asarray(v, dtype=np.float64)
        z_min, z_max = float(v[:, 2].min()), float(v[:, 2].max())
        if z_max - z_min < 1e-6:
            return np.ones(n)
        z_vals = np.linspace(z_min + 0.15 * (z_max - z_min),
                             z_max - 0.15 * (z_max - z_min), n)
        circs = []
        for z in z_vals:
            cs = extract_cross_section(v, f, z)
            if len(cs) >= 3:
                center = cs.mean(axis=0)
                radii = np.linalg.norm(cs - center, axis=1)
                if radii.mean() > 1e-8:
                    circs.append(float(radii.std() / radii.mean()))
                else:
                    circs.append(0.0)
            else:
                circs.append(0.0)
        return np.array(circs)

    c_cad = _circularity_profile(cad_v, cad_f, n_slices)
    c_mesh = _circularity_profile(mesh_v, mesh_f, n_slices)
    return float(np.mean(np.abs(c_cad - c_mesh)) * 100)


def extrude_twist_diff(cad_v, cad_f, mesh_v, mesh_f, n_slices=8):
    """Detect rotational offset between cross-sections at different heights."""
    def _twist_angles(v, f, n):
        v = np.asarray(v, dtype=np.float64)
        z_min, z_max = float(v[:, 2].min()), float(v[:, 2].max())
        if z_max - z_min < 1e-6:
            return np.zeros(n)
        z_vals = np.linspace(z_min + 0.15 * (z_max - z_min),
                             z_max - 0.15 * (z_max - z_min), n)
        angles = []
        prev_angle = None
        for z in z_vals:
            cs = extract_cross_section(v, f, z)
            if len(cs) >= 3:
                center = cs.mean(axis=0)
                # Find the angle of the point farthest from center
                diffs = cs - center
                idx = np.argmax(np.linalg.norm(diffs, axis=1))
                angle = math.atan2(diffs[idx, 1], diffs[idx, 0])
                if prev_angle is not None:
                    delta = angle - prev_angle
                    # Normalize to [-pi, pi]
                    delta = (delta + math.pi) % (2 * math.pi) - math.pi
                    angles.append(abs(delta))
                prev_angle = angle
            else:
                angles.append(0.0)
        return np.array(angles) if angles else np.zeros(1)

    t_cad = _twist_angles(cad_v, cad_f, n_slices)
    t_mesh = _twist_angles(mesh_v, mesh_f, n_slices)
    n = min(len(t_cad), len(t_mesh))
    if n == 0:
        return 0.0
    return float(np.mean(np.abs(t_cad[:n] - t_mesh[:n])) * 100 / math.pi)


# =========================================================================
# Extended extrude/sweep fixers (round 2)
# =========================================================================

def fix_taper_consistency(cad_v, cad_f, mesh_v, mesh_f):
    """Per-Z-slice area scaling to match mesh taper."""
    v = np.asarray(cad_v, dtype=np.float64).copy()
    mv = np.asarray(mesh_v)
    mf = np.asarray(mesh_f)
    z_min = max(v[:, 2].min(), mv[:, 2].min())
    z_max = min(v[:, 2].max(), mv[:, 2].max())
    if z_max - z_min < 1e-6:
        return v

    n_slices = 12
    z_vals = np.linspace(z_min, z_max, n_slices + 2)[1:-1]
    tol = (z_max - z_min) / (n_slices * 2)

    for z in z_vals:
        cad_mask = np.abs(v[:, 2] - z) < tol
        mesh_mask = np.abs(mv[:, 2] - z) < tol
        if np.sum(cad_mask) < 3 or np.sum(mesh_mask) < 3:
            continue

        # Radial scaling from centroid
        c_center = v[cad_mask, :2].mean(axis=0)
        m_center = mv[mesh_mask, :2].mean(axis=0)

        c_radii = np.linalg.norm(v[cad_mask, :2] - c_center, axis=1)
        m_radii = np.linalg.norm(mv[mesh_mask, :2] - m_center, axis=1)

        if c_radii.mean() > 1e-6:
            s = np.clip(m_radii.mean() / c_radii.mean(), 0.5, 2.0)
            v[cad_mask, :2] = c_center + (v[cad_mask, :2] - c_center) * s
            # Also shift centroid
            v[cad_mask, :2] += (m_center - c_center) * 0.3

    return v


def fix_sweep_path_deviation(cad_v, cad_f, mesh_v, mesh_f):
    """Shift vertices along skeleton direction toward mesh skeleton."""
    v = np.asarray(cad_v, dtype=np.float64).copy()
    mv = np.asarray(mesh_v)
    mf = np.asarray(mesh_f)
    f = np.asarray(cad_f)

    skel_mesh, _ = extract_mesh_skeleton(mv, mf, n_points=15)
    if len(skel_mesh) < 2:
        return v

    tree = KDTree(mv)
    _, idx = tree.query(v)
    v = v * 0.65 + mv[idx] * 0.35
    return v


def fix_profile_circularity(cad_v, cad_f, mesh_v, mesh_f):
    """Per-slice radial adjustment to match mesh circularity."""
    v = np.asarray(cad_v, dtype=np.float64).copy()
    mv = np.asarray(mesh_v)
    z_min = max(v[:, 2].min(), mv[:, 2].min())
    z_max = min(v[:, 2].max(), mv[:, 2].max())
    if z_max - z_min < 1e-6:
        return v

    n_slices = 10
    z_vals = np.linspace(z_min, z_max, n_slices + 2)[1:-1]
    tol = (z_max - z_min) / (n_slices * 2)

    for z in z_vals:
        cad_mask = np.abs(v[:, 2] - z) < tol
        mesh_mask = np.abs(mv[:, 2] - z) < tol
        if np.sum(cad_mask) < 3 or np.sum(mesh_mask) < 3:
            continue

        c_center = v[cad_mask, :2].mean(axis=0)
        m_center = mv[mesh_mask, :2].mean(axis=0)
        m_radii = np.linalg.norm(mv[mesh_mask, :2] - m_center, axis=1)
        target_r = float(np.median(m_radii))

        c_diffs = v[cad_mask, :2] - c_center
        c_radii = np.linalg.norm(c_diffs, axis=1, keepdims=True)
        c_radii[c_radii < 1e-8] = 1e-8
        c_dirs = c_diffs / c_radii

        new_radii = c_radii * 0.6 + target_r * 0.4
        v[cad_mask, :2] = c_center + c_dirs * new_radii

    return v


def fix_extrude_twist(cad_v, cad_f, mesh_v, mesh_f):
    """Apply per-slice rotation correction."""
    v = np.asarray(cad_v, dtype=np.float64).copy()
    mv = np.asarray(mesh_v)
    mf = np.asarray(mesh_f)
    z_min = max(v[:, 2].min(), mv[:, 2].min())
    z_max = min(v[:, 2].max(), mv[:, 2].max())
    if z_max - z_min < 1e-6:
        return v

    n_slices = 8
    z_vals = np.linspace(z_min, z_max, n_slices + 2)[1:-1]
    tol = (z_max - z_min) / (n_slices * 2)

    for z in z_vals:
        cad_mask = np.abs(v[:, 2] - z) < tol
        mesh_mask = np.abs(mv[:, 2] - z) < tol
        if np.sum(cad_mask) < 3 or np.sum(mesh_mask) < 3:
            continue

        c_center = v[cad_mask, :2].mean(axis=0)
        m_center = mv[mesh_mask, :2].mean(axis=0)

        # Find principal angle of each slice
        def _principal_angle(pts, center):
            diffs = pts - center
            if len(diffs) < 2:
                return 0.0
            cov = diffs.T @ diffs / len(diffs)
            _, evecs = np.linalg.eigh(cov)
            return math.atan2(evecs[1, -1], evecs[0, -1])

        c_angle = _principal_angle(v[cad_mask, :2], c_center)
        m_angle = _principal_angle(mv[mesh_mask, :2], m_center)
        delta = m_angle - c_angle
        delta = (delta + math.pi) % (2 * math.pi) - math.pi

        # Apply partial rotation correction
        delta *= 0.5
        cos_d, sin_d = math.cos(delta), math.sin(delta)
        centered = v[cad_mask, :2] - c_center
        rotated = np.column_stack([
            centered[:, 0] * cos_d - centered[:, 1] * sin_d,
            centered[:, 0] * sin_d + centered[:, 1] * cos_d,
        ])
        v[cad_mask, :2] = c_center + rotated

    return v
