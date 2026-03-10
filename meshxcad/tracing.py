"""Tracing-based CAD reconstruction from mesh.

Segments a mesh into parts, traces 2D profiles from projections of each
segment, and reconstructs the full model via extrude/revolve/loft/sweep
operations with scaling along paths.

This is the primary entry point for "from scratch" CAD creation when no
input CAD model is given.

Pipeline:
1. (Optional) Template matching — identify object class
2. Segment mesh into parts
3. For each segment, extract 2D profile + path
4. Reconstruct each segment via appropriate CAD operation
5. Combine all segments into one mesh
6. Score quality against original mesh
"""

import numpy as np
from scipy.spatial import KDTree

from .segmentation import (
    segment_mesh, MeshSegment, classify_segment_action,
)
from .objects.builder import revolve_profile, combine_meshes
from .objects.operations import (
    extrude_polygon, sweep_along_path, compute_frenet_frames,
)
from .general_align import hausdorff_distance
from .stl_io import read_binary_stl, write_binary_stl


# ---------------------------------------------------------------------------
# Per-segment reconstruction
# ---------------------------------------------------------------------------

def _reconstruct_revolve(segment):
    """Reconstruct a segment as a revolve solid."""
    if segment.profile is None or len(segment.profile) < 2:
        return _reconstruct_freeform(segment)

    profile_rz = np.asarray(segment.profile)
    # Ensure profile is sorted by Z
    order = np.argsort(profile_rz[:, 1])
    profile_rz = profile_rz[order]

    n_angular = min(64, max(16, len(segment.vertices) // 10))

    close_top = True
    close_bottom = True
    # If the profile doesn't reach r=0 at either end, close the caps
    if profile_rz[0, 0] < 1e-6:
        close_bottom = False
    if profile_rz[-1, 0] < 1e-6:
        close_top = False

    verts, faces = revolve_profile(profile_rz, n_angular,
                                    close_top=close_top,
                                    close_bottom=close_bottom)

    # Align the revolved mesh to the segment's position and orientation
    verts = _align_to_segment(verts, segment)
    return verts, faces


def _reconstruct_extrude(segment):
    """Reconstruct a segment as an extruded 2D profile."""
    if segment.profile is None or len(segment.profile) < 3:
        return _reconstruct_freeform(segment)

    profile_2d = np.asarray(segment.profile)

    # Compute extrusion height from segment extent along primary axis
    v = segment.vertices
    proj = (v - segment.centroid) @ segment.primary_axis
    height = float(proj.max() - proj.min())
    height = max(height, 1e-6)

    n_height = max(2, min(20, int(height * 2)))

    verts, faces = extrude_polygon(profile_2d, height, n_height)

    # Offset to correct position
    verts[:, 2] += float(proj.min())
    verts = _align_to_segment(verts, segment)
    return verts, faces


def _reconstruct_sweep(segment):
    """Reconstruct a segment as a sweep with scaling along path."""
    if segment.path is None or len(segment.path) < 2:
        return _reconstruct_loft(segment)

    path = np.asarray(segment.path)
    n_path = len(path)

    # Build profile (circle if no explicit profile)
    if segment.profile is not None and len(segment.profile) >= 3:
        profile_2d = np.asarray(segment.profile)
        # Center the profile
        profile_2d = profile_2d - profile_2d.mean(axis=0)
    else:
        # Estimate radius from segment
        radii = np.linalg.norm(segment.vertices - segment.centroid, axis=1)
        r = float(np.median(radii)) * 0.5
        n_pts = 16
        angles = np.linspace(0, 2 * np.pi, n_pts, endpoint=False)
        profile_2d = np.column_stack([r * np.cos(angles), r * np.sin(angles)])

    # Scale function from scale_along_path
    if segment.scale_along_path is not None and len(segment.scale_along_path) == n_path:
        scale_arr = segment.scale_along_path
    else:
        # Default: taper from 1.0 to 0.3
        scale_arr = np.linspace(1.0, 0.3, n_path)

    # Interpolate scale as a callable
    t_norm = np.linspace(0, 1, n_path)

    def scale_fn(t):
        return float(np.interp(t, t_norm, scale_arr))

    n_profile = max(12, len(profile_2d))

    try:
        verts, faces = sweep_along_path(profile_2d, path, n_profile, scale_fn)
    except Exception:
        return _reconstruct_loft(segment)

    return verts, faces


def _reconstruct_loft(segment):
    """Reconstruct a segment as a loft (extrude with varying cross-sections)."""
    v = segment.vertices
    center = segment.centroid
    axis = segment.primary_axis

    centered = v - center
    proj = centered @ axis

    n_stations = min(10, max(3, len(v) // 30))
    t_vals = np.linspace(proj.min(), proj.max(), n_stations)
    half_w = (proj.max() - proj.min()) / n_stations * 1.5

    # Gather cross-section rings
    rings = []
    for t in t_vals:
        mask = np.abs(proj - t) < half_w
        if mask.sum() < 3:
            continue
        local = v[mask]
        local_center = local.mean(axis=0)

        # Project to perpendicular plane
        offsets = local - local_center
        # Build local coordinate system
        u, w = _perp_axes(axis)
        pts_2d_u = offsets @ u
        pts_2d_w = offsets @ w

        # Compute convex hull-like profile
        radii = np.sqrt(pts_2d_u**2 + pts_2d_w**2)
        r_median = float(np.median(radii))

        # Generate ring at this station
        n_ring = 16
        angles = np.linspace(0, 2 * np.pi, n_ring, endpoint=False)
        ring = local_center[None, :] + r_median * (
            np.cos(angles)[:, None] * u[None, :] +
            np.sin(angles)[:, None] * w[None, :]
        )
        rings.append(ring)

    if len(rings) < 2:
        return _reconstruct_freeform(segment)

    # Build lofted mesh from rings
    return _loft_rings(rings)


def _reconstruct_freeform(segment):
    """Fallback: return the segment mesh as-is (cleaned)."""
    return segment.vertices.copy(), segment.faces.copy()


def _perp_axes(axis):
    """Compute two axes perpendicular to the given axis."""
    a = np.asarray(axis, dtype=np.float64)
    a = a / max(np.linalg.norm(a), 1e-12)

    # Pick a non-parallel vector
    if abs(a[0]) < 0.9:
        ref = np.array([1, 0, 0.0])
    else:
        ref = np.array([0, 1, 0.0])

    u = np.cross(a, ref)
    u = u / max(np.linalg.norm(u), 1e-12)
    w = np.cross(a, u)
    w = w / max(np.linalg.norm(w), 1e-12)
    return u, w


def _loft_rings(rings):
    """Build a mesh by connecting a sequence of vertex rings."""
    all_verts = []
    all_faces = []
    offset = 0
    n_ring = len(rings[0])

    for ring in rings:
        all_verts.append(ring)

    verts = np.vstack(all_verts)
    n_stations = len(rings)

    # Side faces
    for si in range(n_stations - 1):
        base = si * n_ring
        next_base = (si + 1) * n_ring
        for ri in range(n_ring):
            r_next = (ri + 1) % n_ring
            # Two triangles per quad
            all_faces.append([base + ri, next_base + ri, next_base + r_next])
            all_faces.append([base + ri, next_base + r_next, base + r_next])

    # Cap bottom
    bottom_center = verts[:n_ring].mean(axis=0)
    ci = len(verts)
    verts = np.vstack([verts, bottom_center[None, :]])
    for ri in range(n_ring):
        r_next = (ri + 1) % n_ring
        all_faces.append([ci, ri, r_next])

    # Cap top
    top_base = (n_stations - 1) * n_ring
    top_center = verts[top_base:top_base + n_ring].mean(axis=0)
    ci2 = len(verts)
    verts = np.vstack([verts, top_center[None, :]])
    for ri in range(n_ring):
        r_next = (ri + 1) % n_ring
        all_faces.append([ci2, top_base + r_next, top_base + ri])

    faces = np.array(all_faces, dtype=np.int64)
    return verts, faces


def _align_to_segment(verts, segment):
    """Transform reconstructed vertices to match segment position/orientation."""
    # The reconstructed mesh is in a canonical frame (Z-up, centered at origin).
    # Rotate so Z maps to segment's primary axis, then translate to centroid.
    target_axis = segment.primary_axis / max(np.linalg.norm(segment.primary_axis), 1e-12)
    z_axis = np.array([0, 0, 1.0])

    if np.allclose(target_axis, z_axis, atol=1e-6):
        R = np.eye(3)
    elif np.allclose(target_axis, -z_axis, atol=1e-6):
        R = np.diag([1, -1, -1.0])
    else:
        v_cross = np.cross(z_axis, target_axis)
        s = np.linalg.norm(v_cross)
        c = np.dot(z_axis, target_axis)
        vx = np.array([
            [0, -v_cross[2], v_cross[1]],
            [v_cross[2], 0, -v_cross[0]],
            [-v_cross[1], v_cross[0], 0],
        ])
        R = np.eye(3) + vx + vx @ vx * (1 - c) / max(s * s, 1e-12)

    verts = (R @ verts.T).T

    # Center on segment centroid
    current_center = verts.mean(axis=0)
    verts = verts - current_center + segment.centroid

    return verts


# ---------------------------------------------------------------------------
# Full tracing reconstruction pipeline
# ---------------------------------------------------------------------------

def trace_reconstruct(vertices, faces, template=None, strategy="auto"):
    """Full tracing-based CAD reconstruction.

    Args:
        vertices: (N, 3) input mesh vertices
        faces:    (M, 3) input mesh faces
        template: optional ObjectTemplate to guide segmentation
        strategy: segmentation strategy override

    Returns:
        dict with:
            cad_vertices   — (P, 3) reconstructed vertices
            cad_faces      — (Q, 3) reconstructed faces
            segments       — list of MeshSegment
            quality        — 0-1 quality score
            n_segments     — number of segments
    """
    v = np.asarray(vertices, dtype=np.float64)
    f = np.asarray(faces)

    # 1. Segment
    segments = segment_mesh(v, f, strategy=strategy, template=template)

    if len(segments) == 0:
        # Fallback: treat entire mesh as one segment
        return {
            "cad_vertices": v.copy(),
            "cad_faces": f.copy(),
            "segments": [],
            "quality": 0.5,
            "n_segments": 0,
        }

    # 2. Reconstruct each segment
    reconstructed_parts = []
    for seg in segments:
        dispatch = {
            "revolve": _reconstruct_revolve,
            "extrude": _reconstruct_extrude,
            "sweep": _reconstruct_sweep,
            "loft": _reconstruct_loft,
            "freeform": _reconstruct_freeform,
        }
        fn = dispatch.get(seg.cad_action, _reconstruct_freeform)
        try:
            part_v, part_f = fn(seg)
            if len(part_v) >= 3 and len(part_f) >= 1:
                reconstructed_parts.append((part_v, part_f))
            else:
                reconstructed_parts.append((seg.vertices.copy(), seg.faces.copy()))
        except Exception:
            reconstructed_parts.append((seg.vertices.copy(), seg.faces.copy()))

    # 3. Combine
    cad_verts, cad_faces = combine_meshes(reconstructed_parts)

    # 4. Score
    hd = hausdorff_distance(cad_verts, v)
    bbox_diag = float(np.linalg.norm(v.max(axis=0) - v.min(axis=0)))
    quality = max(0.0, min(1.0, 1.0 - (hd["hausdorff"] / max(bbox_diag, 1e-12)) * 5))

    return {
        "cad_vertices": cad_verts,
        "cad_faces": cad_faces,
        "segments": segments,
        "quality": quality,
        "n_segments": len(segments),
    }


def trace_reconstruct_with_template_search(vertices, faces):
    """Auto-detect the best template and reconstruct.

    Tries the top-3 templates and picks the one with the best quality.
    Also tries a template-free reconstruction as a baseline.

    Returns same dict as trace_reconstruct, plus:
        template_name — name of the best template (or None)
    """
    from .object_templates import match_template

    v = np.asarray(vertices, dtype=np.float64)
    f = np.asarray(faces)

    candidates = match_template(v, f, top_k=3)
    best_result = None
    best_quality = -1
    best_tpl_name = None

    # Try template-free first
    result = trace_reconstruct(v, f, template=None)
    if result["quality"] > best_quality:
        best_quality = result["quality"]
        best_result = result
        best_tpl_name = None

    # Try top templates
    for tpl, score in candidates:
        try:
            result = trace_reconstruct(v, f, template=tpl)
            if result["quality"] > best_quality:
                best_quality = result["quality"]
                best_result = result
                best_tpl_name = tpl.name
        except Exception:
            continue

    best_result["template_name"] = best_tpl_name
    return best_result


def trace_reconstruct_file(input_path, output_path, template_name=None):
    """File-level API: read STL, trace-reconstruct, write STL.

    Args:
        input_path:    path to input STL
        output_path:   path to output STL
        template_name: optional template name override

    Returns:
        dict with quality, n_segments, template_name
    """
    from .object_templates import get_template

    vertices, faces = read_binary_stl(input_path)

    template = get_template(template_name) if template_name else None

    if template is not None:
        result = trace_reconstruct(vertices, faces, template=template)
        result["template_name"] = template_name
    else:
        result = trace_reconstruct_with_template_search(vertices, faces)

    write_binary_stl(output_path, result["cad_vertices"], result["cad_faces"])

    return {
        "quality": result["quality"],
        "n_segments": result["n_segments"],
        "template_name": result.get("template_name"),
    }
