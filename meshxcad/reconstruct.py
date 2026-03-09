"""Mesh-to-CAD reconstruction: create a parametric CAD representation from
just a triangle mesh, with no pre-existing CAD model required.

Pipeline:
1. classify_mesh        — Determine the best parametric shape type
2. fit_sphere / fit_cylinder / fit_cone / fit_box  — Primitive fitting
3. reconstruct_revolve  — Fit revolve profile to mesh
4. reconstruct_extrude  — Fit extrude cross-section to mesh
5. reconstruct_sweep    — Fit sweep path + profile to mesh
6. reconstruct_freeform — Clean mesh approximation via adaptive slicing
7. reconstruct_cad      — Full auto pipeline: classify → fit → generate
8. mesh_to_cad_file     — File-level API (STL in → STL out)
"""

import math
import numpy as np
from scipy.spatial import KDTree

from .stl_io import read_binary_stl, write_binary_stl
from .general_align import hausdorff_distance
from .revolve_align import (
    detect_revolve_axis,
    extract_radial_profile,
    adaptive_revolve,
)
from .extrude_align import (
    extract_cross_section,
    fit_primitive_to_cross_section,
    adaptive_extrude,
    detect_sweep_candidate,
    fit_sweep_to_mesh,
    adaptive_sweep_extrude,
)
from .objects.builder import revolve_profile


# ---------------------------------------------------------------------------
# Primitive fitting
# ---------------------------------------------------------------------------

def fit_sphere(vertices):
    """Fit a sphere to a point cloud via least-squares.

    Args:
        vertices: (N, 3) array

    Returns:
        dict with center (3,), radius, residual (mean radial error)
    """
    v = np.asarray(vertices, dtype=np.float64)
    center = v.mean(axis=0)
    radii = np.linalg.norm(v - center, axis=1)
    radius = float(np.median(radii))

    # Refine center via iterative least-squares (3 iterations)
    for _ in range(3):
        diffs = v - center
        dists = np.linalg.norm(diffs, axis=1, keepdims=True)
        dists = np.maximum(dists, 1e-12)
        surface_pts = center + diffs / dists * radius
        center = v.mean(axis=0) + (v - surface_pts).mean(axis=0)
        radii = np.linalg.norm(v - center, axis=1)
        radius = float(np.median(radii))

    residual = float(np.mean(np.abs(radii - radius)))
    return {"center": center, "radius": radius, "residual": residual}


def fit_cylinder(vertices):
    """Fit a cylinder to a point cloud (axis via PCA, radius via median).

    Args:
        vertices: (N, 3) array

    Returns:
        dict with center (3,), axis (3,), radius, height, residual
    """
    v = np.asarray(vertices, dtype=np.float64)
    center = v.mean(axis=0)
    centered = v - center

    # PCA for axis
    cov = centered.T @ centered / len(v)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    eigvecs = eigvecs[:, order]
    axis = eigvecs[:, 0]
    if axis[2] < 0:
        axis = -axis

    # Project onto axis for height
    proj = centered @ axis
    height = float(proj.max() - proj.min())

    # Radial distance perpendicular to axis
    radial = centered - np.outer(proj, axis)
    radii = np.linalg.norm(radial, axis=1)
    radius = float(np.median(radii))
    residual = float(np.mean(np.abs(radii - radius)))

    # Axis base point (bottom of cylinder)
    base = center + axis * float(proj.min())

    return {
        "center": base + axis * height / 2,
        "axis": axis,
        "radius": radius,
        "height": height,
        "base": base,
        "residual": residual,
    }


def fit_cone(vertices):
    """Fit a cone to a point cloud.

    Assumes the cone axis aligns with the principal axis. Fits a linear
    relationship between height and radius.

    Args:
        vertices: (N, 3) array

    Returns:
        dict with apex, axis, half_angle_deg, height, base_radius, residual
    """
    v = np.asarray(vertices, dtype=np.float64)
    center = v.mean(axis=0)
    centered = v - center

    cov = centered.T @ centered / len(v)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    eigvecs = eigvecs[:, order]
    axis = eigvecs[:, 0]
    if axis[2] < 0:
        axis = -axis

    proj = centered @ axis
    radial = centered - np.outer(proj, axis)
    radii = np.linalg.norm(radial, axis=1)

    # Fit linear: radius = a * t + b where t = projection along axis
    t = proj
    A = np.column_stack([t, np.ones(len(t))])
    result = np.linalg.lstsq(A, radii, rcond=None)
    slope, intercept = result[0]

    predicted = slope * t + intercept
    residual = float(np.mean(np.abs(radii - predicted)))

    height = float(proj.max() - proj.min())
    base_radius = float(slope * proj.min() + intercept)
    top_radius = float(slope * proj.max() + intercept)

    # Apex is where radius would be zero
    if abs(slope) > 1e-8:
        t_apex = -intercept / slope
        apex = center + axis * t_apex
    else:
        apex = center + axis * float(proj.max())

    half_angle = math.degrees(math.atan2(abs(base_radius - top_radius), height))

    return {
        "apex": apex,
        "axis": axis,
        "half_angle_deg": half_angle,
        "height": height,
        "base_radius": max(base_radius, top_radius),
        "top_radius": min(base_radius, top_radius),
        "residual": residual,
    }


def fit_box(vertices):
    """Fit an oriented bounding box to a point cloud.

    Uses PCA axes as box orientation.

    Args:
        vertices: (N, 3) array

    Returns:
        dict with center (3,), axes (3,3), dimensions (3,), residual
    """
    v = np.asarray(vertices, dtype=np.float64)
    center = v.mean(axis=0)
    centered = v - center

    cov = centered.T @ centered / len(v)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    eigvecs = eigvecs[:, order]

    # Project onto PCA axes
    coords = centered @ eigvecs
    mins = coords.min(axis=0)
    maxs = coords.max(axis=0)
    dimensions = maxs - mins

    # Refined center in PCA space
    box_center_pca = (mins + maxs) / 2
    refined_center = center + eigvecs @ box_center_pca

    # Residual: mean distance from point to nearest box face
    half = dimensions / 2
    local = coords - box_center_pca
    face_dists = half - np.abs(local)  # distance to each face pair
    min_face_dist = face_dists.min(axis=1)  # nearest face
    # Points outside box have negative min_face_dist
    residual = float(np.mean(np.abs(min_face_dist)))

    return {
        "center": refined_center,
        "axes": eigvecs,
        "dimensions": dimensions,
        "residual": residual,
    }


# ---------------------------------------------------------------------------
# Shape classification
# ---------------------------------------------------------------------------

def classify_mesh(vertices, faces):
    """Classify a mesh into the best parametric shape type.

    Tests sphere, cylinder, cone, box, revolve, extrude, sweep — returns
    the type with the lowest normalized residual.

    Args:
        vertices: (N, 3) array
        faces:    (M, 3) array

    Returns:
        dict with:
            shape_type  — one of "sphere", "cylinder", "cone", "box",
                          "revolve", "extrude", "sweep", "freeform"
            confidence  — 0-1 confidence score
            details     — shape-specific fit parameters
    """
    v = np.asarray(vertices, dtype=np.float64)
    f = np.asarray(faces)

    # Bounding box diagonal for normalization
    bbox_diag = float(np.linalg.norm(v.max(axis=0) - v.min(axis=0)))
    if bbox_diag < 1e-12:
        return {"shape_type": "freeform", "confidence": 0.0, "details": {}}

    candidates = []

    # Sphere
    sphere = fit_sphere(v)
    sphere_score = sphere["residual"] / bbox_diag
    candidates.append(("sphere", sphere_score, sphere))

    # Cylinder
    cyl = fit_cylinder(v)
    cyl_score = cyl["residual"] / bbox_diag
    candidates.append(("cylinder", cyl_score, cyl))

    # Cone
    cone = fit_cone(v)
    cone_score = cone["residual"] / bbox_diag
    candidates.append(("cone", cone_score, cone))

    # Box
    box = fit_box(v)
    box_score = box["residual"] / bbox_diag
    candidates.append(("box", box_score, box))

    # Revolve: check circularity
    revolve_info = detect_revolve_axis(v)
    revolve_score = 1.0 - revolve_info["circularity"]  # lower is better
    candidates.append(("revolve", revolve_score, revolve_info))

    # Sweep: check elongation + consistency
    sweep_info = detect_sweep_candidate(v, f)
    if sweep_info["is_sweep"]:
        sweep_score = 1.0 - sweep_info["confidence"]
    else:
        sweep_score = 1.0
    candidates.append(("sweep", sweep_score, sweep_info))

    # Extrude: check cross-section consistency along principal axis
    extrude_score = _score_extrude_fit(v, f, bbox_diag)
    candidates.append(("extrude", extrude_score, {}))

    # Sort by score (lower = better fit)
    candidates.sort(key=lambda c: c[1])
    best_type, best_score, best_details = candidates[0]

    # Confidence: inverse of normalized score, clamped
    confidence = max(0.0, min(1.0, 1.0 - best_score))

    # Fall back to freeform if nothing fits well
    if confidence < 0.15:
        best_type = "freeform"

    return {
        "shape_type": best_type,
        "confidence": round(confidence, 3),
        "details": best_details,
        "all_scores": [(c[0], round(c[1], 4)) for c in candidates],
    }


def _score_extrude_fit(vertices, faces, bbox_diag):
    """Score how well a mesh fits a constant-profile extrusion along Z."""
    v = np.asarray(vertices, dtype=np.float64)
    z_min, z_max = float(v[:, 2].min()), float(v[:, 2].max())
    z_range = z_max - z_min
    if z_range < 1e-6:
        return 1.0

    # Sample cross-sections at 5 heights and compare areas
    n_slices = 5
    z_vals = np.linspace(z_min + 0.15 * z_range, z_max - 0.15 * z_range,
                         n_slices)
    areas = []
    for z in z_vals:
        cs = extract_cross_section(v, faces, z)
        if len(cs) >= 3:
            from scipy.spatial import ConvexHull
            try:
                hull = ConvexHull(cs)
                areas.append(hull.volume)  # 2D convex hull "volume" is area
            except Exception:
                pass

    if len(areas) < 3:
        return 1.0

    areas = np.array(areas)
    mean_a = areas.mean()
    if mean_a < 1e-12:
        return 1.0
    consistency = float(areas.std() / mean_a)
    return min(1.0, consistency)


# ---------------------------------------------------------------------------
# Reconstruction methods
# ---------------------------------------------------------------------------

def reconstruct_revolve(vertices, faces, n_slices=40, n_angular=48):
    """Reconstruct a revolve-type CAD mesh from an arbitrary mesh.

    Extracts the radial profile and generates a clean revolve mesh.

    Args:
        vertices:  (N, 3) input mesh
        faces:     (M, 3) input mesh faces (used for quality measurement)
        n_slices:  profile sampling density
        n_angular: angular subdivisions

    Returns:
        dict with cad_vertices, cad_faces, profile, quality
    """
    v = np.asarray(vertices, dtype=np.float64)
    profile = extract_radial_profile(v, n_slices=n_slices)
    if len(profile) < 2:
        return {"cad_vertices": v, "cad_faces": faces,
                "profile": profile, "quality": 0.0}

    cad_v, cad_f = adaptive_revolve(v, n_slices=n_slices,
                                     n_angular=n_angular)
    quality = _measure_quality(cad_v, v)
    return {
        "cad_vertices": cad_v,
        "cad_faces": cad_f,
        "profile": profile,
        "quality": quality,
    }


def reconstruct_extrude(vertices, faces, n_slices=10, n_profile=32):
    """Reconstruct an extrude-type CAD mesh from an arbitrary mesh.

    Samples cross-sections along Z and lofts between them.

    Args:
        vertices: (N, 3) input mesh
        faces:    (M, 3) input mesh faces
        n_slices: number of Z-slices
        n_profile: points per cross-section ring

    Returns:
        dict with cad_vertices, cad_faces, z_range, quality
    """
    v = np.asarray(vertices, dtype=np.float64)
    f = np.asarray(faces)
    z_min, z_max = float(v[:, 2].min()), float(v[:, 2].max())

    cad_v, cad_f = adaptive_extrude(v, f, z_min, z_max,
                                     n_slices=n_slices, n_profile=n_profile)
    quality = _measure_quality(cad_v, v)
    return {
        "cad_vertices": cad_v,
        "cad_faces": cad_f,
        "z_range": (z_min, z_max),
        "quality": quality,
    }


def reconstruct_sweep(vertices, faces, n_path=20, n_profile=24):
    """Reconstruct a sweep-type CAD mesh from an arbitrary mesh.

    Extracts skeleton path and cross-section profile, then sweeps.

    Args:
        vertices: (N, 3) input mesh
        faces:    (M, 3) input mesh faces
        n_path:   path sampling density
        n_profile: profile sampling density

    Returns:
        dict with cad_vertices, cad_faces, path, profile, quality
    """
    v = np.asarray(vertices, dtype=np.float64)
    f = np.asarray(faces)

    cad_v, cad_f, info = adaptive_sweep_extrude(
        v, f, n_path=n_path, n_profile=n_profile)
    quality = _measure_quality(cad_v, v)
    return {
        "cad_vertices": cad_v,
        "cad_faces": cad_f,
        "path": info.get("path"),
        "profile": info.get("profile"),
        "quality": quality,
    }


def reconstruct_sphere(vertices, n_lat=30, n_lon=30):
    """Reconstruct a sphere CAD mesh from a mesh.

    Args:
        vertices: (N, 3) input mesh

    Returns:
        dict with cad_vertices, cad_faces, center, radius, quality
    """
    from .synthetic import make_sphere_mesh

    params = fit_sphere(vertices)
    cad_v, cad_f = make_sphere_mesh(radius=params["radius"],
                                     lat_divs=n_lat, lon_divs=n_lon)
    cad_v = cad_v + params["center"]
    quality = _measure_quality(cad_v, vertices)
    return {
        "cad_vertices": cad_v,
        "cad_faces": cad_f,
        "center": params["center"],
        "radius": params["radius"],
        "quality": quality,
    }


def reconstruct_cylinder(vertices, n_angular=48, n_height=10):
    """Reconstruct a cylinder CAD mesh from a mesh.

    Args:
        vertices: (N, 3) input mesh

    Returns:
        dict with cad_vertices, cad_faces, params, quality
    """
    from .synthetic import make_cylinder_mesh

    params = fit_cylinder(vertices)
    cad_v, cad_f = make_cylinder_mesh(
        radius=params["radius"],
        height=params["height"],
        radial_divs=n_angular,
        height_divs=n_height,
    )

    # Align cylinder: default is along Z centered at origin.
    # Rotate to match fitted axis and translate to fitted center.
    cad_v = _align_to_axis(cad_v, params["axis"], params["base"],
                            params["height"])
    quality = _measure_quality(cad_v, vertices)
    return {
        "cad_vertices": cad_v,
        "cad_faces": cad_f,
        "params": params,
        "quality": quality,
    }


def reconstruct_cone(vertices, n_angular=48, n_height=10):
    """Reconstruct a cone CAD mesh from a mesh.

    Generates a tapered revolve from base_radius to top_radius.

    Args:
        vertices: (N, 3) input mesh

    Returns:
        dict with cad_vertices, cad_faces, params, quality
    """
    params = fit_cone(vertices)
    height = params["height"]
    base_r = params["base_radius"]
    top_r = max(params["top_radius"], 0.01)  # avoid zero

    profile = []
    for i in range(n_height + 1):
        t = i / n_height
        r = base_r + (top_r - base_r) * t
        z = t * height
        profile.append((max(r, 0.01), z))

    cad_v, cad_f = revolve_profile(profile, n_angular)
    cad_v = _align_to_axis(cad_v, params["axis"], params["apex"],
                            height)
    quality = _measure_quality(cad_v, vertices)
    return {
        "cad_vertices": cad_v,
        "cad_faces": cad_f,
        "params": params,
        "quality": quality,
    }


def reconstruct_box(vertices, n_subdiv=4):
    """Reconstruct a box CAD mesh from a mesh.

    Args:
        vertices: (N, 3) input mesh
        n_subdiv: subdivisions per face edge

    Returns:
        dict with cad_vertices, cad_faces, params, quality
    """
    params = fit_box(vertices)
    center = params["center"]
    axes = params["axes"]
    dims = params["dimensions"]
    half = dims / 2

    cad_v, cad_f = _make_box_mesh(half, n_subdiv)
    # Rotate into PCA frame and translate
    cad_v = cad_v @ axes.T + center
    quality = _measure_quality(cad_v, vertices)
    return {
        "cad_vertices": cad_v,
        "cad_faces": cad_f,
        "params": params,
        "quality": quality,
    }


def reconstruct_freeform(vertices, faces, n_slices=20, n_profile=32):
    """Reconstruct a freeform CAD mesh via adaptive Z-slicing.

    This is a fallback for meshes that don't fit any parametric type well.
    It samples cross-sections along the principal axis and lofts between them.

    Args:
        vertices: (N, 3) input mesh
        faces:    (M, 3) input mesh faces
        n_slices: number of slices along principal axis
        n_profile: points per cross-section ring

    Returns:
        dict with cad_vertices, cad_faces, quality
    """
    v = np.asarray(vertices, dtype=np.float64)
    f = np.asarray(faces)

    # Use PCA to find principal axis
    center = v.mean(axis=0)
    centered = v - center
    cov = centered.T @ centered / len(v)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    eigvecs = eigvecs[:, order]
    axis = eigvecs[:, 0]

    # Rotate so principal axis aligns with Z
    rotated_v = _rotate_to_z(v, axis, center)
    z_min = float(rotated_v[:, 2].min())
    z_max = float(rotated_v[:, 2].max())

    # Build rotated faces (same topology)
    cad_v_rot, cad_f = adaptive_extrude(rotated_v, f, z_min, z_max,
                                         n_slices=n_slices,
                                         n_profile=n_profile)

    # Rotate back to original orientation
    cad_v = _unrotate_from_z(cad_v_rot, axis, center)
    quality = _measure_quality(cad_v, v)
    return {
        "cad_vertices": cad_v,
        "cad_faces": cad_f,
        "quality": quality,
    }


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------

def reconstruct_cad(vertices, faces, shape_type=None):
    """Full auto pipeline: classify the mesh and reconstruct a clean CAD mesh.

    Args:
        vertices:   (N, 3) input mesh vertices
        faces:      (M, 3) input mesh faces
        shape_type: override auto-classification (one of "sphere", "cylinder",
                    "cone", "box", "revolve", "extrude", "sweep", "freeform")

    Returns:
        dict with:
            shape_type     — detected or specified shape type
            cad_vertices   — (P, 3) reconstructed CAD mesh vertices
            cad_faces      — (Q, 3) reconstructed CAD mesh faces
            quality        — 0-1 quality score (based on mean distance)
            classification — full classification result (if auto)
            params         — shape-specific parameters
    """
    v = np.asarray(vertices, dtype=np.float64)
    f = np.asarray(faces)

    # Classify if not specified
    if shape_type is None:
        classification = classify_mesh(v, f)
        shape_type = classification["shape_type"]
    else:
        classification = None

    # Dispatch to the appropriate reconstructor
    dispatchers = {
        "sphere": lambda: reconstruct_sphere(v),
        "cylinder": lambda: reconstruct_cylinder(v),
        "cone": lambda: reconstruct_cone(v),
        "box": lambda: reconstruct_box(v),
        "revolve": lambda: reconstruct_revolve(v, f),
        "extrude": lambda: reconstruct_extrude(v, f),
        "sweep": lambda: reconstruct_sweep(v, f),
        "freeform": lambda: reconstruct_freeform(v, f),
    }

    result = dispatchers[shape_type]()

    return {
        "shape_type": shape_type,
        "cad_vertices": result["cad_vertices"],
        "cad_faces": result["cad_faces"],
        "quality": result.get("quality", 0.0),
        "classification": classification,
        "params": {k: v for k, v in result.items()
                   if k not in ("cad_vertices", "cad_faces", "quality")},
    }


def mesh_to_cad_file(input_path, output_path, shape_type=None):
    """File-level API: read a mesh file, reconstruct CAD, write result.

    Args:
        input_path:  path to input STL file
        output_path: path to output STL file
        shape_type:  optional override for shape classification

    Returns:
        dict with shape_type, quality, n_vertices, n_faces
    """
    vertices, faces = read_binary_stl(input_path)
    result = reconstruct_cad(vertices, faces, shape_type=shape_type)
    write_binary_stl(output_path, result["cad_vertices"], result["cad_faces"])
    return {
        "shape_type": result["shape_type"],
        "quality": result["quality"],
        "n_vertices": len(result["cad_vertices"]),
        "n_faces": len(result["cad_faces"]),
        "classification": result["classification"],
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _measure_quality(cad_vertices, mesh_vertices):
    """Measure reconstruction quality as 1 - normalized mean distance."""
    cad_v = np.asarray(cad_vertices, dtype=np.float64)
    mesh_v = np.asarray(mesh_vertices, dtype=np.float64)
    if len(cad_v) == 0 or len(mesh_v) == 0:
        return 0.0

    hd = hausdorff_distance(cad_v, mesh_v)
    bbox_diag = float(np.linalg.norm(mesh_v.max(axis=0) - mesh_v.min(axis=0)))
    if bbox_diag < 1e-12:
        return 0.0

    normalized = hd["mean_symmetric"] / bbox_diag
    return round(max(0.0, min(1.0, 1.0 - normalized * 5)), 3)


def _align_to_axis(vertices, target_axis, base_point, height):
    """Rotate vertices from Z-up orientation to target axis orientation
    and translate so base is at base_point."""
    v = np.asarray(vertices, dtype=np.float64).copy()

    # Current axis is Z
    z_axis = np.array([0.0, 0.0, 1.0])
    target = target_axis / (np.linalg.norm(target_axis) + 1e-12)

    # Rotation from Z to target axis
    rot = _rotation_between(z_axis, target)
    v = v @ rot.T

    # Translate so that bottom (min along target axis) is at base_point
    proj = v @ target
    v = v - target * proj.min() + base_point

    return v


def _rotation_between(a, b):
    """Compute rotation matrix that rotates unit vector a to unit vector b."""
    a = a / (np.linalg.norm(a) + 1e-12)
    b = b / (np.linalg.norm(b) + 1e-12)

    cross = np.cross(a, b)
    dot = float(np.dot(a, b))

    if dot > 0.9999:
        return np.eye(3)
    if dot < -0.9999:
        # 180 degree rotation — pick an arbitrary perpendicular axis
        perp = np.array([1.0, 0.0, 0.0])
        if abs(np.dot(a, perp)) > 0.9:
            perp = np.array([0.0, 1.0, 0.0])
        perp = perp - np.dot(perp, a) * a
        perp = perp / (np.linalg.norm(perp) + 1e-12)
        # Rodrigues for 180 degrees around perp
        return 2.0 * np.outer(perp, perp) - np.eye(3)

    # Rodrigues' formula
    K = np.array([[0, -cross[2], cross[1]],
                  [cross[2], 0, -cross[0]],
                  [-cross[1], cross[0], 0]])
    return np.eye(3) + K + K @ K / (1 + dot)


def _rotate_to_z(vertices, axis, center):
    """Rotate vertices so that 'axis' direction maps to Z."""
    rot = _rotation_between(axis, np.array([0.0, 0.0, 1.0]))
    return (vertices - center) @ rot.T + center


def _unrotate_from_z(vertices, axis, center):
    """Inverse of _rotate_to_z."""
    rot = _rotation_between(np.array([0.0, 0.0, 1.0]), axis)
    return (vertices - center) @ rot.T + center


def _make_box_mesh(half_dims, n_subdiv=4):
    """Generate a subdivided box mesh centered at origin.

    Args:
        half_dims: (3,) half-dimensions along each axis
        n_subdiv: subdivisions per edge

    Returns:
        vertices (P, 3), faces (Q, 3)
    """
    hx, hy, hz = half_dims
    verts = []
    faces = []

    # Six faces, each a subdivided quad
    face_defs = [
        # (normal_axis, sign, u_axis, v_axis)
        (2, +1, 0, 1),  # +Z face
        (2, -1, 0, 1),  # -Z face
        (1, +1, 0, 2),  # +Y face
        (1, -1, 0, 2),  # -Y face
        (0, +1, 1, 2),  # +X face
        (0, -1, 1, 2),  # -X face
    ]

    for norm_ax, sign, u_ax, v_ax in face_defs:
        offset = len(verts)
        n = n_subdiv + 1
        for iu in range(n):
            for iv in range(n):
                pt = [0.0, 0.0, 0.0]
                pt[norm_ax] = sign * half_dims[norm_ax]
                pt[u_ax] = -half_dims[u_ax] + 2 * half_dims[u_ax] * iu / n_subdiv
                pt[v_ax] = -half_dims[v_ax] + 2 * half_dims[v_ax] * iv / n_subdiv
                verts.append(pt)

        for iu in range(n_subdiv):
            for iv in range(n_subdiv):
                p00 = offset + iu * n + iv
                p01 = offset + iu * n + iv + 1
                p10 = offset + (iu + 1) * n + iv
                p11 = offset + (iu + 1) * n + iv + 1
                if sign > 0:
                    faces.append([p00, p01, p10])
                    faces.append([p01, p11, p10])
                else:
                    faces.append([p00, p10, p01])
                    faces.append([p01, p10, p11])

    return np.array(verts, dtype=np.float64), np.array(faces)
