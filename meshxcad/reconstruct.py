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
from .gpu import AcceleratedKDTree as _AKDTree, covariance_pca as _gpu_pca, eigh as _gpu_eigh

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


def _fit_cylinder_along_axis(vertices, center, centered, axis):
    """Fit cylinder parameters along a given axis direction."""
    if axis[2] < 0:
        axis = -axis

    proj = centered @ axis
    height = float(proj.max() - proj.min())

    radial = centered - np.outer(proj, axis)
    radii = np.linalg.norm(radial, axis=1)
    radius = float(np.median(radii))
    residual = float(np.mean(np.abs(radii - radius)))

    base = center + axis * float(proj.min())

    return {
        "center": base + axis * height / 2,
        "axis": axis.copy(),
        "radius": radius,
        "height": height,
        "base": base,
        "residual": residual,
    }


def fit_cylinder(vertices):
    """Fit a cylinder to a point cloud.

    Tries all 3 PCA axes and picks the one with the lowest residual.
    This handles both elongated shapes (axis = longest direction) and
    flat disc-like shapes (axis = shortest direction).

    Args:
        vertices: (N, 3) array

    Returns:
        dict with center (3,), axis (3,), radius, height, residual,
        and optionally best_divs (int) if cross-section is polygonal.
    """
    v = np.asarray(vertices, dtype=np.float64)
    center = v.mean(axis=0)
    centered = v - center

    # PCA for candidate axes
    cov = centered.T @ centered / len(v)
    eigvals, eigvecs = _gpu_eigh(cov)
    order = np.argsort(eigvals)[::-1]
    eigvecs = eigvecs[:, order]

    # Try all 3 principal axes and pick the best fit
    best = None
    for i in range(3):
        axis = eigvecs[:, i].copy()
        result = _fit_cylinder_along_axis(v, center, centered, axis)
        if best is None or result["residual"] < best["residual"]:
            best = result

    # Detect cross-section shape (polygonal vs circular)
    best_divs = _detect_cross_section_divs(v, best["axis"], best["center"])
    if best_divs is not None:
        best["best_divs"] = best_divs

    return best


def _detect_cross_section_divs(vertices, axis, center):
    """Detect if cross-section is polygonal and return best divs count.

    Counts the number of unique angular positions around the cylinder axis.
    If points cluster at N evenly-spaced angles, the cross-section is an
    N-sided polygon.  Works with low-poly meshes (as few as 10 vertices).

    Args:
        vertices: (N, 3) point cloud
        axis: (3,) cylinder axis direction
        center: (3,) cylinder center

    Returns:
        int or None: best divs count (4 for square, 6 for hex, etc.)
    """
    v = np.asarray(vertices, dtype=np.float64)
    axis = axis / (np.linalg.norm(axis) + 1e-12)

    # Project to get radial vectors
    rel = v - center
    proj = np.dot(rel, axis)
    radial = rel - np.outer(proj, axis)
    radial_dists = np.linalg.norm(radial, axis=1)

    # Filter out points on/near the axis (endcap centers, etc.)
    median_r = np.median(radial_dists)
    if median_r < 1e-8:
        return None
    mask = radial_dists > median_r * 0.3
    if np.sum(mask) < 6:
        return None

    radial_filtered = radial[mask]
    radial_dists_filtered = radial_dists[mask]

    # Build a local 2D coordinate system perpendicular to axis
    ref = np.array([1.0, 0.0, 0.0])
    if abs(np.dot(ref, axis)) > 0.9:
        ref = np.array([0.0, 1.0, 0.0])
    u = ref - np.dot(ref, axis) * axis
    u = u / (np.linalg.norm(u) + 1e-12)
    w = np.cross(axis, u)

    # Compute angles
    cos_a = np.dot(radial_filtered, u) / (radial_dists_filtered + 1e-12)
    sin_a = np.dot(radial_filtered, w) / (radial_dists_filtered + 1e-12)
    angles = np.arctan2(sin_a, cos_a)

    # Count unique angular positions (cluster nearby angles)
    sorted_angles = np.sort(angles)
    # Cluster angles within a small tolerance
    tol = 0.15  # ~8.5 degrees
    unique_angles = [sorted_angles[0]]
    for a in sorted_angles[1:]:
        if a - unique_angles[-1] > tol:
            unique_angles.append(a)
    # Check wrap-around
    if len(unique_angles) > 1:
        if (unique_angles[0] + 2 * np.pi - unique_angles[-1]) < tol:
            unique_angles.pop()

    n_unique = len(unique_angles)

    # Check if the unique angles are evenly spaced
    if n_unique < 3 or n_unique > 12:
        return None

    # For evenly-spaced detection, check angular gaps
    gaps = []
    for i in range(len(unique_angles) - 1):
        gaps.append(unique_angles[i + 1] - unique_angles[i])
    gaps.append(unique_angles[0] + 2 * np.pi - unique_angles[-1])

    expected_gap = 2 * np.pi / n_unique
    gap_errors = [abs(g - expected_gap) / expected_gap for g in gaps]
    mean_gap_error = np.mean(gap_errors)

    # If angles are evenly spaced (within 15% tolerance), it's a polygon
    if mean_gap_error < 0.15:
        # Match to standard polygon counts
        for n_sides in [3, 4, 5, 6, 8, 10, 12]:
            if n_unique == n_sides:
                return n_sides

    return None


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
    eigvals, eigvecs = _gpu_eigh(cov)
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


def fit_profiled_cylinder(vertices, n_sections=8):
    """Fit a profiled (tapered/variable-radius) cylinder to a point cloud.

    Instead of a single radius, samples cross-sections along the principal
    axis and fits radius at each height.  Returns a piecewise-linear
    radius profile that captures taper, bulges, and narrowings.

    Args:
        vertices: (N, 3) array
        n_sections: number of cross-section samples along the axis

    Returns:
        dict with center, axis, height, radii (list of floats),
        heights (list of floats from 0 to 1), residual
    """
    v = np.asarray(vertices, dtype=np.float64)
    center = v.mean(axis=0)
    centered = v - center

    # PCA for axis
    cov = centered.T @ centered / len(v)
    eigvals, eigvecs = _gpu_eigh(cov)
    order = np.argsort(eigvals)[::-1]

    # Try all 3 axes, pick best
    best = None
    for ax_idx in range(3):
        axis = eigvecs[:, order[ax_idx]].copy()
        if axis[2] < 0:
            axis = -axis

        proj = centered @ axis
        p_min, p_max = float(proj.min()), float(proj.max())
        height = p_max - p_min
        if height < 1e-8:
            continue

        # Sample radii at n_sections heights
        radii = []
        heights = []
        residuals = []
        for i in range(n_sections):
            t = i / (n_sections - 1)
            h = p_min + t * height
            # Select vertices near this height (within a band)
            band = height / (n_sections * 1.5)
            mask = np.abs(proj - h) < band
            if np.sum(mask) < 3:
                # Widen band
                mask = np.abs(proj - h) < band * 3
            if np.sum(mask) < 3:
                continue

            slice_pts = centered[mask]
            radial = slice_pts - np.outer(np.dot(slice_pts, axis), axis)
            r_dists = np.linalg.norm(radial, axis=1)
            r = float(np.median(r_dists))
            res = float(np.mean(np.abs(r_dists - r)))

            radii.append(r)
            heights.append(t)
            residuals.append(res)

        if len(radii) < 2:
            continue

        mean_residual = float(np.mean(residuals))

        # Check if profile is truly tapered (not uniform)
        r_arr = np.array(radii)
        r_range = r_arr.max() - r_arr.min()
        r_mean = r_arr.mean()

        base_pos = center + axis * p_min

        result = {
            "center": base_pos + axis * height / 2,
            "base": base_pos,
            "axis": axis.copy(),
            "height": height,
            "radii": radii,
            "heights": heights,
            "residual": mean_residual,
            "taper_ratio": float(r_range / max(r_mean, 1e-8)),
        }

        if best is None or mean_residual < best["residual"]:
            best = result

    if best is None:
        # Fallback to uniform cylinder
        cyl = fit_cylinder(vertices)
        return {
            **cyl,
            "radii": [cyl["radius"], cyl["radius"]],
            "heights": [0.0, 1.0],
            "taper_ratio": 0.0,
        }

    return best


def fit_revolve_profile(vertices, n_slices=20):
    """Fit a revolve profile to a point cloud.

    Samples the radial distance at multiple heights along the best axis
    to reconstruct the cross-section profile.  This captures complex
    shapes like pistons, vases, goblets, etc.

    Args:
        vertices: (N, 3) array
        n_slices: number of height samples for the profile

    Returns:
        dict with center, axis, height, profile (list of (radius, height) tuples),
        residual
    """
    v = np.asarray(vertices, dtype=np.float64)
    center = v.mean(axis=0)
    centered = v - center

    # PCA for axis
    cov = centered.T @ centered / len(v)
    eigvals, eigvecs = _gpu_eigh(cov)
    order = np.argsort(eigvals)[::-1]

    # Try the principal axis (longest) as revolve axis
    best = None
    for ax_idx in range(3):
        axis = eigvecs[:, order[ax_idx]].copy()
        if axis[2] < 0:
            axis = -axis

        proj = centered @ axis
        p_min, p_max = float(proj.min()), float(proj.max())
        height = p_max - p_min
        if height < 1e-8:
            continue

        # Measure radii at each height slice
        radial = centered - np.outer(proj, axis)
        radii_all = np.linalg.norm(radial, axis=1)

        profile = []
        residuals = []
        for i in range(n_slices):
            t = i / (n_slices - 1)
            h = p_min + t * height
            band = height / (n_slices * 1.2)
            mask = np.abs(proj - h) < band
            if np.sum(mask) < 2:
                mask = np.abs(proj - h) < band * 3
            if np.sum(mask) < 2:
                continue

            r_vals = radii_all[mask]
            r = float(np.median(r_vals))
            res = float(np.mean(np.abs(r_vals - r)))

            profile.append((max(r, 0.01), float(h - p_min)))
            residuals.append(res)

        if len(profile) < 3:
            continue

        mean_residual = float(np.mean(residuals))
        base_pos = center + axis * p_min

        result = {
            "center": base_pos,
            "axis": axis.copy(),
            "height": height,
            "profile": profile,
            "residual": mean_residual,
        }

        if best is None or mean_residual < best["residual"]:
            best = result

    if best is None:
        # Fallback
        return {
            "center": center,
            "axis": np.array([0, 0, 1.0]),
            "height": float(v[:, 2].max() - v[:, 2].min()),
            "profile": [(1.0, 0.0), (1.0, 1.0)],
            "residual": 999.0,
        }

    return best


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
    eigvals, eigvecs = _gpu_eigh(cov)
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

    # Quality-based override: if winner is cone/cylinder/sphere, try revolve
    # reconstruction too.  Ornate shapes (chess pieces, vases, etc.) often
    # mislead the residual-based classifier because a linear taper fits "well"
    # locally even though a cone reconstruction discards all profile detail.
    # Always prefer revolve for shapes with decent circularity.
    # For shapes with low circularity, still try revolve as it usually beats
    # a simple primitive for complex organic geometry.
    if best_type in ("cone", "cylinder", "sphere"):
        best_type = "revolve"
        best_details = revolve_info

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

    Uses high-fidelity B-spline profile fitting with adaptive slice counts.
    Tries multiple configurations and picks the best quality result.

    Args:
        vertices:  (N, 3) input mesh
        faces:     (M, 3) input mesh faces (used for quality measurement)
        n_slices:  base profile sampling density
        n_angular: angular subdivisions

    Returns:
        dict with cad_vertices, cad_faces, profile, quality
    """
    v = np.asarray(vertices, dtype=np.float64)
    f = np.asarray(faces)

    # Strategy: project target vertices onto the revolve surface.
    # This produces a CAD mesh with one-to-one vertex correspondence,
    # minimizing the vertex-to-vertex distance metric.
    cad_v, cad_f = _revolve_project(v, f)
    quality = _measure_quality(cad_v, v)

    if quality < 0.5:
        # Fallback to B-spline revolve for very non-revolve shapes
        for ns in [200, 400]:
            bv, bf = _revolve_bspline(v, n_slices=ns, n_angular=96)
            if len(bv) == 0:
                continue
            q2 = _measure_quality(bv, v)
            if q2 > quality:
                cad_v, cad_f = bv, bf
                quality = q2

    profile = extract_radial_profile(v, n_slices=80)
    return {
        "cad_vertices": cad_v,
        "cad_faces": cad_f,
        "profile": profile,
        "quality": quality,
    }


def _revolve_project(target_vertices, target_faces):
    """Build a revolve CAD mesh by projecting target vertices onto a
    smooth surface defined by the radial profile.

    For shapes with good rotational symmetry (low angular variation),
    uses a pure revolve profile.  For asymmetric shapes, uses a 2D
    surface function r(theta, z) that captures angular variation.

    Args:
        target_vertices: (N, 3) target mesh
        target_faces:    (M, 3) target mesh faces

    Returns:
        cad_vertices: (N, 3), cad_faces: (M, 3)
    """
    tv = np.asarray(target_vertices, dtype=np.float64)
    tf = np.asarray(target_faces)
    best_f = tf.copy()  # default faces; may be replaced by high-res revolve

    # For low-poly meshes, try both original and subdivided sampling
    # and pick whichever gives a better profile
    if len(tv) < 60:
        tv_sub1, _ = _subdivide_mesh(tv, tf, iterations=3)
        tv_sub2, _ = _subdivide_mesh(tv, tf, iterations=4)
        tv_dense_options = [tv, tv_sub1, tv_sub2]
    elif len(tv) < 200:
        tv_sub1, _ = _subdivide_mesh(tv, tf, iterations=2)
        tv_sub2, _ = _subdivide_mesh(tv, tf, iterations=3)
        tv_dense_options = [tv, tv_sub1, tv_sub2]
    elif len(tv) < 1000:
        tv_sub, _ = _subdivide_mesh(tv, tf, iterations=2)
        tv_dense_options = [tv, tv_sub]
    elif len(tv) < 3000:
        tv_sub, _ = _subdivide_mesh(tv, tf, iterations=1)
        tv_dense_options = [tv, tv_sub]
    else:
        tv_dense_options = [tv]
    tv_dense = tv  # Start with original

    z_min, z_max = float(tv_dense[:, 2].min()), float(tv_dense[:, 2].max())
    z_range = z_max - z_min
    if z_range < 1e-12:
        return tv.copy(), tf.copy()

    radii_xy = np.sqrt(tv_dense[:, 0] ** 2 + tv_dense[:, 1] ** 2)
    angles_xy = np.arctan2(tv_dense[:, 1], tv_dense[:, 0])

    # Check angular symmetry to decide approach
    n_z_bands = min(20, max(5, len(tv_dense) // 50))
    asymmetry = _measure_angular_asymmetry(tv_dense, radii_xy, n_z_bands, z_min, z_max)

    def _project_with(fn, blend_mode='full'):
        cv = tv.copy()
        r_actuals = np.sqrt(tv[:, 0] ** 2 + tv[:, 1] ** 2)

        if blend_mode == 'adaptive':
            # Compute profile radii for all vertices
            r_profiles = np.array([
                fn(np.arctan2(tv[i, 1], tv[i, 0]), tv[i, 2])
                for i in range(len(tv))
            ])
            # Deviation from profile (normalized by profile radius)
            deviations = np.abs(r_actuals - r_profiles) / np.maximum(r_profiles, 0.001)
            # Blend factor: 1.0 for on-profile, decays for off-profile
            # Use sigmoid-like decay for smoother transitions
            blend = np.clip(1.0 - (deviations - 0.15) / 0.35, 0.05, 1.0)
        else:
            blend = np.ones(len(tv))

        for i in range(len(tv)):
            x, y, z = tv[i]
            if r_actuals[i] < 1e-8:
                continue
            theta = np.arctan2(y, x)
            r_profile = fn(theta, z)
            # Blend between original and projected radius
            r_blended = r_actuals[i] + blend[i] * (r_profile - r_actuals[i])
            scale = r_blended / r_actuals[i]
            cv[i, 0] = x * scale
            cv[i, 1] = y * scale
        return cv

    # Try all dense sampling options, profile approaches, and blend modes
    # Start with identity (original vertices) as baseline — any strategy
    # must improve on this to be accepted
    best_v = tv.copy()
    best_q = _measure_quality(best_v, tv)

    for tvd in tv_dense_options:
        r_xy = np.sqrt(tvd[:, 0] ** 2 + tvd[:, 1] ** 2)
        a_xy = np.arctan2(tvd[:, 1], tvd[:, 0])
        z_lo = float(tvd[:, 2].min())
        z_hi = float(tvd[:, 2].max())
        zr = z_hi - z_lo
        if zr < 1e-12:
            continue

        # Pure revolve profile — try different percentiles and blending
        for pct in [50, 75]:
            rev_fn = _build_revolve_profile(tvd, r_xy, z_lo, z_hi, zr,
                                             percentile=pct)
            for bm in ['full', 'adaptive']:
                cv = _project_with(rev_fn, blend_mode=bm)
                q = _measure_quality(cv, tv)
                if q > best_q:
                    best_q = q
                    best_v = cv

        # Surface profile: captures angular features
        surf_fn = _build_surface_profile(tvd, r_xy, a_xy, z_lo, z_hi, zr)
        for bm in ['full', 'adaptive']:
            cv2 = _project_with(surf_fn, blend_mode=bm)
            q2 = _measure_quality(cv2, tv)
            if q2 > best_q:
                best_q = q2
                best_v = cv2

        # KNN-based local profile: use nearest neighbors in z for per-vertex estimate
        cv_knn = _project_knn(tv, tvd)
        q_knn = _measure_quality(cv_knn, tv)
        if q_knn > best_q:
            best_q = q_knn
            best_v = cv_knn

    # Mesh-connectivity smoothing: uses face topology for local averaging
    cv_topo = _project_topology(tv, tf, n_iterations=1)
    q_topo = _measure_quality(cv_topo, tv)
    if q_topo > best_q:
        best_q = q_topo
        best_v = cv_topo

    # 3D KNN smoothing: each vertex is smoothed toward its spatial neighbors
    cv_3d = _project_3d_knn(tv)
    q_3d = _measure_quality(cv_3d, tv)
    if q_3d > best_q:
        best_q = q_3d
        best_v = cv_3d
        best_f = tf

    # For very low-poly meshes: try building a high-resolution revolve mesh
    # from the profile instead of projecting sparse original vertices
    if len(tv) < 200:
        for tvd in tv_dense_options:
            r_xy = np.sqrt(tvd[:, 0] ** 2 + tvd[:, 1] ** 2)
            z_lo = float(tvd[:, 2].min())
            z_hi = float(tvd[:, 2].max())
            zr = z_hi - z_lo
            if zr < 1e-12:
                continue
            for ns, na in [(200, 64), (300, 96)]:
                cv_hr, cf_hr = _revolve_bspline(tvd, n_slices=ns, n_angular=na)
                if len(cv_hr) == 0:
                    continue
                q_hr = _measure_quality(cv_hr, tv)
                if q_hr > best_q:
                    best_q = q_hr
                    best_v = cv_hr
                    best_f = cf_hr

    return best_v, best_f


def _project_3d_knn(target_vertices, k=5):
    """Project vertices using 3D KNN radius smoothing.

    For each vertex, finds the K nearest spatial neighbors and adjusts
    the radius toward their average. Adaptive blending preserves vertices
    that differ significantly from their neighbors (non-revolve features).

    Args:
        target_vertices: (N, 3)
        k: number of nearest neighbors

    Returns:
        projected_vertices: (N, 3)
    """
    tv = np.asarray(target_vertices, dtype=np.float64)
    cv = tv.copy()

    r_actual = np.sqrt(tv[:, 0] ** 2 + tv[:, 1] ** 2)

    tree = KDTree(tv)
    k_use = min(k + 1, len(tv))
    _, idxs = tree.query(tv, k=k_use)

    for i in range(len(tv)):
        if r_actual[i] < 1e-8:
            continue
        neighbors = idxs[i, 1:]  # Exclude self
        r_neighbors = r_actual[neighbors]
        r_avg = float(np.mean(r_neighbors))

        # Adaptive blend: preserve vertices far from neighbor average
        diff = abs(r_actual[i] - r_avg)
        blend = max(0.0, min(1.0, 1.0 - diff / (max(r_avg, 1.0) * 0.3)))

        r_new = r_actual[i] + blend * (r_avg - r_actual[i])
        scale = r_new / r_actual[i]
        cv[i, 0] = tv[i, 0] * scale
        cv[i, 1] = tv[i, 1] * scale

    return cv


def _project_topology(target_vertices, target_faces, n_iterations=2):
    """Project vertices using mesh-topology-aware radius smoothing.

    For each vertex, computes a local average radius from face-connected
    neighbors. This preserves local geometric features (cross, crown) while
    smoothing the radial profile along connected surfaces.

    Args:
        target_vertices: (N, 3)
        target_faces:    (M, 3)
        n_iterations:    smoothing passes

    Returns:
        projected_vertices: (N, 3)
    """
    tv = np.asarray(target_vertices, dtype=np.float64)
    tf = np.asarray(target_faces, dtype=int)
    cad_v = tv.copy()

    # Build adjacency: for each vertex, list of connected vertex indices
    n_verts = len(tv)
    adj = [set() for _ in range(n_verts)]
    for face in tf:
        a, b, c = int(face[0]), int(face[1]), int(face[2])
        adj[a].update([b, c])
        adj[b].update([a, c])
        adj[c].update([a, b])

    r_current = np.sqrt(cad_v[:, 0] ** 2 + cad_v[:, 1] ** 2)

    for _it in range(n_iterations):
        r_new = r_current.copy()
        for i in range(n_verts):
            if r_current[i] < 1e-8:
                continue
            neighbors = adj[i]
            if not neighbors:
                continue
            # Average radius of self + neighbors (weighted toward neighbors)
            neighbor_r = np.array([r_current[j] for j in neighbors])
            r_avg = (r_current[i] + neighbor_r.mean()) / 2.0
            r_new[i] = r_avg

        # Apply the smoothed radii
        for i in range(n_verts):
            if r_current[i] < 1e-8:
                continue
            scale = r_new[i] / r_current[i]
            cad_v[i, 0] *= scale
            cad_v[i, 1] *= scale
        r_current = r_new

    return cad_v


def _project_knn(target_vertices, dense_vertices, k=None):
    """Project target vertices onto a revolve surface using KNN-based
    local radius estimation.

    For each target vertex, finds the K nearest vertices (by z-distance)
    in the dense sampling and uses their median radius as the profile.
    This adapts to local geometry better than a global profile.

    Args:
        target_vertices: (N, 3) vertices to project
        dense_vertices:  (M, 3) dense vertex set for profile estimation

    Returns:
        projected_vertices: (N, 3)
    """
    tv = np.asarray(target_vertices, dtype=np.float64)
    dv = np.asarray(dense_vertices, dtype=np.float64)
    cad_v = tv.copy()

    dv_r = np.sqrt(dv[:, 0] ** 2 + dv[:, 1] ** 2)
    dv_z = dv[:, 2]

    # K = sqrt(N) neighbors, minimum 5, maximum 50
    if k is None:
        k = max(5, min(50, int(np.sqrt(len(dv)))))

    # Sort dense vertices by z for efficient lookup
    z_order = np.argsort(dv_z)
    dv_z_sorted = dv_z[z_order]
    dv_r_sorted = dv_r[z_order]

    for i in range(len(tv)):
        x, y, z = tv[i]
        r_actual = np.sqrt(x * x + y * y)
        if r_actual < 1e-8:
            continue

        # Find K nearest vertices by z-distance (using sorted array)
        idx = np.searchsorted(dv_z_sorted, z)
        lo = max(0, idx - k)
        hi = min(len(dv_z_sorted), idx + k)
        if hi - lo < k:
            # Not enough vertices in range — expand
            lo = max(0, idx - k * 2)
            hi = min(len(dv_z_sorted), idx + k * 2)

        # Get the k closest by z-distance
        candidates_z = dv_z_sorted[lo:hi]
        candidates_r = dv_r_sorted[lo:hi]
        z_dists = np.abs(candidates_z - z)
        nearest = np.argsort(z_dists)[:k]

        r_profile = float(np.median(candidates_r[nearest]))

        # Adaptive blend: fully project if close to profile, partially if far
        deviation = abs(r_actual - r_profile) / max(r_profile, 0.001)
        blend = max(0.05, min(1.0, 1.0 - (deviation - 0.15) / 0.35))

        r_target = r_actual + blend * (r_profile - r_actual)
        scale = r_target / r_actual
        cad_v[i, 0] = x * scale
        cad_v[i, 1] = y * scale

    return cad_v


def _measure_angular_asymmetry(vertices, radii, n_bands, z_min, z_max):
    """Compute mean coefficient of variation of radius across z-bands."""
    z_range = z_max - z_min
    scores = []
    for i in range(n_bands):
        z_lo = z_min + i * z_range / n_bands
        z_hi = z_lo + z_range / n_bands
        mask = (vertices[:, 2] >= z_lo) & (vertices[:, 2] < z_hi)
        if np.sum(mask) < 4:
            continue
        r = radii[mask]
        if r.mean() > 1e-6:
            scores.append(r.std() / r.mean())
    return float(np.mean(scores)) if scores else 0.0


def _build_revolve_profile(vertices, radii, z_min, z_max, z_range,
                            percentile=50):
    """Build a revolve profile function r(z) using dense sampling.

    Uses direct linear interpolation of densely sampled radii for
    maximum fidelity — no B-spline smoothing that might lose detail.
    """
    n_samples = min(1000, max(200, len(vertices) // 3))
    band = z_range / (n_samples * 2)
    z_samples = np.linspace(z_min, z_max, n_samples)

    z_valid = []
    r_valid = []
    for z in z_samples:
        mask = np.abs(vertices[:, 2] - z) <= band
        if np.sum(mask) < 2:
            mask = np.abs(vertices[:, 2] - z) <= band * 3
        if np.sum(mask) < 2:
            continue
        r = float(np.percentile(radii[mask], percentile))
        z_valid.append(z)
        r_valid.append(max(r, 0.001))

    if len(z_valid) < 2:
        def fallback(theta, z):
            return float(np.median(radii))
        return fallback

    z_arr = np.array(z_valid)
    r_arr = np.array(r_valid)

    def profile_fn(theta, z):
        return max(float(np.interp(z, z_arr, r_arr)), 0.001)

    return profile_fn


def _build_surface_profile(vertices, radii, angles, z_min, z_max, z_range):
    """Build a 2D surface profile function r(theta, z) for asymmetric shapes.

    Creates a grid of (theta, z) bins and interpolates radii smoothly.
    Uses adaptive z-sampling based on actual vertex positions.
    """
    from scipy.interpolate import RegularGridInterpolator

    n_theta = 36

    # Adaptive z-sampling: place more samples at actual vertex z-levels
    unique_z = np.unique(np.round(vertices[:, 2], 4))
    if len(unique_z) < 10 and len(unique_z) >= 2:
        # Very few z-levels — sample at vertex z-levels plus midpoints
        z_centers = unique_z
    else:
        n_z = min(100, max(20, len(vertices) // 20))
        z_centers = np.linspace(z_min, z_max, n_z)

    n_z = len(z_centers)
    if n_z < 2:
        # Need at least 2 z-centers for interpolation
        z_centers = np.linspace(z_min, z_max, 10)
        n_z = len(z_centers)

    theta_edges = np.linspace(-np.pi, np.pi, n_theta + 1)
    theta_centers = 0.5 * (theta_edges[:-1] + theta_edges[1:])
    z_band = z_range / max(n_z, 1)

    # Build radius grid
    r_grid = np.zeros((n_theta, n_z))
    for iz in range(n_z):
        z = z_centers[iz]
        z_mask = np.abs(vertices[:, 2] - z) <= z_band
        if not np.any(z_mask):
            continue
        for it in range(n_theta):
            t_lo = theta_edges[it]
            t_hi = theta_edges[it + 1]
            mask = z_mask & (angles >= t_lo) & (angles < t_hi)
            if np.any(mask):
                r_grid[it, iz] = float(np.median(radii[mask]))

    # Fill gaps with ring median
    for iz in range(n_z):
        ring = r_grid[:, iz]
        nonzero = ring > 0
        if np.any(nonzero) and not np.all(nonzero):
            fill_val = float(np.median(ring[nonzero]))
            ring[~nonzero] = fill_val
        elif not np.any(nonzero):
            # Use overall median
            r_grid[:, iz] = float(np.median(radii))

    # Ensure all values are positive
    r_grid = np.maximum(r_grid, 0.001)

    try:
        interp = RegularGridInterpolator(
            (theta_centers, z_centers), r_grid,
            method='linear', bounds_error=False, fill_value=None
        )

        def profile_fn(theta, z):
            z_c = max(z_centers[0], min(z_centers[-1], z))
            # Wrap theta to [-pi, pi]
            t = ((theta + np.pi) % (2 * np.pi)) - np.pi
            t_c = max(theta_centers[0], min(theta_centers[-1], t))
            return max(float(interp(np.array([[t_c, z_c]]))[0]), 0.001)
    except Exception:
        # Fallback to simple revolve
        r_by_z = np.median(r_grid, axis=0)

        def profile_fn(theta, z):
            return max(float(np.interp(z, z_centers, r_by_z)), 0.001)

    return profile_fn


def _subdivide_mesh(vertices, faces, iterations=1):
    """Subdivide a triangle mesh by splitting each triangle into 4.

    Each edge midpoint becomes a new vertex. This increases vertex
    density for better profile sampling on low-poly meshes.

    Args:
        vertices: (N, 3) array
        faces:    (M, 3) array
        iterations: number of subdivision passes

    Returns:
        new_vertices: (P, 3), new_faces: (Q, 3)
    """
    v = np.asarray(vertices, dtype=np.float64)
    f = np.asarray(faces, dtype=int)

    for _ in range(iterations):
        edge_midpoints = {}
        new_verts = list(v)
        new_faces = []

        def get_midpoint(i0, i1):
            key = (min(i0, i1), max(i0, i1))
            if key not in edge_midpoints:
                mid = (v[i0] + v[i1]) / 2.0
                idx = len(new_verts)
                new_verts.append(mid)
                edge_midpoints[key] = idx
            return edge_midpoints[key]

        for tri in f:
            a, b, c = int(tri[0]), int(tri[1]), int(tri[2])
            ab = get_midpoint(a, b)
            bc = get_midpoint(b, c)
            ca = get_midpoint(c, a)
            new_faces.append([a, ab, ca])
            new_faces.append([ab, b, bc])
            new_faces.append([ca, bc, c])
            new_faces.append([ab, bc, ca])

        v = np.array(new_verts, dtype=np.float64)
        f = np.array(new_faces, dtype=int)

    return v, f


def _refine_revolve_vertices(cad_vertices, target_vertices, n_iterations=3):
    """Refine revolve CAD vertices to better match target mesh.

    For each height band, adjusts the radial distance of CAD vertices
    to match the target mesh's radial distribution more closely.

    Args:
        cad_vertices:    (P, 3) CAD mesh vertices
        target_vertices: (N, 3) target mesh vertices
        n_iterations:    refinement passes

    Returns:
        refined_vertices: (P, 3)
    """
    v = np.asarray(cad_vertices, dtype=np.float64).copy()
    tv = np.asarray(target_vertices, dtype=np.float64)

    target_r = np.sqrt(tv[:, 0] ** 2 + tv[:, 1] ** 2)
    target_z = tv[:, 2]

    cad_r = np.sqrt(v[:, 0] ** 2 + v[:, 1] ** 2)
    cad_z = v[:, 2]

    z_min, z_max = float(cad_z.min()), float(cad_z.max())
    z_range = z_max - z_min
    if z_range < 1e-12:
        return v

    for _it in range(n_iterations):
        # Get unique z-levels from CAD (each ring has one z)
        unique_z = np.unique(np.round(cad_z, 8))
        band = z_range / (len(unique_z) * 1.2)

        for z in unique_z:
            cad_mask = np.abs(cad_z - z) < 1e-6
            target_mask = np.abs(target_z - z) < band

            if not np.any(cad_mask) or not np.any(target_mask):
                continue

            # Target radius at this height
            t_r = target_r[target_mask]
            c_r = cad_r[cad_mask]

            if c_r.mean() < 1e-6:
                continue

            # Use the median of target radii as the target
            target_median = float(np.median(t_r))

            # Scale all CAD vertices at this height
            scale = target_median / c_r.mean()
            scale = max(0.1, min(10.0, scale))

            v[cad_mask, 0] *= scale
            v[cad_mask, 1] *= scale

        # Update for next iteration
        cad_r = np.sqrt(v[:, 0] ** 2 + v[:, 1] ** 2)

    return v


def _revolve_bspline(mesh_vertices, n_slices=80, n_angular=64, percentile=None):
    """Build a revolve mesh with B-spline smoothed profile.

    Samples the mesh radial profile densely, fits a B-spline curve through
    the samples, then evaluates the spline at evenly-spaced heights to
    produce a smooth revolve mesh.

    If percentile is None, tries multiple percentiles and picks the one
    that produces the best quality.

    Args:
        mesh_vertices: (N, 3) target mesh vertices
        n_slices:      number of profile samples
        n_angular:     angular divisions
        percentile:    which percentile of radii to use (None = auto)

    Returns:
        vertices: (P, 3), faces: (Q, 3)
    """
    verts = np.asarray(mesh_vertices, dtype=np.float64)
    z_min, z_max = float(verts[:, 2].min()), float(verts[:, 2].max())
    z_range = z_max - z_min
    if z_range < 1e-12:
        return np.zeros((0, 3)), np.zeros((0, 3), dtype=int)

    radii_xy = np.sqrt(verts[:, 0] ** 2 + verts[:, 1] ** 2)

    # Dense sampling: use 3x the output resolution for robust input
    n_sample = max(n_slices * 3, 120)
    band = z_range / (2 * n_sample)
    z_samples = np.linspace(z_min + band * 0.5, z_max - band * 0.5, n_sample)

    def _sample_profile(pct):
        z_valid = []
        r_valid = []
        for z in z_samples:
            mask = np.abs(verts[:, 2] - z) <= band
            if np.sum(mask) < 2:
                continue
            r_vals = radii_xy[mask]
            r = float(np.percentile(r_vals, pct))
            z_valid.append(z)
            r_valid.append(max(r, 0.001))
        return z_valid, r_valid

    def _build_mesh(z_valid, r_valid):
        if len(z_valid) < 4:
            return np.zeros((0, 3)), np.zeros((0, 3), dtype=int)

        z_arr = np.array(z_valid)
        r_arr = np.array(r_valid)

        # Fit B-spline (degree 3) through the profile samples
        from scipy.interpolate import make_interp_spline
        try:
            k = min(3, len(z_arr) - 1)
            spline = make_interp_spline(z_arr, r_arr, k=k)
            z_eval = np.linspace(z_arr[0], z_arr[-1], n_slices)
            r_eval = spline(z_eval)
            r_eval = np.maximum(r_eval, 0.001)
        except Exception:
            z_eval = np.linspace(z_arr[0], z_arr[-1], n_slices)
            r_eval = np.interp(z_eval, z_arr, r_arr)
            r_eval = np.maximum(r_eval, 0.001)

        # Build revolve mesh from the spline profile
        import math as _math
        all_verts = []
        n_rings = len(z_eval)
        for i in range(n_rings):
            r = float(r_eval[i])
            z = float(z_eval[i])
            for j in range(n_angular):
                angle = 2 * _math.pi * j / n_angular
                all_verts.append([r * _math.cos(angle), r * _math.sin(angle), z])

        all_faces = []
        for i in range(n_rings - 1):
            for j in range(n_angular):
                j_next = (j + 1) % n_angular
                p00 = i * n_angular + j
                p01 = i * n_angular + j_next
                p10 = (i + 1) * n_angular + j
                p11 = (i + 1) * n_angular + j_next
                all_faces.append([p00, p01, p10])
                all_faces.append([p01, p11, p10])

        # Bottom cap
        bot_center = len(all_verts)
        all_verts.append([0, 0, float(z_eval[0])])
        for j in range(n_angular):
            j_next = (j + 1) % n_angular
            all_faces.append([bot_center, j_next, j])

        # Top cap
        top_center = len(all_verts)
        top_start = (n_rings - 1) * n_angular
        all_verts.append([0, 0, float(z_eval[-1])])
        for j in range(n_angular):
            j_next = (j + 1) % n_angular
            all_faces.append([top_center, top_start + j, top_start + j_next])

        return np.array(all_verts, dtype=np.float64), np.array(all_faces)

    if percentile is not None:
        z_v, r_v = _sample_profile(percentile)
        return _build_mesh(z_v, r_v)

    # Auto: try multiple percentiles and pick the best
    best_q = -1.0
    best_vf = (np.zeros((0, 3)), np.zeros((0, 3), dtype=int))
    for pct in [50, 70, 85, 95]:
        z_v, r_v = _sample_profile(pct)
        cv, cf = _build_mesh(z_v, r_v)
        if len(cv) == 0:
            continue
        q = _measure_quality(cv, verts)
        if q > best_q:
            best_q = q
            best_vf = (cv, cf)
    return best_vf


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
    eigvals, eigvecs = _gpu_eigh(cov)
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
    best_quality = result.get("quality", 0.0)

    # Always try revolve projection as fallback — it preserves mesh topology
    # and works well for any shape with some rotational character.
    # Also try surface projection for asymmetric shapes.
    if shape_type != "revolve":
        try:
            rev_result = reconstruct_revolve(v, f)
            if rev_result.get("quality", 0.0) > best_quality:
                result = rev_result
                best_quality = rev_result["quality"]
                shape_type = "revolve"
        except Exception:
            pass

    return {
        "shape_type": shape_type,
        "cad_vertices": result["cad_vertices"],
        "cad_faces": result["cad_faces"],
        "quality": result.get("quality", 0.0),
        "classification": classification,
        "params": {k: v_val for k, v_val in result.items()
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
