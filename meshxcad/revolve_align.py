"""Revolve-to-mesh alignment: 12 features for improving how a revolved CAD
profile matches a target mesh.

Revolve-based CAD objects are defined by an (r, z) profile curve rotated
around the Z axis.  Given such a profile and a target mesh, these tools:

1.  extract_radial_profile     — Sample a mesh's radial profile (r vs z).
2.  compare_radial_profiles    — Metric comparison of two (r,z) profiles.
3.  fit_profile_to_mesh        — Least-squares fit of an (r,z) profile to
                                  mesh radial samples.
4.  detect_revolve_axis        — Find the best-fit revolution axis of a mesh.
5.  angular_deviation_map      — Per-angle deviation from circular symmetry.
6.  refine_profile_radii       — Adjust each profile point's radius to the
                                  mesh's median radius at that z.
7.  refine_profile_z_spacing   — Re-distribute profile z-values so that
                                  spacing matches the mesh's feature density.
8.  insert_profile_detail      — Insert new (r,z) points where the mesh has
                                  detail that the current profile misses.
9.  smooth_profile_to_mesh     — Cubic-spline smooth the profile while
                                  minimizing deviation from the mesh.
10. remove_profile_redundancy  — Merge near-colinear points to simplify the
                                  profile without losing accuracy.
11. adaptive_revolve           — Rebuild a full revolve mesh whose per-ring
                                  radius matches the mesh slice-by-slice.
12. suggest_revolve_adjustments — Analyse a CAD profile vs a mesh and return
                                  a ranked list of suggested refinements.
"""

import math
import numpy as np
from scipy.spatial import KDTree
from .gpu import AcceleratedKDTree as _AKDTree
from scipy.interpolate import CubicSpline


# =========================================================================
# Helpers
# =========================================================================

def _radii_at_z(vertices, z, band):
    """Return radial distances of vertices within *band* of height *z*."""
    mask = np.abs(vertices[:, 2] - z) <= band
    pts = vertices[mask]
    if len(pts) == 0:
        return np.array([])
    return np.sqrt(pts[:, 0] ** 2 + pts[:, 1] ** 2)


def _resample_profile(profile, n_out):
    """Resample an (r,z) profile to *n_out* equi-arc-length points."""
    prof = np.asarray(profile, dtype=np.float64)
    if len(prof) < 2:
        return prof
    diffs = np.diff(prof, axis=0)
    seg_lens = np.linalg.norm(diffs, axis=1)
    cum = np.concatenate([[0], np.cumsum(seg_lens)])
    total = cum[-1]
    if total < 1e-12:
        return prof[:n_out]
    targets = np.linspace(0, total, n_out)
    out = np.zeros((n_out, 2))
    for i, t in enumerate(targets):
        idx = np.searchsorted(cum, t, side="right") - 1
        idx = max(0, min(idx, len(prof) - 2))
        seg = seg_lens[idx]
        frac = (t - cum[idx]) / seg if seg > 1e-12 else 0.0
        out[i] = prof[idx] * (1 - frac) + prof[idx + 1] * frac
    return out


# =========================================================================
# 1. extract_radial_profile
# =========================================================================

def extract_radial_profile(vertices, n_slices=40, percentile=50):
    """Sample a mesh's radial profile by computing the radius at many Z heights.

    At each Z height, vertices within a thin band are collected and the
    chosen percentile of their radial distances is recorded.

    Args:
        vertices:   (N, 3) mesh vertices
        n_slices:   number of Z-height samples
        percentile: which percentile of radii to use (50 = median)

    Returns:
        profile: (n_slices, 2) array of (radius, z) pairs, sorted by z.
                 Slices with no nearby vertices are omitted.
    """
    verts = np.asarray(vertices, dtype=np.float64)
    z_min, z_max = float(verts[:, 2].min()), float(verts[:, 2].max())
    z_range = z_max - z_min
    if z_range < 1e-12:
        r = float(np.median(np.sqrt(verts[:, 0] ** 2 + verts[:, 1] ** 2)))
        return np.array([[r, z_min]])

    band = z_range / (2 * n_slices)
    z_values = np.linspace(z_min + band, z_max - band, n_slices)

    profile = []
    for z in z_values:
        radii = _radii_at_z(verts, z, band)
        if len(radii) == 0:
            continue
        r = float(np.percentile(radii, percentile))
        profile.append([r, float(z)])

    if not profile:
        return np.zeros((0, 2))
    return np.array(profile, dtype=np.float64)


# =========================================================================
# 2. compare_radial_profiles
# =========================================================================

def compare_radial_profiles(profile_a, profile_b, n_sample=64):
    """Compare two (r, z) profiles and return similarity metrics.

    Both profiles are resampled to *n_sample* equi-arc-length points, then
    compared point-wise.

    Args:
        profile_a: (K, 2) array of (r, z) — e.g. the CAD profile
        profile_b: (L, 2) array of (r, z) — e.g. the mesh profile

    Returns:
        dict with keys:
            max_r_error  — worst-case radius deviation
            mean_r_error — average radius deviation
            rms_r_error  — root-mean-square radius deviation
            z_overlap    — fraction of z-range that both profiles cover
    """
    a = _resample_profile(np.asarray(profile_a), n_sample)
    b = _resample_profile(np.asarray(profile_b), n_sample)

    # Align by z-range: interpolate both onto shared z grid
    z_min = max(a[:, 1].min(), b[:, 1].min())
    z_max = min(a[:, 1].max(), b[:, 1].max())
    z_span_a = a[:, 1].max() - a[:, 1].min()
    z_span_b = b[:, 1].max() - b[:, 1].min()
    total_span = max(z_span_a, z_span_b)
    z_overlap = max(z_max - z_min, 0) / total_span if total_span > 1e-12 else 0.0

    if z_max <= z_min:
        return {
            "max_r_error": float("inf"),
            "mean_r_error": float("inf"),
            "rms_r_error": float("inf"),
            "z_overlap": 0.0,
        }

    z_grid = np.linspace(z_min, z_max, n_sample)

    # Interpolate r(z) for each profile
    # Sort by z first
    a_sorted = a[np.argsort(a[:, 1])]
    b_sorted = b[np.argsort(b[:, 1])]
    r_a = np.interp(z_grid, a_sorted[:, 1], a_sorted[:, 0])
    r_b = np.interp(z_grid, b_sorted[:, 1], b_sorted[:, 0])

    diffs = np.abs(r_a - r_b)
    return {
        "max_r_error": float(np.max(diffs)),
        "mean_r_error": float(np.mean(diffs)),
        "rms_r_error": float(np.sqrt(np.mean(diffs ** 2))),
        "z_overlap": float(z_overlap),
    }


# =========================================================================
# 3. fit_profile_to_mesh
# =========================================================================

def fit_profile_to_mesh(profile_rz, mesh_vertices, n_slices=40):
    """Least-squares fit of a profile to a mesh by adjusting radii.

    For each profile point's z-value, the mesh's median radius is computed.
    The profile radii are then adjusted toward the mesh radii via a weighted
    combination that preserves smoothness.

    Args:
        profile_rz:    list of (r, z) tuples — the CAD profile
        mesh_vertices: (N, 3)
        n_slices:      resolution for mesh sampling

    Returns:
        fitted_profile: list of (r, z) tuples with adjusted radii
        residuals:      (K,) per-point residual after fit
    """
    prof = np.asarray(profile_rz, dtype=np.float64)
    verts = np.asarray(mesh_vertices, dtype=np.float64)
    z_range = verts[:, 2].max() - verts[:, 2].min()
    band = z_range / (2 * max(n_slices, 1))

    fitted = prof.copy()
    residuals = np.zeros(len(prof))

    for i, (r_cad, z_cad) in enumerate(prof):
        radii = _radii_at_z(verts, z_cad, band)
        if len(radii) == 0:
            residuals[i] = 0.0
            continue
        r_mesh = float(np.median(radii))
        residuals[i] = abs(r_mesh - r_cad)
        fitted[i, 0] = r_mesh

    result = [(max(float(r), 0.01), float(z)) for r, z in fitted]
    return result, residuals


# =========================================================================
# 4. detect_revolve_axis
# =========================================================================

def detect_revolve_axis(vertices):
    """Estimate the best-fit revolution axis for a mesh.

    Uses PCA to find the principal axis of the vertex cloud, then
    returns the axis direction and a point on the axis (centroid).

    For a true body of revolution the longest principal component is
    the revolution axis.

    Args:
        vertices: (N, 3)

    Returns:
        dict with:
            axis_direction — (3,) unit vector (sign chosen so z-component ≥ 0)
            axis_point     — (3,) a point on the axis (centroid)
            circularity    — 0-1 score (ratio of the two smaller eigenvalues;
                             1.0 = perfectly circular cross-section)
    """
    verts = np.asarray(vertices, dtype=np.float64)
    centroid = np.mean(verts, axis=0)
    centered = verts - centroid

    cov = centered.T @ centered / len(verts)
    eigvals, eigvecs = np.linalg.eigh(cov)
    # eigh returns ascending order; largest eigenvalue = revolution axis
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]

    axis = eigvecs[:, 0]
    # Ensure positive z
    if axis[2] < 0:
        axis = -axis

    # Circularity: ratio of the two cross-section eigenvalues
    if eigvals[1] > 1e-12:
        circularity = float(min(eigvals[1], eigvals[2]) /
                            max(eigvals[1], eigvals[2]))
    else:
        circularity = 0.0

    return {
        "axis_direction": axis,
        "axis_point": centroid,
        "circularity": circularity,
    }


# =========================================================================
# 5. angular_deviation_map
# =========================================================================

def angular_deviation_map(vertices, n_angular=36, n_z=20):
    """Compute per-angle deviation from circular symmetry.

    For each (angle_bin, z_bin), computes the mean radius and reports how
    much each angular bin deviates from the ring's mean.

    Args:
        vertices:  (N, 3)
        n_angular: angular bins (0 to 360°)
        n_z:       Z bins

    Returns:
        dict with:
            angles       — (n_angular,) bin centers in radians
            z_values     — (n_z,) bin centers
            deviation    — (n_z, n_angular) signed deviation from ring mean
            max_dev      — scalar, worst-case absolute deviation
            mean_abs_dev — scalar, average absolute deviation
    """
    verts = np.asarray(vertices, dtype=np.float64)
    z_min, z_max = float(verts[:, 2].min()), float(verts[:, 2].max())
    z_edges = np.linspace(z_min, z_max, n_z + 1)
    z_centers = 0.5 * (z_edges[:-1] + z_edges[1:])
    angle_edges = np.linspace(-math.pi, math.pi, n_angular + 1)
    angle_centers = 0.5 * (angle_edges[:-1] + angle_edges[1:])

    radii = np.sqrt(verts[:, 0] ** 2 + verts[:, 1] ** 2)
    angles = np.arctan2(verts[:, 1], verts[:, 0])

    deviation = np.zeros((n_z, n_angular))
    for iz in range(n_z):
        z_mask = (verts[:, 2] >= z_edges[iz]) & (verts[:, 2] < z_edges[iz + 1])
        ring_radii = radii[z_mask]
        ring_angles = angles[z_mask]
        if len(ring_radii) == 0:
            continue
        ring_mean = float(np.mean(ring_radii))
        if ring_mean < 1e-12:
            continue
        for ia in range(n_angular):
            a_mask = ((ring_angles >= angle_edges[ia]) &
                      (ring_angles < angle_edges[ia + 1]))
            if np.any(a_mask):
                deviation[iz, ia] = float(np.mean(ring_radii[a_mask])) - ring_mean

    abs_dev = np.abs(deviation)
    return {
        "angles": angle_centers,
        "z_values": z_centers,
        "deviation": deviation,
        "max_dev": float(np.max(abs_dev)),
        "mean_abs_dev": float(np.mean(abs_dev)),
    }


# =========================================================================
# 6. refine_profile_radii
# =========================================================================

def refine_profile_radii(profile_rz, mesh_vertices, blend=1.0):
    """Adjust each profile point's radius to the mesh's median radius.

    Args:
        profile_rz:    list of (r, z)
        mesh_vertices: (N, 3)
        blend:         0.0 = keep original, 1.0 = fully adopt mesh radius

    Returns:
        refined_profile: list of (r, z)
    """
    prof = np.asarray(profile_rz, dtype=np.float64)
    verts = np.asarray(mesh_vertices, dtype=np.float64)
    z_range = verts[:, 2].max() - verts[:, 2].min()
    band = z_range / (2 * max(len(prof), 1))

    refined = []
    for r_cad, z_cad in prof:
        radii = _radii_at_z(verts, z_cad, band)
        if len(radii) > 0:
            r_mesh = float(np.median(radii))
            r_new = r_cad + blend * (r_mesh - r_cad)
        else:
            r_new = r_cad
        refined.append((max(r_new, 0.01), float(z_cad)))
    return refined


# =========================================================================
# 7. refine_profile_z_spacing
# =========================================================================

def refine_profile_z_spacing(profile_rz, mesh_vertices, n_output=None):
    """Re-distribute profile Z values so spacing concentrates where the
    mesh has more geometric detail (higher curvature).

    The mesh radial profile's second derivative is used as a curvature
    proxy.  Profile points are placed more densely where curvature is high.

    Args:
        profile_rz:    list of (r, z)
        mesh_vertices: (N, 3)
        n_output:      number of output points (default = len(profile_rz))

    Returns:
        respaced_profile: list of (r, z)
    """
    prof = np.asarray(profile_rz, dtype=np.float64)
    if n_output is None:
        n_output = len(prof)
    if n_output < 2:
        return [(float(prof[0, 0]), float(prof[0, 1]))]

    mesh_prof = extract_radial_profile(np.asarray(mesh_vertices), n_slices=80)
    if len(mesh_prof) < 4:
        # Not enough data — keep uniform
        return _uniform_respace(prof, n_output)

    # Sort mesh profile by z
    mp = mesh_prof[np.argsort(mesh_prof[:, 1])]
    z_m, r_m = mp[:, 1], mp[:, 0]

    # Compute curvature proxy: |d²r/dz²|
    if len(z_m) < 3:
        return _uniform_respace(prof, n_output)
    dr = np.gradient(r_m, z_m)
    d2r = np.abs(np.gradient(dr, z_m))
    # Normalise to a density function
    density = d2r + 0.1 * np.max(d2r)  # floor so flat regions get some points
    cum_density = np.concatenate([[0], np.cumsum(density)])
    cum_density /= cum_density[-1]

    # Map uniform [0,1] → z via inverse CDF
    t_uniform = np.linspace(0, 1, n_output)
    z_new = np.interp(t_uniform, cum_density, np.concatenate([[z_m[0]], z_m]))

    # Interpolate radii from original profile at the new z values
    prof_sorted = prof[np.argsort(prof[:, 1])]
    r_new = np.interp(z_new, prof_sorted[:, 1], prof_sorted[:, 0])

    return [(max(float(r), 0.01), float(z)) for r, z in zip(r_new, z_new)]


def _uniform_respace(prof, n_output):
    resampled = _resample_profile(prof, n_output)
    return [(max(float(r), 0.01), float(z)) for r, z in resampled]


# =========================================================================
# 8. insert_profile_detail
# =========================================================================

def insert_profile_detail(profile_rz, mesh_vertices, threshold=0.5):
    """Insert new (r,z) points where the mesh deviates from the profile.

    For each segment of the profile, the mesh is sampled at the segment
    midpoint.  If the mesh radius differs from the linearly interpolated
    profile radius by more than *threshold*, a new point is inserted.

    This process repeats up to 3 passes.

    Args:
        profile_rz:    list of (r, z)
        mesh_vertices: (N, 3)
        threshold:     minimum radius deviation to trigger insertion

    Returns:
        enriched_profile: list of (r, z) with added detail points
    """
    prof = list(map(tuple, profile_rz))
    verts = np.asarray(mesh_vertices, dtype=np.float64)
    z_range = verts[:, 2].max() - verts[:, 2].min()

    for _pass in range(3):
        new_prof = [prof[0]]
        inserted = False
        for i in range(len(prof) - 1):
            r0, z0 = prof[i]
            r1, z1 = prof[i + 1]
            z_mid = (z0 + z1) / 2
            r_interp = (r0 + r1) / 2

            band = max(abs(z1 - z0) / 4, z_range / 200)
            radii = _radii_at_z(verts, z_mid, band)
            if len(radii) > 0:
                r_mesh = float(np.median(radii))
                if abs(r_mesh - r_interp) > threshold:
                    new_prof.append((max(r_mesh, 0.01), z_mid))
                    inserted = True
            new_prof.append(prof[i + 1])

        prof = new_prof
        if not inserted:
            break

    return prof


# =========================================================================
# 9. smooth_profile_to_mesh
# =========================================================================

def smooth_profile_to_mesh(profile_rz, mesh_vertices, smoothing=0.5,
                            n_output=None):
    """Smooth the profile via cubic spline while fitting the mesh.

    Combines the profile points with mesh radial samples and fits a
    smoothing cubic spline.  The *smoothing* parameter (0–1) blends
    between a strict mesh fit (0) and a smooth spline (1).

    Args:
        profile_rz:    list of (r, z)
        mesh_vertices: (N, 3)
        smoothing:     0 = pure mesh fit, 1 = smooth spline through profile
        n_output:      output point count (default = len(profile))

    Returns:
        smoothed_profile: list of (r, z)
    """
    prof = np.asarray(profile_rz, dtype=np.float64)
    if n_output is None:
        n_output = len(prof)

    mesh_prof = extract_radial_profile(np.asarray(mesh_vertices), n_slices=60)
    if len(mesh_prof) < 4:
        # Fall back to just smoothing the profile
        resampled = _resample_profile(prof, n_output)
        return [(max(float(r), 0.01), float(z)) for r, z in resampled]

    # Build combined point set with weights
    mp = mesh_prof[np.argsort(mesh_prof[:, 1])]
    ps = prof[np.argsort(prof[:, 1])]

    # Blend: the final r at each z is (1-smoothing)*mesh_r + smoothing*profile_r
    # Use interpolation on a shared z grid
    z_min = min(ps[0, 1], mp[0, 1])
    z_max = max(ps[-1, 1], mp[-1, 1])
    z_grid = np.linspace(z_min, z_max, n_output)

    r_prof = np.interp(z_grid, ps[:, 1], ps[:, 0])
    r_mesh = np.interp(z_grid, mp[:, 1], mp[:, 0])
    r_blend = smoothing * r_prof + (1 - smoothing) * r_mesh

    # Apply light cubic spline smoothing
    if len(z_grid) >= 4:
        cs = CubicSpline(z_grid, r_blend)
        r_smooth = cs(z_grid)
    else:
        r_smooth = r_blend

    return [(max(float(r), 0.01), float(z)) for r, z in zip(r_smooth, z_grid)]


# =========================================================================
# 10. remove_profile_redundancy
# =========================================================================

def remove_profile_redundancy(profile_rz, angle_threshold_deg=5.0):
    """Remove near-colinear points from a profile.

    If the angle at a point (formed by its predecessor and successor) is
    close to 180° (i.e. the point lies on a straight line), it is removed.

    Args:
        profile_rz:        list of (r, z)
        angle_threshold_deg: remove points where the deviation from a
                             straight line is less than this many degrees

    Returns:
        simplified_profile: list of (r, z)
    """
    prof = [tuple(p) for p in profile_rz]
    if len(prof) <= 3:
        return prof

    threshold = math.radians(angle_threshold_deg)

    changed = True
    while changed:
        changed = False
        new_prof = [prof[0]]
        for i in range(1, len(prof) - 1):
            r0, z0 = prof[i - 1]
            r1, z1 = prof[i]
            r2, z2 = prof[i + 1]
            # Vectors
            v1 = np.array([r1 - r0, z1 - z0])
            v2 = np.array([r2 - r1, z2 - z1])
            len1 = np.linalg.norm(v1)
            len2 = np.linalg.norm(v2)
            if len1 < 1e-12 or len2 < 1e-12:
                # Duplicate point — remove
                changed = True
                continue
            cos_angle = np.dot(v1, v2) / (len1 * len2)
            cos_angle = max(-1.0, min(1.0, cos_angle))
            angle = math.acos(cos_angle)
            # angle ≈ 0 means collinear (same direction); that's redundant
            if angle < threshold:
                # Nearly colinear — skip this point
                changed = True
                continue
            new_prof.append(prof[i])
        new_prof.append(prof[-1])
        prof = new_prof

    return prof


# =========================================================================
# 11. adaptive_revolve
# =========================================================================

def adaptive_revolve(mesh_vertices, n_slices=40, n_angular=48, percentile=50):
    """Rebuild a revolve mesh whose per-ring radius matches the target mesh.

    Instead of using a fixed profile, each ring's radius is set to the
    mesh's radius at that height.

    Args:
        mesh_vertices: (N, 3) target mesh
        n_slices:      number of rings along Z
        n_angular:     angular subdivisions per ring
        percentile:    which percentile of radii to match

    Returns:
        vertices: (P, 3)
        faces:    (Q, 3)
    """
    profile = extract_radial_profile(mesh_vertices, n_slices=n_slices,
                                      percentile=percentile)
    if len(profile) < 2:
        return np.zeros((0, 3)), np.zeros((0, 3), dtype=int)

    # Build rings
    all_verts = []
    n_rings = len(profile)
    for r, z in profile:
        for j in range(n_angular):
            angle = 2 * math.pi * j / n_angular
            all_verts.append([r * math.cos(angle), r * math.sin(angle), z])

    # Side faces
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
    all_verts.append([0, 0, float(profile[0, 1])])
    for j in range(n_angular):
        j_next = (j + 1) % n_angular
        all_faces.append([bot_center, j_next, j])

    # Top cap
    top_center = len(all_verts)
    top_start = (n_rings - 1) * n_angular
    all_verts.append([0, 0, float(profile[-1, 1])])
    for j in range(n_angular):
        j_next = (j + 1) % n_angular
        all_faces.append([top_center, top_start + j, top_start + j_next])

    return np.array(all_verts, dtype=np.float64), np.array(all_faces)


# =========================================================================
# 12. suggest_revolve_adjustments
# =========================================================================

def suggest_revolve_adjustments(profile_rz, mesh_vertices, n_slices=40):
    """Analyse a CAD revolve profile against a mesh and suggest refinements.

    Runs a battery of checks and returns a priority-ordered list of
    concrete suggestions.

    Args:
        profile_rz:    list of (r, z) — the CAD revolve profile
        mesh_vertices: (N, 3)
        n_slices:      sampling resolution

    Returns:
        list of dicts with:
            action   — function name to call
            priority — int (lower = more important)
            params   — suggested kwargs
            reason   — human-readable explanation
    """
    prof = np.asarray(profile_rz, dtype=np.float64)
    verts = np.asarray(mesh_vertices, dtype=np.float64)
    suggestions = []

    # Extract mesh profile
    mesh_prof = extract_radial_profile(verts, n_slices=n_slices)
    if len(mesh_prof) < 3:
        return [{"action": "no_data", "priority": 0,
                 "params": {}, "reason": "Insufficient mesh data for analysis"}]

    # Compare profiles
    metrics = compare_radial_profiles(prof, mesh_prof)

    # 1. Large mean radius error → refine_profile_radii
    if metrics["mean_r_error"] > 0.5:
        suggestions.append({
            "action": "refine_profile_radii",
            "priority": 1,
            "params": {"blend": 1.0},
            "reason": f"Mean radius error {metrics['mean_r_error']:.2f} — "
                      f"profile radii need adjustment",
        })

    # 2. Large max error → insert_profile_detail
    if metrics["max_r_error"] > 2.0:
        suggestions.append({
            "action": "insert_profile_detail",
            "priority": 2,
            "params": {"threshold": metrics["max_r_error"] * 0.3},
            "reason": f"Max radius error {metrics['max_r_error']:.2f} — "
                      f"profile is missing detail in some regions",
        })

    # 3. Angular deviation → detect non-revolve features
    ang_map = angular_deviation_map(verts, n_angular=24, n_z=n_slices)
    if ang_map["mean_abs_dev"] > 0.5:
        suggestions.append({
            "action": "angular_deviation_map",
            "priority": 3,
            "params": {"n_angular": 36, "n_z": n_slices},
            "reason": f"Mean angular deviation {ang_map['mean_abs_dev']:.2f} — "
                      f"mesh is not perfectly axially symmetric",
        })

    # 4. Circularity check → detect_revolve_axis
    axis_info = detect_revolve_axis(verts)
    if axis_info["circularity"] < 0.85:
        suggestions.append({
            "action": "detect_revolve_axis",
            "priority": 4,
            "params": {},
            "reason": f"Circularity {axis_info['circularity']:.2f} — "
                      f"mesh cross-section is not circular, revolve may not be ideal",
        })

    # 5. Z-spacing suboptimal → refine_profile_z_spacing
    # Check if profile point density matches mesh curvature
    if len(prof) > 4:
        ps = prof[np.argsort(prof[:, 1])]
        z_gaps = np.diff(ps[:, 1])
        if len(z_gaps) > 1:
            gap_cv = float(np.std(z_gaps) / (np.mean(z_gaps) + 1e-12))
            if gap_cv < 0.3 and metrics["mean_r_error"] > 0.3:
                # Uniform spacing but there's error — curvature-aware spacing may help
                suggestions.append({
                    "action": "refine_profile_z_spacing",
                    "priority": 5,
                    "params": {"n_output": len(prof)},
                    "reason": "Profile z-spacing is nearly uniform — "
                              "curvature-adaptive spacing would capture detail better",
                })

    # 6. Too many profile points → remove_profile_redundancy
    if len(prof) > 20:
        simplified = remove_profile_redundancy(
            [(float(r), float(z)) for r, z in prof], angle_threshold_deg=3.0
        )
        savings = len(prof) - len(simplified)
        if savings >= 3:
            suggestions.append({
                "action": "remove_profile_redundancy",
                "priority": 6,
                "params": {"angle_threshold_deg": 3.0},
                "reason": f"{savings} near-colinear points can be removed "
                          f"without losing shape accuracy",
            })

    # 7. Smoothing might help → smooth_profile_to_mesh
    if metrics["rms_r_error"] > 0.5 and metrics["mean_r_error"] < 2.0:
        suggestions.append({
            "action": "smooth_profile_to_mesh",
            "priority": 7,
            "params": {"smoothing": 0.3},
            "reason": f"RMS error {metrics['rms_r_error']:.2f} with moderate mean — "
                      f"smoothing would reduce jaggedness",
        })

    # 8. Z overlap mismatch → profile doesn't cover full mesh height
    if metrics["z_overlap"] < 0.9:
        suggestions.append({
            "action": "fit_profile_to_mesh",
            "priority": 8,
            "params": {"n_slices": n_slices},
            "reason": f"Z overlap only {metrics['z_overlap']:.1%} — "
                      f"profile needs to extend to cover full mesh height",
        })

    # 9. Adaptive revolve as last resort for high error
    if metrics["rms_r_error"] > 3.0:
        suggestions.append({
            "action": "adaptive_revolve",
            "priority": 9,
            "params": {"n_slices": n_slices * 2, "n_angular": 48},
            "reason": f"High RMS error ({metrics['rms_r_error']:.2f}) — "
                      f"full adaptive rebuild recommended",
        })

    # 10. Partial blend if error is moderate
    if 0.3 < metrics["mean_r_error"] <= 1.0:
        suggestions.append({
            "action": "refine_profile_radii",
            "priority": 10,
            "params": {"blend": 0.5},
            "reason": f"Moderate error ({metrics['mean_r_error']:.2f}) — "
                      f"partial radius blending (50%) suggested",
        })

    suggestions.sort(key=lambda s: s["priority"])
    return suggestions if suggestions else [
        {"action": "none", "priority": 0, "params": {},
         "reason": "Profile already matches mesh well "
                   f"(mean_r_error={metrics['mean_r_error']:.3f})"}
    ]


# =========================================================================
# Differentiators (moved from adversarial_loop for standard reuse)
# =========================================================================

def radial_profile_rms(cad_v, mesh_v, n_slices=30):
    """RMS of radial profile differences."""
    cad_prof = extract_radial_profile(np.asarray(cad_v), n_slices=n_slices)
    mesh_prof = extract_radial_profile(np.asarray(mesh_v), n_slices=n_slices)
    if len(cad_prof) < 2 or len(mesh_prof) < 2:
        return float("inf")
    z_min = max(cad_prof[:, 1].min(), mesh_prof[:, 1].min())
    z_max = min(cad_prof[:, 1].max(), mesh_prof[:, 1].max())
    if z_max <= z_min:
        return float("inf")
    z_grid = np.linspace(z_min, z_max, n_slices)
    r_c = np.interp(z_grid, cad_prof[np.argsort(cad_prof[:, 1]), 1],
                     cad_prof[np.argsort(cad_prof[:, 1]), 0])
    r_m = np.interp(z_grid, mesh_prof[np.argsort(mesh_prof[:, 1]), 1],
                     mesh_prof[np.argsort(mesh_prof[:, 1]), 0])
    return float(np.sqrt(np.mean((r_c - r_m) ** 2)))


def angular_symmetry_diff(cad_v, mesh_v, n_sectors=12):
    """Compare radial vertex distribution across angular sectors around Z."""
    cad_v, mesh_v = np.asarray(cad_v), np.asarray(mesh_v)
    cad_c = cad_v[:, :2].mean(axis=0)
    mesh_c = mesh_v[:, :2].mean(axis=0)

    cad_angles = np.arctan2(cad_v[:, 1] - cad_c[1], cad_v[:, 0] - cad_c[0])
    mesh_angles = np.arctan2(mesh_v[:, 1] - mesh_c[1], mesh_v[:, 0] - mesh_c[0])

    bins = np.linspace(-np.pi, np.pi, n_sectors + 1)
    cad_hist, _ = np.histogram(cad_angles, bins=bins, density=True)
    mesh_hist, _ = np.histogram(mesh_angles, bins=bins, density=True)

    return float(np.sum(np.abs(cad_hist - mesh_hist)) * (bins[1] - bins[0]))


# =========================================================================
# Fixers (moved from adversarial_loop for standard reuse)
# =========================================================================

def fix_radial_profile(cad_v, cad_f, mesh_v, mesh_f):
    """Radial adjustment per Z-band."""
    v = np.asarray(cad_v, dtype=np.float64).copy()
    mv = np.asarray(mesh_v)
    centroid = np.mean(v, axis=0)
    z_min, z_max = v[:, 2].min(), v[:, 2].max()
    n_bands = 20
    z_edges = np.linspace(z_min, z_max, n_bands + 1)

    for i in range(n_bands):
        cad_mask = (v[:, 2] >= z_edges[i]) & (v[:, 2] < z_edges[i + 1])
        mesh_mask = (mv[:, 2] >= z_edges[i]) & (mv[:, 2] < z_edges[i + 1])
        if not np.any(cad_mask) or not np.any(mesh_mask):
            continue
        cad_r = np.sqrt((v[cad_mask, 0] - centroid[0]) ** 2 +
                         (v[cad_mask, 1] - centroid[1]) ** 2)
        mesh_r = np.sqrt((mv[mesh_mask, 0] - centroid[0]) ** 2 +
                          (mv[mesh_mask, 1] - centroid[1]) ** 2)
        cad_mean = np.mean(cad_r)
        mesh_mean = np.mean(mesh_r)
        if cad_mean > 1e-6:
            s = mesh_mean / cad_mean
            s = max(0.5, min(2.0, s))
            v[cad_mask, 0] = centroid[0] + (v[cad_mask, 0] - centroid[0]) * s
            v[cad_mask, 1] = centroid[1] + (v[cad_mask, 1] - centroid[1]) * s
    return v


def profile_smoothness_diff(cad_v, mesh_v, n_slices=30):
    """Compare profile smoothness (2nd derivative of r(z))."""
    def _profile_smoothness(v, n):
        v = np.asarray(v, dtype=np.float64)
        prof = extract_radial_profile(v, n_slices=n)
        if len(prof) < 3:
            return 0.0
        dr = np.diff(prof[:, 0])
        dz = np.diff(prof[:, 1])
        dz[np.abs(dz) < 1e-8] = 1e-8
        slope = dr / dz
        return float(np.std(np.diff(slope)))

    s_cad = _profile_smoothness(cad_v, n_slices)
    s_mesh = _profile_smoothness(mesh_v, n_slices)
    return float(abs(s_cad - s_mesh) * 10)


def radial_asymmetry_diff(cad_v, mesh_v, n_sectors=16, n_z=10):
    """Compare per-sector radial variation (detects non-axisymmetric features)."""
    def _sector_variation(v, n_sec, n_z_bands):
        v = np.asarray(v, dtype=np.float64)
        z_min, z_max = float(v[:, 2].min()), float(v[:, 2].max())
        if z_max - z_min < 1e-6:
            return np.zeros(n_sec)
        radii = np.sqrt(v[:, 0]**2 + v[:, 1]**2)
        angles = np.arctan2(v[:, 1], v[:, 0])
        sector_edges = np.linspace(-np.pi, np.pi, n_sec + 1)
        sector_means = np.zeros(n_sec)
        for i in range(n_sec):
            mask = (angles >= sector_edges[i]) & (angles < sector_edges[i + 1])
            if np.any(mask):
                sector_means[i] = float(np.mean(radii[mask]))
        total = sector_means.sum()
        return sector_means / max(total, 1e-12)

    sv_cad = _sector_variation(cad_v, n_sectors, n_z)
    sv_mesh = _sector_variation(mesh_v, n_sectors, n_z)
    return float(np.sum(np.abs(sv_cad - sv_mesh)) * 50)


def fix_profile_smoothness(cad_v, cad_f, mesh_v, mesh_f):
    """Smooth the radial profile to match mesh smoothness."""
    v = np.asarray(cad_v, dtype=np.float64).copy()
    mv = np.asarray(mesh_v)
    z_min = min(v[:, 2].min(), mv[:, 2].min())
    z_max = max(v[:, 2].max(), mv[:, 2].max())
    if z_max - z_min < 1e-6:
        return v

    n_z = 20
    z_vals = np.linspace(z_min, z_max, n_z + 2)[1:-1]
    tol = (z_max - z_min) / (n_z * 2)

    for z in z_vals:
        cad_mask = np.abs(v[:, 2] - z) < tol
        mesh_mask = np.abs(mv[:, 2] - z) < tol
        if np.sum(cad_mask) < 2 or np.sum(mesh_mask) < 2:
            continue
        cad_r = np.sqrt(v[cad_mask, 0]**2 + v[cad_mask, 1]**2)
        mesh_r = np.sqrt(mv[mesh_mask, 0]**2 + mv[mesh_mask, 1]**2)
        if cad_r.mean() > 1e-6:
            target_r = np.median(mesh_r)
            s = np.clip(target_r / cad_r.mean(), 0.8, 1.2)
            v[cad_mask, 0] *= s
            v[cad_mask, 1] *= s
    return v


def fix_radial_asymmetry(cad_v, cad_f, mesh_v, mesh_f):
    """Per-sector radial correction to reduce asymmetry."""
    v = np.asarray(cad_v, dtype=np.float64).copy()
    mv = np.asarray(mesh_v)
    c = v.mean(axis=0)
    mc = mv.mean(axis=0)

    cad_angles = np.arctan2(v[:, 1] - c[1], v[:, 0] - c[0])
    mesh_angles = np.arctan2(mv[:, 1] - mc[1], mv[:, 0] - mc[0])

    n_sectors = 16
    sector_edges = np.linspace(-np.pi, np.pi, n_sectors + 1)
    for i in range(n_sectors):
        cad_mask = (cad_angles >= sector_edges[i]) & (cad_angles < sector_edges[i + 1])
        mesh_mask = (mesh_angles >= sector_edges[i]) & (mesh_angles < sector_edges[i + 1])
        if not np.any(cad_mask) or not np.any(mesh_mask):
            continue
        cad_r = np.linalg.norm(v[cad_mask, :2] - c[:2], axis=1)
        mesh_r = np.linalg.norm(mv[mesh_mask, :2] - mc[:2], axis=1)
        if cad_r.mean() > 1e-6:
            s = np.clip(mesh_r.mean() / cad_r.mean(), 0.7, 1.5)
            v[cad_mask, :2] = c[:2] + (v[cad_mask, :2] - c[:2]) * s
    return v


def fix_angular_symmetry(cad_v, cad_f, mesh_v, mesh_f):
    """Per-sector radial + Z correction for angular symmetry."""
    v = np.asarray(cad_v, dtype=np.float64).copy()
    mv = np.asarray(mesh_v)
    c = v.mean(axis=0)
    mc = mv.mean(axis=0)

    # Shift centroid first
    v += (mc - c) * 0.3
    c = v.mean(axis=0)

    cad_angles = np.arctan2(v[:, 1] - c[1], v[:, 0] - c[0])
    mesh_angles = np.arctan2(mv[:, 1] - mc[1], mv[:, 0] - mc[0])

    n_sectors = 16
    sector_edges = np.linspace(-np.pi, np.pi, n_sectors + 1)
    for i in range(n_sectors):
        cad_mask = (cad_angles >= sector_edges[i]) & (cad_angles < sector_edges[i + 1])
        mesh_mask = (mesh_angles >= sector_edges[i]) & (mesh_angles < sector_edges[i + 1])
        if not np.any(cad_mask) or not np.any(mesh_mask):
            continue
        # Radial correction
        cad_r = np.linalg.norm(v[cad_mask, :2] - c[:2], axis=1)
        mesh_r = np.linalg.norm(mv[mesh_mask, :2] - mc[:2], axis=1)
        if cad_r.mean() > 1e-6:
            s = np.clip(mesh_r.mean() / cad_r.mean(), 0.7, 1.5)
            v[cad_mask, :2] = c[:2] + (v[cad_mask, :2] - c[:2]) * s
        # Z correction
        cad_z = v[cad_mask, 2].mean()
        mesh_z = mv[mesh_mask, 2].mean()
        v[cad_mask, 2] += (mesh_z - cad_z) * 0.3
    return v
