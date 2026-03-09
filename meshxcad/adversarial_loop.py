"""Adversarial development loop for CAD-to-mesh alignment.

This module implements a red-team / blue-team loop:
  1. RED:  Generate a CAD mesh and a target mesh that differ.
          Apply all existing alignment methods.
          Then find measurable differences that survive alignment.
  2. BLUE: Add / improve alignment functions that close those gaps.
  3. MEASURE: Render both meshes, compute silhouette-based difference
              metrics from multiple angles, and log results.
  4. Repeat.

Each round produces:
  - A numeric "differentiability score" (higher = more visible difference)
  - A description of the detected artefact
  - A new or improved alignment function that addresses it
  - Before/after metrics proving the fix works
"""

import math
import os
import time
import json
import numpy as np

from meshxcad.synthetic import (
    make_cylinder_mesh, make_sphere_mesh, make_cube_mesh,
    add_cylinder_grooves, add_sphere_dimples,
)
from meshxcad.objects.builder import revolve_profile, combine_meshes, make_torus
from meshxcad.objects.operations import extrude_polygon, make_regular_polygon

from meshxcad.general_align import (
    hausdorff_distance, surface_distance_map, normal_deviation_map,
    laplacian_smooth_toward, project_vertices_to_mesh,
    local_scale_correction, fill_surface_holes,
    _compute_vertex_normals, _compute_face_normals, _face_areas,
    # Differentiators
    surface_area, per_region_area_difference, _face_centroids,
    curvature_histogram_diff, bbox_aspect_diff, vertex_density_diff,
    edge_length_distribution_diff, centroid_drift, median_surface_distance,
    local_roughness_diff, volume_diff, percentile_95_distance,
    face_normal_consistency, multi_scale_distance, shape_diameter_diff,
    moment_of_inertia_diff, distance_histogram_diff,
    # Round 2 differentiators
    convexity_defect_diff, boundary_edge_length_diff,
    principal_curvature_ratio_diff, geodesic_diameter_diff,
    laplacian_spectrum_diff, face_area_variance_diff,
    vertex_normal_divergence, octant_volume_diff,
    edge_angle_distribution_diff, aspect_ratio_diff,
    # Fixers
    fix_silhouette_mismatch, fix_hausdorff_outliers, fix_normal_deviation,
    fix_surface_area, fix_curvature_mismatch, fix_bbox_aspect,
    fix_region_area, fix_vertex_density, fix_edge_length_distribution,
    fix_centroid_drift, fix_worst_angle_silhouette, fix_median_surface_distance,
    fix_local_roughness, fix_volume, fix_percentile_95,
    fix_face_normal_consistency, fix_multi_scale_distance, fix_shape_diameter,
    fix_moment_of_inertia, fix_distance_histogram,
    # Round 2 fixers
    fix_convexity_defect, fix_boundary_edges, fix_principal_curvature,
    fix_geodesic_diameter, fix_laplacian_spectrum, fix_face_area_variance,
    fix_vertex_normal_divergence, fix_octant_volume,
    fix_edge_angle_distribution, fix_aspect_ratio,
)
from meshxcad.revolve_align import (
    extract_radial_profile, refine_profile_radii, smooth_profile_to_mesh,
    adaptive_revolve, insert_profile_detail,
    # Differentiators & Fixers
    radial_profile_rms, angular_symmetry_diff,
    fix_radial_profile, fix_angular_symmetry,
    # Round 2 revolve differentiators & fixers
    profile_smoothness_diff, radial_asymmetry_diff,
    fix_profile_smoothness, fix_radial_asymmetry,
)
from meshxcad.extrude_align import (
    extract_cross_section, compare_cross_sections,
    adaptive_extrude as adaptive_extrude_fn,
    # Differentiators & Fixers
    cross_section_contour_diff, z_profile_area_diff, _convex_hull_area,
    fix_cross_section_contour, fix_z_profile_area,
    # Round 2 extrude/sweep differentiators & fixers
    taper_consistency_diff, sweep_path_deviation,
    profile_circularity_diff, extrude_twist_diff,
    fix_taper_consistency, fix_sweep_path_deviation,
    fix_profile_circularity, fix_extrude_twist,
)

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False


# =========================================================================
# Silhouette rendering & comparison (the "eyes" of the loop)
# =========================================================================

def project_silhouette(vertices, faces, elev_deg, azim_deg):
    """Project a mesh to a 2D silhouette from a given viewpoint.

    Returns a 2D point cloud of projected vertex positions.
    """
    verts = np.asarray(vertices, dtype=np.float64)
    elev = math.radians(elev_deg)
    azim = math.radians(azim_deg)

    # Camera rotation: azimuth around Z, then elevation
    cos_a, sin_a = math.cos(azim), math.sin(azim)
    cos_e, sin_e = math.cos(elev), math.sin(elev)

    Raz = np.array([[cos_a, -sin_a, 0],
                     [sin_a, cos_a, 0],
                     [0, 0, 1]])
    Rel = np.array([[1, 0, 0],
                     [0, cos_e, -sin_e],
                     [0, sin_e, cos_e]])
    R = Rel @ Raz
    rotated = verts @ R.T
    return rotated[:, :2]  # drop depth


def silhouette_difference(verts_a, faces_a, verts_b, faces_b,
                           n_angles=8, resolution=128):
    """Compare two meshes by their silhouettes from multiple angles.

    Rasterises each silhouette onto a grid and computes the pixel-level
    difference (IoU, Dice, pixel error).

    Args:
        verts_a, faces_a: mesh A (e.g. CAD after alignment)
        verts_b, faces_b: mesh B (e.g. target mesh)
        n_angles:         number of viewpoints
        resolution:       grid resolution for rasterisation

    Returns:
        dict:
            mean_pixel_error — average fraction of differing pixels
            max_pixel_error  — worst-case angle
            mean_iou         — average intersection-over-union
            per_angle        — list of per-angle dicts
    """
    angles = [(e, a) for e in [0, 30, 60, 90]
              for a in np.linspace(0, 360, max(n_angles // 4, 2), endpoint=False)]
    angles = angles[:n_angles]

    per_angle = []
    for elev, azim in angles:
        pa = project_silhouette(verts_a, faces_a, elev, azim)
        pb = project_silhouette(verts_b, faces_b, elev, azim)

        # Rasterise both onto the same grid
        all_pts = np.vstack([pa, pb])
        xmin, xmax = all_pts[:, 0].min(), all_pts[:, 0].max()
        ymin, ymax = all_pts[:, 1].min(), all_pts[:, 1].max()
        span = max(xmax - xmin, ymax - ymin, 1e-6)
        margin = span * 0.05

        grid_a = _rasterise(pa, xmin - margin, ymin - margin,
                             span + 2 * margin, resolution)
        grid_b = _rasterise(pb, xmin - margin, ymin - margin,
                             span + 2 * margin, resolution)

        union = np.sum(grid_a | grid_b)
        inter = np.sum(grid_a & grid_b)
        diff = np.sum(grid_a ^ grid_b)
        total = resolution * resolution

        iou = inter / union if union > 0 else 1.0
        pixel_err = diff / total

        per_angle.append({
            "elev": float(elev), "azim": float(azim),
            "iou": float(iou), "pixel_error": float(pixel_err),
        })

    pixel_errors = [p["pixel_error"] for p in per_angle]
    ious = [p["iou"] for p in per_angle]
    return {
        "mean_pixel_error": float(np.mean(pixel_errors)),
        "max_pixel_error": float(np.max(pixel_errors)),
        "mean_iou": float(np.mean(ious)),
        "per_angle": per_angle,
    }


def _rasterise(points_2d, origin_x, origin_y, span, resolution):
    """Rasterise 2D points onto a boolean grid."""
    grid = np.zeros((resolution, resolution), dtype=bool)
    if len(points_2d) == 0:
        return grid
    xi = ((points_2d[:, 0] - origin_x) / span * (resolution - 1)).astype(int)
    yi = ((points_2d[:, 1] - origin_y) / span * (resolution - 1)).astype(int)
    valid = (xi >= 0) & (xi < resolution) & (yi >= 0) & (yi < resolution)
    grid[yi[valid], xi[valid]] = True
    return grid


# =========================================================================
# Alignment pipeline: apply all existing methods
# =========================================================================

def full_alignment_pipeline(cad_v, cad_f, mesh_v, mesh_f):
    """Apply the full battery of alignment corrections.

    Returns the improved CAD vertices (faces unchanged).
    """
    v = np.asarray(cad_v, dtype=np.float64).copy()
    f = np.asarray(cad_f)
    mv = np.asarray(mesh_v, dtype=np.float64)
    mf = np.asarray(mesh_f)

    # 0. Centroid alignment first
    v_centroid = np.mean(v, axis=0)
    m_centroid = np.mean(mv, axis=0)
    v += (m_centroid - v_centroid)

    # 1. Bbox aspect correction
    cad_span = np.ptp(v, axis=0)
    mesh_span = np.ptp(mv, axis=0)
    cad_span[cad_span < 1e-12] = 1e-12
    scales = mesh_span / cad_span
    center = np.mean(v, axis=0)
    v = center + (v - center) * scales

    # 2. Local scale correction
    v, _ = local_scale_correction(v, f, mv, n_regions=8)

    # 3. Project to mesh surface
    v, _ = project_vertices_to_mesh(v, mv, mf)

    # 4. Laplacian smoothing toward target
    v = laplacian_smooth_toward(v, f, mv, iterations=5,
                                 lam=0.3, target_weight=0.4)

    # 5. Another projection pass
    v, _ = project_vertices_to_mesh(v, mv, mf)

    return v


# =========================================================================
# Differentiator battery: ways to spot remaining differences
# =========================================================================

def run_differentiators(cad_v, cad_f, mesh_v, mesh_f):
    """Run all differentiator checks and return scored results.

    Returns list of (score, name, details) sorted by score descending.
    """
    results = []

    # 1. Silhouette difference
    sil = silhouette_difference(cad_v, cad_f, mesh_v, mesh_f,
                                 n_angles=8, resolution=64)
    results.append((sil["mean_pixel_error"] * 100,
                    "silhouette_pixel_error", sil))

    # 2. Hausdorff distance
    hd = hausdorff_distance(cad_v, mesh_v)
    results.append((hd["hausdorff"],
                    "hausdorff_distance", hd))

    # 3. Surface distance
    sd = surface_distance_map(cad_v, cad_f, mesh_v)
    results.append((sd["mean_dist"],
                    "mean_surface_distance", sd))

    # 4. Normal deviation
    nd = normal_deviation_map(cad_v, cad_f, mesh_v, mesh_f)
    results.append((nd["mean_deg"],
                    "mean_normal_deviation_deg", nd))

    # 5. Surface area difference
    sa_cad = surface_area(cad_v, cad_f)
    sa_mesh = surface_area(mesh_v, mesh_f)
    area_pct = abs(sa_cad - sa_mesh) / max(sa_mesh, 1e-6) * 100
    results.append((area_pct,
                    "surface_area_pct_diff",
                    {"cad": sa_cad, "mesh": sa_mesh, "pct": area_pct}))

    # 6. Curvature histogram difference
    curv_diff = curvature_histogram_diff(cad_v, cad_f, mesh_v, mesh_f)
    results.append((curv_diff * 10,
                    "curvature_histogram_diff",
                    {"raw": curv_diff}))

    # 7. Bounding box aspect
    bb = bbox_aspect_diff(cad_v, mesh_v)
    results.append((bb * 100,
                    "bbox_aspect_diff",
                    {"raw": bb}))

    # 8. Radial profile RMS
    rp = radial_profile_rms(cad_v, mesh_v)
    results.append((rp,
                    "radial_profile_rms",
                    {"rms": rp}))

    # 9. Per-region area imbalance
    region_diffs = per_region_area_difference(cad_v, cad_f, mesh_v, mesh_f)
    max_region = max(region_diffs) * 100 if region_diffs else 0
    results.append((max_region,
                    "worst_region_area_diff_pct",
                    {"per_region": region_diffs}))

    # 10. Vertex density mismatch — compares local point spacing
    vd = vertex_density_diff(cad_v, mesh_v)
    results.append((vd * 100, "vertex_density_diff", {"raw": vd}))

    # 11. Cross-section contour difference at multiple Z heights
    cs = cross_section_contour_diff(cad_v, cad_f, mesh_v, mesh_f)
    results.append((cs, "cross_section_contour_diff", {"raw": cs}))

    # 12. Edge length distribution mismatch
    el = edge_length_distribution_diff(cad_v, cad_f, mesh_v, mesh_f)
    results.append((el * 10, "edge_length_dist_diff", {"raw": el}))

    # 13. Centroid distance per region
    cd = centroid_drift(cad_v, mesh_v)
    results.append((cd, "centroid_drift", {"raw": cd}))

    # 14. Angular silhouette from underneath (captures bottom detail)
    sil_bottom = silhouette_difference(cad_v, cad_f, mesh_v, mesh_f,
                                        n_angles=4, resolution=64)
    results.append((sil_bottom["max_pixel_error"] * 100,
                    "worst_angle_silhouette", sil_bottom))

    # 15. Median surface distance (robust to outliers, unlike mean)
    med_sd = median_surface_distance(cad_v, mesh_v)
    results.append((med_sd, "median_surface_distance", {"raw": med_sd}))

    # 16. Angular symmetry difference
    ang_sym = angular_symmetry_diff(cad_v, mesh_v)
    results.append((ang_sym * 100, "angular_symmetry_diff", {"raw": ang_sym}))

    # 17. Local roughness difference
    rough = local_roughness_diff(cad_v, cad_f, mesh_v, mesh_f)
    results.append((rough, "local_roughness_diff", {"raw": rough}))

    # 18. Volume difference
    vol = volume_diff(cad_v, cad_f, mesh_v, mesh_f)
    results.append((vol, "volume_diff", {"raw": vol}))

    # 19. 95th percentile distance
    p95 = percentile_95_distance(cad_v, mesh_v)
    results.append((p95, "percentile_95_distance", {"raw": p95}))

    # 20. Face normal consistency
    fnc = face_normal_consistency(cad_v, cad_f, mesh_v, mesh_f)
    results.append((fnc, "face_normal_consistency", {"raw": fnc}))

    # 21. Multi-scale distance
    msd = multi_scale_distance(cad_v, mesh_v)
    results.append((msd, "multi_scale_distance", {"raw": msd}))

    # 22. Shape diameter difference
    sdd = shape_diameter_diff(cad_v, cad_f, mesh_v, mesh_f)
    results.append((sdd, "shape_diameter_diff", {"raw": sdd}))

    # 23. Moment of inertia difference
    moi = moment_of_inertia_diff(cad_v, mesh_v)
    results.append((moi, "moment_of_inertia_diff", {"raw": moi}))

    # 24. D2 distance histogram difference
    dhd = distance_histogram_diff(cad_v, mesh_v)
    results.append((dhd * 10, "distance_histogram_diff", {"raw": dhd}))

    # 25. Z-profile area difference
    zpa = z_profile_area_diff(cad_v, cad_f, mesh_v, mesh_f)
    results.append((zpa, "z_profile_area_diff", {"raw": zpa}))

    # 26. Convexity defect
    cvx = convexity_defect_diff(cad_v, mesh_v)
    results.append((cvx, "convexity_defect_diff", {"raw": cvx}))

    # 27. Boundary edge length
    bel = boundary_edge_length_diff(cad_v, cad_f, mesh_v, mesh_f)
    results.append((bel, "boundary_edge_length_diff", {"raw": bel}))

    # 28. Principal curvature ratio
    pcr = principal_curvature_ratio_diff(cad_v, cad_f, mesh_v, mesh_f)
    results.append((pcr, "principal_curvature_ratio_diff", {"raw": pcr}))

    # 29. Geodesic diameter
    gd = geodesic_diameter_diff(cad_v, cad_f, mesh_v, mesh_f)
    results.append((gd, "geodesic_diameter_diff", {"raw": gd}))

    # 30. Laplacian spectrum
    ls = laplacian_spectrum_diff(cad_v, cad_f, mesh_v, mesh_f)
    results.append((ls, "laplacian_spectrum_diff", {"raw": ls}))

    # 31. Face area variance
    fav = face_area_variance_diff(cad_v, cad_f, mesh_v, mesh_f)
    results.append((fav, "face_area_variance_diff", {"raw": fav}))

    # 32. Vertex normal divergence
    vnd = vertex_normal_divergence(cad_v, cad_f, mesh_v, mesh_f)
    results.append((vnd, "vertex_normal_divergence", {"raw": vnd}))

    # 33. Octant volume balance
    ovd = octant_volume_diff(cad_v, cad_f, mesh_v, mesh_f)
    results.append((ovd, "octant_volume_diff", {"raw": ovd}))

    # 34. Edge dihedral angle distribution
    ead = edge_angle_distribution_diff(cad_v, cad_f, mesh_v, mesh_f)
    results.append((ead, "edge_angle_distribution_diff", {"raw": ead}))

    # 35. Triangle aspect ratio distribution
    ard = aspect_ratio_diff(cad_v, cad_f, mesh_v, mesh_f)
    results.append((ard, "aspect_ratio_diff", {"raw": ard}))

    # 36. Taper consistency (cross-section area along Z)
    tc = taper_consistency_diff(cad_v, cad_f, mesh_v, mesh_f)
    results.append((tc, "taper_consistency_diff", {"raw": tc}))

    # 37. Sweep path deviation (skeleton centerline)
    spd = sweep_path_deviation(cad_v, cad_f, mesh_v, mesh_f)
    results.append((spd, "sweep_path_deviation", {"raw": spd}))

    # 38. Profile circularity difference
    pcd = profile_circularity_diff(cad_v, cad_f, mesh_v, mesh_f)
    results.append((pcd, "profile_circularity_diff", {"raw": pcd}))

    # 39. Extrude twist (rotational offset per slice)
    etd = extrude_twist_diff(cad_v, cad_f, mesh_v, mesh_f)
    results.append((etd, "extrude_twist_diff", {"raw": etd}))

    # 40. Profile smoothness (revolve 2nd derivative)
    ps = profile_smoothness_diff(cad_v, mesh_v)
    results.append((ps, "profile_smoothness_diff", {"raw": ps}))

    # 41. Radial asymmetry (per-sector variation)
    ra = radial_asymmetry_diff(cad_v, mesh_v)
    results.append((ra, "radial_asymmetry_diff", {"raw": ra}))

    results.sort(key=lambda x: x[0], reverse=True)
    return results


FIXERS = {
    "silhouette_pixel_error": fix_silhouette_mismatch,
    "hausdorff_distance": fix_hausdorff_outliers,
    "mean_normal_deviation_deg": fix_normal_deviation,
    "surface_area_pct_diff": fix_surface_area,
    "curvature_histogram_diff": fix_curvature_mismatch,
    "bbox_aspect_diff": fix_bbox_aspect,
    "radial_profile_rms": fix_radial_profile,
    "worst_region_area_diff_pct": fix_region_area,
    "mean_surface_distance": fix_silhouette_mismatch,
    "vertex_density_diff": fix_vertex_density,
    "cross_section_contour_diff": fix_cross_section_contour,
    "edge_length_dist_diff": fix_edge_length_distribution,
    "centroid_drift": fix_centroid_drift,
    "worst_angle_silhouette": fix_worst_angle_silhouette,
    "median_surface_distance": fix_median_surface_distance,
    "angular_symmetry_diff": fix_angular_symmetry,
    "local_roughness_diff": fix_local_roughness,
    "volume_diff": fix_volume,
    "percentile_95_distance": fix_percentile_95,
    "face_normal_consistency": fix_face_normal_consistency,
    "multi_scale_distance": fix_multi_scale_distance,
    "shape_diameter_diff": fix_shape_diameter,
    "moment_of_inertia_diff": fix_moment_of_inertia,
    "distance_histogram_diff": fix_distance_histogram,
    "z_profile_area_diff": fix_z_profile_area,
    "convexity_defect_diff": fix_convexity_defect,
    "boundary_edge_length_diff": fix_boundary_edges,
    "principal_curvature_ratio_diff": fix_principal_curvature,
    "geodesic_diameter_diff": fix_geodesic_diameter,
    "laplacian_spectrum_diff": fix_laplacian_spectrum,
    "face_area_variance_diff": fix_face_area_variance,
    "vertex_normal_divergence": fix_vertex_normal_divergence,
    "octant_volume_diff": fix_octant_volume,
    "edge_angle_distribution_diff": fix_edge_angle_distribution,
    "aspect_ratio_diff": fix_aspect_ratio,
    "taper_consistency_diff": fix_taper_consistency,
    "sweep_path_deviation": fix_sweep_path_deviation,
    "profile_circularity_diff": fix_profile_circularity,
    "extrude_twist_diff": fix_extrude_twist,
    "profile_smoothness_diff": fix_profile_smoothness,
    "radial_asymmetry_diff": fix_radial_asymmetry,
}


# =========================================================================
# Test-case generators
# =========================================================================

def make_test_pair_vase():
    """Vase: revolve CAD simple vs ornate mesh."""
    from meshxcad.objects.catalog import make_simple, make_ornate
    cad_v, cad_f = make_simple("classical_vase")
    mesh_v, mesh_f = make_ornate("classical_vase")
    return cad_v, cad_f, mesh_v, mesh_f, "vase"


def make_test_pair_goblet():
    from meshxcad.objects.catalog import make_simple, make_ornate
    cad_v, cad_f = make_simple("goblet")
    mesh_v, mesh_f = make_ornate("goblet")
    return cad_v, cad_f, mesh_v, mesh_f, "goblet"


def make_test_pair_gear():
    from meshxcad.objects.complex_catalog import make_complex_simple, make_complex_ornate
    cad_v, cad_f = make_complex_simple("spur_gear")
    mesh_v, mesh_f = make_complex_ornate("spur_gear")
    return cad_v, cad_f, mesh_v, mesh_f, "spur_gear"


def make_test_pair_cylinder():
    cad_v, cad_f = make_cylinder_mesh(radius=5.0, height=15.0,
                                       radial_divs=32, height_divs=20)
    mesh_v = add_cylinder_grooves(cad_v.copy(), radius=5.0, height=15.0)
    return cad_v, cad_f, mesh_v, cad_f.copy(), "grooved_cylinder"


def make_test_pair_sphere():
    cad_v, cad_f = make_sphere_mesh(radius=5.0, lat_divs=20, lon_divs=20)
    mesh_v = add_sphere_dimples(cad_v.copy(), radius=5.0)
    return cad_v, cad_f, mesh_v, cad_f.copy(), "dimpled_sphere"


def make_test_pair_bracket():
    from meshxcad.objects.complex_catalog import make_complex_simple, make_complex_ornate
    cad_v, cad_f = make_complex_simple("shelf_bracket")
    mesh_v, mesh_f = make_complex_ornate("shelf_bracket")
    return cad_v, cad_f, mesh_v, mesh_f, "shelf_bracket"


def make_test_pair_torus():
    """Torus: smooth CAD vs noisy mesh."""
    cad_v, cad_f = make_torus(major_r=8.0, minor_r=3.0, z_center=0.0, n_angular=30, n_cross=15)
    mesh_v = cad_v.copy() + np.random.RandomState(42).randn(*cad_v.shape) * 0.3
    return cad_v, cad_f, mesh_v, cad_f.copy(), "torus"


def make_test_pair_scaled_sphere():
    """Sphere: different resolutions and anisotropic scale."""
    cad_v, cad_f = make_sphere_mesh(radius=5.0, lat_divs=15, lon_divs=15)
    mesh_v, mesh_f = make_sphere_mesh(radius=5.0, lat_divs=25, lon_divs=25)
    # Apply anisotropic scaling to mesh
    mesh_v[:, 0] *= 1.3
    mesh_v[:, 2] *= 0.8
    return cad_v, cad_f, mesh_v, mesh_f, "aniso_sphere"


def make_test_pair_offset_cube():
    """Cube: shifted + rotated."""
    cad_v, cad_f = make_cube_mesh(size=10.0, subdivisions=4)
    mesh_v = cad_v.copy()
    # Rotate 15 degrees around Z
    theta = math.radians(15)
    R = np.array([[math.cos(theta), -math.sin(theta), 0],
                   [math.sin(theta), math.cos(theta), 0],
                   [0, 0, 1]])
    mesh_v = mesh_v @ R.T
    mesh_v += np.array([1.0, 0.5, -0.3])
    return cad_v, cad_f, mesh_v, cad_f.copy(), "offset_cube"


def make_test_pair_bent_tube():
    """Sweep: straight tube vs curved tube."""
    from meshxcad.objects.operations import sweep_along_path
    # Straight tube
    t = np.linspace(0, 20, 20)
    path_straight = np.column_stack([np.zeros_like(t), np.zeros_like(t), t])
    profile = np.column_stack([
        3.0 * np.cos(np.linspace(0, 2 * np.pi, 16, endpoint=False)),
        3.0 * np.sin(np.linspace(0, 2 * np.pi, 16, endpoint=False)),
    ])
    cad_v, cad_f = sweep_along_path(profile, path_straight, n_profile=16)

    # Curved tube
    t2 = np.linspace(0, np.pi, 20)
    path_curved = np.column_stack([
        10.0 * np.sin(t2), np.zeros(20), 10.0 * (1 - np.cos(t2))])
    mesh_v, mesh_f = sweep_along_path(profile, path_curved, n_profile=16)
    return cad_v, cad_f, mesh_v, mesh_f, "bent_tube"


def make_test_pair_hollow_sphere():
    """Thin shell: solid vs hollow sphere (outer shell only)."""
    cad_v, cad_f = make_sphere_mesh(radius=5.0, lat_divs=15, lon_divs=15)
    mesh_v, mesh_f = make_sphere_mesh(radius=5.5, lat_divs=20, lon_divs=20)
    # Add random surface noise to mesh
    rng = np.random.RandomState(99)
    mesh_v = mesh_v + rng.randn(*mesh_v.shape) * 0.2
    return cad_v, cad_f, mesh_v, mesh_f, "hollow_sphere"


def make_test_pair_multirez_cyl():
    """Same cylinder at different tessellation resolutions."""
    cad_v, cad_f = make_cylinder_mesh(radius=5.0, height=15.0,
                                       radial_divs=12, height_divs=5)
    mesh_v, mesh_f = make_cylinder_mesh(radius=5.0, height=15.0,
                                         radial_divs=40, height_divs=30)
    return cad_v, cad_f, mesh_v, mesh_f, "multirez_cylinder"


def make_test_pair_concave():
    """Concave L-shape: simple vs detailed."""
    from meshxcad.objects.complex_catalog import make_complex_simple, make_complex_ornate
    cad_v, cad_f = make_complex_simple("hex_nut")
    mesh_v, mesh_f = make_complex_ornate("hex_nut")
    return cad_v, cad_f, mesh_v, mesh_f, "hex_nut"


def make_test_pair_asymmetric():
    """Asymmetric deformation of a sphere."""
    cad_v, cad_f = make_sphere_mesh(radius=5.0, lat_divs=20, lon_divs=20)
    mesh_v = cad_v.copy()
    # Apply asymmetric deformation: bulge on one side
    dist_x = mesh_v[:, 0]
    mask = dist_x > 0
    mesh_v[mask, 0] *= 1.4
    mesh_v[mask, 1] *= 0.9
    return cad_v, cad_f, mesh_v, cad_f.copy(), "asymmetric_sphere"


def make_test_pair_column():
    """Column: simple vs fluted."""
    from meshxcad.objects.complex_catalog import make_complex_simple, make_complex_ornate
    cad_v, cad_f = make_complex_simple("fluted_column")
    mesh_v, mesh_f = make_complex_ornate("fluted_column")
    return cad_v, cad_f, mesh_v, mesh_f, "fluted_column"


TEST_PAIRS = [
    make_test_pair_vase,
    make_test_pair_goblet,
    make_test_pair_gear,
    make_test_pair_cylinder,
    make_test_pair_sphere,
    make_test_pair_bracket,
    make_test_pair_torus,
    make_test_pair_scaled_sphere,
    make_test_pair_offset_cube,
    make_test_pair_bent_tube,
    make_test_pair_hollow_sphere,
    make_test_pair_multirez_cyl,
    make_test_pair_concave,
    make_test_pair_asymmetric,
    make_test_pair_column,
]


# =========================================================================
# Main adversarial loop
# =========================================================================

def _composite_score(diffs):
    """Compute a single composite score from all differentiator results.

    Weights higher scores more (top-heavy weighting) to focus on the
    worst remaining gaps.
    """
    if not diffs:
        return 0.0
    scores = np.array([d[0] for d in diffs])
    # Top-heavy: weight by rank (1st gets 2x, 2nd 1.5x, rest 1x)
    weights = np.ones(len(scores))
    if len(weights) > 0:
        weights[0] = 2.0
    if len(weights) > 1:
        weights[1] = 1.5
    return float(np.average(scores, weights=weights))


def _select_fixer_strategy(diffs, tried, fixer_scores):
    """Select the best fixer to apply based on historic success rates.

    Prioritizes:
    1. Untried differentiators with high current scores
    2. Fixers with high historical improvement rates
    3. Falls back to the highest-scoring differentiator
    """
    # Build candidates: (priority_score, diff_name)
    candidates = []
    for score, name, _ in diffs:
        if name not in FIXERS:
            continue
        hist_rate = fixer_scores.get(name, {}).get("avg_improvement", 50.0)
        freshness = 2.0 if name not in tried else 0.5
        priority = score * freshness * (hist_rate / 100.0 + 0.5)
        candidates.append((priority, score, name))

    if not candidates:
        return diffs[0][0], diffs[0][1] if diffs else (0.0, "")

    candidates.sort(key=lambda c: c[0], reverse=True)
    return candidates[0][1], candidates[0][2]


def run_adversarial_loop(output_dir="/tmp/adversarial_loop",
                          max_duration_sec=3600,
                          max_rounds=50):
    """Execute the adversarial loop with accumulative state and adaptive strategy.

    The loop maintains a per-pair "current best" CAD mesh.  Each round:
    1. Pick the next test pair (priority-weighted toward worst-scoring pairs).
    2. If first visit: apply full alignment pipeline.
       Otherwise: start from the accumulated result.
    3. RED team: run all 35 differentiators, find the worst remaining gap.
    4. BLUE team: apply the targeted fixer + up to 2 additional fixers
       for the next-worst differentiators, using adaptive strategy selection.
    5. Verify improvement with composite scoring; if improved, keep.
    6. Update fixer success statistics for future rounds.
    7. Log everything.

    This means each pair gets progressively better across rounds, and the
    RED team must find new, harder differentiators each cycle.

    Args:
        output_dir:      directory for logs and renders
        max_duration_sec: time limit in seconds
        max_rounds:       maximum iterations

    Returns:
        list of round result dicts
    """
    os.makedirs(output_dir, exist_ok=True)
    start_time = time.time()
    results = []
    round_num = 0

    # Accumulative state per pair: {pair_name: aligned_vertices}
    pair_state = {}
    # Track which fixers have been tried per pair to force variety
    pair_tried_fixers = {}
    # Per-pair composite scores (for priority scheduling)
    pair_composite = {}
    # Fixer success statistics
    fixer_stats = {}  # {fixer_name: {"attempts": n, "successes": n, "total_improvement": f}}

    while (time.time() - start_time < max_duration_sec and
           round_num < max_rounds):

        # Priority-weighted pair selection: pick the pair with worst composite
        if round_num < len(TEST_PAIRS):
            # First pass: cycle through all pairs
            pair_fn = TEST_PAIRS[round_num % len(TEST_PAIRS)]
        else:
            # After first pass: prioritize worst-scoring pairs
            scored_pairs = []
            for i, pfn in enumerate(TEST_PAIRS):
                try:
                    _, _, _, _, name = pfn()
                except Exception:
                    continue
                score = pair_composite.get(name, float('inf'))
                scored_pairs.append((score, i))
            if scored_pairs:
                scored_pairs.sort(key=lambda x: x[0], reverse=True)
                # Pick from top 3 worst, with some randomness
                top_n = min(3, len(scored_pairs))
                pick = round_num % top_n
                pair_fn = TEST_PAIRS[scored_pairs[pick][1]]
            else:
                pair_fn = TEST_PAIRS[round_num % len(TEST_PAIRS)]

        try:
            cad_v, cad_f, mesh_v, mesh_f, pair_name = pair_fn()
        except Exception:
            round_num += 1
            continue

        # ---- Phase 1: Get current state or initialize ----
        if pair_name not in pair_state:
            aligned_v = full_alignment_pipeline(cad_v, cad_f, mesh_v, mesh_f)
            pair_state[pair_name] = aligned_v
            pair_tried_fixers[pair_name] = set()
        else:
            aligned_v = pair_state[pair_name]

        # ---- Phase 2: Run differentiators (RED team) ----
        diffs = run_differentiators(aligned_v, cad_f, mesh_v, mesh_f)
        composite_before = _composite_score(diffs)
        pair_composite[pair_name] = composite_before

        # Select fixer using adaptive strategy
        tried = pair_tried_fixers[pair_name]
        fixer_avg = {
            k: {"avg_improvement": v["total_improvement"] / max(v["attempts"], 1)}
            for k, v in fixer_stats.items()
        }
        top_diff_score, top_diff_name = _select_fixer_strategy(
            diffs, tried, fixer_avg)

        # If everything tried, reset
        if top_diff_name in tried and len(tried) >= len(FIXERS):
            pair_tried_fixers[pair_name] = set()
            tried = set()
            top_diff_score, top_diff_name = _select_fixer_strategy(
                diffs, tried, fixer_avg)

        # ---- Phase 3: Apply targeted fix (BLUE team) ----
        current_v = aligned_v.copy()
        fixers_applied = []

        # Apply primary fixer — try top candidates until one improves composite
        primary_applied = False
        for _, cand_name, _ in diffs:
            if cand_name in tried and len(diffs) > len(tried):
                continue
            fixer = FIXERS.get(cand_name)
            if not fixer:
                continue
            candidate = fixer(aligned_v.copy(), cad_f, mesh_v, mesh_f)
            comp_cand = _composite_score(
                run_differentiators(candidate, cad_f, mesh_v, mesh_f))
            if comp_cand < composite_before:
                current_v = candidate
                fixers_applied.append(cand_name)
                top_diff_name = cand_name
                primary_applied = True
                break

        if not primary_applied:
            # Fall back: just apply the top fixer regardless
            fixer = FIXERS.get(top_diff_name)
            if fixer:
                current_v = fixer(current_v, cad_f, mesh_v, mesh_f)
                fixers_applied.append(top_diff_name)

        # Apply up to 2 more fixers, keeping each only if composite improves
        comp_current = _composite_score(
            run_differentiators(current_v, cad_f, mesh_v, mesh_f))
        for extra_pass in range(2):
            diffs_now = run_differentiators(current_v, cad_f, mesh_v, mesh_f)
            applied = False
            for _, next_name, _ in diffs_now:
                next_fixer = FIXERS.get(next_name)
                if next_fixer and next_name not in fixers_applied:
                    candidate_v = next_fixer(
                        current_v, cad_f, mesh_v, mesh_f)
                    comp_candidate = _composite_score(
                        run_differentiators(candidate_v, cad_f, mesh_v, mesh_f))
                    if comp_candidate < comp_current:
                        current_v = candidate_v
                        comp_current = comp_candidate
                        fixers_applied.append(next_name)
                        applied = True
                    break
            if not applied:
                break

        # ---- Phase 4: Re-measure (VERIFY) ----
        diffs_after_fix = run_differentiators(current_v, cad_f, mesh_v, mesh_f)
        composite_after = _composite_score(diffs_after_fix)

        # Find the same metric after fix for fair comparison
        fixed_same_metric = top_diff_score
        for s, n, _ in diffs_after_fix:
            if n == top_diff_name:
                fixed_same_metric = s
                break

        # Use composite score for accept/reject decision
        improved = composite_after < composite_before
        improvement_pct = ((composite_before - composite_after) /
                           max(abs(composite_before), 1e-6) * 100
                           if composite_before != 0 else 0)

        # Keep the improvement if it helped
        if improved:
            pair_state[pair_name] = current_v
            pair_composite[pair_name] = composite_after

        pair_tried_fixers[pair_name].add(top_diff_name)

        # Update fixer statistics
        for fname in fixers_applied:
            if fname not in fixer_stats:
                fixer_stats[fname] = {"attempts": 0, "successes": 0,
                                       "total_improvement": 0.0}
            fixer_stats[fname]["attempts"] += 1
            if improved:
                fixer_stats[fname]["successes"] += 1
                fixer_stats[fname]["total_improvement"] += improvement_pct

        # ---- Record ----
        elapsed = time.time() - start_time
        round_result = {
            "round": round_num,
            "elapsed_sec": round(elapsed, 1),
            "pair": pair_name,
            "red_team": {
                "top_differentiator": top_diff_name,
                "score_before_fix": round(top_diff_score, 4),
                "composite_before": round(composite_before, 4),
            },
            "blue_team": {
                "fixers_applied": fixers_applied,
                "score_after_fix": round(fixed_same_metric, 4),
                "composite_after": round(composite_after, 4),
                "improved": improved,
                "improvement_pct": round(improvement_pct, 1),
            },
            "final_top3": [
                {"name": n, "score": round(s, 4)}
                for s, n, _ in diffs_after_fix[:3]
            ],
        }
        results.append(round_result)

        # Print progress
        status = "IMPROVED" if improved else "HELD"
        print(f"[Round {round_num:2d} | {elapsed:6.1f}s] {pair_name:20s} "
              f"RED={top_diff_name:30s} score={top_diff_score:8.3f} → "
              f"{fixed_same_metric:8.3f} composite={composite_before:.2f}→"
              f"{composite_after:.2f} ({improvement_pct:+.1f}%) [{status}]")

        # Save incremental results
        with open(os.path.join(output_dir, "results.json"), "w") as f:
            json.dump(results, f, indent=2)

        round_num += 1

    # ---- Final summary ----
    total_rounds = len(results)
    improved_rounds = sum(1 for r in results if r["blue_team"]["improved"])
    avg_improvement = (np.mean([r["blue_team"]["improvement_pct"]
                                for r in results]) if results else 0)

    # Fixer leaderboard
    fixer_board = []
    for fname, stats in fixer_stats.items():
        rate = stats["successes"] / max(stats["attempts"], 1) * 100
        avg_imp = stats["total_improvement"] / max(stats["successes"], 1)
        fixer_board.append({
            "fixer": fname,
            "attempts": stats["attempts"],
            "successes": stats["successes"],
            "success_rate_pct": round(rate, 1),
            "avg_improvement_pct": round(avg_imp, 1),
        })
    fixer_board.sort(key=lambda x: x["success_rate_pct"], reverse=True)

    summary = {
        "total_rounds": total_rounds,
        "improved_rounds": improved_rounds,
        "held_rounds": total_rounds - improved_rounds,
        "avg_improvement_pct": round(float(avg_improvement), 1),
        "total_elapsed_sec": round(time.time() - start_time, 1),
        "unique_differentiators_found": len(set(
            r["red_team"]["top_differentiator"] for r in results)),
        "total_differentiators": 41,
        "total_fixers": len(FIXERS),
        "total_test_pairs": len(TEST_PAIRS),
        "fixer_leaderboard": fixer_board[:10],
    }
    print(f"\n{'='*70}")
    print(f"ADVERSARIAL LOOP COMPLETE")
    print(f"  Rounds: {total_rounds}")
    print(f"  Improved: {improved_rounds}/{total_rounds}")
    print(f"  Avg improvement: {avg_improvement:.1f}%")
    print(f"  Unique differentiators: {summary['unique_differentiators_found']}/41")
    print(f"  Test pairs: {len(TEST_PAIRS)}")
    if fixer_board:
        print(f"\n  Top fixers:")
        for fb in fixer_board[:5]:
            print(f"    {fb['fixer']:35s} "
                  f"{fb['successes']}/{fb['attempts']} "
                  f"({fb['success_rate_pct']}%) "
                  f"avg +{fb['avg_improvement_pct']}%")
    print(f"  Elapsed: {summary['total_elapsed_sec']:.1f}s")
    print(f"{'='*70}")

    with open(os.path.join(output_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    return results


if __name__ == "__main__":
    run_adversarial_loop()
