"""Tests for meshxcad.revolve_align — 12 revolve-to-mesh alignment features."""

import math
import numpy as np
import pytest

from meshxcad.revolve_align import (
    extract_radial_profile,
    compare_radial_profiles,
    fit_profile_to_mesh,
    detect_revolve_axis,
    angular_deviation_map,
    refine_profile_radii,
    refine_profile_z_spacing,
    insert_profile_detail,
    smooth_profile_to_mesh,
    remove_profile_redundancy,
    adaptive_revolve,
    suggest_revolve_adjustments,
)
from meshxcad.objects.builder import revolve_profile
from meshxcad.synthetic import make_cylinder_mesh, make_sphere_mesh


# ---- helpers ----

def _make_cylinder_verts(radius=5.0, height=20.0, n=1000):
    """Quick cylinder point cloud (no faces needed for most tests)."""
    rng = np.random.RandomState(42)
    angles = rng.uniform(0, 2 * math.pi, n)
    zs = rng.uniform(0, height, n)
    xs = radius * np.cos(angles)
    ys = radius * np.sin(angles)
    return np.column_stack([xs, ys, zs])


def _make_vase_verts(n_per_ring=60, n_rings=40):
    """Vase-shaped point cloud — radius varies sinusoidally along Z."""
    verts = []
    for i in range(n_rings):
        z = 100.0 * i / (n_rings - 1)
        # Vase: wide at bottom, narrow waist at 60%, flare at top
        t = z / 100.0
        r = 15.0 + 10.0 * math.sin(math.pi * t) - 5.0 * math.sin(2 * math.pi * t)
        for j in range(n_per_ring):
            angle = 2 * math.pi * j / n_per_ring
            verts.append([r * math.cos(angle), r * math.sin(angle), z])
    return np.array(verts, dtype=np.float64)


def _simple_profile():
    """Simple constant-radius profile."""
    return [(5.0, z) for z in np.linspace(0, 20, 10)]


def _vase_profile(n=20):
    """Vase profile matching _make_vase_verts."""
    profile = []
    for i in range(n):
        z = 100.0 * i / (n - 1)
        t = z / 100.0
        r = 15.0 + 10.0 * math.sin(math.pi * t) - 5.0 * math.sin(2 * math.pi * t)
        profile.append((r, z))
    return profile


# =========================================================================
# 1. extract_radial_profile
# =========================================================================

class TestExtractRadialProfile:
    def test_cylinder_constant_radius(self):
        verts = _make_cylinder_verts(radius=5.0, height=20.0)
        prof = extract_radial_profile(verts, n_slices=10)
        assert len(prof) > 0
        # All radii should be ~5
        assert np.all(np.abs(prof[:, 0] - 5.0) < 1.0)

    def test_z_ordering(self):
        verts = _make_cylinder_verts()
        prof = extract_radial_profile(verts, n_slices=20)
        # z values should be monotonically increasing
        assert np.all(np.diff(prof[:, 1]) > 0)

    def test_vase_varying_radius(self):
        verts = _make_vase_verts()
        prof = extract_radial_profile(verts, n_slices=20)
        # Radius should vary
        assert np.ptp(prof[:, 0]) > 5.0

    def test_empty_returns_empty(self):
        verts = np.array([[0, 0, 0]], dtype=np.float64)
        prof = extract_radial_profile(verts, n_slices=5)
        assert len(prof) >= 1

    def test_sphere(self):
        verts, _ = make_sphere_mesh(radius=10.0, lat_divs=30, lon_divs=30)
        prof = extract_radial_profile(verts, n_slices=20)
        # Middle slices should have radius ~10
        mid = prof[len(prof) // 2]
        assert abs(mid[0] - 10.0) < 2.0


# =========================================================================
# 2. compare_radial_profiles
# =========================================================================

class TestCompareRadialProfiles:
    def test_identical_profiles(self):
        prof = np.array(_simple_profile())
        metrics = compare_radial_profiles(prof, prof)
        assert metrics["mean_r_error"] < 0.01
        assert metrics["rms_r_error"] < 0.01
        assert metrics["z_overlap"] > 0.99

    def test_different_radii(self):
        a = np.array([(5.0, z) for z in np.linspace(0, 20, 10)])
        b = np.array([(10.0, z) for z in np.linspace(0, 20, 10)])
        metrics = compare_radial_profiles(a, b)
        assert abs(metrics["mean_r_error"] - 5.0) < 0.5
        assert metrics["z_overlap"] > 0.9

    def test_partial_z_overlap(self):
        a = np.array([(5.0, z) for z in np.linspace(0, 20, 10)])
        b = np.array([(5.0, z) for z in np.linspace(10, 30, 10)])
        metrics = compare_radial_profiles(a, b)
        assert metrics["z_overlap"] < 0.7

    def test_no_overlap(self):
        a = np.array([(5.0, z) for z in np.linspace(0, 10, 10)])
        b = np.array([(5.0, z) for z in np.linspace(20, 30, 10)])
        metrics = compare_radial_profiles(a, b)
        assert metrics["z_overlap"] == 0.0
        assert metrics["mean_r_error"] == float("inf")


# =========================================================================
# 3. fit_profile_to_mesh
# =========================================================================

class TestFitProfileToMesh:
    def test_cylinder_with_matching_profile(self):
        verts = _make_cylinder_verts(radius=5.0, height=20.0)
        prof = _simple_profile()
        fitted, residuals = fit_profile_to_mesh(prof, verts)
        # Fitted radii should be close to 5.0
        for r, z in fitted:
            assert abs(r - 5.0) < 1.5

    def test_wrong_radius_gets_corrected(self):
        verts = _make_cylinder_verts(radius=10.0, height=20.0)
        prof = [(5.0, z) for z in np.linspace(0, 20, 10)]
        fitted, residuals = fit_profile_to_mesh(prof, verts)
        # Residuals should be ~5.0 (original offset)
        assert np.mean(residuals) > 3.0
        # Fitted should be close to 10.0
        fitted_radii = [r for r, z in fitted]
        assert abs(np.mean(fitted_radii) - 10.0) < 2.0

    def test_preserves_z_values(self):
        verts = _make_cylinder_verts()
        prof = _simple_profile()
        fitted, _ = fit_profile_to_mesh(prof, verts)
        orig_zs = [z for _, z in prof]
        fitted_zs = [z for _, z in fitted]
        np.testing.assert_allclose(fitted_zs, orig_zs, atol=1e-10)


# =========================================================================
# 4. detect_revolve_axis
# =========================================================================

class TestDetectRevolveAxis:
    def test_cylinder_z_axis(self):
        verts = _make_cylinder_verts(radius=5.0, height=40.0)
        result = detect_revolve_axis(verts)
        # Axis should be close to Z
        assert abs(result["axis_direction"][2]) > 0.8
        assert result["circularity"] > 0.8

    def test_sphere_high_circularity(self):
        verts, _ = make_sphere_mesh(radius=10.0, lat_divs=30, lon_divs=30)
        result = detect_revolve_axis(verts)
        # Sphere is symmetric in all directions
        assert result["circularity"] > 0.5

    def test_tall_cylinder_vs_flat_disk(self):
        tall = _make_cylinder_verts(radius=5.0, height=100.0)
        flat = _make_cylinder_verts(radius=50.0, height=2.0)
        tall_res = detect_revolve_axis(tall)
        flat_res = detect_revolve_axis(flat)
        # Both should find Z axis
        assert abs(tall_res["axis_direction"][2]) > 0.7
        # Flat disk — axis should still roughly be Z but PCA might differ
        assert len(flat_res["axis_direction"]) == 3

    def test_axis_point_near_centroid(self):
        verts = _make_cylinder_verts()
        result = detect_revolve_axis(verts)
        centroid = np.mean(verts, axis=0)
        np.testing.assert_allclose(result["axis_point"], centroid, atol=0.01)


# =========================================================================
# 5. angular_deviation_map
# =========================================================================

class TestAngularDeviationMap:
    def test_perfect_cylinder_small_deviation(self):
        """A perfect cylinder should have near-zero angular deviation."""
        verts, _ = make_cylinder_mesh(radius=5.0, height=10.0,
                                       radial_divs=48, height_divs=20)
        result = angular_deviation_map(verts, n_angular=12, n_z=10)
        assert result["mean_abs_dev"] < 0.5
        assert result["deviation"].shape == (10, 12)

    def test_angles_and_z_shapes(self):
        verts = _make_cylinder_verts()
        result = angular_deviation_map(verts, n_angular=24, n_z=15)
        assert len(result["angles"]) == 24
        assert len(result["z_values"]) == 15
        assert result["deviation"].shape == (15, 24)

    def test_asymmetric_mesh(self):
        """A mesh with one side displaced should show angular deviation."""
        verts = _make_cylinder_verts(radius=5.0, height=20.0, n=2000)
        # Displace positive-x vertices outward
        mask = verts[:, 0] > 0
        verts[mask, 0] += 3.0
        result = angular_deviation_map(verts, n_angular=12, n_z=10)
        assert result["max_dev"] > 0.5


# =========================================================================
# 6. refine_profile_radii
# =========================================================================

class TestRefineProfileRadii:
    def test_blend_0_unchanged(self):
        verts = _make_cylinder_verts(radius=10.0)
        prof = [(5.0, z) for z in np.linspace(0, 20, 10)]
        refined = refine_profile_radii(prof, verts, blend=0.0)
        for (r_orig, _), (r_ref, _) in zip(prof, refined):
            assert abs(r_ref - r_orig) < 0.01

    def test_blend_1_matches_mesh(self):
        verts = _make_cylinder_verts(radius=10.0)
        prof = [(5.0, z) for z in np.linspace(0, 20, 10)]
        refined = refine_profile_radii(prof, verts, blend=1.0)
        for r, z in refined:
            assert abs(r - 10.0) < 2.0

    def test_blend_05_halfway(self):
        verts = _make_cylinder_verts(radius=10.0)
        prof = [(0.0, z) for z in np.linspace(0, 20, 10)]
        refined = refine_profile_radii(prof, verts, blend=0.5)
        for r, z in refined:
            assert r > 2.0  # moved toward 10 from 0

    def test_preserves_z(self):
        verts = _make_cylinder_verts()
        prof = [(5.0, z) for z in [0, 5, 10, 15, 20]]
        refined = refine_profile_radii(prof, verts, blend=1.0)
        for (_, z_orig), (_, z_ref) in zip(prof, refined):
            assert abs(z_ref - z_orig) < 0.01


# =========================================================================
# 7. refine_profile_z_spacing
# =========================================================================

class TestRefineProfileZSpacing:
    def test_output_length(self):
        verts = _make_vase_verts()
        prof = _vase_profile(n=15)
        result = refine_profile_z_spacing(prof, verts, n_output=20)
        assert len(result) == 20

    def test_z_range_preserved(self):
        verts = _make_vase_verts()
        prof = _vase_profile(n=15)
        result = refine_profile_z_spacing(prof, verts)
        z_orig = [z for _, z in prof]
        z_new = [z for _, z in result]
        assert abs(z_new[0] - z_orig[0]) < 2.0
        assert abs(z_new[-1] - z_orig[-1]) < 2.0

    def test_cylinder_roughly_uniform(self):
        """Cylinder has no curvature variation → spacing should stay ~uniform."""
        verts = _make_cylinder_verts(radius=5.0, height=20.0, n=2000)
        prof = _simple_profile()
        result = refine_profile_z_spacing(prof, verts)
        z_vals = [z for _, z in result]
        gaps = np.diff(z_vals)
        # Coefficient of variation should be moderate
        assert np.std(gaps) / (np.mean(gaps) + 1e-12) < 2.0


# =========================================================================
# 8. insert_profile_detail
# =========================================================================

class TestInsertProfileDetail:
    def test_coarse_profile_gets_refined(self):
        """A 3-point profile on a vase mesh should gain points."""
        verts = _make_vase_verts()
        coarse = [(15.0, 0.0), (15.0, 50.0), (15.0, 100.0)]
        enriched = insert_profile_detail(coarse, verts, threshold=1.0)
        assert len(enriched) > len(coarse)

    def test_accurate_profile_unchanged(self):
        """A profile that already matches should gain few/no points."""
        verts = _make_cylinder_verts(radius=5.0, height=20.0, n=2000)
        prof = [(5.0, z) for z in np.linspace(0, 20, 20)]
        enriched = insert_profile_detail(prof, verts, threshold=1.0)
        # May gain at most a couple of points
        assert len(enriched) <= len(prof) + 5

    def test_preserves_endpoints(self):
        verts = _make_vase_verts()
        prof = [(15.0, 0.0), (15.0, 100.0)]
        enriched = insert_profile_detail(prof, verts, threshold=0.5)
        assert enriched[0] == (15.0, 0.0)
        assert enriched[-1] == (15.0, 100.0)


# =========================================================================
# 9. smooth_profile_to_mesh
# =========================================================================

class TestSmoothProfileToMesh:
    def test_smoothing_1_preserves_profile(self):
        verts = _make_cylinder_verts(radius=10.0)
        prof = [(5.0, z) for z in np.linspace(0, 20, 10)]
        result = smooth_profile_to_mesh(prof, verts, smoothing=1.0)
        # With smoothing=1 the profile dominates
        for r, z in result:
            assert abs(r - 5.0) < 2.0

    def test_smoothing_0_matches_mesh(self):
        verts = _make_cylinder_verts(radius=10.0)
        prof = [(5.0, z) for z in np.linspace(0, 20, 10)]
        result = smooth_profile_to_mesh(prof, verts, smoothing=0.0)
        # Should be close to mesh radius 10.0
        radii = [r for r, z in result]
        assert abs(np.mean(radii) - 10.0) < 2.0

    def test_output_length(self):
        verts = _make_cylinder_verts()
        prof = _simple_profile()
        result = smooth_profile_to_mesh(prof, verts, n_output=30)
        assert len(result) == 30

    def test_radii_positive(self):
        verts = _make_vase_verts()
        prof = _vase_profile()
        result = smooth_profile_to_mesh(prof, verts, smoothing=0.5)
        for r, z in result:
            assert r >= 0.01


# =========================================================================
# 10. remove_profile_redundancy
# =========================================================================

class TestRemoveProfileRedundancy:
    def test_straight_line_collapses(self):
        """Colinear points should be removed."""
        prof = [(5.0, z) for z in range(20)]
        simplified = remove_profile_redundancy(prof, angle_threshold_deg=10.0)
        # All interior points are colinear → only endpoints remain
        assert len(simplified) == 2

    def test_preserves_corners(self):
        """Sharp corners should be kept."""
        prof = [(5.0, 0.0), (5.0, 10.0), (10.0, 10.0), (10.0, 20.0)]
        simplified = remove_profile_redundancy(prof, angle_threshold_deg=5.0)
        assert len(simplified) == 4  # all are corners

    def test_preserves_endpoints(self):
        prof = [(5.0, z) for z in range(10)]
        simplified = remove_profile_redundancy(prof)
        assert simplified[0] == prof[0]
        assert simplified[-1] == prof[-1]

    def test_small_profile_unchanged(self):
        prof = [(5.0, 0.0), (10.0, 5.0), (5.0, 10.0)]
        simplified = remove_profile_redundancy(prof)
        assert len(simplified) == 3


# =========================================================================
# 11. adaptive_revolve
# =========================================================================

class TestAdaptiveRevolve:
    def test_cylinder_reconstruction(self):
        verts, faces = make_cylinder_mesh(radius=5.0, height=10.0,
                                           radial_divs=32, height_divs=20)
        av, af = adaptive_revolve(verts, n_slices=10, n_angular=24)
        assert av.shape[1] == 3
        assert af.shape[1] == 3
        # Radii should be ~5.0
        r = np.sqrt(av[:, 0] ** 2 + av[:, 1] ** 2)
        # Exclude cap centers (r=0)
        r_surface = r[r > 0.1]
        assert abs(np.mean(r_surface) - 5.0) < 1.0

    def test_valid_faces(self):
        verts = _make_vase_verts()
        av, af = adaptive_revolve(verts, n_slices=15, n_angular=24)
        assert np.all(af >= 0)
        assert np.all(af < len(av))

    def test_z_range(self):
        verts = _make_cylinder_verts(height=30.0)
        av, af = adaptive_revolve(verts, n_slices=10)
        assert av[:, 2].min() < 2.0
        assert av[:, 2].max() > 28.0

    def test_vase_shape_preserved(self):
        verts = _make_vase_verts()
        av, af = adaptive_revolve(verts, n_slices=20, n_angular=32)
        # Extract profile from result — should vary
        r = np.sqrt(av[:, 0] ** 2 + av[:, 1] ** 2)
        r_surface = r[r > 0.1]
        assert np.ptp(r_surface) > 5.0


# =========================================================================
# 12. suggest_revolve_adjustments
# =========================================================================

class TestSuggestRevolveAdjustments:
    def test_matching_profile_no_major_issues(self):
        verts = _make_cylinder_verts(radius=5.0, height=20.0, n=2000)
        prof = _simple_profile()
        suggestions = suggest_revolve_adjustments(prof, verts)
        assert isinstance(suggestions, list)
        assert len(suggestions) >= 1
        # Should not suggest drastic changes
        actions = [s["action"] for s in suggestions]
        assert "adaptive_revolve" not in actions  # error is small

    def test_wrong_radius_suggests_refine(self):
        verts = _make_cylinder_verts(radius=10.0, height=20.0, n=2000)
        prof = [(3.0, z) for z in np.linspace(0, 20, 10)]
        suggestions = suggest_revolve_adjustments(prof, verts)
        actions = [s["action"] for s in suggestions]
        assert "refine_profile_radii" in actions

    def test_coarse_profile_suggests_detail(self):
        verts = _make_vase_verts()
        prof = [(15.0, 0.0), (15.0, 50.0), (15.0, 100.0)]
        suggestions = suggest_revolve_adjustments(prof, verts)
        actions = [s["action"] for s in suggestions]
        assert any(a in actions for a in [
            "insert_profile_detail", "refine_profile_radii",
            "adaptive_revolve", "smooth_profile_to_mesh",
        ])

    def test_priority_ordered(self):
        verts = _make_cylinder_verts()
        prof = _simple_profile()
        suggestions = suggest_revolve_adjustments(prof, verts)
        if len(suggestions) >= 2:
            priorities = [s["priority"] for s in suggestions]
            assert priorities == sorted(priorities)

    def test_all_suggestions_have_keys(self):
        verts = _make_vase_verts()
        prof = _vase_profile()
        suggestions = suggest_revolve_adjustments(prof, verts)
        for s in suggestions:
            assert "action" in s
            assert "priority" in s
            assert "params" in s
            assert "reason" in s


# =========================================================================
# Integration: end-to-end revolve improvement
# =========================================================================

class TestIntegrationRevolveImprovement:
    def test_full_pipeline(self):
        """Run the full improvement pipeline on a vase mesh."""
        verts = _make_vase_verts()
        # Start with a bad constant-radius profile
        bad_prof = [(15.0, z) for z in np.linspace(0, 100, 10)]

        # Step 1: fit to mesh
        fitted, residuals = fit_profile_to_mesh(bad_prof, verts)
        assert np.mean(residuals) > 0  # there was error

        # Step 2: insert detail where needed
        enriched = insert_profile_detail(fitted, verts, threshold=1.0)
        assert len(enriched) >= len(fitted)

        # Step 3: smooth
        smoothed = smooth_profile_to_mesh(enriched, verts, smoothing=0.3)

        # Step 4: compare final vs mesh
        mesh_prof = extract_radial_profile(verts, n_slices=30)
        metrics_before = compare_radial_profiles(
            np.array(bad_prof), mesh_prof)
        metrics_after = compare_radial_profiles(
            np.array(smoothed), mesh_prof)

        # After pipeline, error should be lower
        assert metrics_after["mean_r_error"] < metrics_before["mean_r_error"]


# =========================================================================
# Round 2 revolve differentiators and fixers
# =========================================================================

class TestProfileSmoothnessDiff:
    def test_identical_is_small(self):
        from meshxcad.revolve_align import profile_smoothness_diff
        v, _ = make_cylinder_mesh(radius=5.0, height=10.0, radial_divs=20, height_divs=10)
        assert profile_smoothness_diff(v, v) < 5.0

    def test_different_shapes(self):
        from meshxcad.revolve_align import profile_smoothness_diff
        v_cyl, _ = make_cylinder_mesh(radius=5.0, height=10.0, radial_divs=20, height_divs=10)
        v_sph, _ = make_sphere_mesh(radius=5.0, lat_divs=15, lon_divs=15)
        d = profile_smoothness_diff(v_cyl, v_sph)
        assert d >= 0.0  # just check it runs


class TestRadialAsymmetryDiff:
    def test_identical_is_small(self):
        from meshxcad.revolve_align import radial_asymmetry_diff
        v, _ = make_sphere_mesh(radius=5.0, lat_divs=15, lon_divs=15)
        assert radial_asymmetry_diff(v, v) < 5.0

    def test_asymmetric_is_larger(self):
        from meshxcad.revolve_align import radial_asymmetry_diff
        v, _ = make_sphere_mesh(radius=5.0, lat_divs=15, lon_divs=15)
        v2 = v.copy()
        v2[v2[:, 0] > 0, 0] *= 1.5
        assert radial_asymmetry_diff(v, v2) > 1.0


class TestFixProfileSmoothness:
    def test_preserves_shape(self):
        from meshxcad.revolve_align import fix_profile_smoothness
        v, f = make_cylinder_mesh(radius=5.0, height=10.0, radial_divs=20, height_divs=10)
        v2 = v * 1.1
        result = fix_profile_smoothness(v2, f, v, f)
        assert result.shape == v2.shape


class TestFixRadialAsymmetry:
    def test_preserves_shape(self):
        from meshxcad.revolve_align import fix_radial_asymmetry
        v, f = make_sphere_mesh(radius=5.0, lat_divs=15, lon_divs=15)
        v2 = v * 1.2
        result = fix_radial_asymmetry(v2, f, v, f)
        assert result.shape == v2.shape
