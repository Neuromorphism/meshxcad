"""Tests for meshxcad.extrude_align — 12 extrude-to-mesh alignment features."""

import math
import numpy as np
import pytest

from meshxcad.extrude_align import (
    extract_cross_section,
    compare_cross_sections,
    fit_primitive_to_cross_section,
    fit_ellipse_to_cross_section,
    offset_profile,
    blend_profiles,
    taper_extrude,
    twist_extrude,
    boolean_union_profiles,
    boolean_difference_profiles,
    adaptive_extrude,
    suggest_extrude_adjustments,
    fit_spline_profile,
)
from meshxcad.objects.operations import extrude_polygon, make_regular_polygon
from meshxcad.synthetic import make_cylinder_mesh, make_cube_mesh


# ---- helpers ----

def _make_circle_polygon(r=5.0, n=32, cx=0.0, cy=0.0):
    t = np.linspace(0, 2 * math.pi, n, endpoint=False)
    return np.column_stack([cx + r * np.cos(t), cy + r * np.sin(t)])


def _make_square_polygon(side=10.0, cx=0.0, cy=0.0):
    h = side / 2
    return np.array([
        [cx - h, cy - h], [cx + h, cy - h],
        [cx + h, cy + h], [cx - h, cy + h],
    ])


# =========================================================================
# 1. extract_cross_section
# =========================================================================

class TestExtractCrossSection:
    def test_cylinder_mid_height(self):
        """Slicing a cylinder at mid-height should yield a roughly circular cross-section."""
        verts, faces = make_cylinder_mesh(radius=5.0, height=10.0,
                                          radial_divs=32, height_divs=20)
        cs = extract_cross_section(verts, faces, z_height=0.0)
        assert len(cs) >= 3
        # Points should be near radius 5
        radii = np.linalg.norm(cs, axis=1)
        assert np.mean(np.abs(radii - 5.0)) < 0.5

    def test_cube_mid_height(self):
        """Slicing a cube at z=0 should yield a square-ish cross-section."""
        verts, faces = make_cube_mesh(size=10.0, subdivisions=6)
        cs = extract_cross_section(verts, faces, z_height=0.0)
        assert len(cs) >= 4

    def test_no_intersection(self):
        """Slicing above the mesh should return empty."""
        verts, faces = make_cylinder_mesh(radius=5.0, height=10.0)
        cs = extract_cross_section(verts, faces, z_height=100.0)
        assert len(cs) == 0

    def test_extruded_hex(self):
        """Slicing an extruded hexagon should return ~6 points."""
        hex_poly = make_regular_polygon(6, 10.0)
        verts, faces = extrude_polygon(hex_poly, height=20.0, n_height=4)
        cs = extract_cross_section(verts, faces, z_height=10.0)
        assert len(cs) >= 6


# =========================================================================
# 2. compare_cross_sections
# =========================================================================

class TestCompareCrossSections:
    def test_identical_profiles(self):
        circ = _make_circle_polygon(5.0)
        metrics = compare_cross_sections(circ, circ)
        assert metrics["hausdorff"] < 0.01
        assert metrics["mean_dist"] < 0.01
        assert abs(metrics["area_ratio"] - 1.0) < 0.01

    def test_different_sizes(self):
        small = _make_circle_polygon(3.0)
        big = _make_circle_polygon(6.0)
        metrics = compare_cross_sections(small, big)
        assert metrics["area_ratio"] < 1.0
        assert metrics["hausdorff"] > 2.0

    def test_circle_vs_square(self):
        circ = _make_circle_polygon(5.0)
        sq = _make_square_polygon(10.0)
        metrics = compare_cross_sections(circ, sq)
        # Shapes differ → hausdorff > 0
        assert metrics["hausdorff"] > 0.1
        # Areas should be roughly similar (pi*25 ≈ 78.5 vs 100)
        assert 0.5 < metrics["area_ratio"] < 1.5

    def test_iou_between_0_and_1(self):
        a = _make_circle_polygon(5.0)
        b = _make_circle_polygon(5.0, cx=2.0)
        metrics = compare_cross_sections(a, b)
        assert 0 <= metrics["iou"] <= 1.0


# =========================================================================
# 3. fit_primitive_to_cross_section
# =========================================================================

class TestFitPrimitive:
    def test_fits_circle(self):
        circ = _make_circle_polygon(5.0, n=64)
        # Add slight noise
        noisy = circ + np.random.RandomState(42).randn(*circ.shape) * 0.05
        result = fit_primitive_to_cross_section(noisy)
        assert result["best"] == "circle"
        assert result["residual"] < 0.3
        assert abs(result["params"]["radius"] - 5.0) < 0.5

    def test_fits_hexagon(self):
        hex_pts = np.array(make_regular_polygon(6, 10.0))
        result = fit_primitive_to_cross_section(hex_pts)
        # Should recognize polygon_6 or circle (hexagon is close to both)
        assert "polygon" in result["best"] or result["best"] == "circle"
        assert result["residual"] < 2.0

    def test_all_fits_populated(self):
        sq = _make_square_polygon(10.0)
        result = fit_primitive_to_cross_section(sq)
        # circle + rectangle + polygon_3..8 = 8 candidates
        assert len(result["all_fits"]) == 8
        assert all(isinstance(f[1], float) for f in result["all_fits"])

    def test_polygon_returned(self):
        circ = _make_circle_polygon(5.0)
        result = fit_primitive_to_cross_section(circ)
        assert result["polygon"].shape[1] == 2
        assert len(result["polygon"]) >= 3


# =========================================================================
# 4. fit_ellipse_to_cross_section
# =========================================================================

class TestFitEllipse:
    def test_circle_is_equal_axes(self):
        circ = _make_circle_polygon(7.0, n=64)
        result = fit_ellipse_to_cross_section(circ)
        a, b = result["semi_axes"]
        assert abs(a - b) < 0.5
        assert abs(a - 7.0) < 0.5

    def test_ellipse_recovery(self):
        t = np.linspace(0, 2 * math.pi, 64, endpoint=False)
        a_true, b_true = 8.0, 4.0
        pts = np.column_stack([a_true * np.cos(t), b_true * np.sin(t)])
        result = fit_ellipse_to_cross_section(pts)
        a, b = sorted(result["semi_axes"], reverse=True)
        assert abs(a - 8.0) < 0.5
        assert abs(b - 4.0) < 0.5
        assert result["residual"] < 0.5

    def test_polygon_shape(self):
        circ = _make_circle_polygon(5.0)
        result = fit_ellipse_to_cross_section(circ)
        assert result["polygon"].shape == (64, 2)


# =========================================================================
# 5. offset_profile
# =========================================================================

class TestOffsetProfile:
    def test_outward_expansion(self):
        sq = _make_square_polygon(10.0)
        expanded = offset_profile(sq, 1.0)
        # Bounding box should be larger
        orig_span = np.ptp(sq, axis=0)
        exp_span = np.ptp(expanded, axis=0)
        assert exp_span[0] > orig_span[0]
        assert exp_span[1] > orig_span[1]

    def test_inward_shrinkage(self):
        sq = _make_square_polygon(10.0)
        shrunk = offset_profile(sq, -1.0)
        # Bounding box should be smaller
        orig_span = np.ptp(sq, axis=0)
        shrunk_span = np.ptp(shrunk, axis=0)
        assert shrunk_span[0] < orig_span[0]
        assert shrunk_span[1] < orig_span[1]

    def test_zero_offset_unchanged(self):
        sq = _make_square_polygon(10.0)
        result = offset_profile(sq, 0.0)
        np.testing.assert_allclose(result, sq, atol=1e-10)

    def test_preserves_shape(self):
        circ = _make_circle_polygon(5.0, n=32)
        result = offset_profile(circ, 2.0)
        # Result should still be roughly circular
        radii = np.linalg.norm(result, axis=1)
        assert np.std(radii) / np.mean(radii) < 0.15


# =========================================================================
# 6. blend_profiles
# =========================================================================

class TestBlendProfiles:
    def test_t0_is_profile_a(self):
        a = _make_circle_polygon(5.0, n=16)
        b = _make_circle_polygon(10.0, n=16)
        result = blend_profiles(a, b, t=0.0, n_sample=16)
        # Should be close to a
        np.testing.assert_allclose(result, a, atol=0.5)

    def test_t1_is_profile_b(self):
        a = _make_circle_polygon(5.0, n=16)
        b = _make_circle_polygon(10.0, n=16)
        result = blend_profiles(a, b, t=1.0, n_sample=16)
        radii = np.linalg.norm(result, axis=1)
        assert np.mean(radii) > 8.0

    def test_t05_is_midpoint(self):
        a = _make_circle_polygon(4.0, n=32)
        b = _make_circle_polygon(8.0, n=32)
        result = blend_profiles(a, b, t=0.5, n_sample=32)
        radii = np.linalg.norm(result, axis=1)
        assert abs(np.mean(radii) - 6.0) < 0.5

    def test_different_point_counts(self):
        a = _make_circle_polygon(5.0, n=16)
        b = _make_circle_polygon(5.0, n=48)
        result = blend_profiles(a, b, t=0.5, n_sample=32)
        assert result.shape == (32, 2)


# =========================================================================
# 7. taper_extrude
# =========================================================================

class TestTaperExtrude:
    def test_no_taper(self):
        sq = _make_square_polygon(10.0).tolist()
        verts, faces = taper_extrude(sq, height=20.0, scale_top=1.0)
        # Top and bottom should have same bounding box width
        z = verts[:, 2]
        bot_verts = verts[z < 0.1]
        top_verts = verts[z > 19.9]
        assert abs(np.ptp(bot_verts[:, 0]) - np.ptp(top_verts[:, 0])) < 0.1

    def test_taper_shrinks_top(self):
        sq = _make_square_polygon(10.0).tolist()
        verts, faces = taper_extrude(sq, height=20.0, scale_top=0.5)
        z = verts[:, 2]
        bot_verts = verts[z < 0.1]
        top_verts = verts[(z > 19.9) & (z < 20.1)]
        # Exclude cap center from check
        if len(top_verts) > 1:
            assert np.ptp(top_verts[:, 0]) < np.ptp(bot_verts[:, 0])

    def test_taper_expands_top(self):
        sq = _make_square_polygon(10.0).tolist()
        verts, faces = taper_extrude(sq, height=20.0, scale_top=1.5)
        z = verts[:, 2]
        bot_verts = verts[z < 0.1]
        top_verts = verts[(z > 19.9) & (z < 20.1)]
        if len(top_verts) > 1:
            assert np.ptp(top_verts[:, 0]) > np.ptp(bot_verts[:, 0])

    def test_valid_mesh(self):
        hex_poly = make_regular_polygon(6, 10.0)
        verts, faces = taper_extrude(hex_poly, height=15.0, scale_top=0.7, n_height=8)
        assert verts.shape[1] == 3
        assert faces.shape[1] == 3
        assert np.all(faces >= 0)
        assert np.all(faces < len(verts))


# =========================================================================
# 8. twist_extrude
# =========================================================================

class TestTwistExtrude:
    def test_zero_twist(self):
        sq = _make_square_polygon(10.0).tolist()
        verts, faces = twist_extrude(sq, height=20.0, total_twist_deg=0.0)
        # Bottom and top should have same angles
        z = verts[:, 2]
        bot = verts[z < 0.1]
        top = verts[(z > 19.9) & (z < 20.1)]
        # Angles should match
        a_bot = np.sort(np.arctan2(bot[:, 1], bot[:, 0]))
        a_top = np.sort(np.arctan2(top[:, 1], top[:, 0]))
        if len(a_bot) == len(a_top):
            np.testing.assert_allclose(a_bot, a_top, atol=0.01)

    def test_90_degree_twist(self):
        sq = _make_square_polygon(10.0).tolist()
        verts, faces = twist_extrude(sq, height=20.0, total_twist_deg=90.0, n_height=20)
        # Valid mesh
        assert np.all(faces >= 0)
        assert np.all(faces < len(verts))
        # Z range should be 0 to 20
        assert verts[:, 2].min() < 0.1
        assert verts[:, 2].max() > 19.9

    def test_triangle_twist(self):
        tri = make_regular_polygon(3, 5.0)
        verts, faces = twist_extrude(tri, height=10.0, total_twist_deg=60.0)
        assert verts.shape[1] == 3
        assert len(faces) > 0


# =========================================================================
# 9. boolean_union_profiles
# =========================================================================

class TestBooleanUnionProfiles:
    def test_identical_profiles(self):
        circ = _make_circle_polygon(5.0)
        result = boolean_union_profiles(circ, circ)
        assert len(result) >= 3

    def test_overlapping_profiles(self):
        a = _make_circle_polygon(5.0)
        b = _make_circle_polygon(5.0, cx=3.0)
        result = boolean_union_profiles(a, b)
        # Union should span wider than either alone
        x_span = np.ptp(result[:, 0])
        assert x_span > 10.0  # wider than single circle diameter

    def test_disjoint_profiles(self):
        a = _make_circle_polygon(3.0, cx=-10)
        b = _make_circle_polygon(3.0, cx=10)
        result = boolean_union_profiles(a, b)
        assert np.ptp(result[:, 0]) > 15.0


# =========================================================================
# 10. boolean_difference_profiles
# =========================================================================

class TestBooleanDifferenceProfiles:
    def test_no_overlap(self):
        outer = _make_circle_polygon(10.0)
        inner = _make_circle_polygon(3.0, cx=50.0)  # far away
        result = boolean_difference_profiles(outer, inner)
        # Should be unchanged
        assert len(result) > 0

    def test_concentric_cut(self):
        outer = _make_circle_polygon(10.0, n=64)
        inner = _make_circle_polygon(3.0, n=16)
        result = boolean_difference_profiles(outer, inner)
        # Points that were inside inner should now be on inner boundary
        inner_pts = result[np.linalg.norm(result, axis=1) < 3.5]
        if len(inner_pts) > 0:
            radii = np.linalg.norm(inner_pts, axis=1)
            assert np.all(radii >= 2.5)  # pushed to ~3.0


# =========================================================================
# 11. adaptive_extrude
# =========================================================================

class TestAdaptiveExtrude:
    def test_cylinder_reconstruction(self):
        """Adaptive extrude of a cylinder should produce a cylinder-like mesh."""
        verts, faces = make_cylinder_mesh(radius=5.0, height=10.0,
                                          radial_divs=32, height_divs=20)
        loft_v, loft_f = adaptive_extrude(verts, faces,
                                           z_bottom=-4.0, z_top=4.0,
                                           n_slices=8, n_profile=16)
        assert loft_v.shape[1] == 3
        assert loft_f.shape[1] == 3
        # Z range should span requested range
        assert loft_v[:, 2].min() <= -3.9
        assert loft_v[:, 2].max() >= 3.9

    def test_valid_faces(self):
        verts, faces = make_cylinder_mesh(radius=5.0, height=10.0)
        loft_v, loft_f = adaptive_extrude(verts, faces,
                                           z_bottom=-3.0, z_top=3.0,
                                           n_slices=5, n_profile=12)
        assert np.all(loft_f >= 0)
        assert np.all(loft_f < len(loft_v))

    def test_small_n_slices(self):
        verts, faces = make_cylinder_mesh(radius=5.0, height=10.0)
        loft_v, loft_f = adaptive_extrude(verts, faces,
                                           z_bottom=-2.0, z_top=2.0,
                                           n_slices=2, n_profile=8)
        assert len(loft_v) > 0
        assert len(loft_f) > 0


# =========================================================================
# 12. suggest_extrude_adjustments
# =========================================================================

class TestSuggestExtrudeAdjustments:
    def test_matching_profile_few_suggestions(self):
        """A circle extrude vs a cylinder mesh should have minimal suggestions."""
        verts, faces = make_cylinder_mesh(radius=5.0, height=10.0,
                                          radial_divs=32, height_divs=20)
        circ = _make_circle_polygon(5.0, n=32)
        suggestions = suggest_extrude_adjustments(circ, verts, faces,
                                                   z_bottom=-4.0, z_top=4.0)
        assert isinstance(suggestions, list)
        # Should not suggest drastic changes
        actions = [s["action"] for s in suggestions]
        # offset shouldn't be needed (area ratio ~1)
        for s in suggestions:
            if s["action"] == "offset_profile":
                assert abs(s["params"]["distance"]) < 2.0

    def test_size_mismatch(self):
        """Small circle profile vs large cylinder should suggest offset."""
        verts, faces = make_cylinder_mesh(radius=10.0, height=10.0,
                                          radial_divs=32, height_divs=20)
        small = _make_circle_polygon(3.0, n=32)
        suggestions = suggest_extrude_adjustments(small, verts, faces,
                                                   z_bottom=-4.0, z_top=4.0)
        actions = [s["action"] for s in suggestions]
        assert "offset_profile" in actions or "adaptive_extrude" in actions

    def test_returns_priority_ordered(self):
        verts, faces = make_cylinder_mesh(radius=5.0, height=10.0)
        circ = _make_circle_polygon(5.0)
        suggestions = suggest_extrude_adjustments(circ, verts, faces,
                                                   z_bottom=-4.0, z_top=4.0)
        if len(suggestions) >= 2:
            priorities = [s["priority"] for s in suggestions]
            assert priorities == sorted(priorities)

    def test_all_suggestions_have_keys(self):
        verts, faces = make_cube_mesh(size=10.0, subdivisions=4)
        sq = _make_square_polygon(8.0)
        suggestions = suggest_extrude_adjustments(sq, verts, faces,
                                                   z_bottom=-3.0, z_top=3.0)
        for s in suggestions:
            assert "action" in s
            assert "priority" in s
            assert "params" in s
            assert "reason" in s


# =========================================================================
# fit_spline_profile
# =========================================================================

class TestFitSplineProfile:
    def test_circle_stays_circular(self):
        circ = _make_circle_polygon(5.0, n=64)
        result = fit_spline_profile(circ, n_control=16, n_output=64)
        radii = np.linalg.norm(result, axis=1)
        assert abs(np.mean(radii) - 5.0) < 0.3
        assert np.std(radii) < 0.3

    def test_output_size(self):
        pts = _make_circle_polygon(5.0, n=20)
        result = fit_spline_profile(pts, n_control=10, n_output=48)
        assert result.shape == (48, 2)

    def test_irregular_shape(self):
        # Star shape
        t = np.linspace(0, 2 * math.pi, 20, endpoint=False)
        r = 5.0 + 2.0 * np.sin(5 * t)
        pts = np.column_stack([r * np.cos(t), r * np.sin(t)])
        result = fit_spline_profile(pts, n_control=16, n_output=64)
        assert result.shape == (64, 2)
        # Should preserve general size
        radii = np.linalg.norm(result, axis=1)
        assert np.mean(radii) > 3.0
        assert np.mean(radii) < 8.0


# =========================================================================
# Round 2 extrude/sweep differentiators and fixers
# =========================================================================

class TestTaperConsistencyDiff:
    def test_identical_is_small(self):
        from meshxcad.extrude_align import taper_consistency_diff
        v, f = make_cylinder_mesh(radius=5.0, height=10.0,
                                   radial_divs=20, height_divs=10)
        assert taper_consistency_diff(v, f, v, f) < 5.0


class TestSweepPathDeviation:
    def test_identical_is_small(self):
        from meshxcad.extrude_align import sweep_path_deviation
        v, f = make_cylinder_mesh(radius=5.0, height=10.0,
                                   radial_divs=20, height_divs=10)
        assert sweep_path_deviation(v, f, v, f) < 5.0


class TestProfileCircularityDiff:
    def test_identical_is_small(self):
        from meshxcad.extrude_align import profile_circularity_diff
        v, f = make_cylinder_mesh(radius=5.0, height=10.0,
                                   radial_divs=20, height_divs=10)
        assert profile_circularity_diff(v, f, v, f) < 5.0


class TestExtrudeTwistDiff:
    def test_identical_is_small(self):
        from meshxcad.extrude_align import extrude_twist_diff
        v, f = make_cylinder_mesh(radius=5.0, height=10.0,
                                   radial_divs=20, height_divs=10)
        assert extrude_twist_diff(v, f, v, f) < 5.0


class TestRound2ExtrudeFixers:
    def _make_pair(self):
        v, f = make_cylinder_mesh(radius=5.0, height=10.0,
                                   radial_divs=20, height_divs=10)
        v2 = v * 1.2 + 0.5
        return v2, f, v, f

    def test_fix_taper_consistency(self):
        from meshxcad.extrude_align import fix_taper_consistency
        cad_v, cad_f, mesh_v, mesh_f = self._make_pair()
        result = fix_taper_consistency(cad_v, cad_f, mesh_v, mesh_f)
        assert result.shape == cad_v.shape

    def test_fix_sweep_path_deviation(self):
        from meshxcad.extrude_align import fix_sweep_path_deviation
        cad_v, cad_f, mesh_v, mesh_f = self._make_pair()
        result = fix_sweep_path_deviation(cad_v, cad_f, mesh_v, mesh_f)
        assert result.shape == cad_v.shape

    def test_fix_profile_circularity(self):
        from meshxcad.extrude_align import fix_profile_circularity
        cad_v, cad_f, mesh_v, mesh_f = self._make_pair()
        result = fix_profile_circularity(cad_v, cad_f, mesh_v, mesh_f)
        assert result.shape == cad_v.shape

    def test_fix_extrude_twist(self):
        from meshxcad.extrude_align import fix_extrude_twist
        cad_v, cad_f, mesh_v, mesh_f = self._make_pair()
        result = fix_extrude_twist(cad_v, cad_f, mesh_v, mesh_f)
        assert result.shape == cad_v.shape
