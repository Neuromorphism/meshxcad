"""Tests for sweep-along-path functionality."""

import math
import numpy as np
import pytest

from meshxcad.objects.operations import (
    compute_frenet_frames,
    sweep_along_path,
    multi_profile_loft,
    make_regular_polygon,
)
from meshxcad.extrude_align import (
    extract_mesh_skeleton,
    extract_sweep_cross_section,
    fit_sweep_to_mesh,
    compare_sweep_to_mesh,
    refine_sweep_path,
    refine_sweep_profile,
    adaptive_sweep_extrude,
    detect_sweep_candidate,
    suggest_sweep_adjustments,
)
from meshxcad.synthetic import make_cylinder_mesh, make_sphere_mesh


# =========================================================================
# Frenet frames
# =========================================================================

class TestComputeFrenetFrames:
    def test_straight_line(self):
        path = np.array([[0, 0, i] for i in range(5)], dtype=float)
        t, n, b = compute_frenet_frames(path)
        assert t.shape == (5, 3)
        # Tangent should be along Z
        np.testing.assert_allclose(np.abs(t[:, 2]), 1.0, atol=0.01)

    def test_unit_tangents(self):
        path = np.array([[i, i**2, 0] for i in range(10)], dtype=float)
        t, n, b = compute_frenet_frames(path)
        norms = np.linalg.norm(t, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-10)

    def test_orthogonal_frames(self):
        angles = np.linspace(0, np.pi, 20)
        path = np.column_stack([np.cos(angles), np.sin(angles),
                                np.zeros(20)])
        t, n, b = compute_frenet_frames(path)
        # t, n, b should be mutually orthogonal
        for i in range(len(path)):
            assert abs(np.dot(t[i], n[i])) < 0.1
            assert abs(np.dot(t[i], b[i])) < 0.1

    def test_single_segment(self):
        path = np.array([[0, 0, 0], [1, 0, 0]], dtype=float)
        t, n, b = compute_frenet_frames(path)
        assert t.shape == (2, 3)

    def test_helix(self):
        angles = np.linspace(0, 4 * np.pi, 50)
        path = np.column_stack([np.cos(angles), np.sin(angles),
                                angles / (2 * np.pi)])
        t, n, b = compute_frenet_frames(path)
        # All tangents should be unit length
        norms = np.linalg.norm(t, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-10)


# =========================================================================
# sweep_along_path
# =========================================================================

class TestSweepAlongPath:
    def test_basic_sweep(self):
        profile = make_regular_polygon(12, 1.0)
        path = np.array([[0, 0, z] for z in range(5)], dtype=float)
        v, f = sweep_along_path(profile, path)
        assert v.shape[1] == 3
        assert f.shape[1] == 3
        assert len(v) > 0
        assert len(f) > 0

    def test_vertex_count(self):
        n_prof = 16
        n_path = 10
        profile = make_regular_polygon(n_prof, 1.0)
        path = np.array([[0, 0, z] for z in range(n_path)], dtype=float)
        v, f = sweep_along_path(profile, path, caps=True)
        # n_prof * n_path ring verts + 2 cap center verts
        assert len(v) == n_prof * n_path + 2

    def test_no_caps(self):
        profile = make_regular_polygon(8, 1.0)
        path = np.array([[0, 0, z] for z in range(5)], dtype=float)
        v_caps, f_caps = sweep_along_path(profile, path, caps=True)
        v_no, f_no = sweep_along_path(profile, path, caps=False)
        assert len(v_no) < len(v_caps)
        assert len(f_no) < len(f_caps)

    def test_curved_path(self):
        profile = make_regular_polygon(12, 0.5)
        angles = np.linspace(0, np.pi / 2, 20)
        path = np.column_stack([5 * np.cos(angles),
                                5 * np.sin(angles),
                                np.zeros(20)])
        v, f = sweep_along_path(profile, path)
        assert len(v) > 0
        # Mesh should span roughly a quarter arc
        assert np.max(v[:, 0]) > 3
        assert np.max(v[:, 1]) > 3

    def test_twist(self):
        profile = np.array([[-1, -0.5], [1, -0.5], [1, 0.5], [-1, 0.5]],
                           dtype=float)
        path = np.array([[0, 0, z] for z in range(10)], dtype=float)
        v_no_twist, _ = sweep_along_path(profile, path, twist_total_deg=0)
        v_twist, _ = sweep_along_path(profile, path, twist_total_deg=90)
        # Twisted version should differ
        assert not np.allclose(v_no_twist, v_twist)

    def test_scale_function(self):
        profile = make_regular_polygon(12, 1.0)
        path = np.array([[0, 0, z] for z in range(10)], dtype=float)
        v_uniform, _ = sweep_along_path(profile, path)
        v_tapered, _ = sweep_along_path(profile, path,
                                         scale_fn=lambda t: 1.0 + t)
        # Tapered end should be larger
        end_radii_uniform = np.linalg.norm(v_uniform[-14:-2, :2], axis=1)
        end_radii_tapered = np.linalg.norm(v_tapered[-14:-2, :2], axis=1)
        assert np.mean(end_radii_tapered) > np.mean(end_radii_uniform)

    def test_resample_profile(self):
        profile = make_regular_polygon(6, 1.0)
        path = np.array([[0, 0, z] for z in range(5)], dtype=float)
        v, f = sweep_along_path(profile, path, n_profile=24)
        # Should have 24 * 5 ring verts + 2 caps
        assert len(v) == 24 * 5 + 2

    def test_valid_faces(self):
        profile = make_regular_polygon(8, 1.0)
        path = np.array([[0, 0, z] for z in range(5)], dtype=float)
        v, f = sweep_along_path(profile, path)
        assert np.all(f >= 0)
        assert np.all(f < len(v))

    def test_3d_path(self):
        """Sweep along a helix in 3D."""
        angles = np.linspace(0, 2 * np.pi, 30)
        path = np.column_stack([3 * np.cos(angles),
                                3 * np.sin(angles),
                                np.linspace(0, 10, 30)])
        profile = make_regular_polygon(8, 0.5)
        v, f = sweep_along_path(profile, path)
        assert len(v) > 0
        # Should span all 3 axes
        assert np.ptp(v[:, 0]) > 4
        assert np.ptp(v[:, 1]) > 4
        assert np.ptp(v[:, 2]) > 8


# =========================================================================
# multi_profile_loft
# =========================================================================

class TestMultiProfileLoft:
    def test_two_profiles(self):
        p1 = make_regular_polygon(16, 1.0)
        p2 = make_regular_polygon(16, 2.0)
        pos = np.array([[0, 0, 0], [0, 0, 5]], dtype=float)
        v, f = multi_profile_loft([p1, p2], pos, n_profile=16)
        assert len(v) > 0
        assert len(f) > 0

    def test_three_profiles(self):
        p1 = make_regular_polygon(12, 1.0)
        p2 = make_regular_polygon(12, 3.0)
        p3 = make_regular_polygon(12, 1.0)
        pos = np.array([[0, 0, 0], [0, 0, 5], [0, 0, 10]], dtype=float)
        v, f = multi_profile_loft([p1, p2, p3], pos, n_profile=12)
        assert len(v) > 0
        # Middle section should be wider
        mid_verts = v[np.abs(v[:, 2] - 5) < 1.5]
        end_verts = v[np.abs(v[:, 2] - 10) < 1.5]
        if len(mid_verts) > 0 and len(end_verts) > 0:
            mid_r = np.mean(np.linalg.norm(mid_verts[:, :2], axis=1))
            end_r = np.mean(np.linalg.norm(end_verts[:, :2], axis=1))
            assert mid_r > end_r

    def test_valid_mesh(self):
        profiles = [make_regular_polygon(8, r) for r in [1, 2, 1.5]]
        pos = np.array([[0, 0, 0], [0, 0, 3], [0, 0, 6]], dtype=float)
        v, f = multi_profile_loft(profiles, pos, n_profile=8, smooth=True)
        assert np.all(f >= 0)
        assert np.all(f < len(v))

    def test_no_smooth(self):
        profiles = [make_regular_polygon(8, r) for r in [1, 2]]
        pos = np.array([[0, 0, 0], [0, 0, 5]], dtype=float)
        v_smooth, _ = multi_profile_loft(profiles, pos, smooth=True)
        v_no, _ = multi_profile_loft(profiles, pos, smooth=False)
        # Smooth version has more rings
        assert len(v_smooth) > len(v_no)


# =========================================================================
# extract_mesh_skeleton
# =========================================================================

class TestExtractMeshSkeleton:
    def test_cylinder(self):
        v, f = make_cylinder_mesh(radius=3.0, height=15.0,
                                   radial_divs=24, height_divs=15)
        skel, axis = extract_mesh_skeleton(v, f, n_points=10)
        assert skel.shape == (10, 3)
        assert len(axis) == 3
        # For a Z-aligned cylinder, axis should be near Z
        assert abs(axis[2]) > 0.8

    def test_skeleton_spans_mesh(self):
        v, f = make_cylinder_mesh(radius=2.0, height=20.0)
        skel, _ = extract_mesh_skeleton(v, f, n_points=15)
        # Skeleton should span most of the mesh height
        assert np.ptp(skel[:, 2]) > 15


# =========================================================================
# extract_sweep_cross_section
# =========================================================================

class TestExtractSweepCrossSection:
    def test_cylinder_cross_section(self):
        v, f = make_cylinder_mesh(radius=3.0, height=15.0,
                                   radial_divs=24, height_divs=15)
        cs, frame = extract_sweep_cross_section(
            v, f, point=np.array([0, 0, 7.5]),
            tangent=np.array([0, 0, 1]), n_output=16)
        assert cs.shape[0] > 0
        assert cs.shape[1] == 2
        assert "normal" in frame
        assert "binormal" in frame

    def test_cross_section_radius(self):
        v, f = make_cylinder_mesh(radius=5.0, height=10.0,
                                   radial_divs=32, height_divs=10)
        cs, _ = extract_sweep_cross_section(
            v, f, point=np.array([0, 0, 5.0]),
            tangent=np.array([0, 0, 1]), n_output=32)
        if len(cs) >= 3:
            radii = np.linalg.norm(cs - cs.mean(axis=0), axis=1)
            mean_r = float(np.mean(radii))
            assert abs(mean_r - 5.0) < 2.0  # within reasonable range


# =========================================================================
# fit_sweep_to_mesh
# =========================================================================

class TestFitSweepToMesh:
    def test_returns_valid_structure(self):
        v, f = make_cylinder_mesh(radius=3.0, height=15.0,
                                   radial_divs=24, height_divs=15)
        fit = fit_sweep_to_mesh(v, f, n_path_points=8, n_profile=12)
        assert "path" in fit
        assert "profile" in fit
        assert "profiles" in fit
        assert "scales" in fit
        assert fit["path"].shape == (8, 3)

    def test_profile_has_points(self):
        v, f = make_cylinder_mesh(radius=3.0, height=15.0,
                                   radial_divs=24, height_divs=15)
        fit = fit_sweep_to_mesh(v, f, n_path_points=10, n_profile=16)
        assert fit["profile"].shape[1] == 2
        assert len(fit["profile"]) > 0


# =========================================================================
# compare_sweep_to_mesh
# =========================================================================

class TestCompareSweepToMesh:
    def test_identical(self):
        v, f = make_cylinder_mesh(radius=3.0, height=10.0)
        metrics = compare_sweep_to_mesh(v, f, v, f)
        assert metrics["hausdorff"] < 0.01
        assert metrics["mean_dist"] < 0.01
        assert metrics["coverage"] > 0.99
        assert metrics["quality_score"] > 90

    def test_offset_mesh(self):
        v, f = make_cylinder_mesh(radius=3.0, height=10.0)
        v2 = v + 2.0
        metrics = compare_sweep_to_mesh(v, f, v2, f)
        assert metrics["mean_dist"] > 1.0
        assert metrics["quality_score"] < 90


# =========================================================================
# refine_sweep_path
# =========================================================================

class TestRefineSweepPath:
    def test_preserves_shape(self):
        v, f = make_cylinder_mesh(radius=3.0, height=15.0,
                                   radial_divs=24, height_divs=15)
        path = np.array([[0, 0, z] for z in np.linspace(0, 15, 10)])
        refined = refine_sweep_path(path, v, f, iterations=2)
        assert refined.shape == path.shape

    def test_endpoints_fixed(self):
        v, f = make_cylinder_mesh(radius=3.0, height=15.0)
        path = np.array([[0, 0, z] for z in np.linspace(0, 15, 8)])
        refined = refine_sweep_path(path, v, f, iterations=3)
        np.testing.assert_allclose(refined[0], path[0], atol=0.01)
        np.testing.assert_allclose(refined[-1], path[-1], atol=0.01)


# =========================================================================
# refine_sweep_profile
# =========================================================================

class TestRefineSweepProfile:
    def test_returns_same_shape(self):
        v, f = make_cylinder_mesh(radius=3.0, height=15.0,
                                   radial_divs=24, height_divs=15)
        profile = np.column_stack([3 * np.cos(np.linspace(0, 2*np.pi, 16, endpoint=False)),
                                   3 * np.sin(np.linspace(0, 2*np.pi, 16, endpoint=False))])
        path = np.array([[0, 0, z] for z in np.linspace(0, 15, 10)])
        refined = refine_sweep_profile(profile, path, v, f, n_samples=3)
        assert refined.shape == profile.shape


# =========================================================================
# adaptive_sweep_extrude
# =========================================================================

class TestAdaptiveSweepExtrude:
    def test_produces_mesh(self):
        v, f = make_cylinder_mesh(radius=3.0, height=15.0,
                                   radial_divs=24, height_divs=15)
        sv, sf, info = adaptive_sweep_extrude(v, f, n_path=8, n_profile=12)
        assert len(sv) > 0
        assert len(sf) > 0
        assert "quality" in info
        assert info["quality"]["quality_score"] >= 0

    def test_quality_positive(self):
        v, f = make_cylinder_mesh(radius=3.0, height=15.0,
                                   radial_divs=24, height_divs=15)
        _, _, info = adaptive_sweep_extrude(v, f, n_path=10, n_profile=16)
        assert info["quality"]["quality_score"] > 0
        assert info["quality"]["coverage"] > 0.5


# =========================================================================
# detect_sweep_candidate
# =========================================================================

class TestDetectSweepCandidate:
    def test_cylinder_is_sweep(self):
        v, f = make_cylinder_mesh(radius=3.0, height=15.0,
                                   radial_divs=24, height_divs=15)
        det = detect_sweep_candidate(v, f)
        assert det["is_sweep"] is True
        assert det["elongation"] > 1.5
        assert det["consistency"] > 0.5
        assert "axis" in det

    def test_sphere_low_elongation(self):
        v, f = make_sphere_mesh(radius=5.0, lat_divs=15, lon_divs=15)
        det = detect_sweep_candidate(v, f)
        # Sphere has low elongation
        assert det["elongation"] < 2.0

    def test_returns_all_keys(self):
        v, f = make_cylinder_mesh(radius=3.0, height=10.0)
        det = detect_sweep_candidate(v, f)
        for key in ["is_sweep", "elongation", "consistency", "axis",
                     "confidence"]:
            assert key in det


# =========================================================================
# suggest_sweep_adjustments
# =========================================================================

class TestSuggestSweepAdjustments:
    def test_identical_no_action(self):
        v, f = make_cylinder_mesh(radius=3.0, height=10.0)
        sugg = suggest_sweep_adjustments(v, f, v, f)
        assert isinstance(sugg, list)
        assert len(sugg) >= 1

    def test_offset_suggests_fixes(self):
        v, f = make_cylinder_mesh(radius=3.0, height=10.0)
        v2 = v + 5.0
        sugg = suggest_sweep_adjustments(v2, f, v, f)
        actions = [s["action"] for s in sugg]
        assert len(sugg) >= 1
        # Should suggest some action
        assert sugg[0]["action"] != "none"

    def test_priority_ordered(self):
        v, f = make_cylinder_mesh(radius=3.0, height=10.0)
        v2 = v * 2.0
        sugg = suggest_sweep_adjustments(v2, f, v, f)
        if len(sugg) >= 2:
            prios = [s["priority"] for s in sugg]
            assert prios == sorted(prios)

    def test_all_keys_present(self):
        v, f = make_cylinder_mesh(radius=3.0, height=10.0)
        sugg = suggest_sweep_adjustments(v, f, v, f)
        for s in sugg:
            assert "action" in s
            assert "priority" in s
            assert "params" in s
            assert "reason" in s
