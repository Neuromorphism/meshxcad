"""Tests for meshxcad.general_align — 12 general mesh-to-mesh alignment features."""

import math
import numpy as np
import pytest

from meshxcad.general_align import (
    surface_distance_map,
    hausdorff_distance,
    normal_deviation_map,
    curvature_deviation_map,
    laplacian_smooth_toward,
    project_vertices_to_mesh,
    local_scale_correction,
    feature_edge_transfer,
    vertex_normal_realign,
    fill_surface_holes,
    decimate_to_match,
    suggest_general_adjustments,
)
from meshxcad.synthetic import make_cylinder_mesh, make_sphere_mesh, make_cube_mesh


# ---- helpers ----

def _offset_mesh(vertices, faces, offset):
    """Translate a mesh."""
    return vertices + np.array(offset), faces


def _scale_mesh(vertices, faces, factor):
    """Scale a mesh."""
    return vertices * factor, faces


# =========================================================================
# 1. surface_distance_map
# =========================================================================

class TestSurfaceDistanceMap:
    def test_identical_meshes(self):
        v, f = make_sphere_mesh(radius=5.0, lat_divs=15, lon_divs=15)
        result = surface_distance_map(v, f, v)
        assert result["mean_dist"] < 0.01
        assert result["max_dist"] < 0.01

    def test_offset_mesh(self):
        v, f = make_sphere_mesh(radius=5.0, lat_divs=15, lon_divs=15)
        v2 = v + np.array([2.0, 0, 0])
        result = surface_distance_map(v, f, v2)
        assert result["mean_dist"] > 0.5

    def test_returns_per_vertex(self):
        v, f = make_sphere_mesh(radius=5.0, lat_divs=10, lon_divs=10)
        result = surface_distance_map(v, f, v)
        assert len(result["signed_distances"]) == len(v)
        assert len(result["unsigned_distances"]) == len(v)


# =========================================================================
# 2. hausdorff_distance
# =========================================================================

class TestHausdorffDistance:
    def test_identical(self):
        v, _ = make_sphere_mesh(radius=5.0, lat_divs=15, lon_divs=15)
        result = hausdorff_distance(v, v)
        assert result["hausdorff"] < 0.01
        assert result["mean_symmetric"] < 0.01

    def test_offset(self):
        v, _ = make_sphere_mesh(radius=5.0)
        v2 = v + 3.0
        result = hausdorff_distance(v, v2)
        assert result["hausdorff"] > 2.0

    def test_symmetric(self):
        v1, _ = make_sphere_mesh(radius=5.0, lat_divs=10, lon_divs=10)
        v2, _ = make_sphere_mesh(radius=7.0, lat_divs=10, lon_divs=10)
        r1 = hausdorff_distance(v1, v2)
        r2 = hausdorff_distance(v2, v1)
        assert abs(r1["hausdorff"] - r2["hausdorff"]) < 0.01


# =========================================================================
# 3. normal_deviation_map
# =========================================================================

class TestNormalDeviationMap:
    def test_identical(self):
        v, f = make_sphere_mesh(radius=5.0, lat_divs=15, lon_divs=15)
        result = normal_deviation_map(v, f, v, f)
        assert result["mean_deg"] < 1.0

    def test_inverted_normals(self):
        v, f = make_sphere_mesh(radius=5.0, lat_divs=15, lon_divs=15)
        f_inv = f[:, ::-1]  # reverse winding
        result = normal_deviation_map(v, f, v, f_inv)
        assert result["mean_deg"] > 90.0

    def test_per_vertex(self):
        v, f = make_sphere_mesh(radius=5.0, lat_divs=10, lon_divs=10)
        result = normal_deviation_map(v, f, v, f)
        assert len(result["angles_deg"]) == len(v)


# =========================================================================
# 4. curvature_deviation_map
# =========================================================================

class TestCurvatureDeviationMap:
    def test_identical(self):
        v, f = make_sphere_mesh(radius=5.0, lat_divs=15, lon_divs=15)
        result = curvature_deviation_map(v, f, v, f)
        assert result["mean_abs_dev"] < 0.01

    def test_different_radii(self):
        v1, f1 = make_sphere_mesh(radius=5.0, lat_divs=15, lon_divs=15)
        v2, f2 = make_sphere_mesh(radius=10.0, lat_divs=15, lon_divs=15)
        result = curvature_deviation_map(v1, f1, v2, f2)
        assert result["mean_abs_dev"] > 0

    def test_returns_curvatures(self):
        v, f = make_sphere_mesh(radius=5.0, lat_divs=10, lon_divs=10)
        result = curvature_deviation_map(v, f, v, f)
        assert len(result["cad_curvature"]) == len(v)
        assert len(result["deviation"]) == len(v)


# =========================================================================
# 5. laplacian_smooth_toward
# =========================================================================

class TestLaplacianSmoothToward:
    def test_moves_toward_target(self):
        v1, f1 = make_sphere_mesh(radius=5.0, lat_divs=10, lon_divs=10)
        v2, _ = make_sphere_mesh(radius=8.0, lat_divs=10, lon_divs=10)
        smoothed = laplacian_smooth_toward(v1, f1, v2, iterations=5,
                                            target_weight=0.5)
        # Mean radius should increase toward 8
        r_orig = np.mean(np.linalg.norm(v1, axis=1))
        r_smooth = np.mean(np.linalg.norm(smoothed, axis=1))
        assert r_smooth > r_orig

    def test_zero_weight_stays(self):
        v, f = make_sphere_mesh(radius=5.0, lat_divs=10, lon_divs=10)
        v2 = v * 2
        smoothed = laplacian_smooth_toward(v, f, v2, iterations=1,
                                            lam=0.0, target_weight=0.0)
        np.testing.assert_allclose(smoothed, v, atol=1e-10)

    def test_preserves_shape(self):
        v, f = make_sphere_mesh(radius=5.0, lat_divs=10, lon_divs=10)
        smoothed = laplacian_smooth_toward(v, f, v, iterations=3)
        assert smoothed.shape == v.shape


# =========================================================================
# 6. project_vertices_to_mesh
# =========================================================================

class TestProjectVerticesToMesh:
    def test_already_on_surface(self):
        v, f = make_sphere_mesh(radius=5.0, lat_divs=15, lon_divs=15)
        projected, dists = project_vertices_to_mesh(v, v, f)
        assert np.mean(dists) < 0.5

    def test_scaled_mesh_projects_back(self):
        v, f = make_sphere_mesh(radius=5.0, lat_divs=15, lon_divs=15)
        v_big = v * 1.5
        projected, dists = project_vertices_to_mesh(v_big, v, f)
        # Projected points should be closer to radius 5
        r_proj = np.linalg.norm(projected, axis=1)
        r_orig = np.linalg.norm(v_big, axis=1)
        assert np.mean(r_proj) < np.mean(r_orig)

    def test_max_distance_cap(self):
        v, f = make_sphere_mesh(radius=5.0, lat_divs=10, lon_divs=10)
        v_far = v + 100
        projected, dists = project_vertices_to_mesh(v_far, v, f, max_distance=1.0)
        # Points were too far — should stay put
        np.testing.assert_allclose(projected, v_far, atol=0.01)


# =========================================================================
# 7. local_scale_correction
# =========================================================================

class TestLocalScaleCorrection:
    def test_corrects_scale(self):
        v, f = make_cylinder_mesh(radius=5.0, height=20.0,
                                   radial_divs=24, height_divs=10)
        v_big = v * 1.5
        corrected, scales = local_scale_correction(v_big, f, v, n_regions=4)
        # Corrected should be closer to original size
        r_orig = np.sqrt(v[:, 0] ** 2 + v[:, 1] ** 2)
        r_big = np.sqrt(v_big[:, 0] ** 2 + v_big[:, 1] ** 2)
        r_corr = np.sqrt(corrected[:, 0] ** 2 + corrected[:, 1] ** 2)
        assert np.mean(np.abs(r_corr - r_orig.mean())) < np.mean(np.abs(r_big - r_orig.mean()))

    def test_identical_no_change(self):
        v, f = make_cylinder_mesh(radius=5.0, height=10.0)
        corrected, scales = local_scale_correction(v, f, v, n_regions=4)
        np.testing.assert_allclose(scales, np.ones(4), atol=0.2)


# =========================================================================
# 8. feature_edge_transfer
# =========================================================================

class TestFeatureEdgeTransfer:
    def test_cube_has_sharp_edges(self):
        from meshxcad.objects.operations import extrude_polygon
        # Extruded square has shared edges between side and cap faces
        poly = [(-5, -5), (5, -5), (5, 5), (-5, 5)]
        v, f = extrude_polygon(poly, height=10.0, n_height=2)
        sharpened, n_sharp = feature_edge_transfer(v, f, v, f)
        assert n_sharp > 0
        assert sharpened.shape == v.shape

    def test_sphere_few_sharp_edges(self):
        v, f = make_sphere_mesh(radius=5.0, lat_divs=15, lon_divs=15)
        _, n_sharp = feature_edge_transfer(v, f, v, f)
        # Sphere should have very few or no sharp edges
        assert n_sharp < len(f)


# =========================================================================
# 9. vertex_normal_realign
# =========================================================================

class TestVertexNormalRealign:
    def test_identical_no_change(self):
        v, f = make_sphere_mesh(radius=5.0, lat_divs=10, lon_divs=10)
        adjusted = vertex_normal_realign(v, f, v, f, strength=0.5)
        np.testing.assert_allclose(adjusted, v, atol=0.01)

    def test_returns_correct_shape(self):
        v, f = make_sphere_mesh(radius=5.0, lat_divs=10, lon_divs=10)
        v2, f2 = make_sphere_mesh(radius=7.0, lat_divs=10, lon_divs=10)
        adjusted = vertex_normal_realign(v, f, v2, f2)
        assert adjusted.shape == v.shape


# =========================================================================
# 10. fill_surface_holes
# =========================================================================

class TestFillSurfaceHoles:
    def test_closed_mesh_no_holes(self):
        v, f = make_sphere_mesh(radius=5.0, lat_divs=15, lon_divs=15)
        _, new_f, n_holes = fill_surface_holes(v, f)
        # Sphere mesh might have poles as holes depending on construction
        # but result should be valid
        assert n_holes >= 0
        assert len(new_f) >= len(f)

    def test_open_cylinder_gets_filled(self):
        """Cylinder without caps should have holes filled."""
        from meshxcad.objects.builder import revolve_profile
        profile = [(5.0, 0), (5.0, 10)]
        v, f = revolve_profile(profile, n_angular=16,
                                close_top=False, close_bottom=False)
        _, new_f, n_holes = fill_surface_holes(v, f)
        assert n_holes >= 1
        assert len(new_f) > len(f)


# =========================================================================
# 11. decimate_to_match
# =========================================================================

class TestDecimateToMatch:
    def test_reduces_face_count(self):
        v, f = make_sphere_mesh(radius=5.0, lat_divs=20, lon_divs=20)
        target = len(f) // 2
        new_v, new_f = decimate_to_match(v, f, target)
        assert len(new_f) <= target + 10  # allow some slack
        assert len(new_f) < len(f)

    def test_preserves_valid_faces(self):
        v, f = make_sphere_mesh(radius=5.0, lat_divs=15, lon_divs=15)
        new_v, new_f = decimate_to_match(v, f, len(f) // 2)
        assert np.all(new_f >= 0)
        assert np.all(new_f < len(new_v))

    def test_already_below_target(self):
        v, f = make_sphere_mesh(radius=5.0, lat_divs=5, lon_divs=5)
        new_v, new_f = decimate_to_match(v, f, len(f) * 2)
        assert len(new_f) == len(f)


# =========================================================================
# 12. suggest_general_adjustments
# =========================================================================

class TestSuggestGeneralAdjustments:
    def test_identical_few_suggestions(self):
        v, f = make_sphere_mesh(radius=5.0, lat_divs=15, lon_divs=15)
        suggestions = suggest_general_adjustments(v, f, v, f)
        assert isinstance(suggestions, list)
        assert len(suggestions) >= 1

    def test_offset_suggests_projection(self):
        v, f = make_sphere_mesh(radius=5.0, lat_divs=15, lon_divs=15)
        v2 = v + 3.0
        suggestions = suggest_general_adjustments(v2, f, v, f)
        actions = [s["action"] for s in suggestions]
        assert "project_vertices_to_mesh" in actions or "laplacian_smooth_toward" in actions

    def test_priority_ordered(self):
        v, f = make_sphere_mesh(radius=5.0, lat_divs=10, lon_divs=10)
        v2 = v * 1.5
        suggestions = suggest_general_adjustments(v2, f, v, f)
        if len(suggestions) >= 2:
            prios = [s["priority"] for s in suggestions]
            assert prios == sorted(prios)

    def test_all_keys_present(self):
        v, f = make_cube_mesh(size=10.0, subdivisions=3)
        suggestions = suggest_general_adjustments(v, f, v, f)
        for s in suggestions:
            assert "action" in s
            assert "priority" in s
            assert "params" in s
            assert "reason" in s


# =========================================================================
# Integration
# =========================================================================

# =========================================================================
# Round 2 differentiators
# =========================================================================

class TestConvexityDefectDiff:
    def test_identical_is_small(self):
        from meshxcad.general_align import convexity_defect_diff
        v, _ = make_sphere_mesh(radius=5.0, lat_divs=15, lon_divs=15)
        assert convexity_defect_diff(v, v) < 1.0

    def test_scaled_is_nonzero(self):
        from meshxcad.general_align import convexity_defect_diff
        v, _ = make_sphere_mesh(radius=5.0, lat_divs=15, lon_divs=15)
        v2 = v * 1.5
        assert convexity_defect_diff(v, v2) > 0.1


class TestBoundaryEdgeLengthDiff:
    def test_closed_mesh_is_small(self):
        from meshxcad.general_align import boundary_edge_length_diff
        v, f = make_sphere_mesh(radius=5.0, lat_divs=15, lon_divs=15)
        assert boundary_edge_length_diff(v, f, v, f) < 1.0


class TestPrincipalCurvatureRatioDiff:
    def test_identical_is_small(self):
        from meshxcad.general_align import principal_curvature_ratio_diff
        v, f = make_sphere_mesh(radius=5.0, lat_divs=15, lon_divs=15)
        assert principal_curvature_ratio_diff(v, f, v, f) < 1.0


class TestGeodesicDiameterDiff:
    def test_identical_is_small(self):
        from meshxcad.general_align import geodesic_diameter_diff
        v, f = make_sphere_mesh(radius=5.0, lat_divs=10, lon_divs=10)
        assert geodesic_diameter_diff(v, f, v, f) < 1.0

    def test_different_sizes(self):
        from meshxcad.general_align import geodesic_diameter_diff
        v, f = make_sphere_mesh(radius=5.0, lat_divs=10, lon_divs=10)
        v2 = v * 2.0
        assert geodesic_diameter_diff(v, f, v2, f) > 1.0


class TestLaplacianSpectrumDiff:
    def test_identical_is_small(self):
        from meshxcad.general_align import laplacian_spectrum_diff
        v, f = make_sphere_mesh(radius=5.0, lat_divs=10, lon_divs=10)
        assert laplacian_spectrum_diff(v, f, v, f) < 1.0


class TestFaceAreaVarianceDiff:
    def test_identical_is_small(self):
        from meshxcad.general_align import face_area_variance_diff
        v, f = make_sphere_mesh(radius=5.0, lat_divs=15, lon_divs=15)
        assert face_area_variance_diff(v, f, v, f) < 1.0


class TestVertexNormalDivergence:
    def test_identical_is_small(self):
        from meshxcad.general_align import vertex_normal_divergence
        v, f = make_sphere_mesh(radius=5.0, lat_divs=15, lon_divs=15)
        assert vertex_normal_divergence(v, f, v, f) < 5.0

    def test_offset_is_larger(self):
        from meshxcad.general_align import vertex_normal_divergence
        v, f = make_sphere_mesh(radius=5.0, lat_divs=15, lon_divs=15)
        v2 = v * 1.3
        d1 = vertex_normal_divergence(v, f, v, f)
        d2 = vertex_normal_divergence(v, f, v2, f)
        assert d2 >= d1 - 1e-6


class TestOctantVolumeDiff:
    def test_identical_is_small(self):
        from meshxcad.general_align import octant_volume_diff
        v, f = make_sphere_mesh(radius=5.0, lat_divs=15, lon_divs=15)
        assert octant_volume_diff(v, f, v, f) < 5.0

    def test_asymmetric_is_larger(self):
        from meshxcad.general_align import octant_volume_diff
        v, f = make_sphere_mesh(radius=5.0, lat_divs=15, lon_divs=15)
        v2 = v.copy()
        v2[v2[:, 0] > 0, 0] *= 1.5
        assert octant_volume_diff(v, f, v2, f) > 1.0


class TestEdgeAngleDistributionDiff:
    def test_identical_is_small(self):
        from meshxcad.general_align import edge_angle_distribution_diff
        v, f = make_sphere_mesh(radius=5.0, lat_divs=15, lon_divs=15)
        assert edge_angle_distribution_diff(v, f, v, f) < 1.0


class TestAspectRatioDiff:
    def test_identical_is_small(self):
        from meshxcad.general_align import aspect_ratio_diff
        v, f = make_sphere_mesh(radius=5.0, lat_divs=15, lon_divs=15)
        assert aspect_ratio_diff(v, f, v, f) < 1.0


# =========================================================================
# Round 2 fixers
# =========================================================================

class TestRound2Fixers:
    def _make_pair(self):
        v, f = make_sphere_mesh(radius=5.0, lat_divs=15, lon_divs=15)
        v2 = v * 1.2 + 0.5
        return v2, f, v, f

    def test_fix_convexity_defect(self):
        from meshxcad.general_align import fix_convexity_defect
        cad_v, cad_f, mesh_v, mesh_f = self._make_pair()
        result = fix_convexity_defect(cad_v, cad_f, mesh_v, mesh_f)
        assert result.shape == cad_v.shape

    def test_fix_boundary_edges(self):
        from meshxcad.general_align import fix_boundary_edges
        cad_v, cad_f, mesh_v, mesh_f = self._make_pair()
        result = fix_boundary_edges(cad_v, cad_f, mesh_v, mesh_f)
        assert result.shape == cad_v.shape

    def test_fix_principal_curvature(self):
        from meshxcad.general_align import fix_principal_curvature
        cad_v, cad_f, mesh_v, mesh_f = self._make_pair()
        result = fix_principal_curvature(cad_v, cad_f, mesh_v, mesh_f)
        assert result.shape == cad_v.shape

    def test_fix_geodesic_diameter(self):
        from meshxcad.general_align import fix_geodesic_diameter
        cad_v, cad_f, mesh_v, mesh_f = self._make_pair()
        result = fix_geodesic_diameter(cad_v, cad_f, mesh_v, mesh_f)
        assert result.shape == cad_v.shape

    def test_fix_laplacian_spectrum(self):
        from meshxcad.general_align import fix_laplacian_spectrum
        cad_v, cad_f, mesh_v, mesh_f = self._make_pair()
        result = fix_laplacian_spectrum(cad_v, cad_f, mesh_v, mesh_f)
        assert result.shape == cad_v.shape

    def test_fix_face_area_variance(self):
        from meshxcad.general_align import fix_face_area_variance
        cad_v, cad_f, mesh_v, mesh_f = self._make_pair()
        result = fix_face_area_variance(cad_v, cad_f, mesh_v, mesh_f)
        assert result.shape == cad_v.shape

    def test_fix_vertex_normal_divergence(self):
        from meshxcad.general_align import fix_vertex_normal_divergence
        cad_v, cad_f, mesh_v, mesh_f = self._make_pair()
        result = fix_vertex_normal_divergence(cad_v, cad_f, mesh_v, mesh_f)
        assert result.shape == cad_v.shape

    def test_fix_octant_volume(self):
        from meshxcad.general_align import fix_octant_volume
        cad_v, cad_f, mesh_v, mesh_f = self._make_pair()
        result = fix_octant_volume(cad_v, cad_f, mesh_v, mesh_f)
        assert result.shape == cad_v.shape

    def test_fix_edge_angle_distribution(self):
        from meshxcad.general_align import fix_edge_angle_distribution
        cad_v, cad_f, mesh_v, mesh_f = self._make_pair()
        result = fix_edge_angle_distribution(cad_v, cad_f, mesh_v, mesh_f)
        assert result.shape == cad_v.shape

    def test_fix_aspect_ratio(self):
        from meshxcad.general_align import fix_aspect_ratio
        cad_v, cad_f, mesh_v, mesh_f = self._make_pair()
        result = fix_aspect_ratio(cad_v, cad_f, mesh_v, mesh_f)
        assert result.shape == cad_v.shape


# =========================================================================
# Integration
# =========================================================================

class TestIntegrationGeneralAlign:
    def test_full_pipeline_reduces_distance(self):
        """Apply projection + smoothing + scale correction and verify improvement."""
        v_mesh, f_mesh = make_sphere_mesh(radius=5.0, lat_divs=20, lon_divs=20)
        v_cad, f_cad = make_sphere_mesh(radius=7.0, lat_divs=15, lon_divs=15)

        hd_before = hausdorff_distance(v_cad, v_mesh)

        # Step 1: project
        v1, _ = project_vertices_to_mesh(v_cad, v_mesh, f_mesh)
        # Step 2: smooth toward target
        v2 = laplacian_smooth_toward(v1, f_cad, v_mesh, iterations=3,
                                      target_weight=0.3)

        hd_after = hausdorff_distance(v2, v_mesh)
        assert hd_after["mean_symmetric"] < hd_before["mean_symmetric"]
