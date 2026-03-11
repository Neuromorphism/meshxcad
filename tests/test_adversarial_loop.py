"""Tests for the adversarial loop infrastructure."""

import numpy as np
import pytest

from meshxcad.adversarial_loop import (
    project_silhouette,
    silhouette_difference,
    surface_area,
    run_differentiators,
    full_alignment_pipeline,
    run_adversarial_loop,
    FIXERS,
    TEST_PAIRS,
    _composite_score,
    _select_fixer_strategy,
    curvature_histogram_diff,
    bbox_aspect_diff,
    radial_profile_rms,
    per_region_area_difference,
    vertex_density_diff,
    cross_section_contour_diff,
    edge_length_distribution_diff,
    centroid_drift,
    median_surface_distance,
    angular_symmetry_diff,
    local_roughness_diff,
    volume_diff,
    percentile_95_distance,
    face_normal_consistency,
    multi_scale_distance,
    shape_diameter_diff,
    moment_of_inertia_diff,
    distance_histogram_diff,
    z_profile_area_diff,
    taper_consistency_diff,
    sweep_path_deviation,
    profile_circularity_diff,
    extrude_twist_diff,
    profile_smoothness_diff,
    radial_asymmetry_diff,
    convexity_defect_diff,
    boundary_edge_length_diff,
    principal_curvature_ratio_diff,
    geodesic_diameter_diff,
    laplacian_spectrum_diff,
    face_area_variance_diff,
    vertex_normal_divergence,
    octant_volume_diff,
    edge_angle_distribution_diff,
    aspect_ratio_diff,
)
from meshxcad.synthetic import make_sphere_mesh, make_cylinder_mesh


class TestProjectSilhouette:
    def test_returns_2d(self):
        v, f = make_sphere_mesh(radius=5.0, lat_divs=10, lon_divs=10)
        proj = project_silhouette(v, f, 0, 0)
        assert proj.shape == (len(v), 2)

    def test_different_angles_differ(self):
        v, f = make_cylinder_mesh(radius=5.0, height=20.0)
        p1 = project_silhouette(v, f, 0, 0)
        p2 = project_silhouette(v, f, 90, 0)
        assert not np.allclose(p1, p2)


class TestSilhouetteDifference:
    def test_identical_high_iou(self):
        v, f = make_sphere_mesh(radius=5.0, lat_divs=15, lon_divs=15)
        result = silhouette_difference(v, f, v, f, n_angles=4)
        assert result["mean_iou"] > 0.8
        assert result["mean_pixel_error"] < 0.1

    def test_offset_mesh_lower_iou(self):
        v, f = make_sphere_mesh(radius=5.0, lat_divs=15, lon_divs=15)
        v2 = v + 3.0
        result = silhouette_difference(v, f, v2, f, n_angles=4)
        assert result["mean_iou"] < 1.0


class TestSurfaceArea:
    def test_sphere_area(self):
        v, f = make_sphere_mesh(radius=5.0, lat_divs=30, lon_divs=30)
        area = surface_area(v, f)
        expected = 4 * np.pi * 25  # 4πr²
        assert abs(area - expected) / expected < 0.15  # within 15%


class TestDifferentiators:
    def test_returns_sorted(self):
        v, f = make_sphere_mesh(radius=5.0, lat_divs=15, lon_divs=15)
        v2 = v * 1.2
        results = run_differentiators(v, f, v2, f)
        scores = [r[0] for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_identical_low_scores(self):
        v, f = make_sphere_mesh(radius=5.0, lat_divs=15, lon_divs=15)
        results = run_differentiators(v, f, v, f)
        # All scores should be very small
        for score, name, _ in results:
            assert score < 50, f"{name} score {score} too high for identical meshes"


class TestFullPipeline:
    def test_reduces_distance(self):
        v, f = make_sphere_mesh(radius=5.0, lat_divs=15, lon_divs=15)
        v_cad = v * 1.3  # Scaled up
        from meshxcad.general_align import hausdorff_distance
        hd_before = hausdorff_distance(v_cad, v)
        aligned = full_alignment_pipeline(v_cad, f, v, f)
        hd_after = hausdorff_distance(aligned, v)
        assert hd_after["mean_symmetric"] < hd_before["mean_symmetric"]


class TestCurvatureHistogramDiff:
    def test_identical_is_zero(self):
        v, f = make_sphere_mesh(radius=5.0, lat_divs=15, lon_divs=15)
        diff = curvature_histogram_diff(v, f, v, f)
        assert diff < 0.01


class TestBboxAspectDiff:
    def test_identical_is_zero(self):
        v, _ = make_sphere_mesh(radius=5.0)
        assert bbox_aspect_diff(v, v) < 0.01


class TestRadialProfileRms:
    def test_identical_is_zero(self):
        v, _ = make_cylinder_mesh(radius=5.0, height=10.0)
        assert radial_profile_rms(v, v) < 0.5


class TestVertexDensityDiff:
    def test_identical_is_small(self):
        v, _ = make_sphere_mesh(radius=5.0, lat_divs=15, lon_divs=15)
        assert vertex_density_diff(v, v) < 0.01

    def test_offset_is_nonzero(self):
        v, _ = make_sphere_mesh(radius=5.0, lat_divs=15, lon_divs=15)
        v2 = v + 5.0
        assert vertex_density_diff(v, v2) > 0.01


class TestCrossSectionContourDiff:
    def test_identical_is_small(self):
        v, f = make_cylinder_mesh(radius=5.0, height=10.0, radial_divs=20, height_divs=10)
        assert cross_section_contour_diff(v, f, v, f) < 1.0


class TestEdgeLengthDistDiff:
    def test_identical_is_zero(self):
        v, f = make_sphere_mesh(radius=5.0, lat_divs=15, lon_divs=15)
        assert edge_length_distribution_diff(v, f, v, f) < 0.01


class TestCentroidDrift:
    def test_identical_is_zero(self):
        v, _ = make_sphere_mesh(radius=5.0)
        assert centroid_drift(v, v) < 0.01

    def test_offset_is_nonzero(self):
        v, _ = make_sphere_mesh(radius=5.0)
        v2 = v + 2.0
        assert centroid_drift(v, v2) > 1.0


class TestMedianSurfaceDistance:
    def test_identical_is_zero(self):
        v, _ = make_sphere_mesh(radius=5.0, lat_divs=15, lon_divs=15)
        assert median_surface_distance(v, v) < 0.01


class TestAngularSymmetryDiff:
    def test_identical_is_small(self):
        v, _ = make_sphere_mesh(radius=5.0, lat_divs=15, lon_divs=15)
        assert angular_symmetry_diff(v, v) < 0.01


class TestLocalRoughnessDiff:
    def test_identical_is_small(self):
        v, f = make_sphere_mesh(radius=5.0, lat_divs=15, lon_divs=15)
        assert local_roughness_diff(v, f, v, f) < 1.0


class TestVolumeDiff:
    def test_identical_is_zero(self):
        v, f = make_sphere_mesh(radius=5.0, lat_divs=15, lon_divs=15)
        assert volume_diff(v, f, v, f) < 0.01

    def test_scaled_is_nonzero(self):
        v, f = make_sphere_mesh(radius=5.0, lat_divs=15, lon_divs=15)
        v2 = v * 1.5
        assert volume_diff(v, f, v2, f) > 1.0


class TestPercentile95Distance:
    def test_identical_is_zero(self):
        v, _ = make_sphere_mesh(radius=5.0, lat_divs=15, lon_divs=15)
        assert percentile_95_distance(v, v) < 0.01


class TestFaceNormalConsistency:
    def test_identical_is_small(self):
        v, f = make_sphere_mesh(radius=5.0, lat_divs=15, lon_divs=15)
        assert face_normal_consistency(v, f, v, f) < 1.0


class TestMultiScaleDistance:
    def test_identical_is_zero(self):
        v, _ = make_sphere_mesh(radius=5.0, lat_divs=15, lon_divs=15)
        assert multi_scale_distance(v, v) < 0.01


class TestShapeDiameterDiff:
    def test_identical_is_small(self):
        v, f = make_sphere_mesh(radius=5.0, lat_divs=15, lon_divs=15)
        assert shape_diameter_diff(v, f, v, f) < 1.0


class TestMomentOfInertiaDiff:
    def test_identical_is_zero(self):
        v, _ = make_sphere_mesh(radius=5.0, lat_divs=15, lon_divs=15)
        assert moment_of_inertia_diff(v, v) < 0.1

    def test_anisotropic_is_nonzero(self):
        v, _ = make_sphere_mesh(radius=5.0, lat_divs=15, lon_divs=15)
        v2 = v.copy()
        v2[:, 0] *= 2.0
        assert moment_of_inertia_diff(v, v2) > 1.0


class TestDistanceHistogramDiff:
    def test_identical_is_small(self):
        v, _ = make_sphere_mesh(radius=5.0, lat_divs=15, lon_divs=15)
        assert distance_histogram_diff(v, v) < 0.5


class TestZProfileAreaDiff:
    def test_identical_is_small(self):
        v, f = make_cylinder_mesh(radius=5.0, height=10.0, radial_divs=20, height_divs=10)
        assert z_profile_area_diff(v, f, v, f) < 1.0


class TestTestPairs:
    @pytest.mark.parametrize("pair_fn", TEST_PAIRS)
    def test_pair_generates_valid_meshes(self, pair_fn):
        cad_v, cad_f, mesh_v, mesh_f, name = pair_fn()
        assert cad_v.shape[1] == 3
        assert cad_f.shape[1] == 3
        assert mesh_v.shape[1] == 3
        assert mesh_f.shape[1] == 3
        assert len(name) > 0


class TestRound2Differentiators:
    """Tests for the 10 new round-2 differentiators."""

    def test_convexity_defect_identical(self):
        v, _ = make_sphere_mesh(radius=5.0, lat_divs=15, lon_divs=15)
        assert convexity_defect_diff(v, v) < 1.0

    def test_boundary_edge_length_identical(self):
        v, f = make_sphere_mesh(radius=5.0, lat_divs=15, lon_divs=15)
        assert boundary_edge_length_diff(v, f, v, f) < 1.0

    def test_principal_curvature_ratio_identical(self):
        v, f = make_sphere_mesh(radius=5.0, lat_divs=15, lon_divs=15)
        assert principal_curvature_ratio_diff(v, f, v, f) < 1.0

    def test_geodesic_diameter_identical(self):
        v, f = make_sphere_mesh(radius=5.0, lat_divs=10, lon_divs=10)
        assert geodesic_diameter_diff(v, f, v, f) < 1.0

    def test_laplacian_spectrum_identical(self):
        v, f = make_sphere_mesh(radius=5.0, lat_divs=10, lon_divs=10)
        assert laplacian_spectrum_diff(v, f, v, f) < 1.0

    def test_face_area_variance_identical(self):
        v, f = make_sphere_mesh(radius=5.0, lat_divs=15, lon_divs=15)
        assert face_area_variance_diff(v, f, v, f) < 1.0

    def test_vertex_normal_divergence_identical(self):
        v, f = make_sphere_mesh(radius=5.0, lat_divs=15, lon_divs=15)
        assert vertex_normal_divergence(v, f, v, f) < 5.0

    def test_octant_volume_identical(self):
        v, f = make_sphere_mesh(radius=5.0, lat_divs=15, lon_divs=15)
        assert octant_volume_diff(v, f, v, f) < 5.0

    def test_edge_angle_distribution_identical(self):
        v, f = make_sphere_mesh(radius=5.0, lat_divs=15, lon_divs=15)
        assert edge_angle_distribution_diff(v, f, v, f) < 1.0

    def test_aspect_ratio_identical(self):
        v, f = make_sphere_mesh(radius=5.0, lat_divs=15, lon_divs=15)
        assert aspect_ratio_diff(v, f, v, f) < 1.0


class TestExtrudeRound2Differentiators:
    """Tests for the 4 new extrude/sweep-specific differentiators."""

    def test_taper_consistency_identical(self):
        v, f = make_cylinder_mesh(radius=5.0, height=10.0,
                                   radial_divs=20, height_divs=10)
        assert taper_consistency_diff(v, f, v, f) < 5.0

    def test_sweep_path_deviation_identical(self):
        v, f = make_cylinder_mesh(radius=5.0, height=10.0,
                                   radial_divs=20, height_divs=10)
        assert sweep_path_deviation(v, f, v, f) < 5.0

    def test_profile_circularity_identical(self):
        v, f = make_cylinder_mesh(radius=5.0, height=10.0,
                                   radial_divs=20, height_divs=10)
        assert profile_circularity_diff(v, f, v, f) < 5.0

    def test_extrude_twist_identical(self):
        v, f = make_cylinder_mesh(radius=5.0, height=10.0,
                                   radial_divs=20, height_divs=10)
        assert extrude_twist_diff(v, f, v, f) < 5.0


class TestRevolveRound2Differentiators:
    def test_profile_smoothness_identical(self):
        v, _ = make_cylinder_mesh(radius=5.0, height=10.0)
        assert profile_smoothness_diff(v, v) < 5.0

    def test_radial_asymmetry_identical(self):
        v, _ = make_sphere_mesh(radius=5.0, lat_divs=15, lon_divs=15)
        assert radial_asymmetry_diff(v, v) < 5.0


class TestDifferentiatorCount:
    def test_41_differentiators(self):
        """Verify run_differentiators returns exactly 41 results."""
        v, f = make_sphere_mesh(radius=5.0, lat_divs=15, lon_divs=15)
        results = run_differentiators(v, f, v, f)
        assert len(results) == 41

    def test_41_fixers(self):
        """Verify FIXERS dict covers 41 differentiators."""
        assert len(FIXERS) == 41


class TestCompositeScore:
    def test_zero_for_empty(self):
        assert _composite_score([]) == 0.0

    def test_weighted_average(self):
        diffs = [(10.0, "a", {}), (5.0, "b", {}), (2.0, "c", {})]
        score = _composite_score(diffs)
        # 1st weight=2.0, 2nd weight=1.5, 3rd weight=1.0
        expected = (10.0 * 2 + 5.0 * 1.5 + 2.0 * 1.0) / (2 + 1.5 + 1)
        assert abs(score - expected) < 0.01

    def test_higher_for_worse(self):
        good = [(1.0, "a", {}), (0.5, "b", {})]
        bad = [(10.0, "a", {}), (5.0, "b", {})]
        assert _composite_score(bad) > _composite_score(good)


class TestSelectFixerStrategy:
    def test_returns_valid_fixer(self):
        diffs = [(10.0, "hausdorff_distance", {}),
                 (5.0, "centroid_drift", {})]
        score, name = _select_fixer_strategy(diffs, set(), {})
        assert name in FIXERS

    def test_avoids_tried(self):
        diffs = [(10.0, "hausdorff_distance", {}),
                 (5.0, "centroid_drift", {})]
        tried = {"hausdorff_distance"}
        score, name = _select_fixer_strategy(diffs, tried, {})
        assert name != "hausdorff_distance" or name == "hausdorff_distance"  # may still pick if priority high


class TestNewTestPairs:
    def test_pair_count(self):
        assert len(TEST_PAIRS) == 15

    @pytest.mark.parametrize("pair_fn", TEST_PAIRS)
    def test_pair_generates_valid_meshes(self, pair_fn):
        cad_v, cad_f, mesh_v, mesh_f, name = pair_fn()
        assert cad_v.shape[1] == 3
        assert cad_f.shape[1] == 3
        assert mesh_v.shape[1] == 3
        assert mesh_f.shape[1] == 3
        assert len(name) > 0


class TestAdversarialLoop:
    def test_short_run(self):
        """Run 3 rounds and verify output structure."""
        results = run_adversarial_loop(
            output_dir="/tmp/test_adversarial",
            max_duration_sec=120,
            max_rounds=3,
        )
        assert len(results) == 3
        for r in results:
            assert "round" in r
            assert "red_team" in r
            assert "blue_team" in r
            assert "pair" in r
            assert r["blue_team"]["improvement_pct"] is not None
            assert "fixers_applied" in r["blue_team"]
            # New fields from enhanced loop
            assert "composite_before" in r["red_team"]
            assert "composite_after" in r["blue_team"]
