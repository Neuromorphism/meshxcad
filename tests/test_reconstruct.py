"""Tests for mesh-to-CAD reconstruction pipeline."""

import numpy as np
import pytest

from meshxcad.reconstruct import (
    fit_sphere,
    fit_cylinder,
    fit_cone,
    fit_box,
    classify_mesh,
    reconstruct_revolve,
    reconstruct_extrude,
    reconstruct_sweep,
    reconstruct_sphere,
    reconstruct_cylinder,
    reconstruct_cone,
    reconstruct_box,
    reconstruct_freeform,
    reconstruct_cad,
    mesh_to_cad_file,
    _rotation_between,
    _make_box_mesh,
)
from meshxcad.synthetic import make_sphere_mesh, make_cylinder_mesh


# ===================================================================
# Primitive fitting
# ===================================================================

class TestFitSphere:
    def test_center_and_radius(self):
        v, f = make_sphere_mesh(radius=5.0, lat_divs=20, lon_divs=20)
        result = fit_sphere(v)
        assert abs(result["radius"] - 5.0) < 0.5
        assert np.linalg.norm(result["center"]) < 0.5

    def test_offset_sphere(self):
        v, f = make_sphere_mesh(radius=3.0, lat_divs=15, lon_divs=15)
        v = v + np.array([10.0, 5.0, -3.0])
        result = fit_sphere(v)
        assert abs(result["radius"] - 3.0) < 0.5
        assert abs(result["center"][0] - 10.0) < 1.0

    def test_residual_is_small(self):
        v, f = make_sphere_mesh(radius=5.0, lat_divs=20, lon_divs=20)
        result = fit_sphere(v)
        assert result["residual"] < 1.0


class TestFitCylinder:
    def test_radius_and_height(self):
        v, f = make_cylinder_mesh(radius=5.0, height=20.0,
                                   radial_divs=20, height_divs=10)
        result = fit_cylinder(v)
        assert abs(result["radius"] - 5.0) < 1.0
        assert abs(result["height"] - 20.0) < 2.0

    def test_axis_direction(self):
        v, f = make_cylinder_mesh(radius=5.0, height=20.0)
        result = fit_cylinder(v)
        # Cylinder is along Z, so axis should be near [0, 0, 1]
        assert abs(result["axis"][2]) > 0.9

    def test_residual_is_small(self):
        v, f = make_cylinder_mesh(radius=5.0, height=20.0,
                                   radial_divs=20, height_divs=10)
        result = fit_cylinder(v)
        assert result["residual"] < 2.0


class TestFitCone:
    def test_returns_valid_params(self):
        # Create a cone-like mesh from a revolve profile
        from meshxcad.objects.builder import revolve_profile
        profile = [(5.0, 0.0), (4.0, 5.0), (3.0, 10.0), (2.0, 15.0), (1.0, 20.0)]
        v, f = revolve_profile(profile, 24)
        result = fit_cone(v)
        assert result["height"] > 0
        assert result["base_radius"] > result["top_radius"]

    def test_half_angle(self):
        from meshxcad.objects.builder import revolve_profile
        profile = [(10.0, 0.0), (5.0, 10.0), (0.5, 20.0)]
        v, f = revolve_profile(profile, 24)
        result = fit_cone(v)
        assert result["half_angle_deg"] > 0


class TestFitBox:
    def test_dimensions(self):
        from meshxcad.objects.operations import extrude_polygon
        poly = [(-5, -3), (5, -3), (5, 3), (-5, 3)]
        v, f = extrude_polygon(poly, 10.0)
        result = fit_box(v)
        dims = sorted(result["dimensions"])
        expected = sorted([10.0, 6.0, 10.0])
        for d, e in zip(dims, expected):
            assert abs(d - e) < 2.0

    def test_center(self):
        from meshxcad.objects.operations import extrude_polygon
        poly = [(-5, -3), (5, -3), (5, 3), (-5, 3)]
        v, f = extrude_polygon(poly, 10.0)
        result = fit_box(v)
        # Center should be roughly at (0, 0, 5) for default extrude
        assert np.linalg.norm(result["center"]) < 10.0


# ===================================================================
# Classification
# ===================================================================

class TestClassifyMesh:
    def test_sphere_classified(self):
        v, f = make_sphere_mesh(radius=5.0, lat_divs=20, lon_divs=20)
        result = classify_mesh(v, f)
        assert result["shape_type"] in ("sphere", "revolve")
        assert result["confidence"] > 0.3

    def test_cylinder_classified(self):
        v, f = make_cylinder_mesh(radius=5.0, height=30.0,
                                   radial_divs=20, height_divs=10)
        result = classify_mesh(v, f)
        assert result["shape_type"] in ("cylinder", "revolve", "extrude", "sweep")
        assert result["confidence"] > 0.2

    def test_box_classified(self):
        from meshxcad.objects.operations import extrude_polygon
        poly = [(-5, -5), (5, -5), (5, 5), (-5, 5)]
        v, f = extrude_polygon(poly, 10.0)
        result = classify_mesh(v, f)
        assert result["shape_type"] in ("box", "extrude")

    def test_has_all_scores(self):
        v, f = make_sphere_mesh(radius=5.0)
        result = classify_mesh(v, f)
        assert "all_scores" in result
        types = [s[0] for s in result["all_scores"]]
        assert "sphere" in types
        assert "cylinder" in types

    def test_confidence_range(self):
        v, f = make_sphere_mesh(radius=5.0, lat_divs=15, lon_divs=15)
        result = classify_mesh(v, f)
        assert 0.0 <= result["confidence"] <= 1.0


# ===================================================================
# Reconstruction methods
# ===================================================================

class TestReconstructRevolve:
    def test_produces_valid_mesh(self):
        v, f = make_cylinder_mesh(radius=5.0, height=20.0,
                                   radial_divs=20, height_divs=10)
        result = reconstruct_revolve(v, f)
        assert result["cad_vertices"].shape[1] == 3
        assert result["cad_faces"].shape[1] == 3
        assert len(result["cad_vertices"]) > 0

    def test_returns_profile(self):
        v, f = make_cylinder_mesh(radius=5.0, height=20.0)
        result = reconstruct_revolve(v, f)
        assert result["profile"].shape[1] == 2
        assert len(result["profile"]) >= 2

    def test_quality_positive(self):
        v, f = make_sphere_mesh(radius=5.0, lat_divs=15, lon_divs=15)
        result = reconstruct_revolve(v, f)
        assert result["quality"] >= 0.0


class TestReconstructExtrude:
    def test_produces_valid_mesh(self):
        from meshxcad.objects.operations import extrude_polygon
        poly = [(-5, -3), (5, -3), (5, 3), (-5, 3)]
        v, f = extrude_polygon(poly, 10.0)
        result = reconstruct_extrude(v, f)
        assert result["cad_vertices"].shape[1] == 3
        assert result["cad_faces"].shape[1] == 3

    def test_z_range(self):
        from meshxcad.objects.operations import extrude_polygon
        poly = [(-5, -3), (5, -3), (5, 3), (-5, 3)]
        v, f = extrude_polygon(poly, 10.0)
        result = reconstruct_extrude(v, f)
        z_min, z_max = result["z_range"]
        assert z_max > z_min


class TestReconstructSweep:
    def test_produces_valid_mesh(self):
        v, f = make_cylinder_mesh(radius=3.0, height=30.0,
                                   radial_divs=20, height_divs=15)
        result = reconstruct_sweep(v, f)
        assert result["cad_vertices"].shape[1] == 3
        assert result["cad_faces"].shape[1] == 3

    def test_quality_nonnegative(self):
        v, f = make_cylinder_mesh(radius=3.0, height=30.0,
                                   radial_divs=20, height_divs=15)
        result = reconstruct_sweep(v, f)
        assert result["quality"] >= 0.0


class TestReconstructSphere:
    def test_produces_sphere(self):
        v, f = make_sphere_mesh(radius=5.0, lat_divs=20, lon_divs=20)
        result = reconstruct_sphere(v)
        assert result["cad_vertices"].shape[1] == 3
        assert abs(result["radius"] - 5.0) < 0.5

    def test_offset_sphere(self):
        v, f = make_sphere_mesh(radius=3.0, lat_divs=15, lon_divs=15)
        v = v + np.array([10.0, 0.0, 0.0])
        result = reconstruct_sphere(v)
        assert abs(result["center"][0] - 10.0) < 1.0


class TestReconstructCylinder:
    def test_produces_cylinder(self):
        v, f = make_cylinder_mesh(radius=5.0, height=20.0,
                                   radial_divs=20, height_divs=10)
        result = reconstruct_cylinder(v)
        assert result["cad_vertices"].shape[1] == 3
        assert result["cad_faces"].shape[1] == 3
        assert abs(result["params"]["radius"] - 5.0) < 1.0


class TestReconstructCone:
    def test_produces_mesh(self):
        from meshxcad.objects.builder import revolve_profile
        profile = [(5.0, 0.0), (3.0, 10.0), (1.0, 20.0)]
        v, f = revolve_profile(profile, 24)
        result = reconstruct_cone(v)
        assert result["cad_vertices"].shape[1] == 3
        assert result["cad_faces"].shape[1] == 3


class TestReconstructBox:
    def test_produces_mesh(self):
        from meshxcad.objects.operations import extrude_polygon
        poly = [(-5, -3), (5, -3), (5, 3), (-5, 3)]
        v, f = extrude_polygon(poly, 10.0)
        result = reconstruct_box(v)
        assert result["cad_vertices"].shape[1] == 3
        assert result["cad_faces"].shape[1] == 3


class TestReconstructFreeform:
    def test_produces_mesh(self):
        v, f = make_sphere_mesh(radius=5.0, lat_divs=15, lon_divs=15)
        result = reconstruct_freeform(v, f)
        assert result["cad_vertices"].shape[1] == 3
        assert result["cad_faces"].shape[1] == 3

    def test_quality_nonnegative(self):
        v, f = make_sphere_mesh(radius=5.0, lat_divs=15, lon_divs=15)
        result = reconstruct_freeform(v, f)
        assert result["quality"] >= 0.0


# ===================================================================
# Full pipeline
# ===================================================================

class TestReconstructCad:
    def test_auto_classify_sphere(self):
        v, f = make_sphere_mesh(radius=5.0, lat_divs=20, lon_divs=20)
        result = reconstruct_cad(v, f)
        assert result["shape_type"] in ("sphere", "revolve", "cylinder")
        assert result["cad_vertices"].shape[1] == 3
        assert result["cad_faces"].shape[1] == 3
        assert result["quality"] >= 0.0

    def test_override_shape_type(self):
        v, f = make_sphere_mesh(radius=5.0, lat_divs=15, lon_divs=15)
        result = reconstruct_cad(v, f, shape_type="revolve")
        assert result["shape_type"] == "revolve"

    def test_override_freeform(self):
        v, f = make_sphere_mesh(radius=5.0, lat_divs=15, lon_divs=15)
        result = reconstruct_cad(v, f, shape_type="freeform")
        assert result["shape_type"] == "freeform"

    def test_cylinder_pipeline(self):
        v, f = make_cylinder_mesh(radius=5.0, height=20.0,
                                   radial_divs=20, height_divs=10)
        result = reconstruct_cad(v, f)
        assert result["cad_vertices"].shape[1] == 3
        assert result["quality"] >= 0.0

    def test_has_classification(self):
        v, f = make_sphere_mesh(radius=5.0)
        result = reconstruct_cad(v, f)
        assert result["classification"] is not None
        assert "shape_type" in result["classification"]

    def test_has_params(self):
        v, f = make_sphere_mesh(radius=5.0)
        result = reconstruct_cad(v, f)
        assert "params" in result


# ===================================================================
# File-level API
# ===================================================================

class TestMeshToCadFile:
    def test_roundtrip(self, tmp_path):
        v, f = make_sphere_mesh(radius=5.0, lat_divs=15, lon_divs=15)
        from meshxcad.stl_io import write_binary_stl, read_binary_stl
        input_path = str(tmp_path / "input.stl")
        output_path = str(tmp_path / "output.stl")
        write_binary_stl(input_path, v, f)

        result = mesh_to_cad_file(input_path, output_path)
        assert result["quality"] >= 0.0
        assert result["n_vertices"] > 0
        assert result["n_faces"] > 0

        # Output file should be readable
        out_v, out_f = read_binary_stl(output_path)
        assert out_v.shape[1] == 3

    def test_with_shape_override(self, tmp_path):
        v, f = make_cylinder_mesh(radius=5.0, height=20.0)
        from meshxcad.stl_io import write_binary_stl
        input_path = str(tmp_path / "cyl.stl")
        output_path = str(tmp_path / "cyl_cad.stl")
        write_binary_stl(input_path, v, f)

        result = mesh_to_cad_file(input_path, output_path, shape_type="cylinder")
        assert result["shape_type"] == "cylinder"


# ===================================================================
# Internal helpers
# ===================================================================

class TestRotationBetween:
    def test_identity(self):
        a = np.array([0.0, 0.0, 1.0])
        R = _rotation_between(a, a)
        assert np.allclose(R, np.eye(3), atol=1e-10)

    def test_z_to_x(self):
        z = np.array([0.0, 0.0, 1.0])
        x = np.array([1.0, 0.0, 0.0])
        R = _rotation_between(z, x)
        result = R @ z
        assert np.allclose(result, x, atol=1e-10)

    def test_opposite(self):
        a = np.array([0.0, 0.0, 1.0])
        b = np.array([0.0, 0.0, -1.0])
        R = _rotation_between(a, b)
        result = R @ a
        assert np.allclose(result, b, atol=1e-10)

    def test_preserves_norm(self):
        a = np.array([1.0, 0.0, 0.0])
        b = np.array([0.0, 1.0, 0.0])
        R = _rotation_between(a, b)
        v = np.array([3.0, 4.0, 5.0])
        assert abs(np.linalg.norm(R @ v) - np.linalg.norm(v)) < 1e-10


class TestMakeBoxMesh:
    def test_has_correct_shape(self):
        v, f = _make_box_mesh(np.array([5.0, 3.0, 2.0]), n_subdiv=2)
        assert v.shape[1] == 3
        assert f.shape[1] == 3

    def test_vertex_count(self):
        n = 3
        v, f = _make_box_mesh(np.array([1.0, 1.0, 1.0]), n_subdiv=n)
        # 6 faces, each (n+1)^2 vertices
        expected = 6 * (n + 1) ** 2
        assert len(v) == expected

    def test_bounds(self):
        half = np.array([5.0, 3.0, 2.0])
        v, f = _make_box_mesh(half)
        for ax in range(3):
            assert abs(v[:, ax].max() - half[ax]) < 1e-10
            assert abs(v[:, ax].min() + half[ax]) < 1e-10
