"""Tests for hourglass models and detail transfer between them."""

import numpy as np
import pytest

from meshxcad.hourglass_synthetic import (
    make_simple_hourglass_mesh,
    make_ornate_hourglass_mesh,
)
from meshxcad.stl_io import write_binary_stl, read_binary_stl
from meshxcad.detail_transfer import transfer_mesh_detail_to_mesh
from meshxcad.alignment import find_correspondences


class TestHourglassMeshGeneration:
    """Test that hourglass meshes are valid."""

    def test_simple_hourglass_valid(self):
        v, f = make_simple_hourglass_mesh(n_angular=24)
        assert len(v) > 100
        assert len(f) > 100
        assert v.shape[1] == 3
        assert f.shape[1] == 3
        # All face indices should be valid
        assert np.all(f >= 0)
        assert np.all(f < len(v))

    def test_ornate_hourglass_valid(self):
        v, f = make_ornate_hourglass_mesh(n_angular=24)
        assert len(v) > 100
        assert len(f) > 100
        assert np.all(f >= 0)
        assert np.all(f < len(v))

    def test_ornate_has_more_geometry(self):
        """Ornate should have more vertices/faces than simple."""
        sv, sf = make_simple_hourglass_mesh(n_angular=24)
        ov, of_ = make_ornate_hourglass_mesh(n_angular=24)
        assert len(ov) > len(sv)
        assert len(of_) > len(sf)

    def test_both_centered_at_origin(self):
        """Both meshes should be roughly centered at origin."""
        for gen in [make_simple_hourglass_mesh, make_ornate_hourglass_mesh]:
            v, _ = gen(n_angular=24)
            centroid = np.mean(v, axis=0)
            assert np.linalg.norm(centroid) < 20  # rough centering


class TestSTLRoundTrip:
    """Test STL export/import."""

    def test_stl_round_trip(self, tmp_path):
        v, f = make_simple_hourglass_mesh(n_angular=12)
        path = str(tmp_path / "test.stl")
        write_binary_stl(path, v, f)
        v2, f2 = read_binary_stl(path)
        # Same number of faces
        assert len(f2) == len(f)
        # Vertices should be close (deduplication may change count)
        assert len(v2) > 0


class TestHourglassDetailTransfer:
    """Test detail transfer between simple and ornate hourglasses.

    Test structure:
      Input: simple hourglass mesh (plain)
      Detail source: ornate hourglass mesh (featured)
      Objective: the result should be closer to ornate than the input was
    """

    @pytest.fixture
    def hourglass_data(self):
        simple_v, simple_f = make_simple_hourglass_mesh(n_angular=24)
        ornate_v, ornate_f = make_ornate_hourglass_mesh(n_angular=24)
        return {
            "simple_v": simple_v,
            "simple_f": simple_f,
            "ornate_v": ornate_v,
            "ornate_f": ornate_f,
        }

    def test_transfer_moves_toward_objective(self, hourglass_data):
        """Transferred mesh should be closer to ornate than the simple input."""
        d = hourglass_data
        result_v = transfer_mesh_detail_to_mesh(
            d["simple_v"], d["simple_f"], d["ornate_v"], d["ornate_f"]
        )

        # Baseline: simple → ornate distance
        _, _, baseline_dists = find_correspondences(d["simple_v"], d["ornate_v"])
        baseline_mean = np.mean(baseline_dists)

        # Result: transferred → ornate distance
        _, _, result_dists = find_correspondences(result_v, d["ornate_v"])
        result_mean = np.mean(result_dists)

        assert result_mean < baseline_mean, (
            f"Transfer should improve distance: {result_mean:.3f} >= {baseline_mean:.3f}"
        )

    def test_transfer_preserves_vertex_count(self, hourglass_data):
        """Result should have same vertex count as the input mesh."""
        d = hourglass_data
        result_v = transfer_mesh_detail_to_mesh(
            d["simple_v"], d["simple_f"], d["ornate_v"], d["ornate_f"]
        )
        assert len(result_v) == len(d["simple_v"])

    def test_transfer_produces_finite_values(self, hourglass_data):
        """Result should have no NaN or inf values."""
        d = hourglass_data
        result_v = transfer_mesh_detail_to_mesh(
            d["simple_v"], d["simple_f"], d["ornate_v"], d["ornate_f"]
        )
        assert np.all(np.isfinite(result_v))

    def test_improvement_is_meaningful(self, hourglass_data):
        """Improvement should be at least 10%."""
        d = hourglass_data
        result_v = transfer_mesh_detail_to_mesh(
            d["simple_v"], d["simple_f"], d["ornate_v"], d["ornate_f"]
        )

        _, _, baseline_dists = find_correspondences(d["simple_v"], d["ornate_v"])
        _, _, result_dists = find_correspondences(result_v, d["ornate_v"])

        improvement = (1 - np.mean(result_dists) / np.mean(baseline_dists)) * 100
        assert improvement > 10, f"Only {improvement:.1f}% improvement — expected >10%"
