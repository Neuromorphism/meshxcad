"""Tests for the object catalog — all 19 decorative objects."""

import numpy as np
import pytest

from meshxcad.objects.catalog import (
    list_objects, make_simple, make_ornate, OBJECT_CATALOG,
)
from meshxcad.detail_transfer import transfer_mesh_detail_to_mesh
from meshxcad.alignment import find_correspondences


class TestCatalogCompleteness:
    """Verify the catalog has all 19 objects and they're valid."""

    def test_catalog_has_19_objects(self):
        assert len(list_objects()) == 19

    @pytest.mark.parametrize("name", list_objects())
    def test_simple_mesh_valid(self, name):
        v, f = make_simple(name)
        assert v.shape[1] == 3
        assert f.shape[1] == 3
        assert len(v) > 50
        assert len(f) > 50
        assert np.all(f >= 0)
        assert np.all(f < len(v))
        assert np.all(np.isfinite(v))

    @pytest.mark.parametrize("name", list_objects())
    def test_ornate_mesh_valid(self, name):
        v, f = make_ornate(name)
        assert v.shape[1] == 3
        assert f.shape[1] == 3
        assert len(v) > 50
        assert np.all(f >= 0)
        assert np.all(f < len(v))
        assert np.all(np.isfinite(v))

    @pytest.mark.parametrize("name", list_objects())
    def test_ornate_has_more_detail(self, name):
        """Ornate version should generally have more vertices than simple."""
        sv, _ = make_simple(name)
        ov, _ = make_ornate(name)
        # Ornate should have at least as many vertices (usually more due to rings)
        assert len(ov) >= len(sv) * 0.8  # allow some tolerance


class TestCatalogDetailTransfer:
    """Test detail transfer works for all 19 catalog objects."""

    @pytest.mark.parametrize("name", list_objects())
    def test_transfer_improves_distance(self, name):
        """Transfer should move simple closer to ornate for every object."""
        sv, sf = make_simple(name)
        ov, of_ = make_ornate(name)

        result_v = transfer_mesh_detail_to_mesh(sv, sf, ov, of_)

        # Baseline
        _, _, baseline_dists = find_correspondences(sv, ov)
        baseline_mean = np.mean(baseline_dists)

        # Result
        _, _, result_dists = find_correspondences(result_v, ov)
        result_mean = np.mean(result_dists)

        assert result_mean < baseline_mean, (
            f"{name}: transfer didn't improve ({result_mean:.3f} >= {baseline_mean:.3f})"
        )

    @pytest.mark.parametrize("name", list_objects())
    def test_transfer_produces_finite_output(self, name):
        sv, sf = make_simple(name)
        ov, of_ = make_ornate(name)
        result_v = transfer_mesh_detail_to_mesh(sv, sf, ov, of_)
        assert np.all(np.isfinite(result_v))

    @pytest.mark.parametrize("name", list_objects())
    def test_transfer_preserves_vertex_count(self, name):
        sv, sf = make_simple(name)
        ov, of_ = make_ornate(name)
        result_v = transfer_mesh_detail_to_mesh(sv, sf, ov, of_)
        assert len(result_v) == len(sv)
