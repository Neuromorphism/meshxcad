"""Tests for complex (non-revolve) object catalog."""

import numpy as np
import pytest

from meshxcad.objects.complex_catalog import (
    list_complex_objects, make_complex_simple, make_complex_ornate,
)
from meshxcad.detail_transfer import transfer_mesh_detail_to_mesh
from meshxcad.alignment import find_correspondences


class TestComplexCatalogCompleteness:
    def test_catalog_has_at_least_10_objects(self):
        assert len(list_complex_objects()) >= 10

    @pytest.mark.parametrize("name", list_complex_objects())
    def test_simple_mesh_valid(self, name):
        v, f = make_complex_simple(name)
        assert v.shape[1] == 3 and f.shape[1] == 3
        assert len(v) > 5 and len(f) > 5
        assert np.all(f >= 0) and np.all(f < len(v))
        assert np.all(np.isfinite(v))

    @pytest.mark.parametrize("name", list_complex_objects())
    def test_ornate_mesh_valid(self, name):
        v, f = make_complex_ornate(name)
        assert v.shape[1] == 3 and f.shape[1] == 3
        assert len(v) > 10
        assert np.all(f >= 0) and np.all(f < len(v))
        assert np.all(np.isfinite(v))


class TestComplexDetailTransfer:
    @pytest.mark.parametrize("name", list_complex_objects())
    def test_transfer_improves_distance(self, name):
        sv, sf = make_complex_simple(name)
        ov, of_ = make_complex_ornate(name)
        rv = transfer_mesh_detail_to_mesh(sv, sf, ov, of_)

        _, _, bd = find_correspondences(sv, ov)
        _, _, rd = find_correspondences(rv, ov)
        assert np.mean(rd) < np.mean(bd), (
            f"{name}: {np.mean(rd):.3f} >= {np.mean(bd):.3f}"
        )

    @pytest.mark.parametrize("name", list_complex_objects())
    def test_transfer_produces_finite_output(self, name):
        sv, sf = make_complex_simple(name)
        ov, of_ = make_complex_ornate(name)
        rv = transfer_mesh_detail_to_mesh(sv, sf, ov, of_)
        assert np.all(np.isfinite(rv))
