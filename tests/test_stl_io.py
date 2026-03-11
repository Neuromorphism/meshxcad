"""Tests for the pure-numpy STL I/O module."""

import numpy as np
import pytest

from meshxcad.stl_io import write_binary_stl, read_binary_stl
from meshxcad.objects.catalog import make_simple, list_objects


class TestSTLIO:
    """Test binary STL read/write."""

    def test_round_trip_cube(self, tmp_path):
        """Simple cube-like mesh should round-trip through STL."""
        # Simple tetrahedron
        verts = np.array([
            [0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]
        ], dtype=float)
        faces = np.array([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]])

        path = str(tmp_path / "test.stl")
        write_binary_stl(path, verts, faces)
        v2, f2 = read_binary_stl(path)

        assert len(f2) == 4
        assert len(v2) == 4

    @pytest.mark.parametrize("name", list_objects()[:5])
    def test_round_trip_catalog_objects(self, name, tmp_path):
        """Catalog objects should survive STL round-trip."""
        v, f = make_simple(name)
        path = str(tmp_path / f"{name}.stl")
        write_binary_stl(path, v, f)
        v2, f2 = read_binary_stl(path)
        assert len(f2) == len(f)
        assert len(v2) > 0
