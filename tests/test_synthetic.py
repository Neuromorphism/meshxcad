"""Tests for synthetic geometry generation."""

import numpy as np
import pytest

from meshxcad.synthetic import (
    make_cube_mesh,
    add_cube_face_pockets,
    make_sphere_mesh,
    add_sphere_dimples,
    make_cylinder_mesh,
    add_cylinder_grooves,
)


class TestCubeGeneration:
    """Test synthetic cube mesh generation."""

    def test_cube_vertex_count(self):
        """Cube should have expected number of vertices."""
        verts, faces = make_cube_mesh(size=10.0, subdivisions=4)
        # 6 faces, each with (subdivisions+1)^2 vertices
        expected = 6 * (4 + 1) ** 2
        assert len(verts) == expected

    def test_cube_bounds(self):
        """Cube vertices should be within [-half, half]."""
        size = 10.0
        half = size / 2
        verts, faces = make_cube_mesh(size=size)
        assert np.all(verts >= -half - 1e-10)
        assert np.all(verts <= half + 1e-10)

    def test_cube_face_count(self):
        """Cube should have expected number of triangular faces."""
        subdivisions = 4
        verts, faces = make_cube_mesh(subdivisions=subdivisions)
        # 6 faces, each with subdivisions^2 quads, 2 triangles per quad
        expected = 6 * subdivisions ** 2 * 2
        assert len(faces) == expected

    def test_pocket_depth(self):
        """Pocketed vertices should be displaced inward by pocket_depth."""
        size = 10.0
        pocket_depth = 1.0
        verts, faces = make_cube_mesh(size=size, subdivisions=10)
        pocketed = add_cube_face_pockets(verts, faces, size=size,
                                          pocket_size=3.0, pocket_depth=pocket_depth)

        # Some vertices should have moved
        diff = np.linalg.norm(pocketed - verts, axis=1)
        assert np.any(diff > 0.5)

        # The max displacement should equal pocket_depth
        assert np.max(diff) <= pocket_depth + 1e-6

    def test_pocket_preserves_non_pocket_vertices(self):
        """Vertices outside pocket regions should not move."""
        size = 10.0
        verts, faces = make_cube_mesh(size=size, subdivisions=10)
        pocketed = add_cube_face_pockets(verts, faces, size=size,
                                          pocket_size=2.0, pocket_depth=1.0)

        # Most vertices should remain unchanged
        unchanged = np.linalg.norm(pocketed - verts, axis=1) < 1e-10
        assert np.sum(unchanged) > len(verts) * 0.5


class TestSphereGeneration:
    """Test synthetic sphere mesh generation."""

    def test_sphere_radius(self):
        """All vertices should be near the sphere radius."""
        radius = 5.0
        verts, faces = make_sphere_mesh(radius=radius)
        radii = np.linalg.norm(verts, axis=1)
        np.testing.assert_allclose(radii, radius, atol=0.01)

    def test_sphere_dimples_reduce_radius(self):
        """Dimpled vertices should have smaller radius than original."""
        radius = 5.0
        verts, faces = make_sphere_mesh(radius=radius, lat_divs=30, lon_divs=30)
        dimpled = add_sphere_dimples(verts, radius=radius)

        # Dimpled vertices should have moved inward
        orig_r = np.linalg.norm(verts, axis=1)
        dimpled_r = np.linalg.norm(dimpled, axis=1)
        assert np.any(dimpled_r < orig_r - 0.1)

        # No vertex should move outward
        assert np.all(dimpled_r <= orig_r + 1e-6)


class TestCylinderGeneration:
    """Test synthetic cylinder mesh generation."""

    def test_cylinder_barrel_radius(self):
        """Barrel vertices should be at the cylinder radius."""
        radius = 5.0
        height = 15.0
        verts, faces = make_cylinder_mesh(radius=radius, height=height)

        # Barrel vertices (not caps)
        half_h = height / 2
        barrel = np.abs(verts[:, 2]) < half_h - 0.01
        barrel_r = np.sqrt(verts[barrel, 0] ** 2 + verts[barrel, 1] ** 2)
        np.testing.assert_allclose(barrel_r, radius, atol=0.01)

    def test_cylinder_height(self):
        """Cylinder should span [-height/2, height/2]."""
        height = 15.0
        verts, faces = make_cylinder_mesh(height=height)
        assert np.min(verts[:, 2]) >= -height / 2 - 0.01
        assert np.max(verts[:, 2]) <= height / 2 + 0.01

    def test_grooves_reduce_radius(self):
        """Grooved vertices should have smaller barrel radius."""
        radius = 5.0
        height = 15.0
        verts, faces = make_cylinder_mesh(radius=radius, height=height,
                                           radial_divs=36, height_divs=40)
        grooved = add_cylinder_grooves(verts, radius=radius, height=height)

        orig_r = np.sqrt(verts[:, 0] ** 2 + verts[:, 1] ** 2)
        grooved_r = np.sqrt(grooved[:, 0] ** 2 + grooved[:, 1] ** 2)

        # Some barrel vertices should have moved inward
        barrel = np.abs(verts[:, 2]) < height / 2 - 0.01
        assert np.any(grooved_r[barrel] < orig_r[barrel] - 0.1)
