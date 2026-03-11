"""Tests for detail transfer between plain and featured geometry.

Each test verifies that transferring detail from a featured source onto
a plain target produces a result close to the known featured target.

Test structure (for each geometry type):
  Inputs:
    - plain mesh vertices/faces (the "blank canvas")
    - featured mesh vertices/faces (the detail source)
  Known objective:
    - featured mesh (the desired end state)
  Test:
    - Transfer featured detail onto plain → result should approximate featured
"""

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
from meshxcad.detail_transfer import (
    compute_detail_displacement,
    apply_displacement_to_mesh,
    transfer_mesh_detail_to_mesh,
    interpolate_displacement_field,
)
from meshxcad.alignment import find_correspondences


class TestCubeDetailTransfer:
    """Test detail transfer with cube geometry.

    Test data set:
    - plain: cube mesh (no features)
    - featured: cube mesh with face pockets
    - objective: plain + featured detail ≈ featured
    """

    @pytest.fixture
    def cube_data(self):
        size = 10.0
        plain_v, plain_f = make_cube_mesh(size=size, subdivisions=8)
        featured_v = add_cube_face_pockets(plain_v.copy(), plain_f,
                                            size=size, pocket_size=3.0,
                                            pocket_depth=1.0)
        return {
            "plain_verts": plain_v,
            "plain_faces": plain_f,
            "featured_verts": featured_v,
            "size": size,
        }

    def test_plain_mesh_modified_by_featured_mesh(self, cube_data):
        """Transfer: plain cube mesh ← featured cube mesh → should ≈ featured cube."""
        result = transfer_mesh_detail_to_mesh(
            cube_data["plain_verts"],
            cube_data["plain_faces"],
            cube_data["featured_verts"],
            cube_data["plain_faces"],  # same topology
        )

        # Result should be close to the featured mesh
        max_error = np.max(np.linalg.norm(result - cube_data["featured_verts"], axis=1))
        mean_error = np.mean(np.linalg.norm(result - cube_data["featured_verts"], axis=1))

        assert mean_error < 0.5, f"Mean error {mean_error} too large"
        assert max_error < 2.0, f"Max error {max_error} too large"

    def test_displacement_captures_pockets(self, cube_data):
        """Displacement field should be non-zero at pocket regions."""
        displacements = compute_detail_displacement(
            cube_data["plain_verts"],
            cube_data["featured_verts"],
            cube_data["plain_faces"],
        )
        disp_magnitudes = np.linalg.norm(displacements, axis=1)

        # Some vertices should have significant displacement (the pockets)
        assert np.max(disp_magnitudes) > 0.3
        # Most vertices should have near-zero displacement (non-pocket areas)
        assert np.median(disp_magnitudes) < 0.1

    def test_roundtrip_same_topology(self, cube_data):
        """With identical topology, transfer should be near-exact."""
        # When source and target have the exact same vertices (just displaced),
        # the transfer should perfectly recover the displacement
        result = transfer_mesh_detail_to_mesh(
            cube_data["plain_verts"],
            cube_data["plain_faces"],
            cube_data["featured_verts"],
            cube_data["plain_faces"],
        )
        error = np.linalg.norm(result - cube_data["featured_verts"], axis=1)
        assert np.mean(error) < 0.3


class TestSphereDetailTransfer:
    """Test detail transfer with sphere geometry."""

    @pytest.fixture
    def sphere_data(self):
        radius = 5.0
        plain_v, plain_f = make_sphere_mesh(radius=radius, lat_divs=25, lon_divs=25)
        featured_v = add_sphere_dimples(plain_v.copy(), radius=radius,
                                         dimple_angle=30.0, dimple_depth=0.8)
        return {
            "plain_verts": plain_v,
            "plain_faces": plain_f,
            "featured_verts": featured_v,
            "radius": radius,
        }

    def test_plain_mesh_modified_by_featured_mesh(self, sphere_data):
        """Transfer sphere dimple detail onto plain sphere."""
        result = transfer_mesh_detail_to_mesh(
            sphere_data["plain_verts"],
            sphere_data["plain_faces"],
            sphere_data["featured_verts"],
            sphere_data["plain_faces"],
        )

        max_error = np.max(np.linalg.norm(result - sphere_data["featured_verts"], axis=1))
        mean_error = np.mean(np.linalg.norm(result - sphere_data["featured_verts"], axis=1))

        assert mean_error < 0.5, f"Mean error {mean_error} too large"

    def test_dimples_are_present(self, sphere_data):
        """Result should show reduced radius at dimple locations."""
        result = transfer_mesh_detail_to_mesh(
            sphere_data["plain_verts"],
            sphere_data["plain_faces"],
            sphere_data["featured_verts"],
            sphere_data["plain_faces"],
        )
        result_r = np.linalg.norm(result, axis=1)
        # Some vertices should be inside the original radius
        assert np.any(result_r < sphere_data["radius"] - 0.2)


class TestCylinderDetailTransfer:
    """Test detail transfer with cylinder geometry."""

    @pytest.fixture
    def cylinder_data(self):
        radius = 5.0
        height = 15.0
        plain_v, plain_f = make_cylinder_mesh(radius=radius, height=height,
                                               radial_divs=36, height_divs=40)
        featured_v = add_cylinder_grooves(plain_v.copy(), radius=radius,
                                           height=height, groove_depth=0.8,
                                           groove_width=1.5, num_grooves=3)
        return {
            "plain_verts": plain_v,
            "plain_faces": plain_f,
            "featured_verts": featured_v,
            "radius": radius,
            "height": height,
        }

    def test_plain_mesh_modified_by_featured_mesh(self, cylinder_data):
        """Transfer groove detail from featured cylinder onto plain cylinder."""
        result = transfer_mesh_detail_to_mesh(
            cylinder_data["plain_verts"],
            cylinder_data["plain_faces"],
            cylinder_data["featured_verts"],
            cylinder_data["plain_faces"],
        )

        max_error = np.max(np.linalg.norm(result - cylinder_data["featured_verts"], axis=1))
        mean_error = np.mean(np.linalg.norm(result - cylinder_data["featured_verts"], axis=1))

        assert mean_error < 0.5, f"Mean error {mean_error} too large"

    def test_grooves_are_present(self, cylinder_data):
        """Result should show reduced radius at groove locations."""
        result = transfer_mesh_detail_to_mesh(
            cylinder_data["plain_verts"],
            cylinder_data["plain_faces"],
            cylinder_data["featured_verts"],
            cylinder_data["plain_faces"],
        )

        barrel = np.abs(result[:, 2]) < cylinder_data["height"] / 2 - 1
        result_r = np.sqrt(result[barrel, 0] ** 2 + result[barrel, 1] ** 2)
        # Some barrel vertices should be grooved (smaller radius)
        assert np.any(result_r < cylinder_data["radius"] - 0.2)


class TestCrossTopologyTransfer:
    """Test transfer between meshes with different tessellation densities."""

    def test_coarse_to_fine_cube(self):
        """Transfer detail from coarse featured cube to fine plain cube."""
        size = 10.0

        # Coarse featured mesh
        coarse_v, coarse_f = make_cube_mesh(size=size, subdivisions=4)
        coarse_featured = add_cube_face_pockets(coarse_v.copy(), coarse_f,
                                                 size=size, pocket_size=3.0,
                                                 pocket_depth=1.0)

        # Fine plain mesh (different tessellation)
        fine_v, fine_f = make_cube_mesh(size=size, subdivisions=8)

        # Transfer detail from coarse featured to fine plain
        result = transfer_mesh_detail_to_mesh(
            fine_v, fine_f, coarse_featured, coarse_f
        )

        # Expected: fine mesh with pockets
        fine_featured = add_cube_face_pockets(fine_v.copy(), fine_f,
                                               size=size, pocket_size=3.0,
                                               pocket_depth=1.0)

        # Should approximate the fine featured version
        mean_error = np.mean(np.linalg.norm(result - fine_featured, axis=1))
        assert mean_error < 1.0, f"Cross-topology mean error {mean_error} too large"


class TestInterpolation:
    """Test displacement field interpolation."""

    def test_rbf_interpolation_constant(self):
        """Constant displacement should be perfectly interpolated."""
        known_pts = np.random.RandomState(42).randn(50, 3)
        offset = np.array([1.0, -0.5, 0.3])
        known_disp = np.tile(offset, (50, 1))

        query_pts = np.random.RandomState(43).randn(20, 3)
        result = interpolate_displacement_field(known_pts, known_disp, query_pts)

        np.testing.assert_allclose(result, np.tile(offset, (20, 1)), atol=0.1)

    def test_rbf_interpolation_linear(self):
        """Linear displacement field should be well-approximated."""
        rng = np.random.RandomState(42)
        known_pts = rng.randn(100, 3)
        # Linear displacement: disp = A * position
        A = np.array([[0.1, 0, 0], [0, 0.2, 0], [0, 0, -0.1]])
        known_disp = known_pts @ A.T

        query_pts = rng.randn(30, 3) * 0.5  # Within the range of known points
        expected = query_pts @ A.T
        result = interpolate_displacement_field(known_pts, known_disp, query_pts)

        np.testing.assert_allclose(result, expected, atol=0.3)
