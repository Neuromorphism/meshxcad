"""Tests for the alignment module (ICP, correspondences, displacement)."""

import numpy as np
import pytest

from meshxcad.alignment import icp, find_correspondences, compute_displacement_field


class TestICP:
    """Test Iterative Closest Point alignment."""

    def test_identity(self):
        """ICP on identical point sets should return near-identity transform."""
        pts = np.random.RandomState(42).randn(100, 3)
        aligned, R, t = icp(pts, pts)
        np.testing.assert_allclose(R, np.eye(3), atol=1e-6)
        np.testing.assert_allclose(t, np.zeros(3), atol=1e-6)
        np.testing.assert_allclose(aligned, pts, atol=1e-6)

    def test_translation(self):
        """ICP should recover a known translation."""
        rng = np.random.RandomState(42)
        source = rng.randn(200, 3)
        offset = np.array([3.0, -2.0, 1.0])
        target = source + offset

        aligned, R, t = icp(source, target)
        np.testing.assert_allclose(t, offset, atol=0.1)
        np.testing.assert_allclose(aligned, target, atol=0.1)

    def test_rotation(self):
        """ICP should recover a known rotation."""
        rng = np.random.RandomState(42)
        source = rng.randn(200, 3)

        # 10-degree rotation around Z (small enough for ICP to converge)
        angle = np.pi / 18
        R_true = np.array([
            [np.cos(angle), -np.sin(angle), 0],
            [np.sin(angle), np.cos(angle), 0],
            [0, 0, 1],
        ])
        target = (R_true @ source.T).T

        aligned, R, t = icp(source, target)
        np.testing.assert_allclose(aligned, target, atol=0.1)

    def test_rigid_transform(self):
        """ICP should recover combined rotation + translation."""
        rng = np.random.RandomState(42)
        source = rng.randn(300, 3)

        angle = np.pi / 6
        R_true = np.array([
            [np.cos(angle), -np.sin(angle), 0],
            [np.sin(angle), np.cos(angle), 0],
            [0, 0, 1],
        ])
        t_true = np.array([1.0, -0.5, 2.0])
        target = (R_true @ source.T).T + t_true

        aligned, R, t = icp(source, target)
        np.testing.assert_allclose(aligned, target, atol=0.15)


class TestCorrespondences:
    """Test correspondence finding."""

    def test_exact_match(self):
        """Identical point sets should have zero-distance correspondences."""
        pts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=float)
        src_idx, tgt_idx, dists = find_correspondences(pts, pts)
        np.testing.assert_array_equal(src_idx, [0, 1, 2])
        np.testing.assert_array_equal(tgt_idx, [0, 1, 2])
        np.testing.assert_allclose(dists, 0, atol=1e-10)

    def test_max_distance(self):
        """Points beyond max_distance should be excluded."""
        source = np.array([[0, 0, 0], [10, 0, 0]], dtype=float)
        target = np.array([[0.1, 0, 0], [5, 0, 0]], dtype=float)
        src_idx, tgt_idx, dists = find_correspondences(source, target, max_distance=1.0)
        assert len(src_idx) == 1
        assert src_idx[0] == 0


class TestDisplacementField:
    """Test displacement field computation."""

    def test_zero_displacement(self):
        """Identical source and target should give zero displacement."""
        pts = np.random.RandomState(42).randn(50, 3)
        disp = compute_displacement_field(pts, pts)
        np.testing.assert_allclose(disp, 0, atol=1e-10)

    def test_known_displacement(self):
        """Known offset should be recovered as displacement."""
        source = np.array([[0, 0, 0], [1, 0, 0]], dtype=float)
        offset = np.array([0.5, 0.5, 0.0])
        target = source + offset
        disp = compute_displacement_field(source, target)
        np.testing.assert_allclose(disp, np.tile(offset, (2, 1)), atol=1e-10)

    def test_signed_displacement_along_normal(self):
        """Displacement projected onto normals should give signed scalars."""
        source = np.array([[0, 0, 0], [1, 0, 0]], dtype=float)
        target = np.array([[0, 0, 1], [1, 0, -1]], dtype=float)
        normals = np.array([[0, 0, 1], [0, 0, 1]], dtype=float)
        disp = compute_displacement_field(source, target, normals)
        np.testing.assert_allclose(disp, [1.0, -1.0], atol=1e-10)
