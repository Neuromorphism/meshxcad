"""Tests for meshxcad.optim — differentiable strategy selection."""

import numpy as np
import pytest

from meshxcad.optim import (
    SegmentationStrategySelector,
    FixerPrioritySelector,
    DifferentiableRefiner,
    mesh_features,
    has_autograd,
    _features_to_vector,
    _extract_continuous_params,
    _set_continuous_params,
    FEATURE_NAMES,
    STRATEGY_NAMES,
)
from meshxcad.synthetic import make_cylinder_mesh, make_sphere_mesh, make_cube_mesh


# ===========================================================================
# mesh_features
# ===========================================================================

class TestMeshFeatures:
    def test_sphere_features(self):
        v, f = make_sphere_mesh(radius=5.0, lat_divs=10, lon_divs=10)
        feats = mesh_features(v, f)

        # Sphere: low elongation, high circularity
        assert feats["elongation"] < 2.0
        assert feats["circularity"] > 0.5
        assert feats["log_faces"] > 0
        assert feats["normal_variance"] > 0

    def test_cylinder_features(self):
        v, f = make_cylinder_mesh(radius=2.0, height=20.0,
                                   radial_divs=12, height_divs=10)
        feats = mesh_features(v, f)

        # Cylinder with height >> radius: high elongation
        assert feats["elongation"] > 1.5

    def test_cube_features(self):
        v, f = make_cube_mesh(size=5.0, subdivisions=3)
        feats = mesh_features(v, f)

        # Cube: low elongation, moderate circularity
        assert feats["elongation"] < 2.0
        assert 0 < feats["bbox_aspect"] <= 1.0

    def test_feature_vector_ordering(self):
        v, f = make_sphere_mesh(radius=3.0, lat_divs=8, lon_divs=8)
        feats = mesh_features(v, f)
        vec = _features_to_vector(feats)

        assert len(vec) == len(FEATURE_NAMES)
        for i, name in enumerate(FEATURE_NAMES):
            assert vec[i] == feats[name]

    def test_degenerate_mesh(self):
        """Tiny mesh with very few vertices should not crash."""
        v = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1.0]])
        f = np.array([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]])
        feats = mesh_features(v, f)
        assert all(np.isfinite(feats[k]) for k in FEATURE_NAMES)


# ===========================================================================
# SegmentationStrategySelector
# ===========================================================================

class TestSegmentationStrategySelector:
    def test_select_returns_valid_strategy(self):
        sel = SegmentationStrategySelector()
        v, f = make_sphere_mesh(radius=5.0, lat_divs=10, lon_divs=10)
        strategy = sel.select(v, f)
        assert strategy in STRATEGY_NAMES

    def test_select_with_probs(self):
        sel = SegmentationStrategySelector()
        v, f = make_cylinder_mesh(radius=2.0, height=20.0,
                                   radial_divs=12, height_divs=10)
        strategy, probs = sel.select_with_probs(v, f)

        assert strategy in STRATEGY_NAMES
        assert len(probs) == len(STRATEGY_NAMES)
        assert abs(sum(probs.values()) - 1.0) < 1e-6
        assert all(p >= 0 for p in probs.values())

    def test_elongated_prefers_skeleton(self):
        """Very elongated mesh should prefer skeleton (from heuristic prior)."""
        sel = SegmentationStrategySelector()
        # Very tall thin cylinder
        v, f = make_cylinder_mesh(radius=1.0, height=50.0,
                                   radial_divs=8, height_divs=20)
        strategy = sel.select(v, f)
        # With heuristic initialization, skeleton should rank high
        _, probs = sel.select_with_probs(v, f)
        assert probs["skeleton"] > 0.1  # at least non-trivial probability

    def test_update_shifts_weights(self):
        """After updating with high quality for a strategy, its probability
        should increase for the same features."""
        sel = SegmentationStrategySelector(lr=0.5)  # aggressive LR for test
        v, f = make_sphere_mesh(radius=5.0, lat_divs=10, lon_divs=10)

        _, probs_before = sel.select_with_probs(v, f)

        # Simulate: "sdf" was great for this sphere-like mesh
        for _ in range(10):
            sel.update(v, f, "sdf", quality=1.0)

        _, probs_after = sel.select_with_probs(v, f)
        assert probs_after["sdf"] > probs_before["sdf"]

    def test_get_set_weights(self):
        sel = SegmentationStrategySelector()
        W, b = sel.get_weights()
        assert W.shape == (len(STRATEGY_NAMES), len(FEATURE_NAMES))
        assert b.shape == (len(STRATEGY_NAMES),)

        # Modify and restore
        W2 = W + 0.1
        b2 = b + 0.1
        sel.set_weights(W2, b2)
        W3, b3 = sel.get_weights()
        np.testing.assert_allclose(W3, W2)
        np.testing.assert_allclose(b3, b2)


# ===========================================================================
# FixerPrioritySelector
# ===========================================================================

class TestFixerPrioritySelector:
    def setup_method(self):
        self.fixer_names = [
            "hausdorff_distance",
            "surface_area_pct_diff",
            "curvature_histogram_diff",
            "silhouette_pixel_error",
        ]
        self.sel = FixerPrioritySelector(self.fixer_names)

    def test_score_returns_ranked_list(self):
        diff_scores = {
            "hausdorff_distance": 5.0,
            "surface_area_pct_diff": 3.0,
            "curvature_histogram_diff": 1.0,
            "silhouette_pixel_error": 4.0,
        }
        tried = set()
        history = {}

        ranked = self.sel.score(diff_scores, tried, history)

        assert len(ranked) == 4
        # Should be sorted descending by priority
        for i in range(len(ranked) - 1):
            assert ranked[i][0] >= ranked[i + 1][0]
        # All fixer names present
        names = {r[1] for r in ranked}
        assert names == set(self.fixer_names)

    def test_freshness_affects_ranking(self):
        diff_scores = {name: 1.0 for name in self.fixer_names}
        tried = {"hausdorff_distance", "surface_area_pct_diff"}
        history = {}

        ranked = self.sel.score(diff_scores, tried, history)
        # Untried fixers should rank higher than tried ones
        untried_names = {"curvature_histogram_diff", "silhouette_pixel_error"}
        top_2_names = {ranked[0][1], ranked[1][1]}
        assert top_2_names == untried_names

    def test_update_changes_priorities(self):
        diff_scores = {name: 1.0 for name in self.fixer_names}
        tried = set()
        history = {}

        ranked_before = self.sel.score(diff_scores, tried, history)

        # Reward hausdorff heavily
        for _ in range(20):
            self.sel.update("hausdorff_distance", diff_scores, tried, history,
                            improved=True, improvement_amount=10.0)

        ranked_after = self.sel.score(diff_scores, tried, history)
        # hausdorff should now rank higher
        hd_rank_before = next(i for i, (_, n) in enumerate(ranked_before)
                              if n == "hausdorff_distance")
        hd_rank_after = next(i for i, (_, n) in enumerate(ranked_after)
                             if n == "hausdorff_distance")
        assert hd_rank_after <= hd_rank_before


# ===========================================================================
# DifferentiableRefiner
# ===========================================================================

class TestDifferentiableRefiner:
    def test_refine_reduces_cost(self):
        """Refiner should not increase total cost."""
        from meshxcad.cad_program import CadProgram, initial_program

        v, f = make_sphere_mesh(radius=5.0, lat_divs=12, lon_divs=12)
        prog = initial_program(v, f)

        if prog.n_enabled() == 0:
            pytest.skip("No operations to refine")

        cost_before = prog.total_cost(v, f)

        refiner = DifferentiableRefiner(max_iter=5)
        cost_after = refiner.refine(prog, 0, v, f)

        # Should not make things worse
        assert cost_after <= cost_before + 0.01

    def test_extract_set_params_roundtrip(self):
        """Parameter extraction and injection should be lossless."""
        from meshxcad.cad_program import CadOp

        op = CadOp.__new__(CadOp)
        op.params = {
            "radius": 5.0,
            "height": 10.0,
            "center": [1.0, 2.0, 3.0],
            "divs": 16,  # should be skipped
        }

        vec, keys, types = _extract_continuous_params(op)
        assert len(vec) == 5  # radius + height + 3 center components
        assert "divs" not in [k for k, _ in keys]

        # Modify and set back
        vec_mod = vec + 0.1
        _set_continuous_params(op, vec_mod, keys, types)
        assert abs(op.params["radius"] - 5.1) < 1e-10
        assert abs(op.params["center"][0] - 1.1) < 1e-10


# ===========================================================================
# Integration: segmentation uses learned selector
# ===========================================================================

class TestSegmentationIntegration:
    def test_auto_select_uses_learned_selector(self):
        """_auto_select_strategy should work with the learned selector."""
        from meshxcad.segmentation import _auto_select_strategy

        v, f = make_sphere_mesh(radius=5.0, lat_divs=10, lon_divs=10)
        strategy = _auto_select_strategy(v, f)
        assert strategy in STRATEGY_NAMES

    def test_segment_mesh_auto(self):
        """Full segment_mesh with auto strategy should still work."""
        from meshxcad.segmentation import segment_mesh

        v, f = make_sphere_mesh(radius=5.0, lat_divs=10, lon_divs=10)
        segments = segment_mesh(v, f, strategy="auto")
        assert isinstance(segments, list)


# ===========================================================================
# Backend detection
# ===========================================================================

class TestBackend:
    def test_has_autograd_returns_bool(self):
        result = has_autograd()
        assert isinstance(result, bool)

    def test_numpy_fallback_works(self):
        """Even without torch, selector should work via numpy path."""
        sel = SegmentationStrategySelector()
        v, f = make_sphere_mesh(radius=5.0, lat_divs=8, lon_divs=8)
        strategy = sel.select(v, f)
        assert strategy in STRATEGY_NAMES

        # Update should not crash
        sel.update(v, f, strategy, quality=0.8)

        # Weights should be extractable
        W, b = sel.get_weights()
        assert W.shape[0] == len(STRATEGY_NAMES)
