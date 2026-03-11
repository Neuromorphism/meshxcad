"""Tests for meshxcad.elegance — CAD elegance scoring and adversarial loops."""

import json
import numpy as np
import pytest

from meshxcad.synthetic import make_sphere_mesh, make_cylinder_mesh
from meshxcad.cad_program import CadOp, CadProgram
from meshxcad.elegance import (
    # Elegance scoring
    OP_TIER, SYMMETRY_OPS, PARAMETRIC_OPS, ELEGANCE_WEIGHTS,
    score_conciseness,
    score_op_hierarchy,
    score_symmetry_exploitation,
    score_no_redundancy,
    score_feature_tree_depth,
    score_parameter_economy,
    score_origin_anchoring,
    score_mesh_quality,
    score_normal_consistency,
    score_op_diversity,
    score_watertightness,
    score_accuracy,
    compute_elegance_score,
    # Discriminator
    DISCRIMINATOR_FEATURES,
    compute_discriminator_features,
    discriminate_cad_vs_mesh,
    _edge_length_regularity,
    _face_area_regularity,
    _normal_smoothness,
    _vertex_valence_regularity,
    _symmetry_score,
    _planarity_fraction,
    # Loop 1
    run_cad_vs_mesh_loop,
    # Loop 2
    compare_elegance,
    run_elegance_tournament,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sphere(r=5.0, center=(0, 0, 0)):
    v, f = make_sphere_mesh(radius=r, lat_divs=16, lon_divs=16)
    v = v + np.array(center)
    return v, f


def _cylinder(r=3.0, h=10.0, center=(0, 0, 0)):
    v, f = make_cylinder_mesh(radius=r, height=h, radial_divs=16, height_divs=8)
    v = v + np.array(center)
    return v, f


def _sphere_program(r=5.0, center=None):
    params = {"radius": r, "divs": 12}
    if center is not None:
        params["center"] = list(center)
    return CadProgram([CadOp("sphere", params)])


def _multi_op_program():
    """A program with several operations for testing metrics."""
    return CadProgram([
        CadOp("sphere", {"radius": 3.0, "center": [0, 0, 0]}),
        CadOp("cylinder", {"radius": 1.0, "height": 5.0, "center": [5, 0, 0]}),
        CadOp("box", {"dimensions": [2, 2, 2], "center": [-5, 0, 0]}),
    ])


# ===========================================================================
# Constants tests
# ===========================================================================

class TestConstants:
    def test_op_tier_covers_all_ops(self):
        from meshxcad.cad_program import OP_COSTS
        for op in OP_COSTS:
            assert op in OP_TIER, f"{op} missing from OP_TIER"

    def test_op_tiers_range(self):
        for op, tier in OP_TIER.items():
            assert 1 <= tier <= 5, f"{op} has out-of-range tier {tier}"

    def test_elegance_weights_sum_to_one(self):
        total = sum(ELEGANCE_WEIGHTS.values())
        assert abs(total - 1.0) < 0.001

    def test_symmetry_ops_defined(self):
        assert "mirror" in SYMMETRY_OPS
        assert "rotate" in SYMMETRY_OPS

    def test_parametric_ops_defined(self):
        assert "revolve" in PARAMETRIC_OPS
        assert "extrude" in PARAMETRIC_OPS


# ===========================================================================
# Individual elegance scorer tests
# ===========================================================================

class TestScoreConciseness:
    def test_single_op_is_perfect(self):
        assert score_conciseness(_sphere_program()) == 1.0

    def test_more_ops_lower_score(self):
        s1 = score_conciseness(_sphere_program())
        s3 = score_conciseness(_multi_op_program())
        assert s1 > s3

    def test_empty_program(self):
        assert score_conciseness(CadProgram()) == 0.0

    def test_score_is_bounded(self):
        prog = CadProgram([CadOp("sphere", {}) for _ in range(20)])
        s = score_conciseness(prog)
        assert 0.0 < s < 1.0


class TestScoreOpHierarchy:
    def test_primitives_only(self):
        prog = CadProgram([CadOp("sphere", {}), CadOp("box", {})])
        assert score_op_hierarchy(prog) == 1.0

    def test_booleans_lower(self):
        prim = CadProgram([CadOp("sphere", {})])
        bool_prog = CadProgram([CadOp("subtract_cylinder", {})])
        assert score_op_hierarchy(prim) > score_op_hierarchy(bool_prog)

    def test_empty(self):
        assert score_op_hierarchy(CadProgram()) == 0.0

    def test_mixed_ops(self):
        prog = CadProgram([
            CadOp("sphere", {}),       # tier 1 → 1.0
            CadOp("extrude", {}),      # tier 2 → 0.8
            CadOp("translate", {}),    # tier 4 → 0.4
        ])
        score = score_op_hierarchy(prog)
        assert 0.5 < score < 1.0


class TestScoreSymmetryExploitation:
    def test_symmetric_target_with_mirror(self):
        target_v, _ = _sphere(r=5.0)
        prog = CadProgram([
            CadOp("sphere", {"radius": 5.0}),
            CadOp("mirror", {"normal": [1, 0, 0]}),
        ])
        score = score_symmetry_exploitation(prog, target_v)
        assert score == 1.0

    def test_no_symmetry_in_target(self):
        # Asymmetric target
        rng = np.random.RandomState(42)
        target_v = rng.randn(100, 3) * 5
        target_v[:, 0] += 10  # shift to break symmetry
        prog = _sphere_program()
        score = score_symmetry_exploitation(prog, target_v)
        assert score >= 0.0


class TestScoreNoRedundancy:
    def test_no_redundant_ops(self):
        target_v, target_f = _sphere(r=5.0)
        prog = _sphere_program(r=5.0)
        score = score_no_redundancy(prog, target_v, target_f)
        assert score == 1.0  # single op can't be redundant

    def test_redundant_op_detected(self):
        target_v, target_f = _sphere(r=5.0)
        # Add a tiny far-away sphere that contributes nothing
        prog = CadProgram([
            CadOp("sphere", {"radius": 5.0, "divs": 12}),
            CadOp("sphere", {"radius": 0.001, "center": [100, 100, 100]}),
        ])
        score = score_no_redundancy(prog, target_v, target_f)
        assert score < 1.0  # should detect the redundant op


class TestScoreFeatureTreeDepth:
    def test_no_modifiers(self):
        prog = CadProgram([CadOp("sphere", {}), CadOp("box", {})])
        assert score_feature_tree_depth(prog) == 1.0

    def test_modifier_chain_reduces_score(self):
        prog = CadProgram([
            CadOp("sphere", {}),
            CadOp("translate", {"offset": [1, 0, 0]}),
            CadOp("scale", {"factors": [2, 2, 2]}),
            CadOp("rotate", {"axis": [0, 0, 1], "angle_deg": 45}),
        ])
        score = score_feature_tree_depth(prog)
        assert score < 1.0

    def test_empty(self):
        assert score_feature_tree_depth(CadProgram()) == 0.0


class TestScoreParameterEconomy:
    def test_minimal_params(self):
        prog = CadProgram([CadOp("sphere", {"radius": 5.0})])
        assert score_parameter_economy(prog) == 1.0

    def test_excessive_params(self):
        # Many params per op
        prog = CadProgram([CadOp("sweep", {
            "profile": [[0, 0], [1, 0], [1, 1], [0, 1], [0.5, 0.5],
                         [-0.5, 0.5], [-1, 0], [-1, -1], [0, -1]],
            "path": [[0, 0, 0], [0, 0, 1], [1, 0, 2], [1, 1, 3],
                     [0, 1, 4], [0, 0, 5], [-1, 0, 6], [-1, -1, 7]],
            "divs": 16,
        })])
        score = score_parameter_economy(prog)
        assert score < 1.0

    def test_empty(self):
        assert score_parameter_economy(CadProgram()) == 0.0


class TestScoreOriginAnchoring:
    def test_at_origin(self):
        prog = CadProgram([CadOp("sphere", {"radius": 5.0, "center": [0, 0, 0]})])
        assert score_origin_anchoring(prog) == 1.0

    def test_no_center_param(self):
        # Default center is [0,0,0]
        prog = CadProgram([CadOp("sphere", {"radius": 5.0})])
        assert score_origin_anchoring(prog) == 1.0

    def test_far_from_origin(self):
        prog = CadProgram([CadOp("sphere", {"radius": 5.0, "center": [100, 200, 300]})])
        assert score_origin_anchoring(prog) < 1.0


class TestScoreMeshQuality:
    def test_sphere_good_quality(self):
        prog = _sphere_program(r=5.0)
        score = score_mesh_quality(prog)
        assert score > 0.8

    def test_empty(self):
        assert score_mesh_quality(CadProgram()) == 0.0


class TestScoreNormalConsistency:
    def test_sphere_consistent_normals(self):
        prog = _sphere_program(r=5.0)
        score = score_normal_consistency(prog)
        assert score > 0.8

    def test_empty(self):
        assert score_normal_consistency(CadProgram()) == 0.0


class TestScoreOpDiversity:
    def test_single_op(self):
        assert score_op_diversity(_sphere_program()) == 1.0

    def test_diverse_ops(self):
        prog = _multi_op_program()
        score = score_op_diversity(prog)
        assert score > 0.7

    def test_repetitive_ops(self):
        prog = CadProgram([CadOp("sphere", {}) for _ in range(6)])
        score = score_op_diversity(prog)
        assert score < 0.5


class TestScoreWatertightness:
    def test_sphere_watertight(self):
        prog = _sphere_program(r=5.0)
        score = score_watertightness(prog)
        assert score > 0.5

    def test_empty(self):
        assert score_watertightness(CadProgram()) == 0.0


class TestScoreAccuracy:
    def test_perfect_match(self):
        target_v, target_f = _sphere(r=5.0)
        prog = CadProgram([CadOp("sphere", {"radius": 5.0, "divs": 16})])
        score = score_accuracy(prog, target_v, target_f)
        assert score > 0.8

    def test_bad_match(self):
        target_v, target_f = _sphere(r=5.0)
        prog = CadProgram([CadOp("sphere", {"radius": 1.0, "center": [50, 50, 50]})])
        score = score_accuracy(prog, target_v, target_f)
        assert score < 0.3

    def test_empty(self):
        target_v, target_f = _sphere(r=5.0)
        assert score_accuracy(CadProgram(), target_v, target_f) == 0.0


# ===========================================================================
# Composite elegance score tests
# ===========================================================================

class TestComputeEleganceScore:
    def test_returns_all_dimensions(self):
        target_v, target_f = _sphere(r=5.0)
        prog = _sphere_program(r=5.0)
        result = compute_elegance_score(prog, target_v, target_f)
        assert "scores" in result
        assert "total" in result
        for key in ELEGANCE_WEIGHTS:
            assert key in result["scores"]

    def test_total_in_range(self):
        target_v, target_f = _sphere(r=5.0)
        prog = _sphere_program(r=5.0)
        result = compute_elegance_score(prog, target_v, target_f)
        assert 0.0 <= result["total"] <= 1.0

    def test_good_program_scores_higher(self):
        target_v, target_f = _sphere(r=5.0)
        good = _sphere_program(r=5.0)
        bad = CadProgram([
            CadOp("sphere", {"radius": 1.0, "center": [50, 50, 50]}),
            CadOp("subtract_cylinder", {"radius": 0.1, "height": 0.1}),
            CadOp("translate", {"offset": [1, 0, 0]}),
            CadOp("scale", {"factors": [2, 2, 2]}),
        ])
        good_score = compute_elegance_score(good, target_v, target_f)["total"]
        bad_score = compute_elegance_score(bad, target_v, target_f)["total"]
        assert good_score > bad_score

    def test_metadata_fields(self):
        target_v, target_f = _sphere(r=5.0)
        prog = _sphere_program(r=5.0)
        result = compute_elegance_score(prog, target_v, target_f)
        assert "n_ops" in result
        assert "complexity" in result
        assert "program_summary" in result


# ===========================================================================
# Discriminator feature tests
# ===========================================================================

class TestDiscriminatorFeatures:
    def test_edge_regularity_cad(self):
        v, f = _sphere(r=5.0)
        reg = _edge_length_regularity(v, f)
        assert reg < 1.0  # CAD sphere has somewhat regular edges

    def test_edge_regularity_noisy(self):
        v, f = _sphere(r=5.0)
        v_noisy = v + np.random.RandomState(42).randn(*v.shape) * 0.5
        reg_clean = _edge_length_regularity(v, f)
        reg_noisy = _edge_length_regularity(v_noisy, f)
        assert reg_noisy > reg_clean  # noisy mesh has more irregular edges

    def test_face_area_regularity(self):
        v, f = _sphere(r=5.0)
        reg = _face_area_regularity(v, f)
        assert reg >= 0.0

    def test_normal_smoothness(self):
        v, f = _sphere(r=5.0)
        ns = _normal_smoothness(v, f)
        assert ns >= 0.0

    def test_vertex_valence_regularity(self):
        v, f = _sphere(r=5.0)
        vr = _vertex_valence_regularity(v, f)
        assert vr >= 0.0

    def test_symmetry_score_sphere(self):
        v, _ = _sphere(r=5.0)
        sym = _symmetry_score(v)
        assert sym > 0.8  # sphere is highly symmetric

    def test_symmetry_score_in_range(self):
        rng = np.random.RandomState(42)
        v = rng.randn(200, 3) * 5
        sym = _symmetry_score(v)
        assert 0.0 <= sym <= 1.0

    def test_planarity_fraction(self):
        v, f = _sphere(r=5.0)
        pf = _planarity_fraction(v, f)
        assert 0.0 <= pf <= 1.0

    def test_compute_all_features(self):
        v, f = _sphere(r=5.0)
        features = compute_discriminator_features(v, f)
        assert len(features) == len(DISCRIMINATOR_FEATURES)
        for name, _ in DISCRIMINATOR_FEATURES:
            assert name in features


class TestDiscriminateCadVsMesh:
    def test_cad_sphere_detected(self):
        v, f = _sphere(r=5.0)
        score = discriminate_cad_vs_mesh(v, f)
        assert 0.0 <= score <= 1.0

    def test_noisy_mesh_less_cad_like(self):
        v, f = _sphere(r=5.0)
        cad_score = discriminate_cad_vs_mesh(v, f)
        v_noisy = v + np.random.RandomState(42).randn(*v.shape) * 1.0
        mesh_score = discriminate_cad_vs_mesh(v_noisy, f)
        # Noisy should look less CAD-like (or at least not more)
        assert mesh_score <= cad_score + 0.1  # allow small tolerance


# ===========================================================================
# Loop 1: CAD vs Mesh tests
# ===========================================================================

class TestCadVsMeshLoop:
    def test_basic_run(self):
        target_v, target_f = _sphere(r=5.0)
        result = run_cad_vs_mesh_loop(target_v, target_f, max_rounds=3)
        assert "program" in result
        assert "history" in result
        assert "final_cad_score" in result
        assert "final_accuracy" in result
        assert len(result["history"]) <= 3

    def test_history_records(self):
        target_v, target_f = _sphere(r=5.0)
        result = run_cad_vs_mesh_loop(target_v, target_f, max_rounds=2)
        for record in result["history"]:
            assert "round" in record
            assert "cad_score" in record
            assert "accuracy" in record
            assert "action" in record

    def test_produces_mesh(self):
        target_v, target_f = _sphere(r=5.0)
        result = run_cad_vs_mesh_loop(target_v, target_f, max_rounds=2)
        assert len(result["cad_vertices"]) > 0
        assert len(result["cad_faces"]) > 0

    def test_cylinder_target(self):
        target_v, target_f = _cylinder(r=3.0, h=10.0)
        result = run_cad_vs_mesh_loop(target_v, target_f, max_rounds=3)
        assert result["final_accuracy"] > 0.0


# ===========================================================================
# Loop 2: Elegance Tournament tests
# ===========================================================================

class TestCompareElegance:
    def test_better_program_wins(self):
        target_v, target_f = _sphere(r=5.0)
        good = CadProgram([CadOp("sphere", {"radius": 5.0, "divs": 12})])
        bad = CadProgram([
            CadOp("sphere", {"radius": 1.0, "center": [50, 50, 50]}),
            CadOp("sphere", {"radius": 0.5, "center": [20, 20, 20]}),
            CadOp("subtract_cylinder", {"radius": 0.1, "height": 0.1}),
        ])
        result = compare_elegance(good, bad, target_v, target_f)
        assert result["winner"] == "A"
        assert result["margin"] > 0

    def test_returns_all_fields(self):
        target_v, target_f = _sphere(r=5.0)
        a = _sphere_program(r=5.0)
        b = _sphere_program(r=4.0)
        result = compare_elegance(a, b, target_v, target_f)
        assert "winner" in result
        assert "margin" in result
        assert "a_total" in result
        assert "b_total" in result
        assert "a_scores" in result
        assert "b_scores" in result

    def test_same_program_ties(self):
        target_v, target_f = _sphere(r=5.0)
        prog = _sphere_program(r=5.0)
        result = compare_elegance(prog, prog.copy(), target_v, target_f)
        assert result["winner"] == "tie"


class TestEleganceTournament:
    def test_basic_run(self):
        target_v, target_f = _sphere(r=5.0)
        result = run_elegance_tournament(target_v, target_f,
                                          max_rounds=3, n_contestants=3)
        assert "program" in result
        assert "elegance_score" in result
        assert "history" in result
        assert "population" in result
        assert len(result["history"]) == 3

    def test_history_structure(self):
        target_v, target_f = _sphere(r=5.0)
        result = run_elegance_tournament(target_v, target_f,
                                          max_rounds=2, n_contestants=2)
        for record in result["history"]:
            assert "round" in record
            assert "champion_score" in record
            assert "champion_summary" in record
            assert "population_scores" in record

    def test_champion_has_positive_score(self):
        target_v, target_f = _sphere(r=5.0)
        result = run_elegance_tournament(target_v, target_f,
                                          max_rounds=3, n_contestants=3)
        assert result["elegance_score"] > 0

    def test_produces_mesh(self):
        target_v, target_f = _sphere(r=5.0)
        result = run_elegance_tournament(target_v, target_f,
                                          max_rounds=2, n_contestants=2)
        assert len(result["cad_vertices"]) > 0
        assert len(result["cad_faces"]) > 0

    def test_cylinder_target(self):
        target_v, target_f = _cylinder(r=3.0, h=10.0)
        result = run_elegance_tournament(target_v, target_f,
                                          max_rounds=3, n_contestants=2)
        assert result["elegance_score"] > 0

    def test_elegance_details_complete(self):
        target_v, target_f = _sphere(r=5.0)
        result = run_elegance_tournament(target_v, target_f,
                                          max_rounds=2, n_contestants=2)
        details = result["elegance_details"]
        assert "scores" in details
        assert "total" in details
        for key in ELEGANCE_WEIGHTS:
            assert key in details["scores"]
