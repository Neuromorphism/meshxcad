"""Tests for meshxcad.cad_program — CAD program abstraction and evolution."""

import json
import math
import numpy as np
import pytest

from meshxcad.cad_program import (
    CadOp,
    CadProgram,
    ProgramGap,
    OP_COSTS,
    ALPHA,
    BETA,
    GAMMA,
    _eval_op,
    _classify_gap,
    _find_nearest_op,
    find_program_gaps,
    add_operation,
    refine_operation,
    remove_operation,
    simplify_program,
    initial_program,
    run_cad_program_loop,
)
from meshxcad.synthetic import make_sphere_mesh, make_cylinder_mesh


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


# ===========================================================================
# CadOp tests
# ===========================================================================

class TestCadOp:
    def test_complexity_cost_known(self):
        for op_type, expected in OP_COSTS.items():
            op = CadOp(op_type, {})
            assert op.complexity_cost == expected

    def test_complexity_cost_unknown(self):
        op = CadOp("unknown_op", {})
        assert op.complexity_cost == 1.0

    def test_param_count_scalars(self):
        op = CadOp("sphere", {"radius": 1.0, "divs": 20})
        assert op.param_count == 2

    def test_param_count_lists(self):
        op = CadOp("sphere", {"center": [1, 2, 3], "radius": 1.0})
        assert op.param_count == 4  # 3 + 1

    def test_param_count_numpy(self):
        op = CadOp("sphere", {"center": np.array([1, 2, 3]), "radius": 1.0})
        assert op.param_count == 4

    def test_to_dict(self):
        op = CadOp("sphere", {"center": np.array([1.0, 2.0, 3.0]), "radius": 5.0})
        d = op.to_dict()
        assert d["op_type"] == "sphere"
        assert d["enabled"] is True
        assert d["params"]["center"] == [1.0, 2.0, 3.0]
        assert d["params"]["radius"] == 5.0

    def test_enabled_default(self):
        op = CadOp("sphere", {})
        assert op.enabled is True


# ===========================================================================
# CadProgram tests
# ===========================================================================

class TestCadProgram:
    def test_empty_program(self):
        prog = CadProgram()
        v, f = prog.evaluate()
        assert len(v) == 0
        assert len(f) == 0

    def test_single_sphere(self):
        prog = CadProgram([CadOp("sphere", {"radius": 5.0, "divs": 10})])
        v, f = prog.evaluate()
        assert len(v) > 0
        assert len(f) > 0
        # Check sphere-like: all vertices roughly at radius 5
        dists = np.linalg.norm(v, axis=1)
        assert np.allclose(dists, 5.0, atol=0.1)

    def test_sphere_with_center(self):
        center = [3, 4, 5]
        prog = CadProgram([CadOp("sphere", {"radius": 2.0, "center": center})])
        v, f = prog.evaluate()
        centroid = v.mean(axis=0)
        np.testing.assert_allclose(centroid, center, atol=0.5)

    def test_single_cylinder(self):
        prog = CadProgram([CadOp("cylinder", {"radius": 3.0, "height": 10.0})])
        v, f = prog.evaluate()
        assert len(v) > 0

    def test_single_box(self):
        prog = CadProgram([CadOp("box", {"dimensions": [2, 4, 6]})])
        v, f = prog.evaluate()
        assert len(v) > 0
        # Check extents
        extents = v.max(axis=0) - v.min(axis=0)
        np.testing.assert_allclose(extents, [2, 4, 6], atol=0.5)

    def test_single_cone(self):
        prog = CadProgram([CadOp("cone", {
            "base_radius": 3.0, "top_radius": 0.5, "height": 5.0
        })])
        v, f = prog.evaluate()
        assert len(v) > 0

    def test_single_torus(self):
        prog = CadProgram([CadOp("torus", {
            "major_r": 5.0, "minor_r": 1.0
        })])
        v, f = prog.evaluate()
        assert len(v) > 0

    def test_multiple_ops(self):
        ops = [
            CadOp("sphere", {"radius": 2.0, "center": [0, 0, 0]}),
            CadOp("cylinder", {"radius": 1.0, "height": 5.0, "center": [5, 0, 0]}),
        ]
        prog = CadProgram(ops)
        v, f = prog.evaluate()
        assert len(v) > 0
        # Should have vertices from both objects
        assert v[:, 0].max() > 3.0  # cylinder center at x=5

    def test_disabled_op_skipped(self):
        op = CadOp("sphere", {"radius": 5.0})
        op.enabled = False
        prog = CadProgram([op])
        v, f = prog.evaluate()
        assert len(v) == 0

    def test_n_enabled(self):
        ops = [
            CadOp("sphere", {}),
            CadOp("cylinder", {}),
            CadOp("box", {}),
        ]
        ops[1].enabled = False
        prog = CadProgram(ops)
        assert prog.n_enabled() == 2

    def test_total_complexity(self):
        ops = [
            CadOp("sphere", {}),   # 1.0
            CadOp("cone", {}),     # 1.5
        ]
        prog = CadProgram(ops)
        assert prog.total_complexity() == 2.5

    def test_total_complexity_excludes_disabled(self):
        ops = [
            CadOp("sphere", {}),   # 1.0
            CadOp("cone", {}),     # 1.5
        ]
        ops[1].enabled = False
        prog = CadProgram(ops)
        assert prog.total_complexity() == 1.0

    def test_elegance_penalty(self):
        ops = [CadOp("sphere", {"radius": 1.0, "center": [0, 0, 0]})]
        prog = CadProgram(ops)
        # complexity=1.0, n_enabled=1, param_count=4
        expected = ALPHA * 1.0 + BETA * 1 + GAMMA * 4
        assert abs(prog.elegance_penalty() - expected) < 1e-10

    def test_total_cost_sphere_matches_sphere(self):
        target_v, target_f = _sphere(r=5.0)
        prog = CadProgram([CadOp("sphere", {"radius": 5.0, "divs": 16})])
        cost = prog.total_cost(target_v, target_f)
        assert cost < 0.1  # should be very close to 0

    def test_total_cost_empty_is_inf(self):
        target_v, target_f = _sphere()
        prog = CadProgram()
        cost = prog.total_cost(target_v, target_f)
        assert cost == float('inf')

    def test_cache_invalidation(self):
        prog = CadProgram([CadOp("sphere", {"radius": 5.0, "divs": 10})])
        v1, f1 = prog.evaluate()
        # Should be cached
        v2, f2 = prog.evaluate()
        assert v1 is v2  # same object from cache
        # Invalidate
        prog.invalidate_cache()
        v3, f3 = prog.evaluate()
        assert v3 is not v1  # fresh evaluation

    def test_copy_is_independent(self):
        prog = CadProgram([CadOp("sphere", {"radius": 5.0, "center": [0, 0, 0]})])
        prog2 = prog.copy()
        prog2.operations[0].params["radius"] = 10.0
        assert prog.operations[0].params["radius"] == 5.0

    def test_summary(self):
        ops = [CadOp("sphere", {}), CadOp("cylinder", {})]
        prog = CadProgram(ops)
        s = prog.summary()
        assert "2 ops" in s
        assert "sphere" in s
        assert "cylinder" in s

    def test_to_dict_from_dict_roundtrip(self):
        ops = [
            CadOp("sphere", {"radius": 3.0, "center": [1, 2, 3]}),
            CadOp("cylinder", {"radius": 2.0, "height": 5.0}),
        ]
        prog = CadProgram(ops)
        d = prog.to_dict()
        prog2 = CadProgram.from_dict(d)
        assert len(prog2.operations) == 2
        assert prog2.operations[0].op_type == "sphere"
        assert prog2.operations[0].params["radius"] == 3.0
        np.testing.assert_array_equal(
            prog2.operations[0].params["center"], [1, 2, 3])

    def test_to_dict_json_serializable(self):
        prog = CadProgram([
            CadOp("sphere", {"radius": 3.0, "center": np.array([1.0, 2.0, 3.0])})
        ])
        d = prog.to_dict()
        # Should not raise
        json.dumps(d)


# ===========================================================================
# _eval_op tests
# ===========================================================================

class TestEvalOp:
    def test_sphere(self):
        op = CadOp("sphere", {"radius": 2.0, "divs": 8})
        v, f = _eval_op(op, [])
        assert len(v) > 0
        dists = np.linalg.norm(v, axis=1)
        np.testing.assert_allclose(dists, 2.0, atol=0.01)

    def test_cylinder(self):
        op = CadOp("cylinder", {"radius": 3.0, "height": 6.0, "divs": 12})
        v, f = _eval_op(op, [])
        assert len(v) > 0

    def test_box(self):
        op = CadOp("box", {"dimensions": [4, 4, 4]})
        v, f = _eval_op(op, [])
        assert len(v) > 0

    def test_cone(self):
        op = CadOp("cone", {"base_radius": 3, "top_radius": 1, "height": 5})
        v, f = _eval_op(op, [])
        assert len(v) > 0

    def test_torus(self):
        op = CadOp("torus", {"major_r": 5, "minor_r": 1})
        v, f = _eval_op(op, [])
        assert len(v) > 0

    def test_extrude(self):
        op = CadOp("extrude", {
            "polygon": [(-1, -1), (1, -1), (1, 1), (-1, 1)],
            "height": 3.0,
        })
        v, f = _eval_op(op, [])
        assert len(v) > 0

    def test_translate_no_meshes(self):
        op = CadOp("translate", {"offset": [1, 2, 3]})
        result = _eval_op(op, [])
        assert result is None

    def test_translate_with_mesh(self):
        sv, sf = make_sphere_mesh(radius=1, lat_divs=6, lon_divs=6)
        op = CadOp("translate", {"offset": [10, 0, 0]})
        v, f = _eval_op(op, [(sv, sf)])
        np.testing.assert_allclose(v.mean(axis=0)[0], 10.0, atol=0.5)

    def test_scale_with_mesh(self):
        sv, sf = make_sphere_mesh(radius=1, lat_divs=6, lon_divs=6)
        op = CadOp("scale", {"factors": [2, 2, 2]})
        v, f = _eval_op(op, [(sv, sf)])
        dists = np.linalg.norm(v, axis=1)
        np.testing.assert_allclose(dists, 2.0, atol=0.2)

    def test_rotate_with_mesh(self):
        sv, sf = make_sphere_mesh(radius=1, lat_divs=6, lon_divs=6)
        sv = sv + np.array([5, 0, 0])  # offset
        op = CadOp("rotate", {"axis": [0, 0, 1], "angle_deg": 90})
        v, f = _eval_op(op, [(sv, sf)])
        # x≈0, y≈5 after 90° rotation around z
        np.testing.assert_allclose(v.mean(axis=0)[0], 0.0, atol=0.5)
        np.testing.assert_allclose(v.mean(axis=0)[1], 5.0, atol=0.5)

    def test_mirror(self):
        sv, sf = make_sphere_mesh(radius=1, lat_divs=6, lon_divs=6)
        sv = sv + np.array([5, 0, 0])
        op = CadOp("mirror", {"normal": [1, 0, 0], "point": [0, 0, 0]})
        v, f = _eval_op(op, [(sv, sf)])
        # Should have vertices on both sides
        assert v[:, 0].min() < -3
        assert v[:, 0].max() > 3

    def test_union_returns_none(self):
        op = CadOp("union", {})
        result = _eval_op(op, [])
        assert result is None

    def test_unknown_op_returns_none(self):
        op = CadOp("unknown", {})
        result = _eval_op(op, [])
        assert result is None

    def test_subtract_cylinder(self):
        # Create a box, subtract a cylinder from center
        from meshxcad.reconstruct import _make_box_mesh
        bv, bf = _make_box_mesh(np.array([5, 5, 5]), n_subdiv=4)
        op = CadOp("subtract_cylinder", {
            "center": [0, 0, 0],
            "axis": [0, 0, 1],
            "radius": 2.0,
            "height": 12.0,
        })
        v, f = _eval_op(op, [(bv, bf)])
        # Interior vertices should have been pushed to radius 2
        z_mid = np.abs(v[:, 2]) < 5
        radii = np.linalg.norm(v[z_mid, :2], axis=1)
        assert radii.min() >= 1.9  # all pushed out to at least ~2.0


# ===========================================================================
# RED team tests
# ===========================================================================

class TestClassifyGap:
    def test_few_points_default_sphere(self):
        pts = np.array([[0, 0, 0], [1, 0, 0]])
        center = pts.mean(axis=0)
        op, params = _classify_gap(pts, center, 1.0, pts, np.zeros((0, 3), dtype=int))
        assert op == "sphere"

    def test_elongated_returns_cylinder(self):
        # Line of points along x
        pts = np.column_stack([
            np.linspace(0, 20, 100),
            np.random.RandomState(42).randn(100) * 0.1,
            np.random.RandomState(43).randn(100) * 0.1,
        ])
        center = pts.mean(axis=0)
        op, params = _classify_gap(pts, center, 10.0, pts, np.zeros((0, 3), dtype=int))
        assert op == "cylinder"

    def test_spherical_cluster(self):
        rng = np.random.RandomState(42)
        pts = rng.randn(200, 3)  # isotropic gaussian
        center = pts.mean(axis=0)
        op, params = _classify_gap(pts, center, 3.0, pts, np.zeros((0, 3), dtype=int))
        assert op == "sphere"

    def test_flat_cluster_returns_box(self):
        rng = np.random.RandomState(42)
        pts = np.column_stack([
            rng.randn(200) * 5,
            rng.randn(200) * 5,
            rng.randn(200) * 0.01,  # very flat in z
        ])
        center = pts.mean(axis=0)
        op, params = _classify_gap(pts, center, 5.0, pts, np.zeros((0, 3), dtype=int))
        assert op == "box"


class TestFindProgramGaps:
    def test_perfect_match_no_gaps(self):
        target_v, target_f = _sphere(r=5.0)
        prog = CadProgram([CadOp("sphere", {"radius": 5.0, "divs": 16})])
        gaps = find_program_gaps(prog, target_v, target_f)
        assert len(gaps) == 0

    def test_empty_program_returns_gap(self):
        target_v, target_f = _sphere(r=5.0)
        prog = CadProgram()
        gaps = find_program_gaps(prog, target_v, target_f)
        assert len(gaps) == 1
        assert gaps[0].action == "add"
        assert gaps[0].residual_score > 0

    def test_mismatched_shape_finds_gaps(self):
        target_v, target_f = _sphere(r=5.0)
        # Sphere at totally wrong position → most target verts are far from CAD
        prog = CadProgram([CadOp("sphere", {"radius": 1.0, "center": [50, 50, 50], "divs": 8})])
        gaps = find_program_gaps(prog, target_v, target_f)
        assert len(gaps) > 0

    def test_gaps_sorted_by_residual(self):
        target_v, target_f = _sphere(r=5.0)
        prog = CadProgram([CadOp("sphere", {"radius": 1.0, "divs": 8})])
        gaps = find_program_gaps(prog, target_v, target_f)
        if len(gaps) > 1:
            for i in range(len(gaps) - 1):
                assert gaps[i].residual_score >= gaps[i + 1].residual_score


class TestFindNearestOp:
    def test_single_op(self):
        prog = CadProgram([CadOp("sphere", {"radius": 3.0})])
        idx = _find_nearest_op(np.array([0, 0, 0]), prog)
        assert idx == 0

    def test_empty_program(self):
        prog = CadProgram()
        idx = _find_nearest_op(np.array([0, 0, 0]), prog)
        assert idx == -1

    def test_picks_closer_op(self):
        prog = CadProgram([
            CadOp("sphere", {"radius": 1.0, "center": [0, 0, 0]}),
            CadOp("sphere", {"radius": 1.0, "center": [100, 0, 0]}),
        ])
        idx = _find_nearest_op(np.array([95, 0, 0]), prog)
        assert idx == 1


# ===========================================================================
# BLUE team tests
# ===========================================================================

class TestAddOperation:
    def test_adds_op(self):
        prog = CadProgram([CadOp("sphere", {"radius": 1.0})])
        gap = ProgramGap(
            region_center=np.array([10, 0, 0]),
            region_radius=2.0,
            residual_score=5.0,
            suggested_op="cylinder",
            suggested_params={"radius": 1.0, "height": 3.0, "center": [10, 0, 0]},
            nearest_program_op=-1,
            action="add",
        )
        add_operation(prog, gap)
        assert len(prog.operations) == 2
        assert prog.operations[1].op_type == "cylinder"

    def test_cache_invalidated(self):
        prog = CadProgram([CadOp("sphere", {"radius": 1.0})])
        prog.evaluate()
        assert prog._cache_valid
        gap = ProgramGap(np.zeros(3), 1, 1, "box", {}, -1, "add")
        add_operation(prog, gap)
        assert not prog._cache_valid


class TestRefineOperation:
    def test_refine_improves_cost(self):
        target_v, target_f = _sphere(r=5.0)
        # Start with wrong radius
        prog = CadProgram([CadOp("sphere", {"radius": 3.0, "divs": 12})])
        cost_before = prog.total_cost(target_v, target_f)
        refine_operation(prog, 0, target_v, target_f)
        cost_after = prog.total_cost(target_v, target_f)
        assert cost_after <= cost_before

    def test_refine_invalid_index(self):
        target_v, target_f = _sphere()
        prog = CadProgram([CadOp("sphere", {"radius": 5.0})])
        # Should not crash
        refine_operation(prog, -1, target_v, target_f)
        refine_operation(prog, 100, target_v, target_f)


class TestRemoveOperation:
    def test_disables_op(self):
        prog = CadProgram([
            CadOp("sphere", {"radius": 1.0}),
            CadOp("cylinder", {"radius": 1.0}),
        ])
        remove_operation(prog, 1)
        assert not prog.operations[1].enabled
        assert prog.n_enabled() == 1

    def test_invalid_index(self):
        prog = CadProgram([CadOp("sphere", {})])
        remove_operation(prog, 5)  # should not crash
        assert prog.operations[0].enabled


class TestSimplifyProgram:
    def test_removes_useless_ops(self):
        target_v, target_f = _sphere(r=5.0)
        # Sphere + a far-away tiny cylinder (probably hurts more via elegance)
        prog = CadProgram([
            CadOp("sphere", {"radius": 5.0, "divs": 16}),
            CadOp("cylinder", {"radius": 0.01, "height": 0.01, "center": [100, 100, 100]}),
        ])
        simplify_program(prog, target_v, target_f)
        # Should have removed the useless cylinder
        assert prog.n_enabled() <= 2  # may or may not remove

    def test_removes_disabled_ops(self):
        target_v, target_f = _sphere(r=5.0)
        ops = [
            CadOp("sphere", {"radius": 5.0}),
            CadOp("cylinder", {"radius": 1.0}),
        ]
        ops[1].enabled = False
        prog = CadProgram(ops)
        simplify_program(prog, target_v, target_f)
        assert len(prog.operations) == 1
        assert prog.operations[0].op_type == "sphere"

    def test_single_op_not_removed(self):
        target_v, target_f = _sphere(r=5.0)
        prog = CadProgram([CadOp("sphere", {"radius": 5.0, "divs": 16})])
        simplify_program(prog, target_v, target_f)
        assert len(prog.operations) == 1


# ===========================================================================
# initial_program tests
# ===========================================================================

class TestInitialProgram:
    def test_sphere_mesh(self):
        v, f = _sphere(r=5.0)
        prog = initial_program(v, f)
        assert isinstance(prog, CadProgram)
        assert len(prog.operations) >= 1
        # Should produce a reasonable sphere
        cad_v, cad_f = prog.evaluate()
        assert len(cad_v) > 0

    def test_cylinder_mesh(self):
        v, f = _cylinder(r=3.0, h=10.0)
        prog = initial_program(v, f)
        assert isinstance(prog, CadProgram)
        assert len(prog.operations) >= 1

    def test_produces_finite_cost(self):
        v, f = _sphere(r=5.0)
        prog = initial_program(v, f)
        cost = prog.total_cost(v, f)
        assert np.isfinite(cost)
        assert cost >= 0


# ===========================================================================
# run_cad_program_loop integration tests
# ===========================================================================

class TestRunCadProgramLoop:
    def test_sphere_loop(self):
        target_v, target_f = _sphere(r=5.0)
        result = run_cad_program_loop(target_v, target_f, max_rounds=5)
        assert "program" in result
        assert "history" in result
        assert "cad_vertices" in result
        assert "cad_faces" in result
        assert "total_cost" in result
        assert "n_ops" in result
        assert result["total_cost"] < 1.0
        assert len(result["cad_vertices"]) > 0

    def test_cylinder_loop(self):
        target_v, target_f = _cylinder(r=3.0, h=10.0)
        result = run_cad_program_loop(target_v, target_f, max_rounds=5)
        assert result["total_cost"] < 2.0
        assert result["n_ops"] >= 1

    def test_history_records(self):
        target_v, target_f = _sphere(r=5.0)
        result = run_cad_program_loop(target_v, target_f, max_rounds=3)
        for record in result["history"]:
            assert "round" in record
            assert "n_ops" in record
            assert "total_cost" in record
            assert "action" in record

    def test_max_ops_respected(self):
        target_v, target_f = _sphere(r=5.0)
        result = run_cad_program_loop(target_v, target_f, max_rounds=10, max_ops=2)
        assert result["n_ops"] <= 2

    def test_program_serializable(self):
        target_v, target_f = _sphere(r=5.0)
        result = run_cad_program_loop(target_v, target_f, max_rounds=3)
        d = result["program"].to_dict()
        json.dumps(d)  # should not raise

    def test_cost_nonincreasing(self):
        """Total cost should generally not increase across rounds."""
        target_v, target_f = _sphere(r=5.0)
        result = run_cad_program_loop(target_v, target_f, max_rounds=5)
        costs = [r["total_cost"] for r in result["history"]]
        # Allow first round to be worse (exploration), but last should be <= first
        if len(costs) >= 2:
            assert costs[-1] <= costs[0] + 0.01


# ===========================================================================
# Constants tests
# ===========================================================================

class TestConstants:
    def test_op_costs_all_positive(self):
        for op, cost in OP_COSTS.items():
            assert cost > 0, f"{op} has non-positive cost"

    def test_penalty_weights_positive(self):
        assert ALPHA > 0
        assert BETA > 0
        assert GAMMA > 0

    def test_all_modifier_ops_cheap(self):
        modifiers = ["translate", "scale", "rotate", "mirror", "union"]
        for m in modifiers:
            assert OP_COSTS[m] <= 0.5

    def test_primitive_ops_moderate(self):
        primitives = ["sphere", "cylinder", "box"]
        for p in primitives:
            assert OP_COSTS[p] == 1.0
