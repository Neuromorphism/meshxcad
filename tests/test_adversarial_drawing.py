"""Tests for adversarial_drawing module.

These tests do NOT require the LLM model. Tests that need the model
are marked with @pytest.mark.vision.
"""

import json
import os
import tempfile

import numpy as np
import pytest

from meshxcad.adversarial_drawing import (
    # Failure tracking
    DrawingFailure,
    FailureDatabase,
    FAILURE_TYPES,
    # Perturbation functions
    apply_gaussian_noise,
    apply_resolution_change,
    apply_rotation,
    apply_line_weight,
    apply_salt_and_pepper,
    apply_gaussian_blur,
    apply_contrast_reduction,
    apply_partial_crop,
    PERTURBATIONS,
    # Program generators
    generate_program_for_round,
    _make_box_program,
    _make_cylinder_program,
    _make_sphere_program,
    _add_holes_to_program,
    _make_composition_program,
    _make_revolve_program,
    # Mutation functions
    MUTATION_FUNCTIONS,
    _mutate_tiny_features,
    _mutate_extreme_aspect,
    _mutate_many_features,
    _mutate_thin_walls,
    _mutate_unusual_orientation,
    # Classification
    _classify_failure,
)
from meshxcad.cad_program import CadOp, CadProgram


# ===================================================================
# Helpers
# ===================================================================

def _make_test_image(size=512, n_lines=5):
    """Create a synthetic drawing-like image (white bg, black lines)."""
    img = np.full((size, size), 255, dtype=np.uint8)
    rng = np.random.RandomState(0)
    for _ in range(n_lines):
        r = rng.randint(50, size - 50)
        img[r, 50:size - 50] = 0
    for _ in range(n_lines):
        c = rng.randint(50, size - 50)
        img[50:size - 50, c] = 0
    return img


def _make_test_rgb_image(size=512):
    """Create a synthetic RGB drawing image."""
    gray = _make_test_image(size)
    return np.stack([gray, gray, gray], axis=2)


# ===================================================================
# FailureDatabase tests
# ===================================================================

class TestFailureDatabase:

    def test_empty_database(self):
        db = FailureDatabase()
        assert db.summary()["total"] == 0
        assert db.cluster_by_type() == {}
        assert db.worst_failure_types() == []

    def test_add_failure(self):
        db = FailureDatabase()
        f = DrawingFailure(
            drawing_image=np.zeros((10, 10), dtype=np.uint8),
            expected_program={"operations": []},
            actual_program=None,
            failure_type="empty_result",
            severity=0.8,
        )
        db.add_failure(f)
        assert db.summary()["total"] == 1
        assert db.summary()["by_type"]["empty_result"] == 1

    def test_cluster_by_type(self):
        db = FailureDatabase()
        for ftype in ["missing_feature", "missing_feature", "scale_error"]:
            db.add_failure(DrawingFailure(
                drawing_image=np.zeros((5, 5), dtype=np.uint8),
                expected_program={}, actual_program=None,
                failure_type=ftype, severity=0.5,
            ))
        clusters = db.cluster_by_type()
        assert len(clusters["missing_feature"]) == 2
        assert len(clusters["scale_error"]) == 1

    def test_worst_failure_types(self):
        db = FailureDatabase()
        # High severity type
        for _ in range(3):
            db.add_failure(DrawingFailure(
                drawing_image=np.zeros((5, 5), dtype=np.uint8),
                expected_program={}, actual_program=None,
                failure_type="hallucination", severity=0.9,
            ))
        # Low severity type
        for _ in range(5):
            db.add_failure(DrawingFailure(
                drawing_image=np.zeros((5, 5), dtype=np.uint8),
                expected_program={}, actual_program=None,
                failure_type="wrong_dimension", severity=0.2,
            ))
        worst = db.worst_failure_types(top_n=2)
        assert len(worst) == 2
        assert worst[0]["failure_type"] == "hallucination"
        assert worst[0]["mean_severity"] == pytest.approx(0.9)

    def test_save_and_load(self, tmp_path):
        db = FailureDatabase()
        db.add_failure(DrawingFailure(
            drawing_image=np.ones((32, 32), dtype=np.uint8) * 128,
            expected_program={"operations": [{"op_type": "box", "params": {}}]},
            actual_program={"operations": []},
            failure_type="wrong_shape_type",
            severity=0.7,
            details={"mesh_dist_norm": 0.42},
            round_info={"strategy": "generator", "round": 5},
        ))
        db.add_failure(DrawingFailure(
            drawing_image=np.zeros((64, 64, 3), dtype=np.uint8),
            expected_program={},
            actual_program=None,
            failure_type="empty_result",
            severity=1.0,
        ))

        path = str(tmp_path / "failures.json")
        db.save(path)

        # Verify JSON is valid
        with open(path) as fh:
            data = json.load(fh)
        assert len(data["failures"]) == 2
        assert data["summary"]["total"] == 2

        # Load back
        db2 = FailureDatabase.load(path)
        assert len(db2.failures) == 2
        assert db2.failures[0].failure_type == "wrong_shape_type"
        assert db2.failures[0].severity == pytest.approx(0.7)
        assert db2.failures[0].drawing_image.shape == (32, 32)
        assert db2.failures[1].drawing_image.shape == (64, 64, 3)

    def test_summary_with_mixed_severities(self):
        db = FailureDatabase()
        for sev in [0.1, 0.5, 0.9]:
            db.add_failure(DrawingFailure(
                drawing_image=np.zeros((5, 5), dtype=np.uint8),
                expected_program={}, actual_program=None,
                failure_type="wrong_dimension", severity=sev,
            ))
        s = db.summary()
        assert s["mean_severity"] == pytest.approx(0.5)
        assert s["max_severity"] == pytest.approx(0.9)


# ===================================================================
# Perturbation function tests
# ===================================================================

class TestPerturbations:

    def test_gaussian_noise_preserves_shape(self):
        img = _make_test_image(256)
        noisy = apply_gaussian_noise(img, 10)
        assert noisy.shape == img.shape
        assert noisy.dtype == np.uint8

    def test_gaussian_noise_increases_with_sigma(self):
        img = _make_test_image(256)
        n5 = apply_gaussian_noise(img, 5)
        n40 = apply_gaussian_noise(img, 40)
        diff5 = np.abs(n5.astype(float) - img.astype(float)).mean()
        diff40 = np.abs(n40.astype(float) - img.astype(float)).mean()
        assert diff40 > diff5

    def test_resolution_change_preserves_shape(self):
        img = _make_test_image(512)
        restored = apply_resolution_change(img, 128)
        assert restored.shape == img.shape

    def test_resolution_change_loses_detail(self):
        img = _make_test_image(512)
        restored = apply_resolution_change(img, 64)
        # Should not be identical to original
        assert not np.array_equal(img, restored)

    def test_rotation_preserves_shape(self):
        img = _make_test_image(256)
        rotated = apply_rotation(img, 3)
        assert rotated.shape == img.shape

    def test_rotation_identity(self):
        img = _make_test_image(256)
        rotated = apply_rotation(img, 0)
        # Zero rotation should preserve most pixels (some interpolation)
        diff = np.abs(rotated.astype(float) - img.astype(float)).mean()
        assert diff < 1.0  # near-zero diff

    def test_line_weight_dilate(self):
        img = _make_test_image(256)
        dilated = apply_line_weight(img, "dilate")
        assert dilated.shape == img.shape
        # Dilation should make dark pixels spread (more dark pixels)
        n_dark_orig = np.sum(img < 128)
        n_dark_dilated = np.sum(dilated < 128)
        assert n_dark_dilated >= n_dark_orig

    def test_line_weight_erode(self):
        img = _make_test_image(256)
        eroded = apply_line_weight(img, "erode")
        assert eroded.shape == img.shape
        # Erosion should reduce dark pixels
        n_dark_orig = np.sum(img < 128)
        n_dark_eroded = np.sum(eroded < 128)
        assert n_dark_eroded <= n_dark_orig

    def test_salt_and_pepper(self):
        img = _make_test_image(256)
        noisy = apply_salt_and_pepper(img, 0.05)
        assert noisy.shape == img.shape
        assert not np.array_equal(img, noisy)

    def test_gaussian_blur(self):
        img = _make_test_image(256)
        blurred = apply_gaussian_blur(img, 2)
        assert blurred.shape == img.shape
        # Blur should reduce edge sharpness (reduce variance)
        assert blurred.astype(float).std() <= img.astype(float).std() + 1

    def test_contrast_reduction(self):
        img = _make_test_image(256)
        reduced = apply_contrast_reduction(img, 0.5)
        assert reduced.shape == img.shape
        # Contrast reduction should reduce dynamic range
        orig_range = img.max() - img.min()
        new_range = reduced.max() - reduced.min()
        assert new_range <= orig_range

    def test_partial_crop_grayscale(self):
        img = _make_test_image(256)
        cropped = apply_partial_crop(img, 0.1)
        assert cropped.shape == img.shape
        # Edges should be white
        margin = int(256 * 0.1)
        assert np.all(cropped[:margin, :] == 255)
        assert np.all(cropped[-margin:, :] == 255)

    def test_partial_crop_rgb(self):
        img = _make_test_rgb_image(256)
        cropped = apply_partial_crop(img, 0.1)
        assert cropped.shape == img.shape
        margin = int(256 * 0.1)
        assert np.all(cropped[:margin, :, :] == 255)

    def test_all_perturbation_registry_entries_callable(self):
        """Every entry in PERTURBATIONS should be callable on a test image."""
        img = _make_test_image(256)
        for cat_name, variants in PERTURBATIONS.items():
            for var_name, fn in variants:
                result = fn(img)
                assert result.shape == img.shape, \
                    f"{cat_name}/{var_name} changed shape"
                assert result.dtype == np.uint8, \
                    f"{cat_name}/{var_name} changed dtype"


# ===================================================================
# Program generator tests
# ===================================================================

class TestProgramGenerators:

    def test_box_program_evaluates(self):
        prog = _make_box_program(20, 15, 10)
        v, f = prog.evaluate()
        assert len(v) > 0
        assert len(f) > 0

    def test_cylinder_program_evaluates(self):
        prog = _make_cylinder_program(8, 20)
        v, f = prog.evaluate()
        assert len(v) > 0

    def test_sphere_program_evaluates(self):
        prog = _make_sphere_program(10)
        v, f = prog.evaluate()
        assert len(v) > 0

    def test_add_holes(self):
        prog = _make_cylinder_program(10, 20)
        n_ops_before = len(prog.operations)
        _add_holes_to_program(prog, 3, 10.0, 20.0)
        assert len(prog.operations) == n_ops_before + 3
        v, f = prog.evaluate()
        assert len(v) > 0

    def test_composition_program(self):
        rng = np.random.RandomState(123)
        prog = _make_composition_program(rng, 3)
        v, f = prog.evaluate()
        assert len(v) > 0
        assert prog.n_enabled() >= 2  # at least 2 primitives

    def test_revolve_program(self):
        rng = np.random.RandomState(99)
        prog = _make_revolve_program(rng)
        v, f = prog.evaluate()
        assert len(v) > 0

    @pytest.mark.parametrize("round_num", [1, 5, 10, 11, 15, 20, 21, 25, 30, 31, 35, 40])
    def test_generate_program_for_round(self, round_num):
        """Each non-catalog round should produce a valid CadProgram."""
        prog, level = generate_program_for_round(round_num)
        assert prog is not None, f"Round {round_num} returned None program"
        v, f = prog.evaluate()
        assert len(v) > 0, f"Round {round_num} ({level}) produced empty mesh"

    def test_generate_catalog_round(self):
        """Rounds 41-50 should return None program and 'catalog_object' level."""
        prog, level = generate_program_for_round(45)
        assert prog is None
        assert level == "catalog_object"

    def test_complexity_increases_with_round(self):
        """Later rounds should generally have more operations."""
        _, level_early = generate_program_for_round(3)
        _, level_mid = generate_program_for_round(25)
        assert level_early == "single_primitive"
        assert level_mid == "multi_primitive"


# ===================================================================
# Mutation function tests
# ===================================================================

class TestMutations:

    def _base_program(self):
        prog = _make_cylinder_program(10, 20)
        _add_holes_to_program(prog, 2, 10.0, 20.0)
        return prog

    def test_tiny_features(self):
        prog = self._base_program()
        rng = np.random.RandomState(0)
        mutated, label = _mutate_tiny_features(prog, rng)
        assert label == "tiny_features"
        # Check holes are scaled down
        for op in mutated.operations:
            if op.op_type == "subtract_cylinder":
                assert op.params["radius"] < 0.2  # 1% of ~10
        v, f = mutated.evaluate()
        assert len(v) > 0

    def test_extreme_aspect(self):
        prog = _make_box_program(10, 10, 10)
        rng = np.random.RandomState(0)
        mutated, label = _mutate_extreme_aspect(prog, rng)
        assert label == "extreme_aspect"
        v, f = mutated.evaluate()
        assert len(v) > 0
        # Bounding box should be non-cubic
        bbox = v.max(axis=0) - v.min(axis=0)
        ratio = bbox.max() / max(bbox.min(), 1e-6)
        assert ratio > 3.0  # significant aspect stretch

    def test_many_features(self):
        prog = _make_cylinder_program(10, 20)
        rng = np.random.RandomState(0)
        mutated, label = _mutate_many_features(prog, rng)
        assert label == "many_features"
        n_holes = sum(1 for op in mutated.operations
                      if op.op_type == "subtract_cylinder")
        assert n_holes >= 8
        v, f = mutated.evaluate()
        assert len(v) > 0

    def test_thin_walls(self):
        prog = _make_cylinder_program(10, 20)
        rng = np.random.RandomState(0)
        mutated, label = _mutate_thin_walls(prog, rng)
        assert label == "thin_walls"
        # Should have a subtract_cylinder nearly as big as the body
        for op in mutated.operations:
            if op.op_type == "subtract_cylinder":
                assert op.params["radius"] == pytest.approx(9.5)
        v, f = mutated.evaluate()
        assert len(v) > 0

    def test_unusual_orientation(self):
        prog = _make_box_program(10, 10, 10)
        rng = np.random.RandomState(0)
        mutated, label = _mutate_unusual_orientation(prog, rng)
        assert label == "unusual_orientation"
        assert any(op.op_type == "rotate" for op in mutated.operations)
        v, f = mutated.evaluate()
        assert len(v) > 0

    def test_all_mutations_produce_valid_programs(self):
        """Every mutation function should produce an evaluable program."""
        base = self._base_program()
        rng = np.random.RandomState(42)
        for mut_fn in MUTATION_FUNCTIONS:
            mutated, label = mut_fn(base, rng)
            v, f = mutated.evaluate()
            assert len(v) > 0, f"Mutation {label} produced empty mesh"


# ===================================================================
# Failure classification tests
# ===================================================================

class TestClassifyFailure:

    def test_success_case(self):
        prog1 = _make_box_program(10, 10, 10)
        prog2 = _make_box_program(10.1, 10.1, 10.1)
        from meshxcad.drawing_spec import DrawingSpec
        spec = DrawingSpec(object_type="box")
        ftype, severity = _classify_failure(0.01, prog1, prog2, spec)
        assert ftype is None
        assert severity == 0.0

    def test_none_spec(self):
        prog1 = _make_box_program()
        ftype, severity = _classify_failure(0.5, prog1, None, None)
        assert ftype == "json_parse_error"
        assert severity == 1.0

    def test_empty_result(self):
        prog1 = _make_box_program()
        prog2 = CadProgram([])
        from meshxcad.drawing_spec import DrawingSpec
        spec = DrawingSpec()
        ftype, severity = _classify_failure(0.5, prog1, prog2, spec)
        assert ftype == "empty_result"

    def test_wrong_shape_type(self):
        prog1 = _make_box_program()
        prog2 = _make_cylinder_program()
        from meshxcad.drawing_spec import DrawingSpec
        spec = DrawingSpec(object_type="cylinder")
        ftype, severity = _classify_failure(0.5, prog1, prog2, spec)
        assert ftype == "wrong_shape_type"


# ===================================================================
# FAILURE_TYPES constant test
# ===================================================================

def test_failure_types_are_strings():
    assert len(FAILURE_TYPES) >= 10
    for ft in FAILURE_TYPES:
        assert isinstance(ft, str)
        assert len(ft) > 0


# ===================================================================
# Vision-dependent tests (skip by default)
# ===================================================================

@pytest.mark.vision
@pytest.mark.skip(reason="Requires LLM model; run explicitly with: pytest -m vision --no-header -rN")
class TestWithVision:
    """Tests that require a loaded DrawingInterpreter (LLM model).

    Run with: pytest -m vision
    """

    def test_generator_loop_smoke(self):
        from meshxcad.vision import DrawingInterpreter
        interp = DrawingInterpreter()
        from meshxcad.adversarial_drawing import adversarial_generator_loop
        result = adversarial_generator_loop(interp, n_rounds=2)
        assert "results" in result
        assert "failure_db" in result

    def test_perturbation_loop_smoke(self):
        from meshxcad.vision import DrawingInterpreter
        interp = DrawingInterpreter()
        from meshxcad.adversarial_drawing import adversarial_perturbation_loop
        progs = [_make_box_program()]
        result = adversarial_perturbation_loop(interp, progs, n_perturbations=1)
        assert "results" in result

    def test_mutation_loop_smoke(self):
        from meshxcad.vision import DrawingInterpreter
        interp = DrawingInterpreter()
        from meshxcad.adversarial_drawing import adversarial_mutation_loop
        progs = [_make_cylinder_program()]
        result = adversarial_mutation_loop(interp, progs, n_mutations=2)
        assert "results" in result
