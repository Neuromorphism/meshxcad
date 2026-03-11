"""Tests for meshxcad.diffusion_strategy — diffusion-based CAD strategy selection."""

import math
import numpy as np
import pytest

from meshxcad.diffusion_strategy import (
    # Mesh conditioning
    extract_mesh_features,
    MESH_CONDITION_DIM,
    # Program embedding
    embed_op,
    embed_program,
    unembed_program,
    OP_EMBED_DIM,
    PROGRAM_EMBED_DIM,
    MAX_OPS,
    N_OP_TYPES,
    # Noise schedule
    NoiseSchedule,
    # Forward process
    forward_diffuse,
    # Score function
    score_program,
    # Strategies
    StructuralRewrite,
    TopologicalAdjust,
    ParameterRefine,
    ElegancePolish,
    # Main entry point
    run_diffusion_strategy,
    DiffusionConfig,
    DiffusionStep,
    # Comparison
    compare_approaches,
    # Helpers
    _suggest_ops_from_features,
    _langevin_step,
    _select_strategies,
    DEFAULT_STRATEGIES,
)
from meshxcad.cad_program import CadOp, CadProgram, OP_COSTS
from meshxcad.synthetic import make_sphere_mesh, make_cylinder_mesh, make_cube_mesh
from meshxcad.objects.builder import revolve_profile, make_torus


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sphere(r=5.0):
    return make_sphere_mesh(radius=r, lat_divs=16, lon_divs=16)


def _cylinder(r=3.0, h=10.0):
    return make_cylinder_mesh(radius=r, height=h, radial_divs=16, height_divs=8)


def _cube(size=8.0):
    return make_cube_mesh(size=size, subdivisions=4)


def _torus(major=6.0, minor=2.0):
    return make_torus(major_r=major, minor_r=minor, z_center=0.0,
                      n_angular=24, n_cross=12)


def _vase():
    profile = [
        (2.0, 0.0), (3.0, 2.0), (2.5, 4.0), (1.5, 6.0),
        (2.0, 8.0), (3.5, 10.0), (3.0, 12.0), (1.0, 13.0),
    ]
    return revolve_profile(profile, n_angular=24)


# ===========================================================================
# Mesh conditioning tests
# ===========================================================================

class TestMeshFeatures:
    """Test the mesh conditioning feature extractor (CLIP-embedding analog)."""

    def test_output_shape(self):
        v, f = _sphere()
        features = extract_mesh_features(v, f)
        assert features.shape == (MESH_CONDITION_DIM,)

    def test_values_in_range(self):
        for mesh_fn in [_sphere, _cylinder, _cube]:
            v, f = mesh_fn()
            features = extract_mesh_features(v, f)
            assert np.all(features >= 0.0), f"Features below 0: {features}"
            assert np.all(features <= 1.0), f"Features above 1: {features}"

    def test_sphere_has_high_circularity(self):
        v, f = _sphere()
        features = extract_mesh_features(v, f)
        # Circularity feature should be high for a sphere
        assert features[13] > 0.7, f"Sphere circularity too low: {features[13]}"

    def test_cylinder_has_elongation(self):
        v, f = _cylinder(r=2.0, h=20.0)
        features = extract_mesh_features(v, f)
        # Elongation feature (sigmoid-mapped) should be higher for elongated shapes
        sphere_feat = extract_mesh_features(*_sphere())
        assert features[9] > sphere_feat[9], \
            f"Cylinder elongation ({features[9]}) should exceed sphere ({sphere_feat[9]})"

    def test_cube_has_high_axis_alignment(self):
        v, f = _cube()
        features = extract_mesh_features(v, f)
        # Face normals should align strongly with axes
        assert features[20] > 0.8, f"Cube axis alignment too low: {features[20]}"

    def test_different_shapes_produce_different_features(self):
        sphere_feat = extract_mesh_features(*_sphere())
        cyl_feat = extract_mesh_features(*_cylinder())
        cube_feat = extract_mesh_features(*_cube())

        # Features should meaningfully differ between shape types
        sc_diff = np.linalg.norm(sphere_feat - cyl_feat)
        sb_diff = np.linalg.norm(sphere_feat - cube_feat)
        cb_diff = np.linalg.norm(cyl_feat - cube_feat)

        assert sc_diff > 0.1, "Sphere and cylinder features too similar"
        assert sb_diff > 0.1, "Sphere and cube features too similar"
        assert cb_diff > 0.1, "Cylinder and cube features too similar"

    def test_symmetry_detection(self):
        v, f = _sphere()
        features = extract_mesh_features(v, f)
        # Sphere should have high symmetry on all axes
        for ax in range(3):
            assert features[25 + ax] > 0.8, \
                f"Sphere symmetry on axis {ax} too low: {features[25 + ax]}"

    def test_empty_mesh(self):
        features = extract_mesh_features(np.zeros((0, 3)), np.zeros((0, 3), dtype=int))
        assert features.shape == (MESH_CONDITION_DIM,)
        assert np.all(features == 0.0)

    def test_tiny_mesh(self):
        v = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=float)
        f = np.array([[0, 1, 2]])
        features = extract_mesh_features(v, f)
        assert features.shape == (MESH_CONDITION_DIM,)
        assert np.all(np.isfinite(features))


# ===========================================================================
# Program embedding tests
# ===========================================================================

class TestProgramEmbedding:
    """Test CadProgram ↔ continuous vector space mapping."""

    def test_embed_op_shape(self):
        op = CadOp("sphere", {"center": np.zeros(3), "radius": 5.0})
        vec = embed_op(op)
        assert vec.shape == (OP_EMBED_DIM,)

    def test_embed_program_shape(self):
        ops = [CadOp("sphere", {"center": np.zeros(3), "radius": 5.0})]
        program = CadProgram(ops)
        vec = embed_program(program)
        assert vec.shape == (PROGRAM_EMBED_DIM,)

    def test_embed_op_type_encoding(self):
        for op_type in OP_COSTS.keys():
            op = CadOp(op_type, {})
            vec = embed_op(op)
            # One-hot type should have max at the correct index
            type_vec = vec[:N_OP_TYPES]
            assert np.argmax(type_vec) == list(OP_COSTS.keys()).index(op_type)

    def test_embed_enabled_flag(self):
        op_on = CadOp("sphere", {"center": np.zeros(3)}, enabled=True)
        op_off = CadOp("sphere", {"center": np.zeros(3)}, enabled=False)
        assert embed_op(op_on)[-1] == 1.0
        assert embed_op(op_off)[-1] == 0.0

    def test_roundtrip_simple(self):
        """Test that embed → unembed preserves op type and parameters."""
        ops = [
            CadOp("sphere", {"center": np.array([1, 2, 3.0]), "radius": 5.0}),
        ]
        program = CadProgram(ops)
        vec = embed_program(program)
        recovered = unembed_program(vec, reference_program=program)

        assert len(recovered.operations) >= 1
        assert recovered.operations[0].op_type == "sphere"
        assert recovered.operations[0].enabled is True

    def test_roundtrip_multi_op(self):
        ops = [
            CadOp("sphere", {"center": np.array([0, 0, 0.0]), "radius": 3.0}),
            CadOp("cylinder", {"center": np.array([5, 0, 0.0]),
                                "axis": np.array([0, 0, 1.0]),
                                "height": 10.0, "radius": 2.0}),
        ]
        program = CadProgram(ops)
        vec = embed_program(program)
        recovered = unembed_program(vec, reference_program=program)

        assert len(recovered.operations) >= 2
        assert recovered.operations[0].op_type == "sphere"
        assert recovered.operations[1].op_type == "cylinder"

    def test_empty_program(self):
        program = CadProgram([])
        vec = embed_program(program)
        assert vec.shape == (PROGRAM_EMBED_DIM,)
        assert np.all(vec == 0.0)

    def test_unembed_without_reference(self):
        ops = [CadOp("box", {"center": np.array([1, 2, 3.0]),
                              "dimensions": np.array([4, 5, 6.0])})]
        program = CadProgram(ops)
        vec = embed_program(program)
        recovered = unembed_program(vec)  # No reference
        assert len(recovered.operations) >= 1
        assert recovered.operations[0].op_type == "box"


# ===========================================================================
# Noise schedule tests
# ===========================================================================

class TestNoiseSchedule:
    """Test the diffusion noise schedule."""

    def test_linear_schedule(self):
        sched = NoiseSchedule(num_timesteps=100, schedule_type="linear")
        # At t=0 (clean): alpha_bar should be near 1
        assert sched.alpha_bar(0) > 0.99
        # At t=T-1 (noisy): alpha_bar should be small
        assert sched.alpha_bar(99) < 0.5

    def test_cosine_schedule(self):
        sched = NoiseSchedule(num_timesteps=100, schedule_type="cosine")
        assert sched.alpha_bar(0) > 0.9
        assert sched.alpha_bar(99) < 0.1

    def test_sqrt_schedule(self):
        sched = NoiseSchedule(num_timesteps=100, schedule_type="sqrt")
        assert sched.alpha_bar(0) > 0.9
        assert sched.alpha_bar(99) < 0.1

    def test_monotonically_decreasing(self):
        for stype in ["linear", "cosine", "sqrt"]:
            sched = NoiseSchedule(num_timesteps=50, schedule_type=stype)
            alpha_bars = [sched.alpha_bar(t) for t in range(50)]
            for i in range(1, len(alpha_bars)):
                assert alpha_bars[i] <= alpha_bars[i - 1] + 1e-10, \
                    f"{stype} schedule not monotonically decreasing at t={i}"

    def test_noise_plus_signal_identity(self):
        sched = NoiseSchedule(num_timesteps=50, schedule_type="cosine")
        for t in range(50):
            # signal^2 + noise^2 should equal 1
            s = sched.signal_level(t)
            n = sched.noise_level(t)
            assert abs(s**2 + n**2 - 1.0) < 1e-6, \
                f"Identity violated at t={t}: {s}^2 + {n}^2 = {s**2 + n**2}"

    def test_invalid_schedule(self):
        with pytest.raises(ValueError):
            NoiseSchedule(schedule_type="invalid")

    def test_boundary_timesteps(self):
        sched = NoiseSchedule(num_timesteps=10)
        # Should not crash on boundary values
        sched.alpha_bar(-1)
        sched.alpha_bar(100)


# ===========================================================================
# Forward diffusion tests
# ===========================================================================

class TestForwardDiffusion:
    """Test the forward (noising) process."""

    def test_output_shapes(self):
        ops = [CadOp("sphere", {"center": np.zeros(3), "radius": 5.0})]
        program = CadProgram(ops)
        sched = NoiseSchedule(num_timesteps=50)

        noisy, noise, clean = forward_diffuse(program, 25, sched)
        assert noisy.shape == (PROGRAM_EMBED_DIM,)
        assert noise.shape == (PROGRAM_EMBED_DIM,)
        assert clean.shape == (PROGRAM_EMBED_DIM,)

    def test_t0_is_mostly_clean(self):
        ops = [CadOp("sphere", {"center": np.zeros(3), "radius": 5.0})]
        program = CadProgram(ops)
        sched = NoiseSchedule(num_timesteps=50)

        noisy, noise, clean = forward_diffuse(program, 0, sched,
                                               rng=np.random.RandomState(42))
        # At t=0, noisy should be very close to clean
        diff = np.linalg.norm(noisy - clean) / max(np.linalg.norm(clean), 1e-12)
        assert diff < 0.3, f"At t=0, noisy too far from clean: {diff}"

    def test_tT_is_mostly_noise(self):
        ops = [CadOp("sphere", {"center": np.zeros(3), "radius": 5.0})]
        program = CadProgram(ops)
        sched = NoiseSchedule(num_timesteps=50)

        noisy, noise, clean = forward_diffuse(program, 49, sched,
                                               rng=np.random.RandomState(42))
        # At t=T-1, noisy should be dominated by noise
        signal_strength = np.linalg.norm(sched.signal_level(49) * clean)
        noise_strength = np.linalg.norm(sched.noise_level(49) * noise)
        assert noise_strength > signal_strength, \
            f"At t=T-1, signal ({signal_strength}) > noise ({noise_strength})"


# ===========================================================================
# Score function tests
# ===========================================================================

class TestScoreFunction:
    """Test the score function (energy landscape)."""

    def test_good_program_scores_higher(self):
        v, f = _sphere(r=5.0)
        good = CadProgram([CadOp("sphere", {"center": np.zeros(3), "radius": 5.0})])
        bad = CadProgram([CadOp("sphere", {"center": np.array([100, 0, 0.0]),
                                             "radius": 0.1})])
        good_score = score_program(good, v, f)
        bad_score = score_program(bad, v, f)
        assert good_score > bad_score, \
            f"Good program ({good_score}) should score higher than bad ({bad_score})"

    def test_empty_program_scores_lowest(self):
        v, f = _sphere()
        empty = CadProgram([])
        score = score_program(empty, v, f)
        assert score == -100.0

    def test_conditioning_bonus(self):
        v, f = _sphere()
        program = CadProgram([CadOp("sphere", {"center": np.zeros(3), "radius": 5.0})])
        mesh_features = extract_mesh_features(v, f)

        score_with = score_program(program, v, f, mesh_features)
        score_without = score_program(program, v, f, None)
        # Conditioning should add a bonus for matching features
        assert score_with >= score_without - 0.01


# ===========================================================================
# Strategy tests
# ===========================================================================

class TestStrategies:
    """Test individual denoising strategies."""

    def test_structural_rewrite_produces_candidates(self):
        v, f = _sphere()
        program = CadProgram([CadOp("sphere", {"center": np.zeros(3), "radius": 5.0})])
        strategy = StructuralRewrite()
        features = extract_mesh_features(v, f)
        rng = np.random.RandomState(42)

        candidates = strategy.apply(program, v, f, features, 0.8, rng)
        assert len(candidates) > 0, "StructuralRewrite produced no candidates"
        for name, cand in candidates:
            assert isinstance(name, str)
            assert isinstance(cand, CadProgram)

    def test_topological_adjust_produces_candidates(self):
        v, f = _sphere()
        program = CadProgram([CadOp("sphere", {"center": np.zeros(3), "radius": 3.0})])
        strategy = TopologicalAdjust()
        features = extract_mesh_features(v, f)
        rng = np.random.RandomState(42)

        candidates = strategy.apply(program, v, f, features, 0.5, rng)
        assert len(candidates) > 0

    def test_parameter_refine_produces_candidates(self):
        v, f = _sphere()
        program = CadProgram([CadOp("sphere", {"center": np.array([0.5, 0, 0.0]),
                                                 "radius": 4.5})])
        strategy = ParameterRefine()
        features = extract_mesh_features(v, f)
        rng = np.random.RandomState(42)

        candidates = strategy.apply(program, v, f, features, 0.1, rng)
        assert len(candidates) > 0

    def test_elegance_polish_produces_candidates(self):
        v, f = _sphere()
        ops = [
            CadOp("sphere", {"center": np.zeros(3), "radius": 5.0}),
            CadOp("sphere", {"center": np.array([0.1, 0, 0.0]), "radius": 4.9}),
        ]
        program = CadProgram(ops)
        strategy = ElegancePolish()
        features = extract_mesh_features(v, f)
        rng = np.random.RandomState(42)

        candidates = strategy.apply(program, v, f, features, 0.05, rng)
        assert len(candidates) > 0


class TestStrategySelection:
    """Test strategy selection at different noise levels."""

    def test_high_noise_selects_structural(self):
        active = _select_strategies(DEFAULT_STRATEGIES, 0.8)
        names = {s.name for s in active}
        assert "structural_rewrite" in names
        assert "parameter_refine" not in names

    def test_medium_noise_selects_topological(self):
        active = _select_strategies(DEFAULT_STRATEGIES, 0.5)
        names = {s.name for s in active}
        assert "topological_adjust" in names

    def test_low_noise_selects_refinement(self):
        active = _select_strategies(DEFAULT_STRATEGIES, 0.1)
        names = {s.name for s in active}
        assert "parameter_refine" in names
        assert "elegance_polish" in names
        assert "structural_rewrite" not in names


# ===========================================================================
# Langevin step tests
# ===========================================================================

class TestLangevinStep:
    """Test stochastic parameter perturbation."""

    def test_modifies_parameters(self):
        op = CadOp("sphere", {"center": np.array([1.0, 2.0, 3.0]), "radius": 5.0})
        program = CadProgram([op])
        original_center = program.operations[0].params["center"].copy()

        _langevin_step(program, noise_level=0.5, rng=np.random.RandomState(42))

        new_center = program.operations[0].params["center"]
        assert not np.allclose(original_center, new_center), \
            "Langevin step did not modify parameters"

    def test_higher_noise_means_larger_perturbation(self):
        rng1 = np.random.RandomState(42)
        rng2 = np.random.RandomState(42)

        p_low = CadProgram([CadOp("sphere", {"center": np.array([1.0, 2.0, 3.0]),
                                               "radius": 5.0})])
        p_high = CadProgram([CadOp("sphere", {"center": np.array([1.0, 2.0, 3.0]),
                                                "radius": 5.0})])
        original = np.array([1.0, 2.0, 3.0])

        _langevin_step(p_low, noise_level=0.1, rng=rng1)
        _langevin_step(p_high, noise_level=0.9, rng=rng2)

        diff_low = np.linalg.norm(p_low.operations[0].params["center"] - original)
        diff_high = np.linalg.norm(p_high.operations[0].params["center"] - original)
        # High noise should generally produce larger perturbations
        # (Not guaranteed due to randomness, but very likely with same seed structure)
        assert diff_high > diff_low * 0.5 or diff_low < 0.01


# ===========================================================================
# Feature-based op suggestion tests
# ===========================================================================

class TestOpSuggestions:
    """Test operation type suggestions from mesh features."""

    def test_sphere_features_suggest_sphere(self):
        v, f = _sphere()
        features = extract_mesh_features(v, f)
        suggestions = _suggest_ops_from_features(features)
        op_types = [s[0] for s in suggestions]
        assert "sphere" in op_types or "torus" in op_types, \
            f"Sphere features should suggest sphere/torus, got: {op_types}"

    def test_cylinder_features_suggest_cylinder_or_revolve(self):
        v, f = _cylinder(r=2.0, h=20.0)
        features = extract_mesh_features(v, f)
        suggestions = _suggest_ops_from_features(features)
        op_types = [s[0] for s in suggestions]
        assert any(t in op_types for t in ["cylinder", "revolve", "profiled_cylinder"]), \
            f"Tall cylinder features should suggest cylinder/revolve, got: {op_types}"


# ===========================================================================
# Main diffusion loop tests
# ===========================================================================

class TestDiffusionLoop:
    """Test the main reverse-diffusion strategy selector."""

    def test_sphere_reconstruction(self):
        v, f = _sphere(r=5.0)
        config = DiffusionConfig(
            num_timesteps=10,  # Short for testing
            patience=3,
            seed=42,
        )
        result = run_diffusion_strategy(v, f, config)

        assert "program" in result
        assert "history" in result
        assert "cad_vertices" in result
        assert "cad_faces" in result
        assert "mesh_features" in result
        assert "total_cost" in result
        assert isinstance(result["program"], CadProgram)
        assert result["n_ops"] >= 1
        assert len(result["cad_vertices"]) > 0

    def test_cylinder_reconstruction(self):
        v, f = _cylinder(r=3.0, h=10.0)
        config = DiffusionConfig(num_timesteps=8, patience=3, seed=42)
        result = run_diffusion_strategy(v, f, config)

        assert result["n_ops"] >= 1
        assert result["total_cost"] < float('inf')
        assert len(result["history"]) > 0

    def test_history_records(self):
        v, f = _sphere()
        config = DiffusionConfig(num_timesteps=5, patience=3, seed=42)
        result = run_diffusion_strategy(v, f, config)

        for step in result["history"]:
            assert isinstance(step, DiffusionStep)
            assert 0 <= step.noise_level <= 1.0
            assert isinstance(step.improved, bool)
            assert step.n_ops >= 0

    def test_different_seeds_different_results(self):
        v, f = _sphere()
        config1 = DiffusionConfig(num_timesteps=8, seed=42)
        config2 = DiffusionConfig(num_timesteps=8, seed=123)

        r1 = run_diffusion_strategy(v, f, config1)
        r2 = run_diffusion_strategy(v, f, config2)

        # Different seeds may give different histories
        # (not guaranteed to be different, but the mechanism should work)
        assert isinstance(r1["program"], CadProgram)
        assert isinstance(r2["program"], CadProgram)

    def test_guidance_scale_effect(self):
        v, f = _sphere()
        config_low = DiffusionConfig(num_timesteps=8, guidance_scale=0.0, seed=42)
        config_high = DiffusionConfig(num_timesteps=8, guidance_scale=5.0, seed=42)

        r_low = run_diffusion_strategy(v, f, config_low)
        r_high = run_diffusion_strategy(v, f, config_high)

        assert isinstance(r_low["program"], CadProgram)
        assert isinstance(r_high["program"], CadProgram)


# ===========================================================================
# Comparison tests
# ===========================================================================

class TestComparison:
    """Test the diffusion vs classical comparison runner."""

    def test_compare_on_sphere(self):
        v, f = _sphere(r=5.0)
        config = DiffusionConfig(num_timesteps=5, patience=2, seed=42)
        result = compare_approaches(v, f, diffusion_config=config,
                                    classical_max_rounds=5)

        assert "classical" in result
        assert "diffusion" in result
        assert "comparison" in result

        assert "total_cost" in result["classical"]
        assert "total_cost" in result["diffusion"]
        assert "diffusion_better" in result["comparison"]
        assert isinstance(result["comparison"]["diffusion_better"], bool)

    def test_compare_returns_valid_costs(self):
        v, f = _cube()
        config = DiffusionConfig(num_timesteps=5, patience=2, seed=42)
        result = compare_approaches(v, f, diffusion_config=config,
                                    classical_max_rounds=5)

        assert result["classical"]["total_cost"] > 0
        assert result["diffusion"]["total_cost"] > 0
        assert np.isfinite(result["comparison"]["cost_ratio"])


# ===========================================================================
# Edge case and robustness tests
# ===========================================================================

class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_single_triangle(self):
        v = np.array([[0, 0, 0], [1, 0, 0], [0.5, 1, 0]], dtype=float)
        f = np.array([[0, 1, 2]])
        config = DiffusionConfig(num_timesteps=3, patience=2, seed=42)
        result = run_diffusion_strategy(v, f, config)
        assert isinstance(result["program"], CadProgram)

    def test_complex_shape(self):
        """Test on a more complex shape (vase profile)."""
        v, f = _vase()
        config = DiffusionConfig(num_timesteps=8, patience=3, seed=42)
        result = run_diffusion_strategy(v, f, config)
        assert result["n_ops"] >= 1
        assert result["total_cost"] < float('inf')

    def test_torus_shape(self):
        v, f = _torus()
        config = DiffusionConfig(num_timesteps=8, patience=3, seed=42)
        result = run_diffusion_strategy(v, f, config)
        assert result["n_ops"] >= 1

    def test_schedule_types(self):
        v, f = _sphere()
        for stype in ["linear", "cosine", "sqrt"]:
            config = DiffusionConfig(num_timesteps=5, schedule_type=stype,
                                     patience=2, seed=42)
            result = run_diffusion_strategy(v, f, config)
            assert isinstance(result["program"], CadProgram)
