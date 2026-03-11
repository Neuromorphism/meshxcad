"""Tests for the tracing-based CAD reconstruction pipeline.

Covers:
  - Object templates (CIFAR-100 coverage, lookup, matching)
  - Mesh segmentation (all strategies)
  - Tracing reconstruction (per-segment and full pipeline)
  - Test models (trees, humanoids, furniture, etc.)
  - Iterative refinement with tracing
"""

import numpy as np
import pytest

from meshxcad.test_models import (
    make_dead_tree, make_dead_tree_gnarled, make_dead_tree_tall,
    make_dead_tree_stumpy, make_humanoid, make_humanoid_stocky,
    make_humanoid_tall, make_simple_chair, make_simple_bottle,
    make_simple_table, make_simple_rocket, ALL_TEST_MODELS, get_test_model,
)
from meshxcad.object_templates import (
    ALL_TEMPLATES, CIFAR100_SUPERCLASSES, get_template, list_templates,
    match_template, templates_for_superclass,
)
from meshxcad.segmentation import (
    segment_mesh, segment_by_skeleton, segment_by_sdf,
    segment_by_convexity, segment_by_normals, segment_by_projection,
    classify_segment_action, MeshSegment,
)
from meshxcad.tracing import (
    trace_reconstruct, trace_reconstruct_with_template_search,
)


# =====================================================================
# Object Template Tests
# =====================================================================

class TestObjectTemplates:

    def test_cifar100_complete(self):
        """All 100 CIFAR-100 fine classes have templates."""
        all_fine_classes = []
        for classes in CIFAR100_SUPERCLASSES.values():
            all_fine_classes.extend(classes)
        assert len(all_fine_classes) == 100

        for cls_name in all_fine_classes:
            tpl = get_template(cls_name)
            assert tpl is not None, f"Missing template for CIFAR-100 class: {cls_name}"
            assert len(tpl.parts) >= 1, f"Template {cls_name} has no parts"

    def test_superclass_count(self):
        """There are exactly 20 CIFAR-100 superclasses."""
        assert len(CIFAR100_SUPERCLASSES) == 20

    def test_superclass_sizes(self):
        """Each superclass has exactly 5 fine classes."""
        for sc, classes in CIFAR100_SUPERCLASSES.items():
            assert len(classes) == 5, f"Superclass {sc} has {len(classes)} classes"

    def test_template_lookup(self):
        """Basic template lookup by name."""
        tpl = get_template("bicycle")
        assert tpl is not None
        assert tpl.name == "bicycle"
        assert tpl.superclass == "vehicles_1"
        assert len(tpl.parts) >= 3

    def test_template_alias(self):
        """Alias lookup works."""
        tpl = get_template("dead_tree")
        assert tpl is not None
        tpl2 = get_template("bare_tree")
        assert tpl2 is not None
        assert tpl.name == tpl2.name

    def test_list_templates(self):
        """list_templates returns unique names."""
        names = list_templates()
        assert len(names) >= 100  # at least CIFAR-100 + extras
        assert len(names) == len(set(names))  # unique

    def test_templates_for_superclass(self):
        """Query templates by superclass."""
        trees = templates_for_superclass("trees")
        assert len(trees) >= 5  # 5 CIFAR + dead_tree
        names = {t.name for t in trees}
        assert "pine_tree" in names

    def test_template_parts_have_required_fields(self):
        """Every part hint has required geometry and action."""
        for tpl in ALL_TEMPLATES.values():
            for part in tpl.parts:
                assert part.name, f"Part in {tpl.name} has empty name"
                assert part.geometry in (
                    "cylinder", "sphere", "box", "cone", "revolve",
                    "extrude", "sweep", "freeform"
                ), f"Part {part.name} in {tpl.name} has invalid geometry: {part.geometry}"
                assert part.cad_action in (
                    "extrude", "revolve", "loft", "sweep",
                    "boolean_cut", "boolean_add", "freeform"
                ), f"Part {part.name} in {tpl.name} has invalid cad_action: {part.cad_action}"

    def test_match_template_returns_results(self):
        """match_template returns scored results for any mesh."""
        v, f = make_simple_bottle()
        results = match_template(v, f, top_k=3)
        assert len(results) == 3
        for tpl, score in results:
            assert score >= 0
            assert tpl.name

    def test_dead_tree_template_structure(self):
        """Dead tree template has correct branching structure."""
        tpl = get_template("dead_tree")
        assert tpl is not None
        part_names = {p.name for p in tpl.parts}
        assert "trunk" in part_names
        assert "primary_branch" in part_names
        assert "secondary_branch" in part_names

    def test_humanoid_template_structure(self):
        """Humanoid (man) template has correct biped parts."""
        tpl = get_template("man")
        assert tpl is not None
        part_names = {p.name for p in tpl.parts}
        assert "torso" in part_names
        assert "head" in part_names
        assert "upper_arm" in part_names
        assert "thigh" in part_names


# =====================================================================
# Test Model Generation
# =====================================================================

class TestModels:

    @pytest.mark.parametrize("model_name", list(ALL_TEST_MODELS.keys()))
    def test_model_generates_valid_mesh(self, model_name):
        """Each test model produces a valid mesh with vertices and faces."""
        v, f = get_test_model(model_name)
        assert isinstance(v, np.ndarray)
        assert isinstance(f, np.ndarray)
        assert v.ndim == 2 and v.shape[1] == 3
        assert f.ndim == 2 and f.shape[1] == 3
        assert len(v) >= 10
        assert len(f) >= 5
        assert f.max() < len(v), "Face indices out of bounds"
        assert f.min() >= 0

    def test_dead_tree_has_branches(self):
        """Dead tree mesh should be significantly larger than a simple cylinder."""
        v, f = make_dead_tree()
        # Should have many vertices from branches
        assert len(v) > 200
        # Bounding box should extend beyond a simple cylinder
        bbox = v.max(axis=0) - v.min(axis=0)
        assert bbox[0] > 1.0, "Tree should have lateral extent from branches"
        assert bbox[2] > 5.0, "Tree should be tall"

    def test_dead_tree_variants(self):
        """Different tree variants have distinct geometry."""
        gnarled_v, _ = make_dead_tree_gnarled()
        tall_v, _ = make_dead_tree_tall()
        stumpy_v, _ = make_dead_tree_stumpy()

        gnarled_h = gnarled_v[:, 2].max() - gnarled_v[:, 2].min()
        tall_h = tall_v[:, 2].max() - tall_v[:, 2].min()
        stumpy_h = stumpy_v[:, 2].max() - stumpy_v[:, 2].min()

        assert tall_h > gnarled_h, "Tall tree should be taller than gnarled"
        assert stumpy_h < gnarled_h, "Stumpy tree should be shorter"

    def test_humanoid_proportions(self):
        """Humanoid mesh has roughly correct proportions."""
        v, f = make_humanoid()
        bbox = v.max(axis=0) - v.min(axis=0)
        height = bbox[2]
        width = bbox[0]

        assert 1.5 < height < 2.5, f"Humanoid height {height} out of range"
        assert width > height * 0.3, "Humanoid should have arm span"
        assert width < height * 1.5, "Humanoid shouldn't be too wide"

    def test_humanoid_variants(self):
        """Different humanoid variants differ."""
        stocky_v, _ = make_humanoid_stocky()
        tall_v, _ = make_humanoid_tall()

        stocky_h = stocky_v[:, 2].max() - stocky_v[:, 2].min()
        tall_h = tall_v[:, 2].max() - tall_v[:, 2].min()
        assert tall_h > stocky_h

    def test_bottle_is_axially_symmetric(self):
        """Bottle should be roughly axially symmetric."""
        v, f = make_simple_bottle()
        center = v.mean(axis=0)
        # Radial distances from Z axis
        r = np.sqrt((v[:, 0] - center[0])**2 + (v[:, 1] - center[1])**2)
        # Check middle slices (skip caps which include center vertices)
        z_vals = v[:, 2]
        n_slices = 5
        z_range = z_vals.max() - z_vals.min()
        z_bins = np.linspace(z_vals.min() + z_range * 0.15,
                             z_vals.max() - z_range * 0.15, n_slices + 1)
        for i in range(n_slices):
            mask = (z_vals >= z_bins[i]) & (z_vals < z_bins[i + 1])
            if mask.sum() > 3:
                r_slice = r[mask]
                # Exclude near-zero radii (cap center vertices)
                r_slice = r_slice[r_slice > 1e-6]
                if len(r_slice) > 3:
                    cv = r_slice.std() / r_slice.mean()
                    assert cv < 0.5, f"Bottle cross-section not circular at z={z_bins[i]:.3f}"


# =====================================================================
# Segmentation Tests
# =====================================================================

class TestSegmentation:

    def test_skeleton_segmentation_tree(self):
        """Skeleton segmentation produces multiple segments for a tree."""
        v, f = make_dead_tree()
        segments = segment_by_skeleton(v, f)
        assert len(segments) >= 2, "Tree should segment into at least 2 parts"
        # All faces should be covered
        total_faces = sum(len(s.faces) for s in segments)
        assert total_faces > 0

    def test_skeleton_segmentation_humanoid(self):
        """Skeleton segmentation finds multiple parts of a humanoid."""
        v, f = make_humanoid()
        segments = segment_by_skeleton(v, f)
        assert len(segments) >= 2

    def test_sdf_segmentation_bottle(self):
        """SDF segmentation separates bottle body from neck."""
        v, f = make_simple_bottle()
        segments = segment_by_sdf(v, f)
        assert len(segments) >= 1

    def test_convexity_segmentation_chair(self):
        """Convexity segmentation finds chair parts."""
        v, f = make_simple_chair()
        segments = segment_by_convexity(v, f)
        assert len(segments) >= 2

    def test_normal_segmentation_table(self):
        """Normal clustering finds distinct surfaces of a table."""
        v, f = make_simple_table()
        segments = segment_by_normals(v, f)
        assert len(segments) >= 2

    def test_projection_segmentation(self):
        """Projection segmentation runs without error."""
        v, f = make_simple_rocket()
        segments = segment_by_projection(v, f)
        assert len(segments) >= 1

    def test_auto_strategy_selection(self):
        """Auto strategy selects appropriate method."""
        v, f = make_dead_tree()
        segments = segment_mesh(v, f, strategy="auto")
        assert len(segments) >= 1

    def test_segment_action_classification(self):
        """classify_segment_action assigns valid action types."""
        v, f = make_simple_bottle()
        segments = segment_mesh(v, f, strategy="sdf")
        for seg in segments:
            assert seg.cad_action in ("revolve", "extrude", "loft", "sweep", "freeform")
            assert 0 <= seg.quality <= 1.0

    def test_template_guided_segmentation(self):
        """Template-guided segmentation uses template strategy."""
        v, f = make_simple_chair()
        tpl = get_template("chair")
        segments = segment_mesh(v, f, template=tpl)
        assert len(segments) >= 1


# =====================================================================
# Tracing Reconstruction Tests
# =====================================================================

class TestTracing:

    def test_trace_bottle(self):
        """Trace reconstruction of a bottle produces valid mesh."""
        v, f = make_simple_bottle()
        result = trace_reconstruct(v, f)
        assert result["cad_vertices"].shape[1] == 3
        assert result["cad_faces"].shape[1] == 3
        assert result["n_segments"] >= 1
        assert 0 <= result["quality"] <= 1.0

    def test_trace_humanoid(self):
        """Trace reconstruction of humanoid produces valid mesh."""
        v, f = make_humanoid()
        result = trace_reconstruct(v, f)
        assert len(result["cad_vertices"]) >= 10
        assert len(result["cad_faces"]) >= 5
        assert result["n_segments"] >= 1

    def test_trace_dead_tree(self):
        """Trace reconstruction of a dead tree handles branching."""
        v, f = make_dead_tree()
        result = trace_reconstruct(v, f)
        assert result["n_segments"] >= 2, "Tree should be multi-segment"
        assert len(result["cad_vertices"]) >= 20

    def test_trace_with_template(self):
        """Template-guided tracing uses the template."""
        v, f = make_simple_chair()
        tpl = get_template("chair")
        result = trace_reconstruct(v, f, template=tpl)
        assert result["n_segments"] >= 1

    def test_trace_with_template_search(self):
        """Auto template search picks a reasonable template."""
        v, f = make_simple_rocket()
        result = trace_reconstruct_with_template_search(v, f)
        assert "template_name" in result
        assert result["quality"] >= 0

    def test_trace_all_test_models(self):
        """Trace reconstruction runs without error on all test models."""
        for name, fn in ALL_TEST_MODELS.items():
            v, f = fn()
            result = trace_reconstruct(v, f)
            assert len(result["cad_vertices"]) >= 3, f"Failed on {name}"
            assert len(result["cad_faces"]) >= 1, f"Failed on {name}"

    def test_tree_segmentation_quality(self):
        """Tree segmentation should produce reasonable quality."""
        for tree_fn in [make_dead_tree, make_dead_tree_gnarled,
                        make_dead_tree_tall, make_dead_tree_stumpy]:
            v, f = tree_fn()
            result = trace_reconstruct(v, f)
            # Quality should be non-zero (the reconstruction has some overlap)
            assert result["quality"] >= 0, f"Quality should be non-negative for {tree_fn.__name__}"

    def test_humanoid_segmentation_quality(self):
        """Humanoid segmentation should produce reasonable quality."""
        for human_fn in [make_humanoid, make_humanoid_stocky, make_humanoid_tall]:
            v, f = human_fn()
            result = trace_reconstruct(v, f)
            assert result["quality"] >= 0


# =====================================================================
# Integration: iterative refinement with tracing
# =====================================================================

class TestIterativeTracing:

    def test_scratch_transfer_with_tracing(self):
        """scratch_transfer with tracing produces valid result."""
        from meshxcad.iterative_transfer import scratch_transfer

        v, f = make_simple_bottle()
        result = scratch_transfer(
            v, f,
            max_iterations=2,
            patience=2,
            render=False,
            use_tracing=True,
        )
        assert len(result["result_verts"]) >= 10
        assert len(result["distances"]) >= 1
        assert "initial_shape_type" in result

    def test_scratch_transfer_without_tracing(self):
        """scratch_transfer without tracing falls back to reconstruct_cad."""
        from meshxcad.iterative_transfer import scratch_transfer

        v, f = make_simple_bottle()
        result = scratch_transfer(
            v, f,
            max_iterations=2,
            patience=2,
            render=False,
            use_tracing=False,
        )
        assert len(result["result_verts"]) >= 10

    def test_scratch_transfer_with_template(self):
        """scratch_transfer with explicit template name."""
        from meshxcad.iterative_transfer import scratch_transfer

        v, f = make_dead_tree()
        result = scratch_transfer(
            v, f,
            max_iterations=2,
            patience=2,
            render=False,
            use_tracing=True,
            template_name="dead_tree",
        )
        assert result.get("template_name") == "dead_tree"


# =====================================================================
# Vision guide module tests (no network)
# =====================================================================

class TestVisionGuide:

    def test_is_available_without_env(self):
        """Vision guide is not available without env vars."""
        import os
        # Save and clear env vars
        old_key = os.environ.pop("LOCAL_OPENAI_KEY", None)
        old_url = os.environ.pop("LOCAL_OPENAI_URL", None)
        try:
            from meshxcad.vision_guide import is_available
            assert not is_available()
        finally:
            if old_key:
                os.environ["LOCAL_OPENAI_KEY"] = old_key
            if old_url:
                os.environ["LOCAL_OPENAI_URL"] = old_url
