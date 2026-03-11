"""Tests for the drawing pipeline (Phase 0-2): renderer, comparator, spec, builder.

All tests are pure geometry — no LLM required.
"""

import numpy as np
import pytest
from meshxcad.drawing import (
    render_orthographic,
    render_drawing_sheet,
    extract_visible_edges,
    _compute_mesh_dimensions,
    _get_view_dimensions,
)
from meshxcad.drawing_compare import (
    compare_drawings,
    extract_edge_pixels,
    chamfer_distance_2d,
)
from meshxcad.drawing_spec import (
    DrawingSpec, Dimension, Feature, ViewSpec,
)
from meshxcad.drawing_to_cad import drawing_to_cad
from meshxcad.synthetic import make_cube_mesh, make_sphere_mesh, make_cylinder_mesh


# ===================================================================
# TestOrthographicRenderer
# ===================================================================

class TestOrthographicRenderer:

    def test_box_front_view(self):
        """Front view of box should produce edges (non-blank image)."""
        v, f = make_cube_mesh(10.0, 4)
        img = render_orthographic(v, f, "front", 256)
        assert img.shape == (256, 256)
        assert img.dtype == np.uint8
        # Should have black pixels (edges)
        assert np.sum(img < 128) > 10

    def test_box_all_views(self):
        """Box should produce edges in all three standard views."""
        v, f = make_cube_mesh(10.0, 4)
        for view in ("front", "side", "top"):
            img = render_orthographic(v, f, view, 256)
            assert np.sum(img < 128) > 10, f"No edges in {view} view"

    def test_cylinder_front_view(self):
        """Front view of cylinder should show a rectangular outline."""
        v, f = make_cylinder_mesh(5.0, 15.0, 24, 10)
        img = render_orthographic(v, f, "front", 256)
        assert img.shape == (256, 256)
        assert np.sum(img < 128) > 10

    def test_cylinder_top_view(self):
        """Top view of cylinder should show a circular outline."""
        v, f = make_cylinder_mesh(5.0, 15.0, 24, 10)
        img = render_orthographic(v, f, "top", 256)
        assert np.sum(img < 128) > 10

    def test_sphere_all_views(self):
        """All views of sphere should produce a circular silhouette."""
        v, f = make_sphere_mesh(5.0, 20, 20)
        for view in ("front", "side", "top"):
            img = render_orthographic(v, f, view, 256)
            assert np.sum(img < 128) > 10, f"No edges in {view} view"

    def test_output_is_binary_ish(self):
        """Output image should be mostly black-and-white."""
        v, f = make_cube_mesh(10.0, 4)
        img = render_orthographic(v, f, "front", 256)
        # Most pixels should be either very white (>200) or very dark (<50)
        bw_pixels = np.sum((img > 200) | (img < 50))
        assert bw_pixels > 0.9 * img.size

    def test_image_size(self):
        """Output matches requested image_size."""
        v, f = make_cube_mesh(10.0, 4)
        for size in (128, 256, 512):
            img = render_orthographic(v, f, "front", size)
            assert img.shape == (size, size)

    def test_multi_view_sheet(self):
        """3 views on one sheet, correct dimensions."""
        v, f = make_cube_mesh(10.0, 4)
        sheet = render_drawing_sheet(v, f, ("front", "side", "top"), 256)
        assert sheet.ndim == 3
        assert sheet.shape[2] == 3  # RGB
        assert sheet.shape[0] == 512  # 2 rows * 256
        assert sheet.shape[1] == 512  # 2 cols * 256

    def test_custom_view_angle(self):
        """Custom (elevation, azimuth) view should work."""
        v, f = make_cube_mesh(10.0, 4)
        img = render_orthographic(v, f, (30, 45), 256)
        assert img.shape == (256, 256)
        assert np.sum(img < 128) > 10

    def test_extract_edges_returns_list(self):
        """extract_visible_edges returns a list of segment pairs."""
        v, f = make_cube_mesh(10.0, 4)
        edges = extract_visible_edges(v, f, "front")
        assert isinstance(edges, list)
        assert len(edges) > 0
        # Each edge is ((x1,y1), (x2,y2))
        for (x1, y1), (x2, y2) in edges:
            assert 0 <= x1 <= 1
            assert 0 <= y1 <= 1


# ===================================================================
# TestDrawingComparison
# ===================================================================

class TestDrawingComparison:

    def test_identical_drawings(self):
        """Same drawing compared to itself → perfect scores."""
        v, f = make_cube_mesh(10.0, 4)
        img = render_orthographic(v, f, "front", 256)
        result = compare_drawings(img, img)
        assert result["pixel_iou"] == 1.0
        assert result["chamfer_distance"] == 0.0
        assert result["edge_precision"] == 1.0
        assert result["edge_recall"] == 1.0

    def test_different_drawings(self):
        """Box vs cylinder → low IoU."""
        v1, f1 = make_cube_mesh(10.0, 4)
        v2, f2 = make_cylinder_mesh(5.0, 15.0, 24, 10)
        img1 = render_orthographic(v1, f1, "front", 256)
        img2 = render_orthographic(v2, f2, "front", 256)
        result = compare_drawings(img1, img2)
        assert result["pixel_iou"] < 0.8  # not identical

    def test_similar_drawings(self):
        """Two similar boxes → high precision/recall."""
        v1, f1 = make_cube_mesh(10.0, 4)
        v2, f2 = make_cube_mesh(10.5, 4)
        img1 = render_orthographic(v1, f1, "front", 256)
        img2 = render_orthographic(v2, f2, "front", 256)
        result = compare_drawings(img1, img2)
        assert result["edge_precision"] > 0.5
        assert result["edge_recall"] > 0.5

    def test_chamfer_is_symmetric(self):
        """Chamfer distance should be approximately symmetric."""
        v1, f1 = make_cube_mesh(10.0, 4)
        v2, f2 = make_cylinder_mesh(5.0, 15.0, 24, 10)
        img1 = render_orthographic(v1, f1, "front", 256)
        img2 = render_orthographic(v2, f2, "front", 256)
        r1 = compare_drawings(img1, img2)
        r2 = compare_drawings(img2, img1)
        assert abs(r1["chamfer_distance"] - r2["chamfer_distance"]) < 1.0

    def test_empty_images(self):
        """Blank white images → IoU=1 (no edges in either)."""
        blank = np.full((256, 256), 255, dtype=np.uint8)
        result = compare_drawings(blank, blank)
        assert result["pixel_iou"] == 1.0

    def test_extract_edge_pixels_rgb(self):
        """extract_edge_pixels works on RGB images."""
        img = np.full((64, 64, 3), 255, dtype=np.uint8)
        img[10, 20] = [0, 0, 0]  # one black pixel
        pts = extract_edge_pixels(img)
        assert len(pts) == 1
        assert pts[0, 0] == 10 and pts[0, 1] == 20


# ===================================================================
# TestDrawingSpec
# ===================================================================

class TestDrawingSpec:

    def test_create_spec(self):
        """Can create DrawingSpec with all fields."""
        spec = DrawingSpec(
            description="Test cylinder",
            object_type="cylinder",
            views=[ViewSpec(view_type="front", features=[
                Feature("cylinder", "front", (0.5, 0.5), (1.0, 1.0)),
            ])],
            dimensions=[Dimension(50.0, "mm", "diameter", "body", "front"),
                        Dimension(30.0, "mm", "height", "body", "front")],
            symmetry="axial",
            overall_size=(50, 30, 50),
        )
        assert spec.object_type == "cylinder"
        assert len(spec.dimensions) == 2

    def test_serialize_roundtrip(self):
        """DrawingSpec roundtrips through JSON."""
        spec = DrawingSpec(
            description="Test box",
            object_type="box",
            views=[ViewSpec("front")],
            dimensions=[Dimension(10.0, "mm", "width", "body", "front")],
            symmetry="none",
            overall_size=(10, 10, 10),
        )
        json_str = spec.to_json()
        spec2 = DrawingSpec.from_json(json_str)
        assert spec2.object_type == "box"
        assert spec2.dimensions[0].value == 10.0
        assert spec2.overall_size == (10, 10, 10)

    def test_from_dict(self):
        """DrawingSpec.from_dict handles nested structures."""
        d = {
            "description": "flange",
            "object_type": "flange",
            "views": [{"view_type": "top", "features": [
                {"feature_type": "hole", "view": "top",
                 "center_2d": [0.3, 0.3], "extent_2d": [0.1, 0.1]},
            ]}],
            "dimensions": [{"value": 80, "unit": "mm", "measurement": "diameter"}],
            "symmetry": "axial",
            "overall_size": [80, 20, 80],
        }
        spec = DrawingSpec.from_dict(d)
        assert spec.object_type == "flange"
        assert len(spec.views) == 1
        assert spec.views[0].features[0].feature_type == "hole"


# ===================================================================
# TestDrawingToCad
# ===================================================================

class TestDrawingToCad:

    def test_box_from_spec(self):
        """3 rectangular views with width/height/depth → box CadOp."""
        spec = DrawingSpec(
            object_type="box",
            dimensions=[
                Dimension(20, "mm", "width"),
                Dimension(15, "mm", "height"),
                Dimension(10, "mm", "depth"),
            ],
            symmetry="none",
            overall_size=(20, 15, 10),
        )
        prog = drawing_to_cad(spec)
        assert prog.n_enabled() >= 1
        assert prog.operations[0].op_type == "box"
        p = prog.operations[0].params
        assert abs(p["dimensions"][0] - 20) < 0.1
        assert abs(p["dimensions"][2] - 15) < 0.1

    def test_cylinder_from_spec(self):
        """Axial symmetric with diameter/height → cylinder CadOp."""
        spec = DrawingSpec(
            object_type="cylinder",
            dimensions=[
                Dimension(50, "mm", "diameter"),
                Dimension(30, "mm", "height"),
            ],
            symmetry="axial",
            overall_size=(50, 30, 50),
        )
        prog = drawing_to_cad(spec)
        assert prog.operations[0].op_type == "cylinder"
        p = prog.operations[0].params
        assert abs(p["radius"] - 25.0) < 0.1
        assert abs(p["height"] - 30.0) < 0.1

    def test_sphere_from_spec(self):
        """Sphere object_type → sphere CadOp."""
        spec = DrawingSpec(
            object_type="sphere",
            dimensions=[Dimension(20, "mm", "diameter")],
            symmetry="axial",
            overall_size=(20, 20, 20),
        )
        prog = drawing_to_cad(spec)
        assert prog.operations[0].op_type == "sphere"
        assert abs(prog.operations[0].params["radius"] - 10.0) < 0.1

    def test_holes_from_spec(self):
        """Spec with holes → base op + subtract_cylinder ops."""
        spec = DrawingSpec(
            object_type="flange",
            dimensions=[
                Dimension(80, "mm", "diameter"),
                Dimension(20, "mm", "height"),
            ],
            views=[ViewSpec("top", features=[
                Feature("hole", "top", (0.3, 0.5), (0.1, 0.1),
                        [Dimension(10, "mm", "diameter")], through=True),
                Feature("hole", "top", (0.7, 0.5), (0.1, 0.1),
                        [Dimension(10, "mm", "diameter")], through=True),
            ])],
            symmetry="axial",
            overall_size=(80, 20, 80),
        )
        prog = drawing_to_cad(spec)
        # Should have base cylinder + 2 holes
        assert prog.n_enabled() >= 3
        hole_ops = [op for op in prog.operations if op.op_type == "subtract_cylinder"]
        assert len(hole_ops) == 2
        assert abs(hole_ops[0].params["radius"] - 5.0) < 0.1

    def test_gear_from_spec(self):
        """Gear object_type → extrude CadOp."""
        spec = DrawingSpec(
            object_type="gear",
            dimensions=[
                Dimension(40, "mm", "diameter"),
                Dimension(8, "mm", "height"),
            ],
            symmetry="radial",
            overall_size=(40, 8, 40),
        )
        prog = drawing_to_cad(spec)
        assert prog.operations[0].op_type == "extrude"

    def test_default_fallback(self):
        """Unknown object type with diameter → cylinder fallback."""
        spec = DrawingSpec(
            object_type="widget",
            dimensions=[
                Dimension(30, "mm", "diameter"),
                Dimension(50, "mm", "height"),
            ],
            symmetry="none",
            overall_size=(30, 50, 30),
        )
        prog = drawing_to_cad(spec)
        # Should get cylinder since diameter is present
        assert prog.operations[0].op_type == "cylinder"

    def test_empty_spec_produces_program(self):
        """Even an empty spec should produce a valid program."""
        spec = DrawingSpec()
        prog = drawing_to_cad(spec)
        assert prog.n_enabled() >= 1


# ===================================================================
# TestDimensionAnnotations
# ===================================================================

class TestDimensionAnnotations:

    def test_compute_box_dimensions(self):
        """Bounding box dimensions for a cube."""
        v, f = make_cube_mesh(10.0, 4)
        dims = _compute_mesh_dimensions(v, f)
        assert abs(dims["width"] - 10.0) < 0.5
        assert abs(dims["height"] - 10.0) < 0.5
        assert abs(dims["depth"] - 10.0) < 0.5
        assert dims["is_cylindrical"] is False

    def test_compute_cylinder_dimensions(self):
        """Cylinder should be detected as cylindrical with correct diameter."""
        v, f = make_cylinder_mesh(5.0, 15.0, 24, 10)
        dims = _compute_mesh_dimensions(v, f)
        assert dims["is_cylindrical"] is True
        assert abs(dims["diameter"] - 10.0) < 0.5
        assert abs(dims["height"] - 15.0) < 0.5

    def test_get_view_dimensions_front_box(self):
        """Front view of box: horizontal=width, vertical=height."""
        dims = {"width": 20.0, "depth": 10.0, "height": 15.0,
                "is_cylindrical": False, "diameter": None}
        anns = _get_view_dimensions(dims, "front")
        assert len(anns) == 2
        horiz = [a for a in anns if a["orientation"] == "horizontal"]
        vert = [a for a in anns if a["orientation"] == "vertical"]
        assert len(horiz) == 1
        assert horiz[0]["value"] == 20.0
        assert vert[0]["value"] == 15.0

    def test_get_view_dimensions_top_box(self):
        """Top view of box: horizontal=width, vertical=depth."""
        dims = {"width": 20.0, "depth": 10.0, "height": 15.0,
                "is_cylindrical": False, "diameter": None}
        anns = _get_view_dimensions(dims, "top")
        horiz = [a for a in anns if a["orientation"] == "horizontal"]
        vert = [a for a in anns if a["orientation"] == "vertical"]
        assert horiz[0]["value"] == 20.0
        assert vert[0]["value"] == 10.0

    def test_get_view_dimensions_cylinder_front(self):
        """Front view of cylinder: horizontal=diameter, vertical=height."""
        dims = {"width": 10.0, "depth": 10.0, "height": 15.0,
                "is_cylindrical": True, "diameter": 10.0}
        anns = _get_view_dimensions(dims, "front")
        horiz = [a for a in anns if a["orientation"] == "horizontal"]
        assert horiz[0]["value"] == 10.0
        assert horiz[0]["label"] == "diameter"

    def test_annotated_render_has_more_pixels(self):
        """Annotated image should have more dark pixels than plain."""
        v, f = make_cube_mesh(10.0, 4)
        plain = render_orthographic(v, f, "front", 256, annotate=False)
        annotated = render_orthographic(v, f, "front", 256, annotate=True)
        assert plain.shape == annotated.shape
        dark_plain = np.sum(plain < 128)
        dark_annot = np.sum(annotated < 128)
        assert dark_annot > dark_plain, "Annotated image should have more drawn content"

    def test_annotated_sheet(self):
        """render_drawing_sheet with annotate=True should produce valid output."""
        v, f = make_cylinder_mesh(5.0, 15.0, 24, 10)
        sheet = render_drawing_sheet(v, f, ("front", "side", "top"), 256,
                                      annotate=True)
        assert sheet.ndim == 3
        assert sheet.shape[2] == 3
        # Should have more dark pixels than non-annotated version
        plain_sheet = render_drawing_sheet(v, f, ("front", "side", "top"), 256,
                                            annotate=False)
        assert np.sum(sheet < 128) > np.sum(plain_sheet < 128)

    def test_annotated_render_size_unchanged(self):
        """Annotated image should have same dimensions as plain."""
        v, f = make_cube_mesh(10.0, 4)
        for size in (128, 256, 512):
            img = render_orthographic(v, f, "front", size, annotate=True)
            assert img.shape == (size, size)

    def test_annotate_false_is_default(self):
        """Default annotate=False should produce same output as explicit False."""
        v, f = make_cube_mesh(10.0, 4)
        default = render_orthographic(v, f, "front", 256)
        explicit = render_orthographic(v, f, "front", 256, annotate=False)
        np.testing.assert_array_equal(default, explicit)
