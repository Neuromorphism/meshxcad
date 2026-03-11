# Plan: Mechanical Drawing → CAD Reconstruction

## Status: READY TO IMPLEMENT

This document is a complete implementation plan for extending MeshXCAD to create
CAD objects from mechanical drawings using Llama-3.2-11B-Vision. It contains all
context needed to execute without reading prior conversation history.

---

## Table of Contents

1. [Project Context](#1-project-context)
2. [Existing Codebase Reference](#2-existing-codebase-reference)
3. [Architecture Overview](#3-architecture-overview)
4. [Phase 0: Orthographic Wireframe Renderer](#4-phase-0-orthographic-wireframe-renderer)
5. [Phase 1: Vision LLM — Drawing Interpretation](#5-phase-1-vision-llm--drawing-interpretation)
6. [Phase 2: Drawing → CAD Builder](#6-phase-2-drawing--cad-builder)
7. [Phase 3: Drawing-Based Optimization Loop](#7-phase-3-drawing-based-optimization-loop)
8. [Phase 4: Roundtrip Verification](#8-phase-4-roundtrip-verification)
9. [Phase 5: Adversarial Development Loops](#9-phase-5-adversarial-development-loops)
10. [Phase 6: Test Suite](#10-phase-6-test-suite)
11. [Implementation Order](#11-implementation-order)
12. [Key Design Decisions](#12-key-design-decisions)

---

## 1. Project Context

### What MeshXCAD Does

MeshXCAD is a Python toolkit for bidirectional detail transfer between triangle
meshes and parametric CAD programs. It evolves a `CadProgram` (a sequence of
parametric operations like sphere, cylinder, box, revolve, subtract_cylinder,
etc.) to match a target mesh through an **alternating coevolution loop** that
balances two objectives:
- **Accuracy**: The generated CAD mesh matches the target mesh
- **Elegance**: The CAD program is clean, simple, and follows professional design patterns

### What We're Adding

The ability to:
1. **Read a mechanical drawing** (PNG/PDF image) → extract geometry via Llama-3.2-11B-Vision
2. **Build a CAD program** from the extracted dimensions and features
3. **Render a CAD program** back to engineering-drawing-style views (wireframe orthographic)
4. **Compare drawings** to measure reconstruction accuracy
5. **Close the roundtrip**: Drawing → CAD → Render → CAD₂ and verify CAD ≈ CAD₂
6. **Harden via adversarial loops** that discover and fix weaknesses

### Environment

- **Platform**: Linux (WSL2), Python 3.12
- **RAM**: 62 GB
- **GPU**: Available in the implementation session (not in planning session)
- **Disk**: ~900 GB free
- **Installed**: numpy, scipy, pillow, torch 2.10.0+cu128, cadquery (brings OCP/OpenCASCADE)
- **Not yet installed**: transformers, accelerate, huggingface_hub (needed for Llama vision model)

### File Layout

```
/app/
├── meshxcad/
│   ├── __init__.py              # v0.1.0
│   ├── __main__.py              # CLI entry: load_mesh(), optimise(), main()
│   ├── cad_program.py           # CadOp, CadProgram, ProgramGap, find_program_gaps()
│   ├── coevolution.py           # run_coevolution(), ObjectState, TechniqueLibrary
│   ├── elegance.py              # 12-dimension scoring, discriminator, mutations (1411 lines)
│   ├── reconstruct.py           # classify_mesh(), fit_sphere/cylinder/cone/box()
│   ├── general_align.py         # hausdorff_distance(), surface_distance_map()
│   ├── render.py                # render_mesh() — matplotlib 3D (NOT engineering-drawing style)
│   ├── synthetic.py             # make_sphere_mesh(), make_cylinder_mesh(), make_cube_mesh()
│   ├── stl_io.py                # read/write binary STL
│   ├── step_io.py               # read_step(), write_step() via OCP/OpenCASCADE
│   ├── mesh_io.py               # generic mesh I/O
│   ├── cad_io.py                # FreeCAD document load/save
│   ├── cad_to_mesh.py           # CAD → mesh conversion
│   ├── mesh_to_cad.py           # mesh → CAD pipeline
│   ├── alignment.py             # spatial registration
│   ├── detail_transfer.py       # detail transfer between models
│   ├── revolve_align.py         # revolve-specific fitting
│   ├── extrude_align.py         # extrude-specific fitting
│   ├── adversarial_loop.py      # earlier adversarial loop (superseded by coevolution.py)
│   ├── hourglass_*.py           # hourglass mesh generation (3 variants)
│   ├── test_data_gen.py         # test mesh generation utilities
│   └── objects/
│       ├── __init__.py
│       ├── builder.py           # revolve_profile(), make_torus(), combine_meshes()
│       ├── catalog.py           # 19 decorative objects (vase, goblet, chess king, etc.)
│       ├── complex_catalog.py   # 10 mechanical parts (gear, flange, bracket, hex nut, etc.)
│       ├── operations.py        # extrude_polygon(), subtract_cylinders(), sweep_along_path()
│       └── freecad_gen.py       # FreeCAD integration
├── tests/
│   ├── test_cad_program.py
│   ├── test_elegance.py
│   ├── test_step_io.py          # 26 tests (all passing)
│   ├── test_reconstruct.py
│   ├── test_general_align.py
│   ├── test_catalog.py
│   ├── test_complex_catalog.py
│   └── ... (16 test files total)
├── setup.py
└── pytest.ini
```

---

## 2. Existing Codebase Reference

### Core Data Structures

#### CadOp
```python
@dataclass
class CadOp:
    op_type: str          # "sphere", "cylinder", "cone", "box", "torus", "revolve",
                          # "extrude", "sweep", "translate", "scale", "rotate", "union",
                          # "subtract_cylinder", "mirror"
    params: dict          # operation-specific parameters
    enabled: bool = True

OP_COSTS = {
    "sphere": 1.0, "cylinder": 1.0, "cone": 1.5, "box": 1.0, "torus": 1.5,
    "revolve": 2.0, "extrude": 1.5, "sweep": 3.0, "translate": 0.2,
    "scale": 0.3, "rotate": 0.3, "union": 0.5, "subtract_cylinder": 1.5,
    "mirror": 0.5,
}
```

#### CadProgram
```python
class CadProgram:
    operations: list[CadOp]

    def evaluate(self) -> (vertices, faces)         # Execute ops → triangle mesh
    def total_complexity(self) -> float
    def total_param_count(self) -> int
    def n_enabled(self) -> int
    def elegance_penalty(self) -> float              # ALPHA*complexity + BETA*n_ops + GAMMA*params
    def total_cost(self, target_v, target_f) -> float
    def copy(self) -> CadProgram
    def to_dict(self) -> dict
    def summary(self) -> str
    @classmethod
    def from_dict(cls, d) -> CadProgram
```

#### ProgramGap
```python
@dataclass
class ProgramGap:
    region_center: np.ndarray   # (3,) where the gap is
    region_radius: float        # spatial extent
    residual_score: float       # severity (0-1)
    suggested_op: str           # what op to add
    suggested_params: dict      # initial params
    nearest_program_op: int     # index of closest existing op (-1 if none)
    action: str                 # "add", "refine", "remove"
```

### Key Functions

```python
# cad_program.py
def find_program_gaps(program, target_v, target_f, max_gaps=5) -> list[ProgramGap]
def add_operation(program, gap)
def refine_operation(program, op_index, target_v, target_f, max_iter=30)
def remove_operation(program, op_index)
def simplify_program(program, target_v, target_f)
def initial_program(target_v, target_f) -> CadProgram

# coevolution.py
def _joint_score(elegance, cad_score, accuracy) -> float
    # = elegance * 0.5 + accuracy * 0.3 + (1.0 - cad_score) * 0.2
def _run_discriminator_pass(state, library, rng, rounds_per_object=5) -> bool
def _run_elegance_pass(state, library, rng, rounds_per_object=5) -> bool
def run_coevolution(target_generators=None, max_sweeps=20, rounds_per_object=5,
                     patience=3, output_dir="/tmp/coevolution") -> dict

# __main__.py
def load_mesh(filepath) -> (vertices, faces)    # STL/OBJ/PLY/STEP dispatch
def _is_step_file(filepath) -> bool
def _load_cad_from_step(step_path, target_v, target_f, quiet=False) -> CadProgram
def optimise(target_v, target_f, *,
             initial_cad=None, max_sweeps=15, rounds=5,
             patience=3, verbose=True) -> dict

# elegance.py
def score_accuracy(program, target_v, target_f) -> float          # 0-1
def compute_elegance_score(program, target_v, target_f) -> dict   # per-dimension + total
def discriminate_cad_vs_mesh(v, f) -> float                       # 0=mesh, 1=CAD
def compare_elegance(prog_a, prog_b, target_v, target_f) -> dict

# reconstruct.py
def classify_mesh(vertices, faces) -> dict      # shape_type, confidence, details
def fit_sphere(vertices) -> dict                # center, radius, residual
def fit_cylinder(vertices) -> dict              # center, axis, radius, height, residual
def fit_cone(vertices) -> dict                  # apex, axis, half_angle, height, residual
def fit_box(vertices) -> dict                   # center, axes, dimensions, residual

# general_align.py
def hausdorff_distance(vertices_a, vertices_b) -> dict
    # {hausdorff, mean_a_to_b, mean_b_to_a, mean_symmetric}
def surface_distance_map(cad_vertices, cad_faces, mesh_vertices) -> dict
    # {signed_distances, unsigned_distances, mean_dist, max_dist}

# render.py (existing — matplotlib 3D, NOT what we need for drawings)
def render_mesh(vertices, faces, output_path, title="Mesh", figsize=(16, 12))
def render_comparison(meshes, labels, output_path, title="Comparison", figsize=(20, 6))

# synthetic.py
def make_sphere_mesh(radius=5.0, lat_divs=20, lon_divs=20) -> (v, f)
def make_cylinder_mesh(radius=5.0, height=15.0, radial_divs=24, height_divs=20) -> (v, f)
def make_cube_mesh(size=10.0, subdivisions=4) -> (v, f)

# objects/builder.py
def revolve_profile(profile_rz, n_angular=48, close_top=True, close_bottom=True)
def make_torus(major_r, minor_r, z_center, n_angular=48, n_cross=12)
def combine_meshes(mesh_list)

# objects/operations.py
def extrude_polygon(polygon_xy, height, n_height=2)
def subtract_cylinders(base_verts, base_faces, holes)
def sweep_along_path(profile_2d, path_3d, n_profile=None, twist_total_deg=0.0, ...)
```

### Available Test Objects (29 total)

**19 decorative (catalog.py)**: classical_vase, goblet, candlestick, chess_king,
chess_queen, bell, trophy_cup, column, baluster, teapot, perfume_bottle,
table_lamp, finial, door_knob, ornamental_egg, wine_decanter, decorative_bowl,
pedestal, spinning_top

**10 complex mechanical (complex_catalog.py)**: spur_gear, pipe_flange,
shelf_bracket, hex_nut, picture_frame, star_knob, fluted_column,
castellated_ring, gear_shift_knob, lattice_panel

### Coevolution Loop Architecture

```
sweep loop (max 15):
  └─ Loop 1: Discriminator Pass
  │  - Mutations: perturb subdivisions, break symmetry, refine accuracy, add gap primitives
  │  - Scoring: joint_score = elegance*0.5 + accuracy*0.3 + (1-cad_score)*0.2
  │
  └─ Loop 2: Elegance Pass
     - Mutations: simplify, mirror-replace, origin-anchor, refine, multi-gap fill
     - Same joint_score

  Convergence: stop after `patience` sweeps with no improvement
  Constants: MIN_IMPROVEMENT_THRESHOLD = 0.002, ACCURACY_FLOOR = 0.40
```

### Elegance Scoring (12 dimensions, weights sum to 1.0)

```python
ELEGANCE_WEIGHTS = {
    "accuracy": 0.20, "conciseness": 0.12, "op_hierarchy": 0.08,
    "symmetry": 0.08, "no_redundancy": 0.10, "tree_depth": 0.06,
    "param_economy": 0.05, "origin_anchoring": 0.04, "mesh_quality": 0.08,
    "normal_consistency": 0.05, "op_diversity": 0.06, "watertightness": 0.08,
}
```

---

## 3. Architecture Overview

```
┌──────────────┐    ┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│  Drawing     │───▶│  Vision LLM  │───▶│  CAD Builder │───▶│  CAD Program │
│  (PNG/PDF)   │    │  (Llama-3.2  │    │  (existing   │    │  (CadOp      │
│              │    │   -11B-Vis)  │    │   mutations) │    │   sequence)  │
└──────────────┘    └──────────────┘    └──────────────┘    └──────────────┘
                                                                    │
                    ┌──────────────┐    ┌──────────────┐            │
                    │  Compare     │◀───│  Render to   │◀───────────┘
                    │  Drawing vs  │    │  Drawing     │
                    │  Render      │    │  (wireframe) │
                    └──────────────┘    └──────────────┘
```

The roundtrip: **Drawing → CAD → Render → CAD₂** and verify CAD ≈ CAD₂.

---

## 4. Phase 0: Orthographic Wireframe Renderer

**Goal**: Render a CadProgram as engineering-drawing-style views (the "CAD → Drawing" direction).

**Why first**: We need this to (a) generate training/test data, (b) measure accuracy
by comparing rendered views to input drawings, and (c) close the roundtrip loop.

### New file: `meshxcad/drawing.py`

```python
def render_orthographic(vertices, faces, view="front", image_size=512) -> np.ndarray:
    """Project mesh to 2D silhouette + visible edges.

    Args:
        vertices: (N, 3) mesh vertices
        faces: (M, 3) triangle indices
        view: "front" (XZ plane, drop Y), "side" (YZ, drop X),
              "top" (XY, drop Z), or tuple (elevation_deg, azimuth_deg)
        image_size: output image dimension in pixels

    Returns:
        Binary image (H, W) uint8 — black lines on white background
    """

def render_drawing_sheet(vertices, faces, views=("front", "side", "top"),
                         image_size=512, dimensions=None) -> np.ndarray:
    """Multi-view engineering drawing on one sheet.

    Args:
        dimensions: optional list of DimensionAnnotation for adding dimension lines

    Returns: RGB image (H, W, 3) with views arranged in standard layout.
    """

def extract_visible_edges(vertices, faces, view_dir) -> list[tuple]:
    """Hidden-line removal: return only visible edges as 2D line segments.

    Algorithm:
    1. Back-face culling: remove faces whose normal dots view_dir < 0
    2. Feature edges: edges where adjacent face normals differ significantly (>30°)
    3. Silhouette edges: edges where one adjacent face is front-facing, one back-facing
    4. Depth test: rasterize front faces to z-buffer, test each edge segment for occlusion

    Returns: list of ((x1,y1), (x2,y2)) in normalized coordinates
    """

def _project_point(point_3d, view) -> tuple:
    """Orthographic projection of a 3D point to 2D for a given view."""

def _rasterize_zbuffer(vertices, faces, view_dir, image_size) -> np.ndarray:
    """Simple z-buffer rasterizer for occlusion testing."""

def add_dimension_lines(edges_2d, known_params) -> list:
    """Add length/radius/angle dimension annotations from known CadOp params."""
```

**Projection math:**
- Front view (standard): project onto XZ plane → x_2d = x_3d, y_2d = z_3d
- Side view: project onto YZ plane → x_2d = y_3d, y_2d = z_3d
- Top view: project onto XY plane → x_2d = x_3d, y_2d = y_3d
- Custom view: rotation matrix from (elevation, azimuth), then project onto XY

**Edge extraction algorithm:**
1. Build edge→face adjacency map from triangle mesh
2. For each edge, classify:
   - **Interior edge**: both faces visible, normals similar → skip (not a feature edge)
   - **Feature edge**: both faces visible, normals differ >30° → include (sharp edge)
   - **Silhouette edge**: one face visible, one hidden → include (outline)
   - **Boundary edge**: only one face → include (open edge)
3. For visible edges, test against z-buffer for occlusion:
   - Sample points along edge, check if any closer geometry blocks them
   - Split edge into visible/hidden segments
4. Draw visible segments as solid lines, optionally hidden as dashed

**Output format**: Pillow (PIL) Image drawn with antialiased lines. Black lines
on white background, matching mechanical drawing visual language.

### New file: `meshxcad/drawing_compare.py`

```python
def compare_drawings(drawing_a, drawing_b) -> dict:
    """Compare two engineering drawings as images.

    Args:
        drawing_a: (H, W) or (H, W, 3) image array
        drawing_b: same

    Returns:
        {
            "pixel_iou": float,          # IoU of edge pixels
            "chamfer_distance": float,   # mean distance between edge pixel sets
            "structural_similarity": float,  # SSIM on binarized edges
            "edge_precision": float,     # fraction of a's edges near b's edges
            "edge_recall": float,        # fraction of b's edges near a's edges
        }
    """

def extract_edge_pixels(image, threshold=128) -> np.ndarray:
    """Binarize image → extract dark pixel coordinates as (N, 2) array."""

def chamfer_distance_2d(points_a, points_b) -> float:
    """Mean bidirectional nearest-neighbor distance between point sets."""
```

**Key metric**: Chamfer distance between edge pixel sets. This is robust to minor
alignment differences and anti-aliasing. The KDTree-based computation is fast.

---

## 5. Phase 1: Vision LLM — Drawing Interpretation

**Goal**: Extract structured geometry from a mechanical drawing image.

### Install dependencies

```bash
pip install transformers accelerate huggingface_hub
# Model: meta-llama/Llama-3.2-11B-Vision-Instruct
# Needs HF token with Llama access grant
```

With GPU available, load in float16 (~22GB VRAM) or int4 (~6GB VRAM).

### New file: `meshxcad/drawing_spec.py`

```python
from dataclasses import dataclass, field

@dataclass
class Dimension:
    value: float
    unit: str              # "mm", "in"
    measurement: str       # "diameter", "height", "width", "radius", "angle", "depth"
    feature: str           # what it measures ("bore", "flange_od", "body", "hole_1")
    view: str              # which view ("front", "side", "top")

@dataclass
class Feature:
    feature_type: str      # "cylinder", "hole", "fillet", "chamfer", "flat",
                           # "thread", "slot", "counterbore", "sphere", "torus"
    view: str              # which view it's visible in
    center_2d: tuple       # (x, y) center in view coordinates (0-1 normalized)
    extent_2d: tuple       # (w, h) bounding box in view coordinates
    dimensions: list       # list of Dimension objects for this feature
    through: bool = False  # True if hole goes all the way through

@dataclass
class ViewSpec:
    view_type: str         # "front", "side", "top", "section", "isometric", "detail"
    features: list         # list[Feature]
    outline: list          # ordered 2D boundary points (normalized)
    bounding_box: tuple    # (x, y, w, h) of this view in the full image

@dataclass
class DrawingSpec:
    description: str       # "Pipe flange with bolt holes"
    object_type: str       # "shaft", "flange", "bracket", "gear", "housing", etc.
    views: list            # list[ViewSpec]
    dimensions: list       # list[Dimension] — all dimensions across all views
    symmetry: str          # "axial", "bilateral", "radial", "none"
    overall_size: tuple    # (width, height, depth) in drawing units
    material: str = ""     # if specified
    notes: list = field(default_factory=list)  # any text notes from drawing
```

### New file: `meshxcad/vision.py`

```python
class DrawingInterpreter:
    """Extract structured geometry from mechanical drawings using Llama-3.2-11B-Vision."""

    def __init__(self, model_path=None, device="auto", quantize="auto"):
        """Load the vision model.

        Args:
            model_path: HuggingFace model ID or local path.
                        Default: "meta-llama/Llama-3.2-11B-Vision-Instruct"
            device: "auto", "cuda", "cpu"
            quantize: "auto" (4-bit if <24GB VRAM, else fp16), "4bit", "8bit", "none"
        """

    def interpret_drawing(self, image_path, views_hint=None) -> DrawingSpec:
        """Full pipeline: image → structured DrawingSpec.

        Multi-stage prompting:
        1. Scene understanding — what object, what views
        2. Dimension extraction — measurements with units
        3. Feature extraction — per-view geometry as JSON
        4. Cross-view reconciliation — consistency check

        Args:
            image_path: path to drawing image (PNG, JPG, PDF)
            views_hint: optional list of known view types present

        Returns: DrawingSpec
        """

    def _prompt_scene(self, image) -> dict:
        """Stage 1: What is depicted and which views are present?

        Prompt template:
        'This is a mechanical engineering drawing showing orthographic views
         of a part. Describe:
         1. What type of part is shown (e.g., shaft, flange, bracket, gear)?
         2. What views are present (front, side, top, section, isometric)?
         3. Is the part axially symmetric, bilaterally symmetric, or asymmetric?
         4. Approximate overall dimensions if visible.
         Reply as JSON: {"object_type": ..., "views": [...], "symmetry": ...,
                         "description": ...}'
        """

    def _prompt_dimensions(self, image, scene) -> list:
        """Stage 2: Extract all dimension annotations.

        Prompt template:
        'Given this is a {scene.object_type}, extract every dimension shown.
         For each dimension, provide:
         - value (number)
         - unit (mm or inches)
         - what it measures (diameter, height, width, radius, etc.)
         - which feature it belongs to
         - which view it appears in
         Reply as JSON array: [{"value": ..., "unit": ..., "measurement": ...,
                                "feature": ..., "view": ...}, ...]'
        """

    def _prompt_features(self, image, scene, dimensions) -> list:
        """Stage 3: Extract geometric features per view.

        Prompt template:
        'This {scene.object_type} has these dimensions: {dims_summary}.
         For each view ({view_list}), describe every geometric feature visible:
         - feature type (cylinder, hole, fillet, chamfer, flat, thread, slot, etc.)
         - approximate center position in the view (as fraction 0-1 of view width/height)
         - approximate size in the view
         - associated dimensions from the list above
         Reply as JSON: {"views": [{"view_type": ..., "features": [...]}]}'
        """

    def _reconcile(self, scene, dimensions, features) -> DrawingSpec:
        """Stage 4: Cross-reference views, resolve conflicts, build DrawingSpec.

        Rules:
        - Height visible in front AND side → must agree (take average if close)
        - Diameter visible in top view (circle) → matches width in front/side
        - Through-hole visible in 2+ views → confirm with both
        - Missing dimension → estimate from view proportions + known dimensions
        """

    def _parse_json_response(self, response_text) -> dict:
        """Robustly parse JSON from LLM output (handles markdown fences, etc.)."""
```

**Model loading strategy:**
1. Check GPU memory with `torch.cuda.get_device_properties(0).total_mem`
2. If ≥24 GB VRAM: load float16
3. If ≥12 GB VRAM: load int8 via bitsandbytes
4. If ≥8 GB VRAM: load int4 via bitsandbytes
5. If no GPU: load int4 on CPU (slow but works with 62 GB RAM)

**Prompt engineering notes:**
- Llama-3.2-11B-Vision works best with clear, structured prompts
- Request JSON output with explicit schema
- Multi-stage prompting avoids overloading the model with one complex query
- Each stage builds on previous results, allowing error correction
- The `_reconcile` stage is deterministic code, not LLM — it fixes inconsistencies

---

## 6. Phase 2: Drawing → CAD Builder

**Goal**: Convert a `DrawingSpec` into a `CadProgram`.

### New file: `meshxcad/drawing_to_cad.py`

```python
def drawing_to_cad(spec: DrawingSpec) -> CadProgram:
    """Convert interpreted drawing to initial CAD program.

    Strategy:
    1. Determine primary shape from symmetry + object_type + views
    2. Build base primitive from overall dimensions
    3. Add features (holes, chamfers, fillets) as additional ops
    4. Cross-reference dimensions across views for consistency

    Returns: CadProgram ready for optimization
    """

def _infer_base_shape(spec: DrawingSpec) -> CadOp:
    """From description + symmetry + views, determine the base primitive.

    Decision tree:
    - axial symmetry + circular front/top view → cylinder or revolve
      - if profile is non-rectangular → revolve with extracted profile
      - if profile is rectangular → cylinder
    - all rectangular views → box
    - circular top + rectangular side → cylinder
    - complex axial profile + symmetry → revolve with profile points
    - gear-like description → extrude with involute profile
    - bracket/L-shape → box + subtract/extrude operations
    """

def _dimensions_to_params(spec: DrawingSpec, shape: str) -> dict:
    """Convert extracted dimensions into CadOp parameter dict.

    Example for cylinder:
        dims has "diameter": 50, "height": 30
        → {"center": [0, 0, 15], "axis": [0, 0, 1],
           "radius": 25.0, "height": 30.0}
    """

def _add_holes(program: CadProgram, spec: DrawingSpec):
    """Add subtract_cylinder ops for each hole feature.

    For each Feature with feature_type == "hole":
    1. Get hole diameter from dimensions
    2. Determine hole center from 2D position + cross-view reconciliation
    3. Determine hole axis from which views it appears circular in
    4. Add CadOp("subtract_cylinder", {center, axis, radius, height})
    """

def _add_fillets_chamfers(program: CadProgram, spec: DrawingSpec):
    """Modify profile or add blend operations for fillets/chamfers.

    For revolve-based objects: modify the profile points to include radius
    For box-based: add torus ops or modify dimensions
    """

def _extract_revolve_profile(spec: DrawingSpec) -> list[tuple]:
    """Extract (r, z) profile from front/side view outline.

    For axially symmetric parts:
    1. Take the front or side view outline
    2. Split at the axis of symmetry
    3. Take the right half as the (radius, height) profile
    4. Scale from view pixels to real dimensions using extracted dimensions
    """

def _cross_reference_views(spec: DrawingSpec) -> dict:
    """Reconcile dimensions across views.

    Returns mapping of feature → 3D dimensions:
    {"body": {"width": 50, "height": 30, "depth": 20},
     "hole_1": {"diameter": 10, "center_3d": [15, 0, 15], "axis": [0, 1, 0]}}
    """

def _estimate_missing_dimensions(spec: DrawingSpec, known: dict) -> dict:
    """Infer dimensions not explicitly annotated.

    Strategies:
    - Proportional: if front shows feature at 30% of width, and width=50 → offset=15
    - Symmetric: if axial, features mirror across axis
    - Standard: common engineering ratios (e.g., fillet radius ~5-10% of adjacent dimension)
    """
```

**This is the hardest module.** The LLM output will be noisy. Key robustness strategies:

1. **Fallback to simpler shapes**: If feature extraction is ambiguous, start with a
   basic primitive (cylinder for round things, box for rectangular) and let the
   coevolution loop refine it.

2. **Dimension priority**: Use explicitly annotated dimensions as ground truth.
   Estimate anything missing from view proportions.

3. **Cross-view validation**: A dimension appearing in 2+ views is more reliable.
   Resolve conflicts by averaging close values, flagging large discrepancies.

---

## 7. Phase 3: Drawing-Based Optimization Loop

**Goal**: Refine a CAD program by comparing its rendered views to the original drawing.

### Modification: `meshxcad/coevolution.py` (or new wrapper)

Add a drawing-based accuracy scorer that plugs into the existing optimization:

```python
def score_drawing_accuracy(program, drawing_image, views=("front", "side", "top"),
                           image_size=512) -> float:
    """Score CAD program by comparing its rendered views to the original drawing.

    1. program.evaluate() → mesh
    2. render_drawing_sheet(mesh, views, image_size) → rendered_image
    3. compare_drawings(drawing_image, rendered_image) → scores
    4. Return weighted score:
       0.5 * (1 - chamfer_distance/max_dist) + 0.3 * pixel_iou + 0.2 * ssim
    """
```

This replaces `score_accuracy(program, target_v, target_f)` when the target is a
drawing rather than a mesh. The coevolution mutations, elegance scoring, and
convergence logic all work unchanged — only the accuracy signal changes.

### New CLI subcommand in `meshxcad/__main__.py`

Add a `drawing` subcommand or mode:

```
python -m meshxcad drawing input_drawing.png [options]
    --model PATH             Path to Llama vision model (or HF model ID)
    --views front,side,top   Which views are in the drawing (auto-detected if omitted)
    --scale FLOAT            Drawing scale (mm/pixel), for absolute dimensions
    -c, --cad FILE           Optional starting CAD program (JSON)
    -o, --output DIR         Output directory
    --sweeps N               Max optimization sweeps (default: 15)
    -r, --rounds N           Rounds per sweep (default: 5)
    --fast                   Quick mode (3 sweeps, 3 rounds)
    -q, --quiet              Suppress output
```

**Flow:**
1. Load drawing image
2. `DrawingInterpreter.interpret_drawing(image)` → `DrawingSpec`
3. `drawing_to_cad(spec)` → initial `CadProgram`
4. Run optimization loop with `score_drawing_accuracy` as the accuracy metric
5. Output: `program.json`, `output.stl`, `comparison.png` (overlay of input + rendered)

---

## 8. Phase 4: Roundtrip Verification

**Goal**: Prove fidelity by going CAD → render → interpret → rebuild → compare.

### New file: `meshxcad/roundtrip.py`

```python
def roundtrip_test(program: CadProgram, interpreter: DrawingInterpreter,
                   views=("front", "side", "top"),
                   optimize_sweeps=10) -> dict:
    """Full roundtrip: CAD₁ → Drawing₁ → CAD₂ → Drawing₂.

    Steps:
    1. program.evaluate() → mesh₁
    2. render_drawing_sheet(mesh₁, views) → drawing₁
    3. interpreter.interpret_drawing(drawing₁) → spec₂
    4. drawing_to_cad(spec₂) → program₂_initial
    5. Optimize program₂ against drawing₁ (drawing-based accuracy)
    6. program₂.evaluate() → mesh₂
    7. render_drawing_sheet(mesh₂, views) → drawing₂
    8. Compare: mesh₁ vs mesh₂, drawing₁ vs drawing₂, program₁ vs program₂

    Returns:
        {
            "mesh_hausdorff": float,        # hausdorff(mesh₁, mesh₂).mean_symmetric
            "mesh_hausdorff_normalized": float,  # normalized by bbox diagonal
            "drawing_chamfer": float,       # chamfer between drawing edge pixels
            "drawing_iou": float,           # pixel IoU of drawings
            "program_op_match": float,      # fraction of op types that match
            "program_param_distance": float,# normalized param difference
            "roundtrip_score": float,       # combined 0-1 score
            "drawing_1": np.ndarray,        # the rendered drawing₁
            "drawing_2": np.ndarray,        # the re-rendered drawing₂
        }
    """

def batch_roundtrip(programs: list, interpreter: DrawingInterpreter,
                    labels: list = None) -> dict:
    """Run roundtrip on multiple objects, report statistics.

    Returns:
        {
            "results": list[dict],          # per-object roundtrip results
            "mean_mesh_distance": float,
            "mean_drawing_iou": float,
            "mean_roundtrip_score": float,
            "worst_object": str,            # name of worst-performing
            "best_object": str,
        }
    """
```

**Test progression** (start simple, expand):

| Stage | Objects | Target Roundtrip Score |
|-------|---------|----------------------|
| 1 | Single primitive: box, cylinder, sphere | >0.95 |
| 2 | Single primitive + 1-2 holes | >0.85 |
| 3 | Revolve profiles: vase, goblet, column | >0.75 |
| 4 | Multi-op: flange, bracket, hex nut | >0.65 |
| 5 | Complex: gear, chess king, star knob | >0.50 |

Each stage should pass before moving to the next. Failures at any stage
indicate gaps in either the renderer, the interpreter, or the builder.

---

## 9. Phase 5: Adversarial Development Loops

**Goal**: Systematically discover and fix weaknesses.

### New file: `meshxcad/adversarial_drawing.py`

Three adversarial strategies:

#### Strategy A: Generator vs Interpreter (GAN-style)

```python
def adversarial_generator_loop(interpreter, n_rounds=50, output_dir=None) -> dict:
    """Generate increasingly hard drawings that challenge the interpreter.

    Each round:
    1. GENERATOR: Create a CadProgram (start simple, increase complexity)
       - Rounds 1-10: single primitives with varying dimensions
       - Rounds 11-20: add 1-4 holes at various positions
       - Rounds 21-30: multi-primitive compositions (union, subtract)
       - Rounds 31-40: revolve profiles with features
       - Rounds 41-50: complex assemblies (gear + shaft, flange + bolts)
    2. RENDER: Program → drawing image
    3. INTERPRET: Drawing → DrawingSpec → CadProgram₂
    4. EVALUATE: Compare programs via mesh Hausdorff distance
    5. CLASSIFY FAILURE: If roundtrip fails, categorize why
    6. ADAPT: Focus next rounds on failure patterns

    Returns:
        {
            "rounds": list[dict],            # per-round results
            "failure_database": FailureDatabase,
            "success_rate_by_complexity": dict,
            "identified_weaknesses": list[str],
        }
    """
```

#### Strategy B: Drawing Perturbation (Robustness)

```python
def adversarial_perturbation_loop(interpreter, test_drawings, n_rounds=30) -> dict:
    """Perturb drawings to find interpreter breaking points.

    Perturbation types:
    1. Gaussian noise (increasing σ)
    2. Resolution change (downsample/upsample)
    3. Rotation (±1° to ±5° simulating scan skew)
    4. Occlusion (remove dimension labels one at a time)
    5. Line weight variation (thin/thick lines)
    6. Partial view removal (crop one view out)
    7. Background noise (add speckling)
    8. Annotation style changes (arrow types, text size)

    Returns: per-perturbation accuracy degradation curves
    """
```

#### Strategy C: Mutation-Based Fuzzing

```python
def adversarial_mutation_loop(interpreter, base_programs, n_rounds=100) -> dict:
    """Mutate known-good CadPrograms to find edge cases.

    Mutations:
    1. Tiny features (hole diameter < 1% of body)
    2. Near-tangent features (hole just touching edge)
    3. Extreme aspect ratios (very thin: 100:1, very tall: 1:100)
    4. Many features (20+ holes in a flange)
    5. Degenerate geometry (zero-height cylinder, flat box)
    6. Overlapping features (intersecting holes)
    7. Very thin walls (subtract cylinder nearly as big as body)
    8. Unusual orientations (tilted cylinders, non-axis-aligned)

    Returns: failure cases with classification
    """
```

#### Failure tracking and auto-repair

```python
@dataclass
class DrawingFailure:
    drawing_image: np.ndarray
    expected_cad: CadProgram
    actual_cad: CadProgram        # None if interpretation completely failed
    failure_type: str             # classification (see below)
    severity: float               # 0-1, based on Hausdorff distance
    details: dict                 # type-specific details

# Failure type taxonomy:
FAILURE_TYPES = [
    "missing_feature",        # hole/fillet/chamfer not detected
    "extra_feature",          # hallucinated feature that doesn't exist
    "wrong_dimension",        # dimension extracted but wrong value
    "wrong_shape_type",       # cylinder misidentified as box, etc.
    "wrong_orientation",      # correct shape, wrong axis
    "scale_error",            # proportions wrong
    "missing_view",           # view not recognized
    "json_parse_error",       # LLM output not parseable
    "hallucination",          # LLM described features not in drawing
    "symmetry_error",         # symmetric pattern not recognized
]

class FailureDatabase:
    """Tracks failures, clusters by type, generates regression tests."""

    failures: list[DrawingFailure]

    def add_failure(self, failure: DrawingFailure)
    def cluster_by_type(self) -> dict[str, list[DrawingFailure]]
    def worst_failure_types(self, top_n=5) -> list[tuple[str, int, float]]
    def generate_regression_suite(self) -> list[tuple]:
        """Return (drawing_image, expected_cad) pairs for the worst failures."""
    def suggest_prompt_improvements(self) -> list[str]:
        """Analyze failure patterns → suggest changes to LLM prompts."""
    def save(self, path)
    def load(cls, path) -> FailureDatabase
```

**Auto-improvement cycle:**
When failures cluster around a pattern (e.g., "always misses counterbore holes"):
1. Analyze which prompt stage failed (_prompt_features is most likely)
2. Generate refined prompt with explicit examples of that feature
3. Create synthetic test drawings emphasizing that feature type
4. Re-run interpreter on the synthetic drawings
5. If improvement, adopt the refined prompt; if not, flag for manual review

---

## 10. Phase 6: Test Suite

### `tests/test_drawing.py`

```python
class TestOrthographicRenderer:
    """Tests for meshxcad/drawing.py — no LLM required."""

    def test_box_front_view():
        """Front view of box should be a rectangle."""
    def test_box_all_views():
        """Box: front=rect, side=rect, top=rect with correct proportions."""
    def test_cylinder_front_view():
        """Front view of cylinder = rectangle with same height."""
    def test_cylinder_top_view():
        """Top view of cylinder = circle."""
    def test_sphere_all_views():
        """All views of sphere = circle."""
    def test_hidden_line_removal():
        """Box with hole: front shows hole outline, back edges hidden."""
    def test_multi_view_sheet():
        """3 views on one image, non-overlapping."""
    def test_edge_count():
        """Box should have 12 edges, front view shows 4 visible."""
    def test_output_is_binary():
        """Output image is binary (only black and white pixels)."""
    def test_image_size():
        """Output matches requested image_size."""

class TestDrawingComparison:
    """Tests for meshxcad/drawing_compare.py — no LLM required."""

    def test_identical_drawings():
        """Same drawing compared to itself → IoU=1.0, chamfer=0."""
    def test_different_drawings():
        """Box vs cylinder drawing → low IoU."""
    def test_similar_drawings():
        """Box 10x10x5 vs box 10x10x6 → high IoU."""
    def test_scale_invariance():
        """Same shape at different scales → should still be comparable."""
    def test_chamfer_metric():
        """Chamfer distance is symmetric."""
    def test_empty_images():
        """Handle edge case of blank images."""

class TestDrawingSpec:
    """Tests for meshxcad/drawing_spec.py — no LLM required."""

    def test_create_spec():
        """Can create DrawingSpec with all fields."""
    def test_serialize_spec():
        """DrawingSpec roundtrips through JSON."""

class TestDrawingToCad:
    """Tests for meshxcad/drawing_to_cad.py — no LLM required.
    Uses synthetic DrawingSpec objects (hand-built, not from vision model)."""

    def test_box_from_spec():
        """3 rectangular views with width/height/depth → box CadOp."""
    def test_cylinder_from_spec():
        """Circle top + rectangle front with diameter/height → cylinder CadOp."""
    def test_sphere_from_spec():
        """Circle in all views with diameter → sphere CadOp."""
    def test_revolve_from_spec():
        """Axial symmetric with complex profile → revolve CadOp."""
    def test_holes_from_spec():
        """Spec with 3 holes → 1 base op + 3 subtract_cylinder ops."""
    def test_cross_reference():
        """Height from front matches height from side."""
    def test_missing_dimension():
        """Missing dimension estimated from view proportions."""
    def test_conflicting_dimensions():
        """Slightly different heights in 2 views → averaged."""

class TestVisionInterpreter:
    """Tests for meshxcad/vision.py — REQUIRES Llama model.
    Skip if model not available."""

    def test_box_drawing():
        """Render box → interpret → verify spec has box-like dimensions."""
    def test_cylinder_drawing():
        """Render cylinder → interpret → verify circular + height."""
    def test_flange_drawing():
        """Render flange → interpret → verify holes detected."""

class TestRoundtrip:
    """Tests for meshxcad/roundtrip.py — REQUIRES Llama model."""

    def test_box_roundtrip():
        """Box → draw → interpret → build → compare. Score > 0.9."""
    def test_cylinder_roundtrip():
        """Cylinder → draw → interpret → build → compare. Score > 0.9."""
    def test_sphere_roundtrip():
        """Sphere → draw → interpret → build → compare. Score > 0.85."""
    def test_flange_roundtrip():
        """Pipe flange → draw → interpret → build → compare. Score > 0.6."""
```

**Test markers:**
- Tests in `TestOrthographicRenderer`, `TestDrawingComparison`, `TestDrawingSpec`,
  `TestDrawingToCad` run without any model → fast, always run.
- Tests in `TestVisionInterpreter`, `TestRoundtrip` require the Llama model → mark
  with `@pytest.mark.vision` and skip if model not loaded.

---

## 11. Implementation Order

| Step | Files | Depends On | Description |
|------|-------|------------|-------------|
| **0** | Install transformers, accelerate, huggingface_hub | GPU available | Package setup |
| **1** | `meshxcad/drawing.py` | Pillow (installed) | Orthographic wireframe renderer |
| **2** | `meshxcad/drawing_compare.py` | step 1 | Image comparison metrics |
| **3** | `meshxcad/drawing_spec.py` | none | Data structures |
| **4** | `meshxcad/drawing_to_cad.py` | step 3 + existing cad_program.py | Spec → CadProgram builder |
| **5** | `tests/test_drawing.py` (non-LLM tests) | steps 1-4 | Test renderer, comparator, builder |
| **6** | `meshxcad/vision.py` | transformers + Llama model | LLM-based drawing interpreter |
| **7** | Update `meshxcad/__main__.py` | steps 4, 6 | Add `drawing` CLI subcommand |
| **8** | `meshxcad/roundtrip.py` | steps 1, 4, 6 | CAD↔Drawing roundtrip verification |
| **9** | `tests/test_drawing.py` (LLM tests) | step 8 | Test roundtrip with actual model |
| **10** | `meshxcad/adversarial_drawing.py` | steps 1-8 | Adversarial hardening loops |
| **11** | Run adversarial loops, fix gaps | step 10 | Iterate on weaknesses |

**Steps 1-5 are pure geometry/rendering** — no ML dependency, fast to develop and test.
Build and verify these first.

**Step 6 introduces the LLM** — the most uncertain component. Expect iteration on
prompt templates and JSON parsing.

**Steps 7-11 integrate everything** and use adversarial pressure to improve.

---

## 12. Key Design Decisions

### 1. Drawing comparison via edge-pixel Chamfer distance
Not pixel-wise diff. Chamfer distance is robust to line weight variation,
anti-aliasing, and minor alignment differences. Computed via KDTree, very fast.

### 2. Vision model as structured-output extractor, not end-to-end predictor
The LLM produces a DrawingSpec (JSON), then deterministic code builds the CAD.
This keeps the system debuggable and allows the coevolution loop to compensate
for LLM mistakes. If the LLM says "cylinder, diameter=50, height=30" but gets
the hole positions slightly wrong, the optimizer can fix that.

### 3. Reuse existing coevolution loop
The only change is swapping the accuracy metric from mesh-Hausdorff to
drawing-edge-comparison. All mutations, elegance scoring, and convergence
logic stay the same. This avoids duplicating complex optimization code.

### 4. Multi-stage prompting
One massive prompt asking for everything at once will fail. Breaking into
scene→dimensions→features→reconciliation stages gives the LLM smaller tasks
and allows deterministic validation between stages.

### 5. Adversarial loops as first-class feature
The failure database feeds back into prompt engineering and test generation.
This creates a self-improving cycle: generate hard cases → find failures →
fix prompts/code → generate harder cases.

### 6. Start with perfect inputs, add noise later
Phase 0-4 use our own rendered drawings (clean, known-correct). This isolates
drawing→CAD from drawing-quality issues. Phase 5 then tests with perturbations,
hand-drawn-style inputs, and edge cases.

### 7. CPU fallback for everything
The system must work without GPU (slower). The LLM loads in 4-bit on CPU if
needed. All rendering and comparison is pure numpy/pillow. Only the LLM
inference benefits significantly from GPU.

---

## Appendix A: Example CadProgram Params

For reference, here's what params look like for each op type:

```python
# sphere
{"center": [0, 0, 0], "radius": 5.0, "divs": 20}

# cylinder
{"center": [0, 0, 5], "axis": [0, 0, 1], "radius": 3.0, "height": 10.0,
 "radial_divs": 24, "height_divs": 10}

# box
{"center": [0, 0, 2.5], "dimensions": [10, 8, 5], "subdivisions": 4}

# cone
{"center": [0, 0, 0], "axis": [0, 0, 1], "base_radius": 5.0,
 "top_radius": 2.0, "height": 10.0}

# torus
{"center": [0, 0, 0], "major_r": 8.0, "minor_r": 2.0, "z_center": 0.0}

# revolve
{"center": [0, 0, 0], "profile": [[r1, z1], [r2, z2], ...], "divs": 48}

# extrude
{"center": [0, 0, 0], "polygon": [[x1, y1], [x2, y2], ...], "height": 5.0}

# subtract_cylinder
{"center": [5, 0, 0], "axis": [0, 0, 1], "radius": 2.0, "height": 10.0}

# mirror
{"plane_normal": [1, 0, 0], "plane_point": [0, 0, 0]}
```

## Appendix B: Available Objects for Testing

All 29 objects can be generated programmatically for roundtrip testing:

```python
# Decorative (revolve-based, from catalog.py)
from meshxcad.objects.catalog import make_ornate, make_simple
v, f = make_ornate("classical_vase")     # 19 shapes available

# Complex mechanical (from complex_catalog.py)
from meshxcad.objects.complex_catalog import make_complex_ornate
v, f = make_complex_ornate("pipe_flange")  # 10 shapes available

# Primitives (from synthetic.py)
from meshxcad.synthetic import make_sphere_mesh, make_cylinder_mesh, make_cube_mesh
v, f = make_sphere_mesh(radius=5.0)
```

## Appendix C: Existing Render Module (NOT what we need)

The existing `render.py` uses matplotlib 3D with shaded faces and perspective view.
This is NOT suitable for engineering drawings. We need a new renderer (`drawing.py`)
that produces:
- Black lines on white background
- Orthographic (not perspective) projection
- Hidden-line removal (dashed or invisible back edges)
- Feature edges (sharp edges between faces)
- Silhouette edges (outline of the shape)
- Optional dimension annotations

The existing `render.py` stays for its current purpose (visualization/debugging).
The new `drawing.py` is specifically for engineering-drawing-style output.
