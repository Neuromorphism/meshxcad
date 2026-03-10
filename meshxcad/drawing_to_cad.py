"""Convert a DrawingSpec (structured drawing interpretation) into a CadProgram.

This is the bridge between drawing interpretation and the CAD system.
It maps extracted dimensions and features to parametric CAD operations.
"""

import numpy as np
from .drawing_spec import DrawingSpec, Dimension, Feature
from .cad_program import CadOp, CadProgram


def drawing_to_cad(spec: DrawingSpec) -> CadProgram:
    """Convert interpreted drawing to initial CAD program.

    Strategy:
    1. Determine primary shape from symmetry + object_type + views.
    2. Build base primitive from overall dimensions.
    3. Add features (holes, chamfers) as additional ops.

    Returns: CadProgram ready for optimisation.
    """
    dims = _collect_dimensions(spec)
    base_op = _infer_base_shape(spec, dims)
    program = CadProgram(operations=[base_op])

    _add_holes(program, spec, dims)

    return program


def _collect_dimensions(spec: DrawingSpec) -> dict:
    """Gather all dimensions keyed by measurement type.

    Returns dict like {"diameter": 50.0, "height": 30.0, "width": 20.0, ...}
    If multiple dimensions of the same type exist, prefer explicit ones and
    average close values.
    """
    result = {}
    by_measurement = {}
    for dim in spec.dimensions:
        by_measurement.setdefault(dim.measurement, []).append(dim.value)

    for meas, vals in by_measurement.items():
        result[meas] = float(np.mean(vals))

    # Also try to get overall size
    w, h, d = spec.overall_size
    if "width" not in result and w > 0:
        result["width"] = float(w)
    if "height" not in result and h > 0:
        result["height"] = float(h)
    if "depth" not in result and d > 0:
        result["depth"] = float(d)

    return result


def _infer_base_shape(spec: DrawingSpec, dims: dict) -> CadOp:
    """Determine the base primitive from the spec.

    Decision tree:
    - axial symmetry + circular view → cylinder or revolve
    - all rectangular views → box
    - sphere-like description → sphere
    - complex profile → revolve
    """
    obj = spec.object_type.lower()
    sym = spec.symmetry.lower()

    # Sphere
    if obj in ("sphere", "ball") or _all_views_circular(spec):
        r = dims.get("radius", dims.get("diameter", 10.0) / 2)
        return CadOp("sphere", {"center": [0, 0, 0], "radius": r, "divs": 20})

    # Axially symmetric → cylinder or revolve
    if sym == "axial" or obj in ("shaft", "cylinder", "rod", "pipe", "tube",
                                  "flange", "column", "vase", "goblet",
                                  "candlestick", "bell", "cup", "bottle"):
        return _make_axial_shape(spec, dims)

    # Box-like (only if explicitly box-typed, not as a fallback)
    if obj in ("box", "block", "plate", "bracket"):
        return _make_box_shape(dims)

    # Gear-like → extrude
    if obj in ("gear", "spur_gear", "sprocket"):
        return _make_gear_shape(dims)

    # Default: try cylinder if we have diameter/height, else box
    if "diameter" in dims or "radius" in dims:
        return _make_axial_shape(spec, dims)

    return _make_box_shape(dims)


def _all_views_circular(spec: DrawingSpec) -> bool:
    """Check if all views suggest a circular outline (sphere)."""
    for view in spec.views:
        for feat in view.features:
            if feat.feature_type not in ("sphere", "circle"):
                return False
    return len(spec.views) > 0


def _make_axial_shape(spec: DrawingSpec, dims: dict) -> CadOp:
    """Build a cylinder or revolve op from dimensions."""
    r = dims.get("radius", dims.get("diameter", 10.0) / 2)
    h = dims.get("height", r * 2)

    # Check if we have a profile for revolve
    profile = _extract_profile(spec)
    if profile is not None and len(profile) >= 3:
        return CadOp("revolve", {
            "center": [0, 0, 0],
            "profile": profile,
            "divs": 48,
        })

    return CadOp("cylinder", {
        "center": [0, 0, h / 2],
        "axis": [0, 0, 1],
        "radius": r,
        "height": h,
        "radial_divs": 24,
        "height_divs": 10,
    })


def _make_box_shape(dims: dict) -> CadOp:
    """Build a box op from dimensions."""
    w = dims.get("width", 10.0)
    h = dims.get("height", 10.0)
    d = dims.get("depth", w)  # default to square cross-section
    return CadOp("box", {
        "center": [0, 0, h / 2],
        "dimensions": [w, d, h],
        "subdivisions": 4,
    })


def _make_gear_shape(dims: dict) -> CadOp:
    """Build an extrude op approximating a gear profile."""
    r = dims.get("radius", dims.get("diameter", 20.0) / 2)
    h = dims.get("height", dims.get("depth", 5.0))
    n_teeth = int(dims.get("teeth", 12))

    # Simple gear-ish polygon
    pts = []
    for i in range(n_teeth * 2):
        angle = 2 * np.pi * i / (n_teeth * 2)
        if i % 2 == 0:
            rr = r
        else:
            rr = r * 0.8
        pts.append([float(rr * np.cos(angle)), float(rr * np.sin(angle))])

    return CadOp("extrude", {
        "center": [0, 0, 0],
        "polygon": pts,
        "height": h,
    })


def _extract_profile(spec: DrawingSpec) -> list | None:
    """Try to extract a (r, z) revolve profile from view outlines.

    For axially symmetric parts, the front or side view outline's right
    half gives the profile.
    """
    for view in spec.views:
        if view.view_type in ("front", "side") and len(view.outline) >= 4:
            pts = np.array(view.outline)
            # Take right half (x > midpoint)
            mid_x = (pts[:, 0].max() + pts[:, 0].min()) / 2
            right = pts[pts[:, 0] >= mid_x]
            if len(right) >= 3:
                # Sort by y (height), convert to (r, z)
                right = right[right[:, 1].argsort()]
                w, h_size, _ = spec.overall_size
                r_scale = (w / 2) if w > 0 else 1.0
                z_scale = h_size if h_size > 0 else 1.0
                profile = []
                for x, y in right:
                    r_norm = (x - mid_x) / max(pts[:, 0].max() - mid_x, 1e-6)
                    z_norm = (y - pts[:, 1].min()) / max(pts[:, 1].max() - pts[:, 1].min(), 1e-6)
                    profile.append([float(r_norm * r_scale), float(z_norm * z_scale)])
                return profile
    return None


def _add_holes(program: CadProgram, spec: DrawingSpec, dims: dict):
    """Add subtract_cylinder ops for each hole feature."""
    for view in spec.views:
        for feat in view.features:
            if feat.feature_type != "hole":
                continue

            # Get hole dimensions
            hole_r = None
            hole_h = None
            for d in feat.dimensions:
                if d.measurement == "diameter":
                    hole_r = d.value / 2
                elif d.measurement == "radius":
                    hole_r = d.value
                elif d.measurement == "depth":
                    hole_h = d.value

            if hole_r is None:
                # Estimate from extent
                base_r = dims.get("radius", dims.get("diameter", 10.0) / 2)
                hole_r = base_r * 0.15  # default small hole

            # Determine axis from which view shows the circle
            if view.view_type == "top":
                axis = [0, 0, 1]
            elif view.view_type == "front":
                axis = [0, 1, 0]
            else:
                axis = [1, 0, 0]

            # Determine center from 2D position
            cx, cy = feat.center_2d
            w, h, d = spec.overall_size
            if view.view_type == "top":
                center = [(cx - 0.5) * w, (cy - 0.5) * d, h / 2]
            elif view.view_type == "front":
                center = [(cx - 0.5) * w, 0, (1 - cy) * h]
            else:
                center = [0, (cx - 0.5) * d, (1 - cy) * h]

            if hole_h is None:
                # Through hole — use full body dimension
                hole_h = max(w, h, d) * 1.5

            program.operations.append(CadOp("subtract_cylinder", {
                "center": [float(c) for c in center],
                "axis": [float(a) for a in axis],
                "radius": float(hole_r),
                "height": float(hole_h),
            }))


def _cross_reference_views(spec: DrawingSpec) -> dict:
    """Reconcile dimensions across views.

    Returns mapping of feature -> 3D dimensions.
    """
    features_3d = {}

    dims_by_view = {}
    for dim in spec.dimensions:
        dims_by_view.setdefault(dim.view, []).append(dim)

    # Overall body dimensions
    body = {}
    for dim in spec.dimensions:
        if dim.feature in ("body", "overall", ""):
            if dim.measurement == "width":
                body["width"] = dim.value
            elif dim.measurement == "height":
                body["height"] = dim.value
            elif dim.measurement == "depth":
                body["depth"] = dim.value
            elif dim.measurement == "diameter":
                body["diameter"] = dim.value

    features_3d["body"] = body
    return features_3d
