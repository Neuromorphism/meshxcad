"""Ornate hourglass model generator for FreeCAD.

Creates a detailed hourglass with:
- Glass body: two hyperbolic bulbs joined at a narrow neck
- Four turned wood pillars with lathe-style decorative profiles
- Top and bottom plates with stepped molding and chamfers
- Decorative rings at the glass-plate junctions
- A finial on top

All dimensions in millimeters. The hourglass is centered at the origin,
standing upright along the Z axis.
"""

import math
import sys
import os

# Add FreeCAD to path if needed
FREECAD_PATHS = [
    "/usr/lib/freecad-python3/lib",
    "/usr/lib/freecad/lib",
    "/usr/share/freecad/lib",
    "/usr/lib64/freecad/lib",
    "/snap/freecad/current/usr/lib/freecad-python3/lib",
    "/Applications/FreeCAD.app/Contents/Resources/lib",
]
for p in FREECAD_PATHS:
    if os.path.isdir(p) and p not in sys.path:
        sys.path.append(p)

import FreeCAD
import Part

# ============================================================================
# Parameters
# ============================================================================

# Overall dimensions
TOTAL_HEIGHT = 200.0       # total height including finial
GLASS_HEIGHT = 120.0       # height of the glass body (both bulbs)
PLATE_THICKNESS = 8.0      # thickness of top/bottom plates
PLATE_RADIUS = 45.0        # radius of the base/top plates

# Glass body
GLASS_BULB_RADIUS = 30.0   # max radius of each glass bulb
GLASS_NECK_RADIUS = 4.0    # radius at the narrow waist
GLASS_WALL_THICKNESS = 2.0 # wall thickness of the glass

# Pillars
NUM_PILLARS = 4
PILLAR_ORBIT_RADIUS = 38.0  # distance from center to pillar axis
PILLAR_RADIUS = 5.0         # main shaft radius
PILLAR_HEIGHT = GLASS_HEIGHT + 2 * PLATE_THICKNESS

# Decorative features
MOLDING_STEPS = 3           # number of stepped rings on plate edges
RING_THICKNESS = 2.0        # thickness of decorative rings
FINIAL_HEIGHT = 15.0        # height of top finial
FINIAL_RADIUS = 6.0         # base radius of finial


def _make_glass_profile():
    """Create the outer profile curve for one glass bulb (half the hourglass).

    The profile is a smooth curve from the neck (bottom) to the equator (top),
    using a series of spline points that create a pleasing bulb shape.

    Returns a list of FreeCAD.Vector points for a BSpline, in the XZ plane,
    where X is the radial distance and Z goes from 0 (neck) to GLASS_HEIGHT/2.
    """
    half = GLASS_HEIGHT / 2.0
    # Control points from neck to equator
    points = [
        FreeCAD.Vector(GLASS_NECK_RADIUS, 0, 0),
        FreeCAD.Vector(GLASS_NECK_RADIUS + 2, 0, half * 0.05),
        FreeCAD.Vector(GLASS_BULB_RADIUS * 0.6, 0, half * 0.15),
        FreeCAD.Vector(GLASS_BULB_RADIUS * 0.85, 0, half * 0.3),
        FreeCAD.Vector(GLASS_BULB_RADIUS * 0.97, 0, half * 0.5),
        FreeCAD.Vector(GLASS_BULB_RADIUS, 0, half * 0.65),
        FreeCAD.Vector(GLASS_BULB_RADIUS * 0.98, 0, half * 0.8),
        FreeCAD.Vector(GLASS_BULB_RADIUS * 0.85, 0, half * 0.92),
        FreeCAD.Vector(GLASS_BULB_RADIUS * 0.6, 0, half * 0.98),
        FreeCAD.Vector(GLASS_NECK_RADIUS + 8, 0, half),
    ]
    return points


def make_glass_body():
    """Create the glass body — two bulbs joined at a narrow neck.

    The glass is a thin-walled revolve: outer surface minus inner surface.
    """
    half = GLASS_HEIGHT / 2.0

    # Build the outer profile for the upper bulb
    upper_pts = _make_glass_profile()
    upper_spline = Part.BSplineCurve()
    upper_spline.interpolate(upper_pts)

    # Mirror for the lower bulb (negate Z)
    lower_pts = [FreeCAD.Vector(p.x, 0, -p.z) for p in reversed(upper_pts)]
    lower_spline = Part.BSplineCurve()
    lower_spline.interpolate(lower_pts)

    # Connect into a closed profile:
    # bottom cap → lower spline → upper spline → top cap → axis line back
    top_pt = upper_pts[-1]
    bot_pt = lower_pts[0]

    # Close the profile with lines along the axis
    top_line = Part.LineSegment(top_pt, FreeCAD.Vector(0, 0, half))
    axis_top = FreeCAD.Vector(0, 0, half)
    axis_bot = FreeCAD.Vector(0, 0, -half)
    axis_line = Part.LineSegment(axis_top, axis_bot)
    bot_line = Part.LineSegment(FreeCAD.Vector(0, 0, -half), bot_pt)

    # Build wire from edges
    wire = Part.Wire([
        bot_line.toShape(),
        lower_spline.toShape(),
        upper_spline.toShape(),
        top_line.toShape(),
        axis_line.toShape(),
    ])

    face = Part.Face(wire)
    outer_solid = face.revolve(FreeCAD.Vector(0, 0, 0), FreeCAD.Vector(0, 0, 1), 360)

    # Inner surface (offset inward by wall thickness)
    inner_pts_upper = []
    for p in _make_glass_profile():
        r = max(p.x - GLASS_WALL_THICKNESS, GLASS_NECK_RADIUS - 0.5)
        inner_pts_upper.append(FreeCAD.Vector(r, 0, p.z))

    inner_upper = Part.BSplineCurve()
    inner_upper.interpolate(inner_pts_upper)

    inner_pts_lower = [FreeCAD.Vector(p.x, 0, -p.z) for p in reversed(inner_pts_upper)]
    inner_lower = Part.BSplineCurve()
    inner_lower.interpolate(inner_pts_lower)

    i_top = inner_pts_upper[-1]
    i_bot = inner_pts_lower[0]

    i_top_line = Part.LineSegment(i_top, FreeCAD.Vector(0, 0, half - GLASS_WALL_THICKNESS))
    i_axis_top = FreeCAD.Vector(0, 0, half - GLASS_WALL_THICKNESS)
    i_axis_bot = FreeCAD.Vector(0, 0, -(half - GLASS_WALL_THICKNESS))
    i_axis_line = Part.LineSegment(i_axis_top, i_axis_bot)
    i_bot_line = Part.LineSegment(i_axis_bot, i_bot)

    i_wire = Part.Wire([
        i_bot_line.toShape(),
        inner_lower.toShape(),
        inner_upper.toShape(),
        i_top_line.toShape(),
        i_axis_line.toShape(),
    ])

    i_face = Part.Face(i_wire)
    inner_solid = i_face.revolve(FreeCAD.Vector(0, 0, 0), FreeCAD.Vector(0, 0, 1), 360)

    glass = outer_solid.cut(inner_solid)
    return glass


def _pillar_profile_points(height):
    """Generate the radial profile for a turned pillar.

    Returns list of (z, radius) tuples from bottom to top.
    The profile includes decorative beads, coves, and tapers.
    """
    r = PILLAR_RADIUS
    # Decorative lathe profile: base bead, taper in, mid-bead, taper in, top bead
    points = [
        (0, r * 1.3),              # base flare
        (2, r * 1.3),              # base flare top
        (4, r * 0.7),              # cove
        (8, r * 1.1),              # first bead
        (10, r * 1.1),             # first bead top
        (12, r * 0.65),            # narrow
        (16, r * 0.85),            # swell
        (height * 0.15, r * 0.75), # taper down
        (height * 0.25, r * 0.9),  # mid bead start
        (height * 0.28, r * 1.0),  # mid bead peak
        (height * 0.31, r * 0.9),  # mid bead end
        (height * 0.35, r * 0.7),  # narrow section
        (height * 0.45, r * 0.65), # long taper
        (height * 0.5, r * 0.8),   # center bead
        (height * 0.52, r * 0.85), # center bead peak
        (height * 0.54, r * 0.8),  # center bead end
        (height * 0.55, r * 0.65), # narrow
        (height * 0.65, r * 0.7),  # gradual swell
        (height * 0.69, r * 0.9),  # upper mid bead
        (height * 0.72, r * 1.0),  # upper mid bead peak
        (height * 0.75, r * 0.9),  # upper mid bead end
        (height * 0.85, r * 0.75), # taper
        (height - 16, r * 0.85),   # approach top
        (height - 12, r * 0.65),   # narrow
        (height - 10, r * 1.1),    # top bead
        (height - 8, r * 1.1),     # top bead base
        (height - 4, r * 0.7),     # cove
        (height - 2, r * 1.3),     # top flare
        (height, r * 1.3),         # top flare end
    ]
    return points


def make_single_pillar(height):
    """Create a single turned pillar as a solid of revolution.

    The pillar stands from z=0 to z=height, centered on the Z axis.
    """
    profile = _pillar_profile_points(height)

    # Build the profile as a wire in the XZ plane
    edges = []

    # Bottom cap: axis to first profile point
    edges.append(Part.LineSegment(
        FreeCAD.Vector(0, 0, profile[0][0]),
        FreeCAD.Vector(profile[0][1], 0, profile[0][0]),
    ).toShape())

    # Profile curve through all points
    spline_pts = [FreeCAD.Vector(r, 0, z) for z, r in profile]
    spline = Part.BSplineCurve()
    spline.interpolate(spline_pts)
    edges.append(spline.toShape())

    # Top cap: last profile point to axis
    edges.append(Part.LineSegment(
        FreeCAD.Vector(profile[-1][1], 0, profile[-1][0]),
        FreeCAD.Vector(0, 0, profile[-1][0]),
    ).toShape())

    # Close along axis
    edges.append(Part.LineSegment(
        FreeCAD.Vector(0, 0, profile[-1][0]),
        FreeCAD.Vector(0, 0, profile[0][0]),
    ).toShape())

    wire = Part.Wire(edges)
    face = Part.Face(wire)
    solid = face.revolve(FreeCAD.Vector(0, 0, 0), FreeCAD.Vector(0, 0, 1), 360)
    return solid


def make_pillars():
    """Create four turned pillars arranged around the glass body."""
    pillar_template = make_single_pillar(PILLAR_HEIGHT)
    pillars = []

    # Position: start at bottom plate top surface
    z_bottom = -GLASS_HEIGHT / 2 - PLATE_THICKNESS

    for i in range(NUM_PILLARS):
        angle = math.radians(45 + i * 90)  # 45°, 135°, 225°, 315°
        x = PILLAR_ORBIT_RADIUS * math.cos(angle)
        y = PILLAR_ORBIT_RADIUS * math.sin(angle)

        pillar = pillar_template.copy()
        pillar.translate(FreeCAD.Vector(x, y, z_bottom))
        pillars.append(pillar)

    result = pillars[0]
    for p in pillars[1:]:
        result = result.fuse(p)
    return result


def _make_plate_profile(thickness, with_molding=True):
    """Create a cross-section profile for a plate with stepped molding.

    The profile is in the XZ plane. X = radial, Z = height.
    Returns a closed wire.
    """
    r = PLATE_RADIUS
    t = thickness
    edges = []

    if with_molding:
        # Stepped molding profile on the outer edge
        pts = [
            FreeCAD.Vector(0, 0, 0),
            FreeCAD.Vector(r - 8, 0, 0),
            # Step 1
            FreeCAD.Vector(r - 8, 0, 1.5),
            FreeCAD.Vector(r - 5, 0, 1.5),
            # Step 2
            FreeCAD.Vector(r - 5, 0, 3.0),
            FreeCAD.Vector(r - 2.5, 0, 3.0),
            # Chamfer to full radius
            FreeCAD.Vector(r, 0, 4.5),
            # Outer edge
            FreeCAD.Vector(r, 0, t - 4.5),
            # Upper chamfer
            FreeCAD.Vector(r - 2.5, 0, t - 3.0),
            FreeCAD.Vector(r - 5, 0, t - 3.0),
            FreeCAD.Vector(r - 5, 0, t - 1.5),
            FreeCAD.Vector(r - 8, 0, t - 1.5),
            FreeCAD.Vector(r - 8, 0, t),
            FreeCAD.Vector(0, 0, t),
        ]
        for j in range(len(pts) - 1):
            edges.append(Part.LineSegment(pts[j], pts[j + 1]).toShape())
        edges.append(Part.LineSegment(pts[-1], pts[0]).toShape())
    else:
        # Simple rectangular profile
        pts = [
            FreeCAD.Vector(0, 0, 0),
            FreeCAD.Vector(r, 0, 0),
            FreeCAD.Vector(r, 0, t),
            FreeCAD.Vector(0, 0, t),
        ]
        for j in range(len(pts) - 1):
            edges.append(Part.LineSegment(pts[j], pts[j + 1]).toShape())
        edges.append(Part.LineSegment(pts[-1], pts[0]).toShape())

    return Part.Wire(edges)


def make_plate(z_position, flip=False):
    """Create a plate (top or bottom) with decorative molding.

    Args:
        z_position: Z coordinate of the plate bottom surface
        flip: if True, the molding faces downward (for top plate)
    """
    wire = _make_plate_profile(PLATE_THICKNESS, with_molding=True)
    face = Part.Face(wire)
    solid = face.revolve(FreeCAD.Vector(0, 0, 0), FreeCAD.Vector(0, 0, 1), 360)

    if flip:
        # Mirror around the plate center
        solid.rotate(FreeCAD.Vector(0, 0, PLATE_THICKNESS / 2),
                     FreeCAD.Vector(1, 0, 0), 180)

    solid.translate(FreeCAD.Vector(0, 0, z_position))
    return solid


def make_decorative_ring(z_position, inner_radius, outer_radius, height):
    """Create a decorative ring (torus-like) at the given Z position."""
    ring = Part.makeTorus(
        (inner_radius + outer_radius) / 2,  # major radius
        (outer_radius - inner_radius) / 2,   # minor radius
        FreeCAD.Vector(0, 0, z_position),
        FreeCAD.Vector(0, 0, 1),
    )
    return ring


def make_finial():
    """Create a decorative finial for the top of the hourglass.

    Shape: a small turned knob with a pointed tip.
    """
    z_base = GLASS_HEIGHT / 2 + PLATE_THICKNESS

    # Profile points (z relative to base, radius)
    pts = [
        FreeCAD.Vector(0, 0, z_base),
        FreeCAD.Vector(FINIAL_RADIUS * 0.8, 0, z_base),
        FreeCAD.Vector(FINIAL_RADIUS, 0, z_base + 3),
        FreeCAD.Vector(FINIAL_RADIUS * 0.9, 0, z_base + 5),
        FreeCAD.Vector(FINIAL_RADIUS * 0.5, 0, z_base + 7),
        FreeCAD.Vector(FINIAL_RADIUS * 0.7, 0, z_base + 9),
        FreeCAD.Vector(FINIAL_RADIUS * 0.6, 0, z_base + 11),
        FreeCAD.Vector(FINIAL_RADIUS * 0.3, 0, z_base + 13),
        FreeCAD.Vector(0.5, 0, z_base + FINIAL_HEIGHT),
        FreeCAD.Vector(0, 0, z_base + FINIAL_HEIGHT + 1),
    ]

    edges = []
    # Axis from top to base
    edges.append(Part.LineSegment(pts[-1], pts[0]).toShape())
    # Bottom line from axis to first profile point
    edges.append(Part.LineSegment(pts[0], pts[1]).toShape())
    # Spline through profile
    spline = Part.BSplineCurve()
    spline.interpolate(pts[1:])
    edges.append(spline.toShape())

    wire = Part.Wire(edges)
    face = Part.Face(wire)
    solid = face.revolve(FreeCAD.Vector(0, 0, 0), FreeCAD.Vector(0, 0, 1), 360)
    return solid


def make_ornate_hourglass(doc_name="OrnateHourglass"):
    """Build the complete ornate hourglass and return the FreeCAD document.

    Returns:
        doc: FreeCAD document with all components
    """
    doc = FreeCAD.newDocument(doc_name)

    # --- Glass body ---
    glass = make_glass_body()
    glass_obj = doc.addObject("Part::Feature", "GlassBody")
    glass_obj.Shape = glass

    # --- Bottom plate ---
    bottom_z = -GLASS_HEIGHT / 2 - PLATE_THICKNESS
    bottom_plate = make_plate(bottom_z, flip=False)
    bp_obj = doc.addObject("Part::Feature", "BottomPlate")
    bp_obj.Shape = bottom_plate

    # --- Top plate ---
    top_z = GLASS_HEIGHT / 2
    top_plate = make_plate(top_z, flip=True)
    tp_obj = doc.addObject("Part::Feature", "TopPlate")
    tp_obj.Shape = top_plate

    # --- Pillars ---
    pillars = make_pillars()
    pil_obj = doc.addObject("Part::Feature", "Pillars")
    pil_obj.Shape = pillars

    # --- Decorative rings at glass-plate junctions ---
    ring_r_inner = GLASS_NECK_RADIUS + 6
    ring_r_outer = ring_r_inner + 4

    # Bottom junction ring
    bot_ring = make_decorative_ring(
        -GLASS_HEIGHT / 2, ring_r_inner, ring_r_outer, RING_THICKNESS
    )
    br_obj = doc.addObject("Part::Feature", "BottomRing")
    br_obj.Shape = bot_ring

    # Top junction ring
    top_ring = make_decorative_ring(
        GLASS_HEIGHT / 2, ring_r_inner, ring_r_outer, RING_THICKNESS
    )
    tr_obj = doc.addObject("Part::Feature", "TopRing")
    tr_obj.Shape = top_ring

    # Center ring at the neck
    center_ring = make_decorative_ring(
        0, GLASS_NECK_RADIUS + 1, GLASS_NECK_RADIUS + 4, RING_THICKNESS
    )
    cr_obj = doc.addObject("Part::Feature", "CenterRing")
    cr_obj.Shape = center_ring

    # --- Finial ---
    finial = make_finial()
    fin_obj = doc.addObject("Part::Feature", "Finial")
    fin_obj.Shape = finial

    doc.recompute()
    return doc


if __name__ == "__main__":
    output_dir = os.path.join(os.path.dirname(__file__), "..", "hourglass")
    os.makedirs(output_dir, exist_ok=True)

    doc = make_ornate_hourglass()

    # Save FreeCAD document
    cad_path = os.path.join(output_dir, "ornate_hourglass.FCStd")
    doc.saveAs(cad_path)
    print(f"Saved CAD: {cad_path}")

    # Export to STL
    import MeshPart
    shapes = []
    for obj in doc.Objects:
        if hasattr(obj, "Shape") and obj.Shape.Solids:
            shapes.append(obj.Shape)
    compound = shapes[0]
    for s in shapes[1:]:
        compound = compound.fuse(s)

    mesh = MeshPart.meshFromShape(Shape=compound, LinearDeflection=0.1, AngularDeflection=0.3)
    stl_path = os.path.join(output_dir, "ornate_hourglass.stl")
    mesh.write(stl_path)
    print(f"Saved STL: {stl_path}")

    FreeCAD.closeDocument(doc.Name)
