"""Simplified hourglass model — major features only, no decorative detail.

This creates a plain hourglass with:
- Simple smooth glass body (two plain bulbs, no wall thickness)
- Four straight cylindrical pillars (no turned details)
- Plain flat circular plates (no molding)
- No decorative rings or finial

This serves as the "plain CAD" that we want to add detail onto from the
ornate mesh.
"""

import math
import sys
import os

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

# Same overall dimensions as the ornate model
GLASS_HEIGHT = 120.0
GLASS_BULB_RADIUS = 30.0
GLASS_NECK_RADIUS = 4.0
PLATE_THICKNESS = 8.0
PLATE_RADIUS = 45.0
NUM_PILLARS = 4
PILLAR_ORBIT_RADIUS = 38.0
PILLAR_RADIUS = 5.0
PILLAR_HEIGHT = GLASS_HEIGHT + 2 * PLATE_THICKNESS


def make_simple_glass_body():
    """Create a simple glass body — solid of revolution with smooth bulb profile.

    No wall thickness, no fine detail — just the basic hourglass silhouette.
    """
    half = GLASS_HEIGHT / 2.0

    # Simplified profile: fewer control points, smoother shape
    upper_pts = [
        FreeCAD.Vector(GLASS_NECK_RADIUS, 0, 0),
        FreeCAD.Vector(GLASS_BULB_RADIUS * 0.7, 0, half * 0.25),
        FreeCAD.Vector(GLASS_BULB_RADIUS, 0, half * 0.55),
        FreeCAD.Vector(GLASS_BULB_RADIUS * 0.7, 0, half * 0.9),
        FreeCAD.Vector(GLASS_NECK_RADIUS + 8, 0, half),
    ]

    upper_spline = Part.BSplineCurve()
    upper_spline.interpolate(upper_pts)

    lower_pts = [FreeCAD.Vector(p.x, 0, -p.z) for p in reversed(upper_pts)]
    lower_spline = Part.BSplineCurve()
    lower_spline.interpolate(lower_pts)

    top_pt = upper_pts[-1]
    bot_pt = lower_pts[0]

    edges = [
        Part.LineSegment(FreeCAD.Vector(0, 0, -half), bot_pt).toShape(),
        lower_spline.toShape(),
        upper_spline.toShape(),
        Part.LineSegment(top_pt, FreeCAD.Vector(0, 0, half)).toShape(),
        Part.LineSegment(FreeCAD.Vector(0, 0, half), FreeCAD.Vector(0, 0, -half)).toShape(),
    ]

    wire = Part.Wire(edges)
    face = Part.Face(wire)
    solid = face.revolve(FreeCAD.Vector(0, 0, 0), FreeCAD.Vector(0, 0, 1), 360)
    return solid


def make_simple_plate(z_position):
    """Create a plain flat circular plate (no molding or chamfers)."""
    plate = Part.makeCylinder(PLATE_RADIUS, PLATE_THICKNESS,
                               FreeCAD.Vector(0, 0, z_position))
    return plate


def make_simple_pillars():
    """Create four plain straight cylindrical pillars."""
    z_bottom = -GLASS_HEIGHT / 2 - PLATE_THICKNESS
    pillars = []

    for i in range(NUM_PILLARS):
        angle = math.radians(45 + i * 90)
        x = PILLAR_ORBIT_RADIUS * math.cos(angle)
        y = PILLAR_ORBIT_RADIUS * math.sin(angle)

        pillar = Part.makeCylinder(
            PILLAR_RADIUS, PILLAR_HEIGHT,
            FreeCAD.Vector(x, y, z_bottom),
        )
        pillars.append(pillar)

    result = pillars[0]
    for p in pillars[1:]:
        result = result.fuse(p)
    return result


def make_simple_hourglass(doc_name="SimpleHourglass"):
    """Build the simplified hourglass and return the FreeCAD document."""
    doc = FreeCAD.newDocument(doc_name)

    # Glass body
    glass = make_simple_glass_body()
    glass_obj = doc.addObject("Part::Feature", "GlassBody")
    glass_obj.Shape = glass

    # Bottom plate
    bottom_z = -GLASS_HEIGHT / 2 - PLATE_THICKNESS
    bottom_plate = make_simple_plate(bottom_z)
    bp_obj = doc.addObject("Part::Feature", "BottomPlate")
    bp_obj.Shape = bottom_plate

    # Top plate
    top_z = GLASS_HEIGHT / 2
    top_plate = make_simple_plate(top_z)
    tp_obj = doc.addObject("Part::Feature", "TopPlate")
    tp_obj.Shape = top_plate

    # Pillars
    pillars = make_simple_pillars()
    pil_obj = doc.addObject("Part::Feature", "Pillars")
    pil_obj.Shape = pillars

    doc.recompute()
    return doc


if __name__ == "__main__":
    output_dir = os.path.join(os.path.dirname(__file__), "..", "hourglass")
    os.makedirs(output_dir, exist_ok=True)

    doc = make_simple_hourglass()

    cad_path = os.path.join(output_dir, "simple_hourglass.FCStd")
    doc.saveAs(cad_path)
    print(f"Saved CAD: {cad_path}")

    import MeshPart
    shapes = []
    for obj in doc.Objects:
        if hasattr(obj, "Shape") and obj.Shape.Solids:
            shapes.append(obj.Shape)
    compound = shapes[0]
    for s in shapes[1:]:
        compound = compound.fuse(s)

    mesh = MeshPart.meshFromShape(Shape=compound, LinearDeflection=0.1, AngularDeflection=0.3)
    stl_path = os.path.join(output_dir, "simple_hourglass.stl")
    mesh.write(stl_path)
    print(f"Saved STL: {stl_path}")

    FreeCAD.closeDocument(doc.Name)
