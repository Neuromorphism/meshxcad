"""FreeCAD generator for all catalog objects.

Creates parametric FreeCAD documents for each object in the catalog,
both simple and ornate versions.

Requires FreeCAD Python environment.
"""

import os
import sys
import math

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


def _profile_to_freecad_wire(profile_rz):
    """Convert a (radius, z) profile to a FreeCAD wire for revolution.

    Builds a closed wire: axis line + profile spline + cap lines.
    """
    import FreeCAD
    import Part

    pts = [FreeCAD.Vector(r, 0, z) for r, z in profile_rz]

    # Build spline through profile points
    spline = Part.BSplineCurve()
    spline.interpolate(pts)

    # Close the wire along the axis
    top = pts[-1]
    bot = pts[0]
    top_axis = FreeCAD.Vector(0, 0, top.z)
    bot_axis = FreeCAD.Vector(0, 0, bot.z)

    edges = [
        Part.LineSegment(bot_axis, bot).toShape(),
        spline.toShape(),
        Part.LineSegment(top, top_axis).toShape(),
        Part.LineSegment(top_axis, bot_axis).toShape(),
    ]

    return Part.Wire(edges)


def profile_to_solid(profile_rz):
    """Convert a (radius, z) profile to a solid of revolution."""
    import FreeCAD
    import Part

    wire = _profile_to_freecad_wire(profile_rz)
    face = Part.Face(wire)
    solid = face.revolve(FreeCAD.Vector(0, 0, 0), FreeCAD.Vector(0, 0, 1), 360)
    return solid


def make_torus_solid(major_r, minor_r, z_center):
    """Create a torus solid."""
    import FreeCAD
    import Part
    return Part.makeTorus(major_r, minor_r,
                           FreeCAD.Vector(0, 0, z_center),
                           FreeCAD.Vector(0, 0, 1))


def generate_freecad_object(name, profile_fn, output_path, extra_tori=None):
    """Generate a FreeCAD document from a profile function.

    Args:
        name: object name for the document
        profile_fn: callable returning list of (radius, z) tuples
        output_path: path to save .FCStd file
        extra_tori: optional list of (major_r, minor_r, z) for decorative rings
    """
    import FreeCAD
    import MeshPart

    doc = FreeCAD.newDocument(name)

    profile = profile_fn()
    if isinstance(profile, tuple) and len(profile) == 2:
        # It's a (vertices, faces) mesh tuple — this is a synthetic generator
        # We need to convert to a profile first
        raise ValueError(f"Expected profile function, got mesh tuple for {name}")

    main_solid = profile_to_solid(profile)

    if extra_tori:
        for major_r, minor_r, z in extra_tori:
            torus = make_torus_solid(major_r, minor_r, z)
            main_solid = main_solid.fuse(torus)

    obj = doc.addObject("Part::Feature", name)
    obj.Shape = main_solid
    doc.recompute()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    doc.saveAs(output_path)

    # Also export STL
    stl_path = output_path.replace(".FCStd", ".stl")
    mesh = MeshPart.meshFromShape(
        Shape=main_solid, LinearDeflection=0.1, AngularDeflection=0.3
    )
    mesh.write(stl_path)

    FreeCAD.closeDocument(doc.Name)
    return output_path, stl_path


def generate_all_freecad(output_dir="freecad_output"):
    """Generate FreeCAD documents for all catalog objects.

    This uses the smooth_profile data from the catalog to build
    actual FreeCAD parametric solids.
    """
    from .catalog import OBJECT_CATALOG
    from .builder import smooth_profile

    os.makedirs(output_dir, exist_ok=True)
    results = {}

    for name, entry in OBJECT_CATALOG.items():
        obj_dir = os.path.join(output_dir, name)
        os.makedirs(obj_dir, exist_ok=True)

        print(f"Generating {name}...")
        for variant in ["simple", "ornate"]:
            path = os.path.join(obj_dir, f"{variant}.FCStd")
            try:
                cad_path, stl_path = generate_freecad_object(
                    f"{name}_{variant}",
                    entry[variant],
                    path,
                )
                results.setdefault(name, {})[variant] = {
                    "cad": cad_path,
                    "stl": stl_path,
                }
            except Exception as e:
                print(f"  Failed {variant}: {e}")

    return results


if __name__ == "__main__":
    results = generate_all_freecad()
    for name, variants in results.items():
        print(f"\n{name}:")
        for variant, paths in variants.items():
            print(f"  {variant}: {paths}")
