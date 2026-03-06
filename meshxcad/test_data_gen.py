"""Generate test data sets for MeshXCAD.

Each test set produces four files:
  - plain.stl: plain mesh
  - featured.stl: mesh with surface features
  - plain.FCStd: plain parametric CAD
  - featured.FCStd: parametric CAD with features

Requires FreeCAD Python environment.
"""

import os
import math


def generate_cube_set(output_dir, size=10.0, feature_depth=1.0, feature_size=3.0):
    """Generate a cube test set.

    The featured version has a rectangular pocket cut into each face.

    Args:
        output_dir: directory to write the 4 files
        size: cube side length
        feature_depth: depth of pocket features
        feature_size: width/height of pocket features
    """
    import FreeCAD
    import Part
    import Mesh
    import MeshPart

    os.makedirs(output_dir, exist_ok=True)

    # --- Plain cube CAD ---
    doc = FreeCAD.newDocument("PlainCube")
    box = doc.addObject("Part::Box", "Cube")
    box.Length = size
    box.Width = size
    box.Height = size
    # Center at origin
    box.Placement.Base = FreeCAD.Vector(-size / 2, -size / 2, -size / 2)
    doc.recompute()

    # Save plain CAD
    plain_cad_path = os.path.join(output_dir, "plain.FCStd")
    doc.saveAs(plain_cad_path)

    # Export plain mesh
    shape = box.Shape
    mesh = MeshPart.meshFromShape(Shape=shape, LinearDeflection=0.1, AngularDeflection=0.5)
    plain_mesh_path = os.path.join(output_dir, "plain.stl")
    mesh.write(plain_mesh_path)

    FreeCAD.closeDocument(doc.Name)

    # --- Featured cube CAD (cube with pockets on each face) ---
    doc = FreeCAD.newDocument("FeaturedCube")
    box = doc.addObject("Part::Box", "Cube")
    box.Length = size
    box.Width = size
    box.Height = size
    box.Placement.Base = FreeCAD.Vector(-size / 2, -size / 2, -size / 2)
    doc.recompute()

    # Create pocket cuts on each face using Part::Cut
    half = size / 2
    fh = feature_size / 2
    pocket_shapes = []

    # Define 6 face-centered pockets (one per cube face)
    face_configs = [
        # (center, direction for extrusion)
        (FreeCAD.Vector(0, 0, half), FreeCAD.Vector(0, 0, -1)),   # +Z face
        (FreeCAD.Vector(0, 0, -half), FreeCAD.Vector(0, 0, 1)),   # -Z face
        (FreeCAD.Vector(half, 0, 0), FreeCAD.Vector(-1, 0, 0)),   # +X face
        (FreeCAD.Vector(-half, 0, 0), FreeCAD.Vector(1, 0, 0)),   # -X face
        (FreeCAD.Vector(0, half, 0), FreeCAD.Vector(0, -1, 0)),   # +Y face
        (FreeCAD.Vector(0, -half, 0), FreeCAD.Vector(0, 1, 0)),   # -Y face
    ]

    current_shape = box.Shape
    for i, (center, direction) in enumerate(face_configs):
        # Create a pocket box at each face
        pocket = Part.makeBox(
            feature_size, feature_size, feature_depth,
            center + direction * feature_depth - FreeCAD.Vector(fh, fh, 0),
        )

        # Orient the pocket box to align with the face
        if abs(direction.z) > 0.5:
            # Z faces: pocket is already aligned
            pocket = Part.makeBox(feature_size, feature_size, feature_depth)
            if direction.z < 0:
                pocket.Placement.Base = FreeCAD.Vector(
                    -fh, -fh, half - feature_depth
                )
            else:
                pocket.Placement.Base = FreeCAD.Vector(-fh, -fh, -half)
        elif abs(direction.x) > 0.5:
            pocket = Part.makeBox(feature_depth, feature_size, feature_size)
            if direction.x < 0:
                pocket.Placement.Base = FreeCAD.Vector(
                    half - feature_depth, -fh, -fh
                )
            else:
                pocket.Placement.Base = FreeCAD.Vector(-half, -fh, -fh)
        else:
            pocket = Part.makeBox(feature_size, feature_depth, feature_size)
            if direction.y < 0:
                pocket.Placement.Base = FreeCAD.Vector(
                    -fh, half - feature_depth, -fh
                )
            else:
                pocket.Placement.Base = FreeCAD.Vector(-fh, -half, -fh)

        current_shape = current_shape.cut(pocket)

    # Add the final shape as a Part::Feature
    feat = doc.addObject("Part::Feature", "FeaturedCube")
    feat.Shape = current_shape
    doc.removeObject("Cube")
    doc.recompute()

    # Save featured CAD
    featured_cad_path = os.path.join(output_dir, "featured.FCStd")
    doc.saveAs(featured_cad_path)

    # Export featured mesh
    mesh = MeshPart.meshFromShape(
        Shape=current_shape, LinearDeflection=0.1, AngularDeflection=0.5
    )
    featured_mesh_path = os.path.join(output_dir, "featured.stl")
    mesh.write(featured_mesh_path)

    FreeCAD.closeDocument(doc.Name)

    return {
        "plain_mesh": plain_mesh_path,
        "featured_mesh": featured_mesh_path,
        "plain_cad": plain_cad_path,
        "featured_cad": featured_cad_path,
    }


def generate_sphere_set(output_dir, radius=5.0, feature_depth=0.8, feature_angle=30.0):
    """Generate a sphere test set.

    The featured version has dimples (spherical indentations) at the poles
    and along the equator.

    Args:
        output_dir: directory to write the 4 files
        radius: sphere radius
        feature_depth: depth of dimple features
        feature_angle: angular size of dimples in degrees
    """
    import FreeCAD
    import Part
    import Mesh
    import MeshPart

    os.makedirs(output_dir, exist_ok=True)

    # --- Plain sphere CAD ---
    doc = FreeCAD.newDocument("PlainSphere")
    sphere = doc.addObject("Part::Sphere", "Sphere")
    sphere.Radius = radius
    doc.recompute()

    plain_cad_path = os.path.join(output_dir, "plain.FCStd")
    doc.saveAs(plain_cad_path)

    mesh = MeshPart.meshFromShape(
        Shape=sphere.Shape, LinearDeflection=0.1, AngularDeflection=0.3
    )
    plain_mesh_path = os.path.join(output_dir, "plain.stl")
    mesh.write(plain_mesh_path)

    FreeCAD.closeDocument(doc.Name)

    # --- Featured sphere CAD (with dimples) ---
    doc = FreeCAD.newDocument("FeaturedSphere")
    sphere_obj = doc.addObject("Part::Sphere", "Sphere")
    sphere_obj.Radius = radius
    doc.recompute()

    current_shape = sphere_obj.Shape

    # Create dimples at 6 cardinal points
    dimple_radius = radius * math.sin(math.radians(feature_angle))
    dimple_centers = [
        FreeCAD.Vector(0, 0, radius),
        FreeCAD.Vector(0, 0, -radius),
        FreeCAD.Vector(radius, 0, 0),
        FreeCAD.Vector(-radius, 0, 0),
        FreeCAD.Vector(0, radius, 0),
        FreeCAD.Vector(0, -radius, 0),
    ]

    for center in dimple_centers:
        dimple = Part.makeSphere(dimple_radius, center)
        current_shape = current_shape.cut(dimple)

    feat = doc.addObject("Part::Feature", "FeaturedSphere")
    feat.Shape = current_shape
    doc.removeObject("Sphere")
    doc.recompute()

    featured_cad_path = os.path.join(output_dir, "featured.FCStd")
    doc.saveAs(featured_cad_path)

    mesh = MeshPart.meshFromShape(
        Shape=current_shape, LinearDeflection=0.1, AngularDeflection=0.3
    )
    featured_mesh_path = os.path.join(output_dir, "featured.stl")
    mesh.write(featured_mesh_path)

    FreeCAD.closeDocument(doc.Name)

    return {
        "plain_mesh": plain_mesh_path,
        "featured_mesh": featured_mesh_path,
        "plain_cad": plain_cad_path,
        "featured_cad": featured_cad_path,
    }


def generate_cylinder_set(output_dir, radius=5.0, height=15.0,
                           groove_depth=0.8, groove_width=1.5, num_grooves=3):
    """Generate a cylinder test set.

    The featured version has circumferential grooves cut around the barrel.

    Args:
        output_dir: directory to write the 4 files
        radius: cylinder radius
        height: cylinder height
        groove_depth: depth of grooves
        groove_width: width of grooves
        num_grooves: number of grooves along the height
    """
    import FreeCAD
    import Part
    import Mesh
    import MeshPart

    os.makedirs(output_dir, exist_ok=True)

    # --- Plain cylinder CAD ---
    doc = FreeCAD.newDocument("PlainCylinder")
    cyl = doc.addObject("Part::Cylinder", "Cylinder")
    cyl.Radius = radius
    cyl.Height = height
    cyl.Placement.Base = FreeCAD.Vector(0, 0, -height / 2)
    doc.recompute()

    plain_cad_path = os.path.join(output_dir, "plain.FCStd")
    doc.saveAs(plain_cad_path)

    mesh = MeshPart.meshFromShape(
        Shape=cyl.Shape, LinearDeflection=0.1, AngularDeflection=0.3
    )
    plain_mesh_path = os.path.join(output_dir, "plain.stl")
    mesh.write(plain_mesh_path)

    FreeCAD.closeDocument(doc.Name)

    # --- Featured cylinder CAD (with grooves) ---
    doc = FreeCAD.newDocument("FeaturedCylinder")
    cyl_obj = doc.addObject("Part::Cylinder", "Cylinder")
    cyl_obj.Radius = radius
    cyl_obj.Height = height
    cyl_obj.Placement.Base = FreeCAD.Vector(0, 0, -height / 2)
    doc.recompute()

    current_shape = cyl_obj.Shape

    # Cut grooves as thin cylinders with larger radius minus the original
    spacing = height / (num_grooves + 1)
    for i in range(num_grooves):
        z_pos = -height / 2 + spacing * (i + 1) - groove_width / 2
        # Groove = hollow cylinder (tube) that cuts into the surface
        outer_cyl = Part.makeCylinder(radius + 1, groove_width,
                                       FreeCAD.Vector(0, 0, z_pos))
        inner_cyl = Part.makeCylinder(radius - groove_depth, groove_width,
                                       FreeCAD.Vector(0, 0, z_pos))
        groove = outer_cyl.cut(inner_cyl)
        current_shape = current_shape.cut(groove)

    feat = doc.addObject("Part::Feature", "FeaturedCylinder")
    feat.Shape = current_shape
    doc.removeObject("Cylinder")
    doc.recompute()

    featured_cad_path = os.path.join(output_dir, "featured.FCStd")
    doc.saveAs(featured_cad_path)

    mesh = MeshPart.meshFromShape(
        Shape=current_shape, LinearDeflection=0.1, AngularDeflection=0.3
    )
    featured_mesh_path = os.path.join(output_dir, "featured.stl")
    mesh.write(featured_mesh_path)

    FreeCAD.closeDocument(doc.Name)

    return {
        "plain_mesh": plain_mesh_path,
        "featured_mesh": featured_mesh_path,
        "plain_cad": plain_cad_path,
        "featured_cad": featured_cad_path,
    }


def generate_all(base_dir="test_data"):
    """Generate all test data sets.

    Args:
        base_dir: base directory for test data output

    Returns:
        dict mapping set name to file paths dict
    """
    results = {}
    results["cube"] = generate_cube_set(os.path.join(base_dir, "cube"))
    results["sphere"] = generate_sphere_set(os.path.join(base_dir, "sphere"))
    results["cylinder"] = generate_cylinder_set(os.path.join(base_dir, "cylinder"))
    return results


if __name__ == "__main__":
    results = generate_all()
    for name, paths in results.items():
        print(f"\n{name}:")
        for key, path in paths.items():
            print(f"  {key}: {path}")
