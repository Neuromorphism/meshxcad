"""CAD I/O operations using FreeCAD."""

import os


def load_cad(filepath):
    """Load a FreeCAD document. Returns the FreeCAD document object."""
    import FreeCAD
    doc = FreeCAD.openDocument(filepath)
    return doc


def save_cad(doc, filepath):
    """Save a FreeCAD document to file."""
    doc.saveAs(filepath)


def cad_to_shape(doc):
    """Extract the compound shape from all solid bodies in a FreeCAD document.

    Returns:
        A FreeCAD TopoShape (the fused result of all bodies).
    """
    import Part
    shapes = []
    for obj in doc.Objects:
        if hasattr(obj, "Shape") and obj.Shape.Solids:
            shapes.append(obj.Shape)
    if not shapes:
        raise ValueError("No solid shapes found in CAD document")
    if len(shapes) == 1:
        return shapes[0]
    result = shapes[0]
    for s in shapes[1:]:
        result = result.fuse(s)
    return result


def shape_to_mesh(shape, linear_deflection=0.1, angular_deflection=0.5):
    """Tessellate a CAD shape into a mesh.

    Args:
        shape: FreeCAD TopoShape
        linear_deflection: max linear error for tessellation
        angular_deflection: max angular error in radians

    Returns:
        FreeCAD Mesh object
    """
    import Mesh
    import MeshPart
    return MeshPart.meshFromShape(
        Shape=shape,
        LinearDeflection=linear_deflection,
        AngularDeflection=angular_deflection,
    )


def new_document(name="MeshXCAD"):
    """Create a new FreeCAD document."""
    import FreeCAD
    return FreeCAD.newDocument(name)


def close_document(doc):
    """Close a FreeCAD document."""
    import FreeCAD
    FreeCAD.closeDocument(doc.Name)
