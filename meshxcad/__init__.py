"""MeshXCAD: Bidirectional detail transfer between meshes and parametric CAD."""

__version__ = "0.1.0"

# Expose optimization / differentiable strategy selection
from .optim import (  # noqa: F401
    SegmentationStrategySelector,
    FixerPrioritySelector,
    DifferentiableRefiner,
    mesh_features,
    has_autograd,
)
