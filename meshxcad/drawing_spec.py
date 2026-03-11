"""Data structures for representing mechanical drawing interpretations.

A DrawingSpec captures the structured geometry extracted from a mechanical
drawing image — dimensions, features, views, and their relationships.
"""

import json
from dataclasses import dataclass, field, asdict


@dataclass
class Dimension:
    """A single dimension annotation from a drawing."""
    value: float
    unit: str = "mm"           # "mm", "in"
    measurement: str = ""      # "diameter", "height", "width", "radius", "angle", "depth"
    feature: str = ""          # what it measures ("bore", "flange_od", "body", "hole_1")
    view: str = ""             # which view ("front", "side", "top")


@dataclass
class Feature:
    """A geometric feature visible in a view."""
    feature_type: str          # "cylinder", "hole", "fillet", "chamfer", "flat",
                               # "thread", "slot", "counterbore", "sphere", "torus"
    view: str = ""             # which view it's visible in
    center_2d: tuple = (0.5, 0.5)  # (x, y) center in normalised view coords (0-1)
    extent_2d: tuple = (1.0, 1.0)  # (w, h) bounding box in view coords
    dimensions: list = field(default_factory=list)   # list[Dimension]
    through: bool = False      # True if hole goes all the way through


@dataclass
class ViewSpec:
    """One view within a multi-view drawing."""
    view_type: str             # "front", "side", "top", "section", "isometric", "detail"
    features: list = field(default_factory=list)  # list[Feature]
    outline: list = field(default_factory=list)   # ordered 2D boundary points (normalised)
    bounding_box: tuple = (0.0, 0.0, 1.0, 1.0)   # (x, y, w, h) in full image


@dataclass
class DrawingSpec:
    """Complete structured description of a mechanical drawing."""
    description: str = ""               # "Pipe flange with bolt holes"
    object_type: str = ""               # "shaft", "flange", "bracket", "gear", etc.
    views: list = field(default_factory=list)       # list[ViewSpec]
    dimensions: list = field(default_factory=list)  # list[Dimension] — all dims
    symmetry: str = "none"              # "axial", "bilateral", "radial", "none"
    overall_size: tuple = (1.0, 1.0, 1.0)  # (width, height, depth) in drawing units
    material: str = ""
    notes: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_dict(cls, d: dict) -> "DrawingSpec":
        views = [ViewSpec(
            view_type=v["view_type"],
            features=[Feature(**f) for f in v.get("features", [])],
            outline=v.get("outline", []),
            bounding_box=tuple(v.get("bounding_box", (0, 0, 1, 1))),
        ) for v in d.get("views", [])]

        dimensions = [Dimension(**dim) for dim in d.get("dimensions", [])]

        return cls(
            description=d.get("description", ""),
            object_type=d.get("object_type", ""),
            views=views,
            dimensions=dimensions,
            symmetry=d.get("symmetry", "none"),
            overall_size=tuple(d.get("overall_size", (1, 1, 1))),
            material=d.get("material", ""),
            notes=d.get("notes", []),
        )

    @classmethod
    def from_json(cls, s: str) -> "DrawingSpec":
        return cls.from_dict(json.loads(s))
