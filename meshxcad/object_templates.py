"""Object templates for template-guided mesh segmentation and CAD reconstruction.

Each template encodes structural knowledge about a class of objects:
  - Expected parts and their geometric types (cylinder, sphere, box, etc.)
  - Spatial relationships between parts
  - Typical proportions and symmetry axes
  - CAD action hints (extrude, revolve, loft, sweep, boolean)

Templates are indexed by CIFAR-100 fine-grained categories (100 classes
across 20 superclasses) and extended with additional common CAD object types.

Usage:
    from meshxcad.object_templates import get_template, match_template, ALL_TEMPLATES
    tpl = get_template("bicycle")
    best = match_template(vertices, faces)  # auto-detect best template
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict


# ---------------------------------------------------------------------------
# Part descriptor
# ---------------------------------------------------------------------------

@dataclass
class PartHint:
    """Geometric hint for one part of an object."""
    name: str
    geometry: str  # cylinder, sphere, box, cone, revolve, extrude, sweep, freeform
    cad_action: str  # extrude, revolve, loft, sweep, boolean_cut, boolean_add, freeform
    symmetry: Optional[str] = None  # axial, bilateral, radial, none
    typical_fraction: float = 0.1  # fraction of total volume
    parent: Optional[str] = None  # name of parent part for hierarchy
    count: int = 1  # expected number of instances
    notes: str = ""


@dataclass
class ObjectTemplate:
    """Template describing the structural decomposition of an object class."""
    name: str
    superclass: str
    parts: List[PartHint]
    symmetry: Optional[str] = None  # overall symmetry
    primary_axis: str = "z"  # dominant axis: x, y, or z
    elongation: str = "moderate"  # compact, moderate, elongated, very_elongated
    segmentation_strategy: str = "skeleton"  # skeleton, sdf, convexity, projection, normal_cluster
    notes: str = ""
    aliases: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# CIFAR-100 templates (100 fine-grained classes across 20 superclasses)
# ---------------------------------------------------------------------------

def _make_templates() -> Dict[str, ObjectTemplate]:
    """Build the full template dictionary."""
    templates = {}

    def _add(t):
        templates[t.name] = t
        for a in t.aliases:
            templates[a] = t

    # ===================================================================
    # Superclass 1: aquatic_mammals
    # ===================================================================
    _add(ObjectTemplate(
        name="beaver", superclass="aquatic_mammals",
        symmetry="bilateral", primary_axis="y", elongation="moderate",
        segmentation_strategy="skeleton",
        parts=[
            PartHint("body", "revolve", "loft", symmetry="bilateral", typical_fraction=0.6),
            PartHint("head", "sphere", "revolve", typical_fraction=0.1, parent="body"),
            PartHint("tail", "extrude", "loft", typical_fraction=0.1, parent="body",
                     notes="Flat paddle-shaped tail"),
            PartHint("leg", "cylinder", "sweep", count=4, typical_fraction=0.04, parent="body"),
        ],
    ))
    _add(ObjectTemplate(
        name="dolphin", superclass="aquatic_mammals",
        symmetry="bilateral", primary_axis="y", elongation="elongated",
        segmentation_strategy="skeleton",
        parts=[
            PartHint("body", "revolve", "loft", symmetry="bilateral", typical_fraction=0.7),
            PartHint("head", "sphere", "revolve", typical_fraction=0.1, parent="body"),
            PartHint("dorsal_fin", "extrude", "loft", typical_fraction=0.03, parent="body"),
            PartHint("pectoral_fin", "extrude", "loft", count=2, typical_fraction=0.02, parent="body"),
            PartHint("tail_fluke", "extrude", "loft", typical_fraction=0.05, parent="body"),
        ],
    ))
    _add(ObjectTemplate(
        name="otter", superclass="aquatic_mammals",
        symmetry="bilateral", primary_axis="y", elongation="elongated",
        segmentation_strategy="skeleton",
        parts=[
            PartHint("body", "revolve", "loft", symmetry="bilateral", typical_fraction=0.5),
            PartHint("head", "sphere", "revolve", typical_fraction=0.1, parent="body"),
            PartHint("tail", "cylinder", "loft", typical_fraction=0.15, parent="body"),
            PartHint("leg", "cylinder", "sweep", count=4, typical_fraction=0.04, parent="body"),
        ],
    ))
    _add(ObjectTemplate(
        name="seal", superclass="aquatic_mammals",
        symmetry="bilateral", primary_axis="y", elongation="elongated",
        segmentation_strategy="skeleton",
        parts=[
            PartHint("body", "revolve", "loft", symmetry="bilateral", typical_fraction=0.7),
            PartHint("head", "sphere", "revolve", typical_fraction=0.1, parent="body"),
            PartHint("flipper", "extrude", "loft", count=4, typical_fraction=0.03, parent="body"),
        ],
    ))
    _add(ObjectTemplate(
        name="whale", superclass="aquatic_mammals",
        symmetry="bilateral", primary_axis="y", elongation="elongated",
        segmentation_strategy="skeleton",
        parts=[
            PartHint("body", "revolve", "loft", symmetry="bilateral", typical_fraction=0.75),
            PartHint("head", "sphere", "revolve", typical_fraction=0.1, parent="body"),
            PartHint("tail_fluke", "extrude", "loft", typical_fraction=0.05, parent="body"),
            PartHint("pectoral_fin", "extrude", "loft", count=2, typical_fraction=0.02, parent="body"),
        ],
    ))

    # ===================================================================
    # Superclass 2: fish
    # ===================================================================
    _add(ObjectTemplate(
        name="aquarium_fish", superclass="fish",
        symmetry="bilateral", primary_axis="y", elongation="moderate",
        segmentation_strategy="skeleton",
        parts=[
            PartHint("body", "revolve", "loft", symmetry="bilateral", typical_fraction=0.7),
            PartHint("tail_fin", "extrude", "loft", typical_fraction=0.1, parent="body"),
            PartHint("dorsal_fin", "extrude", "loft", typical_fraction=0.05, parent="body"),
            PartHint("pectoral_fin", "extrude", "loft", count=2, typical_fraction=0.02, parent="body"),
        ],
    ))
    _add(ObjectTemplate(
        name="flatfish", superclass="fish",
        symmetry="bilateral", primary_axis="y", elongation="moderate",
        segmentation_strategy="projection",
        parts=[
            PartHint("body", "extrude", "loft", symmetry="bilateral", typical_fraction=0.8,
                     notes="Very flat body, nearly planar"),
            PartHint("tail_fin", "extrude", "loft", typical_fraction=0.1, parent="body"),
        ],
    ))
    _add(ObjectTemplate(
        name="ray", superclass="fish",
        symmetry="bilateral", primary_axis="y", elongation="moderate",
        segmentation_strategy="projection",
        parts=[
            PartHint("body", "extrude", "loft", symmetry="bilateral", typical_fraction=0.7,
                     notes="Wide flat diamond/disc body"),
            PartHint("tail", "cylinder", "sweep", typical_fraction=0.1, parent="body"),
            PartHint("wing", "extrude", "loft", count=2, typical_fraction=0.08, parent="body"),
        ],
    ))
    _add(ObjectTemplate(
        name="shark", superclass="fish",
        symmetry="bilateral", primary_axis="y", elongation="elongated",
        segmentation_strategy="skeleton",
        parts=[
            PartHint("body", "revolve", "loft", symmetry="bilateral", typical_fraction=0.65),
            PartHint("head", "cone", "loft", typical_fraction=0.1, parent="body"),
            PartHint("dorsal_fin", "extrude", "loft", typical_fraction=0.05, parent="body"),
            PartHint("tail_fin", "extrude", "loft", typical_fraction=0.1, parent="body"),
            PartHint("pectoral_fin", "extrude", "loft", count=2, typical_fraction=0.03, parent="body"),
        ],
    ))
    _add(ObjectTemplate(
        name="trout", superclass="fish",
        symmetry="bilateral", primary_axis="y", elongation="elongated",
        segmentation_strategy="skeleton",
        parts=[
            PartHint("body", "revolve", "loft", symmetry="bilateral", typical_fraction=0.7),
            PartHint("tail_fin", "extrude", "loft", typical_fraction=0.1, parent="body"),
            PartHint("dorsal_fin", "extrude", "loft", typical_fraction=0.05, parent="body"),
            PartHint("pectoral_fin", "extrude", "loft", count=2, typical_fraction=0.02, parent="body"),
        ],
    ))

    # ===================================================================
    # Superclass 3: flowers
    # ===================================================================
    _add(ObjectTemplate(
        name="orchid", superclass="flowers",
        symmetry="bilateral", primary_axis="z", elongation="moderate",
        segmentation_strategy="normal_cluster",
        parts=[
            PartHint("stem", "cylinder", "extrude", typical_fraction=0.15),
            PartHint("petal", "extrude", "loft", count=5, typical_fraction=0.12,
                     notes="Curved thin petals"),
            PartHint("center", "sphere", "revolve", typical_fraction=0.1),
        ],
    ))
    _add(ObjectTemplate(
        name="poppy", superclass="flowers",
        symmetry="radial", primary_axis="z", elongation="moderate",
        segmentation_strategy="normal_cluster",
        parts=[
            PartHint("stem", "cylinder", "extrude", typical_fraction=0.2),
            PartHint("petal", "extrude", "loft", count=4, typical_fraction=0.15,
                     symmetry="radial"),
            PartHint("center", "sphere", "revolve", typical_fraction=0.1),
        ],
    ))
    _add(ObjectTemplate(
        name="rose", superclass="flowers",
        symmetry="radial", primary_axis="z", elongation="moderate",
        segmentation_strategy="normal_cluster",
        parts=[
            PartHint("stem", "cylinder", "extrude", typical_fraction=0.15),
            PartHint("outer_petals", "extrude", "loft", count=8, typical_fraction=0.08,
                     notes="Layered spiral petals"),
            PartHint("inner_petals", "extrude", "loft", count=5, typical_fraction=0.06),
            PartHint("center", "sphere", "revolve", typical_fraction=0.05),
        ],
    ))
    _add(ObjectTemplate(
        name="sunflower", superclass="flowers",
        symmetry="radial", primary_axis="z", elongation="elongated",
        segmentation_strategy="normal_cluster",
        parts=[
            PartHint("stem", "cylinder", "extrude", typical_fraction=0.3),
            PartHint("petal", "extrude", "loft", count=20, typical_fraction=0.02,
                     symmetry="radial"),
            PartHint("disc", "cylinder", "revolve", typical_fraction=0.15),
        ],
    ))
    _add(ObjectTemplate(
        name="tulip", superclass="flowers",
        symmetry="radial", primary_axis="z", elongation="elongated",
        segmentation_strategy="normal_cluster",
        parts=[
            PartHint("stem", "cylinder", "extrude", typical_fraction=0.3),
            PartHint("petal", "extrude", "loft", count=6, typical_fraction=0.1,
                     notes="Cup-shaped petals"),
            PartHint("leaf", "extrude", "loft", count=2, typical_fraction=0.05),
        ],
    ))

    # ===================================================================
    # Superclass 4: food_containers
    # ===================================================================
    _add(ObjectTemplate(
        name="bottle", superclass="food_containers",
        symmetry="axial", primary_axis="z", elongation="elongated",
        segmentation_strategy="sdf",
        parts=[
            PartHint("body", "revolve", "revolve", symmetry="axial", typical_fraction=0.7),
            PartHint("neck", "cylinder", "revolve", typical_fraction=0.15, parent="body"),
            PartHint("cap", "cylinder", "revolve", typical_fraction=0.05, parent="neck"),
        ],
    ))
    _add(ObjectTemplate(
        name="bowl", superclass="food_containers",
        symmetry="axial", primary_axis="z", elongation="compact",
        segmentation_strategy="sdf",
        parts=[
            PartHint("body", "revolve", "revolve", symmetry="axial", typical_fraction=0.95),
        ],
    ))
    _add(ObjectTemplate(
        name="can", superclass="food_containers",
        symmetry="axial", primary_axis="z", elongation="moderate",
        segmentation_strategy="sdf",
        parts=[
            PartHint("body", "cylinder", "revolve", symmetry="axial", typical_fraction=0.9),
            PartHint("rim", "cylinder", "revolve", typical_fraction=0.05, parent="body"),
        ],
    ))
    _add(ObjectTemplate(
        name="cup", superclass="food_containers",
        symmetry="axial", primary_axis="z", elongation="moderate",
        segmentation_strategy="sdf",
        parts=[
            PartHint("body", "revolve", "revolve", symmetry="axial", typical_fraction=0.8),
            PartHint("handle", "sweep", "sweep", typical_fraction=0.1, parent="body"),
        ],
    ))
    _add(ObjectTemplate(
        name="plate", superclass="food_containers",
        symmetry="axial", primary_axis="z", elongation="compact",
        segmentation_strategy="sdf",
        parts=[
            PartHint("body", "revolve", "revolve", symmetry="axial", typical_fraction=0.95),
        ],
    ))

    # ===================================================================
    # Superclass 5: fruit_and_vegetables
    # ===================================================================
    _add(ObjectTemplate(
        name="apple", superclass="fruit_and_vegetables",
        symmetry="axial", primary_axis="z", elongation="compact",
        segmentation_strategy="sdf",
        parts=[
            PartHint("body", "revolve", "revolve", symmetry="axial", typical_fraction=0.9),
            PartHint("stem", "cylinder", "extrude", typical_fraction=0.02, parent="body"),
        ],
    ))
    _add(ObjectTemplate(
        name="mushroom", superclass="fruit_and_vegetables",
        symmetry="axial", primary_axis="z", elongation="moderate",
        segmentation_strategy="sdf",
        parts=[
            PartHint("cap", "revolve", "revolve", symmetry="axial", typical_fraction=0.6),
            PartHint("stem", "cylinder", "revolve", typical_fraction=0.35, parent="cap"),
        ],
    ))
    _add(ObjectTemplate(
        name="orange", superclass="fruit_and_vegetables",
        symmetry="axial", primary_axis="z", elongation="compact",
        segmentation_strategy="sdf",
        parts=[
            PartHint("body", "sphere", "revolve", symmetry="axial", typical_fraction=0.95),
        ],
    ))
    _add(ObjectTemplate(
        name="pear", superclass="fruit_and_vegetables",
        symmetry="axial", primary_axis="z", elongation="moderate",
        segmentation_strategy="sdf",
        parts=[
            PartHint("body", "revolve", "revolve", symmetry="axial", typical_fraction=0.9),
            PartHint("stem", "cylinder", "extrude", typical_fraction=0.02, parent="body"),
        ],
    ))
    _add(ObjectTemplate(
        name="sweet_pepper", superclass="fruit_and_vegetables",
        symmetry="axial", primary_axis="z", elongation="moderate",
        segmentation_strategy="sdf",
        parts=[
            PartHint("body", "revolve", "revolve", symmetry="axial", typical_fraction=0.85),
            PartHint("stem", "cylinder", "extrude", typical_fraction=0.05, parent="body"),
        ],
    ))

    # ===================================================================
    # Superclass 6: household_electrical_devices
    # ===================================================================
    _add(ObjectTemplate(
        name="clock", superclass="household_electrical_devices",
        symmetry="axial", primary_axis="z", elongation="compact",
        segmentation_strategy="sdf",
        parts=[
            PartHint("face", "cylinder", "revolve", symmetry="axial", typical_fraction=0.8),
            PartHint("frame", "revolve", "revolve", typical_fraction=0.15),
        ],
    ))
    _add(ObjectTemplate(
        name="keyboard", superclass="household_electrical_devices",
        symmetry="bilateral", primary_axis="y", elongation="moderate",
        segmentation_strategy="convexity",
        parts=[
            PartHint("base", "box", "extrude", typical_fraction=0.7),
            PartHint("key", "box", "extrude", count=80, typical_fraction=0.003),
        ],
    ))
    _add(ObjectTemplate(
        name="lamp", superclass="household_electrical_devices",
        symmetry="axial", primary_axis="z", elongation="elongated",
        segmentation_strategy="sdf",
        parts=[
            PartHint("base", "cylinder", "revolve", typical_fraction=0.2),
            PartHint("stem", "cylinder", "extrude", typical_fraction=0.3, parent="base"),
            PartHint("shade", "cone", "revolve", typical_fraction=0.4, parent="stem"),
        ],
    ))
    _add(ObjectTemplate(
        name="telephone", superclass="household_electrical_devices",
        symmetry="bilateral", primary_axis="y", elongation="moderate",
        segmentation_strategy="convexity",
        parts=[
            PartHint("body", "box", "extrude", typical_fraction=0.5),
            PartHint("handset", "sweep", "sweep", typical_fraction=0.3, parent="body"),
        ],
    ))
    _add(ObjectTemplate(
        name="television", superclass="household_electrical_devices",
        symmetry="bilateral", primary_axis="z", elongation="moderate",
        segmentation_strategy="convexity",
        parts=[
            PartHint("screen", "box", "extrude", typical_fraction=0.7),
            PartHint("stand", "cylinder", "extrude", typical_fraction=0.15),
            PartHint("base", "box", "extrude", typical_fraction=0.1),
        ],
    ))

    # ===================================================================
    # Superclass 7: household_furniture
    # ===================================================================
    _add(ObjectTemplate(
        name="bed", superclass="household_furniture",
        symmetry="bilateral", primary_axis="y", elongation="moderate",
        segmentation_strategy="convexity",
        parts=[
            PartHint("mattress", "box", "extrude", typical_fraction=0.5),
            PartHint("headboard", "box", "extrude", typical_fraction=0.15),
            PartHint("leg", "cylinder", "extrude", count=4, typical_fraction=0.03),
            PartHint("frame", "box", "extrude", typical_fraction=0.15),
        ],
    ))
    _add(ObjectTemplate(
        name="chair", superclass="household_furniture",
        symmetry="bilateral", primary_axis="z", elongation="moderate",
        segmentation_strategy="convexity",
        parts=[
            PartHint("seat", "box", "extrude", typical_fraction=0.25),
            PartHint("backrest", "box", "extrude", typical_fraction=0.2, parent="seat"),
            PartHint("leg", "cylinder", "extrude", count=4, typical_fraction=0.08),
            PartHint("armrest", "box", "extrude", count=2, typical_fraction=0.05),
        ],
    ))
    _add(ObjectTemplate(
        name="couch", superclass="household_furniture",
        symmetry="bilateral", primary_axis="y", elongation="moderate",
        segmentation_strategy="convexity",
        parts=[
            PartHint("seat", "box", "extrude", typical_fraction=0.3),
            PartHint("backrest", "box", "extrude", typical_fraction=0.25, parent="seat"),
            PartHint("armrest", "box", "extrude", count=2, typical_fraction=0.1, parent="seat"),
            PartHint("leg", "cylinder", "extrude", count=4, typical_fraction=0.02),
        ],
    ))
    _add(ObjectTemplate(
        name="table", superclass="household_furniture",
        symmetry="bilateral", primary_axis="z", elongation="moderate",
        segmentation_strategy="convexity",
        parts=[
            PartHint("top", "box", "extrude", typical_fraction=0.4),
            PartHint("leg", "cylinder", "extrude", count=4, typical_fraction=0.1),
        ],
    ))
    _add(ObjectTemplate(
        name="wardrobe", superclass="household_furniture",
        symmetry="bilateral", primary_axis="z", elongation="moderate",
        segmentation_strategy="convexity",
        parts=[
            PartHint("body", "box", "extrude", typical_fraction=0.85),
            PartHint("door", "box", "extrude", count=2, typical_fraction=0.05),
        ],
    ))

    # ===================================================================
    # Superclass 8: insects
    # ===================================================================
    _add(ObjectTemplate(
        name="bee", superclass="insects",
        symmetry="bilateral", primary_axis="y", elongation="moderate",
        segmentation_strategy="skeleton",
        parts=[
            PartHint("thorax", "sphere", "revolve", typical_fraction=0.3),
            PartHint("abdomen", "revolve", "revolve", typical_fraction=0.35, parent="thorax"),
            PartHint("head", "sphere", "revolve", typical_fraction=0.1, parent="thorax"),
            PartHint("wing", "extrude", "loft", count=2, typical_fraction=0.08, parent="thorax"),
            PartHint("leg", "cylinder", "sweep", count=6, typical_fraction=0.01, parent="thorax"),
        ],
    ))
    _add(ObjectTemplate(
        name="beetle", superclass="insects",
        symmetry="bilateral", primary_axis="y", elongation="moderate",
        segmentation_strategy="skeleton",
        parts=[
            PartHint("body", "revolve", "revolve", symmetry="bilateral", typical_fraction=0.6),
            PartHint("head", "sphere", "revolve", typical_fraction=0.15, parent="body"),
            PartHint("leg", "cylinder", "sweep", count=6, typical_fraction=0.02, parent="body"),
        ],
    ))
    _add(ObjectTemplate(
        name="butterfly", superclass="insects",
        symmetry="bilateral", primary_axis="y", elongation="moderate",
        segmentation_strategy="projection",
        parts=[
            PartHint("body", "cylinder", "revolve", typical_fraction=0.1),
            PartHint("wing", "extrude", "loft", count=4, typical_fraction=0.2,
                     notes="Thin flat wings — project and extrude"),
        ],
    ))
    _add(ObjectTemplate(
        name="caterpillar", superclass="insects",
        symmetry="bilateral", primary_axis="y", elongation="very_elongated",
        segmentation_strategy="skeleton",
        parts=[
            PartHint("segment", "sphere", "loft", count=10, typical_fraction=0.08,
                     notes="Chain of near-spherical segments along a curve"),
            PartHint("head", "sphere", "revolve", typical_fraction=0.1),
        ],
    ))
    _add(ObjectTemplate(
        name="cockroach", superclass="insects",
        symmetry="bilateral", primary_axis="y", elongation="moderate",
        segmentation_strategy="skeleton",
        parts=[
            PartHint("body", "revolve", "revolve", symmetry="bilateral", typical_fraction=0.5),
            PartHint("head", "sphere", "revolve", typical_fraction=0.1, parent="body"),
            PartHint("leg", "cylinder", "sweep", count=6, typical_fraction=0.02, parent="body"),
            PartHint("antenna", "cylinder", "sweep", count=2, typical_fraction=0.01, parent="head"),
        ],
    ))

    # ===================================================================
    # Superclass 9: large_carnivores
    # ===================================================================
    _quadruped_parts = [
        PartHint("torso", "revolve", "loft", symmetry="bilateral", typical_fraction=0.45),
        PartHint("head", "sphere", "loft", typical_fraction=0.1, parent="torso"),
        PartHint("neck", "cylinder", "loft", typical_fraction=0.05, parent="torso"),
        PartHint("front_leg", "cylinder", "loft", count=2, typical_fraction=0.07, parent="torso"),
        PartHint("rear_leg", "cylinder", "loft", count=2, typical_fraction=0.08, parent="torso"),
        PartHint("tail", "cylinder", "sweep", typical_fraction=0.05, parent="torso"),
    ]
    for name in ["bear", "leopard", "lion", "tiger", "wolf"]:
        _add(ObjectTemplate(
            name=name, superclass="large_carnivores",
            symmetry="bilateral", primary_axis="y", elongation="moderate",
            segmentation_strategy="skeleton",
            parts=list(_quadruped_parts),
        ))

    # ===================================================================
    # Superclass 10: large_man-made_outdoor_things
    # ===================================================================
    _add(ObjectTemplate(
        name="bridge", superclass="large_man-made_outdoor_things",
        symmetry="bilateral", primary_axis="y", elongation="very_elongated",
        segmentation_strategy="convexity",
        parts=[
            PartHint("deck", "box", "extrude", typical_fraction=0.4),
            PartHint("arch", "sweep", "sweep", count=2, typical_fraction=0.15),
            PartHint("pillar", "cylinder", "extrude", count=4, typical_fraction=0.08),
            PartHint("railing", "box", "extrude", count=2, typical_fraction=0.05),
        ],
    ))
    _add(ObjectTemplate(
        name="castle", superclass="large_man-made_outdoor_things",
        symmetry="bilateral", primary_axis="z", elongation="moderate",
        segmentation_strategy="convexity",
        parts=[
            PartHint("wall", "box", "extrude", count=4, typical_fraction=0.15),
            PartHint("tower", "cylinder", "extrude", count=4, typical_fraction=0.1),
            PartHint("turret_cap", "cone", "revolve", count=4, typical_fraction=0.03),
            PartHint("gate", "box", "boolean_cut", typical_fraction=0.05),
        ],
    ))
    _add(ObjectTemplate(
        name="house", superclass="large_man-made_outdoor_things",
        symmetry="bilateral", primary_axis="z", elongation="moderate",
        segmentation_strategy="convexity",
        parts=[
            PartHint("walls", "box", "extrude", typical_fraction=0.5),
            PartHint("roof", "extrude", "extrude", typical_fraction=0.25,
                     notes="Triangular cross-section extruded along ridge"),
            PartHint("window", "box", "boolean_cut", count=4, typical_fraction=0.02),
            PartHint("door", "box", "boolean_cut", typical_fraction=0.03),
        ],
    ))
    _add(ObjectTemplate(
        name="road", superclass="large_man-made_outdoor_things",
        symmetry="bilateral", primary_axis="y", elongation="very_elongated",
        segmentation_strategy="projection",
        parts=[
            PartHint("surface", "box", "extrude", typical_fraction=0.9),
        ],
    ))
    _add(ObjectTemplate(
        name="skyscraper", superclass="large_man-made_outdoor_things",
        symmetry="bilateral", primary_axis="z", elongation="very_elongated",
        segmentation_strategy="convexity",
        parts=[
            PartHint("main_tower", "box", "extrude", typical_fraction=0.7),
            PartHint("setback", "box", "extrude", count=2, typical_fraction=0.1),
            PartHint("antenna", "cylinder", "extrude", typical_fraction=0.02),
        ],
    ))

    # ===================================================================
    # Superclass 11: large_natural_outdoor_scenes
    # ===================================================================
    _add(ObjectTemplate(
        name="cloud", superclass="large_natural_outdoor_scenes",
        symmetry=None, primary_axis="y", elongation="moderate",
        segmentation_strategy="convexity",
        parts=[
            PartHint("lobe", "sphere", "freeform", count=5, typical_fraction=0.2),
        ],
    ))
    _add(ObjectTemplate(
        name="forest", superclass="large_natural_outdoor_scenes",
        symmetry=None, primary_axis="z", elongation="moderate",
        segmentation_strategy="skeleton",
        parts=[
            PartHint("trunk", "cylinder", "loft", count=10, typical_fraction=0.03),
            PartHint("canopy", "sphere", "freeform", count=10, typical_fraction=0.05),
            PartHint("ground", "box", "extrude", typical_fraction=0.1),
        ],
    ))
    _add(ObjectTemplate(
        name="mountain", superclass="large_natural_outdoor_scenes",
        symmetry=None, primary_axis="z", elongation="moderate",
        segmentation_strategy="convexity",
        parts=[
            PartHint("peak", "cone", "revolve", typical_fraction=0.4),
            PartHint("base", "cone", "revolve", typical_fraction=0.5),
        ],
    ))
    _add(ObjectTemplate(
        name="plain", superclass="large_natural_outdoor_scenes",
        symmetry=None, primary_axis="y", elongation="very_elongated",
        segmentation_strategy="projection",
        parts=[
            PartHint("surface", "box", "extrude", typical_fraction=0.95),
        ],
    ))
    _add(ObjectTemplate(
        name="sea", superclass="large_natural_outdoor_scenes",
        symmetry=None, primary_axis="y", elongation="very_elongated",
        segmentation_strategy="projection",
        parts=[
            PartHint("surface", "extrude", "freeform", typical_fraction=0.95),
        ],
    ))

    # ===================================================================
    # Superclass 12: large_omnivores_and_herbivores
    # ===================================================================
    for name in ["camel", "cattle", "elephant", "hippopotamus"]:
        parts = list(_quadruped_parts)
        if name == "elephant":
            parts.append(PartHint("trunk", "cylinder", "sweep", typical_fraction=0.08, parent="head"))
            parts.append(PartHint("ear", "extrude", "loft", count=2, typical_fraction=0.03, parent="head"))
            parts.append(PartHint("tusk", "cone", "revolve", count=2, typical_fraction=0.02, parent="head"))
        elif name == "camel":
            parts.append(PartHint("hump", "sphere", "revolve", count=2, typical_fraction=0.05, parent="torso"))
        _add(ObjectTemplate(
            name=name, superclass="large_omnivores_and_herbivores",
            symmetry="bilateral", primary_axis="y", elongation="moderate",
            segmentation_strategy="skeleton",
            parts=parts,
        ))
    _add(ObjectTemplate(
        name="chimpanzee", superclass="large_omnivores_and_herbivores",
        symmetry="bilateral", primary_axis="z", elongation="moderate",
        segmentation_strategy="skeleton",
        parts=[
            PartHint("torso", "revolve", "loft", symmetry="bilateral", typical_fraction=0.35),
            PartHint("head", "sphere", "revolve", typical_fraction=0.1, parent="torso"),
            PartHint("arm", "cylinder", "loft", count=2, typical_fraction=0.1, parent="torso"),
            PartHint("leg", "cylinder", "loft", count=2, typical_fraction=0.1, parent="torso"),
        ],
    ))

    # ===================================================================
    # Superclass 13: medium_mammals
    # ===================================================================
    for name in ["fox", "porcupine", "possum", "raccoon", "skunk"]:
        _add(ObjectTemplate(
            name=name, superclass="medium_mammals",
            symmetry="bilateral", primary_axis="y", elongation="moderate",
            segmentation_strategy="skeleton",
            parts=list(_quadruped_parts),
        ))

    # ===================================================================
    # Superclass 14: non-insect_invertebrates
    # ===================================================================
    _add(ObjectTemplate(
        name="crab", superclass="non-insect_invertebrates",
        symmetry="bilateral", primary_axis="y", elongation="compact",
        segmentation_strategy="convexity",
        parts=[
            PartHint("body", "revolve", "loft", symmetry="bilateral", typical_fraction=0.4),
            PartHint("claw", "sweep", "sweep", count=2, typical_fraction=0.15),
            PartHint("leg", "cylinder", "sweep", count=8, typical_fraction=0.03),
        ],
    ))
    _add(ObjectTemplate(
        name="lobster", superclass="non-insect_invertebrates",
        symmetry="bilateral", primary_axis="y", elongation="elongated",
        segmentation_strategy="skeleton",
        parts=[
            PartHint("body", "revolve", "loft", typical_fraction=0.3),
            PartHint("tail", "revolve", "loft", typical_fraction=0.2, parent="body"),
            PartHint("claw", "sweep", "sweep", count=2, typical_fraction=0.12),
            PartHint("leg", "cylinder", "sweep", count=8, typical_fraction=0.02),
        ],
    ))
    _add(ObjectTemplate(
        name="snail", superclass="non-insect_invertebrates",
        symmetry=None, primary_axis="z", elongation="compact",
        segmentation_strategy="sdf",
        parts=[
            PartHint("shell", "revolve", "revolve", typical_fraction=0.6,
                     notes="Spiral shell — logarithmic spiral revolve"),
            PartHint("body", "freeform", "freeform", typical_fraction=0.35),
        ],
    ))
    _add(ObjectTemplate(
        name="spider", superclass="non-insect_invertebrates",
        symmetry="bilateral", primary_axis="y", elongation="moderate",
        segmentation_strategy="skeleton",
        parts=[
            PartHint("body", "sphere", "revolve", typical_fraction=0.25),
            PartHint("abdomen", "sphere", "revolve", typical_fraction=0.3, parent="body"),
            PartHint("leg", "cylinder", "sweep", count=8, typical_fraction=0.04),
        ],
    ))
    _add(ObjectTemplate(
        name="worm", superclass="non-insect_invertebrates",
        symmetry="axial", primary_axis="y", elongation="very_elongated",
        segmentation_strategy="skeleton",
        parts=[
            PartHint("body", "cylinder", "sweep", typical_fraction=0.95,
                     notes="Tapered tube along curved path"),
        ],
    ))

    # ===================================================================
    # Superclass 15: people
    # ===================================================================
    _biped_parts = [
        PartHint("torso", "revolve", "loft", symmetry="bilateral", typical_fraction=0.3),
        PartHint("head", "sphere", "revolve", typical_fraction=0.08, parent="torso"),
        PartHint("neck", "cylinder", "loft", typical_fraction=0.02, parent="torso"),
        PartHint("upper_arm", "cylinder", "loft", count=2, typical_fraction=0.05, parent="torso"),
        PartHint("forearm", "cylinder", "loft", count=2, typical_fraction=0.04, parent="upper_arm"),
        PartHint("hand", "box", "freeform", count=2, typical_fraction=0.02, parent="forearm"),
        PartHint("thigh", "cylinder", "loft", count=2, typical_fraction=0.08, parent="torso"),
        PartHint("shin", "cylinder", "loft", count=2, typical_fraction=0.06, parent="thigh"),
        PartHint("foot", "box", "loft", count=2, typical_fraction=0.02, parent="shin"),
    ]
    for name in ["baby", "boy", "girl", "man", "woman"]:
        _add(ObjectTemplate(
            name=name, superclass="people",
            symmetry="bilateral", primary_axis="z", elongation="elongated",
            segmentation_strategy="skeleton",
            parts=list(_biped_parts),
            aliases=["humanoid"] if name == "man" else [],
        ))

    # ===================================================================
    # Superclass 16: reptiles
    # ===================================================================
    _add(ObjectTemplate(
        name="crocodile", superclass="reptiles",
        symmetry="bilateral", primary_axis="y", elongation="very_elongated",
        segmentation_strategy="skeleton",
        parts=[
            PartHint("body", "revolve", "loft", symmetry="bilateral", typical_fraction=0.4),
            PartHint("head", "box", "loft", typical_fraction=0.15, parent="body"),
            PartHint("tail", "cylinder", "loft", typical_fraction=0.2, parent="body"),
            PartHint("leg", "cylinder", "loft", count=4, typical_fraction=0.04, parent="body"),
        ],
    ))
    _add(ObjectTemplate(
        name="dinosaur", superclass="reptiles",
        symmetry="bilateral", primary_axis="z", elongation="elongated",
        segmentation_strategy="skeleton",
        parts=[
            PartHint("torso", "revolve", "loft", symmetry="bilateral", typical_fraction=0.35),
            PartHint("head", "sphere", "loft", typical_fraction=0.1, parent="torso"),
            PartHint("neck", "cylinder", "loft", typical_fraction=0.1, parent="torso"),
            PartHint("tail", "cylinder", "loft", typical_fraction=0.15, parent="torso"),
            PartHint("leg", "cylinder", "loft", count=2, typical_fraction=0.1, parent="torso"),
        ],
    ))
    _add(ObjectTemplate(
        name="lizard", superclass="reptiles",
        symmetry="bilateral", primary_axis="y", elongation="elongated",
        segmentation_strategy="skeleton",
        parts=list(_quadruped_parts),
    ))
    _add(ObjectTemplate(
        name="snake", superclass="reptiles",
        symmetry="axial", primary_axis="y", elongation="very_elongated",
        segmentation_strategy="skeleton",
        parts=[
            PartHint("body", "cylinder", "sweep", typical_fraction=0.85,
                     notes="Tapered tube along sinuous path"),
            PartHint("head", "sphere", "loft", typical_fraction=0.1, parent="body"),
        ],
    ))
    _add(ObjectTemplate(
        name="turtle", superclass="reptiles",
        symmetry="bilateral", primary_axis="y", elongation="compact",
        segmentation_strategy="sdf",
        parts=[
            PartHint("shell", "revolve", "revolve", symmetry="axial", typical_fraction=0.6),
            PartHint("plastron", "revolve", "revolve", typical_fraction=0.15),
            PartHint("head", "sphere", "loft", typical_fraction=0.08, parent="shell"),
            PartHint("leg", "cylinder", "loft", count=4, typical_fraction=0.03, parent="shell"),
        ],
    ))

    # ===================================================================
    # Superclass 17: small_mammals
    # ===================================================================
    for name in ["hamster", "mouse", "rabbit", "shrew", "squirrel"]:
        parts = list(_quadruped_parts)
        if name == "rabbit":
            parts.append(PartHint("ear", "cylinder", "loft", count=2,
                                  typical_fraction=0.04, parent="head"))
        if name == "squirrel":
            # bushy tail override
            for p in parts:
                if p.name == "tail":
                    p.typical_fraction = 0.15
                    p.geometry = "revolve"
        _add(ObjectTemplate(
            name=name, superclass="small_mammals",
            symmetry="bilateral", primary_axis="y", elongation="moderate",
            segmentation_strategy="skeleton",
            parts=parts,
        ))

    # ===================================================================
    # Superclass 18: trees
    # ===================================================================
    _tree_parts = [
        PartHint("trunk", "cylinder", "loft", typical_fraction=0.3,
                 notes="Tapered cylinder, thicker at base"),
        PartHint("branch", "cylinder", "sweep", count=8, typical_fraction=0.05,
                 parent="trunk", notes="Branches taper away from trunk"),
    ]
    _add(ObjectTemplate(
        name="maple_tree", superclass="trees", aliases=["maple"],
        symmetry="axial", primary_axis="z", elongation="elongated",
        segmentation_strategy="skeleton",
        parts=_tree_parts + [
            PartHint("canopy", "sphere", "freeform", typical_fraction=0.3),
        ],
    ))
    _add(ObjectTemplate(
        name="oak_tree", superclass="trees", aliases=["oak"],
        symmetry="axial", primary_axis="z", elongation="moderate",
        segmentation_strategy="skeleton",
        parts=_tree_parts + [
            PartHint("canopy", "sphere", "freeform", typical_fraction=0.35),
        ],
    ))
    _add(ObjectTemplate(
        name="palm_tree", superclass="trees", aliases=["palm"],
        symmetry="axial", primary_axis="z", elongation="very_elongated",
        segmentation_strategy="skeleton",
        parts=[
            PartHint("trunk", "cylinder", "loft", typical_fraction=0.6,
                     notes="Tall narrow trunk, slight taper"),
            PartHint("frond", "extrude", "sweep", count=8, typical_fraction=0.04,
                     parent="trunk", notes="Long thin fronds radiating from top"),
        ],
    ))
    _add(ObjectTemplate(
        name="pine_tree", superclass="trees", aliases=["pine"],
        symmetry="axial", primary_axis="z", elongation="very_elongated",
        segmentation_strategy="skeleton",
        parts=[
            PartHint("trunk", "cylinder", "loft", typical_fraction=0.25),
            PartHint("canopy_tier", "cone", "revolve", count=5, typical_fraction=0.12,
                     parent="trunk", notes="Stacked conical tiers"),
        ],
    ))
    _add(ObjectTemplate(
        name="willow_tree", superclass="trees", aliases=["willow"],
        symmetry="axial", primary_axis="z", elongation="moderate",
        segmentation_strategy="skeleton",
        parts=_tree_parts + [
            PartHint("canopy", "sphere", "freeform", typical_fraction=0.25),
            PartHint("drape", "cylinder", "sweep", count=20, typical_fraction=0.01,
                     parent="branch", notes="Thin hanging tendrils"),
        ],
    ))
    # Extra: dead_tree (not CIFAR but important for this project)
    _add(ObjectTemplate(
        name="dead_tree", superclass="trees", aliases=["bare_tree", "dead_oak"],
        symmetry=None, primary_axis="z", elongation="elongated",
        segmentation_strategy="skeleton",
        notes="No foliage. Pure trunk + branching structure.",
        parts=[
            PartHint("trunk", "cylinder", "loft", typical_fraction=0.35,
                     notes="Tapered, may be gnarled"),
            PartHint("primary_branch", "cylinder", "sweep", count=4, typical_fraction=0.1,
                     parent="trunk", notes="Thick branches tapering away from trunk"),
            PartHint("secondary_branch", "cylinder", "sweep", count=12, typical_fraction=0.02,
                     parent="primary_branch", notes="Thinner sub-branches"),
        ],
    ))

    # ===================================================================
    # Superclass 19: vehicles_1
    # ===================================================================
    _add(ObjectTemplate(
        name="bicycle", superclass="vehicles_1",
        symmetry="bilateral", primary_axis="y", elongation="moderate",
        segmentation_strategy="skeleton",
        parts=[
            PartHint("frame", "cylinder", "sweep", typical_fraction=0.2,
                     notes="Tubular frame — multiple sweep paths"),
            PartHint("wheel", "revolve", "revolve", count=2, typical_fraction=0.2,
                     notes="Torus-like rim + spokes"),
            PartHint("handlebar", "cylinder", "sweep", typical_fraction=0.05),
            PartHint("seat", "extrude", "loft", typical_fraction=0.05),
            PartHint("pedal", "box", "extrude", count=2, typical_fraction=0.02),
        ],
    ))
    _add(ObjectTemplate(
        name="bus", superclass="vehicles_1",
        symmetry="bilateral", primary_axis="y", elongation="elongated",
        segmentation_strategy="convexity",
        parts=[
            PartHint("body", "box", "extrude", typical_fraction=0.7),
            PartHint("wheel", "cylinder", "revolve", count=4, typical_fraction=0.04),
            PartHint("window", "box", "boolean_cut", count=10, typical_fraction=0.01),
            PartHint("wheel_arch", "cylinder", "boolean_cut", count=4, typical_fraction=0.02),
        ],
    ))
    _add(ObjectTemplate(
        name="motorcycle", superclass="vehicles_1",
        symmetry="bilateral", primary_axis="y", elongation="moderate",
        segmentation_strategy="skeleton",
        parts=[
            PartHint("frame", "cylinder", "sweep", typical_fraction=0.15),
            PartHint("wheel", "revolve", "revolve", count=2, typical_fraction=0.15),
            PartHint("engine", "box", "loft", typical_fraction=0.15),
            PartHint("tank", "revolve", "loft", typical_fraction=0.1),
            PartHint("seat", "extrude", "loft", typical_fraction=0.08),
            PartHint("exhaust", "cylinder", "sweep", typical_fraction=0.05),
        ],
    ))
    _add(ObjectTemplate(
        name="pickup_truck", superclass="vehicles_1",
        symmetry="bilateral", primary_axis="y", elongation="elongated",
        segmentation_strategy="convexity",
        parts=[
            PartHint("cab", "box", "loft", typical_fraction=0.3),
            PartHint("bed", "box", "extrude", typical_fraction=0.25),
            PartHint("hood", "box", "loft", typical_fraction=0.15),
            PartHint("wheel", "cylinder", "revolve", count=4, typical_fraction=0.04),
            PartHint("wheel_arch", "cylinder", "boolean_cut", count=4, typical_fraction=0.02),
        ],
    ))
    _add(ObjectTemplate(
        name="train", superclass="vehicles_1",
        symmetry="bilateral", primary_axis="y", elongation="very_elongated",
        segmentation_strategy="convexity",
        parts=[
            PartHint("body", "box", "extrude", typical_fraction=0.6),
            PartHint("roof", "revolve", "loft", typical_fraction=0.1),
            PartHint("wheel", "cylinder", "revolve", count=8, typical_fraction=0.02),
            PartHint("coupling", "cylinder", "extrude", count=2, typical_fraction=0.02),
        ],
    ))

    # ===================================================================
    # Superclass 20: vehicles_2
    # ===================================================================
    _car_body_variants = {
        "sedan": "Smooth tapered body, 3-box profile",
        "hatchback": "Compact, sloped rear",
        "coupe": "Low profile, 2-door",
        "convertible": "Open top, low windshield",
        "suv": "Tall boxy body",
        "wagon": "Extended rear, flat roof",
        "minivan": "Tall rounded body",
        "sports_car": "Very low, wide, aggressive profile",
        "pickup": "Separate cab + bed",
        "crossover": "Between sedan and SUV",
    }
    _add(ObjectTemplate(
        name="lawn_mower", superclass="vehicles_2",
        symmetry="bilateral", primary_axis="y", elongation="moderate",
        segmentation_strategy="convexity",
        parts=[
            PartHint("deck", "box", "extrude", typical_fraction=0.4),
            PartHint("handle", "cylinder", "sweep", typical_fraction=0.2),
            PartHint("wheel", "cylinder", "revolve", count=4, typical_fraction=0.06),
            PartHint("engine", "box", "extrude", typical_fraction=0.15),
        ],
    ))
    _add(ObjectTemplate(
        name="rocket", superclass="vehicles_2",
        symmetry="axial", primary_axis="z", elongation="very_elongated",
        segmentation_strategy="sdf",
        parts=[
            PartHint("body", "cylinder", "revolve", symmetry="axial", typical_fraction=0.6),
            PartHint("nose_cone", "cone", "revolve", typical_fraction=0.1, parent="body"),
            PartHint("fin", "extrude", "loft", count=4, typical_fraction=0.04, parent="body"),
            PartHint("nozzle", "cone", "revolve", typical_fraction=0.08, parent="body"),
        ],
    ))
    _add(ObjectTemplate(
        name="streetcar", superclass="vehicles_2",
        symmetry="bilateral", primary_axis="y", elongation="very_elongated",
        segmentation_strategy="convexity",
        parts=[
            PartHint("body", "box", "extrude", typical_fraction=0.65),
            PartHint("roof", "revolve", "loft", typical_fraction=0.1),
            PartHint("wheel", "cylinder", "revolve", count=8, typical_fraction=0.02),
            PartHint("pantograph", "cylinder", "sweep", typical_fraction=0.03),
        ],
    ))
    _add(ObjectTemplate(
        name="tank", superclass="vehicles_2",
        symmetry="bilateral", primary_axis="y", elongation="moderate",
        segmentation_strategy="convexity",
        parts=[
            PartHint("hull", "box", "extrude", typical_fraction=0.4),
            PartHint("turret", "cylinder", "revolve", typical_fraction=0.15, parent="hull"),
            PartHint("barrel", "cylinder", "extrude", typical_fraction=0.08, parent="turret"),
            PartHint("track", "sweep", "sweep", count=2, typical_fraction=0.12),
        ],
    ))
    _add(ObjectTemplate(
        name="tractor", superclass="vehicles_2",
        symmetry="bilateral", primary_axis="y", elongation="moderate",
        segmentation_strategy="convexity",
        parts=[
            PartHint("body", "box", "extrude", typical_fraction=0.35),
            PartHint("hood", "box", "loft", typical_fraction=0.15),
            PartHint("rear_wheel", "cylinder", "revolve", count=2, typical_fraction=0.12),
            PartHint("front_wheel", "cylinder", "revolve", count=2, typical_fraction=0.06),
            PartHint("exhaust", "cylinder", "extrude", typical_fraction=0.02),
        ],
    ))

    return templates


# Build the global template dictionary
ALL_TEMPLATES = _make_templates()


# Additional structured data: car body variants
CAR_BODY_VARIANTS = [
    "sedan", "hatchback", "coupe", "convertible", "suv",
    "wagon", "minivan", "sports_car", "pickup", "crossover",
]

# CIFAR-100 superclass → fine class mapping (for reference)
CIFAR100_SUPERCLASSES = {
    "aquatic_mammals": ["beaver", "dolphin", "otter", "seal", "whale"],
    "fish": ["aquarium_fish", "flatfish", "ray", "shark", "trout"],
    "flowers": ["orchid", "poppy", "rose", "sunflower", "tulip"],
    "food_containers": ["bottle", "bowl", "can", "cup", "plate"],
    "fruit_and_vegetables": ["apple", "mushroom", "orange", "pear", "sweet_pepper"],
    "household_electrical_devices": ["clock", "keyboard", "lamp", "telephone", "television"],
    "household_furniture": ["bed", "chair", "couch", "table", "wardrobe"],
    "insects": ["bee", "beetle", "butterfly", "caterpillar", "cockroach"],
    "large_carnivores": ["bear", "leopard", "lion", "tiger", "wolf"],
    "large_man-made_outdoor_things": ["bridge", "castle", "house", "road", "skyscraper"],
    "large_natural_outdoor_scenes": ["cloud", "forest", "mountain", "plain", "sea"],
    "large_omnivores_and_herbivores": ["camel", "cattle", "chimpanzee", "elephant", "hippopotamus"],
    "medium_mammals": ["fox", "porcupine", "possum", "raccoon", "skunk"],
    "non-insect_invertebrates": ["crab", "lobster", "snail", "spider", "worm"],
    "people": ["baby", "boy", "girl", "man", "woman"],
    "reptiles": ["crocodile", "dinosaur", "lizard", "snake", "turtle"],
    "small_mammals": ["hamster", "mouse", "rabbit", "shrew", "squirrel"],
    "trees": ["maple_tree", "oak_tree", "palm_tree", "pine_tree", "willow_tree"],
    "vehicles_1": ["bicycle", "bus", "motorcycle", "pickup_truck", "train"],
    "vehicles_2": ["lawn_mower", "rocket", "streetcar", "tank", "tractor"],
}


# ---------------------------------------------------------------------------
# Template matching & lookup API
# ---------------------------------------------------------------------------

def get_template(name: str) -> Optional[ObjectTemplate]:
    """Get a template by name or alias. Returns None if not found."""
    return ALL_TEMPLATES.get(name)


def list_templates() -> List[str]:
    """Return all unique template names (excluding aliases)."""
    seen = set()
    names = []
    for t in ALL_TEMPLATES.values():
        if t.name not in seen:
            seen.add(t.name)
            names.append(t.name)
    return sorted(names)


def templates_for_superclass(superclass: str) -> List[ObjectTemplate]:
    """Return all templates belonging to a CIFAR-100 superclass."""
    seen = set()
    result = []
    for t in ALL_TEMPLATES.values():
        if t.superclass == superclass and t.name not in seen:
            seen.add(t.name)
            result.append(t)
    return result


def match_template(vertices, faces, top_k=3):
    """Score all templates against a mesh and return the best matches.

    Uses coarse geometric features (elongation, symmetry, part count heuristics)
    to rank templates.  Does NOT run full segmentation — this is a fast filter.

    Args:
        vertices: (N, 3) array
        faces:    (M, 3) array
        top_k:    number of top matches to return

    Returns:
        List of (template, score) tuples, highest score first.
    """
    import numpy as np

    v = np.asarray(vertices, dtype=np.float64)
    center = v.mean(axis=0)
    centered = v - center

    # PCA for elongation + symmetry estimation
    cov = np.cov(centered.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    eigvals = eigvals[::-1]
    eigvecs = eigvecs[:, ::-1]

    bbox = v.max(axis=0) - v.min(axis=0)
    bbox_sorted = np.sort(bbox)[::-1]

    # Elongation ratio
    elong_ratio = bbox_sorted[0] / max(bbox_sorted[1], 1e-12)

    # Circularity: ratio of two smallest extents
    circ = bbox_sorted[2] / max(bbox_sorted[1], 1e-12)

    # Primary axis
    primary_idx = np.argmax(bbox)
    axis_map = {0: "x", 1: "y", 2: "z"}
    primary_axis = axis_map[primary_idx]

    # Elongation category
    if elong_ratio < 1.3:
        elong_cat = "compact"
    elif elong_ratio < 2.0:
        elong_cat = "moderate"
    elif elong_ratio < 3.5:
        elong_cat = "elongated"
    else:
        elong_cat = "very_elongated"

    # Symmetry estimate
    if circ > 0.85:
        sym_est = "axial"
    elif circ > 0.5:
        sym_est = "bilateral"
    else:
        sym_est = None

    # Score each template
    scores = []
    unique_templates = {t.name: t for t in ALL_TEMPLATES.values()}

    for tpl in unique_templates.values():
        score = 0.0

        # Elongation match
        if tpl.elongation == elong_cat:
            score += 3.0
        elif abs(["compact", "moderate", "elongated", "very_elongated"].index(tpl.elongation)
                 - ["compact", "moderate", "elongated", "very_elongated"].index(elong_cat)) == 1:
            score += 1.5

        # Primary axis match
        if tpl.primary_axis == primary_axis:
            score += 2.0

        # Symmetry match
        if tpl.symmetry == sym_est:
            score += 2.0
        elif tpl.symmetry is not None and sym_est is not None:
            score += 0.5

        # Circularity bonus for axial-symmetric templates
        if tpl.symmetry == "axial" and circ > 0.8:
            score += 1.5

        scores.append((tpl, score))

    scores.sort(key=lambda x: -x[1])
    return scores[:top_k]
