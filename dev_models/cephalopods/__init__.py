"""Cephalopod model collection for CAD strategy development.

Mixed collection of:
- Downloaded real-world organic 3D models (octopus STL, OBJ)
- Programmatically generated parametric cephalopods (octopus, squid, cthulhu,
  jellyfish, kraken) with diverse tentacle configurations

These models stress-test mesh-to-CAD reconstruction for non-axisymmetric
organic shapes with branching structures and tentacles.

Downloaded models:
    camila_octopus      — camilabga/3d-printing (flexi octopus, high-poly)
    drlex_octopus       — DrLex0/print3d-funny-flexi-octopus
    armadillo           — alecjacobson/common-3d-test-models (OBJ)
    beast               — alecjacobson/common-3d-test-models (OBJ)
    ogre                — alecjacobson/common-3d-test-models (OBJ)
    cheburashka         — alecjacobson/common-3d-test-models (OBJ)
    suzanne             — alecjacobson/common-3d-test-models (Blender monkey, OBJ)

Generated models:
    octopus_basic       — 8-arm octopus with dome body
    octopus_many_arms   — 12-arm octopus variant
    octopus_curly       — heavily curled tentacles
    octopus_long        — long tentacles, smaller body
    octopus_stubby      — short thick tentacles, large body
    squid_basic         — squid with mantle, fins, 8 arms + 2 tentacles
    squid_long          — elongated squid variant
    squid_compact       — compact squid variant
    cthulhu_basic       — humanoid body with face tentacles and wings
    cthulhu_many_tentacles — extra face and arm tentacles
    jellyfish_basic     — bell dome with trailing tentacles
    jellyfish_many      — dense tentacle jellyfish
    kraken_basic        — massive body with 10 thick tentacles
    kraken_massive      — extra-large kraken variant
"""

import os
import numpy as np

_DIR = os.path.dirname(__file__)


def _read_stl(filepath):
    """Read an STL file."""
    import sys
    sys.path.insert(0, os.path.join(_DIR, "..", ".."))
    from meshxcad.stl_io import read_binary_stl
    return read_binary_stl(filepath)


def _read_obj(filepath):
    """Read a Wavefront OBJ file."""
    verts = []
    face_list = []
    with open(filepath, "r") as fh:
        for line in fh:
            parts = line.strip().split()
            if not parts:
                continue
            if parts[0] == "v" and len(parts) >= 4:
                verts.append([float(parts[1]), float(parts[2]), float(parts[3])])
            elif parts[0] == "f":
                idxs = [int(p.split("/")[0]) - 1 for p in parts[1:]]
                for i in range(1, len(idxs) - 1):
                    face_list.append([idxs[0], idxs[i], idxs[i + 1]])
    if not verts:
        raise ValueError(f"No vertices found in OBJ: {filepath}")
    return np.array(verts, dtype=np.float64), np.array(face_list, dtype=np.int64)


def _read_mesh(filepath):
    """Read STL or OBJ based on extension."""
    ext = os.path.splitext(filepath)[1].lower()
    if ext == ".obj":
        return _read_obj(filepath)
    return _read_stl(filepath)


def _normalize_mesh(vertices, faces, target_height=100.0):
    """Center, orient (tallest axis -> Z), and scale to target_height."""
    v = np.array(vertices, dtype=np.float64)
    f = np.array(faces, dtype=np.int64)

    center = (v.max(axis=0) + v.min(axis=0)) / 2
    v = v - center

    bbox = v.max(axis=0) - v.min(axis=0)
    tallest = int(np.argmax(bbox))
    if tallest != 2:
        v[:, [tallest, 2]] = v[:, [2, tallest]]

    bbox = v.max(axis=0) - v.min(axis=0)
    height = bbox[2]
    if height > 1e-12:
        v = v * (target_height / height)

    v[:, 2] -= v[:, 2].min()
    return v, f


# Downloaded model files
DOWNLOADED_FILES = {
    "camila_octopus": "camila_octopus.stl",
    "drlex_octopus": "drlex_octopus.stl",
    "armadillo": "armadillo.obj",
    "beast": "beast.obj",
    "ogre": "ogre.obj",
    "cheburashka": "cheburashka.obj",
    "suzanne": "suzanne.obj",
}

# Generated model names
GENERATED_NAMES = [
    "octopus_basic", "octopus_many_arms", "octopus_curly",
    "octopus_long", "octopus_stubby",
    "squid_basic", "squid_long", "squid_compact",
    "cthulhu_basic", "cthulhu_many_tentacles",
    "jellyfish_basic", "jellyfish_many",
    "kraken_basic", "kraken_massive",
]


def list_models():
    """Return list of all model names (downloaded + generated)."""
    names = []
    # Only include downloaded models that actually exist on disk
    for name, fname in DOWNLOADED_FILES.items():
        if os.path.exists(os.path.join(_DIR, fname)):
            names.append(name)
    names.extend(GENERATED_NAMES)
    return names


def load_model(name, normalize=True, target_height=100.0):
    """Load a cephalopod model by name.

    Args:
        name: model name
        normalize: if True, center/orient/scale
        target_height: target Z-height

    Returns:
        (vertices, faces) numpy arrays
    """
    # Check downloaded models first
    if name in DOWNLOADED_FILES:
        filepath = os.path.join(_DIR, DOWNLOADED_FILES[name])
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")
        v, f = _read_mesh(filepath)
        if normalize:
            v, f = _normalize_mesh(v, f, target_height)
        return v, f

    # Check generated models
    if name in GENERATED_NAMES:
        from dev_models.cephalopods.generators import GENERATORS
        if name not in GENERATORS:
            raise ValueError(f"Generator not found: {name}")
        v, f = GENERATORS[name]()
        if normalize:
            v, f = _normalize_mesh(v, f, target_height)
        return v, f

    raise ValueError(f"Unknown model: {name}. Available: {list_models()}")


def load_all_models(normalize=True, target_height=100.0):
    """Load all available models.

    Returns:
        dict of name -> (vertices, faces)
    """
    result = {}
    for name in list_models():
        try:
            result[name] = load_model(name, normalize=normalize,
                                      target_height=target_height)
        except Exception as e:
            print(f"Warning: failed to load {name}: {e}")
    return result
