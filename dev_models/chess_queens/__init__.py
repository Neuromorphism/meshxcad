"""Chess queen model collection for CAD strategy development.

Twelve diverse chess queen piece meshes downloaded from open-source 3D model
repositories (GitHub).  Each model has different proportions, detail levels,
and stylistic features — complementary to the chess_kings benchmark for
stress-testing mesh-to-CAD reconstruction of crowned/ornate shapes.

Sources:
    staunton_queen       — clarkerubber/Staunton-Pieces (Staunton set)
    duchamp_queen        — clarkerubber/Staunton-Pieces (Duchamp set)
    chessbuds_queen      — kyblacklock/Chessbuds
    kings_gambit_queen   — iamwilhelm/kings_gambit
    openscad_cdr_queen   — chrisrobison/openscad-chess (cdr variant)
    openscad_smiley_queen— chrisrobison/openscad-chess (smiley variant)
    openscad_hillary_queen— chrisrobison/openscad-chess (hillary variant)
    openscad_jib_queen   — chrisrobison/openscad-chess (jib variant)
    openscad_trump_queen — chrisrobison/openscad-chess (trump variant)
    decent_queen         — RLuckom/decent-chess
    scenevr_queen        — scenevr/chess (OBJ format)
    smycynek_queen       — smycynek/chess-puzzles-3d (OBJ format)
    stevenalbert_queen   — stevenalbert/3d-chess-opengl (OBJ format)
    bnolin_queen         — Bnolin/3D-Chess-Game (OBJ format)
"""

import os
import numpy as np

_DIR = os.path.dirname(__file__)


def _read_stl(filepath):
    """Read an STL file (binary or ASCII). Lazy import to avoid circular deps."""
    import sys
    sys.path.insert(0, os.path.join(_DIR, "..", ".."))
    from meshxcad.stl_io import read_binary_stl
    return read_binary_stl(filepath)


def _read_obj(filepath):
    """Read a Wavefront OBJ file, returning (vertices, faces) numpy arrays.

    Handles triangulated and quad faces (quads are split into triangles).
    """
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
                # Fan-triangulate polygons
                for i in range(1, len(idxs) - 1):
                    face_list.append([idxs[0], idxs[i], idxs[i + 1]])
    if not verts:
        raise ValueError(f"No vertices found in OBJ: {filepath}")
    return np.array(verts, dtype=np.float64), np.array(face_list, dtype=np.int64)


def _read_mesh(filepath):
    """Read a mesh file (STL or OBJ) based on extension."""
    ext = os.path.splitext(filepath)[1].lower()
    if ext == ".obj":
        return _read_obj(filepath)
    else:
        return _read_stl(filepath)


def _normalize_mesh(vertices, faces, target_height=100.0):
    """Center and scale a mesh so its bounding box height equals target_height.

    Also reorients so the tallest axis is Z (vertical).
    """
    v = np.array(vertices, dtype=np.float64)
    f = np.array(faces, dtype=np.int64)

    # Center at origin
    center = (v.max(axis=0) + v.min(axis=0)) / 2
    v = v - center

    # Find the tallest axis and rotate if needed
    bbox = v.max(axis=0) - v.min(axis=0)
    tallest = int(np.argmax(bbox))
    if tallest != 2:
        v[:, [tallest, 2]] = v[:, [2, tallest]]

    # Scale to target height
    bbox = v.max(axis=0) - v.min(axis=0)
    height = bbox[2]
    if height > 1e-12:
        scale = target_height / height
        v = v * scale

    # Shift so bottom is at z=0
    v[:, 2] -= v[:, 2].min()

    return v, f


# All queen models
QUEEN_FILES = {
    "staunton":         "staunton_queen.stl",
    "duchamp":          "duchamp_queen.stl",
    "chessbuds":        "chessbuds_queen.stl",
    "kings_gambit":     "kings_gambit_queen.stl",
    "openscad_cdr":     "openscad_cdr_queen.stl",
    "openscad_smiley":  "openscad_smiley_queen.stl",
    "openscad_hillary": "openscad_hillary_queen.stl",
    "openscad_jib":     "openscad_jib_queen.stl",
    "openscad_trump":   "openscad_trump_queen.stl",
    "decent":           "decent_queen.stl",
    "scenevr":          "scenevr_queen.obj",
    "smycynek":         "smycynek_queen.obj",
    "stevenalbert":     "stevenalbert_queen.obj",
    "bnolin":           "bnolin_queen.obj",
}


def list_queens():
    """Return list of all queen model names."""
    return list(QUEEN_FILES.keys())


def load_queen(name, normalize=True, target_height=100.0):
    """Load a queen mesh by name.

    Args:
        name: queen name (e.g. 'staunton', 'duchamp')
        normalize: if True, center, orient, and scale to target_height
        target_height: target Z-height after normalization

    Returns:
        (vertices, faces) numpy arrays
    """
    if name not in QUEEN_FILES:
        raise ValueError(f"Unknown queen: {name}. Available: {list_queens()}")

    filepath = os.path.join(_DIR, QUEEN_FILES[name])
    v, f = _read_mesh(filepath)

    if normalize:
        v, f = _normalize_mesh(v, f, target_height)

    return v, f


def load_all_queens(normalize=True, target_height=100.0):
    """Load all queen meshes.

    Returns:
        dict of name -> (vertices, faces)
    """
    result = {}
    for name in QUEEN_FILES:
        try:
            result[name] = load_queen(name, normalize=normalize,
                                      target_height=target_height)
        except Exception as e:
            print(f"Warning: failed to load {name}: {e}")
    return result
