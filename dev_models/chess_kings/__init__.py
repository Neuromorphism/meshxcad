"""Chess king model collection for CAD strategy development.

Twenty diverse chess king piece meshes downloaded from open-source 3D model
repositories (GitHub).  Each model has different proportions, detail levels,
and stylistic features suitable for stress-testing mesh-to-CAD reconstruction.

Sources:
    staunton_king       — clarkerubber/Staunton-Pieces (Staunton set)
    duchamp_king        — clarkerubber/Staunton-Pieces (Duchamp set)
    chessbuds_king      — kyblacklock/Chessbuds
    nanomars_king       — NanoMars/Cad-Chess
    dav3xor_king        — Dav3xor/chess_set (OpenSCAD)
    kings_gambit_king   — iamwilhelm/kings_gambit
    drosini_king        — drosini/Chess
    courier_king        — K-Francis-H/courier-chess-3dprint
    printable_chess_king— 0x3b29/3D-Printable-Chess
    ernest_king         — ernest-rudnicki/chess-3d
    faisal_king         — faisal004/3d-chess
    react3_king         — tcannon686/react-three-chess
    godot_king          — JEnriquezG88/Godot-3D-Chess
    nemesis_king        — nemesisx00/godot-chess
    webgl_king          — nfriend/webgl-chess
    webxr_king          — Brijesh1005/webxr-chess-game
    yxshee_king         — yxshee/user-interactive-3D-chess
    chess3d_king        — Henrik-Hey/Chess3D
    kuboi_king          — kuboi1/godot-chess-3d
    kessler_king        — thirtythreedown/kessler-chess-set
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
        # Swap tallest axis with Z
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


# All 20 king models (excluding very low-poly aartics and echiquierjs)
KING_FILES = {
    "staunton":       "staunton_king.stl",
    "duchamp":        "duchamp_king.stl",
    "chessbuds":      "chessbuds_king.stl",
    "nanomars":       "nanomars_king.stl",
    "dav3xor":        "dav3xor_king.stl",
    "kings_gambit":   "kings_gambit_king.stl",
    "drosini":        "drosini_king.stl",
    "courier":        "courier_king.stl",
    "printable":      "printable_chess_king.stl",
    "ernest":         "ernest_king.stl",
    "faisal":         "faisal_king.stl",
    "react3":         "react3_king.stl",
    "godot":          "godot_king.stl",
    "nemesis":        "nemesis_king.stl",
    "webgl":          "webgl_king.stl",
    "webxr":          "webxr_king.stl",
    "yxshee":         "yxshee_king.stl",
    "chess3d":        "chess3d_king.stl",
    "kuboi":          "kuboi_king.stl",
    "kessler":        "kessler_king.stl",
}


def list_kings():
    """Return list of all king model names."""
    return list(KING_FILES.keys())


def load_king(name, normalize=True, target_height=100.0):
    """Load a king mesh by name.

    Args:
        name: king name (e.g. 'staunton', 'duchamp')
        normalize: if True, center, orient, and scale to target_height
        target_height: target Z-height after normalization

    Returns:
        (vertices, faces) numpy arrays
    """
    if name not in KING_FILES:
        raise ValueError(f"Unknown king: {name}. Available: {list_kings()}")

    filepath = os.path.join(_DIR, KING_FILES[name])
    v, f = _read_stl(filepath)

    if normalize:
        v, f = _normalize_mesh(v, f, target_height)

    return v, f


def load_all_kings(normalize=True, target_height=100.0):
    """Load all king meshes.

    Returns:
        dict of name -> (vertices, faces)
    """
    result = {}
    for name in KING_FILES:
        try:
            result[name] = load_king(name, normalize=normalize,
                                     target_height=target_height)
        except Exception as e:
            print(f"Warning: failed to load {name}: {e}")
    return result
