#!/usr/bin/env python3
"""Generate spiral_queen.stl — a chess queen piece with a barley-twist stem.

The spiral queen has:
  - A classical revolve base (z = 0..14)
  - A 4-lobed barley-twist column (z = 14..55, 2 full rotations)
  - An ornate collar, head, and crown (z = 55..100)

The twist challenges standard revolve reconstruction because each height slice
has a non-circular cross-section that rotates with z.

Usage:
    python -m dev_models.chess_queens.generate_spiral_queen
or:
    python dev_models/chess_queens/generate_spiral_queen.py
"""

import math
import sys
import os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from scipy.interpolate import CubicSpline
from meshxcad.stl_io import write_binary_stl


# -----------------------------------------------------------------------
# Parametric spiral queen
# -----------------------------------------------------------------------

N_LOBES = 4          # 4-lobed barley twist (square cross-section that rotates)
TWIST_TURNS = 2.0    # full rotations over stem height
STEM_AMPLITUDE = 2.5  # radial lobe amplitude (units = mm in normalised scale)
N_ANG = 96           # angular resolution
N_Z = 160            # z resolution (more = finer detail)

# Control points for the mean revolve profile (radius, z)
PROFILE_CTRL = [
    (14.0,   0.0),  # base bottom edge
    (15.0,   2.0),  # base flange
    (13.5,   4.0),
    (11.0,   8.0),  # base taper
    ( 7.0,  11.0),
    ( 5.0,  14.0),  # transition into stem
    ( 5.0,  25.0),  # stem (near-uniform)
    ( 5.0,  40.0),
    ( 5.0,  55.0),  # transition out of stem
    ( 8.0,  60.0),  # collar expansion
    (11.5,  65.0),
    (12.5,  68.5),  # head bulge
    (13.5,  73.0),
    (12.5,  76.5),  # head upper
    ( 9.0,  79.5),
    ( 7.0,  82.0),  # crown base
    ( 9.0,  85.5),  # crown swell
    (11.0,  88.5),
    (10.0,  91.0),
    ( 7.0,  93.0),  # ball base
    ( 7.5,  95.0),  # ball equator
    ( 6.5,  97.5),
    ( 3.0, 100.0),  # top
]

Z_STEM_BOT = 14.0
Z_STEM_TOP = 55.0
TOTAL_HEIGHT = 100.0


def _smooth_profile_fn():
    """Return a cubic-spline r(z) for the mean revolve profile."""
    z_ctrl = np.array([p[1] for p in PROFILE_CTRL], dtype=np.float64)
    r_ctrl = np.array([p[0] for p in PROFILE_CTRL], dtype=np.float64)
    return CubicSpline(z_ctrl, r_ctrl, extrapolate=True)


def _amplitude_factor(z):
    """Smooth taper [0, 1] for the spiral amplitude, 0 outside the stem."""
    if z <= Z_STEM_BOT or z >= Z_STEM_TOP:
        return 0.0
    t = (z - Z_STEM_BOT) / (Z_STEM_TOP - Z_STEM_BOT)
    taper_frac = 0.12   # taper over first/last 12% of stem
    if t < taper_frac:
        frac = t / taper_frac
    elif t > 1.0 - taper_frac:
        frac = (1.0 - t) / taper_frac
    else:
        frac = 1.0
    # Cosine smoothing for C¹ continuity
    return 0.5 * (1.0 - math.cos(math.pi * frac))


def _twist_angle(z):
    """Cumulative twist angle (radians) at height z."""
    if z <= Z_STEM_BOT:
        return 0.0
    if z >= Z_STEM_TOP:
        return 2.0 * math.pi * TWIST_TURNS
    t = (z - Z_STEM_BOT) / (Z_STEM_TOP - Z_STEM_BOT)
    return 2.0 * math.pi * TWIST_TURNS * t


def make_spiral_queen():
    """Generate the spiral queen mesh.

    Returns:
        vertices : (N, 3) float64 array
        faces    : (M, 3) int64 array
    """
    cs_r = _smooth_profile_fn()

    verts = []

    # Generate a regular (n_ang × (n_z+1)) grid on the surface
    for iz in range(N_Z + 1):
        t = iz / N_Z
        z = t * TOTAL_HEIGHT

        r_base = max(float(cs_r(z)), 0.5)
        amp = STEM_AMPLITUDE * _amplitude_factor(z)
        twist = _twist_angle(z)

        for ia in range(N_ANG):
            theta = 2.0 * math.pi * ia / N_ANG
            # Barley-twist: radial perturbation that rotates with z
            r = r_base + amp * math.cos(N_LOBES * theta + twist)
            r = max(r, 0.2)
            verts.append([r * math.cos(theta), r * math.sin(theta), z])

    verts = np.array(verts, dtype=np.float64)

    # Side faces
    faces = []
    for iz in range(N_Z):
        for ia in range(N_ANG):
            ia_next = (ia + 1) % N_ANG
            p00 = iz * N_ANG + ia
            p01 = iz * N_ANG + ia_next
            p10 = (iz + 1) * N_ANG + ia
            p11 = (iz + 1) * N_ANG + ia_next
            faces.append([p00, p01, p10])
            faces.append([p01, p11, p10])

    # Bottom cap
    bot_center_idx = len(verts)
    verts = np.vstack([verts, [[0.0, 0.0, 0.0]]])
    for ia in range(N_ANG):
        ia_next = (ia + 1) % N_ANG
        faces.append([bot_center_idx, ia_next, ia])

    # Top cap
    top_start = N_Z * N_ANG
    top_center_idx = len(verts)
    verts = np.vstack([verts, [[0.0, 0.0, TOTAL_HEIGHT]]])
    for ia in range(N_ANG):
        ia_next = (ia + 1) % N_ANG
        faces.append([top_center_idx, top_start + ia, top_start + ia_next])

    return verts, np.array(faces, dtype=np.int64)


if __name__ == "__main__":
    out_path = os.path.join(os.path.dirname(__file__), "spiral_queen.stl")
    print("Generating spiral queen mesh...")
    v, f = make_spiral_queen()
    print(f"  vertices: {len(v)}, faces: {len(f)}")
    write_binary_stl(out_path, v, f)
    print(f"  saved → {out_path}")
