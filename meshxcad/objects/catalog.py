"""Catalog of 19 decorative objects, each with simple and ornate versions.

Every object is defined by a pair of functions:
    make_<name>_simple() -> (vertices, faces)
    make_<name>_ornate() -> (vertices, faces)

Object list:
 1. Classical vase      2. Goblet/chalice     3. Candlestick
 4. Chess king          5. Chess queen         6. Bell
 7. Trophy cup          8. Column              9. Baluster
10. Teapot             11. Perfume bottle     12. Table lamp
13. Finial             14. Door knob          15. Ornamental egg
16. Wine decanter      17. Decorative bowl    18. Pedestal
19. Spinning top
"""

import math
import numpy as np
from .builder import revolve_profile, make_torus, combine_meshes, smooth_profile, lerp

N_ANG = 48  # default angular resolution


# ============================================================================
# Registry
# ============================================================================

OBJECT_CATALOG = {}


def _register(name, description, simple_fn, ornate_fn):
    OBJECT_CATALOG[name] = {
        "description": description,
        "simple": simple_fn,
        "ornate": ornate_fn,
    }


# ============================================================================
# 1. Classical Vase
# ============================================================================

def _vase_simple():
    h = 120
    profile = smooth_profile([
        (20, 0), (22, 5), (25, 15), (30, 30),
        (28, 50), (18, 70), (15, 80), (18, 90),
        (22, 100), (20, 110), (15, h),
    ])
    return revolve_profile(profile, N_ANG)


def _vase_ornate():
    h = 120
    profile = smooth_profile([
        (22, 0), (24, 3), (28, 8), (30, 12),  # stepped base
        (26, 16), (24, 20), (28, 25),          # base molding
        (32, 35), (33, 45), (32, 55),          # lower belly
        (28, 65), (20, 75),                     # waist
        (16, 82), (14, 86),                     # neck
        (16, 90), (20, 95), (24, 100),          # lip flare
        (26, 105), (25, 108),                   # lip top
        (22, 112), (18, 115), (12, h),          # inner lip
    ], n_output=60)
    meshes = [revolve_profile(profile, N_ANG)]
    # Decorative rings
    for z in [12, 25, 65]:
        meshes.append(make_torus(28, 1.5, z, N_ANG, 10))
    return combine_meshes(meshes)

_register("classical_vase", "Classical urn-shaped vase with flared lip", _vase_simple, _vase_ornate)


# ============================================================================
# 2. Goblet / Chalice
# ============================================================================

def _goblet_simple():
    profile = smooth_profile([
        (18, 0), (20, 3), (18, 8),   # base
        (5, 15), (4, 40), (5, 55),    # stem
        (15, 65), (22, 75), (24, 85), # cup
        (24, 95), (22, 100),          # rim
    ])
    return revolve_profile(profile, N_ANG)


def _goblet_ornate():
    profile = smooth_profile([
        (20, 0), (22, 2), (24, 5), (22, 8), (18, 12),  # stepped base
        (8, 16), (6, 20),                                 # base-to-stem
        (5, 25), (7, 30), (5, 35),                        # stem bead
        (4, 40), (6, 45), (4, 50),                        # stem bead 2
        (5, 55), (8, 58),                                  # knop
        (12, 60), (8, 62),                                 # knop peak
        (6, 64),                                            # cup base
        (18, 70), (24, 78), (26, 86),                      # cup bowl
        (26, 92), (27, 96), (25, 100),                     # rim
        (24, 101),                                          # rim lip
    ], n_output=60)
    meshes = [revolve_profile(profile, N_ANG)]
    meshes.append(make_torus(22, 1.2, 5, N_ANG, 8))
    meshes.append(make_torus(6, 1.0, 30, N_ANG, 8))
    return combine_meshes(meshes)

_register("goblet", "Chalice/goblet with stem and cup", _goblet_simple, _goblet_ornate)


# ============================================================================
# 3. Candlestick
# ============================================================================

def _candlestick_simple():
    profile = smooth_profile([
        (18, 0), (20, 5), (15, 10),  # base
        (5, 20), (4, 60), (5, 70),   # shaft
        (8, 75), (10, 80),           # drip tray
        (4, 82), (3, 95),            # candle holder
    ])
    return revolve_profile(profile, N_ANG)


def _candlestick_ornate():
    profile = smooth_profile([
        (22, 0), (24, 3), (22, 6), (18, 10),  # ornate base
        (12, 14), (8, 18), (6, 22),
        (5, 28), (7, 32), (5, 36),             # lower bead
        (4, 42), (6, 46), (4, 50),             # mid bead
        (3.5, 56), (5, 60), (3.5, 64),         # upper bead
        (5, 68), (8, 72),                       # approach drip
        (12, 76), (14, 78), (12, 80),          # drip tray
        (8, 81), (4, 83),                       # cup
        (5, 86), (4, 88), (3, 95),             # candle socket
    ], n_output=60)
    meshes = [revolve_profile(profile, N_ANG)]
    for z in [6, 32, 46, 60]:
        meshes.append(make_torus(5, 1.0, z, N_ANG, 8))
    return combine_meshes(meshes)

_register("candlestick", "Turned candlestick holder", _candlestick_simple, _candlestick_ornate)


# ============================================================================
# 4. Chess King
# ============================================================================

def _chess_king_simple():
    profile = smooth_profile([
        (14, 0), (15, 3), (12, 8),    # base
        (5, 15), (4, 30), (5, 38),    # stem
        (8, 42), (10, 48),            # collar
        (12, 55), (11, 65),           # head
        (8, 70), (4, 75),             # crown taper
        (2, 80), (3, 82), (1, 85),   # cross
    ])
    return revolve_profile(profile, N_ANG)


def _chess_king_ornate():
    profile = smooth_profile([
        (16, 0), (18, 2), (16, 4), (14, 7), (10, 10),  # stepped base
        (6, 14), (5, 18),
        (4, 22), (6, 26), (4, 30),   # stem bead
        (5, 34), (7, 38),            # collar
        (10, 42), (12, 45), (10, 48), # ornate collar
        (8, 50), (13, 56), (14, 62), # head
        (13, 66), (10, 70),          # head narrow
        (6, 73), (3, 76),            # crown taper
        (5, 78), (2, 80),            # crown points
        (4, 82), (1.5, 84),          # cross bar
        (3, 85), (1, 88),            # cross top
    ], n_output=60)
    meshes = [revolve_profile(profile, N_ANG)]
    meshes.append(make_torus(12, 1.0, 45, N_ANG, 8))
    meshes.append(make_torus(14, 0.8, 62, N_ANG, 8))
    return combine_meshes(meshes)

_register("chess_king", "Chess king piece with cross", _chess_king_simple, _chess_king_ornate)


# ============================================================================
# 5. Chess Queen
# ============================================================================

def _chess_queen_simple():
    profile = smooth_profile([
        (14, 0), (15, 3), (12, 8),
        (5, 15), (4, 30), (5, 38),
        (8, 42), (10, 48),
        (12, 55), (11, 62),
        (8, 66), (5, 70),
        (7, 73), (3, 76),  # crown
        (1, 78),
    ])
    return revolve_profile(profile, N_ANG)


def _chess_queen_ornate():
    profile = smooth_profile([
        (16, 0), (18, 2), (16, 5), (14, 8),
        (8, 12), (6, 16), (5, 20),
        (4, 24), (6, 28), (4, 32),   # stem bead
        (5, 36), (8, 40), (10, 44),  # collar
        (12, 48), (10, 50),
        (13, 55), (14, 60), (13, 64), # head
        (10, 67), (7, 70),
        (9, 72), (8, 74), (10, 76),  # crown spikes
        (6, 78), (3, 79),
        (4, 80), (2, 81),            # ball top
        (1, 82),
    ], n_output=60)
    meshes = [revolve_profile(profile, N_ANG)]
    meshes.append(make_torus(10, 0.8, 44, N_ANG, 8))
    meshes.append(make_torus(13, 0.7, 60, N_ANG, 8))
    return combine_meshes(meshes)

_register("chess_queen", "Chess queen piece with crown", _chess_queen_simple, _chess_queen_ornate)


# ============================================================================
# 6. Bell
# ============================================================================

def _bell_simple():
    profile = smooth_profile([
        (25, 0), (24, 5), (20, 15),
        (14, 30), (8, 45), (5, 55),
        (3, 60), (2, 65),
        (3, 68), (2, 72),  # clapper mount
    ])
    return revolve_profile(profile, N_ANG)


def _bell_ornate():
    profile = smooth_profile([
        (28, 0), (27, 2), (26, 4), (28, 6),  # lip ring
        (25, 10), (24, 14),
        (22, 20), (20, 26),
        (16, 34), (12, 42),
        (8, 50), (6, 55),
        (4, 58), (3, 60),
        (2, 62), (4, 64), (3, 66),  # crown
        (5, 68), (3, 70),
        (4, 72), (2, 74),           # handle loop
    ], n_output=60)
    meshes = [revolve_profile(profile, N_ANG)]
    meshes.append(make_torus(27, 1.5, 3, N_ANG, 10))
    meshes.append(make_torus(22, 1.0, 20, N_ANG, 8))
    return combine_meshes(meshes)

_register("bell", "Decorative bell with handle", _bell_simple, _bell_ornate)


# ============================================================================
# 7. Trophy Cup
# ============================================================================

def _trophy_simple():
    profile = smooth_profile([
        (16, 0), (18, 5), (14, 10),  # base
        (5, 18), (4, 35),            # stem
        (8, 42), (14, 50),           # cup lower
        (18, 60), (20, 70),          # cup body
        (22, 80), (20, 85),          # rim
    ])
    return revolve_profile(profile, N_ANG)


def _trophy_ornate():
    profile = smooth_profile([
        (18, 0), (20, 2), (22, 5), (20, 8), (16, 12),  # stepped base
        (8, 16), (6, 20),
        (5, 24), (7, 28), (5, 32),  # stem bead
        (4, 36), (6, 40),
        (10, 44), (14, 48),          # knop
        (10, 50), (8, 52),
        (14, 56), (18, 62),          # cup
        (22, 70), (24, 78),
        (25, 84), (24, 87), (26, 90), # ornate rim
        (24, 92), (22, 93),
    ], n_output=60)
    meshes = [revolve_profile(profile, N_ANG)]
    meshes.append(make_torus(20, 1.2, 5, N_ANG, 8))
    meshes.append(make_torus(14, 1.0, 48, N_ANG, 8))
    meshes.append(make_torus(25, 1.0, 87, N_ANG, 8))
    return combine_meshes(meshes)

_register("trophy_cup", "Trophy/loving cup", _trophy_simple, _trophy_ornate)


# ============================================================================
# 8. Column (Classical)
# ============================================================================

def _column_simple():
    h = 160
    r = 10
    profile = [
        (r + 4, 0), (r + 4, 6),   # base
        (r, 10), (r, h - 10),     # shaft
        (r + 4, h - 6), (r + 4, h), # capital
    ]
    return revolve_profile(smooth_profile(profile, 30), N_ANG)


def _column_ornate():
    h = 160
    r = 10
    # Entasis (slight outward curve of shaft) + fluting approximation
    profile_pts = [
        (r + 6, 0), (r + 7, 2), (r + 6, 4), (r + 5, 6),  # base moldings
        (r + 3, 8), (r + 1, 10), (r, 12),
    ]
    # Shaft with entasis
    for i in range(20):
        t = i / 19
        z = 12 + t * (h - 30)
        entasis = 0.8 * math.sin(math.pi * t) ** 0.3
        profile_pts.append((r + entasis, z))
    # Capital
    profile_pts.extend([
        (r, h - 18), (r + 1, h - 16),
        (r + 3, h - 14), (r + 5, h - 12),  # echinus
        (r + 7, h - 10), (r + 8, h - 8),
        (r + 6, h - 6), (r + 8, h - 4),    # abacus
        (r + 8, h - 2), (r + 7, h),
    ])
    meshes = [revolve_profile(smooth_profile(profile_pts, 60), N_ANG)]
    meshes.append(make_torus(r + 6, 1.0, 2, N_ANG, 8))
    meshes.append(make_torus(r + 7, 1.0, h - 8, N_ANG, 8))
    return combine_meshes(meshes)

_register("column", "Classical column with base and capital", _column_simple, _column_ornate)


# ============================================================================
# 9. Baluster
# ============================================================================

def _baluster_simple():
    h = 100
    profile = smooth_profile([
        (8, 0), (10, 5), (8, 10),  # base
        (5, 20), (7, 40),          # lower shaft
        (10, 55),                   # belly
        (7, 70), (5, 80),          # upper shaft
        (8, 90), (10, 95), (8, h),  # capital
    ])
    return revolve_profile(profile, N_ANG)


def _baluster_ornate():
    h = 100
    profile = smooth_profile([
        (10, 0), (12, 2), (10, 4), (8, 8), (6, 12),    # ornate base
        (5, 16), (4, 20),
        (6, 24), (4, 28),                                 # lower bead
        (5, 32), (8, 38), (10, 44),                       # lower swell
        (12, 50), (13, 55), (12, 60),                     # belly
        (10, 64), (8, 68),                                 # upper swell
        (5, 72), (4, 76),
        (6, 80), (4, 84),                                 # upper bead
        (6, 88), (8, 92), (10, 95), (12, 98), (10, h),  # capital
    ], n_output=60)
    meshes = [revolve_profile(profile, N_ANG)]
    meshes.append(make_torus(12, 1.0, 2, N_ANG, 8))
    meshes.append(make_torus(13, 1.0, 55, N_ANG, 8))
    meshes.append(make_torus(12, 1.0, 98, N_ANG, 8))
    return combine_meshes(meshes)

_register("baluster", "Turned baluster/banister post", _baluster_simple, _baluster_ornate)


# ============================================================================
# 10. Teapot (body only — rotationally symmetric approximation)
# ============================================================================

def _teapot_simple():
    profile = smooth_profile([
        (5, 0), (10, 5), (20, 15),
        (25, 25), (26, 35), (25, 45),
        (22, 52), (18, 58),
        (12, 62), (8, 65),
        (5, 67), (3, 70),  # lid knob
        (1, 72),
    ])
    return revolve_profile(profile, N_ANG)


def _teapot_ornate():
    profile = smooth_profile([
        (6, 0), (8, 2), (6, 4),       # foot ring
        (12, 8), (20, 15),
        (26, 22), (28, 30), (27, 38), # belly
        (24, 44), (20, 50),
        (16, 54), (12, 58),
        (10, 60), (12, 62), (10, 64), # shoulder ring
        (8, 65),
        (6, 66), (8, 68), (6, 70),   # lid edge
        (4, 71), (5, 73), (3, 75),   # finial
        (4, 76), (2, 78), (1, 79),
    ], n_output=60)
    meshes = [revolve_profile(profile, N_ANG)]
    meshes.append(make_torus(8, 1.0, 2, N_ANG, 8))
    meshes.append(make_torus(12, 0.8, 62, N_ANG, 8))
    return combine_meshes(meshes)

_register("teapot", "Decorative teapot body", _teapot_simple, _teapot_ornate)


# ============================================================================
# 11. Perfume Bottle
# ============================================================================

def _perfume_simple():
    profile = smooth_profile([
        (12, 0), (14, 5), (16, 15),
        (16, 30), (14, 40),
        (5, 45), (3, 50), (3, 60),  # neck
        (4, 62), (2, 65),           # stopper
    ])
    return revolve_profile(profile, N_ANG)


def _perfume_ornate():
    profile = smooth_profile([
        (14, 0), (16, 2), (14, 4),       # foot
        (16, 8), (18, 14),
        (20, 20), (20, 28),               # body
        (18, 34), (14, 38),
        (8, 42), (4, 46), (3, 50),        # shoulder/neck
        (4, 52), (3, 54),                  # neck ring
        (3, 58), (4, 60),                  # collar
        (6, 62), (8, 64), (7, 66),        # stopper body
        (5, 68), (3, 70), (1, 72),        # stopper tip
    ], n_output=60)
    meshes = [revolve_profile(profile, N_ANG)]
    meshes.append(make_torus(16, 1.0, 2, N_ANG, 8))
    meshes.append(make_torus(4, 0.6, 52, N_ANG, 8))
    return combine_meshes(meshes)

_register("perfume_bottle", "Ornate perfume bottle with stopper", _perfume_simple, _perfume_ornate)


# ============================================================================
# 12. Table Lamp
# ============================================================================

def _lamp_simple():
    profile = smooth_profile([
        (16, 0), (18, 5), (14, 10),  # base
        (5, 20), (4, 50),            # stem
        (6, 55), (8, 58),            # socket
        (3, 60), (3, 65),            # bulb holder
    ])
    return revolve_profile(profile, N_ANG)


def _lamp_ornate():
    profile = smooth_profile([
        (18, 0), (20, 2), (22, 5), (20, 8), (16, 12),  # stepped base
        (10, 15), (8, 18),
        (6, 22), (8, 26), (6, 30),   # lower bead
        (4, 35), (3, 40),
        (5, 44), (3, 48),             # mid bead
        (4, 52), (6, 55),
        (10, 58), (8, 60),            # socket flare
        (5, 62), (6, 64),             # socket ring
        (4, 65), (3, 66),             # top
        (4, 68), (2, 70), (1, 72),   # finial
    ], n_output=60)
    meshes = [revolve_profile(profile, N_ANG)]
    meshes.append(make_torus(20, 1.2, 5, N_ANG, 8))
    meshes.append(make_torus(8, 0.8, 26, N_ANG, 8))
    return combine_meshes(meshes)

_register("table_lamp", "Table lamp base with turned details", _lamp_simple, _lamp_ornate)


# ============================================================================
# 13. Finial (fence post / lamp finial)
# ============================================================================

def _finial_simple():
    profile = smooth_profile([
        (10, 0), (10, 4),            # base collar
        (6, 8), (5, 14),
        (7, 22), (9, 30), (8, 38),  # bulge
        (5, 44), (3, 50),           # taper
        (1, 58),                     # point
    ])
    return revolve_profile(profile, N_ANG)


def _finial_ornate():
    profile = smooth_profile([
        (10, 0), (12, 2), (10, 4),        # base ring
        (8, 6), (6, 8),
        (8, 12), (10, 16), (8, 20),       # lower bead
        (6, 22), (4, 24),
        (6, 28), (9, 32), (10, 36),       # main bulge
        (9, 40), (6, 42),
        (4, 44), (6, 46), (4, 48),        # upper bead
        (3, 50), (2, 52),
        (3, 54), (2, 56), (1, 58),        # spike tip
        (0.5, 60),
    ], n_output=60)
    meshes = [revolve_profile(profile, N_ANG)]
    meshes.append(make_torus(12, 1.0, 2, N_ANG, 8))
    meshes.append(make_torus(10, 0.8, 36, N_ANG, 8))
    return combine_meshes(meshes)

_register("finial", "Decorative finial/post cap", _finial_simple, _finial_ornate)


# ============================================================================
# 14. Door Knob
# ============================================================================

def _doorknob_simple():
    profile = smooth_profile([
        (3, 0), (4, 2),              # stem
        (4, 8), (8, 12),             # flare
        (16, 18), (18, 24),          # knob body
        (16, 30), (10, 34),          # top
        (4, 36), (1, 38),            # crown
    ])
    return revolve_profile(profile, N_ANG)


def _doorknob_ornate():
    profile = smooth_profile([
        (4, 0), (5, 1), (4, 2),      # stem
        (3, 4), (5, 6), (4, 8),      # stem ring
        (8, 10), (12, 12),
        (16, 15), (18, 18),           # knob lower
        (20, 22), (20, 26),           # widest
        (18, 30), (14, 33),           # upper curve
        (8, 35),
        (6, 36), (8, 37), (6, 38),   # decorative ring
        (4, 39), (2, 40),
        (3, 41), (1, 42),            # finial tip
    ], n_output=60)
    meshes = [revolve_profile(profile, N_ANG)]
    meshes.append(make_torus(5, 0.8, 6, N_ANG, 8))
    meshes.append(make_torus(20, 1.0, 24, N_ANG, 8))
    return combine_meshes(meshes)

_register("door_knob", "Decorative door knob", _doorknob_simple, _doorknob_ornate)


# ============================================================================
# 15. Ornamental Egg (Fabergé style)
# ============================================================================

def _egg_simple():
    # Simple egg on a plain stand (must match ornate topology)
    stand = smooth_profile([
        (12, -10), (12, -6),   # plain base
        (6, -2), (6, 0),      # plain pedestal
    ], n_output=20)
    egg = smooth_profile([
        (6, 0),
        (12, 8), (18, 18), (22, 30),
        (23, 42), (22, 54), (18, 64),
        (12, 72), (4, 78),
        (0.5, 82),
    ])
    meshes = [
        revolve_profile(stand, N_ANG),
        revolve_profile(egg, N_ANG, close_top=False, close_bottom=False),
    ]
    return combine_meshes(meshes)


def _egg_ornate():
    # Egg on a stand with decorative band
    stand = smooth_profile([
        (14, -10), (16, -8), (14, -6),  # base ring
        (8, -4), (6, -2), (8, 0),       # pedestal
    ], n_output=20)
    egg = smooth_profile([
        (8, 0), (12, 4),
        (16, 10), (20, 18),
        (22, 28), (23, 38),    # widest
        (22, 48), (20, 56),
        (16, 64), (10, 72),
        (4, 78), (0.5, 82),
    ], n_output=40)
    # Decorative band at equator
    meshes = [
        revolve_profile(stand, N_ANG),
        revolve_profile(egg, N_ANG, close_top=False, close_bottom=False),
    ]
    meshes.append(make_torus(23, 1.2, 38, N_ANG, 10))
    meshes.append(make_torus(16, 0.8, 64, N_ANG, 8))
    meshes.append(make_torus(16, 0.8, 10, N_ANG, 8))
    meshes.append(make_torus(16, 0.5, -8, N_ANG, 8))
    return combine_meshes(meshes)

_register("ornamental_egg", "Fabergé-style ornamental egg on stand", _egg_simple, _egg_ornate)


# ============================================================================
# 16. Wine Decanter
# ============================================================================

def _decanter_simple():
    profile = smooth_profile([
        (8, 0), (20, 10), (24, 25),
        (24, 40), (20, 55),
        (10, 65), (5, 75), (4, 90),  # neck
        (5, 95), (3, 100),           # lip
    ])
    return revolve_profile(profile, N_ANG)


def _decanter_ornate():
    profile = smooth_profile([
        (10, 0), (12, 3), (10, 6),           # foot ring
        (16, 10), (22, 18),
        (26, 28), (27, 38), (26, 48),         # belly
        (22, 56), (16, 62),
        (8, 68), (5, 74),                      # shoulder
        (4, 78), (5, 82), (4, 86),            # neck ring
        (3.5, 90), (4, 94),                    # upper neck
        (6, 96), (7, 98), (6, 100),           # flared lip
        (5, 101), (4, 102),
    ], n_output=60)
    meshes = [revolve_profile(profile, N_ANG)]
    meshes.append(make_torus(12, 1.0, 3, N_ANG, 8))
    meshes.append(make_torus(5, 0.7, 82, N_ANG, 8))
    meshes.append(make_torus(7, 0.8, 98, N_ANG, 8))
    return combine_meshes(meshes)

_register("wine_decanter", "Wine decanter with bulbous body", _decanter_simple, _decanter_ornate)


# ============================================================================
# 17. Decorative Bowl
# ============================================================================

def _bowl_simple():
    profile = smooth_profile([
        (5, 0), (8, 3),             # foot
        (15, 10), (22, 18),
        (26, 26), (28, 34),         # bowl wall
        (28, 38),                    # rim
    ])
    return revolve_profile(profile, N_ANG)


def _bowl_ornate():
    profile = smooth_profile([
        (6, 0), (8, 1), (6, 2),         # foot ring
        (8, 4), (6, 6),                  # foot detail
        (10, 8), (16, 12),
        (22, 18), (26, 24),
        (28, 28), (30, 32),              # wider bowl
        (30, 36), (32, 38),              # rim flare
        (30, 39), (28, 40),              # rim lip
    ], n_output=60)
    meshes = [revolve_profile(profile, N_ANG)]
    meshes.append(make_torus(8, 0.8, 1, N_ANG, 8))
    meshes.append(make_torus(32, 1.0, 38, N_ANG, 8))
    return combine_meshes(meshes)

_register("decorative_bowl", "Decorative serving bowl with foot ring", _bowl_simple, _bowl_ornate)


# ============================================================================
# 18. Pedestal
# ============================================================================

def _pedestal_simple():
    h = 100
    profile = smooth_profile([
        (20, 0), (22, 5), (20, 10),   # base
        (14, 15), (14, h - 15),       # shaft
        (20, h - 10), (22, h - 5), (20, h),
    ])
    return revolve_profile(profile, N_ANG)


def _pedestal_ornate():
    h = 100
    profile = smooth_profile([
        (22, 0), (24, 2), (22, 4), (20, 6), (24, 8),  # base moldings
        (22, 10), (18, 14), (16, 18),
        (14, 22), (15, 30), (14, 40),   # lower shaft
        (16, 48), (18, 52), (16, 56),   # mid ring
        (14, 60), (15, 70), (14, 78),   # upper shaft
        (16, 82), (18, 86),
        (22, 90), (24, 92), (22, 94),   # capital moldings
        (20, 96), (24, 98), (22, h),
    ], n_output=60)
    meshes = [revolve_profile(profile, N_ANG)]
    meshes.append(make_torus(24, 1.2, 2, N_ANG, 8))
    meshes.append(make_torus(18, 1.0, 52, N_ANG, 8))
    meshes.append(make_torus(24, 1.2, 98, N_ANG, 8))
    return combine_meshes(meshes)

_register("pedestal", "Display pedestal with molding", _pedestal_simple, _pedestal_ornate)


# ============================================================================
# 19. Spinning Top
# ============================================================================

def _top_simple():
    profile = smooth_profile([
        (0.5, 0), (2, 2),            # tip
        (10, 10), (18, 18),
        (22, 26),                     # widest
        (18, 32), (12, 38),          # upper dome
        (6, 44), (3, 48),            # handle
        (1, 54),
    ])
    return revolve_profile(profile, N_ANG, close_top=False, close_bottom=False)


def _top_ornate():
    profile = smooth_profile([
        (0.5, 0), (2, 2),             # steel tip
        (4, 4), (10, 10),
        (16, 16), (20, 20),
        (22, 24), (24, 28),           # widest ring
        (22, 30), (18, 33),
        (14, 36), (10, 38),
        (8, 40), (10, 42), (8, 44),  # handle bead
        (5, 46), (3, 48),
        (4, 50), (2, 52), (1, 54),   # handle tip
    ], n_output=60)
    meshes = [revolve_profile(profile, N_ANG, close_top=False, close_bottom=False)]
    meshes.append(make_torus(24, 1.2, 28, N_ANG, 10))
    meshes.append(make_torus(10, 0.8, 42, N_ANG, 8))
    return combine_meshes(meshes)

_register("spinning_top", "Decorative spinning top", _top_simple, _top_ornate)


# ============================================================================
# Public API
# ============================================================================

def list_objects():
    """Return list of all object names."""
    return list(OBJECT_CATALOG.keys())


def get_object(name):
    """Get object entry by name. Returns dict with 'simple' and 'ornate' callables."""
    return OBJECT_CATALOG[name]


def make_simple(name, **kwargs):
    """Generate the simple version of a named object."""
    return OBJECT_CATALOG[name]["simple"]()


def make_ornate(name, **kwargs):
    """Generate the ornate version of a named object."""
    return OBJECT_CATALOG[name]["ornate"]()
