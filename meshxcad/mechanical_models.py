"""20 mechanical test models for CAD reconstruction benchmarking.

Combines the 15 complex catalog models with 5 additional procedural
mechanical parts to reach the 20-model GrabCAD-equivalent target.

Each model is moderately to highly complex with features like:
- Bores, bolt holes, keyways
- Involute gear teeth
- Swept profiles, lofted shapes
- Lattice/array patterns
- Tapered/profiled cylinders
"""

import math
import numpy as np
from .objects.builder import revolve_profile, combine_meshes, make_cylinder
from .objects.operations import (
    extrude_polygon, make_regular_polygon, sweep_along_path,
    circular_array, make_cylinder_at, subtract_cylinders,
)
from .objects.complex_catalog import make_complex_ornate


# ---------------------------------------------------------------------------
# 5 Additional mechanical models (to complement the 15 from complex_catalog)
# ---------------------------------------------------------------------------

def _make_stepped_shaft():
    """Stepped shaft with 4 diameter sections and fillets.

    Common in mechanical engineering: motor shafts, spindles.
    """
    profile = [
        # Section 1: small diameter
        (8.0, 0.0),
        (8.0, 15.0),
        # Fillet/step to section 2
        (9.0, 16.0),
        (12.0, 17.0),
        (12.0, 40.0),
        # Step to section 3
        (13.0, 41.0),
        (16.0, 42.0),
        (16.0, 70.0),
        # Step to section 4
        (13.0, 71.0),
        (10.0, 72.0),
        (10.0, 90.0),
        # Taper to end
        (8.0, 92.0),
        (6.0, 95.0),
    ]
    return revolve_profile(profile, 32)


def _make_piston():
    """Simple piston with crown, ring grooves, skirt, and wrist pin bore.

    Standard mechanical part found in engines.
    """
    # Piston crown and skirt profile
    crown_r = 25.0
    skirt_r = 24.8
    pin_bore_r = 8.0
    total_height = 40.0

    profile = [
        # Crown
        (1.0, total_height),
        (crown_r, total_height),
        (crown_r, total_height - 3.0),
        # Ring groove 1
        (crown_r - 2.0, total_height - 3.5),
        (crown_r - 2.0, total_height - 5.5),
        (crown_r, total_height - 6.0),
        # Ring groove 2
        (crown_r - 2.0, total_height - 6.5),
        (crown_r - 2.0, total_height - 8.5),
        (crown_r, total_height - 9.0),
        # Oil ring groove
        (crown_r - 3.0, total_height - 9.5),
        (crown_r - 3.0, total_height - 13.0),
        (skirt_r, total_height - 13.5),
        # Skirt
        (skirt_r, 5.0),
        (skirt_r - 1.0, 3.0),
        (skirt_r - 3.0, 1.0),
        (pin_bore_r + 2.0, 0.5),
        (pin_bore_r, 0.0),
    ]

    return revolve_profile(profile, 36)


def _make_bearing_housing():
    """Pillow block bearing housing.

    Common mounting bracket for bearings in machinery.
    """
    parts = []

    # Base plate
    base_w, base_h, base_d = 80, 20, 50
    base_poly = [(-base_w/2, -base_d/2), (base_w/2, -base_d/2),
                 (base_w/2, base_d/2), (-base_w/2, base_d/2)]
    base_v, base_f = extrude_polygon(base_poly, base_h)
    parts.append((base_v, base_f))

    # Cylindrical bearing seat
    seat_profile = [
        (30.0, base_h),
        (30.0, base_h + 20),
        (28.0, base_h + 22),
        (25.0, base_h + 24),
        (22.0, base_h + 25),
        (18.0, base_h + 25),
        (15.0, base_h + 24),
        (12.0, base_h + 22),
        (10.0, base_h + 20),
        (10.0, base_h),
    ]
    seat_v, seat_f = revolve_profile(seat_profile, 32)
    parts.append((seat_v, seat_f))

    # Mounting bolt cylinders (4 corners)
    bolt_r = 5.0
    for x_off in [-30, 30]:
        for y_off in [-18, 18]:
            bv, bf = make_cylinder_at(x_off, y_off, 0, base_h + 3, bolt_r, 16, 2)
            parts.append((bv, bf))

    return combine_meshes(parts)


def _make_threaded_rod():
    """Threaded rod with hex head and nut.

    Basic fastener assembly.
    """
    # Hex head
    hex_poly = make_regular_polygon(6, radius=10.0)
    head_v, head_f = extrude_polygon(hex_poly, 8.0)

    # Shank (threaded portion approximated as profiled cylinder)
    profile = []
    shank_length = 60.0
    shank_r = 5.0
    thread_depth = 0.8
    n_threads = 20

    for i in range(n_threads * 4 + 1):
        t = i / (n_threads * 4)
        z = 8.0 + t * shank_length
        # Thread profile: sinusoidal radial variation
        r = shank_r + thread_depth * math.sin(t * n_threads * 2 * math.pi)
        profile.append((max(r, 1.0), z))

    shank_v, shank_f = revolve_profile(profile, 24)

    # Nut at the end
    nut_poly = make_regular_polygon(6, radius=9.0)
    nut_v, nut_f = extrude_polygon(nut_poly, 7.0)
    nut_v[:, 2] += 8.0 + shank_length

    return combine_meshes([(head_v, head_f), (shank_v, shank_f), (nut_v, nut_f)])


def _make_turbine_blade():
    """Simplified turbine blade with airfoil cross-section and twist.

    Tests swept/lofted shapes with varying cross-section.
    """
    # Airfoil profile (simplified NACA-like)
    n_pts = 24
    profile = []
    for i in range(n_pts):
        t = i / n_pts
        angle = 2 * math.pi * t
        # Airfoil shape: thicker near leading edge
        if t < 0.5:
            # Upper surface
            x = 1.0 - math.cos(math.pi * t * 2) * 0.5
            y = 0.5 * math.sin(math.pi * t * 2) * (1.0 - 0.3 * t)
        else:
            # Lower surface
            t2 = (t - 0.5) * 2
            x = math.cos(math.pi * t2) * 0.5 + 0.5
            y = -0.3 * math.sin(math.pi * t2) * (1.0 - 0.3 * t2)
        profile.append((x * 20, y * 20))

    profile_arr = np.array(profile, dtype=np.float64)

    # Blade path: curved and twisted
    n_path = 20
    path = []
    for i in range(n_path):
        t = i / (n_path - 1)
        z = t * 80
        x = 5.0 * math.sin(t * 0.3)
        y = 2.0 * math.sin(t * 0.5)
        path.append([x, y, z])

    path_arr = np.array(path, dtype=np.float64)

    # Scale decreases along blade
    def scale_fn(t):
        return 1.0 - 0.4 * t

    v, f = sweep_along_path(profile_arr, path_arr, twist_total_deg=15,
                            scale_fn=scale_fn)
    return v, f


# ---------------------------------------------------------------------------
# Registry of all 20 mechanical models
# ---------------------------------------------------------------------------

# The 15 complex catalog models
COMPLEX_CATALOG_NAMES = [
    "spur_gear", "pipe_flange", "shelf_bracket", "hex_nut",
    "picture_frame", "star_knob", "fluted_column", "castellated_ring",
    "gear_shift_knob", "lattice_panel", "pulley_sheave", "heat_sink",
    "cam_disc", "hinge_plate", "hex_bolt",
]

# The 5 additional models
ADDITIONAL_MODELS = {
    "stepped_shaft": _make_stepped_shaft,
    "piston": _make_piston,
    "bearing_housing": _make_bearing_housing,
    "threaded_rod": _make_threaded_rod,
    "turbine_blade": _make_turbine_blade,
}


def get_all_mechanical_models():
    """Return dict of name -> (vertices, faces) generator functions.

    Returns 20 mechanical models total.
    """
    models = {}

    # 15 from complex catalog
    for name in COMPLEX_CATALOG_NAMES:
        # Closure to capture name
        def _make(n=name):
            return make_complex_ornate(n)
        models[name] = _make

    # 5 additional
    models.update(ADDITIONAL_MODELS)

    return models


ALL_MECHANICAL_MODELS = get_all_mechanical_models()
