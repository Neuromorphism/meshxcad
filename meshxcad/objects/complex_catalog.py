"""Catalog of 15 complex objects requiring non-revolve CAD operations.

These objects use boolean cuts, extrusions, circular arrays, sweeps,
and pattern features — operations that go beyond simple profile revolution.

Inspired by models found on GrabCAD, TraceParts, and engineering references.

Object list:
 1. Spur gear          — involute tooth extrusion + bore
 2. Pipe flange        — revolve + circular bolt hole array
 3. Shelf bracket      — L-shape extrusion + gusset + filigree holes
 4. Hex nut            — hexagonal prism + threaded bore
 5. Picture frame      — rectangular sweep of molding profile
 6. Star knob          — star extrusion + central bore + knurling
 7. Fluted column      — revolve + longitudinal flute channels
 8. Castellated ring   — revolve + rectangular notch pattern
 9. Gear shift knob    — revolve body + flat cuts + knurling
10. Lattice panel      — grid array of cylindrical bars
11. Pulley sheave      — grooved wheel + hub + keyway
12. Heat sink          — base plate + rectangular fin array
13. Cam disc           — eccentric cam profile + hub + follower track
14. Hinge plate        — flat plate + barrel hinge + screw holes
15. Hex bolt           — hex head + shank + thread detail
"""

import math
import numpy as np
from .builder import (
    revolve_profile, combine_meshes, smooth_profile, make_torus,
    make_cylinder,
)
from .operations import (
    extrude_polygon, make_regular_polygon, make_star_polygon,
    circular_array, make_cylinder_at, subtract_cylinders,
    make_rectangular_frame, make_involute_gear_profile,
    make_knurled_surface,
)

N_ANG = 48

COMPLEX_CATALOG = {}


def _register(name, description, simple_fn, ornate_fn):
    COMPLEX_CATALOG[name] = {
        "description": description,
        "simple": simple_fn,
        "ornate": ornate_fn,
    }


# ============================================================================
# 1. Spur Gear (GrabCAD: spur gear tutorials)
# ============================================================================

def _gear_simple():
    """Simple gear: plain cylinder with a bore hole."""
    outer_r = 30
    bore_r = 8
    thickness = 12
    # Outer ring
    outer = [(outer_r, 0), (outer_r, thickness)]
    outer_mesh = revolve_profile(outer, N_ANG)
    # Bore (inner ring subtracted conceptually — we build a tube)
    profile = [(bore_r, 0), (outer_r, 0), (outer_r, thickness), (bore_r, thickness)]
    return revolve_profile(smooth_profile(profile, 8), N_ANG)


def _gear_ornate():
    """Ornate gear: involute teeth + hub + bore + keyway."""
    module = 2.5
    n_teeth = 20
    thickness = 12
    bore_r = 8
    hub_r = 14
    hub_h = 4

    # Tooth profile
    tooth_poly = make_involute_gear_profile(module, n_teeth)
    gear_body = extrude_polygon(tooth_poly, thickness)

    # Hub extending above
    hub_profile = [(bore_r, thickness), (hub_r, thickness),
                   (hub_r, thickness + hub_h), (bore_r, thickness + hub_h)]
    hub_mesh = revolve_profile(hub_profile, N_ANG)

    # Bore through center (represented as a visible inner ring)
    bore_profile = [(bore_r - 0.5, -1), (bore_r, -1),
                    (bore_r, thickness + hub_h + 1), (bore_r - 0.5, thickness + hub_h + 1)]
    bore_mesh = revolve_profile(bore_profile, N_ANG)

    # Chamfers on teeth edges (decorative ring)
    chamfer_ring = make_torus(module * n_teeth / 2 - module * 0.5, 0.8, thickness, N_ANG, 8)

    return combine_meshes([gear_body, hub_mesh, bore_mesh, chamfer_ring])

_register("spur_gear", "Involute spur gear with hub (GrabCAD-style)", _gear_simple, _gear_ornate)


# ============================================================================
# 2. Pipe Flange (GrabCAD: flange tag)
# ============================================================================

def _flange_simple():
    """Simple flange: flat ring."""
    inner_r = 20
    outer_r = 45
    thickness = 10
    profile = [(inner_r, 0), (outer_r, 0), (outer_r, thickness), (inner_r, thickness)]
    return revolve_profile(profile, N_ANG)


def _flange_ornate():
    """Ornate flange: raised face + bolt holes + pipe neck."""
    inner_r = 20
    outer_r = 45
    bolt_circle_r = 35
    bolt_r = 4
    n_bolts = 8
    thickness = 10
    neck_h = 20
    raised_face_r = 30

    # Main flange body with neck
    profile = smooth_profile([
        (inner_r, -neck_h),
        (inner_r + 3, -neck_h), (inner_r + 3, -2),  # pipe neck
        (outer_r, -2), (outer_r, 0),                  # flange bottom
        (outer_r, thickness),                           # flange top
        (raised_face_r + 2, thickness),                # step down to raised face
        (raised_face_r + 2, thickness + 2),            # raised face
        (inner_r, thickness + 2), (inner_r, -neck_h),
    ], n_output=30)
    body = revolve_profile(profile, N_ANG)

    # Bolt holes (as cylinders placed at bolt positions)
    bolt_meshes = []
    for i in range(n_bolts):
        angle = 2 * math.pi * i / n_bolts
        bx = bolt_circle_r * math.cos(angle)
        by = bolt_circle_r * math.sin(angle)
        # Ring around each bolt hole
        bolt_ring = make_torus(bolt_r + 1, 0.8, thickness, 12, 6)
        bv, bf = bolt_ring
        bv[:, 0] += bx
        bv[:, 1] += by
        bolt_meshes.append((bv, bf))

    # Gasket groove ring on raised face
    gasket_groove = make_torus(raised_face_r - 2, 0.5, thickness + 2, N_ANG, 8)

    return combine_meshes([body] + bolt_meshes + [gasket_groove])

_register("pipe_flange", "Pipe flange with bolt holes (GrabCAD-style)", _flange_simple, _flange_ornate)


# ============================================================================
# 3. Shelf Bracket (GrabCAD: shelf bracket collection)
# ============================================================================

def _bracket_simple():
    """Simple L-shaped bracket."""
    t = 5  # thickness
    arm_h = 80
    arm_w = 60

    # L profile (extruded in Z)
    poly = [
        (0, 0), (arm_w, 0), (arm_w, t),
        (t, t), (t, arm_h), (0, arm_h),
    ]
    return extrude_polygon(poly, 30)


def _bracket_ornate():
    """Ornate bracket: L-shape + diagonal gusset + scalloped edge + mounting holes."""
    t = 5
    arm_h = 80
    arm_w = 60
    depth = 30

    # L profile with curved gusset (more points for the diagonal)
    n_curve = 12
    poly = [(0, 0), (arm_w, 0), (arm_w, t)]
    # Curved gusset from (t, t) arcing out to (t, arm_h)
    for i in range(n_curve + 1):
        frac = i / n_curve
        angle = math.pi / 2 * frac  # 0 to 90 degrees
        gx = t + (arm_w - t - 5) * math.cos(angle)
        gy = t + (arm_h - t - 5) * math.sin(angle)
        poly.append((gx, gy))
    poly.extend([(t, arm_h), (0, arm_h)])

    body = extrude_polygon(poly, depth)

    # Decorative scrollwork holes (circles approximated by polygons)
    holes = []
    scroll_centers = [(25, 20), (15, 35), (25, 45)]
    for cx, cy in scroll_centers:
        # Create a ring at each hole position
        ring = make_torus(5, 1.0, depth / 2, 12, 6)
        rv, rf = ring
        # Rotate ring to face the bracket face (lies in XY plane)
        # The ring is around Z axis; we want it around X axis at (cx, cy)
        rv_rotated = rv.copy()
        rv_rotated[:, 0] = rv[:, 0] + cx  # shift X
        rv_rotated[:, 1] = cy             # fix Y
        rv_rotated[:, 2] = rv[:, 1]       # swap Y->Z (ring now in XZ plane)
        holes.append((rv_rotated, rf))

    # Mounting hole rings on the wall side
    for mz in [8, 22]:
        ring = make_torus(3, 0.8, 0, 12, 6)
        rv, rf = ring
        rv_rot = rv.copy()
        rv_rot[:, 2] = rv[:, 1] + mz
        rv_rot[:, 1] = rv[:, 0] + arm_h - 10
        rv_rot[:, 0] = 0
        holes.append((rv_rot, rf))

    return combine_meshes([body] + holes)

_register("shelf_bracket", "Shelf bracket with gusset and scrollwork (GrabCAD-style)",
          _bracket_simple, _bracket_ornate)


# ============================================================================
# 4. Hex Nut (engineering standard part)
# ============================================================================

def _hexnut_simple():
    """Simple hex nut: plain cylinder (no hex shape, no details)."""
    hex_r = 12
    height = 10
    profile = [(3, 0), (hex_r, 0), (hex_r, height), (3, height)]
    return revolve_profile(profile, N_ANG)


def _hexnut_ornate():
    """Ornate hex nut: hex prism + chamfered edges + thread detail + washer face."""
    hex_r = 12
    height = 10
    bore_r = 5
    washer_face_r = 10

    # Main hex body
    hex_body = extrude_polygon(
        make_regular_polygon(6, hex_r, start_angle=math.pi / 6), height
    )

    # Chamfer rings (top and bottom)
    top_chamfer = make_torus(hex_r * 0.87, 1.0, height, 6, 8)
    bot_chamfer = make_torus(hex_r * 0.87, 1.0, 0, 6, 8)

    # Bore detail ring
    bore_ring_top = make_torus(bore_r + 1, 0.5, height, N_ANG, 6)
    bore_ring_bot = make_torus(bore_r + 1, 0.5, 0, N_ANG, 6)

    # Washer face
    wf_profile = [(washer_face_r - 1, -0.5), (washer_face_r, -0.5),
                  (washer_face_r, 0.5), (washer_face_r - 1, 0.5)]
    washer_face = revolve_profile(wf_profile, N_ANG)

    return combine_meshes([hex_body, top_chamfer, bot_chamfer,
                           bore_ring_top, bore_ring_bot, washer_face])

_register("hex_nut", "Hex nut with chamfers and thread detail", _hexnut_simple, _hexnut_ornate)


# ============================================================================
# 5. Picture Frame (decorative, rectangular sweep)
# ============================================================================

def _frame_simple():
    """Simple picture frame: rectangular frame, flat profile."""
    return make_rectangular_frame(120, 90, 15, 12)


def _frame_ornate():
    """Ornate picture frame: molded profile with step and ogee."""
    profile = [
        (0, 0), (3, 0), (5, 2), (8, 3),  # step up
        (10, 5), (11, 8), (12, 12),       # ogee curve
        (12, 15), (10, 15), (8, 13),      # lip
        (5, 12), (3, 10), (0, 8),         # inner slope
    ]
    body = make_rectangular_frame(120, 90, 15, 12, profile_pts=profile)

    # Corner rosettes (small torus at each corner)
    hw, hh = 60, 45
    corners = [(-hw, -hh), (hw, -hh), (hw, hh), (-hw, hh)]
    rosettes = []
    for cx, cy in corners:
        r = make_torus(4, 1.5, 7.5, 12, 8)
        rv, rf = r
        rv[:, 0] += cx
        rv[:, 1] += cy
        rosettes.append((rv, rf))

    return combine_meshes([body] + rosettes)

_register("picture_frame", "Picture frame with ogee molding and corner rosettes",
          _frame_simple, _frame_ornate)


# ============================================================================
# 6. Star Knob (hand-tightening knob)
# ============================================================================

def _starknob_simple():
    """Simple star knob: plain cylinder."""
    r = 20
    h = 15
    profile = [(0.1, 0), (r, 0), (r, h), (0.1, h)]
    return revolve_profile(profile, N_ANG)


def _starknob_ornate():
    """Ornate star knob: 5-point star + central bore + knurled grip."""
    outer_r = 22
    inner_r = 14
    n_star = 5
    h = 15
    bore_r = 4

    # Star-shaped body
    star_poly = make_star_polygon(n_star, outer_r, inner_r)
    star_body = extrude_polygon(star_poly, h)

    # Top dome (shallow revolve on top)
    dome_profile = smooth_profile([
        (0.1, h), (inner_r * 0.8, h + 0.5),
        (inner_r * 0.5, h + 2), (3, h + 3),
        (0.1, h + 3.5),
    ], n_output=15)
    dome = revolve_profile(dome_profile, N_ANG, close_bottom=False)

    # Bore ring detail
    bore_ring = make_torus(bore_r + 2, 0.8, 0, N_ANG, 8)
    bore_ring_top = make_torus(bore_r + 2, 0.8, h, N_ANG, 8)

    # Edge ring around star
    edge_ring = make_torus(outer_r - 2, 1.0, h / 2, N_ANG, 10)

    return combine_meshes([star_body, dome, bore_ring, bore_ring_top, edge_ring])

_register("star_knob", "Star-shaped hand knob with bore", _starknob_simple, _starknob_ornate)


# ============================================================================
# 7. Fluted Column (revolve + longitudinal channels)
# ============================================================================

def _fluted_col_simple():
    """Simple column: plain cylinder with base and capital."""
    h = 160
    r = 12
    profile = smooth_profile([
        (r + 4, 0), (r + 4, 8),    # base
        (r, 12), (r, h - 12),      # shaft
        (r + 4, h - 8), (r + 4, h), # capital
    ])
    return revolve_profile(profile, N_ANG)


def _fluted_col_ornate():
    """Ornate fluted column: shaft with 20 concave flutes + torus moldings."""
    h = 160
    r = 12
    n_flutes = 20
    flute_depth = 2.0

    # Build shaft with flutes by modifying the angular profile
    shaft_z_bottom = 14
    shaft_z_top = h - 14
    n_z = 40

    verts = []
    faces = []

    # Build the shaft ring-by-ring with fluted radius
    for iz in range(n_z + 1):
        z = shaft_z_bottom + (shaft_z_top - shaft_z_bottom) * iz / n_z
        for ia in range(N_ANG):
            angle = 2 * math.pi * ia / N_ANG
            # Flute pattern: sinusoidal variation in radius
            flute_phase = n_flutes * angle
            flute = flute_depth * (math.cos(flute_phase) - 1) / 2  # always <= 0
            # Entasis
            z_norm = (z - shaft_z_bottom) / (shaft_z_top - shaft_z_bottom)
            entasis = 0.5 * math.sin(math.pi * z_norm) ** 0.4
            current_r = r + entasis + flute
            verts.append([current_r * math.cos(angle),
                          current_r * math.sin(angle), z])

    verts = np.array(verts, dtype=np.float64)

    for iz in range(n_z):
        for ia in range(N_ANG):
            ia_next = (ia + 1) % N_ANG
            p00 = iz * N_ANG + ia
            p01 = iz * N_ANG + ia_next
            p10 = (iz + 1) * N_ANG + ia
            p11 = (iz + 1) * N_ANG + ia_next
            faces.append([p00, p01, p10])
            faces.append([p01, p11, p10])

    shaft_mesh = (verts, np.array(faces))

    # Base and capital (plain revolve)
    base_profile = smooth_profile([
        (r + 6, 0), (r + 7, 2), (r + 6, 4), (r + 4, 6),
        (r + 2, 8), (r, 10), (r, shaft_z_bottom),
    ], n_output=20)
    base = revolve_profile(base_profile, N_ANG)

    cap_profile = smooth_profile([
        (r, shaft_z_top), (r, h - 10),
        (r + 2, h - 8), (r + 4, h - 6),
        (r + 6, h - 4), (r + 7, h - 2), (r + 6, h),
    ], n_output=20)
    capital = revolve_profile(cap_profile, N_ANG)

    # Molding rings
    base_ring = make_torus(r + 6, 1.0, 2, N_ANG, 8)
    cap_ring = make_torus(r + 6, 1.0, h - 2, N_ANG, 8)
    astragal = make_torus(r + 1.5, 1.0, shaft_z_bottom - 1, N_ANG, 8)

    return combine_meshes([shaft_mesh, base, capital, base_ring, cap_ring, astragal])

_register("fluted_column", "Column with longitudinal flutes (Greek/Roman style)",
          _fluted_col_simple, _fluted_col_ornate)


# ============================================================================
# 8. Castellated Ring / Crown Nut
# ============================================================================

def _castle_ring_simple():
    """Simple ring: plain torus/tube shape."""
    inner_r = 15
    outer_r = 25
    h = 12
    profile = [(inner_r, 0), (outer_r, 0), (outer_r, h), (inner_r, h)]
    return revolve_profile(profile, N_ANG)


def _castle_ring_ornate():
    """Castellated ring: ring with rectangular notches cut from the top."""
    inner_r = 15
    outer_r = 25
    h = 12
    n_castles = 12
    castle_h = 5
    castle_width_frac = 0.4  # fraction of each bay that's a notch

    n_z = 3
    ring_h = h - castle_h

    # Lower ring (plain)
    verts = []
    faces = []

    # Build as a tube with castellation on top ring
    n_z_total = 8
    for iz in range(n_z_total + 1):
        z = h * iz / n_z_total
        for ia in range(N_ANG):
            angle = 2 * math.pi * ia / N_ANG

            # Determine if this angle is in a castle (up) or notch (down)
            bay_angle = 2 * math.pi / n_castles
            phase = (angle % bay_angle) / bay_angle
            in_notch = phase < castle_width_frac

            if z > ring_h and in_notch:
                # In a notch region above the ring: push radius inward
                r = inner_r + 2  # thin wall at notch bottom
            else:
                r = outer_r

            verts.append([r * math.cos(angle), r * math.sin(angle), z])

    verts = np.array(verts, dtype=np.float64)

    for iz in range(n_z_total):
        for ia in range(N_ANG):
            ia_next = (ia + 1) % N_ANG
            p00 = iz * N_ANG + ia
            p01 = iz * N_ANG + ia_next
            p10 = (iz + 1) * N_ANG + ia
            p11 = (iz + 1) * N_ANG + ia_next
            faces.append([p00, p01, p10])
            faces.append([p01, p11, p10])

    outer_mesh = (verts, np.array(faces))

    # Inner bore surface
    bore_profile = [(inner_r, 0), (inner_r, h)]
    bore = revolve_profile(bore_profile, N_ANG, close_top=False, close_bottom=False)

    # Decorative ring at base
    base_ring = make_torus(outer_r, 1.0, 0, N_ANG, 8)

    return combine_meshes([outer_mesh, bore, base_ring])

_register("castellated_ring", "Castellated ring/crown nut with notch pattern",
          _castle_ring_simple, _castle_ring_ornate)


# ============================================================================
# 9. Gear Shift Knob (revolve + flat cuts + knurling)
# ============================================================================

def _shiftknob_simple():
    """Simple shift knob: plain spheroid on a stem."""
    profile = smooth_profile([
        (4, 0), (5, 5),             # stem
        (5, 15), (8, 20),
        (14, 28), (16, 36),         # bulb
        (14, 44), (8, 48),
        (3, 50), (0.5, 52),
    ])
    return revolve_profile(profile, N_ANG)


def _shiftknob_ornate():
    """Ornate shift knob: shaped body + knurled grip + flat indicator + trim ring."""
    # Main body profile
    profile = smooth_profile([
        (4, 0), (5, 2), (4, 4),     # stem collar
        (5, 6), (6, 10),
        (5, 14), (8, 18),           # grip transition
        (12, 22), (14, 26),
        (16, 32), (17, 38),         # main body
        (16, 42), (13, 46),
        (8, 50), (4, 52),
        (2, 53), (0.5, 54),         # top button
    ], n_output=60)
    body_v, body_f = revolve_profile(profile, N_ANG)

    # Apply knurling to the grip section (z=8 to z=18)
    body_v = make_knurled_surface(body_v, 5.5, 8, 18, n_knurls=16, knurl_depth=0.4)

    # Trim rings
    trim1 = make_torus(8, 0.8, 18, N_ANG, 8)
    trim2 = make_torus(14, 0.6, 26, N_ANG, 8)
    trim3 = make_torus(14, 0.6, 46, N_ANG, 8)

    # Top button detail
    button = make_torus(3, 0.5, 53, N_ANG, 6)

    return combine_meshes([(body_v, body_f), trim1, trim2, trim3, button])

_register("gear_shift_knob", "Gear shift knob with knurled grip",
          _shiftknob_simple, _shiftknob_ornate)


# ============================================================================
# 10. Lattice Panel (grid of intersecting bars)
# ============================================================================

def _lattice_simple():
    """Simple lattice: solid rectangular panel."""
    w, h, d = 80, 80, 5
    poly = [(-w/2, -h/2), (w/2, -h/2), (w/2, h/2), (-w/2, h/2)]
    return extrude_polygon(poly, d)


def _lattice_ornate():
    """Ornate lattice: diamond pattern of intersecting bars + border frame."""
    w, h, d = 80, 80, 5
    bar_r = 1.5
    n_bars = 7  # number of bars in each diagonal direction
    spacing = w / (n_bars + 1)

    meshes = []

    # Diagonal bars (both directions)
    n_seg = 20
    for direction in [1, -1]:
        for i in range(n_bars + 2):
            offset = -w / 2 + spacing * i
            bar_verts = []
            bar_faces = []

            for j in range(n_seg + 1):
                frac = j / n_seg
                x = -w / 2 + w * frac
                y = offset + direction * w * frac
                y = max(-h / 2, min(h / 2, y))

                for k in range(8):
                    angle = 2 * math.pi * k / 8
                    bx = x + bar_r * math.cos(angle)
                    bz = d / 2 + bar_r * math.sin(angle)
                    bar_verts.append([bx, y, bz])

            bar_verts = np.array(bar_verts, dtype=np.float64)

            for j in range(n_seg):
                for k in range(8):
                    k_next = (k + 1) % 8
                    p00 = j * 8 + k
                    p01 = j * 8 + k_next
                    p10 = (j + 1) * 8 + k
                    p11 = (j + 1) * 8 + k_next
                    bar_faces.append([p00, p01, p10])
                    bar_faces.append([p01, p11, p10])

            meshes.append((bar_verts, np.array(bar_faces)))

    # Border frame
    frame_t = 4
    border_poly = [
        (-w/2, -h/2), (w/2, -h/2), (w/2, h/2), (-w/2, h/2),
        (-w/2, h/2 - frame_t), (w/2 - frame_t, h/2 - frame_t),
        (w/2 - frame_t, -h/2 + frame_t), (-w/2 + frame_t, -h/2 + frame_t),
        (-w/2 + frame_t, h/2 - frame_t), (-w/2, h/2 - frame_t),
    ]
    # Simplified: just use 4 bars as the frame border
    for sx, sy, ex, ey in [
        (-w/2, -h/2, w/2, -h/2),   # bottom
        (w/2, -h/2, w/2, h/2),     # right
        (w/2, h/2, -w/2, h/2),     # top
        (-w/2, h/2, -w/2, -h/2),   # left
    ]:
        fv = []
        ff = []
        bar_w = frame_t
        for j in range(n_seg + 1):
            frac = j / n_seg
            px = sx + (ex - sx) * frac
            py = sy + (ey - sy) * frac
            # Frame cross section
            dx = -(ey - sy)
            dy = (ex - sx)
            length = math.sqrt(dx*dx + dy*dy)
            if length > 0:
                dx, dy = dx / length * bar_w, dy / length * bar_w
            for dz in [0, d]:
                for side in [0, 1]:
                    fx = px + dx * side
                    fy = py + dy * side
                    fv.append([fx, fy, dz])

        fv = np.array(fv, dtype=np.float64)
        n_cross = 4  # 2 depths * 2 sides
        for j in range(n_seg):
            for k in range(n_cross):
                k_next = (k + 1) % n_cross
                p00 = j * n_cross + k
                p01 = j * n_cross + k_next
                p10 = (j + 1) * n_cross + k
                p11 = (j + 1) * n_cross + k_next
                ff.append([p00, p01, p10])
                ff.append([p01, p11, p10])

        meshes.append((fv, np.array(ff)))

    # Decorative rosette at center
    rosette = make_torus(6, 2.0, d / 2, N_ANG, 10)
    meshes.append(rosette)

    return combine_meshes(meshes)

_register("lattice_panel", "Lattice panel with diamond pattern and border",
          _lattice_simple, _lattice_ornate)


# ============================================================================
# 11. Pulley / Sheave (grooved wheel for belt drive)
# ============================================================================

def _pulley_simple():
    """Simple pulley: plain solid cylinder (no bore, no groove, no spokes)."""
    outer_r = 28
    thickness = 8
    profile = [(0.5, 0), (outer_r, 0), (outer_r, thickness), (0.5, thickness)]
    return revolve_profile(profile, N_ANG)


def _pulley_ornate():
    """Ornate pulley: V-groove rim + spoked hub + keyway detail."""
    outer_r = 30
    bore_r = 6
    hub_r = 12
    thickness = 10
    groove_depth = 4
    groove_angle_half = 2.5  # half-width of the V-groove at rim

    # Rim profile with V-groove
    rim_profile = smooth_profile([
        (outer_r - 2, 0), (outer_r, 0),                         # bottom lip
        (outer_r, groove_angle_half - 0.5),                       # outer wall
        (outer_r - groove_depth, thickness / 2),                  # groove bottom
        (outer_r, thickness - groove_angle_half + 0.5),           # outer wall
        (outer_r, thickness), (outer_r - 2, thickness),           # top lip
    ], n_output=25)
    rim = revolve_profile(rim_profile, N_ANG, close_top=False, close_bottom=False)

    # Hub
    hub_profile = smooth_profile([
        (bore_r, 0), (hub_r, 0),
        (hub_r + 1, 1), (hub_r + 1, thickness - 1),
        (hub_r, thickness), (bore_r, thickness),
    ], n_output=18)
    hub = revolve_profile(hub_profile, N_ANG)

    # Spokes connecting hub to rim (4 flat spokes)
    spokes = []
    spoke_w = 3
    for i in range(4):
        angle = 2 * math.pi * i / 4
        cos_a, sin_a = math.cos(angle), math.sin(angle)
        # Each spoke is a short extruded bar from hub_r to outer_r-2
        sv = []
        sf = []
        n_seg = 8
        for j in range(n_seg + 1):
            frac = j / n_seg
            r = hub_r + (outer_r - 2 - hub_r) * frac
            cx = r * cos_a
            cy = r * sin_a
            # Cross-section: small rectangle perpendicular to radial direction
            nx, ny = -sin_a, cos_a  # tangent direction
            for dz in [thickness * 0.35, thickness * 0.65]:
                for side in [-1, 1]:
                    sv.append([cx + side * nx * spoke_w / 2,
                               cy + side * ny * spoke_w / 2, dz])
        sv = np.array(sv, dtype=np.float64)
        n_cross = 4
        for j in range(n_seg):
            for k in range(n_cross):
                k_next = (k + 1) % n_cross
                p00 = j * n_cross + k
                p01 = j * n_cross + k_next
                p10 = (j + 1) * n_cross + k
                p11 = (j + 1) * n_cross + k_next
                sf.append([p00, p01, p10])
                sf.append([p01, p11, p10])
        spokes.append((sv, np.array(sf)))

    # Keyway detail ring on bore
    keyway_ring = make_torus(bore_r + 1.5, 0.6, thickness / 2, N_ANG, 6)

    # Groove guide rings at rim edges
    guide_top = make_torus(outer_r - 1, 0.5, thickness, N_ANG, 6)
    guide_bot = make_torus(outer_r - 1, 0.5, 0, N_ANG, 6)

    return combine_meshes([rim, hub] + spokes + [keyway_ring, guide_top, guide_bot])

_register("pulley_sheave", "V-groove pulley with spoked hub (GrabCAD-style)",
          _pulley_simple, _pulley_ornate)


# ============================================================================
# 12. Heat Sink (base plate + fin array)
# ============================================================================

def _heatsink_simple():
    """Simple heat sink: plain rectangular block."""
    w, d, h = 60, 40, 20
    poly = [(-w / 2, -d / 2), (w / 2, -d / 2), (w / 2, d / 2), (-w / 2, d / 2)]
    return extrude_polygon(poly, h)


def _heatsink_ornate():
    """Ornate heat sink: base plate + array of rectangular fins + mounting holes."""
    w, d = 60, 40
    base_h = 5
    fin_h = 18
    n_fins = 9
    fin_t = 1.5
    total_h = base_h + fin_h

    meshes = []

    # Base plate
    base_poly = [(-w / 2, -d / 2), (w / 2, -d / 2), (w / 2, d / 2), (-w / 2, d / 2)]
    base = extrude_polygon(base_poly, base_h)
    meshes.append(base)

    # Fins along X direction, spaced evenly along Y
    fin_spacing = (d - 4) / (n_fins - 1)
    for i in range(n_fins):
        y_center = -d / 2 + 2 + fin_spacing * i
        fin_poly = [
            (-w / 2 + 2, y_center - fin_t / 2),
            (w / 2 - 2, y_center - fin_t / 2),
            (w / 2 - 2, y_center + fin_t / 2),
            (-w / 2 + 2, y_center + fin_t / 2),
        ]
        fv, ff = extrude_polygon(fin_poly, fin_h)
        # Shift fins up to sit on base plate
        fv[:, 2] += base_h
        meshes.append((fv, ff))

    # Mounting hole rings (4 corners of base)
    for sx, sy in [(-1, -1), (-1, 1), (1, -1), (1, 1)]:
        hx = sx * (w / 2 - 5)
        hy = sy * (d / 2 - 5)
        ring = make_torus(2.5, 0.6, 0, 12, 6)
        rv, rf = ring
        rv_shifted = rv.copy()
        rv_shifted[:, 0] += hx
        rv_shifted[:, 1] += hy
        meshes.append((rv_shifted, rf))

    # Top edge chamfer rings on outermost fins
    for y_edge in [-d / 2 + 2, d / 2 - 2]:
        edge_ring = make_torus(w / 2 - 2, 0.4, total_h, 24, 6)
        rv, rf = edge_ring
        rv_shifted = rv.copy()
        rv_shifted[:, 1] += y_edge
        meshes.append((rv_shifted, rf))

    return combine_meshes(meshes)

_register("heat_sink", "Heat sink with rectangular fin array and mounting holes",
          _heatsink_simple, _heatsink_ornate)


# ============================================================================
# 13. Cam Disc (eccentric cam with follower track)
# ============================================================================

def _cam_simple():
    """Simple cam: plain circular disc."""
    r = 25
    h = 10
    bore_r = 5
    profile = [(bore_r, 0), (r, 0), (r, h), (bore_r, h)]
    return revolve_profile(profile, N_ANG)


def _cam_ornate():
    """Ornate cam: eccentric cam profile + hub + keyway + follower track."""
    base_r = 20
    max_lift = 12
    h = 10
    bore_r = 5
    hub_r = 10
    hub_extend = 4

    # Cam profile: radius varies with angle (single-lobe)
    n_ang_cam = 72
    cam_verts = []
    cam_faces = []

    for iz in range(3):  # bottom, mid, top
        z = h * iz / 2
        for ia in range(n_ang_cam):
            angle = 2 * math.pi * ia / n_ang_cam
            # Single harmonic cam: r = base_r + lift * (1 + cos(angle)) / 2
            lift = max_lift * (1 + math.cos(angle)) / 2
            r = base_r + lift
            cam_verts.append([r * math.cos(angle), r * math.sin(angle), z])

    cam_verts = np.array(cam_verts, dtype=np.float64)

    for iz in range(2):
        for ia in range(n_ang_cam):
            ia_next = (ia + 1) % n_ang_cam
            p00 = iz * n_ang_cam + ia
            p01 = iz * n_ang_cam + ia_next
            p10 = (iz + 1) * n_ang_cam + ia
            p11 = (iz + 1) * n_ang_cam + ia_next
            cam_faces.append([p00, p01, p10])
            cam_faces.append([p01, p11, p10])

    cam_body = (cam_verts, np.array(cam_faces))

    # Hub (extended boss on one side)
    hub_profile = smooth_profile([
        (bore_r, -hub_extend), (hub_r, -hub_extend),
        (hub_r + 1, -hub_extend + 1), (hub_r + 1, h + hub_extend - 1),
        (hub_r, h + hub_extend), (bore_r, h + hub_extend),
    ], n_output=20)
    hub = revolve_profile(hub_profile, N_ANG)

    # Bore detail
    bore_ring_top = make_torus(bore_r + 1.5, 0.5, h + hub_extend, N_ANG, 6)
    bore_ring_bot = make_torus(bore_r + 1.5, 0.5, -hub_extend, N_ANG, 6)

    # Follower track (raised rail on the cam face)
    track = make_torus(base_r + max_lift / 2, 0.8, h, N_ANG, 8)

    # Keyway indicator
    keyway = make_torus(hub_r - 1, 0.4, -hub_extend + 0.5, N_ANG, 6)

    return combine_meshes([cam_body, hub, bore_ring_top, bore_ring_bot, track, keyway])

_register("cam_disc", "Eccentric cam disc with hub and follower track",
          _cam_simple, _cam_ornate)


# ============================================================================
# 14. Hinge Plate (flat plate + barrel hinge + screw holes)
# ============================================================================

def _hinge_simple():
    """Simple hinge: plain thick rectangular plate (no barrel, no holes)."""
    w, d, t = 45, 25, 5
    poly = [(2, 2), (w - 2, 2), (w - 2, d - 2), (2, d - 2)]
    return extrude_polygon(poly, t)


def _hinge_ornate():
    """Ornate hinge: plate + barrel hinge + screw holes + rounded corners."""
    w, d, t = 50, 30, 3
    barrel_r = 4
    n_barrels = 3

    meshes = []

    # Main plate with slight inset around edges
    poly = [(0, 0), (w, 0), (w, d), (0, d)]
    plate = extrude_polygon(poly, t)
    meshes.append(plate)

    # Barrel hinge along the Y=0 edge (series of cylinders)
    barrel_len = d / (n_barrels * 2 - 1)
    for i in range(n_barrels):
        y_start = i * 2 * barrel_len
        y_end = y_start + barrel_len
        # Barrel as a horizontal cylinder along Y
        bv = []
        bf = []
        n_seg = 10
        n_circ = 16
        for iy in range(n_seg + 1):
            y = y_start + (y_end - y_start) * iy / n_seg
            for ic in range(n_circ):
                angle = 2 * math.pi * ic / n_circ
                bx = 0 + barrel_r * math.cos(angle)
                bz = t / 2 + barrel_r * math.sin(angle)
                bv.append([bx, y, bz])
        bv = np.array(bv, dtype=np.float64)
        for iy in range(n_seg):
            for ic in range(n_circ):
                ic_next = (ic + 1) % n_circ
                p00 = iy * n_circ + ic
                p01 = iy * n_circ + ic_next
                p10 = (iy + 1) * n_circ + ic
                p11 = (iy + 1) * n_circ + ic_next
                bf.append([p00, p01, p10])
                bf.append([p01, p11, p10])
        meshes.append((bv, np.array(bf)))

    # Hinge pin (thin cylinder through all barrels)
    pin_profile = [(barrel_r * 0.25, -1), (barrel_r * 0.25, d + 1)]
    pin = revolve_profile(pin_profile, 12, close_top=True, close_bottom=True)
    pv, pf = pin
    pv_shifted = pv.copy()
    # Pin runs along Y, but revolve_profile creates along Z — rotate
    pv_rotated = pv_shifted.copy()
    pv_rotated[:, 0] = pv_shifted[:, 0]  # X stays
    pv_rotated[:, 1] = pv_shifted[:, 2]  # Z -> Y
    pv_rotated[:, 2] = pv_shifted[:, 1] + t / 2  # Y -> Z, shift up
    meshes.append((pv_rotated, pf))

    # Screw hole rings (countersunk pattern)
    hole_positions = [(15, 10), (35, 10), (15, 20), (35, 20)]
    for hx, hy in hole_positions:
        ring = make_torus(2.5, 0.6, t, 12, 6)
        rv, rf = ring
        rv_shifted = rv.copy()
        rv_shifted[:, 0] += hx
        rv_shifted[:, 1] += hy
        meshes.append((rv_shifted, rf))
        # Countersink ring (slightly larger, on bottom)
        csink = make_torus(3.5, 0.4, 0, 12, 6)
        cv, cf = csink
        cv_shifted = cv.copy()
        cv_shifted[:, 0] += hx
        cv_shifted[:, 1] += hy
        meshes.append((cv_shifted, cf))

    # Edge bead along long edges
    edge_top = make_torus(w / 2, 0.3, t, 24, 6)
    ev, ef = edge_top
    ev[:, 1] += d
    meshes.append((ev, ef))

    return combine_meshes(meshes)

_register("hinge_plate", "Hinge plate with barrel and screw holes",
          _hinge_simple, _hinge_ornate)


# ============================================================================
# 15. Hex Bolt (hex head + shank + thread detail)
# ============================================================================

def _hexbolt_simple():
    """Simple hex bolt: plain cylinder (shank only)."""
    shank_r = 5
    head_r = 10
    total_h = 50
    head_h = 7
    profile = [
        (0.1, 0), (shank_r, 0), (shank_r, total_h - head_h),
        (head_r, total_h - head_h), (head_r, total_h), (0.1, total_h),
    ]
    return revolve_profile(profile, N_ANG)


def _hexbolt_ornate():
    """Ornate hex bolt: hex head + chamfered edges + shank + thread rings."""
    shank_r = 5
    head_r = 10
    total_h = 50
    head_h = 7
    thread_start = 0
    thread_end = total_h - head_h - 2  # leave unthreaded section near head

    # Hex head
    hex_body = extrude_polygon(
        make_regular_polygon(6, head_r, start_angle=math.pi / 6),
        head_h,
    )
    # Shift hex head to top
    hv, hf = hex_body
    hv_shifted = hv.copy()
    hv_shifted[:, 2] += total_h - head_h
    hex_mesh = (hv_shifted, hf)

    # Head chamfers
    top_chamfer = make_torus(head_r * 0.87, 0.8, total_h, 6, 8)
    bot_chamfer = make_torus(head_r * 0.87, 0.8, total_h - head_h, 6, 8)

    # Shank (plain cylinder)
    shank_profile = [
        (shank_r, thread_end), (shank_r, total_h - head_h),
    ]
    shank = revolve_profile(shank_profile, N_ANG, close_top=False, close_bottom=True)

    # Thread section: helical ridges approximated as series of torus rings
    thread_rings = []
    thread_pitch = 2.0
    n_threads = int((thread_end - thread_start) / thread_pitch)
    for i in range(n_threads):
        z = thread_start + thread_pitch * (i + 0.5)
        ring = make_torus(shank_r + 0.3, 0.35, z, N_ANG, 6)
        thread_rings.append(ring)

    # Thread body (cylinder underneath the rings)
    thread_body_profile = [(shank_r - 0.2, thread_start), (shank_r, thread_start),
                           (shank_r, thread_end), (shank_r - 0.2, thread_end)]
    thread_body = revolve_profile(thread_body_profile, N_ANG)

    # Tip chamfer
    tip_cone = make_torus(shank_r * 0.6, 0.5, 0, N_ANG, 6)

    # Washer face ring under head
    washer_ring = make_torus(shank_r + 2, 0.4, total_h - head_h, N_ANG, 6)

    return combine_meshes([hex_mesh, top_chamfer, bot_chamfer, shank,
                           thread_body, tip_cone, washer_ring] + thread_rings)

_register("hex_bolt", "Hex bolt with thread detail and chamfered head",
          _hexbolt_simple, _hexbolt_ornate)


# ============================================================================
# Public API
# ============================================================================

def list_complex_objects():
    return list(COMPLEX_CATALOG.keys())


def get_complex_object(name):
    return COMPLEX_CATALOG[name]


def make_complex_simple(name):
    return COMPLEX_CATALOG[name]["simple"]()


def make_complex_ornate(name):
    return COMPLEX_CATALOG[name]["ornate"]()
