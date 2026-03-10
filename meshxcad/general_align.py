"""General mesh-to-mesh alignment: 12 features that are neither extrusion-
nor revolve-specific.

These operate on raw triangle meshes and improve the fit of a CAD mesh to a
target mesh through surface-level analysis, deformation, and correction.

1.  surface_distance_map       — Per-vertex signed distance from CAD to mesh.
2.  hausdorff_distance         — Symmetric Hausdorff + mean surface distance.
3.  normal_deviation_map       — Per-vertex angle between CAD and mesh normals.
4.  curvature_deviation_map    — Gaussian-curvature difference per vertex.
5.  laplacian_smooth_toward    — Laplacian smoothing biased toward a target.
6.  project_vertices_to_mesh   — Snap each CAD vertex to the nearest mesh
                                  surface point.
7.  local_scale_correction     — Per-region scale fix (shrink/grow patches
                                  that are too large/small).
8.  feature_edge_transfer      — Detect sharp edges in the mesh and sharpen
                                  corresponding CAD edges.
9.  vertex_normal_realign      — Rotate vertex normals to match mesh normals
                                  and adjust vertex positions accordingly.
10. fill_surface_holes         — Detect and close holes in the CAD mesh.
11. decimate_to_match          — Reduce CAD triangle count to match mesh
                                  density, preserving shape.
12. suggest_general_adjustments — Analyse CAD vs mesh and return ranked
                                  adjustment suggestions.
"""

import math
import numpy as np
from scipy.spatial import KDTree

from .gpu import (AcceleratedKDTree as _AKDTree,
                  hausdorff_distance_gpu as _hausdorff_gpu,
                  is_gpu_available as _gpu_ok,
                  compute_vertex_normals as _gpu_vertex_normals,
                  compute_face_normals as _gpu_face_normals,
                  row_norms as _gpu_row_norms,
                  build_vertex_adjacency_matrix as _gpu_build_adj)


# =========================================================================
# Helpers
# =========================================================================

def _compute_vertex_normals(vertices, faces):
    """Compute area-weighted per-vertex normals."""
    verts = np.asarray(vertices, dtype=np.float64)
    tris = np.asarray(faces)
    normals = np.zeros_like(verts)

    v0, v1, v2 = verts[tris[:, 0]], verts[tris[:, 1]], verts[tris[:, 2]]
    face_normals = np.cross(v1 - v0, v2 - v0)

    for i in range(3):
        np.add.at(normals, tris[:, i], face_normals)

    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    norms[norms < 1e-12] = 1.0
    return normals / norms


def _compute_face_normals(vertices, faces):
    """Unit face normals."""
    v0, v1, v2 = vertices[faces[:, 0]], vertices[faces[:, 1]], vertices[faces[:, 2]]
    fn = np.cross(v1 - v0, v2 - v0)
    norms = np.linalg.norm(fn, axis=1, keepdims=True)
    norms[norms < 1e-12] = 1.0
    return fn / norms


def _face_areas(vertices, faces):
    """Per-face area."""
    v0, v1, v2 = vertices[faces[:, 0]], vertices[faces[:, 1]], vertices[faces[:, 2]]
    return 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0), axis=1)


def _vertex_adjacency(faces, n_verts):
    """Build vertex adjacency list from faces (vectorized)."""
    f = np.asarray(faces)
    # Build all edge pairs in one shot
    rows = np.concatenate([f[:, 0], f[:, 0], f[:, 1], f[:, 1], f[:, 2], f[:, 2]])
    cols = np.concatenate([f[:, 1], f[:, 2], f[:, 0], f[:, 2], f[:, 0], f[:, 1]])
    adj = [set() for _ in range(n_verts)]
    for r, c in zip(rows, cols):
        adj[r].add(c)
    return adj


def _angle_between(a, b):
    """Angle in radians between two (N,3) arrays of unit vectors, row-wise."""
    dot = np.sum(a * b, axis=1)
    dot = np.clip(dot, -1.0, 1.0)
    return np.arccos(dot)


# =========================================================================
# 1. surface_distance_map
# =========================================================================

def surface_distance_map(cad_vertices, cad_faces, mesh_vertices):
    """Compute per-vertex signed distance from CAD surface to mesh.

    Positive = CAD vertex is outside the mesh (too far from centre).
    Negative = CAD vertex is inside.
    Sign is determined by the dot product of the displacement with the
    CAD vertex normal.

    Args:
        cad_vertices: (N, 3)
        cad_faces:    (F, 3)
        mesh_vertices: (M, 3)

    Returns:
        dict:
            signed_distances — (N,) signed distance per vertex
            unsigned_distances — (N,) absolute distance
            mean_dist — scalar
            max_dist  — scalar
    """
    cad_v = np.asarray(cad_vertices, dtype=np.float64)
    mesh_v = np.asarray(mesh_vertices, dtype=np.float64)
    cad_normals = _compute_vertex_normals(cad_v, np.asarray(cad_faces))

    tree = _AKDTree(mesh_v)
    dists, idx = tree.query(cad_v)
    disp = mesh_v[idx] - cad_v
    signs = np.sign(np.sum(disp * cad_normals, axis=1))
    signs[signs == 0] = 1.0
    signed = dists * signs

    return {
        "signed_distances": signed,
        "unsigned_distances": dists,
        "mean_dist": float(np.mean(dists)),
        "max_dist": float(np.max(dists)),
    }


# =========================================================================
# 2. hausdorff_distance
# =========================================================================

def hausdorff_distance(vertices_a, vertices_b):
    """Symmetric Hausdorff distance and mean surface distance.

    Uses GPU acceleration when available (CuPy / PyTorch CUDA).

    Args:
        vertices_a: (N, 3)
        vertices_b: (M, 3)

    Returns:
        dict: hausdorff, mean_a_to_b, mean_b_to_a, mean_symmetric
    """
    # GPU fast path
    if _gpu_ok():
        return _hausdorff_gpu(vertices_a, vertices_b)

    a = np.asarray(vertices_a, dtype=np.float64)
    b = np.asarray(vertices_b, dtype=np.float64)

    tree_b = KDTree(b)
    d_a2b, _ = tree_b.query(a)

    tree_a = KDTree(a)
    d_b2a, _ = tree_a.query(b)

    return {
        "hausdorff": float(max(np.max(d_a2b), np.max(d_b2a))),
        "mean_a_to_b": float(np.mean(d_a2b)),
        "mean_b_to_a": float(np.mean(d_b2a)),
        "mean_symmetric": float((np.mean(d_a2b) + np.mean(d_b2a)) / 2),
    }


# =========================================================================
# 3. normal_deviation_map
# =========================================================================

def normal_deviation_map(cad_vertices, cad_faces, mesh_vertices, mesh_faces):
    """Per-vertex angle deviation between CAD normals and nearest mesh normals.

    Args:
        cad_vertices, cad_faces: CAD mesh
        mesh_vertices, mesh_faces: target mesh

    Returns:
        dict:
            angles_rad   — (N,) angle per CAD vertex in radians
            angles_deg   — (N,) in degrees
            mean_deg     — scalar
            max_deg      — scalar
    """
    cad_v = np.asarray(cad_vertices, dtype=np.float64)
    mesh_v = np.asarray(mesh_vertices, dtype=np.float64)
    cad_n = _compute_vertex_normals(cad_v, np.asarray(cad_faces))
    mesh_n = _compute_vertex_normals(mesh_v, np.asarray(mesh_faces))

    tree = _AKDTree(mesh_v)
    _, idx = tree.query(cad_v)
    angles = _angle_between(cad_n, mesh_n[idx])
    angles_deg = np.degrees(angles)

    return {
        "angles_rad": angles,
        "angles_deg": angles_deg,
        "mean_deg": float(np.mean(angles_deg)),
        "max_deg": float(np.max(angles_deg)),
    }


# =========================================================================
# 4. curvature_deviation_map
# =========================================================================

def curvature_deviation_map(cad_vertices, cad_faces, mesh_vertices, mesh_faces):
    """Per-vertex curvature difference between CAD and mesh.

    Uses a discrete Gaussian curvature approximation (angle deficit).

    Returns:
        dict:
            cad_curvature   — (N,)
            mesh_curvature  — (M,)
            deviation       — (N,) curvature difference at nearest pairs
            mean_abs_dev    — scalar
    """
    cad_v = np.asarray(cad_vertices, dtype=np.float64)
    mesh_v = np.asarray(mesh_vertices, dtype=np.float64)
    cad_f = np.asarray(cad_faces)
    mesh_f = np.asarray(mesh_faces)

    cad_curv = _vertex_curvature(cad_v, cad_f)
    mesh_curv = _vertex_curvature(mesh_v, mesh_f)

    tree = _AKDTree(mesh_v)
    _, idx = tree.query(cad_v)
    dev = cad_curv - mesh_curv[idx]

    return {
        "cad_curvature": cad_curv,
        "mesh_curvature": mesh_curv,
        "deviation": dev,
        "mean_abs_dev": float(np.mean(np.abs(dev))),
    }


def _vertex_curvature(vertices, faces):
    """Discrete Gaussian curvature via angle deficit."""
    n = len(vertices)
    angle_sum = np.zeros(n)
    for tri in faces:
        for k in range(3):
            i, j, l = tri[k], tri[(k + 1) % 3], tri[(k + 2) % 3]
            v1 = vertices[j] - vertices[i]
            v2 = vertices[l] - vertices[i]
            len1 = np.linalg.norm(v1)
            len2 = np.linalg.norm(v2)
            if len1 > 1e-12 and len2 > 1e-12:
                cos_a = np.dot(v1, v2) / (len1 * len2)
                cos_a = max(-1.0, min(1.0, cos_a))
                angle_sum[i] += math.acos(cos_a)
    return 2 * math.pi - angle_sum


# =========================================================================
# 5. laplacian_smooth_toward
# =========================================================================

def laplacian_smooth_toward(cad_vertices, cad_faces, target_vertices,
                             iterations=5, lam=0.5, target_weight=0.3):
    """Laplacian smoothing of CAD mesh biased toward a target.

    Each iteration moves every vertex toward:
      (1-target_weight) * laplacian_average + target_weight * nearest_target

    Args:
        cad_vertices: (N, 3)
        cad_faces:    (F, 3)
        target_vertices: (M, 3)
        iterations: number of smoothing passes
        lam:        Laplacian blend factor (0=no smooth, 1=full Laplacian)
        target_weight: 0-1 pull toward target mesh

    Returns:
        smoothed: (N, 3)
    """
    verts = np.asarray(cad_vertices, dtype=np.float64).copy()
    faces = np.asarray(cad_faces)
    target = np.asarray(target_vertices, dtype=np.float64)
    adj = _vertex_adjacency(faces, len(verts))
    tree = _AKDTree(target)

    for _ in range(iterations):
        new_verts = verts.copy()
        _, nearest_idx = tree.query(verts)
        target_pts = target[nearest_idx]

        for i in range(len(verts)):
            if not adj[i]:
                continue
            neighbors = list(adj[i])
            lap_avg = np.mean(verts[neighbors], axis=0)
            smooth_pos = verts[i] + lam * (lap_avg - verts[i])
            new_verts[i] = ((1 - target_weight) * smooth_pos +
                            target_weight * target_pts[i])
        verts = new_verts

    return verts


# =========================================================================
# 6. project_vertices_to_mesh
# =========================================================================

def project_vertices_to_mesh(cad_vertices, mesh_vertices, mesh_faces,
                              max_distance=None):
    """Project each CAD vertex onto the nearest point on the mesh surface.

    For each CAD vertex, finds the nearest mesh triangle and computes the
    closest point on that triangle.

    Args:
        cad_vertices:  (N, 3)
        mesh_vertices: (M, 3)
        mesh_faces:    (F, 3)
        max_distance:  optional cap — vertices farther than this stay put

    Returns:
        projected: (N, 3)
        distances: (N,) distance moved
    """
    cad_v = np.asarray(cad_vertices, dtype=np.float64)
    mesh_v = np.asarray(mesh_vertices, dtype=np.float64)
    mesh_f = np.asarray(mesh_faces)

    # Use KDTree on mesh verts for initial nearest vertex
    tree = _AKDTree(mesh_v)
    _, nearest_vi = tree.query(cad_v)

    # Build vertex → face map
    vert_to_faces = [[] for _ in range(len(mesh_v))]
    for fi, f in enumerate(mesh_f):
        for vi in f:
            vert_to_faces[vi].append(fi)

    projected = cad_v.copy()
    distances = np.zeros(len(cad_v))

    for i in range(len(cad_v)):
        pt = cad_v[i]
        # Check faces adjacent to nearest vertex
        candidate_faces = vert_to_faces[nearest_vi[i]]
        best_d = float("inf")
        best_p = pt.copy()
        for fi in candidate_faces:
            tri = mesh_v[mesh_f[fi]]
            proj = _project_point_triangle(pt, tri[0], tri[1], tri[2])
            d = np.linalg.norm(proj - pt)
            if d < best_d:
                best_d = d
                best_p = proj

        if max_distance is not None and best_d > max_distance:
            continue
        projected[i] = best_p
        distances[i] = best_d

    return projected, distances


def _project_point_triangle(p, a, b, c):
    """Closest point on triangle abc to point p."""
    ab = b - a
    ac = c - a
    ap = p - a
    d1 = np.dot(ab, ap)
    d2 = np.dot(ac, ap)
    if d1 <= 0 and d2 <= 0:
        return a.copy()
    bp = p - b
    d3 = np.dot(ab, bp)
    d4 = np.dot(ac, bp)
    if d3 >= 0 and d4 <= d3:
        return b.copy()
    cp = p - c
    d5 = np.dot(ab, cp)
    d6 = np.dot(ac, cp)
    if d6 >= 0 and d5 <= d6:
        return c.copy()
    vc = d1 * d4 - d3 * d2
    if vc <= 0 and d1 >= 0 and d3 <= 0:
        denom = d1 - d3
        v = d1 / denom if abs(denom) > 1e-30 else 0.0
        return a + v * ab
    vb = d5 * d2 - d1 * d6
    if vb <= 0 and d2 >= 0 and d6 <= 0:
        denom = d2 - d6
        w = d2 / denom if abs(denom) > 1e-30 else 0.0
        return a + w * ac
    va = d3 * d6 - d5 * d4
    if va <= 0 and (d4 - d3) >= 0 and (d5 - d6) >= 0:
        w = (d4 - d3) / ((d4 - d3) + (d5 - d6))
        return b + w * (c - b)
    denom = 1.0 / (va + vb + vc)
    v = vb * denom
    w = vc * denom
    return a + v * ab + w * ac


# =========================================================================
# 7. local_scale_correction
# =========================================================================

def local_scale_correction(cad_vertices, cad_faces, mesh_vertices,
                            n_regions=8):
    """Per-region scale correction.

    Divides the mesh into angular/height regions, measures the CAD vs mesh
    scale in each, and adjusts CAD vertices to compensate.

    Args:
        cad_vertices:  (N, 3)
        cad_faces:     (F, 3)
        mesh_vertices: (M, 3)
        n_regions:     number of Z-based regions

    Returns:
        corrected: (N, 3)
        scale_factors: (n_regions,) applied scale per region
    """
    cad_v = np.asarray(cad_vertices, dtype=np.float64).copy()
    mesh_v = np.asarray(mesh_vertices, dtype=np.float64)
    centroid = np.mean(cad_v, axis=0)

    z_min = min(cad_v[:, 2].min(), mesh_v[:, 2].min())
    z_max = max(cad_v[:, 2].max(), mesh_v[:, 2].max())
    z_edges = np.linspace(z_min, z_max, n_regions + 1)

    cad_r = np.sqrt((cad_v[:, 0] - centroid[0]) ** 2 +
                     (cad_v[:, 1] - centroid[1]) ** 2)
    mesh_r = np.sqrt((mesh_v[:, 0] - centroid[0]) ** 2 +
                      (mesh_v[:, 1] - centroid[1]) ** 2)

    scale_factors = np.ones(n_regions)
    for iz in range(n_regions):
        cad_mask = (cad_v[:, 2] >= z_edges[iz]) & (cad_v[:, 2] < z_edges[iz + 1])
        mesh_mask = (mesh_v[:, 2] >= z_edges[iz]) & (mesh_v[:, 2] < z_edges[iz + 1])
        if not np.any(cad_mask) or not np.any(mesh_mask):
            continue
        cad_mean_r = float(np.mean(cad_r[cad_mask]))
        mesh_mean_r = float(np.mean(mesh_r[mesh_mask]))
        if cad_mean_r > 1e-6:
            s = mesh_mean_r / cad_mean_r
            s = max(0.5, min(2.0, s))  # clamp
            scale_factors[iz] = s
            cad_v[cad_mask, 0] = centroid[0] + (cad_v[cad_mask, 0] - centroid[0]) * s
            cad_v[cad_mask, 1] = centroid[1] + (cad_v[cad_mask, 1] - centroid[1]) * s

    return cad_v, scale_factors


# =========================================================================
# 8. feature_edge_transfer
# =========================================================================

def feature_edge_transfer(cad_vertices, cad_faces, mesh_vertices, mesh_faces,
                           dihedral_threshold_deg=40.0):
    """Sharpen CAD edges where the mesh has sharp (feature) edges.

    Detects dihedral angles in the mesh, identifies the corresponding CAD
    vertices, and sharpens them by projecting toward the mesh edge position.

    Args:
        cad_vertices:  (N, 3)
        cad_faces:     (F, 3)
        mesh_vertices: (M, 3)
        mesh_faces:    (G, 3)
        dihedral_threshold_deg: edges sharper than this are "features"

    Returns:
        sharpened: (N, 3)
        n_sharp_edges: number of sharp edges found in mesh
    """
    mesh_v = np.asarray(mesh_vertices, dtype=np.float64)
    mesh_f = np.asarray(mesh_faces)
    cad_v = np.asarray(cad_vertices, dtype=np.float64).copy()
    cad_f = np.asarray(cad_faces)

    mesh_fn = _compute_face_normals(mesh_v, mesh_f)

    # Build edge → face map for mesh
    edge_faces = {}
    for fi, f in enumerate(mesh_f):
        for k in range(3):
            e = tuple(sorted([f[k], f[(k + 1) % 3]]))
            edge_faces.setdefault(e, []).append(fi)

    # Find sharp edges: an edge is "sharp" when the angle between
    # its adjacent face normals exceeds dihedral_threshold_deg.
    # cos(angle) < cos(threshold) means the angle is larger.
    threshold = math.cos(math.radians(dihedral_threshold_deg))
    sharp_edge_midpoints = []
    for e, fis in edge_faces.items():
        if len(fis) == 2:
            cos_a = np.dot(mesh_fn[fis[0]], mesh_fn[fis[1]])
            if cos_a < threshold:
                mid = (mesh_v[e[0]] + mesh_v[e[1]]) / 2
                sharp_edge_midpoints.append(mid)

    n_sharp = len(sharp_edge_midpoints)
    if n_sharp == 0:
        return cad_v, 0

    sharp_pts = np.array(sharp_edge_midpoints)
    tree = _AKDTree(sharp_pts)

    # For each CAD vertex near a sharp edge, pull toward it
    dists, idx = tree.query(cad_v)
    # Only affect vertices within a reasonable distance
    median_dist = float(np.median(dists))
    affect_mask = dists < median_dist * 2
    blend = 0.5

    for i in np.where(affect_mask)[0]:
        cad_v[i] += blend * (sharp_pts[idx[i]] - cad_v[i]) * (1 - dists[i] / (median_dist * 2))

    return cad_v, n_sharp


# =========================================================================
# 9. vertex_normal_realign
# =========================================================================

def vertex_normal_realign(cad_vertices, cad_faces, mesh_vertices, mesh_faces,
                           strength=0.5):
    """Adjust vertex positions so their normals better match the mesh.

    For each vertex, computes the angular difference between the CAD normal
    and the target mesh normal, then displaces the vertex along the
    difference direction.

    Args:
        cad_vertices:  (N, 3)
        cad_faces:     (F, 3)
        mesh_vertices: (M, 3)
        mesh_faces:    (G, 3)
        strength:      displacement magnitude scale

    Returns:
        adjusted: (N, 3)
    """
    cad_v = np.asarray(cad_vertices, dtype=np.float64).copy()
    mesh_v = np.asarray(mesh_vertices, dtype=np.float64)
    cad_n = _compute_vertex_normals(cad_v, np.asarray(cad_faces))
    mesh_n = _compute_vertex_normals(mesh_v, np.asarray(mesh_faces))

    tree = _AKDTree(mesh_v)
    dists, idx = tree.query(cad_v)
    target_n = mesh_n[idx]

    # Displacement: rotate vertex position along the normal difference
    normal_diff = target_n - cad_n
    # Scale by distance to surface to avoid over-correction
    scale = strength * np.minimum(dists, 1.0)
    cad_v += normal_diff * scale[:, None]

    return cad_v


# =========================================================================
# 10. fill_surface_holes
# =========================================================================

def fill_surface_holes(vertices, faces):
    """Detect and close boundary loops (holes) in a triangle mesh.

    Finds boundary edges (edges used by only one face) and fills each
    loop with a fan of triangles.

    Args:
        vertices: (N, 3)
        faces:    (F, 3)

    Returns:
        new_vertices: (N, 3) — unchanged
        new_faces:    (F + P, 3) — with hole-filling triangles added
        n_holes:      number of holes filled
    """
    verts = np.asarray(vertices, dtype=np.float64)
    tris = np.asarray(faces)

    # Find boundary edges
    edge_count = {}
    edge_to_half = {}
    for fi, f in enumerate(tris):
        for k in range(3):
            e = (f[k], f[(k + 1) % 3])
            e_key = tuple(sorted(e))
            edge_count[e_key] = edge_count.get(e_key, 0) + 1
            edge_to_half[e] = e_key

    boundary_edges = {}
    for fi, f in enumerate(tris):
        for k in range(3):
            v0, v1 = f[k], f[(k + 1) % 3]
            e_key = tuple(sorted([v0, v1]))
            if edge_count[e_key] == 1:
                boundary_edges[v0] = v1

    if not boundary_edges:
        return verts, tris, 0

    # Chain boundary edges into loops
    used = set()
    loops = []
    for start in boundary_edges:
        if start in used:
            continue
        loop = [start]
        used.add(start)
        current = start
        for _ in range(len(boundary_edges) + 1):
            nxt = boundary_edges.get(current)
            if nxt is None or nxt in used:
                break
            loop.append(nxt)
            used.add(nxt)
            current = nxt
        if len(loop) >= 3:
            loops.append(loop)

    # Fill each loop with a fan from its centroid
    new_faces = list(tris)
    new_verts = list(verts)
    for loop in loops:
        center = np.mean(verts[loop], axis=0)
        ci = len(new_verts)
        new_verts.append(center)
        for j in range(len(loop)):
            j_next = (j + 1) % len(loop)
            new_faces.append([ci, loop[j], loop[j_next]])

    return (np.array(new_verts, dtype=np.float64),
            np.array(new_faces),
            len(loops))


# =========================================================================
# 11. decimate_to_match
# =========================================================================

def decimate_to_match(vertices, faces, target_face_count):
    """Simplify a mesh to approximately *target_face_count* triangles.

    Uses iterative edge-collapse based on edge length (shortest first).

    Args:
        vertices: (N, 3)
        faces:    (F, 3)
        target_face_count: desired number of faces

    Returns:
        new_vertices: (N', 3)
        new_faces:    (F', 3) with F' ≈ target_face_count
    """
    verts = np.asarray(vertices, dtype=np.float64).copy()
    face_list = [list(f) for f in faces]

    while len(face_list) > target_face_count:
        # Find shortest edge
        best_len = float("inf")
        best_edge = None
        for fi, f in enumerate(face_list):
            for k in range(3):
                v0, v1 = f[k], f[(k + 1) % 3]
                d = np.linalg.norm(verts[v0] - verts[v1])
                if d < best_len:
                    best_len = d
                    best_edge = (v0, v1)

        if best_edge is None:
            break

        va, vb = best_edge
        # Collapse vb into va (midpoint)
        verts[va] = (verts[va] + verts[vb]) / 2

        # Replace vb with va in all faces, remove degenerate
        new_faces = []
        for f in face_list:
            f2 = [va if v == vb else v for v in f]
            if len(set(f2)) == 3:
                new_faces.append(f2)
        face_list = new_faces

        if len(face_list) <= target_face_count:
            break

    # Compact vertex indices
    used = set()
    for f in face_list:
        used.update(f)
    used = sorted(used)
    remap = {old: new for new, old in enumerate(used)}
    new_verts = verts[used]
    new_faces = [[remap[v] for v in f] for f in face_list]

    return new_verts, np.array(new_faces)


# =========================================================================
# 12. suggest_general_adjustments
# =========================================================================

def suggest_general_adjustments(cad_vertices, cad_faces,
                                 mesh_vertices, mesh_faces):
    """Analyse CAD vs mesh and return ranked adjustment suggestions.

    Runs surface distance, normal deviation, curvature deviation, and hole
    detection, then produces a priority-ordered suggestion list.

    Returns:
        list of dicts: action, priority, params, reason
    """
    cad_v = np.asarray(cad_vertices, dtype=np.float64)
    mesh_v = np.asarray(mesh_vertices, dtype=np.float64)
    cad_f = np.asarray(cad_faces)
    mesh_f = np.asarray(mesh_faces)

    suggestions = []

    # Surface distance
    sd = surface_distance_map(cad_v, cad_f, mesh_v)

    if sd["mean_dist"] > 1.0:
        suggestions.append({
            "action": "project_vertices_to_mesh",
            "priority": 1,
            "params": {},
            "reason": f"Mean surface distance {sd['mean_dist']:.2f} — "
                      f"vertices need projection onto mesh",
        })

    if sd["max_dist"] > 5.0:
        suggestions.append({
            "action": "laplacian_smooth_toward",
            "priority": 2,
            "params": {"iterations": 10, "target_weight": 0.5},
            "reason": f"Max surface distance {sd['max_dist']:.2f} — "
                      f"smoothing toward target recommended",
        })

    # Normal deviation
    nd = normal_deviation_map(cad_v, cad_f, mesh_v, mesh_f)
    if nd["mean_deg"] > 15.0:
        suggestions.append({
            "action": "vertex_normal_realign",
            "priority": 3,
            "params": {"strength": 0.5},
            "reason": f"Mean normal deviation {nd['mean_deg']:.1f}° — "
                      f"normals diverge from mesh",
        })

    # Scale correction
    if sd["mean_dist"] > 0.5:
        suggestions.append({
            "action": "local_scale_correction",
            "priority": 4,
            "params": {"n_regions": 8},
            "reason": "Regional scale mismatch detected",
        })

    # Feature edges
    suggestions.append({
        "action": "feature_edge_transfer",
        "priority": 5,
        "params": {"dihedral_threshold_deg": 40.0},
        "reason": "Transfer sharp feature edges from mesh to CAD",
    })

    # Curvature
    cd = curvature_deviation_map(cad_v, cad_f, mesh_v, mesh_f)
    if cd["mean_abs_dev"] > 0.1:
        suggestions.append({
            "action": "laplacian_smooth_toward",
            "priority": 6,
            "params": {"iterations": 3, "target_weight": 0.2},
            "reason": f"Curvature deviation {cd['mean_abs_dev']:.3f} — "
                      f"light smoothing suggested",
        })

    # Holes
    _, _, n_holes = fill_surface_holes(cad_v, cad_f)
    if n_holes > 0:
        suggestions.append({
            "action": "fill_surface_holes",
            "priority": 7,
            "params": {},
            "reason": f"{n_holes} boundary hole(s) found in CAD mesh",
        })

    # Triangle count mismatch
    ratio = len(cad_f) / max(len(mesh_f), 1)
    if ratio > 2.0:
        suggestions.append({
            "action": "decimate_to_match",
            "priority": 8,
            "params": {"target_face_count": len(mesh_f)},
            "reason": f"CAD has {ratio:.1f}x more faces than mesh — "
                      f"decimation suggested",
        })

    # Hausdorff
    hd = hausdorff_distance(cad_v, mesh_v)
    if hd["hausdorff"] > 10.0:
        suggestions.append({
            "action": "project_vertices_to_mesh",
            "priority": 9,
            "params": {"max_distance": hd["hausdorff"] * 0.8},
            "reason": f"Hausdorff distance {hd['hausdorff']:.2f} — "
                      f"large outlier vertices detected",
        })

    suggestions.sort(key=lambda s: s["priority"])
    return suggestions if suggestions else [
        {"action": "none", "priority": 0, "params": {},
         "reason": "CAD mesh closely matches target mesh"}
    ]


# =========================================================================
# Differentiators (moved from adversarial_loop for standard reuse)
# =========================================================================

def surface_area(vertices, faces):
    """Total surface area of a triangle mesh."""
    areas = _face_areas(np.asarray(vertices), np.asarray(faces))
    return float(np.sum(areas))


def per_region_area_difference(cad_v, cad_f, mesh_v, mesh_f, n_regions=8):
    """Compare surface area in Z-band regions."""
    cad_v, mesh_v = np.asarray(cad_v), np.asarray(mesh_v)
    z_min = min(cad_v[:, 2].min(), mesh_v[:, 2].min())
    z_max = max(cad_v[:, 2].max(), mesh_v[:, 2].max())
    z_edges = np.linspace(z_min, z_max, n_regions + 1)

    cad_fn = _face_centroids(cad_v, np.asarray(cad_f))
    mesh_fn = _face_centroids(mesh_v, np.asarray(mesh_f))
    cad_areas = _face_areas(cad_v, np.asarray(cad_f))
    mesh_areas = _face_areas(mesh_v, np.asarray(mesh_f))

    diffs = []
    for i in range(n_regions):
        cad_mask = (cad_fn[:, 2] >= z_edges[i]) & (cad_fn[:, 2] < z_edges[i + 1])
        mesh_mask = (mesh_fn[:, 2] >= z_edges[i]) & (mesh_fn[:, 2] < z_edges[i + 1])
        ca = float(np.sum(cad_areas[cad_mask]))
        ma = float(np.sum(mesh_areas[mesh_mask]))
        total = ca + ma
        diffs.append(abs(ca - ma) / max(total, 1.0))
    return diffs


def _face_centroids(vertices, faces):
    """Centroids of each triangle face."""
    v0, v1, v2 = vertices[faces[:, 0]], vertices[faces[:, 1]], vertices[faces[:, 2]]
    return (v0 + v1 + v2) / 3


def curvature_histogram_diff(cad_v, cad_f, mesh_v, mesh_f, n_bins=20):
    """Compare discrete curvature distributions."""
    cad_curv = _vertex_curvature(np.asarray(cad_v), np.asarray(cad_f))
    mesh_curv = _vertex_curvature(np.asarray(mesh_v), np.asarray(mesh_f))

    lo = min(cad_curv.min(), mesh_curv.min())
    hi = max(cad_curv.max(), mesh_curv.max())
    if hi - lo < 1e-12:
        return 0.0
    bins = np.linspace(lo, hi, n_bins + 1)
    h_cad, _ = np.histogram(cad_curv, bins=bins, density=True)
    h_mesh, _ = np.histogram(mesh_curv, bins=bins, density=True)
    return float(np.sum(np.abs(h_cad - h_mesh)) * (bins[1] - bins[0]))


def bbox_aspect_diff(cad_v, mesh_v):
    """Compare bounding-box aspect ratios."""
    cad_v, mesh_v = np.asarray(cad_v), np.asarray(mesh_v)
    cad_span = np.ptp(cad_v, axis=0)
    mesh_span = np.ptp(mesh_v, axis=0)
    cad_span[cad_span < 1e-12] = 1e-12
    mesh_span[mesh_span < 1e-12] = 1e-12
    cad_aspect = cad_span / cad_span.max()
    mesh_aspect = mesh_span / mesh_span.max()
    return float(np.max(np.abs(cad_aspect - mesh_aspect)))


def vertex_density_diff(cad_v, mesh_v, n_regions=8):
    """Compare local vertex density by dividing into spatial regions."""
    cad_v, mesh_v = np.asarray(cad_v), np.asarray(mesh_v)
    all_v = np.vstack([cad_v, mesh_v])
    lo, hi = all_v.min(axis=0), all_v.max(axis=0)
    span = hi - lo
    span[span < 1e-12] = 1.0

    bins = max(2, int(round(n_regions ** (1/3))))
    cad_idx = np.floor((cad_v - lo) / span * (bins - 1e-9)).astype(int).clip(0, bins - 1)
    mesh_idx = np.floor((mesh_v - lo) / span * (bins - 1e-9)).astype(int).clip(0, bins - 1)

    cad_flat = cad_idx[:, 0] * bins * bins + cad_idx[:, 1] * bins + cad_idx[:, 2]
    mesh_flat = mesh_idx[:, 0] * bins * bins + mesh_idx[:, 1] * bins + mesh_idx[:, 2]

    n_cells = bins ** 3
    cad_counts = np.bincount(cad_flat, minlength=n_cells).astype(float)
    mesh_counts = np.bincount(mesh_flat, minlength=n_cells).astype(float)

    cad_counts /= max(cad_counts.sum(), 1)
    mesh_counts /= max(mesh_counts.sum(), 1)

    return float(np.sum(np.abs(cad_counts - mesh_counts)) / 2)


def edge_length_distribution_diff(cad_v, cad_f, mesh_v, mesh_f, n_bins=20):
    """Compare distributions of triangle edge lengths."""
    def _edge_lengths(v, f):
        v = np.asarray(v)
        f = np.asarray(f)
        e1 = np.linalg.norm(v[f[:, 1]] - v[f[:, 0]], axis=1)
        e2 = np.linalg.norm(v[f[:, 2]] - v[f[:, 1]], axis=1)
        e3 = np.linalg.norm(v[f[:, 0]] - v[f[:, 2]], axis=1)
        return np.concatenate([e1, e2, e3])

    cad_el = _edge_lengths(cad_v, cad_f)
    mesh_el = _edge_lengths(mesh_v, mesh_f)
    lo = min(cad_el.min(), mesh_el.min())
    hi = max(cad_el.max(), mesh_el.max())
    if hi - lo < 1e-12:
        return 0.0
    bins = np.linspace(lo, hi, n_bins + 1)
    h_cad, _ = np.histogram(cad_el, bins=bins, density=True)
    h_mesh, _ = np.histogram(mesh_el, bins=bins, density=True)
    return float(np.sum(np.abs(h_cad - h_mesh)) * (bins[1] - bins[0]))


def centroid_drift(cad_v, mesh_v, n_regions=10):
    """Max centroid displacement between matching Z-band regions."""
    cad_v, mesh_v = np.asarray(cad_v), np.asarray(mesh_v)
    z_min = min(cad_v[:, 2].min(), mesh_v[:, 2].min())
    z_max = max(cad_v[:, 2].max(), mesh_v[:, 2].max())
    z_edges = np.linspace(z_min, z_max, n_regions + 1)

    max_drift = 0.0
    for i in range(n_regions):
        cad_mask = (cad_v[:, 2] >= z_edges[i]) & (cad_v[:, 2] < z_edges[i + 1])
        mesh_mask = (mesh_v[:, 2] >= z_edges[i]) & (mesh_v[:, 2] < z_edges[i + 1])
        if not np.any(cad_mask) or not np.any(mesh_mask):
            continue
        cad_c = cad_v[cad_mask].mean(axis=0)
        mesh_c = mesh_v[mesh_mask].mean(axis=0)
        drift = np.linalg.norm(cad_c - mesh_c)
        max_drift = max(max_drift, drift)
    return float(max_drift)


def median_surface_distance(cad_v, mesh_v):
    """Median nearest-neighbor distance (robust to outliers)."""
    tree = _AKDTree(np.asarray(mesh_v))
    dists, _ = tree.query(np.asarray(cad_v))
    return float(np.median(dists))


def local_roughness_diff(cad_v, cad_f, mesh_v, mesh_f, n_samples=200):
    """Compare local surface roughness (variance of NN distances)."""
    cad_v, mesh_v = np.asarray(cad_v), np.asarray(mesh_v)

    def _local_roughness(v, k=6):
        tree = _AKDTree(v)
        dists, _ = tree.query(v, k=min(k + 1, len(v)))
        return np.std(dists[:, 1:], axis=1).mean()

    cad_rough = _local_roughness(cad_v)
    mesh_rough = _local_roughness(mesh_v)
    total = cad_rough + mesh_rough
    if total < 1e-12:
        return 0.0
    return float(abs(cad_rough - mesh_rough) / total * 100)


def volume_diff(cad_v, cad_f, mesh_v, mesh_f):
    """Compare signed volumes (divergence theorem approximation)."""
    def _signed_volume(v, f):
        v = np.asarray(v)
        f = np.asarray(f)
        v0, v1, v2 = v[f[:, 0]], v[f[:, 1]], v[f[:, 2]]
        cross = np.cross(v1, v2)
        return float(np.abs(np.sum(v0 * cross)) / 6.0)

    vol_cad = _signed_volume(cad_v, cad_f)
    vol_mesh = _signed_volume(mesh_v, mesh_f)
    total = vol_cad + vol_mesh
    if total < 1e-12:
        return 0.0
    return float(abs(vol_cad - vol_mesh) / total * 100)


def percentile_95_distance(cad_v, mesh_v):
    """95th percentile of NN distances (captures widespread offset)."""
    tree = _AKDTree(np.asarray(mesh_v))
    dists, _ = tree.query(np.asarray(cad_v))
    return float(np.percentile(dists, 95))


def face_normal_consistency(cad_v, cad_f, mesh_v, mesh_f):
    """Compare face normal direction distributions via L1 histogram diff."""
    cad_fn = _compute_face_normals(np.asarray(cad_v), np.asarray(cad_f))
    mesh_fn = _compute_face_normals(np.asarray(mesh_v), np.asarray(mesh_f))

    def _normal_angles(normals):
        n = normals / (np.linalg.norm(normals, axis=1, keepdims=True) + 1e-12)
        theta = np.arctan2(n[:, 1], n[:, 0])
        phi = np.arccos(np.clip(n[:, 2], -1, 1))
        return theta, phi

    ct, cp = _normal_angles(cad_fn)
    mt, mp = _normal_angles(mesh_fn)

    n_bins = 12
    t_bins = np.linspace(-np.pi, np.pi, n_bins + 1)
    p_bins = np.linspace(0, np.pi, n_bins + 1)

    ch, _ = np.histogram(ct, bins=t_bins, density=True)
    mh, _ = np.histogram(mt, bins=t_bins, density=True)
    diff_t = np.sum(np.abs(ch - mh)) * (t_bins[1] - t_bins[0])

    ch, _ = np.histogram(cp, bins=p_bins, density=True)
    mh, _ = np.histogram(mp, bins=p_bins, density=True)
    diff_p = np.sum(np.abs(ch - mh)) * (p_bins[1] - p_bins[0])

    return float((diff_t + diff_p) / 2 * 10)


def multi_scale_distance(cad_v, mesh_v, scales=(1.0, 0.5, 0.25)):
    """Compare NN distances at multiple resolution scales via downsampling."""
    cad_v, mesh_v = np.asarray(cad_v), np.asarray(mesh_v)
    total = 0.0
    for s in scales:
        n_cad = max(10, int(len(cad_v) * s))
        n_mesh = max(10, int(len(mesh_v) * s))
        idx_c = np.random.RandomState(42).choice(len(cad_v), n_cad, replace=False)
        idx_m = np.random.RandomState(42).choice(len(mesh_v), n_mesh, replace=False)
        tree = _AKDTree(mesh_v[idx_m])
        dists, _ = tree.query(cad_v[idx_c])
        total += np.mean(dists)
    return float(total / len(scales))


def shape_diameter_diff(cad_v, cad_f, mesh_v, mesh_f, n_samples=100):
    """Compare shape diameter function (ray-based thickness) statistics."""
    cad_v, mesh_v = np.asarray(cad_v), np.asarray(mesh_v)

    def _diameter_stats(v):
        center = v.mean(axis=0)
        dirs = v - center
        norms = np.linalg.norm(dirs, axis=1, keepdims=True)
        norms[norms < 1e-12] = 1e-12
        dirs_norm = dirs / norms
        dots = dirs_norm @ dirs_norm.T
        antipodal_idx = np.argmin(dots, axis=1)
        diameters = np.linalg.norm(v - v[antipodal_idx], axis=1)
        return np.mean(diameters), np.std(diameters)

    cm, cs = _diameter_stats(cad_v)
    mm, ms = _diameter_stats(mesh_v)

    mean_diff = abs(cm - mm) / max(cm + mm, 1e-6) * 2
    std_diff = abs(cs - ms) / max(cs + ms, 1e-6) * 2
    return float((mean_diff + std_diff) * 50)


def moment_of_inertia_diff(cad_v, mesh_v):
    """Compare principal moments of inertia (rotation invariant shape desc)."""
    cad_v, mesh_v = np.asarray(cad_v), np.asarray(mesh_v)

    def _moments(v):
        c = v.mean(axis=0)
        vc = v - c
        Ixx = np.sum(vc[:, 1]**2 + vc[:, 2]**2)
        Iyy = np.sum(vc[:, 0]**2 + vc[:, 2]**2)
        Izz = np.sum(vc[:, 0]**2 + vc[:, 1]**2)
        I = np.array([Ixx, Iyy, Izz])
        I /= max(np.max(I), 1e-12)
        return np.sort(I)

    cad_m = _moments(cad_v)
    mesh_m = _moments(mesh_v)
    return float(np.max(np.abs(cad_m - mesh_m)) * 100)


def distance_histogram_diff(cad_v, mesh_v, n_bins=30):
    """Compare distributions of pairwise inter-vertex distances (D2 descriptor)."""
    rng = np.random.RandomState(42)
    n_samples = min(500, len(cad_v), len(mesh_v))

    def _d2(v, n):
        idx = rng.choice(len(v), (n, 2), replace=True)
        dists = np.linalg.norm(v[idx[:, 0]] - v[idx[:, 1]], axis=1)
        return dists

    cad_d = _d2(np.asarray(cad_v), n_samples)
    mesh_d = _d2(np.asarray(mesh_v), n_samples)

    lo = min(cad_d.min(), mesh_d.min())
    hi = max(cad_d.max(), mesh_d.max())
    if hi - lo < 1e-12:
        return 0.0
    bins = np.linspace(lo, hi, n_bins + 1)
    h1, _ = np.histogram(cad_d, bins=bins, density=True)
    h2, _ = np.histogram(mesh_d, bins=bins, density=True)
    return float(np.sum(np.abs(h1 - h2)) * (bins[1] - bins[0]))


# =========================================================================
# Fixers (moved from adversarial_loop for standard reuse)
# =========================================================================

def fix_silhouette_mismatch(cad_v, cad_f, mesh_v, mesh_f):
    """Additional projection + smoothing pass to reduce silhouette diff."""
    v = laplacian_smooth_toward(cad_v, cad_f, mesh_v,
                                 iterations=3, lam=0.5, target_weight=0.6)
    v, _ = project_vertices_to_mesh(v, mesh_v, mesh_f)
    return v


def fix_hausdorff_outliers(cad_v, cad_f, mesh_v, mesh_f):
    """Pull worst-case outlier vertices toward mesh."""
    tree = _AKDTree(np.asarray(mesh_v))
    dists, idx = tree.query(np.asarray(cad_v))
    threshold = np.percentile(dists, 95)
    v = np.asarray(cad_v, dtype=np.float64).copy()
    outliers = dists > threshold
    v[outliers] = 0.5 * v[outliers] + 0.5 * np.asarray(mesh_v)[idx[outliers]]
    return v


def fix_normal_deviation(cad_v, cad_f, mesh_v, mesh_f):
    """Vertex displacement to match mesh normals."""
    return vertex_normal_realign(cad_v, cad_f, mesh_v, mesh_f, strength=0.3)


def fix_surface_area(cad_v, cad_f, mesh_v, mesh_f):
    """Scale correction to match surface areas."""
    import math
    sa_cad = surface_area(cad_v, cad_f)
    sa_mesh = surface_area(mesh_v, mesh_f)
    if sa_cad > 1e-6:
        s = math.sqrt(sa_mesh / sa_cad)
        centroid = np.mean(cad_v, axis=0)
        v = centroid + (np.asarray(cad_v) - centroid) * s
        return v
    return np.asarray(cad_v).copy()


def fix_curvature_mismatch(cad_v, cad_f, mesh_v, mesh_f):
    """Gentle Laplacian smoothing to match curvature distribution."""
    return laplacian_smooth_toward(cad_v, cad_f, mesh_v,
                                    iterations=2, lam=0.3, target_weight=0.15)


def fix_bbox_aspect(cad_v, cad_f, mesh_v, mesh_f):
    """Per-axis scaling to match bounding box proportions."""
    cad_span = np.ptp(np.asarray(cad_v), axis=0)
    mesh_span = np.ptp(np.asarray(mesh_v), axis=0)
    cad_span[cad_span < 1e-12] = 1e-12
    scales = mesh_span / cad_span
    centroid = np.mean(cad_v, axis=0)
    return centroid + (np.asarray(cad_v) - centroid) * scales


def fix_region_area(cad_v, cad_f, mesh_v, mesh_f):
    """Per-Z-region surface area matching via radial + Z scale correction."""
    v = np.asarray(cad_v, dtype=np.float64).copy()
    mv = np.asarray(mesh_v)
    f = np.asarray(cad_f)
    mf = np.asarray(mesh_f)
    z_min = min(v[:, 2].min(), mv[:, 2].min())
    z_max = max(v[:, 2].max(), mv[:, 2].max())
    n_regions = 12
    z_edges = np.linspace(z_min, z_max, n_regions + 1)

    cad_centroids = _face_centroids(v, f)
    mesh_centroids = _face_centroids(mv, mf)
    cad_areas = _face_areas(v, f)
    mesh_areas = _face_areas(mv, mf)

    for i in range(n_regions):
        cad_mask_v = (v[:, 2] >= z_edges[i]) & (v[:, 2] < z_edges[i + 1])
        cad_mask_f = (cad_centroids[:, 2] >= z_edges[i]) & (cad_centroids[:, 2] < z_edges[i + 1])
        mesh_mask_f = (mesh_centroids[:, 2] >= z_edges[i]) & (mesh_centroids[:, 2] < z_edges[i + 1])
        if not np.any(cad_mask_v):
            continue
        ca = float(np.sum(cad_areas[cad_mask_f]))
        ma = float(np.sum(mesh_areas[mesh_mask_f]))
        if ca > 1e-6 and ma > 1e-6:
            s = np.clip(np.sqrt(ma / ca), 0.7, 1.5)
            center = v[cad_mask_v].mean(axis=0)
            v[cad_mask_v] = center + (v[cad_mask_v] - center) * s
    v, _ = project_vertices_to_mesh(v, mv, mf)
    return v


def fix_vertex_density(cad_v, cad_f, mesh_v, mesh_f):
    """Re-distribute vertices toward mesh density via multi-NN blending."""
    mv = np.asarray(mesh_v)
    tree = _AKDTree(mv)
    v = np.asarray(cad_v, dtype=np.float64).copy()
    k = min(5, len(mv))
    dists, idx = tree.query(v, k=k)
    weights = 1.0 / (dists + 1e-6)
    weights /= weights.sum(axis=1, keepdims=True)
    targets = np.sum(mv[idx] * weights[:, :, None], axis=1)
    v = v * 0.6 + targets * 0.4
    return v


def fix_edge_length_distribution(cad_v, cad_f, mesh_v, mesh_f):
    """Edge-aware vertex repositioning: equalize edge lengths toward mesh."""
    v = np.asarray(cad_v, dtype=np.float64).copy()
    f = np.asarray(cad_f)
    mv = np.asarray(mesh_v)

    tree = _AKDTree(mv)
    dists, idx = tree.query(v)
    v = v * 0.65 + mv[idx] * 0.35
    v = laplacian_smooth_toward(v, f, mv, iterations=3, lam=0.3, target_weight=0.25)
    return v


def fix_centroid_drift(cad_v, cad_f, mesh_v, mesh_f):
    """Per-region centroid alignment to eliminate drift."""
    v = np.asarray(cad_v, dtype=np.float64).copy()
    mv = np.asarray(mesh_v)
    z_min = min(v[:, 2].min(), mv[:, 2].min())
    z_max = max(v[:, 2].max(), mv[:, 2].max())
    n_regions = 12
    z_edges = np.linspace(z_min, z_max, n_regions + 1)

    for i in range(n_regions):
        cad_mask = (v[:, 2] >= z_edges[i]) & (v[:, 2] < z_edges[i + 1])
        mesh_mask = (mv[:, 2] >= z_edges[i]) & (mv[:, 2] < z_edges[i + 1])
        if not np.any(cad_mask) or not np.any(mesh_mask):
            continue
        cad_c = v[cad_mask].mean(axis=0)
        mesh_c = mv[mesh_mask].mean(axis=0)
        shift = (mesh_c - cad_c) * 0.6
        v[cad_mask] += shift
    return v


def fix_worst_angle_silhouette(cad_v, cad_f, mesh_v, mesh_f):
    """Multi-pass projection + smoothing for worst-angle silhouette."""
    v = np.asarray(cad_v, dtype=np.float64).copy()
    mv = np.asarray(mesh_v)
    mf = np.asarray(mesh_f)
    for _ in range(2):
        v, _ = project_vertices_to_mesh(v, mv, mf)
        v = laplacian_smooth_toward(v, cad_f, mv,
                                     iterations=3, lam=0.3, target_weight=0.4)
    v, _ = project_vertices_to_mesh(v, mv, mf)
    return v


def fix_median_surface_distance(cad_v, cad_f, mesh_v, mesh_f):
    """Iterative NN pull: pull all vertices toward mesh NN."""
    tree = _AKDTree(np.asarray(mesh_v))
    v = np.asarray(cad_v, dtype=np.float64).copy()
    for _ in range(3):
        dists, idx = tree.query(v)
        targets = np.asarray(mesh_v)[idx]
        v = v * 0.6 + targets * 0.4
    return v


def fix_local_roughness(cad_v, cad_f, mesh_v, mesh_f):
    """Adapt local spacing to match mesh roughness via NN averaging."""
    tree = _AKDTree(np.asarray(mesh_v))
    v = np.asarray(cad_v, dtype=np.float64).copy()
    dists, idx = tree.query(v, k=min(4, len(mesh_v)))
    weights = 1.0 / (dists + 1e-6)
    weights /= weights.sum(axis=1, keepdims=True)
    targets = np.sum(np.asarray(mesh_v)[idx] * weights[:, :, None], axis=1)
    v = v * 0.7 + targets * 0.3
    return v


def fix_volume(cad_v, cad_f, mesh_v, mesh_f):
    """Uniform scale to match volume."""
    def _vol(v, f):
        v, f = np.asarray(v), np.asarray(f)
        v0, v1, v2 = v[f[:, 0]], v[f[:, 1]], v[f[:, 2]]
        return abs(np.sum(v0 * np.cross(v1, v2))) / 6.0

    vol_cad = _vol(cad_v, cad_f)
    vol_mesh = _vol(mesh_v, mesh_f)
    if vol_cad > 1e-12:
        s = (vol_mesh / vol_cad) ** (1.0 / 3.0)
        centroid = np.mean(cad_v, axis=0)
        return centroid + (np.asarray(cad_v) - centroid) * s
    return np.asarray(cad_v).copy()


def fix_percentile_95(cad_v, cad_f, mesh_v, mesh_f):
    """Aggressively pull the worst 5% of vertices."""
    tree = _AKDTree(np.asarray(mesh_v))
    dists, idx = tree.query(np.asarray(cad_v))
    threshold = np.percentile(dists, 90)
    v = np.asarray(cad_v, dtype=np.float64).copy()
    outliers = dists > threshold
    v[outliers] = 0.3 * v[outliers] + 0.7 * np.asarray(mesh_v)[idx[outliers]]
    return v


def fix_face_normal_consistency(cad_v, cad_f, mesh_v, mesh_f):
    """Normal-guided vertex displacement."""
    return vertex_normal_realign(cad_v, cad_f, mesh_v, mesh_f, strength=0.5)


def fix_multi_scale_distance(cad_v, cad_f, mesh_v, mesh_f):
    """Progressive NN pull at coarse then fine scale."""
    v = np.asarray(cad_v, dtype=np.float64).copy()
    mv = np.asarray(mesh_v)
    tree = _AKDTree(mv)
    dists, idx = tree.query(v)
    v = v * 0.5 + mv[idx] * 0.5
    v = laplacian_smooth_toward(v, cad_f, mv, iterations=2,
                                 lam=0.2, target_weight=0.3)
    return v


def fix_shape_diameter(cad_v, cad_f, mesh_v, mesh_f):
    """Adjust thickness by scaling along vertex normals."""
    v = np.asarray(cad_v, dtype=np.float64).copy()
    normals = _compute_vertex_normals(v, np.asarray(cad_f))
    tree = _AKDTree(np.asarray(mesh_v))
    dists, idx = tree.query(v)
    targets = np.asarray(mesh_v)[idx]
    disp = targets - v
    normal_comp = np.sum(disp * normals, axis=1, keepdims=True)
    v += normals * normal_comp * 0.4
    return v


def fix_moment_of_inertia(cad_v, cad_f, mesh_v, mesh_f):
    """Per-axis scaling to match inertia tensor."""
    cad_v, mesh_v = np.asarray(cad_v), np.asarray(mesh_v)
    c_cad = cad_v.mean(axis=0)
    c_mesh = mesh_v.mean(axis=0)
    vc = cad_v - c_cad
    vm = mesh_v - c_mesh

    cad_spread = np.sqrt(np.mean(vc**2, axis=0))
    mesh_spread = np.sqrt(np.mean(vm**2, axis=0))
    cad_spread[cad_spread < 1e-12] = 1e-12
    scales = mesh_spread / cad_spread
    return c_mesh + vc * scales


def fix_distance_histogram(cad_v, cad_f, mesh_v, mesh_f):
    """Combined projection + smoothing."""
    v, _ = project_vertices_to_mesh(cad_v, mesh_v, mesh_f)
    v = laplacian_smooth_toward(v, cad_f, mesh_v,
                                 iterations=3, lam=0.3, target_weight=0.4)
    return v


# =========================================================================
# Extended differentiators (round 2)
# =========================================================================

def convexity_defect_diff(cad_v, mesh_v):
    """Compare convex hull volume ratio between two meshes.

    Measures how much of the convex hull is 'empty' — high for concave shapes.
    """
    from scipy.spatial import ConvexHull
    cad_v = np.asarray(cad_v, dtype=np.float64)
    mesh_v = np.asarray(mesh_v, dtype=np.float64)

    def _convexity(v):
        if len(v) < 4:
            return 1.0
        try:
            hull = ConvexHull(v)
            return hull.volume
        except Exception:
            return 0.0

    ch_cad = _convexity(cad_v)
    ch_mesh = _convexity(mesh_v)
    denom = max(ch_cad + ch_mesh, 1e-12)
    return float(abs(ch_cad - ch_mesh) / denom * 100)


def boundary_edge_length_diff(cad_v, cad_f, mesh_v, mesh_f):
    """Compare total boundary (non-manifold / open-edge) length."""
    def _boundary_length(v, f):
        v, f = np.asarray(v), np.asarray(f)
        edge_count = {}
        for tri in f:
            for i in range(3):
                e = tuple(sorted((int(tri[i]), int(tri[(i + 1) % 3]))))
                edge_count[e] = edge_count.get(e, 0) + 1
        total = 0.0
        for (a, b), cnt in edge_count.items():
            if cnt == 1:
                total += float(np.linalg.norm(v[a] - v[b]))
        return total

    bl_cad = _boundary_length(cad_v, cad_f)
    bl_mesh = _boundary_length(mesh_v, mesh_f)
    denom = max(bl_cad + bl_mesh, 1e-12)
    return float(abs(bl_cad - bl_mesh) / denom * 100)


def principal_curvature_ratio_diff(cad_v, cad_f, mesh_v, mesh_f, n_samples=200):
    """Compare distributions of the ratio of principal curvatures k1/k2."""
    def _curvature_ratios(v, f, n):
        v, f = np.asarray(v, dtype=np.float64), np.asarray(f)
        curv = _vertex_curvature(v, f)
        if len(curv) == 0:
            return np.array([1.0])
        # Use absolute curvature as proxy for k1; neighbor-mean as k2
        adj = _vertex_adjacency(f, len(v))
        ratios = []
        indices = np.random.RandomState(42).choice(
            len(v), min(n, len(v)), replace=False)
        for i in indices:
            nbrs = adj[i]
            if not nbrs:
                continue
            nbr_curv = np.mean([abs(curv[j]) for j in nbrs])
            if nbr_curv > 1e-8:
                ratios.append(abs(curv[i]) / nbr_curv)
        return np.array(ratios) if ratios else np.array([1.0])

    r_cad = _curvature_ratios(cad_v, cad_f, n_samples)
    r_mesh = _curvature_ratios(mesh_v, mesh_f, n_samples)
    return float(abs(np.median(r_cad) - np.median(r_mesh)) * 10)


def geodesic_diameter_diff(cad_v, cad_f, mesh_v, mesh_f):
    """Approximate geodesic diameter comparison via mesh graph BFS."""
    def _approx_geodesic_diam(v, f):
        v, f = np.asarray(v, dtype=np.float64), np.asarray(f)
        if len(v) < 2:
            return 0.0
        # Build adjacency with edge weights
        adj = {}
        for tri in f:
            for i in range(3):
                a, b = int(tri[i]), int(tri[(i + 1) % 3])
                d = float(np.linalg.norm(v[a] - v[b]))
                adj.setdefault(a, []).append((b, d))
                adj.setdefault(b, []).append((a, d))

        # BFS from vertex 0 to find farthest, then BFS from there
        def _bfs_farthest(start):
            import heapq
            dist = {start: 0.0}
            heap = [(0.0, start)]
            while heap:
                d, u = heapq.heappop(heap)
                if d > dist.get(u, float('inf')):
                    continue
                for nb, w in adj.get(u, []):
                    nd = d + w
                    if nd < dist.get(nb, float('inf')):
                        dist[nb] = nd
                        heapq.heappush(heap, (nd, nb))
            if not dist:
                return 0, 0.0
            farthest = max(dist, key=dist.get)
            return farthest, dist[farthest]

        _, d1 = _bfs_farthest(0)
        far1, _ = _bfs_farthest(0)
        _, diam = _bfs_farthest(far1)
        return diam

    d_cad = _approx_geodesic_diam(cad_v, cad_f)
    d_mesh = _approx_geodesic_diam(mesh_v, mesh_f)
    denom = max(d_cad + d_mesh, 1e-12)
    return float(abs(d_cad - d_mesh) / denom * 100)


def laplacian_spectrum_diff(cad_v, cad_f, mesh_v, mesh_f, n_eigenvalues=10):
    """Compare mesh Laplacian spectra (first n non-zero eigenvalues)."""
    def _laplacian_eigenvalues(v, f, n_eig):
        v, f = np.asarray(v, dtype=np.float64), np.asarray(f)
        n = len(v)
        if n < n_eig + 2:
            return np.zeros(n_eig)
        # Build combinatorial Laplacian
        adj = _vertex_adjacency(f, n)
        # Use dense for small meshes, subsample for large
        if n > 500:
            indices = np.random.RandomState(42).choice(n, 500, replace=False)
            v_sub = v[indices]
            tree = _AKDTree(v_sub)
            L = np.zeros((500, 500))
            for i in range(500):
                dists, nbrs = tree.query(v_sub[i], k=min(7, 500))
                for j, d in zip(nbrs[1:], dists[1:]):
                    if d > 0:
                        w = 1.0 / max(d, 1e-12)
                        L[i, j] -= w
                        L[j, i] -= w
                        L[i, i] += w
                        L[j, j] += w
        else:
            L = np.zeros((n, n))
            for i in range(n):
                for j in adj[i]:
                    d = max(np.linalg.norm(v[i] - v[j]), 1e-12)
                    w = 1.0 / d
                    L[i, j] -= w
                    L[i, i] += w

        eigvals = np.linalg.eigvalsh(L)
        # Skip zero eigenvalue(s)
        positive = eigvals[eigvals > 1e-8]
        if len(positive) < n_eig:
            result = np.zeros(n_eig)
            result[:len(positive)] = positive[:n_eig]
            return result
        return positive[:n_eig]

    eig_cad = _laplacian_eigenvalues(cad_v, cad_f, n_eigenvalues)
    eig_mesh = _laplacian_eigenvalues(mesh_v, mesh_f, n_eigenvalues)

    # Normalize by max eigenvalue
    max_eig = max(eig_cad.max(), eig_mesh.max(), 1e-12)
    return float(np.mean(np.abs(eig_cad - eig_mesh)) / max_eig * 100)


def face_area_variance_diff(cad_v, cad_f, mesh_v, mesh_f):
    """Compare face area uniformity (coefficient of variation)."""
    def _area_cv(v, f):
        areas = _face_areas(np.asarray(v), np.asarray(f))
        if len(areas) == 0 or areas.mean() < 1e-12:
            return 0.0
        return float(areas.std() / areas.mean())

    cv_cad = _area_cv(cad_v, cad_f)
    cv_mesh = _area_cv(mesh_v, mesh_f)
    return float(abs(cv_cad - cv_mesh) * 50)


def vertex_normal_divergence(cad_v, cad_f, mesh_v, mesh_f, n_samples=300):
    """Mean angle between CAD vertex normals and nearest-mesh vertex normals."""
    cad_v = np.asarray(cad_v, dtype=np.float64)
    mesh_v = np.asarray(mesh_v, dtype=np.float64)
    cad_f = np.asarray(cad_f)
    mesh_f = np.asarray(mesh_f)

    cn = _compute_vertex_normals(cad_v, cad_f)
    mn = _compute_vertex_normals(mesh_v, mesh_f)

    tree = _AKDTree(mesh_v)
    indices = np.random.RandomState(42).choice(
        len(cad_v), min(n_samples, len(cad_v)), replace=False)
    _, nbr_idx = tree.query(cad_v[indices])

    dots = np.sum(cn[indices] * mn[nbr_idx], axis=1)
    dots = np.clip(dots, -1.0, 1.0)
    angles = np.degrees(np.arccos(np.abs(dots)))
    return float(np.mean(angles))


def octant_volume_diff(cad_v, cad_f, mesh_v, mesh_f):
    """Compare volume distribution across 8 octants around centroid."""
    def _octant_volumes(v, f):
        v = np.asarray(v, dtype=np.float64)
        f = np.asarray(f)
        center = v.mean(axis=0)
        areas = _face_areas(v, f)
        centroids = (v[f[:, 0]] + v[f[:, 1]] + v[f[:, 2]]) / 3.0
        relative = centroids - center
        vols = np.zeros(8)
        for octant in range(8):
            sx = 1 if octant & 1 else -1
            sy = 1 if octant & 2 else -1
            sz = 1 if octant & 4 else -1
            mask = ((relative[:, 0] * sx >= 0) &
                    (relative[:, 1] * sy >= 0) &
                    (relative[:, 2] * sz >= 0))
            vols[octant] = float(areas[mask].sum())
        total = vols.sum()
        return vols / max(total, 1e-12)

    ov_cad = _octant_volumes(cad_v, cad_f)
    ov_mesh = _octant_volumes(mesh_v, mesh_f)
    return float(np.sum(np.abs(ov_cad - ov_mesh)) * 50)


def edge_angle_distribution_diff(cad_v, cad_f, mesh_v, mesh_f, n_bins=18):
    """Compare distributions of dihedral angles between adjacent faces."""
    def _dihedral_angles(v, f):
        v, f = np.asarray(v, dtype=np.float64), np.asarray(f)
        fn = np.cross(v[f[:, 1]] - v[f[:, 0]], v[f[:, 2]] - v[f[:, 0]])
        norms = np.linalg.norm(fn, axis=1, keepdims=True)
        norms[norms < 1e-12] = 1.0
        fn = fn / norms

        # Build edge→face map
        edge_faces = {}
        for fi in range(len(f)):
            for i in range(3):
                e = tuple(sorted((int(f[fi, i]), int(f[fi, (i + 1) % 3]))))
                edge_faces.setdefault(e, []).append(fi)

        angles = []
        for faces_list in edge_faces.values():
            if len(faces_list) == 2:
                dot = float(np.dot(fn[faces_list[0]], fn[faces_list[1]]))
                dot = max(-1.0, min(1.0, dot))
                angles.append(math.degrees(math.acos(dot)))
        return np.array(angles) if angles else np.array([0.0])

    ang_cad = _dihedral_angles(cad_v, cad_f)
    ang_mesh = _dihedral_angles(mesh_v, mesh_f)

    bins = np.linspace(0, 180, n_bins + 1)
    h_cad, _ = np.histogram(ang_cad, bins=bins, density=True)
    h_mesh, _ = np.histogram(ang_mesh, bins=bins, density=True)
    return float(np.sum(np.abs(h_cad - h_mesh)) * (bins[1] - bins[0]))


def aspect_ratio_diff(cad_v, cad_f, mesh_v, mesh_f):
    """Compare distributions of triangle aspect ratios (circumradius/inradius)."""
    def _aspect_ratios(v, f):
        v, f = np.asarray(v, dtype=np.float64), np.asarray(f)
        a = np.linalg.norm(v[f[:, 1]] - v[f[:, 0]], axis=1)
        b = np.linalg.norm(v[f[:, 2]] - v[f[:, 1]], axis=1)
        c = np.linalg.norm(v[f[:, 0]] - v[f[:, 2]], axis=1)
        s = (a + b + c) / 2
        area = np.sqrt(np.clip(s * (s - a) * (s - b) * (s - c), 0, None))
        # Aspect ratio = abc / (8 * area^2) * perimeter ... simplified:
        # Use longest_edge / shortest_altitude
        area[area < 1e-12] = 1e-12
        return (a * b * c) / (8 * area * area) * (a + b + c)

    ar_cad = _aspect_ratios(cad_v, cad_f)
    ar_mesh = _aspect_ratios(mesh_v, mesh_f)
    return float(abs(np.median(ar_cad) - np.median(ar_mesh)))


# =========================================================================
# Extended fixers (round 2)
# =========================================================================

def fix_convexity_defect(cad_v, cad_f, mesh_v, mesh_f):
    """Pull vertices toward convex hull where mesh is more convex."""
    from scipy.spatial import ConvexHull
    v = np.asarray(cad_v, dtype=np.float64).copy()
    mv = np.asarray(mesh_v)
    tree = _AKDTree(mv)
    dists, idx = tree.query(v)
    # Blend toward nearest mesh vertex
    v = v * 0.7 + mv[idx] * 0.3
    return v


def fix_boundary_edges(cad_v, cad_f, mesh_v, mesh_f):
    """Snap boundary (open-edge) vertices to nearest mesh vertices."""
    v = np.asarray(cad_v, dtype=np.float64).copy()
    f = np.asarray(cad_f)
    mv = np.asarray(mesh_v)

    # Find boundary vertices
    edge_count = {}
    for tri in f:
        for i in range(3):
            e = tuple(sorted((int(tri[i]), int(tri[(i + 1) % 3]))))
            edge_count[e] = edge_count.get(e, 0) + 1

    boundary_verts = set()
    for (a, b), cnt in edge_count.items():
        if cnt == 1:
            boundary_verts.add(a)
            boundary_verts.add(b)

    if not boundary_verts:
        return v

    bv_idx = np.array(sorted(boundary_verts))
    tree = _AKDTree(mv)
    _, nn_idx = tree.query(v[bv_idx])
    v[bv_idx] = v[bv_idx] * 0.4 + mv[nn_idx] * 0.6
    return v


def fix_principal_curvature(cad_v, cad_f, mesh_v, mesh_f):
    """Curvature-weighted vertex adjustment."""
    v = np.asarray(cad_v, dtype=np.float64).copy()
    mv = np.asarray(mesh_v)
    f = np.asarray(cad_f)
    curv = np.abs(_vertex_curvature(v, f))
    max_curv = max(curv.max(), 1e-12)
    weights = curv / max_curv  # High-curvature vertices get more correction

    tree = _AKDTree(mv)
    _, idx = tree.query(v)
    blend = 0.3 * weights[:, None]
    v = v * (1 - blend) + mv[idx] * blend
    return v


def fix_geodesic_diameter(cad_v, cad_f, mesh_v, mesh_f):
    """Scale mesh to match geodesic diameter via global uniform scale."""
    v = np.asarray(cad_v, dtype=np.float64).copy()
    mv = np.asarray(mesh_v)
    # Quick diameter proxy: max pairwise distance of a sample
    rng = np.random.RandomState(42)
    n = min(100, len(v), len(mv))
    idx_c = rng.choice(len(v), n, replace=len(v) < n)
    idx_m = rng.choice(len(mv), n, replace=len(mv) < n)

    def _max_dist(pts):
        d = np.linalg.norm(pts[:, None, :] - pts[None, :, :], axis=2)
        return d.max()

    d_cad = _max_dist(v[idx_c])
    d_mesh = _max_dist(mv[idx_m])
    if d_cad > 1e-6:
        s = np.clip(d_mesh / d_cad, 0.7, 1.5)
        center = v.mean(axis=0)
        v = center + (v - center) * s
    return v


def fix_laplacian_spectrum(cad_v, cad_f, mesh_v, mesh_f):
    """Multi-pass smoothing + projection to improve spectral match."""
    v = np.asarray(cad_v, dtype=np.float64).copy()
    mv = np.asarray(mesh_v)
    mf = np.asarray(mesh_f)
    f = np.asarray(cad_f)
    # Smooth then project twice
    for _ in range(2):
        v = laplacian_smooth_toward(v, f, mv,
                                     iterations=3, lam=0.4, target_weight=0.35)
        v, _ = project_vertices_to_mesh(v, mv, mf)
    return v


def fix_face_area_variance(cad_v, cad_f, mesh_v, mesh_f):
    """Equalize face areas by relaxing vertices toward neighbor centroids."""
    v = np.asarray(cad_v, dtype=np.float64).copy()
    f = np.asarray(cad_f)
    adj = _vertex_adjacency(f, len(v))

    for _ in range(3):
        new_v = v.copy()
        for i in range(len(v)):
            nbrs = list(adj[i])
            if not nbrs:
                continue
            nbr_center = v[nbrs].mean(axis=0)
            new_v[i] = v[i] * 0.7 + nbr_center * 0.3
        v = new_v

    # Pull back toward mesh
    tree = _AKDTree(np.asarray(mesh_v))
    _, idx = tree.query(v)
    v = v * 0.6 + np.asarray(mesh_v)[idx] * 0.4
    return v


def fix_vertex_normal_divergence(cad_v, cad_f, mesh_v, mesh_f):
    """Adjust vertices along normals to improve normal alignment."""
    v = np.asarray(cad_v, dtype=np.float64).copy()
    f = np.asarray(cad_f)
    mv = np.asarray(mesh_v)
    mf = np.asarray(mesh_f)

    cn = _compute_vertex_normals(v, f)
    mn = _compute_vertex_normals(mv, mf)

    tree = _AKDTree(mv)
    _, idx = tree.query(v)
    target_normals = mn[idx]

    # Displacement = difference projected onto vertex normal
    disp = mv[idx] - v
    normal_comp = np.sum(disp * cn, axis=1, keepdims=True)
    v += cn * normal_comp * 0.5
    return v


def fix_octant_volume(cad_v, cad_f, mesh_v, mesh_f):
    """Per-octant scaling to balance volume distribution."""
    v = np.asarray(cad_v, dtype=np.float64).copy()
    f = np.asarray(cad_f)
    mv = np.asarray(mesh_v)
    center = v.mean(axis=0)
    m_center = mv.mean(axis=0)

    for octant in range(8):
        sx = 1 if octant & 1 else -1
        sy = 1 if octant & 2 else -1
        sz = 1 if octant & 4 else -1

        c_mask = ((v[:, 0] - center[0]) * sx >= 0) & \
                 ((v[:, 1] - center[1]) * sy >= 0) & \
                 ((v[:, 2] - center[2]) * sz >= 0)
        m_mask = ((mv[:, 0] - m_center[0]) * sx >= 0) & \
                 ((mv[:, 1] - m_center[1]) * sy >= 0) & \
                 ((mv[:, 2] - m_center[2]) * sz >= 0)

        if np.sum(c_mask) < 2 or np.sum(m_mask) < 2:
            continue

        c_spread = np.linalg.norm(v[c_mask] - center, axis=1).mean()
        m_spread = np.linalg.norm(mv[m_mask] - m_center, axis=1).mean()
        if c_spread > 1e-6:
            s = np.clip(m_spread / c_spread, 0.7, 1.5)
            v[c_mask] = center + (v[c_mask] - center) * s

    return v


def fix_edge_angle_distribution(cad_v, cad_f, mesh_v, mesh_f):
    """Smooth sharp edges and project to mesh for better dihedral match."""
    v = np.asarray(cad_v, dtype=np.float64).copy()
    f = np.asarray(cad_f)
    mv = np.asarray(mesh_v)
    mf = np.asarray(mesh_f)
    v = laplacian_smooth_toward(v, f, mv,
                                 iterations=4, lam=0.35, target_weight=0.3)
    v, _ = project_vertices_to_mesh(v, mv, mf)
    return v


def fix_aspect_ratio(cad_v, cad_f, mesh_v, mesh_f):
    """Improve triangle quality by relaxing toward neighbor centroids + mesh."""
    v = np.asarray(cad_v, dtype=np.float64).copy()
    f = np.asarray(cad_f)
    mv = np.asarray(mesh_v)
    adj = _vertex_adjacency(f, len(v))

    for _ in range(2):
        new_v = v.copy()
        for i in range(len(v)):
            nbrs = list(adj[i])
            if not nbrs:
                continue
            nbr_center = v[nbrs].mean(axis=0)
            new_v[i] = v[i] * 0.75 + nbr_center * 0.25
        v = new_v

    tree = _AKDTree(mv)
    _, idx = tree.query(v)
    v = v * 0.5 + mv[idx] * 0.5
    return v
