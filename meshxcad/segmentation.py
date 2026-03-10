"""Mesh segmentation engine for tracing-based CAD reconstruction.

Decomposes a triangle mesh into parts that can each be reproduced by a
small number of CAD operations (extrude, revolve, loft/sweep).

Segmentation strategies (selected by template or auto-detected):
  - skeleton   : medial-axis skeleton → branch decomposition
  - sdf        : shape-diameter-function → thin/thick clustering
  - convexity  : approximate convex decomposition via face normals
  - projection : 2D projection silhouette analysis
  - normal_cluster : face-normal k-means clustering

Each segment is annotated with a recommended CAD action and a quality
score indicating how well that action can reproduce the geometry.
"""

import math
import numpy as np
from scipy.spatial import KDTree
from .gpu import AcceleratedKDTree as _AKDTree
from dataclasses import dataclass, field
from typing import List, Optional, Tuple


@dataclass
class MeshSegment:
    """One segment of a decomposed mesh, with CAD reconstruction metadata."""
    segment_id: int
    vertices: np.ndarray       # (N, 3)
    faces: np.ndarray          # (M, 3)  — indices into self.vertices
    centroid: np.ndarray       # (3,)
    primary_axis: np.ndarray   # (3,) — dominant direction
    cad_action: str            # extrude, revolve, loft, sweep, freeform
    path: Optional[np.ndarray] = None    # (K, 3) sweep/loft path
    profile: Optional[np.ndarray] = None # (P, 2) 2D cross-section
    scale_along_path: Optional[np.ndarray] = None  # (K,) per-path-point scale
    quality: float = 0.0
    parent_id: Optional[int] = None
    label: str = ""
    is_fillet: bool = False    # True if this segment is a blend/fillet region


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def _face_normals(vertices, faces):
    """Compute per-face unit normals."""
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]
    n = np.cross(v1 - v0, v2 - v0)
    norms = np.linalg.norm(n, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    return n / norms


def _face_centroids(vertices, faces):
    """Compute per-face centroids."""
    return (vertices[faces[:, 0]] + vertices[faces[:, 1]] + vertices[faces[:, 2]]) / 3.0


def _face_areas(vertices, faces):
    """Compute per-face areas."""
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]
    return 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0), axis=1)


def _build_adjacency(faces, n_faces):
    """Build face adjacency via shared edges.  Returns dict {face_idx: set(neighbor_idxs)}."""
    edge_to_faces = {}
    for fi in range(n_faces):
        for ei in range(3):
            e = tuple(sorted((int(faces[fi, ei]), int(faces[fi, (ei + 1) % 3]))))
            edge_to_faces.setdefault(e, []).append(fi)

    adj = {i: set() for i in range(n_faces)}
    for flist in edge_to_faces.values():
        for i in range(len(flist)):
            for j in range(i + 1, len(flist)):
                adj[flist[i]].add(flist[j])
                adj[flist[j]].add(flist[i])
    return adj


def _extract_submesh(vertices, faces, face_mask):
    """Extract a submesh for selected faces, re-indexing vertices."""
    sel_faces = faces[face_mask]
    unique_verts = np.unique(sel_faces.ravel())
    vmap = np.full(len(vertices), -1, dtype=np.int64)
    vmap[unique_verts] = np.arange(len(unique_verts))
    new_verts = vertices[unique_verts]
    new_faces = vmap[sel_faces]
    return new_verts, new_faces


# ---------------------------------------------------------------------------
# Strategy: skeleton-based segmentation
# ---------------------------------------------------------------------------

def _extract_skeleton(vertices, faces, n_samples=500):
    """Extract a medial-axis-like skeleton from a mesh.

    Uses PCA-guided slicing: project vertices onto the principal axis,
    sample cross-sectional centroids, and connect them.

    Returns:
        path: (K, 3) skeleton path points
        assignments: (N_faces,) which path segment each face belongs to
    """
    v = np.asarray(vertices, dtype=np.float64)
    center = v.mean(axis=0)
    centered = v - center

    cov = np.cov(centered.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    # Primary axis = largest eigenvalue
    primary = eigvecs[:, np.argmax(eigvals)]

    # Project onto primary axis
    proj = centered @ primary  # (N,)

    p_min, p_max = proj.min(), proj.max()
    n_slices = min(n_samples, max(10, len(v) // 50))
    t_values = np.linspace(p_min, p_max, n_slices)

    # For each slice, find nearby vertices and compute centroid
    path_points = []
    half_width = (p_max - p_min) / n_slices * 1.5

    for t in t_values:
        mask = np.abs(proj - t) < half_width
        if mask.sum() < 3:
            continue
        local_center = v[mask].mean(axis=0)
        path_points.append(local_center)

    if len(path_points) < 2:
        # Fallback: just use centroid
        path_points = [v.mean(axis=0)]

    path = np.array(path_points)

    # Assign faces to nearest path point
    fc = _face_centroids(vertices, faces)
    tree = _AKDTree(path)
    _, assignments = tree.query(fc)

    return path, assignments


def segment_by_skeleton(vertices, faces, min_segment_faces=20):
    """Segment mesh by skeleton branching.

    Extracts a skeleton, finds branches via path discontinuities and
    face-normal coherence, and groups faces into segments.

    Returns list of MeshSegment.
    """
    v = np.asarray(vertices, dtype=np.float64)
    f = np.asarray(faces)
    n_faces = len(f)

    path, assignments = _extract_skeleton(v, f)

    # Group consecutive path assignments into segments via connected components
    # with normal coherence
    normals = _face_normals(v, f)
    adj = _build_adjacency(f, n_faces)

    # Region-growing segmentation seeded by path groups
    n_path = len(path)
    # Merge path points that have very similar face populations
    labels = np.full(n_faces, -1, dtype=np.int64)
    segment_id = 0
    visited = np.zeros(n_faces, dtype=bool)

    # Sort faces by path assignment, process contiguous groups
    for pi in range(n_path):
        seed_faces = np.where((assignments == pi) & (~visited))[0]
        if len(seed_faces) == 0:
            continue

        # Region-grow from these seeds using normal coherence
        queue = list(seed_faces)
        region = set(seed_faces.tolist())
        visited[seed_faces] = True
        head = 0

        while head < len(queue):
            fi = queue[head]
            head += 1
            for ni in adj.get(fi, set()):
                if visited[ni]:
                    continue
                # Check normal similarity
                cos_angle = np.dot(normals[fi], normals[ni])
                if cos_angle > 0.3:  # ~73 degrees
                    visited[ni] = True
                    region.add(ni)
                    queue.append(ni)

        if len(region) >= min_segment_faces:
            for fi in region:
                labels[fi] = segment_id
            segment_id += 1

    # Assign orphan faces to nearest segment
    orphans = np.where(labels == -1)[0]
    if len(orphans) > 0 and segment_id > 0:
        fc = _face_centroids(v, f)
        labeled_centroids = []
        labeled_ids = []
        for si in range(segment_id):
            seg_mask = labels == si
            if seg_mask.any():
                labeled_centroids.append(fc[seg_mask].mean(axis=0))
                labeled_ids.append(si)
        if labeled_centroids:
            tree = _AKDTree(np.array(labeled_centroids))
            _, nearest = tree.query(fc[orphans])
            for oi, ni in zip(orphans, nearest):
                labels[oi] = labeled_ids[ni]

    # If no segments were created, put everything in one segment
    if segment_id == 0:
        labels[:] = 0
        segment_id = 1

    # Build MeshSegment objects
    segments = []
    for si in range(segment_id):
        mask = labels == si
        if mask.sum() == 0:
            continue
        seg_v, seg_f = _extract_submesh(v, f, mask)
        seg_centroid = seg_v.mean(axis=0)

        # Determine primary axis via PCA
        centered = seg_v - seg_centroid
        if len(centered) >= 3:
            cov = np.cov(centered.T)
            evals, evecs = np.linalg.eigh(cov)
            primary_axis = evecs[:, np.argmax(evals)]
        else:
            primary_axis = np.array([0, 0, 1.0])

        segments.append(MeshSegment(
            segment_id=si,
            vertices=seg_v,
            faces=seg_f,
            centroid=seg_centroid,
            primary_axis=primary_axis,
            cad_action="loft",  # default, refined later
            label=f"segment_{si}",
        ))

    return segments


# ---------------------------------------------------------------------------
# Strategy: SDF (shape diameter function) segmentation
# ---------------------------------------------------------------------------

def _compute_sdf(vertices, faces, n_rays=10):
    """Approximate shape-diameter function per face.

    For each face, cast rays inward (opposite to normal) and measure the
    distance to the opposite side of the mesh.
    """
    fc = _face_centroids(vertices, faces)
    fn = _face_normals(vertices, faces)
    n_faces = len(faces)

    tree = _AKDTree(fc)

    sdf_values = np.zeros(n_faces)
    for i in range(n_faces):
        # Inward direction
        inward = -fn[i]

        # Query faces in the inward direction by shooting along -normal
        # Approximate: find faces whose centroids are roughly opposite
        dist_k = min(50, n_faces)
        dists, idxs = tree.query(fc[i], k=dist_k)

        max_d = 0.0
        count = 0
        for j, idx in enumerate(idxs):
            if idx == i:
                continue
            direction = fc[idx] - fc[i]
            d = np.linalg.norm(direction)
            if d < 1e-12:
                continue
            direction_norm = direction / d
            # Must be roughly in the inward direction
            if np.dot(direction_norm, inward) > 0.3:
                # And the opposite face should be facing us
                if np.dot(fn[idx], inward) < -0.1:
                    max_d += d
                    count += 1
                    if count >= n_rays:
                        break

        sdf_values[i] = max_d / max(count, 1)

    return sdf_values


def segment_by_sdf(vertices, faces, n_clusters=None, min_segment_faces=20):
    """Segment mesh by shape-diameter function (thin vs thick regions).

    Returns list of MeshSegment.
    """
    v = np.asarray(vertices, dtype=np.float64)
    f = np.asarray(faces)
    n_faces = len(f)

    sdf = _compute_sdf(v, f)

    # Determine number of clusters from SDF distribution
    if n_clusters is None:
        # Use Sturges' rule as starting point, cap at 8
        n_clusters = min(8, max(2, int(1 + 3.322 * np.log10(max(n_faces, 1)))))
        n_clusters = min(n_clusters, n_faces // max(min_segment_faces, 1))
        n_clusters = max(2, n_clusters)

    # Simple k-means on SDF values
    labels = _kmeans_1d(sdf, n_clusters)

    # Refine with connected components + adjacency
    adj = _build_adjacency(f, n_faces)
    labels = _connected_component_refine(labels, adj, n_faces, min_segment_faces)

    return _labels_to_segments(v, f, labels)


def _kmeans_1d(values, k, max_iter=20):
    """1D k-means clustering."""
    n = len(values)
    # Initialize centers evenly across range
    vmin, vmax = values.min(), values.max()
    if vmax - vmin < 1e-12:
        return np.zeros(n, dtype=np.int64)

    centers = np.linspace(vmin, vmax, k)
    labels = np.zeros(n, dtype=np.int64)

    for _ in range(max_iter):
        # Assign
        dists = np.abs(values[:, None] - centers[None, :])
        new_labels = dists.argmin(axis=1)

        if np.array_equal(new_labels, labels):
            break
        labels = new_labels

        # Update centers
        for ci in range(k):
            mask = labels == ci
            if mask.any():
                centers[ci] = values[mask].mean()

    return labels


# ---------------------------------------------------------------------------
# Strategy: convexity-based segmentation
# ---------------------------------------------------------------------------

def segment_by_convexity(vertices, faces, concavity_threshold=0.5,
                          min_segment_faces=20):
    """Segment mesh by approximate convex decomposition.

    Identifies concave edges (high dihedral angle) and uses them as
    cut boundaries for region growing.
    """
    v = np.asarray(vertices, dtype=np.float64)
    f = np.asarray(faces)
    n_faces = len(f)
    normals = _face_normals(v, f)

    adj = _build_adjacency(f, n_faces)

    # Compute dihedral angles at shared edges
    concave_pairs = set()
    for fi in range(n_faces):
        for ni in adj.get(fi, set()):
            if ni <= fi:
                continue
            cos_a = np.dot(normals[fi], normals[ni])
            if cos_a < concavity_threshold:  # concave edge
                concave_pairs.add((fi, ni))
                concave_pairs.add((ni, fi))

    # Region growing, not crossing concave edges
    labels = np.full(n_faces, -1, dtype=np.int64)
    segment_id = 0

    for seed in range(n_faces):
        if labels[seed] >= 0:
            continue

        queue = [seed]
        region = [seed]
        labels[seed] = segment_id
        head = 0

        while head < len(queue):
            fi = queue[head]
            head += 1
            for ni in adj.get(fi, set()):
                if labels[ni] >= 0:
                    continue
                if (fi, ni) in concave_pairs:
                    continue  # Don't cross concave boundary
                labels[ni] = segment_id
                region.append(ni)
                queue.append(ni)

        segment_id += 1

    labels = _connected_component_refine(labels, adj, n_faces, min_segment_faces)
    return _labels_to_segments(v, f, labels)


# ---------------------------------------------------------------------------
# Strategy: normal clustering
# ---------------------------------------------------------------------------

def segment_by_normals(vertices, faces, n_clusters=6, min_segment_faces=20):
    """Segment mesh by face-normal k-means clustering.

    Groups faces with similar normals, then refines with connected components.
    """
    v = np.asarray(vertices, dtype=np.float64)
    f = np.asarray(faces)
    n_faces = len(f)
    normals = _face_normals(v, f)

    # K-means on 3D normals
    n_clusters = min(n_clusters, n_faces // max(min_segment_faces, 1))
    n_clusters = max(2, n_clusters)

    # Initialize with evenly spaced directions
    rng = np.random.RandomState(42)
    centers = rng.randn(n_clusters, 3)
    centers /= np.linalg.norm(centers, axis=1, keepdims=True)

    labels = np.zeros(n_faces, dtype=np.int64)
    for _ in range(30):
        # Assign — use absolute dot product (normals can be flipped)
        dots = np.abs(normals @ centers.T)
        new_labels = dots.argmax(axis=1)
        if np.array_equal(new_labels, labels):
            break
        labels = new_labels
        for ci in range(n_clusters):
            mask = labels == ci
            if mask.any():
                avg = normals[mask].mean(axis=0)
                norm = np.linalg.norm(avg)
                centers[ci] = avg / max(norm, 1e-12)

    adj = _build_adjacency(f, n_faces)
    labels = _connected_component_refine(labels, adj, n_faces, min_segment_faces)
    return _labels_to_segments(v, f, labels)


# ---------------------------------------------------------------------------
# Strategy: projection-based segmentation
# ---------------------------------------------------------------------------

def segment_by_projection(vertices, faces, min_segment_faces=20):
    """Segment mesh by analyzing 2D projections along principal axes.

    Projects the mesh onto XY, XZ, YZ planes, identifies connected
    silhouette regions, and maps back to 3D faces.
    """
    v = np.asarray(vertices, dtype=np.float64)
    f = np.asarray(faces)
    n_faces = len(f)

    fc = _face_centroids(v, f)
    center = v.mean(axis=0)
    centered = v - center

    cov = np.cov(centered.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    # Sort by eigenvalue descending
    order = np.argsort(eigvals)[::-1]
    eigvecs = eigvecs[:, order]

    # Project face centroids onto the two minor axes (perpendicular to primary)
    proj2d = (fc - center) @ eigvecs[:, 1:3]  # (N_faces, 2)

    # Grid-based clustering in projection space
    bbox_range = proj2d.max(axis=0) - proj2d.min(axis=0)
    grid_res = max(bbox_range) / 20  # ~20 grid cells across
    if grid_res < 1e-12:
        grid_res = 1.0

    grid_coords = ((proj2d - proj2d.min(axis=0)) / grid_res).astype(int)

    # Hash grid cells, group faces
    cell_to_faces = {}
    for fi in range(n_faces):
        key = (int(grid_coords[fi, 0]), int(grid_coords[fi, 1]))
        cell_to_faces.setdefault(key, []).append(fi)

    # Connected-component flood fill on grid
    cell_labels = {}
    label = 0
    for key in cell_to_faces:
        if key in cell_labels:
            continue
        queue = [key]
        cell_labels[key] = label
        head = 0
        while head < len(queue):
            cx, cy = queue[head]
            head += 1
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nk = (cx + dx, cy + dy)
                if nk in cell_to_faces and nk not in cell_labels:
                    cell_labels[nk] = label
                    queue.append(nk)
        label += 1

    face_labels = np.zeros(n_faces, dtype=np.int64)
    for key, faces_in_cell in cell_to_faces.items():
        lbl = cell_labels[key]
        for fi in faces_in_cell:
            face_labels[fi] = lbl

    adj = _build_adjacency(f, n_faces)
    face_labels = _connected_component_refine(face_labels, adj, n_faces, min_segment_faces)
    return _labels_to_segments(v, f, face_labels)


# ---------------------------------------------------------------------------
# Shared refinement helpers
# ---------------------------------------------------------------------------

def _connected_component_refine(labels, adj, n_faces, min_segment_faces):
    """Split labels into true connected components and merge tiny segments."""
    # Split each label into connected components
    new_labels = np.full(n_faces, -1, dtype=np.int64)
    new_id = 0

    for old_label in np.unique(labels):
        faces_in_label = np.where(labels == old_label)[0]
        visited = set()

        for seed in faces_in_label:
            if seed in visited:
                continue
            # BFS
            component = []
            queue = [seed]
            visited.add(seed)
            head = 0
            while head < len(queue):
                fi = queue[head]
                head += 1
                component.append(fi)
                for ni in adj.get(fi, set()):
                    if ni not in visited and labels[ni] == old_label:
                        visited.add(ni)
                        queue.append(ni)

            for fi in component:
                new_labels[fi] = new_id
            new_id += 1

    # Merge tiny segments into nearest neighbor
    seg_sizes = {}
    for i in range(new_id):
        seg_sizes[i] = (new_labels == i).sum()

    fc = None  # lazy-compute only if needed
    for si in list(seg_sizes.keys()):
        if seg_sizes[si] < min_segment_faces and seg_sizes[si] > 0:
            if fc is None:
                # We need face centroids, but we don't have vertices here.
                # Instead merge into the most common adjacent label.
                pass
            faces_in_seg = np.where(new_labels == si)[0]
            neighbor_counts = {}
            for fi in faces_in_seg:
                for ni in adj.get(fi, set()):
                    nl = new_labels[ni]
                    if nl != si and nl >= 0:
                        neighbor_counts[nl] = neighbor_counts.get(nl, 0) + 1
            if neighbor_counts:
                best = max(neighbor_counts, key=neighbor_counts.get)
                new_labels[faces_in_seg] = best

    # Re-number contiguously
    unique_labels = np.unique(new_labels)
    remap = {old: new for new, old in enumerate(unique_labels)}
    for i in range(n_faces):
        new_labels[i] = remap[new_labels[i]]

    return new_labels


def _labels_to_segments(vertices, faces, labels):
    """Convert face labels to list of MeshSegment."""
    segments = []
    for si in np.unique(labels):
        mask = labels == si
        seg_v, seg_f = _extract_submesh(vertices, faces, mask)
        if len(seg_v) < 3 or len(seg_f) == 0:
            continue
        seg_centroid = seg_v.mean(axis=0)

        centered = seg_v - seg_centroid
        if len(centered) >= 3:
            cov = np.cov(centered.T)
            evals, evecs = np.linalg.eigh(cov)
            primary_axis = evecs[:, np.argmax(evals)]
        else:
            primary_axis = np.array([0, 0, 1.0])

        segments.append(MeshSegment(
            segment_id=int(si),
            vertices=seg_v,
            faces=seg_f,
            centroid=seg_centroid,
            primary_axis=primary_axis,
            cad_action="freeform",
            label=f"segment_{si}",
        ))

    return segments


# ---------------------------------------------------------------------------
# CAD action classification for segments
# ---------------------------------------------------------------------------

def classify_segment_action(segment):
    """Determine the best CAD action for a segment.

    Analyzes the segment geometry and assigns one of:
        revolve, extrude, loft, sweep, freeform

    Also populates segment.path, segment.profile, segment.scale_along_path
    when applicable.
    """
    from .revolve_align import detect_revolve_axis
    from .extrude_align import detect_sweep_candidate

    v = segment.vertices
    f = segment.faces

    if len(v) < 10 or len(f) < 5:
        segment.cad_action = "freeform"
        segment.quality = 0.3
        return segment

    # Test revolve suitability
    axis_info = detect_revolve_axis(v)
    circularity = axis_info["circularity"]

    # Test sweep suitability
    sweep_info = detect_sweep_candidate(v, f)
    is_sweep = sweep_info.get("is_sweep", False)

    # Decision tree
    if circularity > 0.7:
        segment.cad_action = "revolve"
        segment.primary_axis = axis_info["axis_direction"]
        segment.quality = min(1.0, circularity)
        _extract_revolve_profile(segment)
    elif is_sweep and sweep_info.get("consistency", 0) > 0.5:
        segment.cad_action = "sweep"
        segment.quality = sweep_info.get("consistency", 0.5)
        _extract_sweep_path_and_profile(segment)
    elif _is_extrudable(v, f):
        segment.cad_action = "extrude"
        segment.quality = 0.7
        _extract_extrude_profile(segment)
    else:
        # Default: loft along primary axis with varying cross-sections
        segment.cad_action = "loft"
        segment.quality = 0.5
        _extract_loft_path_and_profiles(segment)

    return segment


def _is_extrudable(vertices, faces, threshold=0.8):
    """Check if a segment has consistent cross-sections (extrudable)."""
    v = np.asarray(vertices)
    center = v.mean(axis=0)
    centered = v - center

    cov = np.cov(centered.T)
    evals, evecs = np.linalg.eigh(cov)
    order = np.argsort(evals)[::-1]
    evals = evals[order]

    if evals[0] < 1e-12:
        return False

    # Elongation along primary axis
    elongation = np.sqrt(evals[0] / max(evals[1], 1e-12))
    if elongation < 1.3:
        return False

    # Check cross-section consistency
    axis = evecs[:, order[0]]
    proj = centered @ axis
    n_slices = 5
    t_vals = np.linspace(np.percentile(proj, 10), np.percentile(proj, 90), n_slices)
    half_w = (proj.max() - proj.min()) / n_slices * 1.2

    areas = []
    for t in t_vals:
        mask = np.abs(proj - t) < half_w
        if mask.sum() < 3:
            continue
        local = centered[mask]
        # Project onto perpendicular plane
        perp = local - np.outer(local @ axis, axis)
        area = _convex_hull_area_2d(perp, evecs[:, order[1]], evecs[:, order[2]])
        areas.append(area)

    if len(areas) < 3:
        return False

    areas = np.array(areas)
    mean_area = areas.mean()
    if mean_area < 1e-12:
        return False

    consistency = 1.0 - min(areas.std() / mean_area, 1.0)
    return consistency > threshold


def _convex_hull_area_2d(points_3d, u_axis, v_axis):
    """Approximate area of points projected onto a 2D plane."""
    u = points_3d @ u_axis
    v = points_3d @ v_axis
    return (u.max() - u.min()) * (v.max() - v.min())


def _extract_revolve_profile(segment):
    """Extract a 2D (r, z) revolve profile for a segment."""
    from .revolve_align import extract_radial_profile

    profile = extract_radial_profile(segment.vertices, n_slices=30)
    if profile is not None and len(profile) >= 2:
        segment.profile = profile


def _extract_extrude_profile(segment):
    """Extract a 2D cross-section profile for extrusion."""
    from .extrude_align import extract_cross_section

    v = segment.vertices
    f = segment.faces
    center = v.mean(axis=0)

    # Extract cross-section at midpoint along primary axis
    proj = (v - center) @ segment.primary_axis
    mid_height = center @ np.array([0, 0, 1]) + np.median(proj)

    profile = extract_cross_section(v, f, mid_height)
    if profile is not None and len(profile) >= 3:
        segment.profile = profile


def _extract_sweep_path_and_profile(segment):
    """Extract sweep path and cross-section with scaling."""
    from .extrude_align import extract_mesh_skeleton, extract_sweep_cross_section

    v = segment.vertices
    f = segment.faces

    # Extract skeleton path
    path = extract_mesh_skeleton(v, f)
    if path is None or len(path) < 2:
        return

    segment.path = path

    # Extract cross-sections along path and compute scale factors
    scales = []
    profile_ref = None
    for i in range(len(path)):
        if i == 0:
            tangent = path[min(1, len(path) - 1)] - path[0]
        else:
            tangent = path[i] - path[i - 1]
        tn = np.linalg.norm(tangent)
        if tn < 1e-12:
            scales.append(1.0)
            continue
        tangent = tangent / tn

        cs = extract_sweep_cross_section(v, f, path[i], tangent)
        if cs is not None and len(cs) >= 3:
            # Compute "radius" as max distance from centroid
            cs_center = cs.mean(axis=0)
            radii = np.linalg.norm(cs - cs_center, axis=1)
            r = float(np.median(radii))

            if profile_ref is None:
                profile_ref = cs
                scales.append(1.0)
            else:
                ref_center = profile_ref.mean(axis=0)
                ref_radii = np.linalg.norm(profile_ref - ref_center, axis=1)
                ref_r = float(np.median(ref_radii))
                scales.append(r / max(ref_r, 1e-12))
        else:
            scales.append(scales[-1] if scales else 1.0)

    segment.scale_along_path = np.array(scales)
    if profile_ref is not None:
        segment.profile = profile_ref


def _extract_loft_path_and_profiles(segment):
    """Extract a loft path (centerline) with per-station cross-sections."""
    v = segment.vertices
    center = v.mean(axis=0)
    centered = v - center
    axis = segment.primary_axis

    proj = centered @ axis
    n_stations = min(10, max(3, len(v) // 30))
    t_vals = np.linspace(proj.min(), proj.max(), n_stations)
    half_w = (proj.max() - proj.min()) / n_stations * 1.5

    path_pts = []
    scales = []

    for t in t_vals:
        mask = np.abs(proj - t) < half_w
        if mask.sum() < 3:
            continue
        local = v[mask]
        local_center = local.mean(axis=0)
        path_pts.append(local_center)

        radii = np.linalg.norm(local - local_center, axis=1)
        scales.append(float(np.median(radii)))

    if len(path_pts) >= 2:
        segment.path = np.array(path_pts)
        scales = np.array(scales)
        if scales[0] > 1e-12:
            segment.scale_along_path = scales / scales[0]
        else:
            segment.scale_along_path = np.ones(len(scales))


# ---------------------------------------------------------------------------
# Fillet / blend detection
# ---------------------------------------------------------------------------

def detect_fillets(vertices, faces, segments):
    """Detect fillet/blend regions between CAD primitives.

    Fillets are transition surfaces with roughly triangular cross-section
    extruded along the path where two geometric primitives intersect.
    They blend the sharp intersection edge into a smooth curve.

    Types of fillets handled:
    - **Concave fillets**: Fill inner corners (e.g., cylinder meeting a plate).
      Cross-section is concave-triangular.
    - **Convex fillets**: Round outer corners (e.g., chamfered cube edges).
      Cross-section is convex-triangular.
    - **Variable fillets**: Around curved intersections (e.g., sphere embedded
      in a cube face).  The fillet curves along the intersection path but may
      be straight in some sections.

    The detection is two-phase:
    Phase 1 (curvature): Identify high-curvature band regions (narrow strips
      of faces with elevated curvature relative to the mesh average).
    Phase 2 (topology): Verify these bands border two different segments
      (i.e., they're transitions between primitives, not features of a
      single primitive like a cone tip).

    Segments marked is_fillet=True should be excluded from primitive fitting
    because fillet geometry masks the underlying sharp-edge intersection.

    Returns:
        fillet_face_mask: boolean array (n_faces,) — True for fillet faces
    """
    v = np.asarray(vertices, dtype=np.float64)
    f = np.asarray(faces)
    n_verts = len(v)
    n_faces = len(f)

    if n_faces < 10 or n_verts < 10:
        return np.zeros(n_faces, dtype=bool)

    # Phase 1: Curvature-based fillet candidate detection
    # Compute per-vertex discrete curvature (angle deficit)
    angle_sum = np.zeros(n_verts)
    for tri in f:
        for k in range(3):
            i, j, l = int(tri[k]), int(tri[(k + 1) % 3]), int(tri[(k + 2) % 3])
            v1 = v[j] - v[i]
            v2 = v[l] - v[i]
            len1 = np.linalg.norm(v1)
            len2 = np.linalg.norm(v2)
            if len1 > 1e-12 and len2 > 1e-12:
                cos_a = np.clip(np.dot(v1, v2) / (len1 * len2), -1, 1)
                angle_sum[i] += math.acos(cos_a)

    curvature = np.abs(2 * math.pi - angle_sum)

    if np.max(curvature) < 1e-8:
        return np.zeros(n_faces, dtype=bool)

    curv_median = np.median(curvature)
    curv_std = max(np.std(curvature), 1e-8)
    curv_threshold = curv_median + 1.5 * curv_std
    high_curv_mask = curvature > curv_threshold

    if high_curv_mask.sum() < 3:
        return np.zeros(n_faces, dtype=bool)

    # Identify fillet faces: >= 2 of 3 vertices are high-curvature
    face_curv_count = np.zeros(n_faces, dtype=int)
    for fi in range(n_faces):
        for vi in f[fi]:
            if high_curv_mask[vi]:
                face_curv_count[fi] += 1
    fillet_face_mask = face_curv_count >= 2

    # Avoid marking > 30% of faces as fillet
    if fillet_face_mask.sum() / max(n_faces, 1) > 0.3:
        curv_threshold = curv_median + 2.5 * curv_std
        high_curv_mask = curvature > curv_threshold
        face_curv_count = np.zeros(n_faces, dtype=int)
        for fi in range(n_faces):
            for vi in f[fi]:
                if high_curv_mask[vi]:
                    face_curv_count[fi] += 1
        fillet_face_mask = face_curv_count >= 2

    # Phase 2: Topological validation — confirm fillet segments border
    # multiple other segments (true fillets are transitions between
    # two distinct primitives)
    global_tree = _AKDTree(v)

    for seg in segments:
        if len(seg.faces) < 3 or len(seg.vertices) < 3:
            continue

        seg_v = seg.vertices
        _, nearest = global_tree.query(seg_v)
        seg_curvatures = curvature[nearest]
        high_frac = float(np.mean(seg_curvatures > curv_threshold))
        is_small = len(seg_v) / max(n_verts, 1) < 0.15

        if not (high_frac > 0.4 and is_small):
            continue

        # Check topology: does this segment border multiple other segments?
        # A true fillet borders at least 2 other segments.
        # Use centroid proximity to count neighboring segments.
        seg_center = seg.centroid
        neighbor_ids = set()
        for other in segments:
            if other.segment_id == seg.segment_id:
                continue
            if other.is_fillet:
                continue
            dist = float(np.linalg.norm(seg_center - other.centroid))
            # Also check if any vertices are close
            if len(other.vertices) > 0:
                other_tree = _AKDTree(other.vertices)
                d, _ = other_tree.query(seg_v)
                close_count = (d < np.linalg.norm(
                    v.max(0) - v.min(0)) * 0.05).sum()
                if close_count > 1:
                    neighbor_ids.add(other.segment_id)

        # Check aspect ratio: fillets are typically narrow bands
        centered = seg_v - seg_center
        if len(centered) >= 3:
            cov = centered.T @ centered / len(centered)
            eigvals = np.sort(np.linalg.eigvalsh(cov))[::-1]
            spreads = np.sqrt(np.maximum(eigvals, 0))
            # Elongated = high first-to-second spread ratio
            if spreads[1] > 1e-12:
                elongation = spreads[0] / spreads[1]
            else:
                elongation = 10.0
        else:
            elongation = 1.0

        # Mark as fillet if:
        # - High curvature fraction (>40%)
        # - Small relative to mesh (<15% of vertices)
        # - Borders 2+ other segments OR is highly elongated (band-like)
        if len(neighbor_ids) >= 2 or (elongation > 2.0 and high_frac > 0.5):
            seg.is_fillet = True
            seg.cad_action = "fillet"
            seg.label = f"fillet_{seg.segment_id}"

    return fillet_face_mask


def detect_intersection_fillets(program, target_v, target_f):
    """Post-fitting fillet detection: find blend regions at primitive intersections.

    This implements the three-phase approach:
    Phase 1: Primitives are already fitted (the program argument)
    Phase 2: Compute pairwise intersection zones between ops
    Phase 3: Find target mesh faces in these zones that aren't well-covered
             by any single primitive — these are fillet/blend surfaces

    A fillet has roughly triangular cross-section with one edge optionally
    concave, extruded along the path where two primitives intersect.

    Returns:
        list of dicts, each describing a detected fillet:
            op_a, op_b: indices of the two primitives
            fillet_vertices: target vertices in the fillet zone
            path: approximate intersection path (centerline)
            concavity: estimated concavity of the fillet (0=flat, 1=fully concave)
    """
    from .cad_program import _eval_op

    target_v = np.asarray(target_v, dtype=np.float64)
    target_f = np.asarray(target_f)

    ops = [(i, op) for i, op in enumerate(program.operations) if op.enabled]
    if len(ops) < 2:
        return []

    # Evaluate each op to get its mesh
    op_meshes = {}
    for i, op in ops:
        try:
            result = _eval_op(op, [])
            if result is not None:
                op_meshes[i] = result  # (vertices, faces)
        except Exception:
            pass

    if len(op_meshes) < 2:
        return []

    # Build KDTrees for each op mesh
    op_trees = {}
    for i, (v, f) in op_meshes.items():
        if len(v) > 0:
            op_trees[i] = _AKDTree(v)

    target_tree = _AKDTree(target_v)
    bbox_diag = float(np.linalg.norm(target_v.max(0) - target_v.min(0)))
    proximity_threshold = bbox_diag * 0.08  # 8% of bbox diagonal

    fillets = []
    op_indices = list(op_meshes.keys())

    for ai in range(len(op_indices)):
        for bi in range(ai + 1, len(op_indices)):
            idx_a, idx_b = op_indices[ai], op_indices[bi]
            va, _ = op_meshes[idx_a]
            vb, _ = op_meshes[idx_b]

            if idx_a not in op_trees or idx_b not in op_trees:
                continue

            # Find vertices of A that are close to B (intersection zone)
            d_a_to_b, _ = op_trees[idx_b].query(va)
            near_a = va[d_a_to_b < proximity_threshold]

            if len(near_a) < 3:
                continue

            # Intersection zone center and extent
            zone_center = near_a.mean(axis=0)
            zone_radius = float(np.max(np.linalg.norm(near_a - zone_center, axis=1)))

            # Find target vertices in this zone that are NOT well-covered
            # by either primitive alone
            d_target_to_a, _ = op_trees[idx_a].query(target_v)
            d_target_to_b, _ = op_trees[idx_b].query(target_v)
            d_target_to_zone = np.linalg.norm(target_v - zone_center, axis=1)

            # Fillet candidates: near the intersection zone but not close
            # to either primitive surface
            # Use a tighter threshold to avoid including vertices that are
            # clearly on one primitive's surface
            coverage_thresh = proximity_threshold * 0.5
            in_zone = d_target_to_zone < zone_radius * 1.2
            not_on_a = d_target_to_a > coverage_thresh
            not_on_b = d_target_to_b > coverage_thresh
            fillet_mask = in_zone & not_on_a & not_on_b

            fillet_verts = target_v[fillet_mask]

            # Cap fillet vertices to < 20% of target to avoid marking
            # too much of the mesh as fillet
            if len(fillet_verts) > len(target_v) * 0.2:
                # Tighten: increase coverage threshold
                coverage_thresh = proximity_threshold * 0.8
                not_on_a = d_target_to_a > coverage_thresh
                not_on_b = d_target_to_b > coverage_thresh
                fillet_mask = in_zone & not_on_a & not_on_b
                fillet_verts = target_v[fillet_mask]

            if len(fillet_verts) < 5:
                continue

            # Estimate intersection path (PCA primary direction of near_a)
            centered = near_a - zone_center
            if len(centered) >= 3:
                cov = centered.T @ centered / len(centered)
                eigvals, eigvecs = np.linalg.eigh(cov)
                path_dir = eigvecs[:, np.argmax(eigvals)]
            else:
                path_dir = np.array([0, 0, 1.0])

            # Estimate concavity: check if fillet vertices are inside
            # the convex hull of the two primitives
            # (approximation: check if fillet is between the two surfaces)
            mean_dist_a = float(np.mean(d_target_to_a[fillet_mask]))
            mean_dist_b = float(np.mean(d_target_to_b[fillet_mask]))
            concavity = min(1.0, (mean_dist_a + mean_dist_b) / max(proximity_threshold, 1e-8))

            fillets.append({
                "op_a": idx_a,
                "op_b": idx_b,
                "fillet_vertices": fillet_verts,
                "near_a": near_a,
                "zone_center": zone_center,
                "zone_radius": zone_radius,
                "path_direction": path_dir,
                "concavity": round(concavity, 3),
                "n_vertices": len(fillet_verts),
            })

    return fillets


def fit_fillet_op(fillet_info, target_v, target_f):
    """Convert detected fillet info into a CadOp('fillet', params).

    Uses the intersection zone geometry (from the primitives' proximity)
    to determine the fillet path, then estimates cross-section radius from
    the uncovered fillet vertices.

    Args:
        fillet_info: dict from detect_intersection_fillets()
        target_v: full target mesh vertices
        target_f: full target mesh faces

    Returns:
        CadOp or None
    """
    from .cad_program import CadOp

    fillet_verts = np.asarray(fillet_info["fillet_vertices"], dtype=np.float64)
    zone_center = np.asarray(fillet_info["zone_center"])
    zone_radius = float(fillet_info["zone_radius"])
    concavity = float(fillet_info["concavity"])
    # Use near_a (intersection zone points from primitive A near B) for path
    near_a = np.asarray(fillet_info.get("near_a", fillet_verts), dtype=np.float64)

    if len(fillet_verts) < 5:
        return None

    # Determine path shape from intersection zone geometry (near_a)
    centered_zone = near_a - zone_center
    cov = centered_zone.T @ centered_zone / len(centered_zone)
    eigvals, eigvecs = np.linalg.eigh(cov)
    sort_idx = np.argsort(eigvals)[::-1]
    eigvecs = eigvecs[:, sort_idx]
    eigvals = eigvals[sort_idx]
    spreads = np.sqrt(np.maximum(eigvals, 0))

    # Ring detection: two large spreads, one small = planar ring
    # Also detect when fillet vertices themselves suggest a ring
    # by checking if they wrap around the zone center
    centered_fv = fillet_verts - zone_center
    proj_2d_fv = centered_fv @ eigvecs[:, :2]
    angles_fv = np.arctan2(proj_2d_fv[:, 1], proj_2d_fv[:, 0])
    # Check angular coverage — if vertices span >270° it's likely a ring
    angle_range = float(np.ptp(angles_fv))
    n_angle_bins = 12
    bin_edges = np.linspace(-np.pi, np.pi, n_angle_bins + 1)
    occupied_bins = sum(1 for k in range(n_angle_bins)
                        if np.any((angles_fv >= bin_edges[k]) &
                                  (angles_fv < bin_edges[k + 1])))
    angular_coverage = occupied_bins / n_angle_bins

    is_ring = (angular_coverage > 0.7 and
               spreads[0] > 1e-6 and spreads[1] > 1e-6 and
               spreads[1] / spreads[0] > 0.3)

    if is_ring:
        # Ring fillet: build path around the intersection circle
        # Use near_a points (they lie on the intersection) to define the ring
        proj_2d_zone = centered_zone @ eigvecs[:, :2]
        angles_zone = np.arctan2(proj_2d_zone[:, 1], proj_2d_zone[:, 0])

        n_path = min(32, max(12, len(near_a) // 3))
        bin_edges_path = np.linspace(-np.pi, np.pi, n_path + 1)
        path_pts = []
        mean_r = float(np.mean(np.linalg.norm(proj_2d_zone, axis=1)))

        for k in range(n_path):
            mask = (angles_zone >= bin_edges_path[k]) & (angles_zone < bin_edges_path[k + 1])
            if mask.sum() > 0:
                path_pts.append(near_a[mask].mean(axis=0))
            else:
                t = (bin_edges_path[k] + bin_edges_path[k + 1]) / 2
                pt_2d = np.array([mean_r * math.cos(t), mean_r * math.sin(t)])
                pt_3d = zone_center + eigvecs[:, :2] @ pt_2d
                path_pts.append(pt_3d)

        path = np.array(path_pts)
        closed = True
    else:
        # Open fillet: project onto primary axis and sample along it
        path_dir = eigvecs[:, 0]
        proj = centered_fv @ path_dir
        sorted_idx = np.argsort(proj)
        n_path = min(20, max(4, len(fillet_verts) // 5))
        indices = np.linspace(0, len(sorted_idx) - 1, n_path, dtype=int)
        path = fillet_verts[sorted_idx[indices]]
        closed = False

    # Estimate fillet radius from distance of fillet vertices to path
    path_tree = _AKDTree(path)
    d_to_path, _ = path_tree.query(fillet_verts)
    fillet_radius = float(np.percentile(d_to_path, 50))
    if fillet_radius < 1e-6:
        fillet_radius = zone_radius * 0.15

    # Determine up direction: perpendicular to the ring plane (if ring)
    # or the smallest PCA component direction
    if is_ring:
        # Normal to the ring plane = smallest eigenvector
        up_dir = eigvecs[:, 2].tolist()
    else:
        # Use the direction perpendicular to both path and radial spread
        up_dir = eigvecs[:, 2].tolist() if len(eigvecs) > 2 else [0, 0, 1]

    return CadOp("fillet", {
        "path": path.tolist(),
        "radius": fillet_radius,
        "concavity": min(1.0, max(0.0, concavity * 0.5)),
        "closed": closed,
        "n_cross": 6,
        "up_dir": up_dir,
    })


# ---------------------------------------------------------------------------
# Top-level API
# ---------------------------------------------------------------------------

def segment_mesh(vertices, faces, strategy="auto", template=None,
                  min_segment_faces=20, detect_fillets_flag=True):
    """Segment a mesh into parts suitable for CAD reconstruction.

    Args:
        vertices: (N, 3) array
        faces: (M, 3) array
        strategy: "skeleton", "sdf", "convexity", "projection",
                  "normal_cluster", or "auto"
        template: optional ObjectTemplate to guide strategy selection
        min_segment_faces: minimum faces per segment

    Returns:
        list of MeshSegment, each annotated with a recommended CAD action
    """
    v = np.asarray(vertices, dtype=np.float64)
    f = np.asarray(faces)

    # Auto-select strategy
    if strategy == "auto":
        if template is not None:
            strategy = template.segmentation_strategy
        else:
            strategy = _auto_select_strategy(v, f)

    # Dispatch
    dispatch = {
        "skeleton": lambda: segment_by_skeleton(v, f, min_segment_faces),
        "sdf": lambda: segment_by_sdf(v, f, min_segment_faces=min_segment_faces),
        "convexity": lambda: segment_by_convexity(v, f, min_segment_faces=min_segment_faces),
        "projection": lambda: segment_by_projection(v, f, min_segment_faces),
        "normal_cluster": lambda: segment_by_normals(v, f, min_segment_faces=min_segment_faces),
    }

    segments = dispatch.get(strategy, dispatch["skeleton"])()

    # Classify each segment's CAD action
    for seg in segments:
        classify_segment_action(seg)

    # Detect fillets/blends that mask underlying primitive intersections
    if detect_fillets_flag and len(segments) >= 2:
        detect_fillets(v, f, segments)

    return segments


def _auto_select_strategy(vertices, faces):
    """Pick the best segmentation strategy from mesh properties."""
    v = np.asarray(vertices)
    center = v.mean(axis=0)
    centered = v - center

    cov = np.cov(centered.T)
    eigvals = np.sort(np.linalg.eigvalsh(cov))[::-1]

    # Elongation
    elong = np.sqrt(eigvals[0] / max(eigvals[1], 1e-12))

    # Circularity
    circ = np.sqrt(eigvals[2] / max(eigvals[1], 1e-12))

    if elong > 3.0:
        return "skeleton"  # Long thin objects → skeleton decomposition
    elif circ > 0.85:
        return "sdf"  # Compact symmetric → SDF thickness
    elif len(faces) > 10000:
        return "normal_cluster"  # Large meshes → fast normal clustering
    else:
        return "convexity"  # General → convexity decomposition
