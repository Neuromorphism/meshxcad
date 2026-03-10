"""B-rep detail transfer: apply CAD operations to a STEP solid to match a mesh.

Given a plain STEP model and a detailed scan mesh, this module:

1. Tessellates the STEP model and the scan mesh.
2. Runs the pre-alignment pipeline to bring them into the same frame.
3. Computes a displacement field and identifies spatial regions of mismatch.
4. For each region, classifies the required geometric operation
   (pocket, boss, hole, fillet, chamfer) and fits parameters.
5. Applies those operations as OCC boolean cuts/fuses on the original
   B-rep solid, producing a proper STEP-compatible TopoDS_Shape.
"""

import math
import numpy as np
from scipy.spatial import KDTree

from . import alignment
from .stl_io import read_binary_stl
from .step_io import read_step, _read_step_shape, _tessellate_shape


# ---------------------------------------------------------------------------
# Region classification
# ---------------------------------------------------------------------------

def _classify_region(points, displacements, plain_center):
    """Classify a displacement region as a cut, boss, or hole.

    Args:
        points: (K, 3) vertices in the region
        displacements: (K, 3) displacement vectors at those vertices
        plain_center: (3,) centroid of the plain mesh (for inward/outward)

    Returns:
        dict with op_type, params
    """
    center = points.mean(axis=0)
    mean_disp = displacements.mean(axis=0)
    disp_mag = np.linalg.norm(mean_disp)

    if disp_mag < 1e-8:
        return None

    # Direction of displacement relative to part centre
    outward = center - plain_center
    outward_norm = np.linalg.norm(outward)
    if outward_norm > 1e-8:
        outward /= outward_norm

    # Positive dot = moving away from centre (boss), negative = inward (cut)
    direction = float(np.dot(mean_disp / disp_mag, outward))

    # PCA of region shape
    centered = points - center
    cov = centered.T @ centered / max(len(points), 1)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]
    spreads = np.sqrt(np.maximum(eigvals, 0))

    # Circularity check (in the plane perpendicular to displacement)
    disp_dir = mean_disp / disp_mag
    perp = centered - np.outer(centered @ disp_dir, disp_dir)
    radii = np.linalg.norm(perp, axis=1)
    mean_r = float(np.mean(radii))
    std_r = float(np.std(radii))
    circularity = 1.0 - (std_r / max(mean_r, 1e-8))

    depth = float(disp_mag)

    if circularity > 0.7 and mean_r > 0 and spreads[0] / max(spreads[1], 1e-8) < 3:
        # Circular region
        if direction < -0.3:
            return {
                "op_type": "hole",
                "center": center.tolist(),
                "axis": disp_dir.tolist(),
                "radius": mean_r,
                "depth": depth,
            }
        elif direction > 0.3:
            return {
                "op_type": "boss_cylinder",
                "center": center.tolist(),
                "axis": disp_dir.tolist(),
                "radius": mean_r,
                "height": depth,
            }

    # Non-circular region
    if direction < -0.3:
        # Inward displacement → pocket/cut
        extent = (spreads[:2] * 2).tolist()
        return {
            "op_type": "pocket",
            "center": center.tolist(),
            "normal": disp_dir.tolist(),
            "extent": extent,
            "depth": depth,
            "axes": eigvecs[:, :2].T.tolist(),
        }
    elif direction > 0.3:
        extent = (spreads[:2] * 2).tolist()
        return {
            "op_type": "boss",
            "center": center.tolist(),
            "normal": disp_dir.tolist(),
            "extent": extent,
            "height": depth,
            "axes": eigvecs[:, :2].T.tolist(),
        }

    return None


# ---------------------------------------------------------------------------
# Displacement → regions
# ---------------------------------------------------------------------------

def identify_detail_regions(plain_verts, detail_verts, plain_faces,
                             threshold_percentile=70, min_region_points=10):
    """Find distinct regions of significant displacement.

    Returns:
        list of dicts, each with op_type and geometric parameters.
    """
    # Full alignment
    aligned_detail, scale, R, t = alignment.full_align(
        detail_verts, plain_verts)

    # Correspondences
    tree = KDTree(aligned_detail)
    distances, indices = tree.query(plain_verts)
    displacements = aligned_detail[indices] - plain_verts

    disp_mags = np.linalg.norm(displacements, axis=1)
    threshold = np.percentile(disp_mags, threshold_percentile)
    threshold = max(threshold, 1e-6)

    significant = disp_mags > threshold
    if not np.any(significant):
        return []

    sig_pts = plain_verts[significant]
    sig_disp = displacements[significant]

    # Cluster significant points
    regions = _cluster_points(sig_pts, sig_disp, min_region_points)
    plain_center = plain_verts.mean(axis=0)

    operations = []
    for pts, disps in regions:
        op = _classify_region(pts, disps, plain_center)
        if op is not None:
            operations.append(op)

    return operations


def _cluster_points(points, displacements, min_points, max_clusters=8):
    """Simple spatial clustering of displacement regions."""
    if len(points) < min_points:
        return [(points, displacements)]

    n_clusters = min(max_clusters, max(1, len(points) // max(min_points, 1)))
    rng = np.random.RandomState(42)
    centers = points[rng.choice(len(points), n_clusters, replace=False)]

    # K-means (10 iterations)
    for _ in range(10):
        dist_to_centers = np.linalg.norm(
            points[:, None, :] - centers[None, :, :], axis=2)
        labels = dist_to_centers.argmin(axis=1)
        for k in range(n_clusters):
            mask = labels == k
            if np.any(mask):
                centers[k] = points[mask].mean(axis=0)

    regions = []
    for k in range(n_clusters):
        mask = labels == k
        if np.sum(mask) >= min_points:
            regions.append((points[mask], displacements[mask]))

    return regions if regions else [(points, displacements)]


# ---------------------------------------------------------------------------
# OCC B-rep operations
# ---------------------------------------------------------------------------

def _make_occ_cylinder(center, axis, radius, height):
    """Create an OCC cylindrical solid."""
    from OCP.gp import gp_Pnt, gp_Dir, gp_Ax2
    from OCP.BRepPrimAPI import BRepPrimAPI_MakeCylinder

    ax = np.asarray(axis, dtype=np.float64)
    ax = ax / (np.linalg.norm(ax) + 1e-12)
    c = np.asarray(center, dtype=np.float64)

    # Place base at center - axis * height/2
    base = c - ax * height / 2

    occ_ax = gp_Ax2(gp_Pnt(*base), gp_Dir(*ax))
    cyl = BRepPrimAPI_MakeCylinder(occ_ax, radius, height)
    return cyl.Shape()


def _make_occ_box(center, normal, extent, depth, axes):
    """Create an OCC box solid for pocket/boss operations."""
    from OCP.gp import gp_Pnt, gp_Dir, gp_Ax2, gp_Vec
    from OCP.BRepPrimAPI import BRepPrimAPI_MakeBox
    from OCP.BRepBuilderAPI import BRepBuilderAPI_Transform
    from OCP.gp import gp_Trsf

    n = np.asarray(normal, dtype=np.float64)
    n = n / (np.linalg.norm(n) + 1e-12)
    c = np.asarray(center, dtype=np.float64)
    ax = np.asarray(axes, dtype=np.float64)  # (2, 3) local u, v axes

    half_u = extent[0] / 2
    half_v = extent[1] / 2

    # Create box at origin, then transform
    from OCP.BRepPrimAPI import BRepPrimAPI_MakeBox
    box = BRepPrimAPI_MakeBox(extent[0], extent[1], depth)
    shape = box.Shape()

    # Build rotation matrix: columns = u, v, normal
    u = ax[0] / (np.linalg.norm(ax[0]) + 1e-12)
    v = ax[1] / (np.linalg.norm(ax[1]) + 1e-12)
    # Ensure right-handed
    w = np.cross(u, v)
    if np.dot(w, n) < 0:
        w = -w
        v = np.cross(w, u)

    origin = c - u * half_u - v * half_v - w * depth / 2

    trsf = gp_Trsf()
    trsf.SetValues(
        u[0], v[0], w[0], origin[0],
        u[1], v[1], w[1], origin[1],
        u[2], v[2], w[2], origin[2],
    )
    transformer = BRepBuilderAPI_Transform(shape, trsf, True)
    return transformer.Shape()


def apply_operations_to_shape(base_shape, operations):
    """Apply a list of classified operations as OCC booleans on base_shape.

    Args:
        base_shape: OCC TopoDS_Shape (the plain CAD solid)
        operations: list of dicts from identify_detail_regions()

    Returns:
        Modified TopoDS_Shape (B-rep)
    """
    from OCP.BRepAlgoAPI import BRepAlgoAPI_Cut, BRepAlgoAPI_Fuse

    result = base_shape
    applied = 0

    for op in operations:
        try:
            tool_shape = None
            is_cut = False

            if op["op_type"] == "hole":
                tool_shape = _make_occ_cylinder(
                    op["center"], op["axis"], op["radius"], op["depth"] * 3)
                is_cut = True

            elif op["op_type"] == "pocket":
                tool_shape = _make_occ_box(
                    op["center"], op["normal"], op["extent"],
                    op["depth"], op["axes"])
                is_cut = True

            elif op["op_type"] == "boss_cylinder":
                tool_shape = _make_occ_cylinder(
                    op["center"], op["axis"], op["radius"], op["height"])
                is_cut = False

            elif op["op_type"] == "boss":
                tool_shape = _make_occ_box(
                    op["center"], op["normal"], op["extent"],
                    op["height"], op["axes"])
                is_cut = False

            if tool_shape is not None:
                if is_cut:
                    boolean = BRepAlgoAPI_Cut(result, tool_shape)
                else:
                    boolean = BRepAlgoAPI_Fuse(result, tool_shape)
                if boolean.IsDone():
                    result = boolean.Shape()
                    applied += 1
        except Exception:
            # Skip operations that fail (bad geometry, etc.)
            continue

    return result, applied


def write_step_shape(shape, filepath):
    """Write an OCC TopoDS_Shape to a STEP file."""
    from OCP.STEPControl import STEPControl_Writer, STEPControl_AsIs
    from OCP.IFSelect import IFSelect_RetDone

    writer = STEPControl_Writer()
    writer.Transfer(shape, STEPControl_AsIs)
    status = writer.Write(filepath)
    if status != IFSelect_RetDone:
        raise IOError(f"Failed to write STEP file: {filepath}")


# ---------------------------------------------------------------------------
# High-level API
# ---------------------------------------------------------------------------

def transfer_to_step(plain_step_path, detail_mesh_path, output_step_path,
                     linear_deflection=0.05):
    """Transfer mesh detail onto a STEP model, outputting a modified STEP B-rep.

    Args:
        plain_step_path: path to plain .step file
        detail_mesh_path: path to detail .stl file
        output_step_path: output .step path
        linear_deflection: tessellation quality for analysis

    Returns:
        dict with n_operations, output_path
    """
    # Read the original B-rep shape
    base_shape = _read_step_shape(plain_step_path)

    # Tessellate for analysis
    plain_verts, plain_faces = _tessellate_shape(
        base_shape, linear_deflection, 0.5)

    # Load detail mesh
    detail_verts, detail_faces = read_binary_stl(detail_mesh_path)

    print(f"  Plain: {len(plain_verts)} verts, Detail: {len(detail_verts)} verts")

    # Identify detail regions (displacement analysis with pre-alignment)
    operations = identify_detail_regions(
        plain_verts, detail_verts, plain_faces)
    print(f"  Identified {len(operations)} detail regions:")
    for i, op in enumerate(operations):
        print(f"    {i+1}. {op['op_type']}")

    if not operations:
        print("  No significant detail differences found; writing unmodified shape.")
        write_step_shape(base_shape, output_step_path)
        return {"n_operations": 0, "output_path": output_step_path}

    # Apply operations as B-rep booleans
    result_shape, n_applied = apply_operations_to_shape(base_shape, operations)
    print(f"  Applied {n_applied}/{len(operations)} CAD operations")

    # Write result
    write_step_shape(result_shape, output_step_path)

    return {
        "n_operations": n_applied,
        "operations": operations,
        "output_path": output_step_path,
    }
