"""STEP file I/O — read STEP/STP/IGES files via OpenCASCADE (OCP).

Tessellates the CAD solid into a triangle mesh for use with the
meshxcad optimisation pipeline.
"""

import numpy as np


def read_step(filepath, linear_deflection=0.1, angular_deflection=0.5):
    """Read a STEP or IGES file and tessellate it to a triangle mesh.

    Args:
        filepath: path to .step, .stp, or .iges file
        linear_deflection: mesh resolution (smaller = finer, default 0.1)
        angular_deflection: angular resolution in radians (default 0.5)

    Returns:
        vertices: (N, 3) float64 array
        faces: (M, 3) int64 array of triangle indices
    """
    ext = filepath.rsplit(".", 1)[-1].lower() if "." in filepath else ""

    if ext in ("step", "stp"):
        shape = _read_step_shape(filepath)
    elif ext in ("iges", "igs"):
        shape = _read_iges_shape(filepath)
    else:
        raise ValueError(
            f"Unsupported CAD format: .{ext}  (use .step, .stp, .iges, .igs)")

    return _tessellate_shape(shape, linear_deflection, angular_deflection)


def _read_step_shape(filepath):
    """Read a STEP file and return the OCC shape."""
    from OCP.STEPControl import STEPControl_Reader
    from OCP.IFSelect import IFSelect_RetDone

    reader = STEPControl_Reader()
    status = reader.ReadFile(filepath)
    if status != IFSelect_RetDone:
        raise IOError(f"Failed to read STEP file: {filepath} (status={status})")

    reader.TransferRoots()
    shape = reader.OneShape()
    return shape


def _read_iges_shape(filepath):
    """Read an IGES file and return the OCC shape."""
    from OCP.IGESControl import IGESControl_Reader
    from OCP.IFSelect import IFSelect_RetDone

    reader = IGESControl_Reader()
    status = reader.ReadFile(filepath)
    if status != IFSelect_RetDone:
        raise IOError(f"Failed to read IGES file: {filepath} (status={status})")

    reader.TransferRoots()
    shape = reader.OneShape()
    return shape


def _tessellate_shape(shape, linear_deflection, angular_deflection):
    """Tessellate an OCC shape into (vertices, faces) arrays."""
    from OCP.BRepMesh import BRepMesh_IncrementalMesh
    from OCP.TopExp import TopExp_Explorer
    from OCP.TopAbs import TopAbs_FACE
    from OCP.BRep import BRep_Tool
    from OCP.TopLoc import TopLoc_Location
    from OCP.TopoDS import TopoDS

    # Tessellate the shape
    BRepMesh_IncrementalMesh(shape, linear_deflection, False,
                             angular_deflection, True)

    all_verts = []
    all_faces = []
    vert_offset = 0

    explorer = TopExp_Explorer(shape, TopAbs_FACE)
    while explorer.More():
        face = TopoDS.Face_s(explorer.Current())
        location = TopLoc_Location()
        triangulation = BRep_Tool.Triangulation_s(face, location)

        if triangulation is not None:
            n_nodes = triangulation.NbNodes()
            n_tris = triangulation.NbTriangles()

            # Extract vertices
            trsf = location.Transformation()
            for i in range(1, n_nodes + 1):
                node = triangulation.Node(i)
                pt = node.Transformed(trsf)
                all_verts.append([pt.X(), pt.Y(), pt.Z()])

            # Extract triangles (1-indexed in OCC)
            for i in range(1, n_tris + 1):
                tri = triangulation.Triangle(i)
                i1, i2, i3 = tri.Get()
                all_faces.append([
                    i1 - 1 + vert_offset,
                    i2 - 1 + vert_offset,
                    i3 - 1 + vert_offset,
                ])

            vert_offset += n_nodes

        explorer.Next()

    if not all_verts:
        raise ValueError(f"No geometry found in shape (empty tessellation)")

    vertices = np.array(all_verts, dtype=np.float64)
    faces = np.array(all_faces, dtype=np.int64)

    # Deduplicate coincident vertices (from shared edges between faces)
    vertices, faces = _deduplicate(vertices, faces)

    return vertices, faces


def _deduplicate(vertices, faces, tol=1e-8):
    """Merge coincident vertices within tolerance."""
    # Round to tolerance to group nearby vertices
    rounded = np.round(vertices / tol) * tol
    unique, inverse = np.unique(rounded, axis=0, return_inverse=True)
    new_faces = inverse[faces]
    return unique, new_faces


def write_step(filepath, vertices, faces):
    """Write a triangle mesh as a STEP file (as a sewed shell).

    This creates a TopoDS_Shell from the triangles and writes it as STEP.
    Useful for exporting optimised CAD meshes back to STEP format.

    Args:
        filepath: output .step path
        vertices: (N, 3) array
        faces: (M, 3) array of triangle indices
    """
    from OCP.STEPControl import STEPControl_Writer, STEPControl_AsIs
    from OCP.IFSelect import IFSelect_RetDone
    from OCP.BRepBuilderAPI import BRepBuilderAPI_MakePolygon, BRepBuilderAPI_Sewing
    from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeFace
    from OCP.gp import gp_Pnt
    from OCP.TopoDS import TopoDS

    sewing = BRepBuilderAPI_Sewing(1e-6)

    for tri in faces:
        v0 = vertices[tri[0]]
        v1 = vertices[tri[1]]
        v2 = vertices[tri[2]]

        wire = BRepBuilderAPI_MakePolygon(
            gp_Pnt(*v0), gp_Pnt(*v1), gp_Pnt(*v2), True)
        face_maker = BRepBuilderAPI_MakeFace(wire.Wire(), True)
        if face_maker.IsDone():
            sewing.Add(face_maker.Face())

    sewing.Perform()
    shape = sewing.SewedShape()

    writer = STEPControl_Writer()
    writer.Transfer(shape, STEPControl_AsIs)
    status = writer.Write(filepath)
    if status != IFSelect_RetDone:
        raise IOError(f"Failed to write STEP file: {filepath}")


# ---------------------------------------------------------------------------
# B-Rep face analysis
# ---------------------------------------------------------------------------

def analyze_brep_faces(shape):
    """Analyze the B-Rep faces of an OCC shape.

    Returns:
        list of dicts, each with 'face' (TopoDS_Face), 'area', 'surface_type',
        'surface_type_name'.
    """
    from OCP.TopExp import TopExp_Explorer
    from OCP.TopAbs import TopAbs_FACE
    from OCP.BRepAdaptor import BRepAdaptor_Surface
    from OCP.GProp import GProp_GProps
    from OCP.BRepGProp import BRepGProp
    from OCP.TopoDS import TopoDS

    _TYPE_NAMES = {
        0: "Plane", 1: "Cylinder", 2: "Cone", 3: "Sphere", 4: "Torus",
        5: "BezierSurface", 6: "BSplineSurface", 7: "SurfaceOfRevolution",
        8: "SurfaceOfExtrusion", 9: "OffsetSurface", 10: "OtherSurface",
    }

    faces_info = []
    explorer = TopExp_Explorer(shape, TopAbs_FACE)
    while explorer.More():
        face = TopoDS.Face_s(explorer.Current())
        adaptor = BRepAdaptor_Surface(face)
        stype = int(adaptor.GetType())
        props = GProp_GProps()
        BRepGProp.SurfaceProperties_s(face, props)
        area = props.Mass()
        faces_info.append({
            "face": face,
            "area": area,
            "surface_type": stype,
            "surface_type_name": _TYPE_NAMES.get(stype, f"Type_{stype}"),
        })
        explorer.Next()

    return faces_info


# ---------------------------------------------------------------------------
# Defeaturization: remove small features from a B-Rep shape
# ---------------------------------------------------------------------------

def defeature_shape(shape, area_fraction=0.1, max_faces=None):
    """Remove small B-Rep faces (fillets, chamfers, small holes) from a shape.

    Uses BRepAlgoAPI_Defeaturing to remove faces whose area is below a
    fraction of the median face area.  This simplifies the model while
    preserving the overall geometry.

    Args:
        shape: OCC TopoDS_Shape
        area_fraction: faces with area < median_area * area_fraction are
                       candidates for removal (default 0.1 = 10%)
        max_faces: maximum number of faces to remove in one pass (None = all)

    Returns:
        dict with:
            shape          — simplified TopoDS_Shape
            n_removed      — number of faces removed
            n_original     — original face count
            n_remaining    — remaining face count
            removed_faces  — list of removed face info dicts
    """
    from OCP.BRepAlgoAPI import BRepAlgoAPI_Defeaturing

    faces_info = analyze_brep_faces(shape)
    if not faces_info:
        return {"shape": shape, "n_removed": 0, "n_original": 0,
                "n_remaining": 0, "removed_faces": []}

    areas = np.array([fi["area"] for fi in faces_info])
    median_area = float(np.median(areas))
    threshold = median_area * area_fraction

    # Select small faces as removal candidates — prefer smallest first
    candidates = [(i, fi) for i, fi in enumerate(faces_info)
                  if fi["area"] < threshold]
    candidates.sort(key=lambda x: x[1]["area"])

    if max_faces is not None:
        candidates = candidates[:max_faces]

    if not candidates:
        return {"shape": shape, "n_removed": 0,
                "n_original": len(faces_info),
                "n_remaining": len(faces_info), "removed_faces": []}

    # Use BRepAlgoAPI_Defeaturing to remove faces
    defeaturer = BRepAlgoAPI_Defeaturing()
    defeaturer.SetShape(shape)
    for _idx, fi in candidates:
        defeaturer.AddFaceToRemove(fi["face"])

    defeaturer.Build()
    if not defeaturer.IsDone():
        # If full defeature fails, try removing faces one at a time
        removed = []
        current_shape = shape
        for _idx, fi in candidates:
            d = BRepAlgoAPI_Defeaturing()
            d.SetShape(current_shape)
            d.AddFaceToRemove(fi["face"])
            d.Build()
            if d.IsDone():
                current_shape = d.Shape()
                removed.append({
                    "area": fi["area"],
                    "surface_type_name": fi["surface_type_name"],
                })

        remaining = analyze_brep_faces(current_shape)
        return {"shape": current_shape, "n_removed": len(removed),
                "n_original": len(faces_info),
                "n_remaining": len(remaining),
                "removed_faces": removed}

    result_shape = defeaturer.Shape()
    remaining = analyze_brep_faces(result_shape)
    removed_info = [{"area": fi["area"],
                     "surface_type_name": fi["surface_type_name"]}
                    for _idx, fi in candidates]

    return {"shape": result_shape, "n_removed": len(candidates),
            "n_original": len(faces_info),
            "n_remaining": len(remaining),
            "removed_faces": removed_info}


def defeature_step(input_path, output_path, area_fraction=0.1,
                   linear_deflection=0.1, angular_deflection=0.5):
    """Load a STEP file, remove small features, and write the defeatured model.

    Uses two strategies:
    1. B-Rep defeaturing via OCP (preferred — preserves exact geometry)
    2. Mesh-level coarsening fallback (removes high-curvature detail)

    Args:
        input_path:  path to input STEP/STP file
        output_path: path to output STEP file
        area_fraction: removal threshold (fraction of median face area)
        linear_deflection: tessellation resolution for the output mesh
        angular_deflection: tessellation angular resolution

    Returns:
        dict with defeature stats and tessellated defeatured mesh
    """
    from OCP.STEPControl import STEPControl_Writer, STEPControl_AsIs
    from OCP.IFSelect import IFSelect_RetDone

    shape = _read_step_shape(input_path)

    # Try B-Rep defeaturing first
    result = defeature_shape(shape, area_fraction=area_fraction)

    if result["n_removed"] > 0:
        # B-Rep defeaturing succeeded
        writer = STEPControl_Writer()
        writer.Transfer(result["shape"], STEPControl_AsIs)
        status = writer.Write(output_path)
        if status != IFSelect_RetDone:
            raise IOError(f"Failed to write defeatured STEP: {output_path}")
        def_v, def_f = _tessellate_shape(
            result["shape"], linear_deflection, angular_deflection)
    else:
        # B-Rep defeaturing didn't remove anything — use coarse tessellation
        # to create a simplified mesh, then write as STEP shell
        coarse_v, coarse_f = _tessellate_shape(
            shape,
            linear_deflection * 3.0,   # coarser mesh
            angular_deflection * 2.0,   # fewer angular samples
        )
        # Smooth the coarse mesh to remove small feature detail
        coarse_v = _laplacian_smooth(coarse_v, coarse_f, iterations=3,
                                     factor=0.3)
        write_step(output_path, coarse_v, coarse_f)
        def_v, def_f = coarse_v, coarse_f
        result["method"] = "mesh_coarsening"
        result["n_removed"] = -1  # indicate mesh-level defeature

    # Also get fine tessellation of original for comparison
    orig_v, orig_f = _tessellate_shape(
        shape, linear_deflection, angular_deflection)

    result["defeatured_vertices"] = def_v
    result["defeatured_faces"] = def_f
    result["original_vertices"] = orig_v
    result["original_faces"] = orig_f
    return result


def _laplacian_smooth(vertices, faces, iterations=3, factor=0.3):
    """Apply Laplacian smoothing to a mesh.

    Moves each vertex toward the average of its neighbors while
    preserving overall shape.
    """
    v = vertices.copy()
    n = len(v)

    # Build adjacency
    neighbors = [set() for _ in range(n)]
    for face in faces:
        for i in range(3):
            a, b = int(face[i]), int(face[(i + 1) % 3])
            neighbors[a].add(b)
            neighbors[b].add(a)

    for _ in range(iterations):
        new_v = v.copy()
        for i in range(n):
            nbrs = neighbors[i]
            if nbrs:
                avg = v[list(nbrs)].mean(axis=0)
                new_v[i] = v[i] + factor * (avg - v[i])
        v = new_v

    return v


# ---------------------------------------------------------------------------
# Re-featurization: reconstruct detailed STEP from mesh + defeatured STEP
# ---------------------------------------------------------------------------

def refeature(mesh_vertices, mesh_faces, defeatured_path,
              linear_deflection=0.1, angular_deflection=0.5):
    """Re-create a detailed CAD model from a detailed mesh and defeatured STEP.

    Strategy:
    1. Tessellate the defeatured STEP to get a simplified mesh
    2. Compute per-vertex displacement from defeatured mesh to detailed mesh
    3. Identify regions where the detailed mesh differs significantly
       (these are the removed features — fillets, holes, chamfers)
    4. Use the defeatured B-Rep as a base and overlay feature geometry
       from the detailed mesh

    Args:
        mesh_vertices:   (N, 3) detailed mesh vertices
        mesh_faces:      (M, 3) detailed mesh faces
        defeatured_path: path to defeatured STEP file
        linear_deflection: tessellation resolution
        angular_deflection: tessellation angular resolution

    Returns:
        dict with:
            refeature_vertices  — (P, 3) recomposed mesh vertices
            refeature_faces     — (Q, 3) recomposed mesh faces
            n_feature_regions   — number of detected feature regions
            feature_coverage    — fraction of mesh covered by features
            accuracy            — accuracy of reconstruction vs original mesh
    """
    from scipy.spatial import KDTree

    mesh_v = np.asarray(mesh_vertices, dtype=np.float64)
    mesh_f = np.asarray(mesh_faces, dtype=np.int64)

    # Load and tessellate defeatured model
    def_shape = _read_step_shape(defeatured_path)
    def_v, def_f = _tessellate_shape(
        def_shape, linear_deflection, angular_deflection)

    # Build KDTree for defeatured mesh to find correspondences
    def_tree = KDTree(def_v)

    # For each vertex in the detailed mesh, find nearest defeatured vertex
    dists, indices = def_tree.query(mesh_v)

    # Compute displacement field
    bbox_diag = float(np.linalg.norm(mesh_v.max(0) - mesh_v.min(0)))
    rel_dists = dists / max(bbox_diag, 1e-12)

    # Feature threshold: vertices displaced more than 0.5% of bbox
    feature_threshold = 0.005
    feature_mask = rel_dists > feature_threshold

    n_feature_verts = int(feature_mask.sum())
    feature_coverage = n_feature_verts / max(len(mesh_v), 1)

    # Label connected feature regions using face adjacency
    n_regions = 0
    if n_feature_verts > 0:
        # Simple region counting via union-find on feature faces
        feature_vert_set = set(np.where(feature_mask)[0])
        feature_faces_mask = np.array([
            mesh_f[i, 0] in feature_vert_set or
            mesh_f[i, 1] in feature_vert_set or
            mesh_f[i, 2] in feature_vert_set
            for i in range(len(mesh_f))
        ])
        n_feature_faces = int(feature_faces_mask.sum())

        # Count connected components among feature faces
        if n_feature_faces > 0:
            parent = list(range(len(mesh_v)))

            def find(x):
                while parent[x] != x:
                    parent[x] = parent[parent[x]]
                    x = parent[x]
                return x

            def union(a, b):
                ra, rb = find(a), find(b)
                if ra != rb:
                    parent[ra] = rb

            for fi in np.where(feature_faces_mask)[0]:
                face = mesh_f[fi]
                union(face[0], face[1])
                union(face[1], face[2])

            roots = set()
            for vi in np.where(feature_mask)[0]:
                roots.add(find(vi))
            n_regions = len(roots)

    # Compose the refeature mesh by combining:
    #   - Defeatured base geometry (from STEP B-Rep)
    #   - Reconstructed feature regions (from detailed mesh)
    #
    # For each feature region, extract submesh and try to reconstruct it
    # as a proper CAD primitive (fillet torus, chamfer plane, hole cylinder).
    # Non-feature vertices snap to the defeatured surface.

    refeature_v = mesh_v.copy()
    non_feature = ~feature_mask
    refeature_v[non_feature] = def_v[indices[non_feature]]
    refeature_f = mesh_f.copy()

    # Extract and classify each feature region
    region_info = []
    if n_regions > 0 and n_feature_verts > 0:
        from .reconstruct import fit_sphere, fit_cylinder

        # Group vertices by region root
        region_map = {}  # root -> list of vertex indices
        for vi in np.where(feature_mask)[0]:
            root = find(vi)
            if root not in region_map:
                region_map[root] = []
            region_map[root].append(vi)

        for root, vert_indices in region_map.items():
            vert_indices = np.array(vert_indices)
            region_v = mesh_v[vert_indices]
            if len(region_v) < 10:
                region_info.append({
                    "n_verts": len(vert_indices),
                    "type": "small_detail",
                })
                continue

            region_bbox = float(np.linalg.norm(
                region_v.max(0) - region_v.min(0)))

            # Try primitive fitting to characterize the feature
            feature_type = "freeform"
            try:
                sph = fit_sphere(region_v)
                sph_score = sph["residual"] / max(region_bbox, 1e-12)
                if sph_score < 0.1:
                    feature_type = "fillet_sphere"
            except Exception:
                sph_score = 1.0

            try:
                cyl = fit_cylinder(region_v)
                cyl_score = cyl["residual"] / max(region_bbox, 1e-12)
                if cyl_score < 0.08:
                    feature_type = "hole_or_fillet"
            except Exception:
                cyl_score = 1.0

            # Compute mean displacement direction for this region
            displacements = mesh_v[vert_indices] - def_v[indices[vert_indices]]
            mean_disp = float(np.linalg.norm(displacements.mean(0)))
            max_disp = float(np.linalg.norm(displacements, axis=1).max())

            region_info.append({
                "n_verts": len(vert_indices),
                "type": feature_type,
                "bbox_diag": region_bbox,
                "mean_displacement": mean_disp,
                "max_displacement": max_disp,
                "sphere_score": round(sph_score, 4),
                "cylinder_score": round(cyl_score, 4),
            })

    # Measure accuracy vs original detailed mesh
    from .general_align import hausdorff_distance
    hd = hausdorff_distance(refeature_v, mesh_v)
    accuracy = max(0.0, 1.0 - hd["mean_symmetric"] / max(bbox_diag, 1e-12) * 5)

    return {
        "refeature_vertices": refeature_v,
        "refeature_faces": refeature_f,
        "n_feature_regions": n_regions,
        "n_feature_vertices": n_feature_verts,
        "feature_coverage": feature_coverage,
        "accuracy": accuracy,
        "defeatured_vertices": def_v,
        "defeatured_faces": def_f,
        "feature_regions": region_info,
    }
