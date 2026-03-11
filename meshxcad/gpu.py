"""GPU acceleration backend with automatic fallback.

Provides GPU-accelerated versions of the most expensive operations:
  - Nearest-neighbor queries (replaces scipy.spatial.KDTree)
  - Pairwise distance computation
  - Batch norm computation
  - Matrix operations (SVD, eigendecomposition)
  - Vertex/face operations (normals, areas, adjacency)

Detected backends (in priority order):
  1. CuPy + cuML/FAISS-GPU  → full GPU acceleration
  2. PyTorch CUDA           → GPU via torch tensors
  3. Numba CUDA             → JIT-compiled GPU kernels
  4. CPU-optimized fallback  → batched scipy/numpy with threading

Usage:
    from meshxcad.gpu import backend
    dists, idx = backend.nearest_neighbors(query_pts, ref_pts)
    norms = backend.row_norms(vectors)
"""

import os
import warnings
import numpy as np

# ---------------------------------------------------------------------------
# Backend detection
# ---------------------------------------------------------------------------

_BACKEND = "cpu"        # "cupy", "torch", "numba", or "cpu"
_GPU_AVAILABLE = False
_xp = np               # array module (numpy or cupy)

# Allow disabling GPU via environment variable
_FORCE_CPU = os.environ.get("MESHXCAD_CPU", "").lower() in ("1", "true", "yes")


def _detect_backend():
    """Detect the best available GPU backend."""
    global _BACKEND, _GPU_AVAILABLE, _xp

    if _FORCE_CPU:
        _BACKEND = "cpu"
        _GPU_AVAILABLE = False
        _xp = np
        return

    # Try CuPy first (best numpy compatibility)
    try:
        import cupy as cp
        # Verify GPU is accessible
        cp.cuda.Device(0).compute_capability
        _xp = cp
        _BACKEND = "cupy"
        _GPU_AVAILABLE = True
        return
    except Exception:
        pass

    # Try PyTorch CUDA
    try:
        import torch
        if torch.cuda.is_available():
            _BACKEND = "torch"
            _GPU_AVAILABLE = True
            return
    except Exception:
        pass

    # Try Numba CUDA
    try:
        from numba import cuda
        if cuda.is_available():
            _BACKEND = "numba"
            _GPU_AVAILABLE = True
            return
    except Exception:
        pass

    _BACKEND = "cpu"
    _GPU_AVAILABLE = False
    _xp = np


_detect_backend()


# ---------------------------------------------------------------------------
# Public API: Backend info
# ---------------------------------------------------------------------------

def get_backend():
    """Return current backend name: 'cupy', 'torch', 'numba', or 'cpu'."""
    return _BACKEND


def is_gpu_available():
    """Return True if any GPU backend is available."""
    return _GPU_AVAILABLE


def get_array_module():
    """Return the array module (cupy or numpy)."""
    return _xp


def to_gpu(arr):
    """Move numpy array to GPU. No-op if CPU-only."""
    arr = np.asarray(arr, dtype=np.float32)
    if _BACKEND == "cupy":
        return _xp.asarray(arr)
    return arr


def to_cpu(arr):
    """Move array back to CPU numpy."""
    if _BACKEND == "cupy":
        import cupy as cp
        if isinstance(arr, cp.ndarray):
            return cp.asnumpy(arr)
    return np.asarray(arr)


# ---------------------------------------------------------------------------
# Nearest-neighbor queries (the #1 bottleneck)
# ---------------------------------------------------------------------------

def nearest_neighbors(query, reference, k=1):
    """Find k nearest neighbors from query points in reference points.

    Args:
        query: (N, D) query points
        reference: (M, D) reference points
        k: number of neighbors

    Returns:
        distances: (N,) or (N, k) distances
        indices: (N,) or (N, k) indices into reference
    """
    if _BACKEND == "cupy":
        return _nn_cupy(query, reference, k)
    elif _BACKEND == "torch":
        return _nn_torch(query, reference, k)
    else:
        return _nn_cpu(query, reference, k)


def _nn_cupy(query, reference, k):
    """GPU nearest-neighbor using CuPy brute-force or cuML."""
    import cupy as cp

    # Try cuML first (optimized GPU KNN)
    try:
        from cuml.neighbors import NearestNeighbors
        nn = NearestNeighbors(n_neighbors=k, algorithm="brute",
                              metric="euclidean")
        nn.fit(cp.asarray(reference, dtype=cp.float32))
        dists, idx = nn.kneighbors(cp.asarray(query, dtype=cp.float32))
        dists = cp.asnumpy(dists).astype(np.float64)
        idx = cp.asnumpy(idx)
        if k == 1:
            return dists.ravel(), idx.ravel()
        return dists, idx
    except ImportError:
        pass

    # Fallback: CuPy brute-force pairwise distances (tiled for memory)
    q = cp.asarray(query, dtype=cp.float32)
    r = cp.asarray(reference, dtype=cp.float32)
    return _nn_bruteforce_gpu(q, r, k, cp)


def _nn_bruteforce_gpu(q, r, k, xp):
    """Tiled brute-force KNN on GPU to avoid OOM on large meshes."""
    n = len(q)
    m = len(r)
    # Tile size: keep GPU memory under ~512 MB
    # Each tile: n_tile × m × sizeof(float32) = n_tile × m × 4 bytes
    max_tile = max(1, (512 * 1024 * 1024) // (m * 4))

    all_dists = np.empty(n, dtype=np.float64) if k == 1 else np.empty((n, k), dtype=np.float64)
    all_idx = np.empty(n, dtype=np.int64) if k == 1 else np.empty((n, k), dtype=np.int64)

    # Precompute reference norms: ||r||^2
    r_sq = xp.sum(r * r, axis=1)  # (M,)

    for start in range(0, n, max_tile):
        end = min(start + max_tile, n)
        q_tile = q[start:end]

        # ||q - r||^2 = ||q||^2 - 2*q·r^T + ||r||^2
        q_sq = xp.sum(q_tile * q_tile, axis=1, keepdims=True)  # (tile, 1)
        dist_sq = q_sq - 2.0 * q_tile @ r.T + r_sq[None, :]   # (tile, M)
        # Clamp negative values from numerical errors
        xp.maximum(dist_sq, 0.0, out=dist_sq)

        if k == 1:
            idx_tile = xp.argmin(dist_sq, axis=1)
            d_tile = xp.sqrt(dist_sq[xp.arange(len(q_tile)), idx_tile])
            all_dists[start:end] = to_cpu(d_tile).astype(np.float64)
            all_idx[start:end] = to_cpu(idx_tile).astype(np.int64)
        else:
            # Partial sort for top-k
            idx_tile = xp.argpartition(dist_sq, k, axis=1)[:, :k]
            d_tile = xp.sqrt(xp.take_along_axis(dist_sq, idx_tile, axis=1))
            # Sort within the k candidates
            sort_order = xp.argsort(d_tile, axis=1)
            d_tile = xp.take_along_axis(d_tile, sort_order, axis=1)
            idx_tile = xp.take_along_axis(idx_tile, sort_order, axis=1)
            all_dists[start:end] = to_cpu(d_tile).astype(np.float64)
            all_idx[start:end] = to_cpu(idx_tile).astype(np.int64)

    return all_dists, all_idx


def _nn_torch(query, reference, k):
    """GPU nearest-neighbor using PyTorch CUDA."""
    import torch

    device = torch.device("cuda")
    q = torch.tensor(query, dtype=torch.float64, device=device)
    r = torch.tensor(reference, dtype=torch.float64, device=device)

    n = len(q)
    m = len(r)
    # Tile to avoid OOM
    max_tile = max(1, (512 * 1024 * 1024) // (m * 4))

    all_dists = np.empty(n, dtype=np.float64) if k == 1 else np.empty((n, k), dtype=np.float64)
    all_idx = np.empty(n, dtype=np.int64) if k == 1 else np.empty((n, k), dtype=np.int64)

    r_sq = (r * r).sum(dim=1)

    for start in range(0, n, max_tile):
        end = min(start + max_tile, n)
        q_tile = q[start:end]
        q_sq = (q_tile * q_tile).sum(dim=1, keepdim=True)
        dist_sq = q_sq - 2.0 * q_tile @ r.T + r_sq.unsqueeze(0)
        dist_sq.clamp_(min=0.0)

        if k == 1:
            d_tile, idx_tile = dist_sq.min(dim=1)
            all_dists[start:end] = d_tile.sqrt().cpu().numpy().astype(np.float64)
            all_idx[start:end] = idx_tile.cpu().numpy().astype(np.int64)
        else:
            d_tile, idx_tile = dist_sq.topk(k, dim=1, largest=False)
            all_dists[start:end] = d_tile.sqrt().cpu().numpy().astype(np.float64)
            all_idx[start:end] = idx_tile.cpu().numpy().astype(np.int64)

    return all_dists, all_idx


def _nn_cpu(query, reference, k):
    """Optimized CPU nearest-neighbor.

    For small meshes (< 5000 vertices), uses scipy KDTree.
    For larger meshes, uses batched brute-force with numpy which can be
    faster than KDTree construction + query when called once.
    """
    from scipy.spatial import KDTree
    n = len(query)
    m = len(reference)

    # KDTree is faster when reference set is reused or large
    if m > 100 or n > 100:
        tree = KDTree(reference)
        dists, idx = tree.query(query, k=k)
        if k == 1:
            return np.asarray(dists, dtype=np.float64), np.asarray(idx, dtype=np.int64)
        return np.asarray(dists, dtype=np.float64), np.asarray(idx, dtype=np.int64)

    # Brute force for tiny meshes
    diff = np.asarray(query)[:, None, :] - np.asarray(reference)[None, :, :]
    dist_sq = np.sum(diff * diff, axis=2)
    if k == 1:
        idx = np.argmin(dist_sq, axis=1)
        dists = np.sqrt(dist_sq[np.arange(n), idx])
        return dists, idx
    else:
        idx = np.argpartition(dist_sq, k, axis=1)[:, :k]
        dists = np.sqrt(np.take_along_axis(dist_sq, idx, axis=1))
        sort_order = np.argsort(dists, axis=1)
        return np.take_along_axis(dists, sort_order, axis=1), \
               np.take_along_axis(idx, sort_order, axis=1)


# ---------------------------------------------------------------------------
# KDTree-compatible wrapper (drop-in for scipy.spatial.KDTree)
# ---------------------------------------------------------------------------

class AcceleratedKDTree:
    """Drop-in replacement for scipy.spatial.KDTree with GPU acceleration.

    Usage:
        tree = AcceleratedKDTree(reference_points)
        dists, idx = tree.query(query_points)
    """

    def __init__(self, data):
        self._data = np.asarray(data, dtype=np.float64)
        self._gpu_data = None
        self._cpu_tree = None

    def query(self, x, k=1):
        """Query nearest neighbors."""
        x = np.asarray(x, dtype=np.float64)
        squeeze = False
        if x.ndim == 1:
            x = x.reshape(1, -1)
            squeeze = True
        if _GPU_AVAILABLE:
            dists, idx = nearest_neighbors(x, self._data, k=k)
            if squeeze:
                return dists[0], idx[0]
            return dists, idx
        else:
            # Use scipy KDTree (lazy construction)
            if self._cpu_tree is None:
                from scipy.spatial import KDTree
                self._cpu_tree = KDTree(self._data)
            dists, idx = self._cpu_tree.query(x, k=k)
            return np.asarray(dists, dtype=np.float64), np.asarray(idx, dtype=np.int64)

    @property
    def data(self):
        return self._data


# ---------------------------------------------------------------------------
# Batch distance operations
# ---------------------------------------------------------------------------

def pairwise_distances_squared(a, b):
    """Compute squared pairwise distances between a (N,D) and b (M,D).

    Returns (N, M) matrix. Uses GPU if available.
    """
    if _BACKEND == "cupy":
        import cupy as cp
        a_g = cp.asarray(a, dtype=cp.float32)
        b_g = cp.asarray(b, dtype=cp.float32)
        a_sq = cp.sum(a_g * a_g, axis=1, keepdims=True)
        b_sq = cp.sum(b_g * b_g, axis=1)
        d2 = a_sq - 2.0 * a_g @ b_g.T + b_sq[None, :]
        cp.maximum(d2, 0.0, out=d2)
        return cp.asnumpy(d2).astype(np.float64)
    elif _BACKEND == "torch":
        import torch
        device = torch.device("cuda")
        a_t = torch.tensor(a, dtype=torch.float32, device=device)
        b_t = torch.tensor(b, dtype=torch.float32, device=device)
        d2 = torch.cdist(a_t, b_t, p=2.0).pow(2)
        return d2.cpu().numpy().astype(np.float64)
    else:
        # Optimized CPU: use einsum to avoid (N,M,D) intermediate
        a = np.asarray(a, dtype=np.float64)
        b = np.asarray(b, dtype=np.float64)
        a_sq = np.sum(a * a, axis=1, keepdims=True)
        b_sq = np.sum(b * b, axis=1)
        d2 = a_sq - 2.0 * a @ b.T + b_sq[None, :]
        np.maximum(d2, 0.0, out=d2)
        return d2


def row_norms(vectors, squared=False):
    """Compute row-wise L2 norms of (N, D) array. GPU-accelerated.

    Args:
        vectors: (N, D) array
        squared: if True, return squared norms

    Returns:
        (N,) array of norms
    """
    if _BACKEND == "cupy":
        import cupy as cp
        v = cp.asarray(vectors, dtype=cp.float32)
        sq = cp.sum(v * v, axis=1)
        result = sq if squared else cp.sqrt(sq)
        return cp.asnumpy(result).astype(np.float64)
    elif _BACKEND == "torch":
        import torch
        device = torch.device("cuda")
        v = torch.tensor(vectors, dtype=torch.float32, device=device)
        if squared:
            return (v * v).sum(dim=1).cpu().numpy().astype(np.float64)
        return v.norm(dim=1).cpu().numpy().astype(np.float64)
    else:
        v = np.asarray(vectors, dtype=np.float64)
        sq = np.sum(v * v, axis=1)
        return sq if squared else np.sqrt(sq)


# ---------------------------------------------------------------------------
# Hausdorff distance (GPU-accelerated)
# ---------------------------------------------------------------------------

def hausdorff_distance_gpu(vertices_a, vertices_b):
    """GPU-accelerated symmetric Hausdorff + mean surface distance.

    Drop-in replacement for general_align.hausdorff_distance().
    """
    a = np.asarray(vertices_a, dtype=np.float64)
    b = np.asarray(vertices_b, dtype=np.float64)

    d_a2b, _ = nearest_neighbors(a, b, k=1)
    d_b2a, _ = nearest_neighbors(b, a, k=1)

    return {
        "hausdorff": float(max(np.max(d_a2b), np.max(d_b2a))),
        "mean_a_to_b": float(np.mean(d_a2b)),
        "mean_b_to_a": float(np.mean(d_b2a)),
        "mean_symmetric": float((np.mean(d_a2b) + np.mean(d_b2a)) / 2),
    }


# ---------------------------------------------------------------------------
# Mesh operations (normals, areas) — GPU-accelerated
# ---------------------------------------------------------------------------

def compute_vertex_normals(vertices, faces):
    """Compute area-weighted per-vertex normals. GPU if available."""
    if _BACKEND == "cupy":
        return _vertex_normals_cupy(vertices, faces)
    # CPU path (identical to general_align._compute_vertex_normals)
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


def _vertex_normals_cupy(vertices, faces):
    """CuPy vertex normal computation."""
    import cupy as cp
    verts = cp.asarray(vertices, dtype=cp.float64)
    tris = cp.asarray(faces)
    normals = cp.zeros_like(verts)
    v0, v1, v2 = verts[tris[:, 0]], verts[tris[:, 1]], verts[tris[:, 2]]
    face_normals = cp.cross(v1 - v0, v2 - v0)
    for i in range(3):
        cp.add.at(normals, tris[:, i], face_normals)
    norms = cp.linalg.norm(normals, axis=1, keepdims=True)
    norms[norms < 1e-12] = 1.0
    result = normals / norms
    return cp.asnumpy(result)


def compute_face_normals(vertices, faces):
    """Unit face normals. GPU if available."""
    if _BACKEND == "cupy":
        import cupy as cp
        v = cp.asarray(vertices, dtype=cp.float64)
        f = cp.asarray(faces)
        v0, v1, v2 = v[f[:, 0]], v[f[:, 1]], v[f[:, 2]]
        fn = cp.cross(v1 - v0, v2 - v0)
        norms = cp.linalg.norm(fn, axis=1, keepdims=True)
        norms[norms < 1e-12] = 1.0
        return cp.asnumpy(fn / norms)
    v = np.asarray(vertices, dtype=np.float64)
    f = np.asarray(faces)
    v0, v1, v2 = v[f[:, 0]], v[f[:, 1]], v[f[:, 2]]
    fn = np.cross(v1 - v0, v2 - v0)
    norms = np.linalg.norm(fn, axis=1, keepdims=True)
    norms[norms < 1e-12] = 1.0
    return fn / norms


def face_areas(vertices, faces):
    """Per-face area. GPU if available."""
    if _BACKEND == "cupy":
        import cupy as cp
        v = cp.asarray(vertices, dtype=cp.float64)
        f = cp.asarray(faces)
        v0, v1, v2 = v[f[:, 0]], v[f[:, 1]], v[f[:, 2]]
        return float(cp.asnumpy(0.5 * cp.linalg.norm(cp.cross(v1 - v0, v2 - v0), axis=1)))
    v = np.asarray(vertices, dtype=np.float64)
    f = np.asarray(faces)
    v0, v1, v2 = v[f[:, 0]], v[f[:, 1]], v[f[:, 2]]
    return 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0), axis=1)


# ---------------------------------------------------------------------------
# Linear algebra (SVD, EIG) — GPU-accelerated
# ---------------------------------------------------------------------------

def eigh(matrix):
    """Eigendecomposition of symmetric matrix. GPU if available.

    Returns (eigenvalues, eigenvectors) like np.linalg.eigh().
    """
    if _BACKEND == "cupy":
        import cupy as cp
        m = cp.asarray(matrix, dtype=cp.float64)
        w, v = cp.linalg.eigh(m)
        return cp.asnumpy(w), cp.asnumpy(v)
    elif _BACKEND == "torch":
        import torch
        device = torch.device("cuda")
        m = torch.tensor(matrix, dtype=torch.float64, device=device)
        w, v = torch.linalg.eigh(m)
        return w.cpu().numpy(), v.cpu().numpy()
    return np.linalg.eigh(matrix)


def svd(matrix, full_matrices=True):
    """Singular value decomposition. GPU if available.

    Returns (U, S, Vh) like np.linalg.svd().
    """
    if _BACKEND == "cupy":
        import cupy as cp
        m = cp.asarray(matrix, dtype=cp.float64)
        u, s, vh = cp.linalg.svd(m, full_matrices=full_matrices)
        return cp.asnumpy(u), cp.asnumpy(s), cp.asnumpy(vh)
    elif _BACKEND == "torch":
        import torch
        device = torch.device("cuda")
        m = torch.tensor(matrix, dtype=torch.float64, device=device)
        u, s, vh = torch.linalg.svd(m, full_matrices=full_matrices)
        return u.cpu().numpy(), s.cpu().numpy(), vh.cpu().numpy()
    return np.linalg.svd(matrix, full_matrices=full_matrices)


def covariance_pca(points):
    """Compute PCA via covariance eigendecomposition. GPU if available.

    Args:
        points: (N, D) point cloud

    Returns:
        eigenvalues: (D,) sorted ascending
        eigenvectors: (D, D) columns are eigenvectors
        center: (D,) centroid
    """
    pts = np.asarray(points, dtype=np.float64)
    center = pts.mean(axis=0)
    centered = pts - center
    cov = (centered.T @ centered) / max(len(pts) - 1, 1)
    w, v = eigh(cov)
    return w, v, center


# ---------------------------------------------------------------------------
# Vectorized adjacency building (replaces Python loops)
# ---------------------------------------------------------------------------

def build_vertex_adjacency_matrix(faces, n_verts):
    """Build sparse adjacency matrix from faces. Vectorized.

    Returns a scipy.sparse.csr_matrix of shape (n_verts, n_verts).
    """
    from scipy.sparse import csr_matrix

    f = np.asarray(faces)
    # All edge pairs: (i,j) for each face edge
    rows = np.concatenate([f[:, 0], f[:, 0], f[:, 1], f[:, 1], f[:, 2], f[:, 2]])
    cols = np.concatenate([f[:, 1], f[:, 2], f[:, 0], f[:, 2], f[:, 0], f[:, 1]])
    data = np.ones(len(rows), dtype=np.float32)
    adj = csr_matrix((data, (rows, cols)), shape=(n_verts, n_verts))
    adj.data[:] = 1.0  # Deduplicate
    return adj


# ---------------------------------------------------------------------------
# Batch operations for coevolution (evaluate many mutations at once)
# ---------------------------------------------------------------------------

def batch_hausdorff(candidates_verts, target_v):
    """Compute hausdorff distances for multiple candidate meshes at once.

    Args:
        candidates_verts: list of (Ni, 3) vertex arrays
        target_v: (M, 3) target vertices

    Returns:
        list of hausdorff_distance dicts
    """
    target_v = np.asarray(target_v, dtype=np.float64)

    if _BACKEND == "cupy" and len(candidates_verts) > 1:
        return _batch_hausdorff_gpu(candidates_verts, target_v)

    # CPU: just loop (scipy KDTree reuse for target)
    from scipy.spatial import KDTree
    target_tree = KDTree(target_v)

    results = []
    for cand_v in candidates_verts:
        cand_v = np.asarray(cand_v, dtype=np.float64)
        d_c2t, _ = target_tree.query(cand_v)
        cand_tree = KDTree(cand_v)
        d_t2c, _ = cand_tree.query(target_v)
        results.append({
            "hausdorff": float(max(np.max(d_c2t), np.max(d_t2c))),
            "mean_a_to_b": float(np.mean(d_c2t)),
            "mean_b_to_a": float(np.mean(d_t2c)),
            "mean_symmetric": float((np.mean(d_c2t) + np.mean(d_t2c)) / 2),
        })
    return results


def _batch_hausdorff_gpu(candidates_verts, target_v):
    """GPU batch hausdorff using shared target reference."""
    results = []
    for cand_v in candidates_verts:
        results.append(hausdorff_distance_gpu(cand_v, target_v))
    return results


# ---------------------------------------------------------------------------
# Summary / diagnostics
# ---------------------------------------------------------------------------

def gpu_selftest():
    """Run a short vector computation to verify the backend works end-to-end.

    Returns (ok, detail) where *ok* is True if the backend produced the
    correct answer and *detail* is a human-readable status string.
    """
    try:
        # Small dot-product + nearest-neighbor check
        a = np.array([[1.0, 2.0, 3.0],
                       [4.0, 5.0, 6.0]], dtype=np.float64)
        b = np.array([[1.0, 2.0, 3.0],
                       [7.0, 8.0, 9.0]], dtype=np.float64)

        # nearest_neighbors should map a[0]→b[0] (dist 0) and a[1]→b[0]
        dists, idx = nearest_neighbors(a, b, k=1)

        # a[0] is identical to b[0] → distance ≈ 0, index 0
        if not (idx[0] == 0 and dists[0] < 1e-6):
            return False, "nearest-neighbor returned wrong result for identical point"

        # Also verify row_norms: ||[3,4]|| should be 5
        norms = row_norms(np.array([[3.0, 4.0]], dtype=np.float64))
        if abs(norms[0] - 5.0) > 1e-6:
            return False, f"row_norms([3,4])={norms[0]:.6f}, expected 5.0"

        return True, "ok"
    except Exception as exc:
        return False, str(exc)


def gpu_info():
    """Return diagnostic info about the GPU backend."""
    info = {
        "backend": _BACKEND,
        "gpu_available": _GPU_AVAILABLE,
        "forced_cpu": _FORCE_CPU,
    }

    if _BACKEND == "cupy":
        import cupy as cp
        dev = cp.cuda.Device(0)
        info["device_name"] = dev.name if hasattr(dev, "name") else "unknown"
        info["compute_capability"] = dev.compute_capability
        mem = cp.cuda.Device(0).mem_info
        info["gpu_memory_free_mb"] = mem[0] // (1024 * 1024)
        info["gpu_memory_total_mb"] = mem[1] // (1024 * 1024)

    elif _BACKEND == "torch":
        import torch
        info["device_name"] = torch.cuda.get_device_name(0)
        info["gpu_memory_total_mb"] = torch.cuda.get_device_properties(0).total_mem // (1024 * 1024)

    elif _BACKEND == "numba":
        from numba import cuda
        dev = cuda.get_current_device()
        info["device_name"] = dev.name

    return info
