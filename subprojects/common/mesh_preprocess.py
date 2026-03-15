"""Shared mesh preprocessing for local mesh-to-CAD integrations."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path

import numpy as np
import trimesh


@dataclass
class Normalization:
    center: np.ndarray
    scale: float
    target: str


def load_mesh(mesh_path: str | Path) -> trimesh.Trimesh:
    mesh = trimesh.load_mesh(mesh_path)
    if isinstance(mesh, trimesh.Scene):
        geometries = [g for g in mesh.geometry.values() if len(g.vertices) and len(g.faces)]
        if not geometries:
            raise ValueError(f"No triangle geometry found in {mesh_path}")
        mesh = trimesh.util.concatenate(tuple(geometries))
    if not isinstance(mesh, trimesh.Trimesh):
        raise TypeError(f"Unsupported mesh type for {mesh_path}: {type(mesh)!r}")
    if mesh.faces is None or len(mesh.faces) == 0:
        raise ValueError(f"Mesh has no faces: {mesh_path}")
    return mesh


def normalize_mesh(mesh: trimesh.Trimesh, target: str) -> tuple[trimesh.Trimesh, Normalization]:
    mesh = mesh.copy()
    bounds = mesh.bounds.astype(np.float64)
    center = (bounds[0] + bounds[1]) / 2.0
    extents = bounds[1] - bounds[0]
    max_extent = float(np.max(extents))
    if max_extent <= 0:
        raise ValueError("Mesh has zero extent")

    scale = 1.0
    if target == "unit_symmetric":
        mesh.apply_translation(-center)
        scale = 2.0 / max_extent
        mesh.apply_scale(scale)
    elif target == "unit_cube":
        mesh.apply_translation(-center)
        scale = 1.0 / max_extent
        mesh.apply_scale(scale)
        mesh.apply_translation([0.5, 0.5, 0.5])
    elif target == "raw":
        scale = 1.0
    else:
        raise ValueError(f"Unknown normalization target: {target}")

    return mesh, Normalization(center=center, scale=scale, target=target)


def sample_point_cloud(
    mesh: trimesh.Trimesh,
    n_points: int,
    n_pre_points: int = 8192,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    points, face_idx = trimesh.sample.sample_surface(mesh, n_pre_points, seed=seed)
    chosen = farthest_point_sample(points, n_points, rng)
    return points[chosen].astype(np.float32), face_idx[chosen].astype(np.int32)


def farthest_point_sample(points: np.ndarray, n_points: int, rng: np.random.Generator) -> np.ndarray:
    if len(points) < n_points:
        raise ValueError(f"Requested {n_points} points from only {len(points)} samples")
    chosen = np.empty(n_points, dtype=np.int64)
    distances = np.full(len(points), np.inf, dtype=np.float64)
    chosen[0] = int(rng.integers(len(points)))
    current = points[chosen[0]]
    for i in range(1, n_points):
        delta = points - current
        distances = np.minimum(distances, np.einsum("ij,ij->i", delta, delta))
        chosen[i] = int(np.argmax(distances))
        current = points[chosen[i]]
    return chosen


def label_sampled_faces(
    vertices: np.ndarray,
    faces: np.ndarray,
    sampled_face_idx: np.ndarray,
    n_clusters: int = 8,
    min_component_faces: int = 32,
) -> np.ndarray:
    normals = face_normals(vertices, faces)
    n_faces = len(faces)
    n_clusters = max(2, min(n_clusters, max(2, n_faces // max(min_component_faces, 1))))

    rng = np.random.default_rng(42)
    centers = rng.normal(size=(n_clusters, 3))
    centers /= np.linalg.norm(centers, axis=1, keepdims=True)

    labels = np.zeros(n_faces, dtype=np.int32)
    for _ in range(30):
        dots = np.abs(normals @ centers.T)
        new_labels = np.argmax(dots, axis=1).astype(np.int32)
        if np.array_equal(labels, new_labels):
            break
        labels = new_labels
        for cluster_idx in range(n_clusters):
            mask = labels == cluster_idx
            if not np.any(mask):
                continue
            avg = normals[mask].mean(axis=0)
            norm = np.linalg.norm(avg)
            if norm > 1e-12:
                centers[cluster_idx] = avg / norm

    refined = refine_connected_components(faces, labels, min_component_faces)
    return refined[sampled_face_idx]


def face_normals(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]
    normals = np.cross(v1 - v0, v2 - v0)
    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    return normals / norms


def refine_connected_components(
    faces: np.ndarray,
    initial_labels: np.ndarray,
    min_component_faces: int,
) -> np.ndarray:
    adjacency = build_face_adjacency(faces)
    labels = np.full(len(faces), -1, dtype=np.int32)
    next_label = 0

    for face_index in range(len(faces)):
        if labels[face_index] != -1:
            continue
        seed_label = initial_labels[face_index]
        queue = [face_index]
        component = []
        labels[face_index] = next_label
        head = 0
        while head < len(queue):
            current = queue[head]
            head += 1
            component.append(current)
            for neighbor in adjacency[current]:
                if labels[neighbor] != -1:
                    continue
                if initial_labels[neighbor] != seed_label:
                    continue
                labels[neighbor] = next_label
                queue.append(neighbor)

        if len(component) < min_component_faces:
            for idx in component:
                labels[idx] = -2
        else:
            next_label += 1

    if next_label == 0:
        return np.zeros(len(faces), dtype=np.int32)

    return reassign_small_components(faces, labels)


def reassign_small_components(faces: np.ndarray, labels: np.ndarray) -> np.ndarray:
    adjacency = build_face_adjacency(faces)
    output = labels.copy()
    for face_index in np.where(output == -2)[0]:
        neighbor_labels = [output[n] for n in adjacency[face_index] if output[n] >= 0]
        output[face_index] = neighbor_labels[0] if neighbor_labels else 0
    return output


def build_face_adjacency(faces: np.ndarray) -> list[list[int]]:
    adjacency = [set() for _ in range(len(faces))]
    edge_to_faces: dict[tuple[int, int], list[int]] = {}
    for face_index, face in enumerate(faces):
        for edge_idx in range(3):
            edge = tuple(sorted((int(face[edge_idx]), int(face[(edge_idx + 1) % 3]))))
            edge_to_faces.setdefault(edge, []).append(face_index)
    for face_group in edge_to_faces.values():
        for i in range(len(face_group)):
            for j in range(i + 1, len(face_group)):
                adjacency[face_group[i]].add(face_group[j])
                adjacency[face_group[j]].add(face_group[i])
    return [sorted(neighbors) for neighbors in adjacency]


def save_point_cloud_xyz(points: np.ndarray, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(path, points, fmt="%.8f")
    return path


def save_point_cloud_xyzc(points: np.ndarray, labels: np.ndarray, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    stacked = np.column_stack((points, labels.astype(np.int32)))
    np.savetxt(path, stacked, fmt=["%.8f", "%.8f", "%.8f", "%d"])
    return path


def save_mesh(mesh: trimesh.Trimesh, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    mesh.export(path)
    return path


def save_metadata(path: str | Path, **payload: object) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    serializable = json.loads(json.dumps(payload, default=_json_default))
    path.write_text(json.dumps(serializable, indent=2) + "\n")
    return path


def _json_default(value: object) -> object:
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Unsupported JSON value: {type(value)!r}")
